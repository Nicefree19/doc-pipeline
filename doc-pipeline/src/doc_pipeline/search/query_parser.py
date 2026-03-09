"""Metadata-aware query parser for search queries.

Extracts project name, year, document type, category, and domain topics
from free-text search queries to feed into SearchAggregator for metadata
bonus scoring.

Parse priority:
  1. Year extraction (regex)
  2. Document type matching (TypeRegistry keywords)
  3. Project name matching (known projects from registry)
  4. Domain topic tagging (fixed keyword dictionary)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher


# Fixed domain keyword dictionary for structural engineering topics
_DOMAIN_TOPICS: dict[str, str] = {
    "균열": "균열",
    "크랙": "균열",
    "보강": "보강",
    "슬래브": "슬래브",
    "기둥": "기둥",
    "기초": "기초",
    "철골": "철골",
    "내진": "내진",
    "처짐": "처짐",
    "도면": "도면",
    "배근": "배근",
    "전단": "전단",
    "휨": "휨",
    "콘크리트": "콘크리트",
    "철근": "철근",
    "앵커": "앵커",
    "용접": "용접",
    "하중": "하중",
    "지반": "지반",
    "파일": "파일",
    "벽체": "벽체",
    "보": "보",
    "옥탑": "옥탑",
    "지하": "지하",
    "옹벽": "옹벽",
    "데크": "데크",
    "합성보": "합성보",
}

# Year pattern: 4-digit year optionally followed by 년
_YEAR_RE = re.compile(r"((?:19|20)\d{2})년?")


@dataclass
class ParsedQuery:
    """Result of parsing a search query for metadata hints."""

    raw_query: str
    cleaned_query: str
    project: str = ""
    year: int = 0
    doc_type: str = ""
    category: str = ""
    topics: list[str] = field(default_factory=list)


class QueryParser:
    """Extracts metadata from free-text search queries.

    Args:
        known_projects: Set of project names from the document registry.
        type_keywords: Map of ``{doc_type_ext: [keywords]}`` from TypeRegistry.
    """

    _FUZZY_THRESHOLD = 0.7

    def __init__(
        self,
        known_projects: set[str] | None = None,
        type_keywords: dict[str, list[str]] | None = None,
        type_category_map: dict[str, str] | None = None,
    ) -> None:
        self._known_projects: set[str] = known_projects or set()
        # type_name -> category mapping for _extract_doc_type
        self._type_category_map: dict[str, str] = type_category_map or {}
        # Build reverse map: keyword -> (doc_type, keyword_len) sorted longest first
        self._type_keyword_map: list[tuple[str, str]] = []
        if type_keywords:
            for doc_type, keywords in type_keywords.items():
                for kw in keywords:
                    self._type_keyword_map.append((kw, doc_type))
            # Sort by keyword length descending for longest-match-first
            self._type_keyword_map.sort(key=lambda x: len(x[0]), reverse=True)

    def parse(self, query: str) -> ParsedQuery:
        """Parse a query string and extract metadata hints.

        Returns:
            ParsedQuery with extracted metadata and cleaned query text.
        """
        if not query or not query.strip():
            return ParsedQuery(raw_query=query, cleaned_query="")

        raw = query.strip()
        working = raw

        # 1. Extract year
        year, working = self._extract_year(working)

        # 2. Extract document type
        doc_type, category = self._extract_doc_type(working)

        # 3. Extract project name
        project = self._extract_project(working)

        # 4. Extract domain topics
        topics = self._extract_topics(working)

        # Build cleaned query (remove year, keep the rest)
        cleaned = working.strip()
        # Collapse multiple spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return ParsedQuery(
            raw_query=raw,
            cleaned_query=cleaned,
            project=project,
            year=year,
            doc_type=doc_type,
            category=category,
            topics=topics,
        )

    def _extract_year(self, text: str) -> tuple[int, str]:
        """Extract year from text. Returns (year, text_with_year_removed)."""
        match = _YEAR_RE.search(text)
        if not match:
            return 0, text
        year = int(match.group(1))
        # Remove the matched year (with optional 년) from text
        cleaned = text[:match.start()] + text[match.end():]
        return year, cleaned

    def _extract_doc_type(self, text: str) -> tuple[str, str]:
        """Extract document type using keyword matching. Returns (doc_type, category)."""
        if not self._type_keyword_map:
            return "", ""

        text_lower = text.lower()
        for keyword, doc_type in self._type_keyword_map:
            if keyword.lower() in text_lower:
                category = self._type_category_map.get(doc_type, "")
                return doc_type, category
        return "", ""

    def _extract_project(self, text: str) -> str:
        """Extract project name by matching against known projects.

        Match strategy (in order):
        1. Exact match: query contains an exact known project name
        2. Reverse containment: a known project contains a query token
        3. Fuzzy match: SequenceMatcher ratio >= 0.7
        """
        if not self._known_projects or not text.strip():
            return ""

        text_stripped = text.strip()

        # Phase 1: Exact substring match — longest project name first
        exact_matches: list[str] = []
        for proj in sorted(self._known_projects, key=len, reverse=True):
            if not proj:
                continue
            if proj in text_stripped:
                exact_matches.append(proj)

        if exact_matches:
            return exact_matches[0]  # Longest match

        # Phase 2: Reverse containment — query token is substring of known project
        # Split text into meaningful tokens (2+ chars)
        tokens = [t for t in text_stripped.split() if len(t) >= 2]
        reverse_matches: list[str] = []
        for token in tokens:
            for proj in sorted(self._known_projects, key=len, reverse=True):
                if not proj:
                    continue
                if token in proj:
                    reverse_matches.append(proj)

        if reverse_matches:
            # Return the longest matching project
            return max(reverse_matches, key=len)

        # Phase 3: Fuzzy match using SequenceMatcher
        best_ratio = 0.0
        best_proj = ""
        for proj in self._known_projects:
            if not proj:
                continue
            for token in tokens:
                ratio = SequenceMatcher(None, token, proj).ratio()
                if ratio >= self._FUZZY_THRESHOLD and ratio > best_ratio:
                    best_ratio = ratio
                    best_proj = proj

        return best_proj

    def _extract_topics(self, text: str) -> list[str]:
        """Extract domain-specific topics from text."""
        found: list[str] = []
        seen: set[str] = set()
        for keyword, canonical in _DOMAIN_TOPICS.items():
            if keyword in text and canonical not in seen:
                found.append(canonical)
                seen.add(canonical)
        return found
