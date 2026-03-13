"""Search profile resolution and relevance policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from doc_pipeline.search.aggregator import DocumentResult
    from doc_pipeline.search.query_parser import ParsedQuery

SearchProfile = Literal[
    "auto",
    "technical_qa",
    "project_lookup",
    "contract_lookup",
    "method_docs",
]

ResolvedSearchProfile = Literal[
    "technical_qa",
    "project_lookup",
    "contract_lookup",
    "method_docs",
]

_CONTRACT_HINTS = frozenset({
    "계약",
    "계약서",
    "계약금",
    "용역계약",
    "도급",
    "금액",
    "계약금액",
})

_METHOD_DOC_HINTS = frozenset({
    "공법자료",
    "소개자료",
    "요약자료",
    "공법소개",
    "제안서",
    "소개",
    "자료",
    "인정서",
    "ve",
})

_METHOD_TOPIC_HINTS = frozenset({
    "공법",
    "시공법",
    "탄소섬유",
    "cfrp",
    "데크플레이트",
    "deck plate",
    "합성보",
    "내화뿜칠",
    "tsc",
})

_TECHNICAL_HINTS = frozenset({
    "검토",
    "구조",
    "내진",
    "슬래브",
    "기둥",
    "보강",
    "안전성",
    "접합부",
    "처짐",
    "철골",
    "균열",
    "기초",
    "응력",
    "계산",
    "심의",
    "조치계획",
})

_PROFILE_DOC_PRIORS: dict[str, dict[str, float]] = {
    "technical_qa": {
        "계약서": -0.30,
        "공법자료": -0.35,
        "의견서": 0.18,
        "조치계획서": 0.12,
    },
    "project_lookup": {
        "계약서": 0.14,
        "공법자료": -0.30,
        "의견서": 0.08,
        "조치계획서": 0.08,
    },
    "contract_lookup": {
        "계약서": 0.35,
        "공법자료": -0.25,
        "의견서": -0.08,
        "조치계획서": -0.08,
    },
    "method_docs": {
        "계약서": -0.25,
        "공법자료": 0.45,
        "의견서": -0.10,
        "조치계획서": -0.08,
    },
}


@dataclass(frozen=True)
class SearchProfilePolicy:
    """Ranking/filter policy for a resolved search profile."""

    name: ResolvedSearchProfile
    doc_type_priors: dict[str, float]
    include_doc_types: frozenset[str] = frozenset()
    prefer_non_doc_types: frozenset[str] = frozenset()
    default_doc_type_filter: str = ""
    fetch_multiplier: int = 5


def get_search_profile_policy(profile: ResolvedSearchProfile) -> SearchProfilePolicy:
    """Return ranking/filter policy for a resolved profile."""
    if profile == "contract_lookup":
        return SearchProfilePolicy(
            name=profile,
            doc_type_priors=_PROFILE_DOC_PRIORS[profile],
            prefer_non_doc_types=frozenset({"공법자료"}),
            default_doc_type_filter="계약서",
            fetch_multiplier=4,
        )
    if profile == "method_docs":
        return SearchProfilePolicy(
            name=profile,
            doc_type_priors=_PROFILE_DOC_PRIORS[profile],
            include_doc_types=frozenset({"공법자료"}),
            default_doc_type_filter="공법자료",
            fetch_multiplier=6,
        )
    if profile == "project_lookup":
        return SearchProfilePolicy(
            name=profile,
            doc_type_priors=_PROFILE_DOC_PRIORS[profile],
            prefer_non_doc_types=frozenset({"공법자료"}),
            fetch_multiplier=4,
        )
    return SearchProfilePolicy(
        name="technical_qa",
        doc_type_priors=_PROFILE_DOC_PRIORS["technical_qa"],
        prefer_non_doc_types=frozenset({"계약서", "공법자료"}),
        fetch_multiplier=6,
    )


def resolve_search_profile(
    query_text: str,
    *,
    search_profile: SearchProfile = "auto",
    parsed: ParsedQuery | None = None,
    doc_type_filter: str | None = None,
    doc_type_ext_filter: str | None = None,
) -> ResolvedSearchProfile:
    """Resolve the active search profile from explicit input or query heuristics."""
    if search_profile and search_profile != "auto":
        return search_profile

    selected_type = (doc_type_ext_filter or doc_type_filter or "").strip()
    if selected_type == "계약서":
        return "contract_lookup"
    if selected_type == "공법자료":
        return "method_docs"

    text = query_text.lower()
    if any(term in text for term in _CONTRACT_HINTS):
        return "contract_lookup"

    has_method_doc_hint = any(term in text for term in _METHOD_DOC_HINTS)
    has_method_topic_hint = any(term in text for term in _METHOD_TOPIC_HINTS)

    if parsed and parsed.project:
        if has_method_doc_hint:
            return "method_docs"
        return "project_lookup"

    if has_method_doc_hint or has_method_topic_hint:
        return "method_docs"

    return "technical_qa"


def get_doc_type_prior(
    doc_type: str,
    *,
    doc_type_ext: str = "",
    profile: ResolvedSearchProfile,
) -> float:
    """Return doc-type prior for the resolved profile."""
    resolved_type = _normalize_doc_type(doc_type, doc_type_ext)
    return get_search_profile_policy(profile).doc_type_priors.get(resolved_type, 0.0)


def rank_profile_results(
    doc_results: list[DocumentResult],
    *,
    profile: ResolvedSearchProfile,
) -> list[DocumentResult]:
    """Apply profile-specific filtering preference and re-rank results."""
    if not doc_results:
        return []

    policy = get_search_profile_policy(profile)
    scored: list[DocumentResult] = []
    fallbacks: list[DocumentResult] = []

    for doc in doc_results:
        resolved_type = _normalize_doc_type(doc.doc_type, doc.doc_type_ext)
        if policy.include_doc_types and resolved_type not in policy.include_doc_types:
            fallbacks.append(doc)
            continue
        if policy.prefer_non_doc_types and resolved_type in policy.prefer_non_doc_types:
            fallbacks.append(doc)
            continue
        scored.append(doc)

    scored.sort(key=lambda d: d.doc_score, reverse=True)
    fallbacks.sort(key=lambda d: d.doc_score, reverse=True)
    return scored + fallbacks


def _normalize_doc_type(doc_type: str, doc_type_ext: str) -> str:
    """Map doc_type/doc_type_ext into stable profile buckets."""
    if doc_type:
        return doc_type
    if doc_type_ext == "계약서":
        return "계약서"
    return doc_type_ext
