"""Unified search entry point.

All search callers (API, CLI, eval, drafter) should use ``unified_search()``
to ensure they go through the same improved pipeline:

    query_parser.parse() → search_rrf()/search_hybrid() → SearchAggregator.aggregate()

This prevents the problem where eval/CLI/drafter bypass RRF and aggregation
by calling ``store.search()`` directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from doc_pipeline.search.aggregator import DocumentResult, SearchAggregator
from doc_pipeline.search.profiles import (
    SearchProfile,
    get_search_profile_policy,
    rank_profile_results,
    resolve_search_profile,
)
from doc_pipeline.search.query_parser import ParsedQuery

if TYPE_CHECKING:
    from doc_pipeline.search.query_parser import QueryParser
    from doc_pipeline.storage.registry import DocumentRegistry
    from doc_pipeline.storage.vectordb import ChunkFTS, VectorStore

logger = logging.getLogger(__name__)


def unified_search(
    store: VectorStore,
    query_text: str,
    query_embedding: list[float],
    *,
    n_results: int = 5,
    doc_type_filter: str | None = None,
    category_filter: str | None = None,
    doc_type_ext_filter: str | None = None,
    year_filter: int | None = None,
    project_name_filter: str | None = None,
    search_profile: SearchProfile = "auto",
    exclude_doc_ids: list[str] | None = None,
    query_parser: QueryParser | None = None,
    registry: DocumentRegistry | None = None,
    chunk_fts: ChunkFTS | None = None,
) -> tuple[list[DocumentResult], ParsedQuery | None]:
    """Single search function that all callers should use.

    Internally:
      1. Parse query for metadata hints (if parser provided).
      2. Run ``search_hybrid()`` (vector+FTS) or ``search_rrf()`` (vector-only).
      3. Aggregate chunks into document-level results.
      4. Apply doc-level FTS bonus from registry (if available).

    Args:
        store: VectorStore instance.
        query_text: Raw query string.
        query_embedding: Embedding vector for the query.
        n_results: Desired number of document-level results.
        doc_type_filter: Explicit doc_type filter (user-selected).
        category_filter: Explicit category filter (user-selected).
        doc_type_ext_filter: Explicit doc_type_ext filter (user-selected).
        year_filter: Explicit year override (takes precedence over parsed).
        project_name_filter: Explicit project override (takes precedence over parsed).
        search_profile: Optional retrieval profile (or ``"auto"``).
        exclude_doc_ids: Doc IDs to exclude from results.
        query_parser: Optional QueryParser for metadata extraction.
        registry: Optional DocumentRegistry for doc-level FTS bonus.
        chunk_fts: Optional ChunkFTS for chunk-level FTS search.

    Returns:
        Tuple of (document_results, parsed_query_or_None).
    """
    # 1. Parse query for metadata hints
    parsed: ParsedQuery | None = None
    if query_parser:
        try:
            parsed = query_parser.parse(query_text)
        except Exception:
            logger.debug("QueryParser.parse() failed", exc_info=True)

    # Resolve effective filters: explicit > parsed
    effective_year = year_filter
    if not effective_year and parsed and parsed.year:
        effective_year = parsed.year

    effective_project = project_name_filter
    if not effective_project and parsed and parsed.project:
        effective_project = parsed.project

    effective_profile = resolve_search_profile(
        query_text,
        search_profile=search_profile,
        parsed=parsed,
        doc_type_filter=doc_type_filter,
        doc_type_ext_filter=doc_type_ext_filter,
    )
    policy = get_search_profile_policy(effective_profile)
    fetch_multiplier = policy.fetch_multiplier
    if effective_project and not _is_project_filter_reliable(query_text, effective_project):
        effective_project = None

    effective_doc_type_filter = doc_type_filter or policy.default_doc_type_filter

    # 2. Run search — hybrid (vector+FTS) or vector-only
    fts_settings = _get_fts_settings()
    use_hybrid = fts_settings["enabled"] and chunk_fts is not None

    if use_hybrid:
        chunk_results = store.search_hybrid(
            query_embedding,
            query_text=query_text,
            n_results=n_results * fetch_multiplier,
            doc_type_filter=effective_doc_type_filter,
            category_filter=category_filter,
            doc_type_ext_filter=doc_type_ext_filter,
            exclude_doc_ids=exclude_doc_ids,
            year_filter=effective_year,
            project_name_filter=effective_project,
            chunk_fts=chunk_fts,
            fts_weight=fts_settings["fts_weight"],
            vector_weight=fts_settings["vector_weight"],
        )
    else:
        chunk_results = store.search_rrf(
            query_embedding,
            n_results=n_results * fetch_multiplier,
            query_text=query_text,
            doc_type_filter=effective_doc_type_filter,
            category_filter=category_filter,
            doc_type_ext_filter=doc_type_ext_filter,
            exclude_doc_ids=exclude_doc_ids,
            year_filter=effective_year,
            project_name_filter=effective_project,
        )

    # 3. Aggregate into document-level results
    aggregator = SearchAggregator()

    # Collect doc-level FTS bonus scores (with metadata filters)
    fts_doc_scores: dict[str, float] = {}
    if registry and query_text:
        fts_doc_scores = _get_fts_doc_bonus(
            registry, query_text,
            project_name=effective_project,
            year=effective_year,
        )

    doc_results = aggregator.aggregate(
        chunk_results,
        query_project=effective_project or "",
        query_year=parsed.year if parsed else 0,
        query_doc_type=parsed.doc_type if parsed else "",
        search_profile=effective_profile,
        fts_doc_scores=fts_doc_scores,
    )

    doc_map: dict[str, dict[str, Any]] = {}
    if registry and doc_results:
        doc_map = _hydrate_doc_results(doc_results, registry)
    if doc_map:
        _apply_profile_metadata_bonus(
            doc_results,
            effective_profile,
            query_text,
            doc_map,
        )
    doc_results = rank_profile_results(doc_results, profile=effective_profile)

    return doc_results[:n_results], parsed


def _get_fts_settings() -> dict[str, Any]:
    """Get FTS settings with graceful fallback."""
    try:
        from doc_pipeline.config import settings

        return {
            "enabled": settings.fts.enabled,
            "fts_weight": settings.fts.fts_weight,
            "vector_weight": settings.fts.vector_weight,
        }
    except Exception:
        return {"enabled": False, "fts_weight": 0.3, "vector_weight": 0.7}


def _get_fts_doc_bonus(
    registry: Any,
    query_text: str,
    project_name: str | None = None,
    year: int | None = None,
) -> dict[str, float]:
    """Get doc-level FTS scores from registry, normalized to [0, 1]."""
    try:
        fts_results = registry.search_fts(
            query_text, limit=20,
            project_name=project_name,
            year=year,
        )
        if not fts_results:
            return {}
        # Normalize rank scores (FTS5 rank is negative, lower = better)
        scores: dict[str, float] = {}
        if fts_results:
            best_rank = abs(fts_results[0]["rank"]) if fts_results[0]["rank"] else 1.0
            for r in fts_results:
                # Convert rank to [0, 1] score (best match = 1.0)
                raw = abs(r["rank"]) if r["rank"] else best_rank
                scores[r["doc_id"]] = best_rank / raw if raw > 0 else 0.0
        return scores
    except Exception:
        logger.debug("Doc-level FTS bonus failed", exc_info=True)
        return {}


def _hydrate_doc_results(
    doc_results: list[DocumentResult],
    registry: Any,
) -> dict[str, dict[str, Any]]:
    """Fill missing metadata on aggregated results using registry batch fetch."""
    try:
        doc_map = registry.get_documents_batch([d.doc_id for d in doc_results])
    except Exception:
        logger.debug("Registry batch hydration failed", exc_info=True)
        return {}

    for doc in doc_results:
        record = doc_map.get(doc.doc_id)
        if not record:
            continue
        if not doc.doc_type:
            doc.doc_type = record.get("doc_type", "")
        if not doc.doc_type_ext:
            doc.doc_type_ext = record.get("doc_type_ext", "")
        if not doc.category:
            doc.category = record.get("category", "")
        if not doc.project_name:
            doc.project_name = record.get("project_name", "")
        if not doc.year:
            doc.year = int(record.get("year", 0) or 0)
        _hydrate_chunk(doc.best_chunk, record)
        for chunk in doc.top_chunks:
            _hydrate_chunk(chunk, record)
    return doc_map


def _hydrate_chunk(chunk: Any, record: dict[str, Any]) -> None:
    """Fill missing chunk metadata from registry metadata when possible."""
    if not chunk.doc_type:
        chunk.doc_type = record.get("doc_type", "")
    if not chunk.doc_type_ext:
        chunk.doc_type_ext = record.get("doc_type_ext", "")
    if not chunk.category:
        chunk.category = record.get("category", "")
    if not chunk.project_name:
        chunk.project_name = record.get("project_name", "")
    if not chunk.year:
        chunk.year = int(record.get("year", 0) or 0)


def _apply_profile_metadata_bonus(
    doc_results: list[DocumentResult],
    profile: str,
    query_text: str,
    doc_map: dict[str, dict[str, Any]],
) -> None:
    """Apply profile-specific bonuses using hydrated registry metadata."""
    if not doc_results or not query_text:
        return

    base_bonus = {
        "technical_qa": 0.08,
        "project_lookup": 0.22,
        "contract_lookup": 0.18,
        "method_docs": 0.28,
    }.get(profile, 0.0)
    if base_bonus <= 0:
        return

    query_tokens = _query_tokens(query_text)
    if not query_tokens:
        return

    for doc in doc_results:
        record = doc_map.get(doc.doc_id, {})
        project_name = str(record.get("project_name", "") or doc.project_name or "").lower()
        file_name = str(record.get("file_name_original", "") or "").lower()
        if profile == "project_lookup" and not project_name:
            continue
        overlap = sum(1 for token in query_tokens if token in project_name or token in file_name)
        if overlap == 0:
            continue
        ratio = overlap / len(query_tokens)
        bonus = base_bonus * ratio
        if profile == "method_docs" and doc.doc_type == "공법자료":
            bonus *= 1.2
        doc.doc_score += bonus


def _query_tokens(query_text: str) -> list[str]:
    """Extract simple overlap tokens from a raw query."""
    generic = {"구조", "검토", "설계", "자료", "공법", "용역", "계약서", "계약"}
    tokens = []
    for token in query_text.lower().replace("-", " ").split():
        cleaned = token.strip()
        if len(cleaned) < 2:
            continue
        if cleaned in generic:
            continue
        tokens.append(cleaned)
    return tokens


def _is_project_filter_reliable(query_text: str, project_name: str) -> bool:
    """Return True when the parsed project has enough overlap with the raw query."""
    if not query_text or not project_name:
        return False

    query_lower = query_text.lower()
    project_lower = project_name.lower()
    if project_lower in query_lower:
        return True

    overlap = [token for token in _query_tokens(query_text) if token in project_lower]
    if len(overlap) >= 2:
        return True
    return sum(len(token) for token in overlap) >= 5
