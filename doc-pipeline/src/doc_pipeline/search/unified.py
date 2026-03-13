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

    # 2. Run search — hybrid (vector+FTS) or vector-only
    fts_settings = _get_fts_settings()
    use_hybrid = fts_settings["enabled"] and chunk_fts is not None

    if use_hybrid:
        chunk_results = store.search_hybrid(
            query_embedding,
            query_text=query_text,
            n_results=n_results * 3,  # Fetch 3x for aggregation headroom
            doc_type_filter=doc_type_filter,
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
            n_results=n_results * 3,
            query_text=query_text,
            doc_type_filter=doc_type_filter,
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
        query_project=parsed.project if parsed else "",
        query_year=parsed.year if parsed else 0,
        query_doc_type=parsed.doc_type if parsed else "",
        fts_doc_scores=fts_doc_scores,
    )

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
