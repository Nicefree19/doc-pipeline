"""Unified search entry point.

All search callers (API, CLI, eval, drafter) should use ``unified_search()``
to ensure they go through the same improved pipeline:

    query_parser.parse() → search_rrf() → SearchAggregator.aggregate()

This prevents the problem where eval/CLI/drafter bypass RRF and aggregation
by calling ``store.search()`` directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from doc_pipeline.search.aggregator import DocumentResult, SearchAggregator
from doc_pipeline.search.query_parser import ParsedQuery

if TYPE_CHECKING:
    from doc_pipeline.search.query_parser import QueryParser
    from doc_pipeline.storage.vectordb import VectorStore

logger = logging.getLogger(__name__)


def unified_search(
    store: VectorStore,
    query_text: str,
    query_embedding: list[float],
    *,
    n_results: int = 5,
    doc_type_filter: str | None = None,
    category_filter: str | None = None,
    year_filter: int | None = None,
    project_name_filter: str | None = None,
    exclude_doc_ids: list[str] | None = None,
    query_parser: QueryParser | None = None,
) -> tuple[list[DocumentResult], ParsedQuery | None]:
    """Single search function that all callers should use.

    Internally:
      1. Parse query for metadata hints (if parser provided).
      2. Run ``search_rrf()`` with metadata pre-filtering.
      3. Aggregate chunks into document-level results.

    Args:
        store: VectorStore instance.
        query_text: Raw query string.
        query_embedding: Embedding vector for the query.
        n_results: Desired number of document-level results.
        doc_type_filter: Explicit doc_type filter (user-selected).
        category_filter: Explicit category filter (user-selected).
        year_filter: Explicit year override (takes precedence over parsed).
        project_name_filter: Explicit project override (takes precedence over parsed).
        exclude_doc_ids: Doc IDs to exclude from results.
        query_parser: Optional QueryParser for metadata extraction.

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

    # 2. Run RRF search with larger candidate pool
    chunk_results = store.search_rrf(
        query_embedding,
        n_results=n_results * 3,  # Fetch 3x for aggregation headroom
        query_text=query_text,
        doc_type_filter=doc_type_filter,
        category_filter=category_filter,
        exclude_doc_ids=exclude_doc_ids,
        year_filter=effective_year,
        project_name_filter=effective_project,
    )

    # 3. Aggregate into document-level results
    aggregator = SearchAggregator()
    doc_results = aggregator.aggregate(
        chunk_results,
        query_project=parsed.project if parsed else "",
        query_year=parsed.year if parsed else 0,
        query_doc_type=parsed.doc_type if parsed else "",
    )

    return doc_results[:n_results], parsed
