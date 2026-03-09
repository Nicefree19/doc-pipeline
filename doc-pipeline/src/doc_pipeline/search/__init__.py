"""Search aggregation and document-level ranking."""

from doc_pipeline.search.aggregator import DocumentResult, SearchAggregator
from doc_pipeline.search.query_parser import ParsedQuery, QueryParser
from doc_pipeline.search.unified import unified_search

__all__ = [
    "DocumentResult",
    "ParsedQuery",
    "QueryParser",
    "SearchAggregator",
    "unified_search",
]
