"""Search aggregation and document-level ranking."""

from doc_pipeline.search.aggregator import DocumentResult, SearchAggregator
from doc_pipeline.search.profiles import (
    SearchProfile,
    get_search_profile_policy,
    resolve_search_profile,
)
from doc_pipeline.search.query_parser import ParsedQuery, QueryParser
from doc_pipeline.search.unified import unified_search

__all__ = [
    "DocumentResult",
    "ParsedQuery",
    "QueryParser",
    "SearchProfile",
    "SearchAggregator",
    "get_search_profile_policy",
    "resolve_search_profile",
    "unified_search",
]
