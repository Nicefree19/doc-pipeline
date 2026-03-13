"""Document-level search result aggregation.

Converts chunk-level search results into document-level rankings by
combining per-chunk RRF scores with metadata matching bonuses.

The key insight: a user searching for a document cares about *which
document* is relevant, not which individual chunk is closest. This
module aggregates chunks by doc_id and produces a document-level score.

Score formula::

    doc_score = best_chunk.rrf_score * best_weight
              + avg(top_3_rrf_scores) * avg_top3_weight
              + metadata_bonus * metadata_weight

    metadata_bonus = project_match * 0.5 + year_match * 0.3 + type_match * 0.2
"""

from __future__ import annotations

from dataclasses import dataclass

from doc_pipeline.search.profiles import SearchProfile, get_doc_type_prior
from doc_pipeline.storage.vectordb import SearchResult


@dataclass
class DocumentResult:
    """A document-level search result aggregated from chunks."""

    doc_id: str
    doc_type: str
    doc_type_ext: str
    category: str
    project_name: str
    year: int
    doc_score: float
    best_chunk: SearchResult
    top_chunks: list[SearchResult]
    chunk_count: int
    quality_score: float = 50.0


class SearchAggregator:
    """Aggregates chunk-level SearchResults into document-level rankings.

    Args:
        best_weight: Weight for the best chunk's RRF score.
        avg_top3_weight: Weight for the average of top-3 chunk scores.
        metadata_weight: Weight for metadata matching bonus.
    """

    def __init__(
        self,
        best_weight: float = 0.5,
        avg_top3_weight: float = 0.3,
        metadata_weight: float = 0.2,
        fts_doc_weight: float = 0.15,
    ) -> None:
        self.best_weight = best_weight
        self.avg_top3_weight = avg_top3_weight
        self.metadata_weight = metadata_weight
        self.fts_doc_weight = fts_doc_weight

    def aggregate(
        self,
        chunk_results: list[SearchResult],
        *,
        query_project: str = "",
        query_year: int = 0,
        query_doc_type: str = "",
        search_profile: SearchProfile = "auto",
        fts_doc_scores: dict[str, float] | None = None,
    ) -> list[DocumentResult]:
        """Aggregate chunk results into document-level results.

        Args:
            chunk_results: Chunk-level search results (from search_rrf).
            query_project: Extracted project name from query for bonus.
            query_year: Extracted year from query for bonus.
            query_doc_type: Extracted document type from query for bonus.
            fts_doc_scores: Optional doc-level FTS scores {doc_id: score}.
                Applied as additive bonus to doc_score.

        Returns:
            Document-level results sorted by doc_score descending.
        """
        if not chunk_results:
            return []

        # Group chunks by doc_id
        doc_chunks: dict[str, list[SearchResult]] = {}
        for r in chunk_results:
            doc_chunks.setdefault(r.doc_id, []).append(r)

        doc_results: list[DocumentResult] = []

        for doc_id, chunks in doc_chunks.items():
            # Sort by RRF score descending
            chunks.sort(key=lambda c: c.rrf_score, reverse=True)

            best = chunks[0]
            top3 = chunks[:3]

            # Compute score components
            best_score = best.rrf_score
            avg_top3 = sum(c.rrf_score for c in top3) / len(top3)
            metadata_bonus = self._metadata_bonus(
                best, query_project, query_year, query_doc_type,
            )
            type_prior = 0.0
            if search_profile != "auto":
                type_prior = get_doc_type_prior(
                    best.doc_type,
                    doc_type_ext=best.doc_type_ext,
                    profile=search_profile,
                )

            doc_score = (
                best_score * self.best_weight
                + avg_top3 * self.avg_top3_weight
                + metadata_bonus * self.metadata_weight
                + type_prior
            )

            # Add doc-level FTS bonus (if available)
            if fts_doc_scores and doc_id in fts_doc_scores:
                doc_score += fts_doc_scores[doc_id] * self.fts_doc_weight

            doc_results.append(DocumentResult(
                doc_id=doc_id,
                doc_type=best.doc_type,
                doc_type_ext=best.doc_type_ext,
                category=best.category,
                project_name=best.project_name,
                year=best.year,
                doc_score=doc_score,
                best_chunk=best,
                top_chunks=top3,
                chunk_count=len(chunks),
            ))

        # Sort by doc_score descending
        doc_results.sort(key=lambda d: d.doc_score, reverse=True)
        return doc_results

    @staticmethod
    def _metadata_bonus(
        chunk: SearchResult,
        query_project: str,
        query_year: int,
        query_doc_type: str = "",
    ) -> float:
        """Compute metadata matching bonus ∈ [0.0, 1.0].

        - Project name match: 0.5 weight
        - Year match: 0.3 weight
        - Document type match: 0.2 weight
        """
        bonus = 0.0

        if query_project and chunk.project_name:
            # Exact or substring match
            qp = query_project.lower()
            cp = chunk.project_name.lower()
            if qp in cp or cp in qp:
                bonus += 0.5

        if query_year and chunk.year and chunk.year == query_year:
            bonus += 0.3

        if query_doc_type and chunk.doc_type_ext:
            if query_doc_type == chunk.doc_type_ext:
                bonus += 0.2
            elif query_doc_type in chunk.doc_type_ext or chunk.doc_type_ext in query_doc_type:
                bonus += 0.1

        return bonus
