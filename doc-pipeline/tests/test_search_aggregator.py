"""Tests for doc_pipeline.search.aggregator module."""

from __future__ import annotations

import pytest

from doc_pipeline.search.aggregator import DocumentResult, SearchAggregator
from doc_pipeline.storage.vectordb import SearchResult


def _sr(
    doc_id: str = "d1",
    rrf_score: float = 0.016,
    chunk_index: int = 0,
    project_name: str = "테스트PJ",
    year: int = 2024,
    text: str = "chunk text",
    doc_type: str = "의견서",
    doc_type_ext: str = "",
    category: str = "",
) -> SearchResult:
    """Helper to create a SearchResult with sane defaults."""
    return SearchResult(
        doc_id=doc_id,
        doc_type=doc_type,
        project_name=project_name,
        text=text,
        distance=0.3,
        chunk_index=chunk_index,
        year=year,
        rrf_score=rrf_score,
        doc_type_ext=doc_type_ext,
        category=category,
    )


class TestSearchAggregator:
    def test_empty_input(self) -> None:
        agg = SearchAggregator()
        assert agg.aggregate([]) == []

    def test_single_chunk_single_doc(self) -> None:
        agg = SearchAggregator()
        results = agg.aggregate([_sr(rrf_score=0.016)])
        assert len(results) == 1
        assert results[0].doc_id == "d1"
        assert results[0].chunk_count == 1
        assert results[0].doc_score > 0

    def test_multiple_chunks_same_doc(self) -> None:
        agg = SearchAggregator()
        chunks = [
            _sr(chunk_index=0, rrf_score=0.016),
            _sr(chunk_index=1, rrf_score=0.015),
            _sr(chunk_index=2, rrf_score=0.014),
            _sr(chunk_index=3, rrf_score=0.010),
        ]
        results = agg.aggregate(chunks)
        assert len(results) == 1
        assert results[0].chunk_count == 4
        assert len(results[0].top_chunks) == 3  # top 3

    def test_multiple_docs_ranking(self) -> None:
        """Doc with higher best-chunk score should rank first."""
        agg = SearchAggregator()
        chunks = [
            _sr(doc_id="d1", chunk_index=0, rrf_score=0.010),
            _sr(doc_id="d2", chunk_index=0, rrf_score=0.016),
        ]
        results = agg.aggregate(chunks)
        assert len(results) == 2
        assert results[0].doc_id == "d2"  # Higher score
        assert results[1].doc_id == "d1"

    def test_chunk_imbalance(self) -> None:
        """Doc with many low-score chunks vs doc with few high-score chunks."""
        agg = SearchAggregator()
        # d1: 5 low-score chunks
        many_chunks = [_sr(doc_id="d1", chunk_index=i, rrf_score=0.005) for i in range(5)]
        # d2: 1 high-score chunk
        one_chunk = [_sr(doc_id="d2", chunk_index=0, rrf_score=0.016)]

        results = agg.aggregate(many_chunks + one_chunk)
        assert results[0].doc_id == "d2"  # Quality over quantity

    def test_doc_score_formula(self) -> None:
        """Verify score formula with known weights."""
        agg = SearchAggregator(best_weight=0.5, avg_top3_weight=0.3, metadata_weight=0.2)
        chunks = [
            _sr(chunk_index=0, rrf_score=0.020),
            _sr(chunk_index=1, rrf_score=0.015),
            _sr(chunk_index=2, rrf_score=0.010),
        ]
        results = agg.aggregate(chunks)
        r = results[0]

        expected_best = 0.020
        expected_avg = (0.020 + 0.015 + 0.010) / 3
        expected = expected_best * 0.5 + expected_avg * 0.3 + 0.0 * 0.2

        assert r.doc_score == pytest.approx(expected, abs=0.0001)

    def test_project_match_bonus(self) -> None:
        """Project name matching boosts doc_score."""
        agg = SearchAggregator()
        chunks = [
            _sr(doc_id="d1", project_name="화성동탄", rrf_score=0.016),
            _sr(doc_id="d2", project_name="판교신도시", rrf_score=0.016),
        ]
        results = agg.aggregate(chunks, query_project="화성동탄")
        # d1 should be boosted by project match
        assert results[0].doc_id == "d1"
        assert results[0].doc_score > results[1].doc_score

    def test_year_match_bonus(self) -> None:
        """Year matching boosts doc_score."""
        agg = SearchAggregator()
        chunks = [
            _sr(doc_id="d1", year=2024, rrf_score=0.016),
            _sr(doc_id="d2", year=2023, rrf_score=0.016),
        ]
        results = agg.aggregate(chunks, query_year=2024)
        assert results[0].doc_id == "d1"

    def test_project_and_year_bonus(self) -> None:
        """Both project + year match gives significant bonus."""
        agg = SearchAggregator()
        chunks = [
            _sr(doc_id="d1", project_name="A", year=2024, rrf_score=0.016),
            _sr(doc_id="d2", project_name="B", year=2023, rrf_score=0.016),
        ]
        results = agg.aggregate(chunks, query_project="A", query_year=2024)
        assert results[0].doc_id == "d1"
        # Project(0.5) + Year(0.3) = 0.8 (type match adds 0.2 when present)
        expected_bonus = agg.metadata_weight * 0.8
        no_bonus_score = results[1].doc_score
        boosted_score = results[0].doc_score
        assert boosted_score - no_bonus_score == pytest.approx(expected_bonus, abs=0.001)

    def test_full_metadata_bonus(self) -> None:
        """Project + year + doc_type match gives maximum bonus."""
        agg = SearchAggregator()
        chunks = [
            _sr(doc_id="d1", project_name="A", year=2024,
                doc_type_ext="의견서", rrf_score=0.016),
            _sr(doc_id="d2", project_name="B", year=2023,
                doc_type_ext="계약서", rrf_score=0.016),
        ]
        results = agg.aggregate(
            chunks, query_project="A", query_year=2024, query_doc_type="의견서",
        )
        assert results[0].doc_id == "d1"
        # Full bonus = 0.5 + 0.3 + 0.2 = 1.0
        full_bonus = agg.metadata_weight * 1.0
        no_bonus_score = results[1].doc_score
        boosted_score = results[0].doc_score
        assert boosted_score - no_bonus_score == pytest.approx(full_bonus, abs=0.001)

    def test_doc_type_partial_match_bonus(self) -> None:
        """Substring doc_type match gives half bonus (0.1)."""
        agg = SearchAggregator()
        chunks = [
            _sr(doc_id="d1", doc_type_ext="구조검토의견서", rrf_score=0.016),
            _sr(doc_id="d2", doc_type_ext="계약서", rrf_score=0.016),
        ]
        results = agg.aggregate(chunks, query_doc_type="의견서")
        assert results[0].doc_id == "d1"

    def test_best_chunk_preserved(self) -> None:
        """best_chunk should be the chunk with the highest RRF score."""
        agg = SearchAggregator()
        chunks = [
            _sr(chunk_index=0, rrf_score=0.010, text="low"),
            _sr(chunk_index=1, rrf_score=0.020, text="high"),
        ]
        results = agg.aggregate(chunks)
        assert results[0].best_chunk.text == "high"
        assert results[0].best_chunk.rrf_score == 0.020

    def test_document_result_fields(self) -> None:
        """DocumentResult carries correct metadata from best chunk."""
        agg = SearchAggregator()
        chunks = [_sr(
            doc_type_ext="구조검토의견서", category="구조",
            project_name="테스트PJ", year=2024,
        )]
        results = agg.aggregate(chunks)
        r = results[0]
        assert r.doc_type_ext == "구조검토의견서"
        assert r.category == "구조"
        assert r.project_name == "테스트PJ"
        assert r.year == 2024

    def test_substring_project_match(self) -> None:
        """Partial project name match should also give bonus."""
        agg = SearchAggregator()
        chunks = [
            _sr(doc_id="d1", project_name="화성동탄2 아파트"),
        ]
        results = agg.aggregate(chunks, query_project="화성동탄")
        # Should get project bonus from substring match
        no_bonus = agg.aggregate(chunks)
        assert results[0].doc_score > no_bonus[0].doc_score

    def test_fts_doc_bonus_increases_score(self) -> None:
        """Doc-level FTS bonus should increase the document score."""
        agg = SearchAggregator()
        chunks = [_sr(doc_id="d1"), _sr(doc_id="d2")]
        no_fts = agg.aggregate(chunks)
        with_fts = agg.aggregate(chunks, fts_doc_scores={"d1": 1.0})
        # d1 should score higher with FTS bonus
        d1_no = next(d for d in no_fts if d.doc_id == "d1")
        d1_yes = next(d for d in with_fts if d.doc_id == "d1")
        assert d1_yes.doc_score > d1_no.doc_score

    def test_fts_doc_weight_configurable(self) -> None:
        """fts_doc_weight parameter should control FTS bonus magnitude."""
        chunks = [_sr(doc_id="d1")]
        agg_low = SearchAggregator(fts_doc_weight=0.05)
        agg_high = SearchAggregator(fts_doc_weight=0.3)
        low = agg_low.aggregate(chunks, fts_doc_scores={"d1": 1.0})
        high = agg_high.aggregate(chunks, fts_doc_scores={"d1": 1.0})
        assert high[0].doc_score > low[0].doc_score

    def test_technical_qa_prefers_opinion_over_contract(self) -> None:
        """technical_qa should boost opinion docs over contracts at equal base score."""
        agg = SearchAggregator()
        chunks = [
            _sr(doc_id="contract", doc_type="계약서", rrf_score=0.016),
            _sr(doc_id="opinion", doc_type="의견서", rrf_score=0.016),
        ]
        results = agg.aggregate(chunks, search_profile="technical_qa")
        assert results[0].doc_id == "opinion"

    def test_contract_lookup_prefers_contract(self) -> None:
        """contract_lookup should strongly boost contract docs."""
        agg = SearchAggregator()
        chunks = [
            _sr(doc_id="contract", doc_type="계약서", rrf_score=0.016),
            _sr(doc_id="opinion", doc_type="의견서", rrf_score=0.016),
        ]
        results = agg.aggregate(chunks, search_profile="contract_lookup")
        assert results[0].doc_id == "contract"

    def test_method_docs_prefers_method_doc(self) -> None:
        """method_docs should prefer 공법자료 docs over contracts."""
        agg = SearchAggregator()
        chunks = [
            _sr(doc_id="contract", doc_type="계약서", rrf_score=0.016),
            _sr(doc_id="method", doc_type="공법자료", rrf_score=0.016),
        ]
        results = agg.aggregate(chunks, search_profile="method_docs")
        assert results[0].doc_id == "method"

    def test_fts_doc_bonus_reranks(self) -> None:
        """FTS bonus can rerank documents when vector scores are close."""
        chunks = [
            _sr(doc_id="d1", rrf_score=0.016),
            _sr(doc_id="d2", rrf_score=0.015),
        ]
        agg = SearchAggregator(fts_doc_weight=0.15)
        # Without FTS: d1 should be first (higher rrf)
        no_fts = agg.aggregate(chunks)
        assert no_fts[0].doc_id == "d1"
        # With FTS bonus on d2: d2 should overtake d1
        with_fts = agg.aggregate(chunks, fts_doc_scores={"d2": 1.0})
        assert with_fts[0].doc_id == "d2"


class TestRegistryBatchLookup:
    """Test get_documents_batch (requires registry fixture from conftest)."""

    def test_batch_empty(self, tmp_path) -> None:
        from doc_pipeline.storage.registry import DocumentRegistry
        reg = DocumentRegistry(db_path=str(tmp_path / "test.db"))
        assert reg.get_documents_batch([]) == {}

    def test_batch_single(self, tmp_path) -> None:
        from doc_pipeline.models.schemas import DocMaster, DocType, ProcessStatus, SecurityGrade, SourceFormat
        from doc_pipeline.storage.registry import DocumentRegistry

        reg = DocumentRegistry(db_path=str(tmp_path / "test.db"))
        doc = DocMaster(
            doc_id="d1", file_name_original="test.pdf",
            doc_type=DocType.OPINION, project_name="PJ",
            year=2024, page_count=1,
            process_status=ProcessStatus.COMPLETED,
            security_grade=SecurityGrade.C,
            source_format=SourceFormat.PDF,
        )
        reg.insert_document(doc, source_path="/test")
        result = reg.get_documents_batch(["d1"])
        assert "d1" in result
        assert result["d1"]["project_name"] == "PJ"

    def test_batch_multiple(self, tmp_path) -> None:
        from doc_pipeline.models.schemas import DocMaster, DocType, ProcessStatus, SecurityGrade, SourceFormat
        from doc_pipeline.storage.registry import DocumentRegistry

        reg = DocumentRegistry(db_path=str(tmp_path / "test.db"))
        for i in range(3):
            doc = DocMaster(
                doc_id=f"d{i}", file_name_original=f"test{i}.pdf",
                doc_type=DocType.OPINION, project_name=f"PJ{i}",
                year=2024, page_count=1,
                process_status=ProcessStatus.COMPLETED,
                security_grade=SecurityGrade.C,
                source_format=SourceFormat.PDF,
            )
            reg.insert_document(doc, source_path=f"/test{i}")

        result = reg.get_documents_batch(["d0", "d1", "d2", "nonexistent"])
        assert len(result) == 3
        assert "nonexistent" not in result
