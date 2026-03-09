"""Tests for doc_pipeline.storage.vectordb module."""

from __future__ import annotations

import pytest

from doc_pipeline.models.schemas import ChunkRecord, DocType, SecurityGrade
from doc_pipeline.storage.vectordb import SearchResult, VectorStore


def _make_chunk(
    chunk_id: str = "c1",
    doc_id: str = "d1",
    text: str = "테스트 청크 텍스트",
    chunk_index: int = 0,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_type=DocType.OPINION,
        project_name="테스트",
        year=2024,
        chunk_index=chunk_index,
        text=text,
        security_grade=SecurityGrade.C,
    )


class TestVectorStore:
    def test_init_creates_collection(self, tmp_chromadb: str) -> None:
        store = VectorStore(persist_dir=tmp_chromadb)
        assert store.count == 0

    def test_add_chunks(self, tmp_chromadb: str) -> None:
        store = VectorStore(persist_dir=tmp_chromadb)
        chunks = [_make_chunk("c1"), _make_chunk("c2", chunk_index=1)]
        embeddings = [[0.1] * 10, [0.2] * 10]
        store.add_chunks(chunks, embeddings)
        assert store.count == 2

    def test_upsert_chunks_idempotent(self, tmp_chromadb: str) -> None:
        store = VectorStore(persist_dir=tmp_chromadb)
        chunk = _make_chunk("c1")
        emb = [[0.1] * 10]
        store.upsert_chunks([chunk], emb)
        store.upsert_chunks([chunk], emb)  # Same ID — no duplicate
        assert store.count == 1

    def test_search_empty_store(self, tmp_chromadb: str) -> None:
        store = VectorStore(persist_dir=tmp_chromadb)
        results = store.search([0.1] * 10, n_results=5)
        assert results == []

    def test_search_with_results(self, tmp_chromadb: str) -> None:
        store = VectorStore(persist_dir=tmp_chromadb)
        chunks = [
            _make_chunk("c1", text="슬래브 균열 보강"),
            _make_chunk("c2", chunk_index=1, text="기둥 보강 설계"),
        ]
        embeddings = [[0.1] * 10, [0.9] * 10]
        store.add_chunks(chunks, embeddings)

        results = store.search([0.1] * 10, n_results=2)
        assert len(results) == 2
        assert results[0].text in ("슬래브 균열 보강", "기둥 보강 설계")

    def test_validate_empty_chunks_raises(self, tmp_chromadb: str) -> None:
        store = VectorStore(persist_dir=tmp_chromadb)
        with pytest.raises(ValueError, match="empty"):
            store.add_chunks([], [])

    def test_validate_mismatch_raises(self, tmp_chromadb: str) -> None:
        store = VectorStore(persist_dir=tmp_chromadb)
        with pytest.raises(ValueError, match="mismatch"):
            store.add_chunks([_make_chunk()], [[0.1] * 10, [0.2] * 10])

    def test_search_with_doc_type_filter(self, tmp_chromadb: str) -> None:
        store = VectorStore(persist_dir=tmp_chromadb)
        chunks = [_make_chunk("c1", text="의견서 텍스트")]
        store.add_chunks(chunks, [[0.1] * 10])

        # Filter for existing type
        results = store.search([0.1] * 10, n_results=5, doc_type_filter="의견서")
        assert len(results) == 1

        # Filter for non-existing type
        results = store.search([0.1] * 10, n_results=5, doc_type_filter="계약서")
        assert len(results) == 0


class TestSearchResultExtended:
    """Test SearchResult extended fields backward compat."""

    def test_defaults(self) -> None:
        r = SearchResult(doc_id="d1", doc_type="t", project_name="p", text="x", distance=0.5)
        assert r.source_collection == ""
        assert r.doc_type_ext == ""
        assert r.chunk_index == 0
        assert r.rrf_score == 0.0
        assert r.page_number is None

    def test_extended_fields(self) -> None:
        r = SearchResult(
            doc_id="d1", doc_type="t", project_name="p", text="x",
            distance=0.5, source_collection="api", doc_type_ext="검토의견서",
            category="구조", page_number=3, chunk_index=2, year=2024,
            rrf_score=0.016,
        )
        assert r.source_collection == "api"
        assert r.doc_type_ext == "검토의견서"
        assert r.page_number == 3
        assert r.year == 2024
        assert r.rrf_score == 0.016


class TestSearchRRF:
    """Test RRF-based search fusion."""

    def test_rrf_api_only(self, tmp_chromadb: str) -> None:
        """RRF with only C-grade (API) collection data."""
        store = VectorStore(persist_dir=tmp_chromadb)
        chunks = [
            _make_chunk("c1", text="슬래브 보강"),
            _make_chunk("c2", chunk_index=1, text="기둥 설계"),
        ]
        store.add_chunks(chunks, [[0.1] * 10, [0.9] * 10])

        results = store.search_rrf([0.1] * 10, n_results=2, query_text="슬래브")
        assert len(results) == 2
        assert all(r.rrf_score > 0 for r in results)
        assert results[0].rrf_score >= results[1].rrf_score

    def test_rrf_local_only(self, tmp_chromadb: str) -> None:
        """RRF with only B-grade (local) collection data."""
        store = VectorStore(persist_dir=tmp_chromadb)
        chunks = [_make_chunk("c1", text="기초 구조 검토")]
        store.upsert_chunks_local(chunks)

        results = store.search_rrf(
            [0.0] * 10, n_results=5, query_text="기초 구조",
        )
        assert len(results) == 1
        assert results[0].rrf_score > 0
        assert results[0].source_collection == "local"

    def test_rrf_both_collections(self, tmp_chromadb: str) -> None:
        """RRF merges results from both collections."""
        store = VectorStore(persist_dir=tmp_chromadb)

        # API collection (C-grade)
        api_chunks = [_make_chunk("api_c1", doc_id="d1", text="슬래브 균열 보강 설계")]
        store.add_chunks(api_chunks, [[0.1] * 10])

        # Local collection (B-grade) — different doc
        local_chunks = [_make_chunk("loc_c1", doc_id="d2", text="기둥 내진 보강 설계")]
        store.upsert_chunks_local(local_chunks)

        results = store.search_rrf(
            [0.1] * 10, n_results=5, query_text="보강 설계",
        )
        assert len(results) == 2
        doc_ids = {r.doc_id for r in results}
        assert doc_ids == {"d1", "d2"}

    def test_rrf_duplicate_chunk_score_sum(self, tmp_chromadb: str) -> None:
        """When same chunk appears in both collections, RRF scores should sum."""
        store = VectorStore(persist_dir=tmp_chromadb)

        # Same doc_id+chunk_index in both collections
        chunk = _make_chunk("c1", doc_id="d1", text="공통 검토 내용")
        store.add_chunks([chunk], [[0.1] * 10])
        store.upsert_chunks_local([chunk])

        results = store.search_rrf(
            [0.1] * 10, n_results=5, query_text="공통 검토",
        )
        # Should be merged into 1 result with summed score
        assert len(results) == 1
        # Score should be sum of both RRF scores
        expected_score = 1.0 / (60 + 1) + 1.0 / (60 + 1)
        assert results[0].rrf_score == pytest.approx(expected_score, abs=0.0001)

    def test_rrf_empty_store(self, tmp_chromadb: str) -> None:
        """RRF on empty store returns empty list."""
        store = VectorStore(persist_dir=tmp_chromadb)
        results = store.search_rrf([0.1] * 10, n_results=5, query_text="test")
        assert results == []

    def test_rrf_empty_embedding(self, tmp_chromadb: str) -> None:
        """RRF with empty embedding returns empty list."""
        store = VectorStore(persist_dir=tmp_chromadb)
        results = store.search_rrf([], n_results=5)
        assert results == []

    def test_rrf_with_filter(self, tmp_chromadb: str) -> None:
        """RRF respects doc_type_filter."""
        store = VectorStore(persist_dir=tmp_chromadb)
        chunks = [_make_chunk("c1", text="의견서 내용")]
        store.add_chunks(chunks, [[0.1] * 10])

        # Should find with matching filter
        results = store.search_rrf(
            [0.1] * 10, n_results=5, doc_type_filter="의견서",
        )
        assert len(results) == 1

        # Should not find with non-matching filter
        results = store.search_rrf(
            [0.1] * 10, n_results=5, doc_type_filter="계약서",
        )
        assert len(results) == 0

    def test_rrf_k_parameter(self, tmp_chromadb: str) -> None:
        """Different rrf_k values affect scores."""
        store = VectorStore(persist_dir=tmp_chromadb)
        chunks = [_make_chunk("c1", text="테스트")]
        store.add_chunks(chunks, [[0.1] * 10])

        r_k60 = store.search_rrf([0.1] * 10, n_results=1, rrf_k=60)
        r_k10 = store.search_rrf([0.1] * 10, n_results=1, rrf_k=10)

        # Lower k → higher score for same rank
        assert r_k10[0].rrf_score > r_k60[0].rrf_score

    def test_enriched_metadata_fields(self, tmp_chromadb: str) -> None:
        """Verify extended metadata fields are populated."""
        store = VectorStore(persist_dir=tmp_chromadb)
        chunk = ChunkRecord(
            chunk_id="c1", doc_id="d1", doc_type=DocType.OPINION,
            project_name="테스트PJ", year=2024, chunk_index=3,
            text="메타데이터 테스트", security_grade=SecurityGrade.C,
            doc_type_ext="구조검토의견서", category="구조",
            page_number=5, ocr_confidence=0.95,
        )
        store.add_chunks([chunk], [[0.1] * 10])

        results = store.search_rrf([0.1] * 10, n_results=1)
        r = results[0]
        assert r.doc_type_ext == "구조검토의견서"
        assert r.category == "구조"
        assert r.page_number == 5
        assert r.chunk_index == 3
        assert r.year == 2024
        assert r.ocr_confidence == pytest.approx(0.95)
        assert r.source_collection == "api"


class TestBuildWhereClause:
    """Test _build_where_clause with year/project filters."""

    def test_where_clause_year_filter(self) -> None:
        clause = VectorStore._build_where_clause(None, None, None, year_filter=2024)
        assert clause == {"year": 2024}

    def test_where_clause_project_filter(self) -> None:
        clause = VectorStore._build_where_clause(None, None, None, project_name_filter="화성동탄")
        assert clause == {"project_name": "화성동탄"}

    def test_where_clause_combined_all(self) -> None:
        clause = VectorStore._build_where_clause(
            "의견서", "구조", ["ex1"], year_filter=2024, project_name_filter="화성동탄",
        )
        assert "$and" in clause
        conditions = clause["$and"]
        assert {"doc_type": "의견서"} in conditions
        assert {"category": "구조"} in conditions
        assert {"doc_id": {"$nin": ["ex1"]}} in conditions
        assert {"year": 2024} in conditions
        assert {"project_name": "화성동탄"} in conditions
        assert len(conditions) == 5

    def test_where_clause_zero_year_ignored(self) -> None:
        clause = VectorStore._build_where_clause(None, None, None, year_filter=0)
        assert clause is None

    def test_rrf_larger_pool(self, tmp_chromadb: str) -> None:
        """search_rrf uses 5x candidate pool."""
        store = VectorStore(persist_dir=tmp_chromadb)
        # Add enough chunks to verify larger pool is requested
        chunks = [
            _make_chunk(f"c{i}", doc_id=f"d{i}", chunk_index=0, text=f"테스트 {i}")
            for i in range(10)
        ]
        embeddings = [[0.1 * (i + 1)] * 10 for i in range(10)]
        store.add_chunks(chunks, embeddings)

        # n_results=2 → internal fetch_n should be 10 (2*5)
        results = store.search_rrf([0.1] * 10, n_results=2)
        assert len(results) == 2
        # All 10 chunks should have been considered (verified by getting top 2)
        assert results[0].rrf_score >= results[1].rrf_score
