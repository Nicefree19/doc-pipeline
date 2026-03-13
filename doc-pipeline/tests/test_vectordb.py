"""Tests for doc_pipeline.storage.vectordb module."""

from __future__ import annotations

from pathlib import Path

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


    def test_rrf_larger_pool(self, tmp_chromadb: str) -> None:
        """search_rrf uses 5x candidate pool."""
        store = VectorStore(persist_dir=tmp_chromadb)
        chunks = [
            _make_chunk(f"c{i}", doc_id=f"d{i}", chunk_index=0, text=f"테스트 {i}")
            for i in range(10)
        ]
        embeddings = [[0.1 * (i + 1)] * 10 for i in range(10)]
        store.add_chunks(chunks, embeddings)

        # n_results=2 → internal fetch_n should be 10 (2*5)
        results = store.search_rrf([0.1] * 10, n_results=2)
        assert len(results) == 2
        assert results[0].rrf_score >= results[1].rrf_score


class TestBuildWhereClause:
    """Test _build_where_clause with all filter parameters."""

    def test_where_clause_year_filter(self) -> None:
        clause = VectorStore._build_where_clause(None, None, None, None, year_filter=2024)
        assert clause == {"year": 2024}

    def test_where_clause_project_filter(self) -> None:
        clause = VectorStore._build_where_clause(None, None, None, None, project_name_filter="화성동탄")
        assert clause == {"project_name": "화성동탄"}

    def test_where_clause_combined_all(self) -> None:
        clause = VectorStore._build_where_clause(
            "의견서", "구조", None, ["ex1"], year_filter=2024, project_name_filter="화성동탄",
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
        clause = VectorStore._build_where_clause(None, None, None, None, year_filter=0)
        assert clause is None

    def test_where_clause_all_none(self) -> None:
        """All args None returns None (no filter)."""
        clause = VectorStore._build_where_clause(None, None, None, None)
        assert clause is None

    def test_where_clause_doc_type_ext_only(self) -> None:
        """doc_type_ext_filter alone returns single condition (no $and)."""
        clause = VectorStore._build_where_clause(None, None, "매뉴얼", None)
        assert clause == {"doc_type_ext": "매뉴얼"}

    def test_where_clause_category_only(self) -> None:
        """category_filter alone returns single condition (no $and)."""
        clause = VectorStore._build_where_clause(None, "구조", None, None)
        assert clause == {"category": "구조"}

    def test_where_clause_exclude_only(self) -> None:
        """exclude_doc_ids alone returns single $nin condition."""
        clause = VectorStore._build_where_clause(None, None, None, ["ex1"])
        assert clause == {"doc_id": {"$nin": ["ex1"]}}


class TestDeleteByDocIds:
    def test_delete_by_doc_ids(self, tmp_chromadb: str) -> None:
        """delete_by_doc_ids removes chunks for specified doc_ids."""
        store = VectorStore(tmp_chromadb)
        # Insert chunks for 3 doc_ids
        for doc_id in ("d1", "d2", "d3"):
            chunks = [_make_chunk(f"{doc_id}_0", doc_id=doc_id, text=f"text for {doc_id}")]
            store.upsert_chunks_local(chunks)
        assert store.count == 3

        deleted = store.delete_by_doc_ids(["d1", "d2"])
        assert deleted == 2
        assert store.count == 1

        # d3 still exists
        remaining = store.get_chunks_by_doc_id("d3")
        assert len(remaining) == 1

    def test_delete_by_doc_ids_empty(self, tmp_chromadb: str) -> None:
        """delete_by_doc_ids with empty list returns 0."""
        store = VectorStore(tmp_chromadb)
        assert store.delete_by_doc_ids([]) == 0

    def test_delete_by_doc_ids_both_collections(self, tmp_chromadb: str) -> None:
        """delete_by_doc_ids removes from both API and local collections."""
        store = VectorStore(tmp_chromadb)

        # Insert into API collection (needs embeddings)
        api_chunk = _make_chunk("d1_api_0", doc_id="d1", text="api text")
        store.upsert_chunks([api_chunk], [[0.1] * 384])

        # Insert into local collection
        local_chunk = _make_chunk("d1_local_0", doc_id="d1", text="local text")
        store.upsert_chunks_local([local_chunk])

        assert store._collection.count() == 1
        assert store._local_collection.count() == 1
        assert store.count == 2

        deleted = store.delete_by_doc_ids(["d1"])
        assert deleted == 2
        assert store.count == 0


class TestFTSDeleteByDocIds:
    def test_fts_delete_by_doc_ids(self, tmp_path: Path) -> None:
        """ChunkFTS.delete_by_doc_ids removes entries for given doc_ids."""
        from doc_pipeline.storage.vectordb import ChunkFTS

        fts = ChunkFTS(db_path=str(tmp_path / "test_fts.db"))

        # Insert chunks for 3 doc_ids
        for doc_id in ("d1", "d2", "d3"):
            chunks = [_make_chunk(f"{doc_id}_0", doc_id=doc_id, text=f"text for {doc_id}")]
            fts.upsert(chunks)
        assert fts.count == 3

        deleted = fts.delete_by_doc_ids(["d1", "d2"])
        assert deleted == 2
        assert fts.count == 1


class TestChunkFTSPhraseFirst:
    """ChunkFTS.search uses phrase -> AND -> OR fallback."""

    def test_chunk_fts_phrase_first(self, tmp_path: Path) -> None:
        from doc_pipeline.storage.vectordb import ChunkFTS

        fts = ChunkFTS(db_path=str(tmp_path / "fts_phrase.db"))
        c1 = _make_chunk("c1", doc_id="d1", text="강릉시 송정동 공동주택 구조검토")
        c2 = _make_chunk("c2", doc_id="d2", text="강릉시 다른문서 내용")
        c3 = _make_chunk("c3", doc_id="d3", text="송정동 별도 자료")
        fts.upsert([c1, c2, c3])

        results = fts.search("강릉시 송정동")
        assert len(results) >= 1
        # Phrase match should rank d1 highest
        assert results[0]["doc_id"] == "d1"

    def test_chunk_fts_metadata_filter(self, tmp_path: Path) -> None:
        """project_name_filter limits results to matching chunks."""
        from doc_pipeline.storage.vectordb import ChunkFTS

        fts = ChunkFTS(db_path=str(tmp_path / "fts_filter.db"))
        c1 = ChunkRecord(
            chunk_id="c1", doc_id="d1", doc_type=DocType.OPINION,
            project_name="프로젝트A", year=2024, chunk_index=0,
            text="구조검토 의견서", security_grade=SecurityGrade.C,
        )
        c2 = ChunkRecord(
            chunk_id="c2", doc_id="d2", doc_type=DocType.OPINION,
            project_name="프로젝트B", year=2024, chunk_index=0,
            text="구조검토 보고서", security_grade=SecurityGrade.C,
        )
        fts.upsert([c1, c2])

        results = fts.search("구조검토", project_name_filter="프로젝트A")
        assert len(results) == 1
        assert results[0]["doc_id"] == "d1"
        assert results[0]["project_name"] == "프로젝트A"

    def test_chunk_fts_returns_metadata(self, tmp_path: Path) -> None:
        """FTS results include project_name and doc_type_ext."""
        from doc_pipeline.storage.vectordb import ChunkFTS

        fts = ChunkFTS(db_path=str(tmp_path / "fts_meta.db"))
        c1 = ChunkRecord(
            chunk_id="c1", doc_id="d1", doc_type=DocType.OPINION,
            doc_type_ext="구조검토의견서", project_name="테스트PJ",
            year=2024, chunk_index=0, text="구조검토 의견서 슬래브 보강",
            security_grade=SecurityGrade.C,
        )
        fts.upsert([c1])

        results = fts.search("구조검토 슬래브")
        assert len(results) == 1
        assert results[0]["project_name"] == "테스트PJ"
        assert results[0]["doc_type_ext"] == "구조검토의견서"


class TestHybridFTSOnlyHitMetadata:
    """search_hybrid fills metadata for FTS-only hits."""

    def test_hybrid_fts_only_hit_metadata(self, tmp_chromadb: str, tmp_path: Path) -> None:
        from doc_pipeline.storage.vectordb import ChunkFTS

        store = VectorStore(persist_dir=tmp_chromadb)
        fts = ChunkFTS(db_path=str(tmp_path / "fts_hybrid.db"))

        # Only add to FTS (not vector store) — becomes FTS-only hit
        c1 = ChunkRecord(
            chunk_id="d1_0", doc_id="d1", doc_type=DocType.OPINION,
            doc_type_ext="의견서", project_name="테스트프로젝트",
            year=2024, chunk_index=0, text="강릉시 구조검토",
            security_grade=SecurityGrade.B,
        )
        fts.upsert([c1])

        # Add a different chunk to vector store so search_rrf returns something
        c2 = _make_chunk("d2_0", doc_id="d2", text="완전 다른 문서")
        store.upsert_chunks_local([c2])

        results = store.search_hybrid(
            [0.0] * 10, query_text="강릉시 구조검토",
            n_results=5, chunk_fts=fts,
        )

        # Find the FTS-only hit
        fts_hits = [r for r in results if r.doc_id == "d1"]
        assert len(fts_hits) == 1
        assert fts_hits[0].project_name == "테스트프로젝트"
        assert fts_hits[0].doc_type_ext == "의견서"
