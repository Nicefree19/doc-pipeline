"""Tests for doc_pipeline.search.unified module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from doc_pipeline.models.schemas import ChunkRecord, DocType, SecurityGrade
from doc_pipeline.search.query_parser import QueryParser
from doc_pipeline.search.unified import unified_search
from doc_pipeline.storage.vectordb import ChunkFTS, VectorStore


def _make_chunk(
    chunk_id: str = "c1",
    doc_id: str = "d1",
    text: str = "테스트 청크 텍스트",
    chunk_index: int = 0,
    project_name: str = "테스트",
    year: int = 2024,
    doc_type_ext: str = "",
    category: str = "",
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_type=DocType.OPINION,
        project_name=project_name,
        year=year,
        chunk_index=chunk_index,
        text=text,
        security_grade=SecurityGrade.C,
        doc_type_ext=doc_type_ext,
        category=category,
    )


class TestUnifiedSearch:
    def test_returns_document_results(self, tmp_chromadb: str) -> None:
        """unified_search returns DocumentResult list."""
        store = VectorStore(persist_dir=tmp_chromadb)
        chunks = [
            _make_chunk("c1", doc_id="d1", text="슬래브 균열 보강"),
            _make_chunk("c2", doc_id="d2", chunk_index=0, text="기둥 설계 검토"),
        ]
        store.add_chunks(chunks, [[0.1] * 10, [0.9] * 10])

        results, parsed = unified_search(
            store, "슬래브", [0.1] * 10, n_results=5,
        )
        assert len(results) > 0
        assert all(hasattr(r, "doc_id") for r in results)
        assert all(hasattr(r, "doc_score") for r in results)
        assert parsed is None  # No parser provided

    def test_passes_metadata_filters(self, tmp_chromadb: str) -> None:
        """unified_search forwards doc_type_filter to search_rrf."""
        store = VectorStore(persist_dir=tmp_chromadb)
        chunks = [_make_chunk("c1", text="의견서 내용")]
        store.add_chunks(chunks, [[0.1] * 10])

        # Should find with matching filter
        results, _ = unified_search(
            store, "의견서", [0.1] * 10, n_results=5,
            doc_type_filter="의견서",
        )
        assert len(results) == 1

        # Should not find with non-matching filter
        results, _ = unified_search(
            store, "계약서", [0.1] * 10, n_results=5,
            doc_type_filter="계약서",
        )
        assert len(results) == 0

    def test_excludes_doc_ids(self, tmp_chromadb: str) -> None:
        """unified_search excludes specified doc IDs."""
        store = VectorStore(persist_dir=tmp_chromadb)
        chunks = [
            _make_chunk("c1", doc_id="d1", text="문서 하나"),
            _make_chunk("c2", doc_id="d2", chunk_index=0, text="문서 둘"),
        ]
        store.add_chunks(chunks, [[0.1] * 10, [0.2] * 10])

        results, _ = unified_search(
            store, "문서", [0.1] * 10, n_results=5,
            exclude_doc_ids=["d1"],
        )
        doc_ids = {r.doc_id for r in results}
        assert "d1" not in doc_ids

    def test_no_parser_works(self, tmp_chromadb: str) -> None:
        """unified_search works without a QueryParser."""
        store = VectorStore(persist_dir=tmp_chromadb)
        chunks = [_make_chunk("c1", text="검토 내용")]
        store.add_chunks(chunks, [[0.1] * 10])

        results, parsed = unified_search(
            store, "검토", [0.1] * 10, n_results=5,
        )
        assert parsed is None
        assert len(results) == 1

    def test_empty_results(self, tmp_chromadb: str) -> None:
        """unified_search returns empty list for empty store."""
        store = VectorStore(persist_dir=tmp_chromadb)
        results, _ = unified_search(
            store, "아무거나", [0.1] * 10, n_results=5,
        )
        assert results == []

    def test_with_query_parser(self, tmp_chromadb: str) -> None:
        """unified_search uses QueryParser to extract metadata."""
        store = VectorStore(persist_dir=tmp_chromadb)
        chunks = [
            _make_chunk("c1", doc_id="d1", text="화성동탄 구조검토", project_name="화성동탄", year=2024),
        ]
        store.add_chunks(chunks, [[0.1] * 10])

        parser = QueryParser(
            known_projects={"화성동탄"},
            type_keywords={"구조검토의견서": ["구조검토"]},
        )
        results, parsed = unified_search(
            store, "화성동탄 구조검토", [0.1] * 10, n_results=5,
            query_parser=parser,
        )
        assert parsed is not None
        assert parsed.project == "화성동탄"
        assert len(results) >= 1

    def test_n_results_limit(self, tmp_chromadb: str) -> None:
        """unified_search respects n_results limit."""
        store = VectorStore(persist_dir=tmp_chromadb)
        chunks = [
            _make_chunk(f"c{i}", doc_id=f"d{i}", chunk_index=0, text=f"문서 {i}")
            for i in range(5)
        ]
        embeddings = [[0.1 * (i + 1)] * 10 for i in range(5)]
        store.add_chunks(chunks, embeddings)

        results, _ = unified_search(
            store, "문서", [0.1] * 10, n_results=2,
        )
        assert len(results) <= 2


class TestUnifiedSearchIntegration:
    """Integration tests: parser + hybrid FTS + doc-level FTS bonus together."""

    def test_parser_year_project_with_hybrid_fts(self, tmp_path, tmp_chromadb: str) -> None:
        """Full pipeline: parser extracts year+project, hybrid FTS filters,
        doc-level FTS bonus applied, correct doc ranked first."""
        store = VectorStore(persist_dir=tmp_chromadb)

        # Two docs: one matching project+year, one not
        chunks = [
            _make_chunk("c1", doc_id="d1", text="화성동탄 구조검토 의견서 슬래브",
                        project_name="화성동탄", year=2024, doc_type_ext="구조검토의견서"),
            _make_chunk("c2", doc_id="d2", text="강릉시 구조검토 의견서 기둥",
                        project_name="강릉시", year=2023, doc_type_ext="구조검토의견서"),
        ]
        store.add_chunks(chunks, [[0.5] * 10, [0.5] * 10])

        # Set up ChunkFTS
        fts_db = str(tmp_path / "fts_test.db")
        chunk_fts = ChunkFTS(db_path=fts_db)
        chunk_fts.upsert(chunks)

        # Set up registry mock with search_fts returning d1 as top FTS hit
        registry = MagicMock()
        registry.search_fts.return_value = [
            {"doc_id": "d1", "rank": -5.0},
        ]

        # Set up parser
        parser = QueryParser(
            known_projects={"화성동탄", "강릉시"},
            type_keywords={"구조검토의견서": ["구조검토"]},
        )

        results, parsed = unified_search(
            store, "2024년 화성동탄 구조검토", [0.5] * 10,
            n_results=5,
            query_parser=parser,
            registry=registry,
            chunk_fts=chunk_fts,
        )

        # Parser should extract metadata
        assert parsed is not None
        assert parsed.project == "화성동탄"
        assert parsed.year == 2024

        # d1 should appear in results (project+year match + FTS bonus)
        result_doc_ids = [r.doc_id for r in results]
        assert "d1" in result_doc_ids
        # d1 should rank first (matching project/year/FTS vs d2 which doesn't)
        if len(results) >= 2:
            assert results[0].doc_id == "d1"

    def test_fts_only_hit_filtered_by_year(self, tmp_path, tmp_chromadb: str) -> None:
        """FTS-only hits (not in vector results) are filtered when year_filter is set."""
        store = VectorStore(persist_dir=tmp_chromadb)

        # Only d1 in vector store
        chunks = [
            _make_chunk("c1", doc_id="d1", text="화성동탄 슬래브 보강",
                        project_name="화성동탄", year=2024),
        ]
        store.add_chunks(chunks, [[0.5] * 10])

        # FTS has d1 and d2, but d2 has no year info (FTS doesn't store year)
        fts_db = str(tmp_path / "fts_test2.db")
        chunk_fts = ChunkFTS(db_path=fts_db)
        chunk_fts.upsert(chunks)
        # Add a second chunk to FTS manually
        extra_chunk = _make_chunk("c2", doc_id="d2", text="화성동탄 슬래브 검토",
                                  project_name="화성동탄", year=2023)
        chunk_fts.upsert([extra_chunk])

        results, _ = unified_search(
            store, "화성동탄 슬래브", [0.5] * 10,
            n_results=5,
            year_filter=2024,
            chunk_fts=chunk_fts,
        )

        # d2 is FTS-only (not in vector store) and has year=0 in SearchResult
        # With year_filter=2024, FTS-only hits with year=0 should be filtered out
        result_doc_ids = [r.doc_id for r in results]
        assert "d1" in result_doc_ids
        assert "d2" not in result_doc_ids, "FTS-only hit with year=0 should be filtered by year_filter=2024"

    def test_doc_level_fts_bonus_affects_ranking(self, tmp_chromadb: str) -> None:
        """Doc-level FTS bonus from registry boosts matching docs."""
        store = VectorStore(persist_dir=tmp_chromadb)

        # d1 closer to query embedding, d2 further away
        chunks = [
            _make_chunk("c1", doc_id="d1", text="보강 설계 검토"),
            _make_chunk("c2", doc_id="d2", text="보강 설계 검토"),
        ]
        # d1 gets better vector match (closer embedding)
        store.add_chunks(chunks, [[0.50] * 10, [0.90] * 10])

        # Without FTS bonus: verify d1 ranks first (baseline)
        results_no_fts, _ = unified_search(
            store, "보강 설계", [0.50] * 10,
            n_results=5,
        )
        assert len(results_no_fts) >= 2
        assert results_no_fts[0].doc_id == "d1", "Without FTS bonus, d1 should rank first"

        # With FTS bonus: registry strongly favors d2
        registry = MagicMock()
        registry.search_fts.return_value = [
            {"doc_id": "d2", "rank": -10.0},
        ]

        results_with_fts, _ = unified_search(
            store, "보강 설계", [0.50] * 10,
            n_results=5,
            registry=registry,
        )

        # d2 should now rank first thanks to FTS bonus overcoming vector gap
        assert len(results_with_fts) >= 2
        assert results_with_fts[0].doc_id == "d2", "FTS bonus should push d2 to rank 1"
        registry.search_fts.assert_called_once()
