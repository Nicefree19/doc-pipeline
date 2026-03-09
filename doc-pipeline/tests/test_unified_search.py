"""Tests for doc_pipeline.search.unified module."""

from __future__ import annotations

import pytest

from doc_pipeline.models.schemas import ChunkRecord, DocType, SecurityGrade
from doc_pipeline.search.query_parser import QueryParser
from doc_pipeline.search.unified import unified_search
from doc_pipeline.storage.vectordb import VectorStore


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
