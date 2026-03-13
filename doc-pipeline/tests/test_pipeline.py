"""Tests for doc_pipeline.processor.pipeline module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from doc_pipeline.models.schemas import (
    BlockType,
    ChunkRecord,
    DocType,
    OCRBlock,
    SecurityGrade,
)
from doc_pipeline.processor.pipeline import (
    PipelineResult,
    _build_chunks,
    _chunk_text,
    _compute_file_hash,
    _next_standard_name,
    _resolve_doc_type,
    _run_ocr_isolated,
    process_pdf,
)


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


class TestChunkText:
    def test_basic_chunking(self) -> None:
        text = "a" * 2000
        chunks = _chunk_text(text, chunk_size=800, chunk_overlap=200)
        assert len(chunks) > 1
        assert all(len(c) <= 800 for c in chunks)

    def test_empty_text(self) -> None:
        assert _chunk_text("") == []

    def test_whitespace_only_skipped(self) -> None:
        assert _chunk_text("   ") == []

    def test_short_text_single_chunk(self) -> None:
        text = "hello"
        chunks = _chunk_text(text, chunk_size=800)
        assert len(chunks) == 1
        assert chunks[0] == "hello"


class TestResolveDocType:
    def test_valid_type(self) -> None:
        from doc_pipeline.models.schemas import DocType
        assert _resolve_doc_type("계약서") == DocType.CONTRACT

    def test_invalid_defaults_to_opinion(self) -> None:
        from doc_pipeline.models.schemas import DocType
        assert _resolve_doc_type("알수없음") == DocType.OPINION


class TestNextStandardName:
    def test_first_file(self, tmp_path: Path) -> None:
        name = _next_standard_name(2024, "테스트", "의견서", tmp_path)
        assert name == "2024-테스트-의견서-001.pdf"

    def test_collision_avoidance(self, tmp_path: Path) -> None:
        # Create existing file
        (tmp_path / "2024-테스트-의견서-001.pdf").touch()
        name = _next_standard_name(2024, "테스트", "의견서", tmp_path)
        assert name == "2024-테스트-의견서-002.pdf"


class TestProcessPdf:
    def test_grade_a_skips(self, sample_pdf: Path) -> None:
        result = process_pdf(sample_pdf, grade=SecurityGrade.A)
        assert result.skipped is True
        assert "grade A" in result.skip_reason.lower() or "A" in result.skip_reason

    @patch("doc_pipeline.processor.pipeline._store_embeddings", return_value=5)
    @patch("doc_pipeline.processor.pipeline._save_to_sheets")
    @patch("doc_pipeline.processor.pipeline._classify_and_extract")
    def test_grade_c_full_pipeline(
        self,
        mock_classify: MagicMock,
        mock_sheets: MagicMock,
        mock_embed: MagicMock,
        sample_pdf: Path,
    ) -> None:
        from doc_pipeline.models.schemas import DocType
        mock_classify.return_value = (
            DocType.OPINION,
            "테스트프로젝트",
            2024,
            {"site_name": "테스트"},
            "요약 텍스트",
        )

        result = process_pdf(sample_pdf, grade=SecurityGrade.C)
        assert result.doc is not None
        assert result.doc.doc_type.value == "의견서"
        assert result.chunks_stored == 5

    @patch("doc_pipeline.processor.pipeline._classify_and_extract")
    def test_grade_b_no_embed(
        self,
        mock_classify: MagicMock,
        sample_pdf: Path,
    ) -> None:
        from doc_pipeline.models.schemas import DocType
        mock_classify.return_value = (
            DocType.OPINION,
            "미분류",
            0,
            {"note": "Grade B"},
            "",
        )

        result = process_pdf(sample_pdf, grade=SecurityGrade.B, no_embed=True)
        assert result.doc is not None
        assert result.chunks_stored == 0


class TestPipelineResult:
    def test_default_values(self) -> None:
        r = PipelineResult()
        assert r.doc is None
        assert r.chunks_stored == 0
        assert r.skipped is False
        assert r.error == ""


class TestComputeFileHash:
    def test_hash_deterministic(self, sample_pdf: Path) -> None:
        h1 = _compute_file_hash(sample_pdf)
        h2 = _compute_file_hash(sample_pdf)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex length

    def test_different_files_different_hashes(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content A")
        f2.write_text("content B")
        assert _compute_file_hash(f1) != _compute_file_hash(f2)


class TestRegistryIntegration:
    """Test that pipeline integrates with registry on process_pdf."""

    @patch("doc_pipeline.processor.pipeline._store_embeddings", return_value=5)
    @patch("doc_pipeline.processor.pipeline._save_to_sheets")
    @patch("doc_pipeline.processor.pipeline._classify_and_extract")
    @patch("doc_pipeline.processor.pipeline._save_to_registry")
    def test_registry_called_on_process(
        self,
        mock_registry_save: MagicMock,
        mock_classify: MagicMock,
        mock_sheets: MagicMock,
        mock_embed: MagicMock,
        sample_pdf: Path,
    ) -> None:
        from doc_pipeline.models.schemas import DocType

        mock_classify.return_value = (
            DocType.OPINION, "테스트", 2024,
            {"site_name": "테스트"}, "요약",
        )
        result = process_pdf(sample_pdf, grade=SecurityGrade.C)
        assert result.doc is not None
        mock_registry_save.assert_called_once()

    @patch("doc_pipeline.processor.pipeline._store_embeddings", return_value=0)
    @patch("doc_pipeline.processor.pipeline._save_to_sheets")
    @patch("doc_pipeline.processor.pipeline._classify_and_extract")
    def test_registry_disabled_skips_dedup(
        self,
        mock_classify: MagicMock,
        mock_sheets: MagicMock,
        mock_embed: MagicMock,
        sample_pdf: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from doc_pipeline.config import settings
        from doc_pipeline.models.schemas import DocType

        monkeypatch.setattr(settings.registry, "enabled", False)
        mock_classify.return_value = (
            DocType.OPINION, "테스트", 2024, {}, "",
        )
        result = process_pdf(sample_pdf, grade=SecurityGrade.C)
        assert result.doc is not None
        assert result.skipped is False


# ---------------------------------------------------------------------------
# Block-aware chunking in _build_chunks (Task 3)
# ---------------------------------------------------------------------------


class TestBuildChunksBlockAware:
    """Test _build_chunks with and without OCR blocks."""

    def _make_doc(self) -> MagicMock:
        doc = MagicMock()
        doc.doc_id = "test_doc_1"
        doc.doc_type_ext = "구조검토의견서"
        doc.category = "구조"
        return doc

    def test_legacy_fallback_without_blocks(self) -> None:
        """Without ocr_blocks, uses legacy _chunk_text path."""
        doc = self._make_doc()
        chunks = _build_chunks(
            doc, DocType.OPINION, "테스트", 2024,
            SecurityGrade.C, "테스트 " * 200,
        )
        assert len(chunks) > 0
        assert all(c.page_number is None for c in chunks)
        assert all(c.block_type is None for c in chunks)

    def test_block_aware_with_ocr_blocks(self) -> None:
        """With ocr_blocks, uses chunk_blocks path with page metadata."""
        doc = self._make_doc()
        blocks = [
            OCRBlock(text="첫 번째 블록 텍스트 " * 20, page=1, block_type=BlockType.TEXT, confidence=0.95),
            OCRBlock(text="두 번째 블록 텍스트 " * 20, page=2, block_type=BlockType.TEXT, confidence=0.88),
        ]
        chunks = _build_chunks(
            doc, DocType.OPINION, "테스트", 2024,
            SecurityGrade.C, "dummy text",
            ocr_blocks=blocks,
        )
        assert len(chunks) > 0
        # Block-aware chunks have page_number metadata
        assert any(c.page_number is not None for c in chunks)
        # doc-level metadata enrichment
        assert all(c.doc_type_ext == "구조검토의견서" for c in chunks)
        assert all(c.category == "구조" for c in chunks)

    def test_empty_blocks_falls_back_to_legacy(self) -> None:
        """Empty ocr_blocks list falls back to legacy path."""
        doc = self._make_doc()
        chunks = _build_chunks(
            doc, DocType.OPINION, "테스트", 2024,
            SecurityGrade.C, "테스트 텍스트 " * 50,
            ocr_blocks=[],
        )
        assert len(chunks) > 0
        # Legacy path — no page metadata
        assert all(c.page_number is None for c in chunks)


# ---------------------------------------------------------------------------
# OCR subprocess isolation tests
# ---------------------------------------------------------------------------


class TestRunOcrIsolated:
    @patch("doc_pipeline.processor.pipeline.multiprocessing")
    def test_run_ocr_isolated_success(self, mock_mp: MagicMock) -> None:
        """Successful OCR returns OCRResult."""
        from doc_pipeline.processor.ocr import OCRResult

        mock_queue = MagicMock()
        mock_queue.empty.return_value = False
        mock_queue.get_nowait.return_value = {
            "text": "Hello OCR", "engine": "marker",
            "elapsed_seconds": 1.5, "page_count": 1,
            "blocks": [], "markdown": "", "avg_confidence": 0.9,
        }
        mock_mp.Queue.return_value = mock_queue

        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False
        mock_proc.exitcode = 0
        mock_mp.Process.return_value = mock_proc

        result = _run_ocr_isolated("marker", Path("test.pdf"), timeout=10)
        assert result is not None
        assert result.text == "Hello OCR"
        assert result.engine == "marker"

    @patch("doc_pipeline.processor.pipeline.multiprocessing")
    def test_run_ocr_isolated_crash(self, mock_mp: MagicMock) -> None:
        """Segfault (exitcode=-11) returns None."""
        mock_queue = MagicMock()
        mock_mp.Queue.return_value = mock_queue

        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False
        mock_proc.exitcode = -11  # SIGSEGV
        mock_mp.Process.return_value = mock_proc

        result = _run_ocr_isolated("marker", Path("test.pdf"), timeout=10)
        assert result is None

    @patch("doc_pipeline.processor.pipeline.multiprocessing")
    def test_run_ocr_isolated_timeout(self, mock_mp: MagicMock) -> None:
        """Timeout returns None after killing process."""
        mock_queue = MagicMock()
        mock_mp.Queue.return_value = mock_queue

        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True  # still running after join
        mock_mp.Process.return_value = mock_proc

        result = _run_ocr_isolated("marker", Path("test.pdf"), timeout=1)
        assert result is None
        mock_proc.kill.assert_called_once()

    @patch("doc_pipeline.processor.pipeline._run_ocr_isolated", return_value=None)
    @patch("doc_pipeline.processor.pipeline._classify_and_extract")
    def test_process_pdf_ocr_fallback(
        self,
        mock_classify: MagicMock,
        mock_ocr: MagicMock,
        sample_pdf: Path,
    ) -> None:
        """When OCR fails, process_pdf uses adapter text."""
        mock_classify.return_value = (
            DocType.OPINION, "테스트", 2024, {}, "",
        )

        # Monkeypatch extraction to say it's scanned
        with patch("doc_pipeline.processor.pipeline.extract_text") as mock_extract:
            mock_extract.return_value = MagicMock(
                text="adapter text", page_count=1, is_scanned=True,
            )
            result = process_pdf(sample_pdf, grade=SecurityGrade.B, no_embed=True)

        assert result.doc is not None
        # No crash — adapter text was used as fallback


# ---------------------------------------------------------------------------
# Embed extracted_text caching tests
# ---------------------------------------------------------------------------


class TestEmbedExtractedText:
    @patch("doc_pipeline.processor.pipeline._run_ocr_isolated", return_value=None)
    @patch("doc_pipeline.processor.pipeline._build_chunks")
    @patch("doc_pipeline.processor.pipeline.mask_text", side_effect=lambda t: t)
    def test_embed_stores_extracted_text(
        self,
        mock_mask: MagicMock,
        mock_chunks: MagicMock,
        mock_ocr: MagicMock,
        sample_pdf: Path,
        tmp_path: Path,
    ) -> None:
        """embed_document saves extracted_text to metadata."""
        from doc_pipeline.processor.pipeline import embed_document
        from doc_pipeline.storage.registry import DocumentRegistry

        mock_chunks.return_value = []

        db_path = str(tmp_path / "test_cache.db")
        registry = DocumentRegistry(db_path=db_path)

        from doc_pipeline.models.schemas import DocMaster, ProcessStatus
        doc = DocMaster(
            doc_id="cache001",
            file_name_original=sample_pdf.name,
            doc_type=DocType.OPINION,
            process_status=ProcessStatus.COMPLETED,
            security_grade=SecurityGrade.B,
        )
        registry.insert_document(doc, source_path=str(sample_pdf))
        doc_record = registry.get_document("cache001")

        with patch("doc_pipeline.collector.adapters.get_adapter") as mock_adapter:
            mock_norm = MagicMock()
            mock_norm.text = "sample text content"
            mock_norm.is_scanned = False
            mock_adapter.return_value.extract.return_value = mock_norm

            embed_document(doc_record, sample_pdf, SecurityGrade.B, registry=registry)

        meta = registry.get_metadata("cache001")
        assert meta is not None
        assert "extracted_text" in meta.get("metadata", {})
        assert meta["metadata"]["extracted_text"] == "sample text content"

    @patch("doc_pipeline.processor.pipeline._run_ocr_isolated", return_value=None)
    @patch("doc_pipeline.processor.pipeline.mask_text", side_effect=lambda t: t)
    def test_embed_cached_text_fallback(
        self,
        mock_mask: MagicMock,
        mock_ocr: MagicMock,
        sample_pdf: Path,
        tmp_path: Path,
    ) -> None:
        """When adapter returns empty text, fallback to cached extracted_text."""
        from doc_pipeline.processor.pipeline import embed_document
        from doc_pipeline.storage.registry import DocumentRegistry

        db_path = str(tmp_path / "test_fallback.db")
        registry = DocumentRegistry(db_path=db_path)

        from doc_pipeline.models.schemas import DocMaster, ProcessStatus
        doc = DocMaster(
            doc_id="fb001",
            file_name_original=sample_pdf.name,
            doc_type=DocType.OPINION,
            process_status=ProcessStatus.COMPLETED,
            security_grade=SecurityGrade.B,
        )
        registry.insert_document(doc, source_path=str(sample_pdf))
        # Pre-populate cached text
        registry.save_metadata("fb001", {"extracted_text": "cached text content " * 20})
        doc_record = registry.get_document("fb001")

        with patch("doc_pipeline.collector.adapters.get_adapter") as mock_adapter:
            mock_norm = MagicMock()
            mock_norm.text = ""  # Empty text from adapter
            mock_norm.is_scanned = False
            mock_adapter.return_value.extract.return_value = mock_norm

            with patch("doc_pipeline.storage.vectordb.VectorStore") as mock_store_cls:
                mock_store = MagicMock()
                mock_store_cls.return_value = mock_store

                chunks = embed_document(
                    doc_record, sample_pdf, SecurityGrade.B,
                    store=mock_store, registry=registry,
                )

        # Should have used cached text and produced chunks
        assert chunks > 0


class TestChunkExplosionGuard:
    def test_chunk_guard_truncates(self, tmp_path: Path) -> None:
        """Chunks exceeding max_chunks_per_doc are truncated."""
        from doc_pipeline.models.schemas import DocMaster

        doc = DocMaster(
            doc_id="explosion001",
            file_name_original="big.pdf",
            doc_type=DocType.OPINION,
            project_name="테스트",
            year=2024,
            security_grade=SecurityGrade.B,
        )

        # Large text that will produce many chunks
        big_text = "가나다라마바사 " * 500  # ~4000 chars with small chunks

        with patch("doc_pipeline.processor.pipeline.settings") as mock_settings:
            mock_settings.chroma.chunk_size = 20
            mock_settings.chroma.chunk_overlap = 5
            mock_settings.chroma.max_chunks_per_doc = 5

            chunks = _build_chunks(
                doc, DocType.OPINION, "테스트", 2024, SecurityGrade.B, big_text,
            )

        assert len(chunks) == 5


class TestReplaceSemanticsStaleChunkPrune:
    def test_prune_stale_chunks(self, tmp_path: Path) -> None:
        """After re-embed, stale chunks not in new set are pruned."""
        from doc_pipeline.processor.pipeline import _prune_stale_chunks
        from doc_pipeline.storage.vectordb import VectorStore

        store = VectorStore(str(tmp_path / "prune_chroma"))

        # Initial embed: 10 chunks
        initial_chunks = [
            _make_chunk(f"doc1_{i}", doc_id="doc1", text=f"old chunk {i}", chunk_index=i)
            for i in range(10)
        ]
        store.upsert_chunks_local(initial_chunks)
        assert store.count == 10

        # Re-embed: 5 chunks (new set)
        new_chunks = [
            _make_chunk(f"doc1_{i}", doc_id="doc1", text=f"new chunk {i}", chunk_index=i)
            for i in range(5)
        ]
        store.upsert_chunks_local(new_chunks)

        new_ids = {f"doc1_{i}" for i in range(5)}
        pruned = _prune_stale_chunks(store, "doc1", new_ids)

        assert pruned == 5  # stale doc1_5 through doc1_9
        assert store.count == 5  # only new chunks remain

    def test_prune_empty_new_ids_noop(self, tmp_path: Path) -> None:
        """Empty new_chunk_ids → no-op (safety guard)."""
        from doc_pipeline.processor.pipeline import _prune_stale_chunks
        from doc_pipeline.storage.vectordb import VectorStore

        store = VectorStore(str(tmp_path / "prune_noop"))
        chunks = [_make_chunk("doc1_0", doc_id="doc1")]
        store.upsert_chunks_local(chunks)

        pruned = _prune_stale_chunks(store, "doc1", set())
        assert pruned == 0
        assert store.count == 1  # preserved


class TestCrossGradeReembed:
    """Verify that re-embedding with a different grade clears the opposite collection."""

    def test_api_to_local_clears_api_collection(self, tmp_path: Path) -> None:
        """C-grade chunks in API collection → B-grade re-embed → API collection cleared."""
        from doc_pipeline.processor.pipeline import _clear_opposite_collection
        from doc_pipeline.storage.vectordb import VectorStore

        store = VectorStore(str(tmp_path / "cross_c_to_b"))

        # Simulate C-grade: chunks in API collection (need embeddings)
        api_chunks = [
            _make_chunk(f"doc1_{i}", doc_id="doc1", text=f"api chunk {i}", chunk_index=i)
            for i in range(3)
        ]
        embeddings = [[0.1] * 384 for _ in range(3)]
        store.upsert_chunks(api_chunks, embeddings)
        assert store._collection.count() == 3
        assert store._local_collection.count() == 0

        # Now re-embed as B-grade: clear opposite (API) collection first
        cleared = _clear_opposite_collection(store, "doc1", SecurityGrade.B)
        assert cleared == 3
        assert store._collection.count() == 0

        # Then upsert into local
        local_chunks = [
            _make_chunk(f"doc1_{i}", doc_id="doc1", text=f"local chunk {i}", chunk_index=i)
            for i in range(3)
        ]
        store.upsert_chunks_local(local_chunks)
        assert store.count == 3  # exactly one set, not 6

    def test_local_to_api_clears_local_collection(self, tmp_path: Path) -> None:
        """B-grade chunks in local collection → C-grade re-embed → local collection cleared."""
        from doc_pipeline.processor.pipeline import _clear_opposite_collection
        from doc_pipeline.storage.vectordb import VectorStore

        store = VectorStore(str(tmp_path / "cross_b_to_c"))

        # Simulate B-grade: chunks in local collection
        local_chunks = [
            _make_chunk(f"doc1_{i}", doc_id="doc1", text=f"local chunk {i}", chunk_index=i)
            for i in range(4)
        ]
        store.upsert_chunks_local(local_chunks)
        assert store._local_collection.count() == 4
        assert store._collection.count() == 0

        # Now re-embed as C-grade: clear opposite (local) collection first
        cleared = _clear_opposite_collection(store, "doc1", SecurityGrade.C)
        assert cleared == 4
        assert store._local_collection.count() == 0

        # Then upsert into API collection
        api_chunks = [
            _make_chunk(f"doc1_{i}", doc_id="doc1", text=f"api chunk {i}", chunk_index=i)
            for i in range(4)
        ]
        embeddings = [[0.1] * 384 for _ in range(4)]
        store.upsert_chunks(api_chunks, embeddings)
        assert store.count == 4  # exactly one set

    def test_same_grade_reembed_no_opposite_clear(self, tmp_path: Path) -> None:
        """B→B re-embed: opposite (API) collection is empty, cleared == 0."""
        from doc_pipeline.processor.pipeline import _clear_opposite_collection
        from doc_pipeline.storage.vectordb import VectorStore

        store = VectorStore(str(tmp_path / "same_grade"))

        local_chunks = [_make_chunk("doc1_0", doc_id="doc1")]
        store.upsert_chunks_local(local_chunks)

        # API collection is empty, so clearing it is a no-op
        cleared = _clear_opposite_collection(store, "doc1", SecurityGrade.B)
        assert cleared == 0
        assert store.count == 1

    def test_cross_grade_embed_failure_preserves_old_chunks(
        self, tmp_path: Path, sample_pdf: Path,
    ) -> None:
        """If cross-grade re-embed fails mid-flight, old chunks must survive.

        Scenario: doc1 was B-grade (local collection, 3 chunks).
        Re-embed as C-grade, but get_embeddings raises → exception propagates.
        Old local chunks must still be present for search continuity.
        """
        from doc_pipeline.models.schemas import DocMaster, ProcessStatus
        from doc_pipeline.processor.pipeline import embed_document
        from doc_pipeline.storage.registry import DocumentRegistry
        from doc_pipeline.storage.vectordb import VectorStore

        chroma_dir = str(tmp_path / "fail_chroma")
        store = VectorStore(chroma_dir)

        # Pre-populate: B-grade chunks in local collection
        old_chunks = [
            _make_chunk(f"doc1_{i}", doc_id="doc1", text=f"old B chunk {i}", chunk_index=i)
            for i in range(3)
        ]
        store.upsert_chunks_local(old_chunks)
        assert store.count == 3
        assert store._local_collection.count() == 3

        # Set up registry with the doc
        db_path = str(tmp_path / "fail_reg.db")
        registry = DocumentRegistry(db_path=db_path)
        doc = DocMaster(
            doc_id="doc1",
            file_name_original=sample_pdf.name,
            doc_type=DocType.OPINION,
            process_status=ProcessStatus.COMPLETED,
            security_grade=SecurityGrade.B,
        )
        registry.insert_document(doc, source_path=str(sample_pdf))

        doc_record = registry.get_document("doc1")

        # Attempt C-grade re-embed with failing get_embeddings
        with patch("doc_pipeline.collector.adapters.get_adapter") as mock_adapter:
            mock_norm = MagicMock()
            mock_norm.text = "텍스트 " * 100
            mock_norm.is_scanned = False
            mock_adapter.return_value.extract.return_value = mock_norm

            with patch(
                "doc_pipeline.processor.llm.get_embeddings",
                side_effect=RuntimeError("API unavailable"),
            ):
                with pytest.raises(RuntimeError, match="API unavailable"):
                    embed_document(
                        doc_record, sample_pdf, SecurityGrade.C,
                        store=store, registry=registry,
                    )

        # Key invariant: old B-grade chunks survive in local collection
        assert store._local_collection.count() == 3
        assert store.count >= 3  # old chunks still searchable

    def test_cross_grade_c_to_b_failure_preserves_api_chunks(
        self, tmp_path: Path, sample_pdf: Path,
    ) -> None:
        """Symmetric case: C-grade API chunks survive if B-grade re-embed fails.

        Scenario: doc1 was C-grade (API collection, 3 chunks with embeddings).
        Re-embed as B-grade, but upsert_chunks_local raises → exception propagates.
        Old API chunks must remain for search continuity.
        """
        from doc_pipeline.models.schemas import DocMaster, ProcessStatus
        from doc_pipeline.processor.pipeline import embed_document
        from doc_pipeline.storage.registry import DocumentRegistry
        from doc_pipeline.storage.vectordb import VectorStore

        chroma_dir = str(tmp_path / "fail_c2b_chroma")
        store = VectorStore(chroma_dir)

        # Pre-populate: C-grade chunks in API collection
        old_chunks = [
            _make_chunk(f"doc1_{i}", doc_id="doc1", text=f"old C chunk {i}", chunk_index=i)
            for i in range(3)
        ]
        embeddings = [[0.1] * 384 for _ in range(3)]
        store.upsert_chunks(old_chunks, embeddings)
        assert store._collection.count() == 3

        # Set up registry
        db_path = str(tmp_path / "fail_c2b_reg.db")
        registry = DocumentRegistry(db_path=db_path)
        doc = DocMaster(
            doc_id="doc1",
            file_name_original=sample_pdf.name,
            doc_type=DocType.OPINION,
            process_status=ProcessStatus.COMPLETED,
            security_grade=SecurityGrade.C,
        )
        registry.insert_document(doc, source_path=str(sample_pdf))
        doc_record = registry.get_document("doc1")

        # Attempt B-grade re-embed with failing upsert_chunks_local
        with patch("doc_pipeline.collector.adapters.get_adapter") as mock_adapter:
            mock_norm = MagicMock()
            mock_norm.text = "텍스트 " * 100
            mock_norm.is_scanned = False
            mock_adapter.return_value.extract.return_value = mock_norm

            with patch.object(
                store, "upsert_chunks_local",
                side_effect=RuntimeError("ChromaDB write failed"),
            ):
                with pytest.raises(RuntimeError, match="ChromaDB write failed"):
                    embed_document(
                        doc_record, sample_pdf, SecurityGrade.B,
                        store=store, registry=registry,
                    )

        # Key invariant: old C-grade API chunks survive
        assert store._collection.count() == 3
        assert store.count >= 3


class TestEmbedFailureTracking:
    """embed_document records failure once, clears on success."""

    def test_embed_failure_single_count(self, tmp_path: Path, sample_pdf: Path) -> None:
        """OCR timeout + no text → single failure record, not double-counted."""
        from doc_pipeline.models.schemas import DocMaster, ProcessStatus
        from doc_pipeline.processor.pipeline import embed_document
        from doc_pipeline.storage.registry import DocumentRegistry

        db_path = str(tmp_path / "fail_track.db")
        registry = DocumentRegistry(db_path=db_path)
        doc = DocMaster(
            doc_id="fail1",
            file_name_original="scan.pdf",
            doc_type=DocType.OPINION,
            process_status=ProcessStatus.COMPLETED,
            security_grade=SecurityGrade.B,
        )
        registry.insert_document(doc, source_path=str(sample_pdf))
        doc_record = registry.get_document("fail1")

        with patch("doc_pipeline.collector.adapters.get_adapter") as mock_adapter:
            mock_norm = MagicMock()
            mock_norm.text = ""  # adapter returns no text
            mock_norm.is_scanned = True
            mock_adapter.return_value.extract.return_value = mock_norm

            with patch(
                "doc_pipeline.processor.pipeline._run_ocr_isolated",
                return_value=None,  # OCR timeout
            ):
                result = embed_document(
                    doc_record, sample_pdf, SecurityGrade.B,
                    registry=registry,
                )
                assert result == 0

        meta = registry.get_metadata("fail1")
        assert meta is not None
        # Single invocation → exactly 1 attempt recorded
        assert meta["metadata"]["embed_attempts"] == 1
        assert meta["metadata"]["embed_error_type"] in ("ocr_timeout", "no_text")

    def test_embed_success_clears_failure(self, tmp_path: Path, sample_pdf: Path) -> None:
        """After successful embedding, failure metadata is cleared."""
        from doc_pipeline.models.schemas import DocMaster, ProcessStatus
        from doc_pipeline.processor.pipeline import embed_document
        from doc_pipeline.storage.registry import DocumentRegistry
        from doc_pipeline.storage.vectordb import VectorStore

        db_path = str(tmp_path / "clear_fail.db")
        registry = DocumentRegistry(db_path=db_path)
        chroma_dir = str(tmp_path / "clear_chroma")
        store = VectorStore(persist_dir=chroma_dir)

        doc = DocMaster(
            doc_id="clear1",
            file_name_original="ok.pdf",
            doc_type=DocType.OPINION,
            process_status=ProcessStatus.COMPLETED,
            security_grade=SecurityGrade.B,
        )
        registry.insert_document(doc, source_path=str(sample_pdf))

        # Pre-set failure metadata
        registry.update_embed_failure("clear1", "ocr_timeout", "Timeout")
        meta_before = registry.get_metadata("clear1")
        assert meta_before["metadata"]["embed_error_type"] == "ocr_timeout"

        doc_record = registry.get_document("clear1")

        # Successful embed
        with patch("doc_pipeline.collector.adapters.get_adapter") as mock_adapter:
            mock_norm = MagicMock()
            mock_norm.text = "텍스트 " * 100
            mock_norm.is_scanned = False
            mock_adapter.return_value.extract.return_value = mock_norm

            result = embed_document(
                doc_record, sample_pdf, SecurityGrade.B,
                store=store, registry=registry,
            )
            assert result > 0

        # Failure metadata should be cleared
        meta_after = registry.get_metadata("clear1")
        assert meta_after is not None
        assert "embed_error_type" not in meta_after["metadata"]
        assert "embed_attempts" not in meta_after["metadata"]
