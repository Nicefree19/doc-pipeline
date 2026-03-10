"""Tests for doc_pipeline.processor.pipeline module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from doc_pipeline.models.schemas import (
    BlockType,
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
