"""Tests for OCR operational reporting helpers."""

from __future__ import annotations

from pathlib import Path

from doc_pipeline.models.schemas import DocMaster, DocType, ProcessStatus, SecurityGrade
from doc_pipeline.processor.ocr_ops import (
    build_retry_report,
    classify_retry_strategy,
    collect_embed_failures,
    summarize_ocr_engine_stats,
)
from doc_pipeline.storage.registry import DocumentRegistry


def _insert_doc(
    registry: DocumentRegistry,
    *,
    doc_id: str,
    file_name_original: str = "test.pdf",
    doc_type: DocType = DocType.CONTRACT,
    page_count: int = 5,
    ocr_engine: str = "marker",
) -> None:
    doc = DocMaster(
        doc_id=doc_id,
        file_name_original=file_name_original,
        doc_type=doc_type,
        page_count=page_count,
        ocr_engine=ocr_engine,
        process_status=ProcessStatus.COMPLETED,
        security_grade=SecurityGrade.B,
    )
    registry.insert_document(doc, source_path=f"/{file_name_original}")


def test_collect_embed_failures(tmp_path: Path) -> None:
    registry = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    _insert_doc(registry, doc_id="d1")
    _insert_doc(registry, doc_id="d2", ocr_engine="mineru")
    registry.update_embed_failure("d1", "ocr_timeout", "Timeout 300s")
    registry.update_document("d2", embedded_at="2026-03-13T00:00:00", process_status="인덱싱완료")

    failures = collect_embed_failures(registry)
    assert len(failures) == 1
    assert failures[0]["doc_id"] == "d1"
    assert failures[0]["embed_error_type"] == "ocr_timeout"


def test_collect_embed_failures_infers_legacy_unembedded(tmp_path: Path) -> None:
    registry = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    _insert_doc(registry, doc_id="legacy", page_count=12)
    failures = collect_embed_failures(registry)
    assert len(failures) == 1
    assert failures[0]["embed_error_type"] == "legacy_untracked"
    assert failures[0]["failure_source"] == "inferred"


def test_classify_retry_strategy() -> None:
    assert classify_retry_strategy({
        "embed_error_type": "ocr_timeout",
        "embed_attempts": 1,
        "page_count": 8,
        "ocr_engine": "marker",
        "source_format": "pdf",
    }) == "gateway_retry"
    assert classify_retry_strategy({
        "embed_error_type": "ocr_timeout",
        "embed_attempts": 1,
        "page_count": 50,
        "ocr_engine": "marker",
        "source_format": "pdf",
    }) == "timeout_increase"
    assert classify_retry_strategy({
        "embed_error_type": "no_text",
        "embed_attempts": 1,
        "page_count": 1,
        "ocr_engine": "marker",
        "source_format": "pdf",
    }) == "terminal_failure"


def test_retry_report_and_engine_stats(tmp_path: Path) -> None:
    registry = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    _insert_doc(registry, doc_id="gateway", page_count=8, ocr_engine="marker")
    _insert_doc(registry, doc_id="timeout", page_count=40, ocr_engine="marker")
    _insert_doc(registry, doc_id="ok", doc_type=DocType.METHOD_DOC, ocr_engine="none")
    registry.update_embed_failure("gateway", "ocr_timeout", "Timeout")
    registry.update_embed_failure("timeout", "ocr_timeout", "Timeout")
    registry.update_document("ok", embedded_at="2026-03-13T00:00:00", process_status="인덱싱완료")

    report = build_retry_report(registry)
    assert report["total_failures"] == 2
    assert report["by_strategy"]["gateway_retry"] == 1
    assert report["by_strategy"]["timeout_increase"] == 1

    stats = summarize_ocr_engine_stats(registry)
    assert stats["doc_count"] == 3
    assert stats["failure_by_type"]["ocr_timeout"] == 2
