"""Operational helpers for OCR/embed failure reporting."""

from __future__ import annotations

from collections import Counter
from typing import Any


def collect_embed_failures(registry) -> list[dict[str, Any]]:
    """Return document records augmented with embed failure metadata."""
    docs = registry.list_documents(limit=None, order_by="ingested_at DESC")
    failures: list[dict[str, Any]] = []
    for doc in docs:
        meta_record = registry.get_metadata(doc["doc_id"]) or {}
        metadata = meta_record.get("metadata", {})
        error_type = metadata.get("embed_error_type")
        failure_source = "explicit"
        if not error_type:
            embedded_at = str(doc.get("embedded_at", "") or "").strip()
            process_status = str(doc.get("process_status", "") or "")
            if embedded_at or process_status == "인덱싱완료":
                continue
            error_type = "legacy_untracked"
            failure_source = "inferred"
        failures.append({
            **doc,
            "embed_error_type": error_type,
            "embed_error_msg": metadata.get(
                "embed_error_msg",
                "No embedded_at / not indexed; inferred backlog item",
            ),
            "embed_attempts": int(metadata.get("embed_attempts", 0) or 0),
            "last_embed_error_at": metadata.get("last_embed_error_at", ""),
            "failure_source": failure_source,
        })
    return failures


def classify_retry_strategy(row: dict[str, Any]) -> str:
    """Classify OCR retry strategy for an embed-failure row."""
    error_type = row.get("embed_error_type", "")
    attempts = int(row.get("embed_attempts", 0) or 0)
    page_count = int(row.get("page_count", 0) or 0)
    ocr_engine = row.get("ocr_engine", "")
    source_format = row.get("source_format", "")

    if error_type != "ocr_timeout":
        if error_type == "legacy_untracked":
            if page_count >= 20:
                return "timeout_increase"
            return "gateway_retry"
        return "terminal_failure"
    if attempts >= 3:
        return "terminal_failure"
    if source_format == "pdf" and ocr_engine in ("marker", "none") and page_count <= 20:
        return "gateway_retry"
    return "timeout_increase"


def build_retry_report(registry) -> dict[str, Any]:
    """Build a retry backlog report grouped by strategy."""
    failures = collect_embed_failures(registry)
    grouped: dict[str, list[dict[str, Any]]] = {
        "gateway_retry": [],
        "timeout_increase": [],
        "terminal_failure": [],
    }
    for row in failures:
        grouped[classify_retry_strategy(row)].append(row)
    return {
        "total_failures": len(failures),
        "by_strategy": {key: len(rows) for key, rows in grouped.items()},
        "docs": grouped,
    }


def summarize_ocr_engine_stats(registry) -> dict[str, Any]:
    """Return OCR engine usage/failure statistics from registry state."""
    docs = registry.list_documents(limit=None, order_by="ingested_at DESC")
    by_engine: Counter[str] = Counter()
    failure_by_engine: Counter[str] = Counter()
    failure_by_type: Counter[str] = Counter()

    for doc in docs:
        engine = doc.get("ocr_engine", "") or "none"
        by_engine[engine] += 1
        meta_record = registry.get_metadata(doc["doc_id"]) or {}
        metadata = meta_record.get("metadata", {})
        error_type = metadata.get("embed_error_type", "")
        if error_type:
            failure_by_engine[engine] += 1
            failure_by_type[error_type] += 1

    return {
        "doc_count": sum(by_engine.values()),
        "by_engine": dict(sorted(by_engine.items())),
        "failure_by_engine": dict(sorted(failure_by_engine.items())),
        "failure_by_type": dict(sorted(failure_by_type.items())),
    }
