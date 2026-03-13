"""Tests for doc_pipeline.cli module."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from doc_pipeline.cli import (
    classify_local,
    cmd_process,
    cmd_batch,
    cmd_embed,
    cmd_purge,
    cmd_reclassify,
    cmd_report,
    cmd_health,
)


class TestCmdProcess:
    def test_process_no_files(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        args = argparse.Namespace(
            path=str(tmp_path / "nonexistent"),
            grade=None,
            no_embed=True,
        )
        cmd_process(args)
        # No crash, just logs error

    @patch("doc_pipeline.processor.pipeline.process_pdf")
    def test_process_single_file(
        self,
        mock_process: MagicMock,
        sample_pdf: Path,
    ) -> None:
        from doc_pipeline.processor.pipeline import PipelineResult
        mock_process.return_value = PipelineResult(skipped=True, skip_reason="test skip")

        args = argparse.Namespace(
            path=str(sample_pdf),
            grade="A",
            no_embed=True,
        )
        cmd_process(args)
        mock_process.assert_called_once()


class TestCmdBatch:
    @patch("doc_pipeline.processor.pipeline.process_pdf")
    def test_batch_with_limit(
        self,
        mock_process: MagicMock,
        sample_pdf: Path,
        tmp_path: Path,
    ) -> None:
        from doc_pipeline.processor.pipeline import PipelineResult
        from doc_pipeline.models.schemas import DocMaster, DocType, ProcessStatus, SecurityGrade

        mock_doc = DocMaster(
            file_name_original="test.pdf",
            doc_type=DocType.OPINION,
            project_name="테스트",
        )
        mock_process.return_value = PipelineResult(doc=mock_doc)

        args = argparse.Namespace(
            path=str(sample_pdf.parent),
            grade="B",
            limit=1,
            no_embed=True,
            output=str(tmp_path / "results.json"),
        )
        cmd_batch(args)

        import json
        result_file = tmp_path / "results.json"
        assert result_file.exists()
        data = json.loads(result_file.read_text(encoding="utf-8"))
        assert len(data) >= 1


class TestClassifyLocal:
    def test_contract_by_keyword(self, tmp_path: Path) -> None:
        path = tmp_path / "2024_용역_계약.pdf"
        result = classify_local(path, "계약서 내용")
        assert result["doc_type"] == "계약서"

    def test_opinion_by_keyword(self, tmp_path: Path) -> None:
        path = tmp_path / "의견서_검토.pdf"
        result = classify_local(path, "구조안전 검토 의견")
        assert result["doc_type"] == "의견서"

    def test_action_plan_by_keyword(self, tmp_path: Path) -> None:
        path = tmp_path / "조치계획서.pdf"
        result = classify_local(path, "지적사항 조치")
        assert result["doc_type"] == "조치계획서"

    def test_year_extraction(self, tmp_path: Path) -> None:
        path = tmp_path / "2023_문서.pdf"
        result = classify_local(path, "텍스트")
        assert result["year"] == "2023"

    def test_no_year(self, tmp_path: Path) -> None:
        path = tmp_path / "문서.pdf"
        result = classify_local(path, "텍스트")
        assert result["year"] == "0"

    def test_project_name_from_korean_path(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "화성동탄프로젝트"
        project_dir.mkdir()
        path = project_dir / "문서.pdf"
        result = classify_local(path, "텍스트")
        assert result["project_name"] == "화성동탄프로젝트"

    def test_default_project_name(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.pdf"
        result = classify_local(path, "텍스트")
        assert result["project_name"] == "미분류"


# ---------------------------------------------------------------------------
# Embed command tests
# ---------------------------------------------------------------------------


class TestCmdEmbed:
    def _make_registry_with_doc(self, tmp_path: Path, sample_pdf: Path):
        """Helper: create a registry with one document, return (registry, doc_record)."""
        from doc_pipeline.storage.registry import DocumentRegistry
        from doc_pipeline.models.schemas import DocMaster, DocType, ProcessStatus, SecurityGrade

        db_path = str(tmp_path / "test_embed.db")
        registry = DocumentRegistry(db_path=db_path)

        doc = DocMaster(
            doc_id="emb001",
            file_name_original=sample_pdf.name,
            file_name_standard="2024-test-의견서-001.pdf",
            doc_type=DocType.OPINION,
            project_name="테스트",
            year=2024,
            process_status=ProcessStatus.COMPLETED,
            security_grade=SecurityGrade.B,
        )
        registry.insert_document(doc, source_path=str(sample_pdf))
        return registry, registry.get_document("emb001")

    @patch("doc_pipeline.cli.settings")
    @patch("doc_pipeline.processor.pipeline.embed_document")
    def test_embed_no_documents(self, mock_embed, mock_settings, tmp_path: Path) -> None:
        """No documents need embedding → graceful exit."""
        db_path = str(tmp_path / "empty.db")
        mock_settings.registry.db_path = db_path

        from doc_pipeline.storage.registry import DocumentRegistry
        DocumentRegistry(db_path=db_path)  # init tables

        args = argparse.Namespace(
            doc_id=None, embed_all=True, grade="B", limit=None, dry_run=False, force=False,
        )
        cmd_embed(args)
        mock_embed.assert_not_called()

    @patch("doc_pipeline.cli.settings")
    @patch("doc_pipeline.processor.pipeline.embed_document", return_value=5)
    def test_embed_single_doc(
        self, mock_embed, mock_settings, tmp_path: Path, sample_pdf: Path,
    ) -> None:
        """Embed a specific document by ID."""
        registry, doc_record = self._make_registry_with_doc(tmp_path, sample_pdf)
        mock_settings.registry.db_path = str(tmp_path / "test_embed.db")

        args = argparse.Namespace(
            doc_id="emb001", embed_all=False, grade="B", limit=None, dry_run=False, force=False,
        )
        cmd_embed(args)
        mock_embed.assert_called_once()

    @patch("doc_pipeline.cli.settings")
    @patch("doc_pipeline.processor.pipeline.embed_document")
    def test_embed_dry_run(
        self, mock_embed, mock_settings, tmp_path: Path, sample_pdf: Path,
    ) -> None:
        """Dry run should not call embed_document."""
        registry, _ = self._make_registry_with_doc(tmp_path, sample_pdf)
        mock_settings.registry.db_path = str(tmp_path / "test_embed.db")

        args = argparse.Namespace(
            doc_id=None, embed_all=True, grade="B", limit=None, dry_run=True, force=False,
        )
        cmd_embed(args)
        mock_embed.assert_not_called()

    @patch("doc_pipeline.cli.settings")
    @patch("doc_pipeline.processor.pipeline.embed_document")
    def test_embed_file_not_found(
        self, mock_embed, mock_settings, tmp_path: Path,
    ) -> None:
        """Missing file should skip, not crash."""
        from doc_pipeline.storage.registry import DocumentRegistry
        from doc_pipeline.models.schemas import DocMaster, DocType, ProcessStatus, SecurityGrade

        db_path = str(tmp_path / "test_embed_miss.db")
        mock_settings.registry.db_path = db_path
        registry = DocumentRegistry(db_path=db_path)

        doc = DocMaster(
            doc_id="miss001",
            file_name_original="nonexistent.pdf",
            doc_type=DocType.OPINION,
            process_status=ProcessStatus.COMPLETED,
            security_grade=SecurityGrade.B,
        )
        registry.insert_document(doc, source_path="/fake/path/nonexistent.pdf")

        args = argparse.Namespace(
            doc_id="miss001", embed_all=False, grade="B", limit=None, dry_run=False, force=False,
        )
        cmd_embed(args)
        mock_embed.assert_not_called()

    @patch("doc_pipeline.cli.settings")
    @patch("doc_pipeline.processor.pipeline.embed_document", return_value=3)
    def test_embed_already_embedded(
        self, mock_embed, mock_settings, tmp_path: Path, sample_pdf: Path,
    ) -> None:
        """Already-embedded documents should be excluded from --all."""
        from doc_pipeline.storage.registry import DocumentRegistry
        from doc_pipeline.models.schemas import DocMaster, DocType, ProcessStatus, SecurityGrade

        db_path = str(tmp_path / "test_embed_done.db")
        mock_settings.registry.db_path = db_path
        registry = DocumentRegistry(db_path=db_path)

        doc = DocMaster(
            doc_id="done001",
            file_name_original=sample_pdf.name,
            doc_type=DocType.OPINION,
            process_status=ProcessStatus.COMPLETED,
            security_grade=SecurityGrade.B,
        )
        registry.insert_document(doc, source_path=str(sample_pdf))
        registry.update_document("done001", embedded_at="2024-01-01T00:00:00")

        args = argparse.Namespace(
            doc_id=None, embed_all=True, grade="B", limit=None, dry_run=False, force=False,
        )
        cmd_embed(args)
        mock_embed.assert_not_called()

    @patch("doc_pipeline.cli.settings")
    @patch("doc_pipeline.processor.pipeline.embed_document", return_value=3)
    def test_embed_force_flag(
        self, mock_embed, mock_settings, tmp_path: Path, sample_pdf: Path,
    ) -> None:
        """force=True should embed even documents with embedded_at set."""
        from doc_pipeline.storage.registry import DocumentRegistry
        from doc_pipeline.models.schemas import DocMaster, DocType, ProcessStatus, SecurityGrade

        db_path = str(tmp_path / "test_embed_force.db")
        mock_settings.registry.db_path = db_path
        mock_settings.chroma.persist_dir = str(tmp_path / "chroma")
        registry = DocumentRegistry(db_path=db_path)

        doc = DocMaster(
            doc_id="force001",
            file_name_original=sample_pdf.name,
            doc_type=DocType.OPINION,
            process_status=ProcessStatus.COMPLETED,
            security_grade=SecurityGrade.B,
        )
        registry.insert_document(doc, source_path=str(sample_pdf))
        registry.update_document("force001", embedded_at="2024-01-01T00:00:00")

        args = argparse.Namespace(
            doc_id=None, embed_all=False, grade="B", limit=None, dry_run=False, force=True,
        )
        cmd_embed(args)
        mock_embed.assert_called_once()

    @patch("doc_pipeline.cli.settings")
    @patch("doc_pipeline.processor.pipeline.embed_document", return_value=3)
    def test_embed_force_disabled(
        self, mock_embed, mock_settings, tmp_path: Path, sample_pdf: Path,
    ) -> None:
        """force=False should NOT embed documents with embedded_at set."""
        from doc_pipeline.storage.registry import DocumentRegistry
        from doc_pipeline.models.schemas import DocMaster, DocType, ProcessStatus, SecurityGrade

        db_path = str(tmp_path / "test_embed_noforce.db")
        mock_settings.registry.db_path = db_path
        registry = DocumentRegistry(db_path=db_path)

        doc = DocMaster(
            doc_id="noforce001",
            file_name_original=sample_pdf.name,
            doc_type=DocType.OPINION,
            process_status=ProcessStatus.COMPLETED,
            security_grade=SecurityGrade.B,
        )
        registry.insert_document(doc, source_path=str(sample_pdf))
        registry.update_document("noforce001", embedded_at="2024-01-01T00:00:00")

        args = argparse.Namespace(
            doc_id=None, embed_all=True, grade="B", limit=None, dry_run=False, force=False,
        )
        cmd_embed(args)
        mock_embed.assert_not_called()


# ---------------------------------------------------------------------------
# Reclassify command tests
# ---------------------------------------------------------------------------


class TestCmdReclassify:
    @patch("doc_pipeline.cli.settings")
    @patch("doc_pipeline.processor.pipeline.reclassify_document")
    def test_reclassify_single(
        self, mock_reclass, mock_settings, tmp_path: Path, sample_pdf: Path,
    ) -> None:
        from doc_pipeline.storage.registry import DocumentRegistry
        from doc_pipeline.models.schemas import DocMaster, DocType, ProcessStatus, SecurityGrade

        db_path = str(tmp_path / "test_reclass.db")
        mock_settings.registry.db_path = db_path
        registry = DocumentRegistry(db_path=db_path)

        doc = DocMaster(
            doc_id="rcl001",
            file_name_original=sample_pdf.name,
            doc_type=DocType.OPINION,
            process_status=ProcessStatus.COMPLETED,
            security_grade=SecurityGrade.B,
        )
        registry.insert_document(doc, source_path=str(sample_pdf))
        mock_reclass.return_value = {"changed": True, "old_type": "의견서", "new_type": "구조계산서"}

        args = argparse.Namespace(
            doc_id="rcl001", reclass_all=False, grade="B", dry_run=False,
        )
        cmd_reclassify(args)
        mock_reclass.assert_called_once()

    @patch("doc_pipeline.cli.settings")
    @patch("doc_pipeline.processor.pipeline.reclassify_document")
    def test_reclassify_dry_run(
        self, mock_reclass, mock_settings, tmp_path: Path, sample_pdf: Path,
    ) -> None:
        from doc_pipeline.storage.registry import DocumentRegistry
        from doc_pipeline.models.schemas import DocMaster, DocType, ProcessStatus, SecurityGrade

        db_path = str(tmp_path / "test_reclass_dry.db")
        mock_settings.registry.db_path = db_path
        registry = DocumentRegistry(db_path=db_path)

        doc = DocMaster(
            doc_id="rcl002",
            file_name_original=sample_pdf.name,
            doc_type=DocType.OPINION,
            doc_type_ext="의견서",
            process_status=ProcessStatus.COMPLETED,
            security_grade=SecurityGrade.B,
        )
        registry.insert_document(doc, source_path=str(sample_pdf))
        mock_reclass.return_value = {"changed": True, "old_type": "의견서", "new_type": "구조계산서"}

        args = argparse.Namespace(
            doc_id=None, reclass_all=True, grade="B", dry_run=True,
        )
        cmd_reclassify(args)
        # Verify dry_run=True is passed to the pipeline function
        mock_reclass.assert_called_once()
        call_kwargs = mock_reclass.call_args
        assert call_kwargs.kwargs.get("dry_run") is True

        # Verify DB was NOT changed (doc_type_ext should still be "의견서")
        doc_after = registry.get_document("rcl002")
        assert doc_after["doc_type_ext"] == "의견서"

    @patch("doc_pipeline.cli.settings")
    def test_reclassify_not_found(self, mock_settings, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test_reclass_miss.db")
        mock_settings.registry.db_path = db_path
        from doc_pipeline.storage.registry import DocumentRegistry
        DocumentRegistry(db_path=db_path)

        args = argparse.Namespace(
            doc_id="nonexistent", reclass_all=False, grade="B", dry_run=False,
        )
        cmd_reclassify(args)  # should not crash


# ---------------------------------------------------------------------------
# Report command tests
# ---------------------------------------------------------------------------


class TestCmdReport:
    @patch("doc_pipeline.cli.settings")
    def test_report_table_format(
        self, mock_settings, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from doc_pipeline.storage.registry import DocumentRegistry
        from doc_pipeline.models.schemas import DocMaster, DocType, ProcessStatus, SecurityGrade

        db_path = str(tmp_path / "test_report.db")
        mock_settings.registry.db_path = db_path
        registry = DocumentRegistry(db_path=db_path)

        doc = DocMaster(
            doc_id="rpt001",
            file_name_original="test.pdf",
            doc_type=DocType.OPINION,
            doc_type_ext="의견서",
            process_status=ProcessStatus.COMPLETED,
            security_grade=SecurityGrade.B,
        )
        registry.insert_document(doc, source_path="/의견서/test.pdf")

        args = argparse.Namespace(format="table")
        cmd_report(args)
        captured = capsys.readouterr()
        assert "Accuracy" in captured.out

    @patch("doc_pipeline.cli.settings")
    def test_report_json_format(
        self, mock_settings, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        import json

        from doc_pipeline.storage.registry import DocumentRegistry
        from doc_pipeline.models.schemas import DocMaster, DocType, ProcessStatus, SecurityGrade

        db_path = str(tmp_path / "test_report_json.db")
        mock_settings.registry.db_path = db_path
        registry = DocumentRegistry(db_path=db_path)

        doc = DocMaster(
            doc_id="rptj001",
            file_name_original="test.pdf",
            doc_type=DocType.OPINION,
            doc_type_ext="의견서",
            process_status=ProcessStatus.COMPLETED,
            security_grade=SecurityGrade.B,
        )
        registry.insert_document(doc, source_path="/의견서/test.pdf")

        args = argparse.Namespace(format="json")
        cmd_report(args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "accuracy_pct" in data

    @patch("doc_pipeline.cli.settings")
    def test_report_empty_registry(
        self, mock_settings, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        db_path = str(tmp_path / "test_report_empty.db")
        mock_settings.registry.db_path = db_path
        from doc_pipeline.storage.registry import DocumentRegistry
        DocumentRegistry(db_path=db_path)

        args = argparse.Namespace(format="table")
        cmd_report(args)
        captured = capsys.readouterr()
        assert "No documents" in captured.out

    @patch("doc_pipeline.cli.settings")
    def test_report_empty_json_format(
        self, mock_settings, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Empty registry with --format json should still produce valid JSON."""
        import json

        db_path = str(tmp_path / "test_report_empty_json.db")
        mock_settings.registry.db_path = db_path
        from doc_pipeline.storage.registry import DocumentRegistry
        DocumentRegistry(db_path=db_path)

        args = argparse.Namespace(format="json")
        cmd_report(args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["total_documents"] == 0
        assert data["accuracy_pct"] == 0.0

    @patch("doc_pipeline.cli.settings")
    def test_report_empty_csv_format(
        self, mock_settings, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Empty registry with --format csv should still produce CSV header."""
        db_path = str(tmp_path / "test_report_empty_csv.db")
        mock_settings.registry.db_path = db_path
        from doc_pipeline.storage.registry import DocumentRegistry
        DocumentRegistry(db_path=db_path)

        args = argparse.Namespace(format="csv")
        cmd_report(args)
        captured = capsys.readouterr()
        assert "doc_id" in captured.out
        assert "folder" in captured.out


# ---------------------------------------------------------------------------
# Health command tests
# ---------------------------------------------------------------------------


class TestCmdHealth:
    def test_health_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Health check runs and prints output (may exit 0 or 1 depending on env)."""
        args = argparse.Namespace()
        with pytest.raises(SystemExit) as exc_info:
            cmd_health(args)
        assert exc_info.value.code in (0, 1)
        captured = capsys.readouterr()
        assert "Health Check" in captured.out

    @patch("doc_pipeline.cli.settings")
    def test_health_with_registry(
        self, mock_settings, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        db_path = str(tmp_path / "test_health.db")
        mock_settings.registry.db_path = db_path
        mock_settings.registry.enabled = True
        mock_settings.gemini.api_key = "test-key-1234"
        mock_settings.chroma.persist_dir = str(tmp_path / "chroma")
        mock_settings.watch.contracts_dir = ""
        mock_settings.watch.action_plans_dir = ""
        mock_settings.watch.opinions_dir = ""
        mock_settings.ocr_engine = "marker"

        from doc_pipeline.storage.registry import DocumentRegistry
        DocumentRegistry(db_path=db_path)

        args = argparse.Namespace()
        with pytest.raises(SystemExit) as exc_info:
            cmd_health(args)
        assert exc_info.value.code in (0, 1)
        captured = capsys.readouterr()
        assert "Health Check" in captured.out
        assert "Registry DB" in captured.out

    @patch("doc_pipeline.cli.settings")
    def test_health_missing_api_key(
        self, mock_settings, capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_settings.gemini.api_key = ""
        mock_settings.chroma.persist_dir = "/tmp/nonexistent_chroma"
        mock_settings.registry.db_path = "/tmp/nonexistent_reg.db"
        mock_settings.watch.contracts_dir = ""
        mock_settings.watch.action_plans_dir = ""
        mock_settings.watch.opinions_dir = ""
        mock_settings.ocr_engine = "marker"

        args = argparse.Namespace()
        with pytest.raises(SystemExit) as exc_info:
            cmd_health(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "MISSING" in captured.out


class TestCmdPurge:
    def test_purge_fix_status_standalone(self, tmp_path: Path) -> None:
        """--fix-status alone updates process_status for embedded docs."""
        import sqlite3

        from doc_pipeline.models.schemas import DocMaster, DocType
        from doc_pipeline.storage.registry import DocumentRegistry

        db_path = str(tmp_path / "purge_reg.db")
        registry = DocumentRegistry(db_path=db_path)

        # Insert docs with different statuses
        for i, status_val in enumerate(["대기", "완료", "추출완료", "실패"]):
            doc = DocMaster(
                doc_id=f"pfix{i:03d}",
                file_name_original=f"fix{i}.pdf",
                doc_type=DocType.OPINION,
            )
            registry.insert_document(doc, source_path=f"/p/{i}")
            # Mark first 3 as embedded
            if i < 3:
                registry.update_document(f"pfix{i:03d}", embedded_at="2026-03-11T00:00:00")
            # Set process_status directly
            conn = sqlite3.connect(db_path)
            conn.execute(
                "UPDATE documents SET process_status = ? WHERE doc_id = ?",
                (status_val, f"pfix{i:03d}"),
            )
            conn.commit()
            conn.close()

        args = argparse.Namespace(fix_status=True, broken_paths=False, doc_ids=None, dry_run=False)
        with patch("doc_pipeline.cli.settings") as mock_settings:
            mock_settings.registry.db_path = db_path
            cmd_purge(args)

        # Verify: 대기, 완료, 추출완료 (embedded) → 인덱싱완료; 실패 (not embedded) unchanged
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT doc_id, process_status FROM documents ORDER BY doc_id").fetchall()
        conn.close()
        status_map = {r["doc_id"]: r["process_status"] for r in rows}
        assert status_map["pfix000"] == "인덱싱완료"
        assert status_map["pfix001"] == "인덱싱완료"
        assert status_map["pfix002"] == "인덱싱완료"
        assert status_map["pfix003"] == "실패"  # not embedded, unchanged


class TestCmdEmbed4Counter:
    @patch("doc_pipeline.cli._resolve_doc_file_path")
    @patch("doc_pipeline.processor.pipeline.embed_document")
    @patch("doc_pipeline.storage.vectordb.VectorStore")
    @patch("doc_pipeline.storage.registry.DocumentRegistry")
    def test_embed_4_counter(
        self,
        mock_reg_cls,
        mock_store_cls,
        mock_embed,
        mock_resolve,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """4-counter: success, skipped_missing, skipped_zero, errors."""
        mock_registry = MagicMock()
        mock_reg_cls.return_value = mock_registry
        mock_store_cls.return_value = MagicMock()

        docs = [
            {"doc_id": "d1", "file_name_original": "f1.pdf"},
            {"doc_id": "d2", "file_name_original": "f2.pdf"},
            {"doc_id": "d3", "file_name_original": "f3.pdf"},
            {"doc_id": "d4", "file_name_original": "f4.pdf"},
        ]
        mock_registry.list_unembedded.return_value = docs

        # d1: file not found; d2: success (5 chunks); d3: 0 chunks; d4: error
        mock_resolve.side_effect = [None, Path("/fake/f2.pdf"), Path("/fake/f3.pdf"), Path("/fake/f4.pdf")]
        mock_embed.side_effect = [5, 0, RuntimeError("test error")]

        args = argparse.Namespace(
            doc_id=None, embed_all=True, grade="B", limit=None, dry_run=False, force=False,
        )
        with caplog.at_level(logging.INFO, logger="doc_pipeline"):
            cmd_embed(args)

        assert "1 success" in caplog.text
        assert "1 skipped (no file)" in caplog.text
        assert "1 skipped (no text)" in caplog.text
        assert "1 errors" in caplog.text
