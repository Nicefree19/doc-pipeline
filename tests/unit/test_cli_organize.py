"""Tests for CLI organize command."""

from __future__ import annotations

from typer.testing import CliRunner

from src.filehub.actions.models import ActionRule, ActionType, TriggerType
from src.filehub.cli.main import app
from src.filehub.config.schema import FileHubConfig

runner = CliRunner()


def _make_base_config() -> FileHubConfig:
    cfg = FileHubConfig()
    cfg.naming.enabled = False
    cfg.actions = []
    cfg.ignore.prefixes = []
    cfg.ignore.extensions = []
    cfg.ignore.globs = []
    return cfg


class TestOrganizeCommand:
    """Tests for the organize command."""

    def test_analyze_only_does_not_move(self, tmp_path, monkeypatch):
        """Analyze-only mode should not modify files."""
        src = tmp_path / "source"
        dst = tmp_path / "target"
        src.mkdir()
        (src / "a.pdf").write_text("a", encoding="utf-8")
        (src / "b.dwg").write_text("b", encoding="utf-8")
        (src / "c.jpg").write_text("c", encoding="utf-8")

        cfg = _make_base_config()
        monkeypatch.setattr("src.filehub.config.load_config", lambda _=None: cfg)

        result = runner.invoke(
            app,
            ["organize", str(src), "--target", str(dst), "--analyze-only"],
        )

        assert result.exit_code == 0
        assert "Folder Analysis" in result.output
        assert "Files scanned: 3" in result.output
        assert (src / "a.pdf").exists()
        assert not (dst / "documents" / "pdf" / "a.pdf").exists()

    def test_default_rule_moves_by_extension_group(self, tmp_path, monkeypatch):
        """Without configured actions, fallback extension grouping should be used."""
        src = tmp_path / "source"
        dst = tmp_path / "target"
        src.mkdir()
        (src / "a.pdf").write_text("a", encoding="utf-8")
        (src / "b.dwg").write_text("b", encoding="utf-8")
        (src / "c.png").write_text("c", encoding="utf-8")

        cfg = _make_base_config()
        monkeypatch.setattr("src.filehub.config.load_config", lambda _=None: cfg)

        result = runner.invoke(
            app,
            ["organize", str(src), "--target", str(dst)],
        )

        assert result.exit_code == 0
        assert (dst / "documents" / "pdf" / "a.pdf").exists()
        assert (dst / "drawings" / "dwg" / "b.dwg").exists()
        assert (dst / "images" / "png" / "c.png").exists()
        assert not (src / "a.pdf").exists()

    def test_config_rule_respects_user_target_root(self, tmp_path, monkeypatch):
        """Relative action targets should be resolved under --target."""
        src = tmp_path / "source"
        dst = tmp_path / "target"
        src.mkdir()
        (src / "report.pdf").write_text("content", encoding="utf-8")

        cfg = _make_base_config()
        cfg.actions = [
            ActionRule(
                name="Custom Rule",
                action=ActionType.MOVE,
                trigger=TriggerType.ALWAYS,
                target="CUSTOM/{ext_no_dot}",
                conflict="rename",
            )
        ]
        monkeypatch.setattr("src.filehub.config.load_config", lambda _=None: cfg)

        result = runner.invoke(
            app,
            ["organize", str(src), "--target", str(dst)],
        )

        assert result.exit_code == 0
        assert (dst / "CUSTOM" / "pdf" / "report.pdf").exists()

    def test_dry_run_keeps_source_files(self, tmp_path, monkeypatch):
        """Dry-run mode should not move any files."""
        src = tmp_path / "source"
        dst = tmp_path / "target"
        src.mkdir()
        (src / "sample.pdf").write_text("content", encoding="utf-8")

        cfg = _make_base_config()
        monkeypatch.setattr("src.filehub.config.load_config", lambda _=None: cfg)

        result = runner.invoke(
            app,
            ["organize", str(src), "--target", str(dst), "--dry-run"],
        )

        assert result.exit_code == 0
        assert "Dry-run mode: no files were moved." in result.output
        assert (src / "sample.pdf").exists()
        assert not (dst / "documents" / "pdf" / "sample.pdf").exists()
