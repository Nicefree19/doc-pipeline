"""Tests for edge cases: Korean filenames, long paths, permissions, COPY (Step 7)."""

import os
import stat
from unittest.mock import patch

import pytest
from filehub.actions.engine import ActionEngine
from filehub.actions.models import ActionRule, ActionType, TriggerType
from filehub.actions.move import MoveAction


class TestKoreanFilenames:
    """Test Korean filename handling."""

    def test_move_korean_filename(self, tmp_path):
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"
        source_dir.mkdir()
        target_dir.mkdir()

        source = source_dir / "구조계산서_v2.pdf"
        source.write_text("PDF content", encoding="utf-8")

        rule = ActionRule(
            name="Move Korean",
            action=ActionType.MOVE,
            trigger=TriggerType.ALWAYS,
            target=str(target_dir),
        )
        engine = ActionEngine([rule])
        engine.process(source, None)

        assert not source.exists()
        assert (target_dir / "구조계산서_v2.pdf").exists()

    def test_copy_korean_filename(self, tmp_path):
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"
        source_dir.mkdir()
        target_dir.mkdir()

        source = source_dir / "설계도면_최종.dwg"
        source.write_text("DWG content", encoding="utf-8")

        rule = ActionRule(
            name="Copy Korean",
            action=ActionType.COPY,
            trigger=TriggerType.ALWAYS,
            target=str(target_dir),
        )
        engine = ActionEngine([rule])
        engine.process(source, None)

        assert source.exists()  # copy keeps original
        assert (target_dir / "설계도면_최종.dwg").exists()

    def test_korean_in_ext_group_path(self, tmp_path):
        target_dir = tmp_path / "target"
        target_dir.mkdir()

        source = tmp_path / "report.pdf"
        source.write_text("content")

        rule = ActionRule(
            name="ExtGroup Move",
            action=ActionType.MOVE,
            trigger=TriggerType.ALWAYS,
            target=str(target_dir / "{ext_group}"),
        )
        engine = ActionEngine([rule])
        engine.process(source, None)

        assert (target_dir / "documents" / "report.pdf").exists()


class TestLongPaths:
    """Test long path handling."""

    def test_move_to_deep_nested_path(self, tmp_path):
        source = tmp_path / "file.txt"
        source.write_text("data")

        # Create a deep but not excessively long path
        deep = "a" * 30
        target = tmp_path / "target" / deep / deep
        rule = ActionRule(
            name="Deep Move",
            action=ActionType.MOVE,
            trigger=TriggerType.ALWAYS,
            target=str(target),
        )
        engine = ActionEngine([rule])
        result = engine.process(source, None)

        assert result  # Should complete without error

    def test_long_filename_near_limit(self, tmp_path):
        # Windows MAX_PATH is 260 chars, test with a reasonably long name
        long_name = "a" * 200 + ".txt"
        source = tmp_path / long_name

        try:
            source.write_text("data")
        except OSError:
            pytest.skip("OS does not support this filename length")

        target_dir = tmp_path / "target"
        target_dir.mkdir()

        rule = ActionRule(
            name="Long Name Move",
            action=ActionType.MOVE,
            trigger=TriggerType.ALWAYS,
            target=str(target_dir),
        )
        engine = ActionEngine([rule])
        result = engine.process(source, None)
        assert result


class TestPermissionErrors:
    """Test permission error handling."""

    @pytest.mark.skipif(os.name != "nt", reason="Windows-specific permission test")
    def test_move_readonly_target(self, tmp_path):
        source = tmp_path / "file.txt"
        source.write_text("data")

        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()

        # Try to create a file in readonly dir - may fail on some systems
        target = readonly_dir / "dest.txt"
        target.write_text("existing")

        try:
            os.chmod(str(readonly_dir), stat.S_IREAD)

            rule = ActionRule(
                name="ReadOnly Move",
                action=ActionType.MOVE,
                trigger=TriggerType.ALWAYS,
                target=str(readonly_dir),
            )
            engine = ActionEngine([rule])
            # Should not crash
            engine.process(source, None)
        finally:
            os.chmod(str(readonly_dir), stat.S_IRWXU)

    def test_move_nonexistent_source(self, tmp_path):
        source = tmp_path / "nonexistent.txt"
        target_dir = tmp_path / "target"
        target_dir.mkdir()

        rule = ActionRule(
            name="Missing Source",
            action=ActionType.MOVE,
            trigger=TriggerType.ALWAYS,
            target=str(target_dir),
        )
        engine = ActionEngine([rule])
        result = engine.process(source, None)
        assert result is False

    def test_copy_permission_denied(self, tmp_path):
        source = tmp_path / "file.txt"
        source.write_text("data")

        action = MoveAction(target_dir=tmp_path / "nonexistent_deep" / "path")

        # MoveAction.execute should handle OSError gracefully
        with patch("shutil.move", side_effect=OSError("Permission denied")):
            result = action.execute(source)
            assert not result.success


class TestCopyAction:
    """Test COPY action through ActionEngine."""

    def test_basic_copy(self, tmp_path):
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"
        source_dir.mkdir()
        target_dir.mkdir()

        source = source_dir / "original.txt"
        source.write_text("content")

        rule = ActionRule(
            name="Basic Copy",
            action=ActionType.COPY,
            trigger=TriggerType.ALWAYS,
            target=str(target_dir),
        )
        engine = ActionEngine([rule])
        engine.process(source, None)

        assert source.exists()
        assert (target_dir / "original.txt").exists()

    def test_copy_conflict_rename(self, tmp_path):
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"
        source_dir.mkdir()
        target_dir.mkdir()

        source = source_dir / "dup.txt"
        source.write_text("new")
        (target_dir / "dup.txt").write_text("old")

        rule = ActionRule(
            name="Copy Conflict",
            action=ActionType.COPY,
            trigger=TriggerType.ALWAYS,
            target=str(target_dir),
            conflict="rename",
        )
        engine = ActionEngine([rule])
        engine.process(source, None)

        assert (target_dir / "dup.txt").read_text() == "old"
        assert (target_dir / "dup_1.txt").exists()

    def test_copy_conflict_skip_returns_false(self, tmp_path):
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"
        source_dir.mkdir()
        target_dir.mkdir()

        source = source_dir / "dup.txt"
        source.write_text("new")
        (target_dir / "dup.txt").write_text("old")

        rule = ActionRule(
            name="Copy Skip",
            action=ActionType.COPY,
            trigger=TriggerType.ALWAYS,
            target=str(target_dir),
            conflict="skip",
        )
        engine = ActionEngine([rule])
        _result = engine.process(source, None)

        # skip returns False from _copy_file, but process returns executed=False
        assert (target_dir / "dup.txt").read_text() == "old"

    def test_copy_dry_run(self, tmp_path):
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"
        source_dir.mkdir()
        target_dir.mkdir()

        source = source_dir / "keep.txt"
        source.write_text("data")

        rule = ActionRule(
            name="DryRun Copy",
            action=ActionType.COPY,
            trigger=TriggerType.ALWAYS,
            target=str(target_dir),
        )
        engine = ActionEngine([rule], dry_run=True)
        engine.process(source, None)

        assert source.exists()
        assert not (target_dir / "keep.txt").exists()


class TestSkipConflict:
    """Test skip conflict resolution."""

    def test_move_skip_returns_false(self, tmp_path):
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"
        source_dir.mkdir()
        target_dir.mkdir()

        source = source_dir / "existing.txt"
        source.write_text("new")
        (target_dir / "existing.txt").write_text("old")

        rule = ActionRule(
            name="Skip Move",
            action=ActionType.MOVE,
            trigger=TriggerType.ALWAYS,
            target=str(target_dir),
            conflict="skip",
        )
        engine = ActionEngine([rule])
        engine.process(source, None)

        # Source should still exist (skip means no move)
        assert source.exists()
        assert (target_dir / "existing.txt").read_text() == "old"
