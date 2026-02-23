"""Tests for MoveAction class (Step 8)."""

from unittest.mock import patch

from filehub.actions.move import MoveAction
from filehub.core.models import ValidationResult


class TestMoveActionExecute:
    """Test MoveAction.execute method."""

    def test_basic_move(self, tmp_path):
        target = tmp_path / "target"
        target.mkdir()
        source = tmp_path / "file.txt"
        source.write_text("data")

        action = MoveAction(target_dir=target)
        result = action.execute(source)

        assert result.success
        assert not source.exists()
        assert (target / "file.txt").exists()

    def test_file_not_found(self, tmp_path):
        action = MoveAction(target_dir=tmp_path / "target")
        result = action.execute(tmp_path / "nonexistent.txt")

        assert not result.success
        assert "File not found" in result.message

    def test_already_in_target(self, tmp_path):
        target = tmp_path / "target"
        target.mkdir()
        source = target / "file.txt"
        source.write_text("data")

        action = MoveAction(target_dir=target)
        result = action.execute(source)

        assert result.success
        assert "already in target" in result.message

    def test_avoid_overwrite_unique_path(self, tmp_path):
        target = tmp_path / "target"
        target.mkdir()
        (target / "dup.txt").write_text("existing")
        source = tmp_path / "dup.txt"
        source.write_text("new")

        action = MoveAction(target_dir=target)
        result = action.execute(source)

        assert result.success
        assert (target / "dup_1.txt").exists()

    def test_dry_run(self, tmp_path):
        target = tmp_path / "target"
        target.mkdir()
        source = tmp_path / "file.txt"
        source.write_text("data")

        action = MoveAction(target_dir=target, dry_run=True)
        result = action.execute(source)

        assert result.success
        assert result.dry_run
        assert source.exists()
        assert not (target / "file.txt").exists()

    def test_os_error_handling(self, tmp_path):
        target = tmp_path / "target"
        target.mkdir()
        source = tmp_path / "file.txt"
        source.write_text("data")

        action = MoveAction(target_dir=target)
        with patch("shutil.move", side_effect=OSError("disk full")):
            result = action.execute(source)

        assert not result.success
        assert "Move failed" in result.message

    def test_create_dirs_false(self, tmp_path):
        target = tmp_path / "nonexistent_dir"
        source = tmp_path / "file.txt"
        source.write_text("data")

        action = MoveAction(target_dir=target, create_dirs=False)
        with patch("shutil.move", side_effect=OSError("No such directory")):
            result = action.execute(source)
            assert not result.success


class TestResolveTargetDir:
    """Test MoveAction._resolve_target_dir."""

    def test_organize_by_validation_group(self, tmp_path):
        target = tmp_path / "target"
        action = MoveAction(target_dir=target, organize_by="project")

        vr = ValidationResult.valid("file.txt", {"project": "Alpha"})
        result = action._resolve_target_dir(tmp_path / "file.txt", {"validation_result": vr})

        assert "Alpha" in str(result)

    def test_organize_by_extension_map(self, tmp_path):
        target = tmp_path / "target"
        action = MoveAction(target_dir=target, extension_map={".pdf": "documents", ".dwg": "drawings"})

        result = action._resolve_target_dir(tmp_path / "report.pdf", {})
        assert result == target / "documents"

    def test_organize_by_falls_back_to_base(self, tmp_path):
        target = tmp_path / "target"
        action = MoveAction(target_dir=target, organize_by="project")

        # No validation_result in context
        result = action._resolve_target_dir(tmp_path / "file.txt", {})
        assert result == target

    def test_no_organize_returns_base(self, tmp_path):
        target = tmp_path / "target"
        action = MoveAction(target_dir=target)

        result = action._resolve_target_dir(tmp_path / "file.txt", {})
        assert result == target


class TestUniquePath:
    """Test MoveAction._unique_path."""

    def test_no_conflict(self, tmp_path):
        path = tmp_path / "unique.txt"
        result = MoveAction._unique_path(path)
        assert result == path

    def test_single_conflict(self, tmp_path):
        (tmp_path / "dup.txt").write_text("a")
        result = MoveAction._unique_path(tmp_path / "dup.txt")
        assert result == tmp_path / "dup_1.txt"

    def test_multiple_conflicts(self, tmp_path):
        (tmp_path / "dup.txt").write_text("a")
        (tmp_path / "dup_1.txt").write_text("b")
        (tmp_path / "dup_2.txt").write_text("c")
        result = MoveAction._unique_path(tmp_path / "dup.txt")
        assert result == tmp_path / "dup_3.txt"
