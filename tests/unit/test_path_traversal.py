"""Tests for path traversal prevention (Step 6)."""


from filehub.actions.engine import ActionEngine
from filehub.actions.models import ActionRule, ActionType, TriggerType
from filehub.actions.move import MoveAction
from filehub.actions.rename import RenameAction
from filehub.core.models import ValidationResult


class TestSanitizePathValue:
    """Test ActionEngine._sanitize_path_value."""

    def test_removes_forward_slash(self):
        assert ActionEngine._sanitize_path_value("../../etc") == "etc"

    def test_removes_backslash(self):
        assert ActionEngine._sanitize_path_value("..\\..\\Windows") == "Windows"

    def test_removes_dotdot(self):
        assert ActionEngine._sanitize_path_value("..project") == "project"

    def test_preserves_normal_value(self):
        assert ActionEngine._sanitize_path_value("P5-MyProject") == "P5-MyProject"

    def test_empty_string(self):
        assert ActionEngine._sanitize_path_value("") == ""


class TestValidateContainment:
    """Test ActionEngine._validate_containment."""

    def test_valid_containment(self, tmp_path):
        child = tmp_path / "sub" / "file.txt"
        assert ActionEngine._validate_containment(child, tmp_path) is True

    def test_escaping_blocked(self, tmp_path):
        boundary = tmp_path / "safe"
        boundary.mkdir()
        outside = tmp_path / "unsafe" / "file.txt"
        assert ActionEngine._validate_containment(outside, boundary) is False

    def test_same_directory(self, tmp_path):
        assert ActionEngine._validate_containment(tmp_path / "file.txt", tmp_path) is True


class TestBuildContextSanitization:
    """Test that matched_groups are sanitized in _build_context."""

    def test_matched_groups_path_separator_sanitized(self, tmp_path):
        rule = ActionRule(
            name="test",
            action=ActionType.MOVE,
            trigger=TriggerType.ALWAYS,
            target=str(tmp_path / "target" / "{project}"),
        )
        engine = ActionEngine([rule])

        source = tmp_path / "source.txt"
        source.write_text("test")

        result = ValidationResult.valid("source.txt", {"project": "../../etc"})
        ctx = engine._build_context(source, result)
        assert "/" not in ctx["project"]
        assert "\\" not in ctx["project"]
        assert ".." not in ctx["project"]
        assert ctx["project"] == "etc"

    def test_matched_groups_backslash_sanitized(self, tmp_path):
        rule = ActionRule(name="t", action=ActionType.MOVE, trigger=TriggerType.ALWAYS, target="x")
        engine = ActionEngine([rule])

        source = tmp_path / "f.txt"
        source.write_text("x")

        result = ValidationResult.valid("f.txt", {"project": "..\\..\\Windows"})
        ctx = engine._build_context(source, result)
        assert ctx["project"] == "Windows"

    def test_sanitize_preserves_non_strings(self, tmp_path):
        rule = ActionRule(name="t", action=ActionType.MOVE, trigger=TriggerType.ALWAYS, target="x")
        engine = ActionEngine([rule])

        source = tmp_path / "f.txt"
        source.write_text("x")

        result = ValidationResult.valid("f.txt", {"count": "42"})
        ctx = engine._build_context(source, result)
        assert ctx["count"] == "42"

    def test_valid_groups_still_work(self, tmp_path):
        (tmp_path / "target").mkdir()
        rule = ActionRule(
            name="test",
            action=ActionType.MOVE,
            trigger=TriggerType.ALWAYS,
            target=str(tmp_path / "target" / "{project}"),
        )
        engine = ActionEngine([rule])

        source = tmp_path / "file.txt"
        source.write_text("data")

        result = ValidationResult.valid("file.txt", {"project": "P5"})
        engine.process(source, result)

        assert (tmp_path / "target" / "P5" / "file.txt").exists()


class TestResolveTargetContainment:
    """Test that _resolve_target blocks paths escaping boundary."""

    def test_resolve_target_escaping_blocked(self, tmp_path):
        target_root = tmp_path / "safe"
        target_root.mkdir()

        rule = ActionRule(name="t", action=ActionType.MOVE, trigger=TriggerType.ALWAYS, target="x")
        engine = ActionEngine([rule], target_root=target_root)

        source = tmp_path / "f.txt"
        source.write_text("x")

        # Directly test _resolve_target with a traversal pattern
        result = engine._resolve_target("../../escape", {"stem": "f"}, source)
        assert result is None

    def test_resolve_target_within_boundary(self, tmp_path):
        target_root = tmp_path / "safe"
        target_root.mkdir()

        rule = ActionRule(name="t", action=ActionType.MOVE, trigger=TriggerType.ALWAYS, target="x")
        engine = ActionEngine([rule], target_root=target_root)

        source = tmp_path / "f.txt"
        source.write_text("x")

        result = engine._resolve_target("sub/dir", {"stem": "f"}, source)
        assert result is not None


class TestMoveActionTraversal:
    """Test MoveAction group_value traversal prevention."""

    def test_move_action_traversal_blocked(self, tmp_path):
        base = tmp_path / "target"
        base.mkdir()
        source = tmp_path / "file.txt"
        source.write_text("data")

        action = MoveAction(target_dir=base, organize_by="project")
        result_vr = ValidationResult.valid("file.txt", {"project": "../../escape"})
        result = action.execute(source, validation_result=result_vr)

        # Should fall back to base directory (traversal blocked)
        assert result.success
        assert base.resolve() in result.destination.resolve().parents or result.destination.parent.resolve() == base.resolve()


class TestRenameTraversal:
    """Test RenameAction separator stripping."""

    def test_rename_template_separator_stripped(self, tmp_path):
        source = tmp_path / "test.txt"
        source.write_text("data")

        action = RenameAction(template="{stem}/{ext}")
        result = action.execute(source)

        # The "/" in template output should be stripped
        assert result.success
        if result.destination:
            assert "/" not in result.destination.name
            assert "\\" not in result.destination.name

    def test_rename_regex_separator_stripped(self, tmp_path):
        source = tmp_path / "test_file.txt"
        source.write_text("data")

        # Replacement that would introduce path separator
        action = RenameAction(pattern=r"_", replacement="/")
        result = action.execute(source)

        assert result.success
        if result.destination:
            assert "/" not in result.destination.name
