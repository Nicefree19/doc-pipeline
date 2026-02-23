"""Unit tests for ActionEngine."""

from unittest.mock import patch

import pytest
from filehub.actions.engine import ActionEngine
from filehub.actions.models import ActionRule, ActionType, TriggerType
from filehub.core.models import ValidationResult


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace."""
    (tmp_path / "source").mkdir()
    (tmp_path / "target").mkdir()
    return tmp_path


def test_action_trigger_valid(temp_workspace):
    """Test action triggering on valid result."""
    rule = ActionRule(
        name="Move Valid",
        action=ActionType.MOVE,
        trigger=TriggerType.VALID,
        target=str(temp_workspace / "target"),
    )
    engine = ActionEngine([rule])

    source_file = temp_workspace / "source" / "test_valid.txt"
    source_file.touch()

    result = ValidationResult.valid("test_valid.txt", {})

    engine.process(source_file, result)

    assert not source_file.exists()
    assert (temp_workspace / "target" / "test_valid.txt").exists()


def test_action_trigger_invalid(temp_workspace):
    """Test action triggering on invalid result."""
    rule = ActionRule(
        name="Rename Invalid",
        action=ActionType.RENAME,
        trigger=TriggerType.INVALID,
        target="{original_name}.bad",
    )
    engine = ActionEngine([rule])

    source_file = temp_workspace / "source" / "test_invalid.txt"
    source_file.touch()

    result = ValidationResult(is_valid=False, filename="test_invalid.txt", message="Error")

    engine.process(source_file, result)

    assert not source_file.exists()
    assert (temp_workspace / "source" / "test_invalid.txt.bad").exists()


def test_variable_substitution(temp_workspace):
    """Test variable substitution in target path."""
    rule = ActionRule(
        name="Move with Vars",
        action=ActionType.MOVE,
        trigger=TriggerType.ALWAYS,
        target=str(temp_workspace / "target" / "{project}" / "{year}"),
    )
    engine = ActionEngine([rule])

    source_file = temp_workspace / "source" / "project_file.txt"
    source_file.touch()

    # Mock year
    with patch("filehub.actions.engine.datetime") as mock_date:
        mock_date.fromtimestamp.return_value.strftime.return_value = "2023"
        mock_date.now.return_value.strftime.return_value = "2023"

        result = ValidationResult.valid("project_file.txt", {"project": "P123"})

        engine.process(source_file, result)

    expected_path = temp_workspace / "target" / "P123" / "2023" / "project_file.txt"
    assert expected_path.exists()


def test_conflict_resolution_rename(temp_workspace):
    """Test conflict resolution (rename)."""
    rule = ActionRule(
        name="Move Conflict",
        action=ActionType.MOVE,
        trigger=TriggerType.ALWAYS,
        target=str(temp_workspace / "target"),
        conflict="rename",
    )
    engine = ActionEngine([rule])

    source_file = temp_workspace / "source" / "duplicate.txt"
    source_file.touch()

    # Create existing file in target
    (temp_workspace / "target" / "duplicate.txt").touch()

    engine.process(source_file, None)

    assert (temp_workspace / "target" / "duplicate.txt").exists()
    assert (temp_workspace / "target" / "duplicate_1.txt").exists()


def test_delete_action(temp_workspace):
    """Test delete action."""
    rule = ActionRule(name="Delete Me", action=ActionType.DELETE, trigger=TriggerType.ALWAYS)
    engine = ActionEngine([rule])

    source_file = temp_workspace / "source" / "todelete.txt"
    source_file.touch()

    engine.process(source_file, None)

    assert not source_file.exists()


def test_relative_target_uses_target_root(temp_workspace):
    """Relative targets should be resolved under target_root when provided."""
    rule = ActionRule(
        name="Relative Move",
        action=ActionType.MOVE,
        trigger=TriggerType.ALWAYS,
        target="drawings/{ext_no_dot}",
    )
    engine = ActionEngine([rule], target_root=temp_workspace / "target")

    source_file = temp_workspace / "source" / "sample.dwg"
    source_file.touch()

    engine.process(source_file, None)

    assert not source_file.exists()
    assert (temp_workspace / "target" / "drawings" / "dwg" / "sample.dwg").exists()


def test_dry_run_does_not_move_file(temp_workspace):
    """Dry-run mode should not change files on disk."""
    rule = ActionRule(
        name="DryRun Move",
        action=ActionType.MOVE,
        trigger=TriggerType.ALWAYS,
        target=str(temp_workspace / "target"),
    )
    engine = ActionEngine([rule], dry_run=True)

    source_file = temp_workspace / "source" / "keep.txt"
    source_file.touch()

    engine.process(source_file, None)

    assert source_file.exists()
    assert not (temp_workspace / "target" / "keep.txt").exists()


def test_missing_variable_falls_back_to_next_rule(temp_workspace):
    """If a rule needs missing variables, next rule should still apply."""
    rules = [
        ActionRule(
            name="Needs project var",
            action=ActionType.MOVE,
            trigger=TriggerType.VALID,
            target="{project}/{year}",
        ),
        ActionRule(
            name="Fallback",
            action=ActionType.MOVE,
            trigger=TriggerType.ALWAYS,
            target=str(temp_workspace / "target"),
        ),
    ]
    engine = ActionEngine(rules)

    source_file = temp_workspace / "source" / "fallback.pdf"
    source_file.touch()

    # Valid result with no matched groups -> first rule cannot resolve {project}
    result = ValidationResult.valid("fallback.pdf", {})
    engine.process(source_file, result)

    assert not source_file.exists()
    assert (temp_workspace / "target" / "fallback.pdf").exists()
