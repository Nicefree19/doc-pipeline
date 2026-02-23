"""Tests for RenameAction class (Step 8)."""

from unittest.mock import patch

from filehub.actions.rename import RenameAction


class TestRenameTemplate:
    """Test template-based renaming."""

    def test_stem_variable(self, tmp_path):
        source = tmp_path / "report.pdf"
        source.write_text("data")

        action = RenameAction(template="{stem}_final{ext}")
        result = action.execute(source)

        assert result.success
        assert result.destination.name == "report_final.pdf"

    def test_lower_variable(self, tmp_path):
        source = tmp_path / "LOUD_NAME.txt"
        source.write_text("data")

        action = RenameAction(template="{lower}{ext}")
        result = action.execute(source)

        assert result.success
        assert result.destination.name == "loud_name.txt"

    def test_upper_variable(self, tmp_path):
        source = tmp_path / "quiet.txt"
        source.write_text("data")

        action = RenameAction(template="{upper}{ext}")
        result = action.execute(source)

        assert result.success
        assert result.destination.name == "QUIET.txt"

    def test_date_variable(self, tmp_path):
        source = tmp_path / "doc.pdf"
        source.write_text("data")

        action = RenameAction(template="{date}_{stem}{ext}")
        result = action.execute(source)

        assert result.success
        # Date format is YYYYMMDD - just verify it's 8 digits followed by underscore
        import re
        assert re.match(r"\d{8}_doc\.pdf", result.destination.name)

    def test_counter_variable(self, tmp_path):
        action = RenameAction(template="{counter}_{stem}{ext}")

        f1 = tmp_path / "a.txt"
        f1.write_text("1")
        r1 = action.execute(f1)
        assert "0001" in r1.destination.name

        f2 = tmp_path / "b.txt"
        f2.write_text("2")
        r2 = action.execute(f2)
        assert "0002" in r2.destination.name

    def test_template_separator_stripped(self, tmp_path):
        source = tmp_path / "file.txt"
        source.write_text("data")

        action = RenameAction(template="{stem}/{ext}")
        result = action.execute(source)

        assert result.success
        assert "/" not in result.destination.name
        assert "\\" not in result.destination.name


class TestRenameRegex:
    """Test regex-based renaming."""

    def test_basic_regex(self, tmp_path):
        source = tmp_path / "old_name.txt"
        source.write_text("data")

        action = RenameAction(pattern=r"old", replacement="new")
        result = action.execute(source)

        assert result.success
        assert result.destination.name == "new_name.txt"

    def test_regex_group_capture(self, tmp_path):
        source = tmp_path / "doc_2023_v1.txt"
        source.write_text("data")

        action = RenameAction(pattern=r"(\w+)_(\d+)_v(\d+)", replacement=r"\1_v\3_\2")
        result = action.execute(source)

        assert result.success
        assert result.destination.name == "doc_v1_2023.txt"

    def test_regex_separator_stripped(self, tmp_path):
        source = tmp_path / "test_file.txt"
        source.write_text("data")

        action = RenameAction(pattern=r"_", replacement="/")
        result = action.execute(source)

        assert result.success
        assert "/" not in result.destination.name


class TestRenameEdgeCases:
    """Test edge cases for renaming."""

    def test_same_name_no_rename(self, tmp_path):
        source = tmp_path / "unchanged.txt"
        source.write_text("data")

        action = RenameAction(template="{stem}{ext}")
        result = action.execute(source)

        assert result.success
        assert "No rename needed" in result.message

    def test_file_not_found(self, tmp_path):
        action = RenameAction(template="{stem}_new{ext}")
        result = action.execute(tmp_path / "nonexistent.txt")

        assert not result.success
        assert "File not found" in result.message

    def test_dry_run(self, tmp_path):
        source = tmp_path / "file.txt"
        source.write_text("data")

        action = RenameAction(template="{lower}{ext}", dry_run=True)
        result = action.execute(source)

        assert result.success
        assert result.dry_run
        assert source.exists()
        assert source.name == "file.txt"  # unchanged

    def test_os_error_handling(self, tmp_path):
        source = tmp_path / "file.txt"
        source.write_text("data")

        action = RenameAction(template="{upper}{ext}")
        with patch("shutil.move", side_effect=OSError("access denied")):
            result = action.execute(source)

        assert not result.success
        assert "Rename failed" in result.message

    def test_conflict_auto_rename(self, tmp_path):
        (tmp_path / "target.txt").write_text("existing")
        source = tmp_path / "source.txt"
        source.write_text("new")

        action = RenameAction(template="target{ext}")
        result = action.execute(source)

        assert result.success
        assert result.destination.name == "target_1.txt"


class TestIsSameFile:
    """Test _is_same_file helper."""

    def test_same_file(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("data")
        assert RenameAction._is_same_file(f, f) is True

    def test_different_files(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("a")
        b.write_text("b")
        assert RenameAction._is_same_file(a, b) is False

    def test_nonexistent_file(self, tmp_path):
        a = tmp_path / "exists.txt"
        a.write_text("x")
        b = tmp_path / "ghost.txt"
        assert RenameAction._is_same_file(a, b) is False
