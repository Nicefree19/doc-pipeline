"""Tests for FileHub CLI interface."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from src.filehub.cli.main import app

runner = CliRunner()


# ===================================================================
# watch command
# ===================================================================


class TestWatchCommand:
    """Tests for the watch command."""

    @patch("src.filehub.cli.main.run_app", create=True)
    def test_watch_command_invokes_run_app(self, mock_run_app):
        """watch command calls run_app with correct arguments."""
        # Patch at the point of import inside the command
        mock_run_app.return_value = 0

        with patch("src.filehub.app.run_app", mock_run_app):
            runner.invoke(app, ["watch"])

        mock_run_app.assert_called_once_with(watch_paths=None, config_path=None, target_root=None)

    @patch("src.filehub.app.run_app")
    def test_watch_command_with_paths(self, mock_run_app, tmp_path):
        """watch command passes paths to run_app."""
        mock_run_app.return_value = 0
        dir_a = tmp_path / "a"
        dir_a.mkdir()

        runner.invoke(app, ["watch", str(dir_a)])

        mock_run_app.assert_called_once_with(
            watch_paths=[str(dir_a)], config_path=None, target_root=None
        )

    @patch("src.filehub.app.run_app")
    def test_watch_command_with_config(self, mock_run_app, tmp_path):
        """watch command passes config path to run_app."""
        mock_run_app.return_value = 0
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("watcher:\n  paths: []\n", encoding="utf-8")

        runner.invoke(app, ["watch", "--config", str(cfg_file)])

        mock_run_app.assert_called_once_with(
            watch_paths=None, config_path=str(cfg_file), target_root=None
        )


# ===================================================================
# validate command
# ===================================================================


class TestValidateCommand:
    """Tests for the validate command."""

    def test_validate_valid_file(self, tmp_path):
        """validate shows PASS for a valid filename."""
        valid_file = tmp_path / "PROJ-ABC-ZZ-01-DR-A-0001.pdf"
        valid_file.write_text("content")

        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.message = "Validation passed"

        mock_validator = MagicMock()
        mock_validator.validate.return_value = mock_result

        with patch(
            "src.filehub.naming.ISO19650Validator", return_value=mock_validator
        ):
            result = runner.invoke(app, ["validate", str(valid_file)])

        assert result.exit_code == 0
        assert "[PASS]" in result.output
        assert valid_file.name in result.output

    def test_validate_invalid_file(self, tmp_path):
        """validate shows FAIL and exits with code 1 for invalid filename."""
        invalid_file = tmp_path / "bad-name.pdf"
        invalid_file.write_text("content")

        mock_result = MagicMock()
        mock_result.is_valid = False
        mock_result.message = "Type code is missing."

        mock_validator = MagicMock()
        mock_validator.validate.return_value = mock_result

        with patch(
            "src.filehub.naming.ISO19650Validator", return_value=mock_validator
        ):
            result = runner.invoke(app, ["validate", str(invalid_file)])

        assert result.exit_code == 1
        assert "[FAIL]" in result.output
        assert "Type code is missing." in result.output

    def test_validate_missing_file(self, tmp_path):
        """validate shows SKIP for a non-existent file."""
        missing = tmp_path / "nonexistent.pdf"

        mock_validator = MagicMock()

        with patch(
            "src.filehub.naming.ISO19650Validator", return_value=mock_validator
        ):
            result = runner.invoke(app, ["validate", str(missing)])

        assert "[SKIP]" in result.output
        assert "not found" in result.output
        # Validator should not have been called for missing files
        mock_validator.validate.assert_not_called()


# ===================================================================
# config command
# ===================================================================


class TestConfigCommand:
    """Tests for the config command."""

    def test_config_show_displays_content(self, tmp_path):
        """config show displays file path and content."""
        cfg_file = tmp_path / "config.yaml"
        cfg_content = "watcher:\n  paths:\n    - ~/Documents\n"
        cfg_file.write_text(cfg_content, encoding="utf-8")

        with patch(
            "src.filehub.config.get_config_path", return_value=cfg_file
        ):
            result = runner.invoke(app, ["config", "show"])

        assert result.exit_code == 0
        assert str(cfg_file) in result.output
        assert "---" in result.output
        assert "~/Documents" in result.output

    def test_config_show_with_explicit_path(self, tmp_path):
        """config show --path displays the specified file."""
        cfg_file = tmp_path / "my_config.yaml"
        cfg_file.write_text("logging:\n  level: DEBUG\n", encoding="utf-8")

        result = runner.invoke(
            app, ["config", "show", "--path", str(cfg_file)]
        )

        assert result.exit_code == 0
        assert str(cfg_file) in result.output
        assert "DEBUG" in result.output

    def test_config_init_creates_file(self, tmp_path):
        """config init --path creates a new config file."""
        new_cfg = tmp_path / "new_config" / "config.yaml"

        with patch(
            "src.filehub.config.create_default_config"
        ) as mock_create:
            result = runner.invoke(
                app, ["config", "init", "--path", str(new_cfg)]
            )

        assert result.exit_code == 0
        assert "Configuration file created" in result.output
        mock_create.assert_called_once_with(new_cfg)

    def test_config_unknown_action(self):
        """config with unknown action shows error."""
        result = runner.invoke(app, ["config", "bogus"])

        assert result.exit_code == 1
        assert "Unknown action" in result.output


# ===================================================================
# status command
# ===================================================================


class TestStatusCommand:
    """Tests for the status command."""

    def test_status_shows_version(self):
        """status displays the FileHub version."""
        from src.filehub import __version__

        with patch(
            "src.filehub.config.loader.get_config_path",
            side_effect=Exception("no config"),
        ):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert f"FileHub v{__version__}" in result.output
        assert "---" in result.output
