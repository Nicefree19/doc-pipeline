from unittest.mock import patch

from typer.testing import CliRunner

from src.filehub.cli.main import app

runner = CliRunner()


def test_scaffold_command(tmp_path):
    """Test the scaffold command execution."""
    target_dir = tmp_path / "new_project"

    # Mock TemplateEngine to avoid actual file creation and dependency on config
    with patch("src.filehub.templates.engine.TemplateEngine") as MockEngine:
        mock_instance = MockEngine.return_value

        result = runner.invoke(app, ["scaffold", "epc_standard", str(target_dir)])

        assert result.exit_code == 0
        assert "Successfully scaffolded" in result.stdout
        mock_instance.scaffold.assert_called_once()
