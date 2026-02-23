"""Tests for extended CLI commands (stats, plugins)."""

from __future__ import annotations

from typer.testing import CliRunner

from src.filehub.cli.main import app

runner = CliRunner()


# ===================================================================
# stats command
# ===================================================================


class TestStatsCommand:
    """Tests for the stats command."""

    def test_stats_no_db_shows_message(self, tmp_path):
        """stats command with non-existent DB shows informative message."""
        fake_db = tmp_path / "nonexistent" / "stats.db"
        result = runner.invoke(app, ["stats", "--db", str(fake_db)])

        assert result.exit_code == 0
        assert "No statistics database found" in result.output

    def test_stats_with_db_shows_report(self, tmp_path):
        """stats command with a real DB shows the text report."""
        from src.filehub.reporting.store import StatsStore

        db_path = tmp_path / "stats.db"
        store = StatsStore(db_path=db_path)
        store.record_event("/test/file.txt", "CREATED")
        store.record_validation("/test/file.txt", is_valid=True)
        store.close()

        result = runner.invoke(app, ["stats", "--db", str(db_path)])

        assert result.exit_code == 0
        assert "FileHub Statistics Report" in result.output
        assert "Validation" in result.output

    def test_stats_default_path(self):
        """stats command without --db uses default ~/.filehub/stats.db path."""
        # The default DB won't exist, so we expect the 'not found' message
        result = runner.invoke(app, ["stats"])

        assert result.exit_code == 0
        assert "FileHub Statistics" in result.output


# ===================================================================
# plugins command
# ===================================================================


class TestPluginsCommand:
    """Tests for the plugins command."""

    def test_plugins_no_plugins_shows_message(self):
        """plugins command with no registered plugins shows 'No plugins'."""
        result = runner.invoke(app, ["plugins"])

        assert result.exit_code == 0
        assert "No plugins installed" in result.output

    def test_plugins_runs_without_error(self):
        """plugins command completes with exit code 0."""
        result = runner.invoke(app, ["plugins"])

        assert result.exit_code == 0

    def test_plugins_shows_header(self):
        """plugins command shows the 'FileHub Plugins' header."""
        result = runner.invoke(app, ["plugins"])

        assert result.exit_code == 0
        assert "FileHub Plugins" in result.output
