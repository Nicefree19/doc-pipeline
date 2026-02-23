"""Tests for TrayStatsDisplay module."""

from __future__ import annotations

import threading
from pathlib import Path

from src.filehub.ui.stats_display import TrayStatsDisplay


class TestFormatTooltip:
    """Test format_tooltip method."""

    def test_format_tooltip_no_stats_shows_zeros(self):
        """Tooltip with a fresh collector should show zero counts."""
        from src.filehub.reporting.collector import StatsCollector

        collector = StatsCollector()
        display = TrayStatsDisplay(collector=collector)
        tooltip = display.format_tooltip()

        assert "Processed: 0 files" in tooltip
        assert "Valid: 0 | Invalid: 0" in tooltip
        assert "Last: N/A" in tooltip

    def test_format_tooltip_with_stats_populated(self):
        """Tooltip should reflect actual stats from the collector."""
        from src.filehub.reporting.collector import StatsCollector

        collector = StatsCollector()
        collector.record_file_processed(Path("/a.txt"))
        collector.record_file_processed(Path("/b.txt"))
        collector.record_file_processed(Path("/c.txt"))
        collector.record_validation_result(Path("/a.txt"), is_valid=True)
        collector.record_validation_result(Path("/b.txt"), is_valid=True)
        collector.record_validation_result(Path("/c.txt"), is_valid=False)

        display = TrayStatsDisplay(collector=collector)
        tooltip = display.format_tooltip()

        assert "Processed: 3 files" in tooltip
        assert "Valid: 2 | Invalid: 1" in tooltip
        assert "Last: N/A" not in tooltip

    def test_format_tooltip_with_none_collector(self):
        """Tooltip with None collector should show zeros and N/A."""
        display = TrayStatsDisplay(collector=None)
        tooltip = display.format_tooltip()

        assert "Processed: 0 files" in tooltip
        assert "Valid: 0 | Invalid: 0" in tooltip
        assert "Last: N/A" in tooltip

    def test_format_tooltip_includes_version(self):
        """Tooltip should include the FileHub version string."""
        from src.filehub import __version__

        display = TrayStatsDisplay(collector=None)
        tooltip = display.format_tooltip()

        assert f"FileHub v{__version__}" in tooltip


class TestFormatMenuStats:
    """Test format_menu_stats method."""

    def test_format_menu_stats_returns_list_of_dicts(self):
        """Menu stats should return a list of dicts with label and value."""
        display = TrayStatsDisplay(collector=None)
        result = display.format_menu_stats()

        assert isinstance(result, list)
        assert len(result) > 0
        for item in result:
            assert "label" in item
            assert "value" in item

    def test_format_menu_stats_with_data(self):
        """Menu stats should reflect actual collector data."""
        from src.filehub.reporting.collector import StatsCollector

        collector = StatsCollector()
        collector.record_file_processed(Path("/a.txt"))
        collector.record_file_processed(Path("/b.txt"))
        collector.record_validation_result(Path("/a.txt"), is_valid=True)
        collector.record_validation_result(Path("/b.txt"), is_valid=False)

        display = TrayStatsDisplay(collector=collector)
        result = display.format_menu_stats()

        labels = {item["label"]: item["value"] for item in result}
        assert labels["Files Processed"] == "2"
        assert labels["Valid"] == "1"
        assert labels["Invalid"] == "1"


class TestRecordError:
    """Test record_error and get_recent_errors methods."""

    def test_record_error_stores_error(self):
        """Recording an error should make it retrievable."""
        display = TrayStatsDisplay()
        display.record_error("bad_file.txt", "Invalid naming convention")

        errors = display.get_recent_errors()
        assert len(errors) == 1
        assert errors[0]["filename"] == "bad_file.txt"
        assert errors[0]["message"] == "Invalid naming convention"

    def test_record_error_max_10_limit_fifo(self):
        """Errors beyond 10 should evict the oldest entries."""
        display = TrayStatsDisplay()
        for i in range(15):
            display.record_error(f"file_{i}.txt", f"Error {i}")

        # Internal buffer should be capped at 10
        errors = display.get_recent_errors(limit=20)
        assert len(errors) == 10
        # Newest first, so the first returned should be the last recorded
        assert errors[0]["filename"] == "file_14.txt"
        # Oldest kept should be file_5
        assert errors[-1]["filename"] == "file_5.txt"

    def test_get_recent_errors_default_limit(self):
        """Default limit should return at most 5 errors."""
        display = TrayStatsDisplay()
        for i in range(8):
            display.record_error(f"file_{i}.txt", f"Error {i}")

        errors = display.get_recent_errors()
        assert len(errors) == 5

    def test_get_recent_errors_custom_limit(self):
        """Custom limit should cap the returned list."""
        display = TrayStatsDisplay()
        for i in range(8):
            display.record_error(f"file_{i}.txt", f"Error {i}")

        errors = display.get_recent_errors(limit=3)
        assert len(errors) == 3

    def test_get_recent_errors_returns_newest_first(self):
        """Errors should be returned with the newest entry first."""
        display = TrayStatsDisplay()
        display.record_error("old.txt", "Old error")
        display.record_error("new.txt", "New error")

        errors = display.get_recent_errors()
        assert errors[0]["filename"] == "new.txt"
        assert errors[1]["filename"] == "old.txt"

    def test_clear_errors_empties_list(self):
        """Clearing errors should remove all stored entries."""
        display = TrayStatsDisplay()
        display.record_error("file.txt", "Error")
        display.record_error("file2.txt", "Error 2")

        display.clear_errors()

        errors = display.get_recent_errors()
        assert len(errors) == 0

    def test_error_dict_structure(self):
        """Each error dict should have timestamp, filename, and message keys."""
        display = TrayStatsDisplay()
        display.record_error("test.txt", "Bad name")

        errors = display.get_recent_errors()
        assert len(errors) == 1
        error = errors[0]
        assert "timestamp" in error
        assert "filename" in error
        assert "message" in error
        # Timestamp should be a valid ISO format string
        assert isinstance(error["timestamp"], str)
        assert "T" in error["timestamp"]


class TestThreadSafety:
    """Test thread safety of TrayStatsDisplay."""

    def test_record_from_multiple_threads(self):
        """Recording errors from multiple threads should not lose data."""
        display = TrayStatsDisplay()
        errors_per_thread = 5
        num_threads = 4
        barrier = threading.Barrier(num_threads)

        def record_errors(thread_id: int) -> None:
            barrier.wait()
            for i in range(errors_per_thread):
                display.record_error(
                    f"thread{thread_id}_file{i}.txt",
                    f"Error from thread {thread_id}",
                )

        threads = [
            threading.Thread(target=record_errors, args=(t,))
            for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Total recorded is 20, but max buffer is 10
        total_expected = min(
            errors_per_thread * num_threads, 10
        )
        errors = display.get_recent_errors(limit=20)
        assert len(errors) == total_expected
