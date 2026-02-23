"""Tray statistics display formatter for FileHub.

Formats statistics from StatsCollector for display in system tray
tooltips and menus. Standalone module with no pystray dependency.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

from .. import __version__
from ..reporting.collector import StatsCollector


class TrayStatsDisplay:
    """Format and manage statistics for system tray display.

    Thread-safe class that formats StatsCollector data for tray tooltips
    and menus, and tracks recent validation errors.
    """

    _MAX_ERRORS = 10

    def __init__(self, collector: StatsCollector | None = None) -> None:
        """Initialize the tray stats display.

        Args:
            collector: Optional StatsCollector instance to pull stats from.
        """
        self._collector = collector
        self._lock = threading.Lock()
        self._recent_errors: list[dict] = []

    def format_tooltip(self) -> str:
        """Format statistics for the tray tooltip.

        Returns:
            Multi-line string suitable for a system tray tooltip.
        """
        if self._collector is None:
            return (
                f"FileHub v{__version__}\n"
                f"Processed: 0 files\n"
                f"Valid: 0 | Invalid: 0\n"
                f"Last: N/A"
            )

        summary = self._collector.get_summary()
        total = summary["total_files_processed"]
        passed = summary["validation_passed"]
        failed = summary["validation_failed"]
        last_activity = summary["last_activity"]

        if last_activity is not None:
            timestamp = last_activity
        else:
            timestamp = "N/A"

        return (
            f"FileHub v{__version__}\n"
            f"Processed: {total} files\n"
            f"Valid: {passed} | Invalid: {failed}\n"
            f"Last: {timestamp}"
        )

    def format_menu_stats(self) -> list[dict]:
        """Format statistics for tray menu display.

        Returns:
            List of dicts with 'label' and 'value' keys.
        """
        if self._collector is None:
            return [
                {"label": "Files Processed", "value": "0"},
                {"label": "Valid", "value": "0"},
                {"label": "Invalid", "value": "0"},
                {"label": "Last Activity", "value": "N/A"},
            ]

        summary = self._collector.get_summary()
        total = summary["total_files_processed"]
        passed = summary["validation_passed"]
        failed = summary["validation_failed"]
        last_activity = summary["last_activity"]

        return [
            {"label": "Files Processed", "value": str(total)},
            {"label": "Valid", "value": str(passed)},
            {"label": "Invalid", "value": str(failed)},
            {"label": "Last Activity", "value": last_activity or "N/A"},
        ]

    def record_error(self, filename: str, message: str) -> None:
        """Store a recent validation error.

        Maintains a FIFO buffer of at most 10 errors, discarding the
        oldest when the limit is reached.

        Args:
            filename: Name of the file that failed validation.
            message: Description of the validation error.
        """
        error = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filename": filename,
            "message": message,
        }
        with self._lock:
            self._recent_errors.append(error)
            if len(self._recent_errors) > self._MAX_ERRORS:
                self._recent_errors = self._recent_errors[-self._MAX_ERRORS :]

    def get_recent_errors(self, limit: int = 5) -> list[dict]:
        """Return recent validation errors, newest first.

        Args:
            limit: Maximum number of errors to return.

        Returns:
            List of dicts with 'timestamp', 'filename', and 'message' keys.
        """
        with self._lock:
            # Return newest first by reversing, then apply limit
            return list(reversed(self._recent_errors))[:limit]

    def clear_errors(self) -> None:
        """Remove all stored validation errors."""
        with self._lock:
            self._recent_errors.clear()
