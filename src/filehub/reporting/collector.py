"""In-memory statistics collector for FileHub."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path


class StatsCollector:
    """Thread-safe in-memory statistics collector.

    Tracks file processing counts, validation results, and event counts
    without requiring persistent storage.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_files_processed: int = 0
        self._validation_passed: int = 0
        self._validation_failed: int = 0
        self._events_by_type: dict[str, int] = {}
        self._last_activity: datetime | None = None

    def record_file_processed(
        self, path: Path, processing_time: float = 0.0
    ) -> None:
        """Record that a file was processed.

        Args:
            path: Path to the processed file.
            processing_time: Time in seconds the processing took.
        """
        with self._lock:
            self._total_files_processed += 1
            self._last_activity = datetime.now(timezone.utc)

    def record_validation_result(self, path: Path, is_valid: bool) -> None:
        """Record a validation result for a file.

        Args:
            path: Path to the validated file.
            is_valid: Whether the file passed validation.
        """
        with self._lock:
            if is_valid:
                self._validation_passed += 1
            else:
                self._validation_failed += 1
            self._last_activity = datetime.now(timezone.utc)

    def record_event(self, event_type: str) -> None:
        """Record a file system event by type.

        Args:
            event_type: The type of event (e.g. 'CREATED', 'MODIFIED').
        """
        with self._lock:
            self._events_by_type[event_type] = (
                self._events_by_type.get(event_type, 0) + 1
            )
            self._last_activity = datetime.now(timezone.utc)

    def get_summary(self) -> dict:
        """Return a summary dict of all collected statistics.

        Returns:
            Dictionary with keys: total_files_processed, validation_passed,
            validation_failed, events_by_type, last_activity.
        """
        with self._lock:
            return {
                "total_files_processed": self._total_files_processed,
                "validation_passed": self._validation_passed,
                "validation_failed": self._validation_failed,
                "events_by_type": dict(self._events_by_type),
                "last_activity": (
                    self._last_activity.isoformat()
                    if self._last_activity is not None
                    else None
                ),
            }

    def reset(self) -> None:
        """Clear all counters and reset to initial state."""
        with self._lock:
            self._total_files_processed = 0
            self._validation_passed = 0
            self._validation_failed = 0
            self._events_by_type.clear()
            self._last_activity = None
