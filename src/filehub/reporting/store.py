"""SQLite-based statistics persistence store for FileHub."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class StatsStore:
    """SQLite-backed persistent statistics store.

    Provides durable storage for file events and validation results
    with querying capabilities for reporting.

    Uses ``check_same_thread=False`` for thread safety across threads.
    """

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(
            self._db_path, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they do not already exist."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS file_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS validation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                is_valid INTEGER NOT NULL,
                message TEXT NOT NULL DEFAULT '',
                timestamp TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def record_event(self, path: str, event_type: str) -> None:
        """Record a file system event.

        Args:
            path: File path associated with the event.
            event_type: The type of event (e.g. 'CREATED').
        """
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO file_events (path, event_type, timestamp) VALUES (?, ?, ?)",
            (path, event_type, now),
        )
        self._conn.commit()

    def record_validation(
        self, path: str, is_valid: bool, message: str = ""
    ) -> None:
        """Record a validation result.

        Args:
            path: File path that was validated.
            is_valid: Whether the file passed validation.
            message: Optional validation message or reason.
        """
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO validation_results (path, is_valid, message, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (path, int(is_valid), message, now),
        )
        self._conn.commit()

    def get_event_counts(self, since: str | None = None) -> dict[str, int]:
        """Get event counts grouped by event type.

        Args:
            since: Optional ISO timestamp to filter events from.

        Returns:
            Dictionary mapping event_type to count.
        """
        if since is not None:
            cursor = self._conn.execute(
                "SELECT event_type, COUNT(*) as cnt "
                "FROM file_events WHERE timestamp >= ? "
                "GROUP BY event_type",
                (since,),
            )
        else:
            cursor = self._conn.execute(
                "SELECT event_type, COUNT(*) as cnt "
                "FROM file_events GROUP BY event_type"
            )
        return {row["event_type"]: row["cnt"] for row in cursor.fetchall()}

    def get_validation_stats(
        self, since: str | None = None
    ) -> dict:
        """Get validation statistics.

        Args:
            since: Optional ISO timestamp to filter results from.

        Returns:
            Dictionary with keys: passed, failed, total.
        """
        if since is not None:
            cursor = self._conn.execute(
                "SELECT is_valid, COUNT(*) as cnt "
                "FROM validation_results WHERE timestamp >= ? "
                "GROUP BY is_valid",
                (since,),
            )
        else:
            cursor = self._conn.execute(
                "SELECT is_valid, COUNT(*) as cnt "
                "FROM validation_results GROUP BY is_valid"
            )

        passed = 0
        failed = 0
        for row in cursor.fetchall():
            if row["is_valid"]:
                passed = row["cnt"]
            else:
                failed = row["cnt"]

        return {"passed": passed, "failed": failed, "total": passed + failed}

    def get_recent_events(self, limit: int = 50) -> list[dict]:
        """Get the most recent file events.

        Args:
            limit: Maximum number of events to return.

        Returns:
            List of event dictionaries ordered by timestamp descending.
        """
        cursor = self._conn.execute(
            "SELECT id, path, event_type, timestamp "
            "FROM file_events ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
