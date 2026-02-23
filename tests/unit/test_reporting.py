"""Tests for the reporting module."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.filehub.reporting.collector import StatsCollector
from src.filehub.reporting.report import ReportGenerator
from src.filehub.reporting.store import StatsStore


class TestStatsCollector:
    """Test StatsCollector in-memory counters."""

    def test_initial_summary_all_zeros(self):
        """A fresh collector should report all zeroes and no last activity."""
        collector = StatsCollector()
        summary = collector.get_summary()

        assert summary["total_files_processed"] == 0
        assert summary["validation_passed"] == 0
        assert summary["validation_failed"] == 0
        assert summary["events_by_type"] == {}
        assert summary["last_activity"] is None

    def test_record_file_processed_increments_count(self):
        """Recording file processed should increment the counter."""
        collector = StatsCollector()
        collector.record_file_processed(Path("/test/file.txt"), processing_time=0.1)
        collector.record_file_processed(Path("/test/file2.txt"), processing_time=0.2)

        summary = collector.get_summary()
        assert summary["total_files_processed"] == 2

    def test_record_validation_passed(self):
        """Recording a passed validation should increment validation_passed."""
        collector = StatsCollector()
        collector.record_validation_result(Path("/test/file.txt"), is_valid=True)

        summary = collector.get_summary()
        assert summary["validation_passed"] == 1
        assert summary["validation_failed"] == 0

    def test_record_validation_failed(self):
        """Recording a failed validation should increment validation_failed."""
        collector = StatsCollector()
        collector.record_validation_result(Path("/test/file.txt"), is_valid=False)

        summary = collector.get_summary()
        assert summary["validation_passed"] == 0
        assert summary["validation_failed"] == 1

    def test_record_event_by_type(self):
        """Recording events should group counts by event type."""
        collector = StatsCollector()
        collector.record_event("CREATED")
        collector.record_event("CREATED")
        collector.record_event("MODIFIED")

        summary = collector.get_summary()
        assert summary["events_by_type"]["CREATED"] == 2
        assert summary["events_by_type"]["MODIFIED"] == 1

    def test_reset_clears_all(self):
        """Resetting should return all counters to initial state."""
        collector = StatsCollector()
        collector.record_file_processed(Path("/test/file.txt"))
        collector.record_validation_result(Path("/test/file.txt"), is_valid=True)
        collector.record_event("CREATED")

        collector.reset()
        summary = collector.get_summary()

        assert summary["total_files_processed"] == 0
        assert summary["validation_passed"] == 0
        assert summary["validation_failed"] == 0
        assert summary["events_by_type"] == {}
        assert summary["last_activity"] is None

    def test_last_activity_timestamp(self):
        """Last activity should be set after any recording operation."""
        collector = StatsCollector()
        assert collector.get_summary()["last_activity"] is None

        collector.record_file_processed(Path("/test/file.txt"))
        last_activity = collector.get_summary()["last_activity"]

        assert last_activity is not None
        # Should be a valid ISO timestamp
        parsed = datetime.fromisoformat(last_activity)
        assert parsed.tzinfo is not None


class TestStatsStore:
    """Test StatsStore SQLite persistence."""

    def test_init_creates_tables(self):
        """Initializing a store should create the required tables."""
        store = StatsStore(db_path=":memory:")
        try:
            cursor = store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row["name"] for row in cursor.fetchall()]
            assert "file_events" in tables
            assert "validation_results" in tables
        finally:
            store.close()

    def test_record_and_get_events(self):
        """Recording events and retrieving them should return correct data."""
        store = StatsStore(db_path=":memory:")
        try:
            store.record_event("/test/file.txt", "CREATED")
            store.record_event("/test/file2.txt", "MODIFIED")

            events = store.get_recent_events(limit=10)
            assert len(events) == 2
            paths = {e["path"] for e in events}
            assert "/test/file.txt" in paths
            assert "/test/file2.txt" in paths
        finally:
            store.close()

    def test_record_and_get_validation_stats(self):
        """Recording validations and retrieving stats should be accurate."""
        store = StatsStore(db_path=":memory:")
        try:
            store.record_validation("/test/good.txt", is_valid=True, message="OK")
            store.record_validation("/test/bad.txt", is_valid=False, message="Invalid")
            store.record_validation("/test/good2.txt", is_valid=True)

            stats = store.get_validation_stats()
            assert stats["passed"] == 2
            assert stats["failed"] == 1
            assert stats["total"] == 3
        finally:
            store.close()

    def test_get_event_counts_by_type(self):
        """Event counts should be grouped correctly by event type."""
        store = StatsStore(db_path=":memory:")
        try:
            store.record_event("/a.txt", "CREATED")
            store.record_event("/b.txt", "CREATED")
            store.record_event("/c.txt", "MODIFIED")
            store.record_event("/d.txt", "DELETED")

            counts = store.get_event_counts()
            assert counts["CREATED"] == 2
            assert counts["MODIFIED"] == 1
            assert counts["DELETED"] == 1
        finally:
            store.close()

    def test_get_recent_events_with_limit(self):
        """Recent events should respect the limit parameter."""
        store = StatsStore(db_path=":memory:")
        try:
            for i in range(10):
                store.record_event(f"/file{i}.txt", "CREATED")

            events = store.get_recent_events(limit=3)
            assert len(events) == 3
        finally:
            store.close()

    def test_get_validation_stats_since_date(self):
        """Validation stats should filter by the 'since' parameter."""
        store = StatsStore(db_path=":memory:")
        try:
            # Insert records with explicit timestamps in the past
            past = "2020-01-01T00:00:00+00:00"
            store._conn.execute(
                "INSERT INTO validation_results (path, is_valid, message, timestamp) "
                "VALUES (?, ?, ?, ?)",
                ("/old.txt", 1, "OK", past),
            )
            store._conn.commit()

            # Insert a recent record via the normal API
            store.record_validation("/new.txt", is_valid=True, message="OK")

            # Query since a date after the old record
            since = "2024-01-01T00:00:00+00:00"
            stats = store.get_validation_stats(since=since)
            assert stats["passed"] == 1
            assert stats["total"] == 1
        finally:
            store.close()

    def test_close_does_not_raise(self):
        """Closing the store should not raise an exception."""
        store = StatsStore(db_path=":memory:")
        store.close()


class TestReportGenerator:
    """Test ReportGenerator output."""

    def test_generate_summary_structure(self):
        """Summary dict should contain all required top-level keys."""
        store = StatsStore(db_path=":memory:")
        try:
            gen = ReportGenerator(store)
            summary = gen.generate_summary()

            assert "generated_at" in summary
            assert "event_counts" in summary
            assert "validation_stats" in summary
            assert "recent_events" in summary
        finally:
            store.close()

    def test_generate_summary_with_data(self):
        """Summary should reflect data stored in the database."""
        store = StatsStore(db_path=":memory:")
        try:
            store.record_event("/a.txt", "CREATED")
            store.record_event("/b.txt", "MODIFIED")
            store.record_validation("/a.txt", is_valid=True, message="OK")
            store.record_validation("/c.txt", is_valid=False, message="Bad name")

            gen = ReportGenerator(store)
            summary = gen.generate_summary()

            assert summary["event_counts"]["CREATED"] == 1
            assert summary["event_counts"]["MODIFIED"] == 1
            assert summary["validation_stats"]["passed"] == 1
            assert summary["validation_stats"]["failed"] == 1
            assert summary["validation_stats"]["total"] == 2
            assert len(summary["recent_events"]) == 2
        finally:
            store.close()

    def test_generate_text_report_contains_info(self):
        """Text report should contain key sections and data."""
        store = StatsStore(db_path=":memory:")
        try:
            store.record_event("/a.txt", "CREATED")
            store.record_validation("/a.txt", is_valid=True)

            gen = ReportGenerator(store)
            text = gen.generate_text_report()

            assert "FileHub Statistics Report" in text
            assert "Validation" in text
            assert "Events" in text
            assert "CREATED" in text
            assert "Passed:" in text
        finally:
            store.close()

    def test_generate_summary_empty_db(self):
        """Summary from an empty database should return zero/empty values."""
        store = StatsStore(db_path=":memory:")
        try:
            gen = ReportGenerator(store)
            summary = gen.generate_summary()

            assert summary["event_counts"] == {}
            assert summary["validation_stats"]["passed"] == 0
            assert summary["validation_stats"]["failed"] == 0
            assert summary["validation_stats"]["total"] == 0
            assert summary["recent_events"] == []
        finally:
            store.close()
