"""Report generation for FileHub statistics."""

from __future__ import annotations

from datetime import datetime, timezone

from .store import StatsStore


class ReportGenerator:
    """Generates human-readable and machine-readable reports from stored statistics.

    Args:
        store: A StatsStore instance providing the data for reports.
    """

    def __init__(self, store: StatsStore) -> None:
        self._store = store

    def generate_summary(self, since: str | None = None) -> dict:
        """Generate a comprehensive summary dictionary.

        Args:
            since: Optional ISO timestamp to filter data from.

        Returns:
            Dictionary with keys: generated_at, event_counts,
            validation_stats, recent_events.
        """
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "event_counts": self._store.get_event_counts(since=since),
            "validation_stats": self._store.get_validation_stats(since=since),
            "recent_events": self._store.get_recent_events(),
        }

    def generate_text_report(self, since: str | None = None) -> str:
        """Generate a human-readable text report.

        Args:
            since: Optional ISO timestamp to filter data from.

        Returns:
            Formatted multi-line string report.
        """
        summary = self.generate_summary(since=since)

        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("FileHub Statistics Report")
        lines.append("=" * 60)
        lines.append(f"Generated: {summary['generated_at']}")
        if since:
            lines.append(f"Since: {since}")
        lines.append("")

        # Validation stats
        vs = summary["validation_stats"]
        lines.append("--- Validation ---")
        lines.append(f"  Total:  {vs['total']}")
        lines.append(f"  Passed: {vs['passed']}")
        lines.append(f"  Failed: {vs['failed']}")
        lines.append("")

        # Event counts
        ec = summary["event_counts"]
        lines.append("--- Events ---")
        if ec:
            for event_type, count in sorted(ec.items()):
                lines.append(f"  {event_type}: {count}")
        else:
            lines.append("  No events recorded.")
        lines.append("")

        # Recent events
        recent = summary["recent_events"]
        lines.append(f"--- Recent Events (last {len(recent)}) ---")
        if recent:
            for evt in recent:
                lines.append(
                    f"  [{evt['timestamp']}] {evt['event_type']} - {evt['path']}"
                )
        else:
            lines.append("  No recent events.")

        lines.append("=" * 60)
        return "\n".join(lines)
