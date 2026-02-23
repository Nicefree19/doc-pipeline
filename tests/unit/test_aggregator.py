"""Tests for Aggregator module."""

import time
from pathlib import Path

from src.filehub.core.models import AggregatorState, EventType, FileEventDTO
from src.filehub.core.pipeline.aggregator import Aggregator


class TestAggregator:
    """Test Aggregator functionality."""

    def test_add_event_creates_state(self):
        """Test that add_event creates a new file state."""
        agg = Aggregator(debounce_seconds=0.1)
        event = FileEventDTO(
            path=Path("/test/file.txt"), event_type=EventType.CREATED, timestamp=time.time()
        )

        agg.add_event(event)

        # State should exist
        assert event.path in agg._states
        state = agg._states[event.path]
        assert state.state == AggregatorState.DEBOUNCING

    def test_add_event_updates_existing_state(self):
        """Test that subsequent events update existing state."""
        agg = Aggregator(debounce_seconds=0.5)
        path = Path("/test/file.txt")

        event1 = FileEventDTO(path=path, event_type=EventType.CREATED, timestamp=1.0)
        event2 = FileEventDTO(path=path, event_type=EventType.MODIFIED, timestamp=1.3)

        agg.add_event(event1)
        first_due = agg._states[path].due_time

        agg.add_event(event2)
        second_due = agg._states[path].due_time

        # Due time should be updated
        assert second_due > first_due

    def test_get_due_items_returns_ready(self):
        """Test that get_due_items returns items past due time."""
        agg = Aggregator(debounce_seconds=0.0)  # No debounce
        event = FileEventDTO(
            path=Path("/test/file.txt"), event_type=EventType.CREATED, timestamp=time.time() - 1.0
        )

        agg.add_event(event)

        # Get due items
        due_items = list(agg.get_due_items(time.time()))

        assert len(due_items) == 1
        assert due_items[0].state == AggregatorState.STABILITY_CHECK

    def test_get_due_items_skips_not_ready(self):
        """Test that get_due_items skips items not yet due."""
        agg = Aggregator(debounce_seconds=10.0)  # Long debounce
        event = FileEventDTO(
            path=Path("/test/file.txt"), event_type=EventType.CREATED, timestamp=time.time()
        )

        agg.add_event(event)

        # Get due items (nothing should be ready)
        due_items = list(agg.get_due_items(time.time()))

        assert len(due_items) == 0

    def test_reschedule_updates_due_time(self):
        """Test that reschedule updates the due time."""
        agg = Aggregator(debounce_seconds=0.0)
        event = FileEventDTO(
            path=Path("/test/file.txt"), event_type=EventType.CREATED, timestamp=time.time() - 1.0
        )

        agg.add_event(event)
        due_items = list(agg.get_due_items(time.time()))
        state = due_items[0]

        old_due = state.due_time
        agg.reschedule(state, delay=1.0)

        assert state.due_time > old_due

    def test_mark_notified_sets_cooldown(self):
        """Test that mark_notified sets cooldown state."""
        agg = Aggregator(debounce_seconds=0.0, cooldown_seconds=60.0)
        path = Path("/test/file.txt")
        event = FileEventDTO(path=path, event_type=EventType.CREATED, timestamp=time.time())

        agg.add_event(event)
        agg.mark_notified(path)

        state = agg._states[path]
        assert state.state == AggregatorState.COOLDOWN
        assert state.last_notified > 0

    def test_is_in_cooldown(self):
        """Test cooldown detection."""
        agg = Aggregator(debounce_seconds=0.0, cooldown_seconds=60.0)
        path = Path("/test/file.txt")
        event = FileEventDTO(path=path, event_type=EventType.CREATED, timestamp=time.time())

        agg.add_event(event)

        # Not in cooldown yet
        assert not agg.is_in_cooldown(path, time.time())

        # Mark notified
        agg.mark_notified(path)

        # Now in cooldown
        assert agg.is_in_cooldown(path, time.time())

        # Not in cooldown after cooldown period
        assert not agg.is_in_cooldown(path, time.time() + 61.0)

    def test_remove_clears_state(self):
        """Test that remove clears file state."""
        agg = Aggregator(debounce_seconds=0.1)
        path = Path("/test/file.txt")
        event = FileEventDTO(path=path, event_type=EventType.CREATED, timestamp=time.time())

        agg.add_event(event)
        assert path in agg._states

        agg.remove(path)
        assert path not in agg._states

    def test_clear_resets_all(self):
        """Test that clear resets all states."""
        agg = Aggregator(debounce_seconds=0.1)

        for i in range(5):
            event = FileEventDTO(
                path=Path(f"/test/file{i}.txt"), event_type=EventType.CREATED, timestamp=time.time()
            )
            agg.add_event(event)

        assert len(agg._states) == 5

        agg.clear()

        assert len(agg._states) == 0
        assert len(agg._heap) == 0
