"""Aggregator: State machine + Heap-based due time management.

State flow:
DEBOUNCING → (due_time reached) → STABILITY_CHECK → (stable) → READY
                                                 ↓ (not stable)
                                            back to STABILITY_CHECK
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from ..models import AggregatorState, FileEventDTO, FileState

logger = logging.getLogger("filehub")


class Aggregator:
    """Event aggregator (state machine + heap)."""

    def __init__(self, debounce_seconds: float = 1.0, cooldown_seconds: float = 1800.0):
        self._debounce = debounce_seconds
        self._cooldown = cooldown_seconds

        # Path → FileState
        self._states: dict[Path, FileState] = {}

        # (due_time, path) heap
        self._heap: list[tuple[float, Path]] = []

        self._lock = threading.RLock()  # Reentrant lock for nested calls

    def add_event(self, event: FileEventDTO) -> None:
        """Add event (debounce)."""
        with self._lock:
            path = event.path

            if path in self._states:
                # Update existing state (reset debounce)
                state = self._states[path]
                state.update_event(event.timestamp, self._debounce)
            else:
                # Create new state
                state = FileState(path=path)
                state.update_event(event.timestamp, self._debounce)
                self._states[path] = state

            heapq.heappush(self._heap, (state.due_time, path))

    def get_due_items(self, current_time: float) -> Iterator[FileState]:
        """Return items past due_time (DEBOUNCING → STABILITY_CHECK)."""
        with self._lock:
            while self._heap:
                due_time, path = self._heap[0]

                if due_time > current_time:
                    break

                heapq.heappop(self._heap)

                state = self._states.get(path)
                if state is None:
                    continue

                # Skip if there's a more recent due_time
                if state.due_time > due_time:
                    continue

                # State transition: DEBOUNCING → STABILITY_CHECK
                if state.state == AggregatorState.DEBOUNCING:
                    state.state = AggregatorState.STABILITY_CHECK

                yield state

    def reschedule(self, state: FileState, delay: float) -> None:
        """Reschedule state (for stability recheck, etc.)."""
        with self._lock:
            new_due = time.time() + delay
            state.due_time = new_due
            heapq.heappush(self._heap, (new_due, state.path))

    def mark_ready(self, state: FileState) -> None:
        """Transition to READY state."""
        with self._lock:
            state.state = AggregatorState.READY

    def mark_notified(self, path: Path) -> None:
        """Set cooldown after notification."""
        with self._lock:
            state = self._states.get(path)
            if state:
                state.last_notified = time.time()
                state.state = AggregatorState.COOLDOWN

    def is_in_cooldown(self, path: Path, current_time: float) -> bool:
        """Check if in cooldown period."""
        with self._lock:
            state = self._states.get(path)
            if state is None:
                return False

            if state.last_notified <= 0:
                return False

            elapsed = current_time - state.last_notified
            return elapsed < self._cooldown

    def remove(self, path: Path) -> None:
        """Remove state (on file deletion)."""
        with self._lock:
            self._states.pop(path, None)

    def cleanup_expired(self, current_time: float) -> int:
        """Remove states whose cooldown has expired.

        Returns:
            Number of states removed.
        """
        with self._lock:
            expired = [
                path
                for path, state in self._states.items()
                if state.state == AggregatorState.COOLDOWN
                and state.last_notified > 0
                and (current_time - state.last_notified) >= self._cooldown
            ]
            for path in expired:
                del self._states[path]
            return len(expired)

    def clear(self) -> None:
        """Clear all states."""
        with self._lock:
            self._states.clear()
            self._heap.clear()
