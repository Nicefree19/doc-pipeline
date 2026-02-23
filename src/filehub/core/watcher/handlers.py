"""Watchdog event handlers (Producer).

Converts watchdog events to FileEventDTO and enqueues them.
- Gating: events are not enqueued when paused
- Queue full: log and drop
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from queue import Full, Queue

from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)

from ..models import EventType, FileEventDTO

logger = logging.getLogger("filehub")


class FileEventHandler(FileSystemEventHandler):
    """Watchdog event handler."""

    EVENT_TYPE_MAP = {
        FileCreatedEvent: EventType.CREATED,
        FileModifiedEvent: EventType.MODIFIED,
        FileMovedEvent: EventType.MOVED,
        FileDeletedEvent: EventType.DELETED,
    }

    def __init__(
        self,
        queue: Queue[FileEventDTO],
        is_paused: Callable[[], bool],
        on_overflow: Callable[[], None] | None = None,
    ):
        super().__init__()
        self._queue = queue
        self._is_paused = is_paused
        self._on_overflow = on_overflow

    def on_any_event(self, event: FileSystemEvent) -> None:
        # Overflow event triggers reconcile scan immediately
        if self._is_overflow_event(event):
            logger.warning("WATCHDOG_OVERFLOW detected - triggering reconcile scan")
            if self._on_overflow:
                self._on_overflow()
            return

        if event.is_directory:
            return

        if self._is_paused():
            return

        event_type = self.EVENT_TYPE_MAP.get(type(event))
        if event_type is None:
            return

        dto = self._convert_event(event, event_type)
        if dto is None:
            return

        try:
            self._queue.put_nowait(dto)
        except Full:
            logger.warning(
                "Event queue full - dropping: %s", dto.path, extra={"file_path": dto.path}
            )

    @staticmethod
    def _is_overflow_event(event: FileSystemEvent) -> bool:
        """Detect watchdog overflow events."""
        event_type = getattr(event, "event_type", "")
        if isinstance(event_type, str) and event_type.lower() == "overflow":
            return True
        name = event.__class__.__name__.lower()
        return "overflow" in name

    def _convert_event(self, event: FileSystemEvent, event_type: EventType) -> FileEventDTO | None:
        try:
            path = Path(str(event.src_path))
        except (TypeError, ValueError):
            return None

        src_path = None
        if event_type == EventType.MOVED and hasattr(event, "dest_path"):
            src_path = path
            path = Path(str(event.dest_path))

        return FileEventDTO(
            path=path, event_type=event_type, timestamp=time.time(), src_path=src_path
        )
