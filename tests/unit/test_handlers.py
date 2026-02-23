"""Tests for FileEventHandler module."""

from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock

from src.filehub.core.models import EventType, FileEventDTO
from src.filehub.core.watcher.handlers import FileEventHandler


def _make_handler(queue=None, is_paused=None, on_overflow=None):
    """Helper to create a FileEventHandler."""
    if queue is None:
        queue = Queue(maxsize=100)
    if is_paused is None:

        def is_paused():
            return False
    return FileEventHandler(queue=queue, is_paused=is_paused, on_overflow=on_overflow)


def _make_file_event(event_class_name, src_path="/test/file.txt", dest_path=None):
    """Helper to create a mock watchdog FileSystemEvent."""
    from watchdog.events import (
        FileCreatedEvent,
        FileDeletedEvent,
        FileModifiedEvent,
        FileMovedEvent,
    )

    cls_map = {
        "FileCreatedEvent": FileCreatedEvent,
        "FileDeletedEvent": FileDeletedEvent,
        "FileModifiedEvent": FileModifiedEvent,
        "FileMovedEvent": FileMovedEvent,
    }

    cls = cls_map[event_class_name]
    if event_class_name == "FileMovedEvent":
        return cls(src_path=src_path, dest_path=dest_path or "/test/moved_file.txt")
    return cls(src_path=src_path)


class TestFileEventHandler:
    """Test FileEventHandler functionality."""

    def test_on_created_enqueues_dto(self):
        """Test that a FileCreatedEvent produces a CREATED DTO in the queue."""
        queue = Queue(maxsize=100)
        handler = _make_handler(queue=queue)

        event = _make_file_event("FileCreatedEvent", "/test/new_file.txt")
        handler.on_any_event(event)

        assert queue.qsize() == 1
        dto = queue.get_nowait()
        assert isinstance(dto, FileEventDTO)
        assert dto.event_type == EventType.CREATED
        assert dto.path == Path("/test/new_file.txt")

    def test_on_modified_enqueues_dto(self):
        """Test that a FileModifiedEvent produces a MODIFIED DTO in the queue."""
        queue = Queue(maxsize=100)
        handler = _make_handler(queue=queue)

        event = _make_file_event("FileModifiedEvent", "/test/changed.txt")
        handler.on_any_event(event)

        assert queue.qsize() == 1
        dto = queue.get_nowait()
        assert dto.event_type == EventType.MODIFIED

    def test_on_deleted_enqueues_dto(self):
        """Test that a FileDeletedEvent produces a DELETED DTO in the queue."""
        queue = Queue(maxsize=100)
        handler = _make_handler(queue=queue)

        event = _make_file_event("FileDeletedEvent", "/test/removed.txt")
        handler.on_any_event(event)

        assert queue.qsize() == 1
        dto = queue.get_nowait()
        assert dto.event_type == EventType.DELETED

    def test_on_moved_enqueues_dto_with_dest(self):
        """Test that a FileMovedEvent produces a MOVED DTO with dest as path."""
        queue = Queue(maxsize=100)
        handler = _make_handler(queue=queue)

        event = _make_file_event(
            "FileMovedEvent",
            src_path="/test/old_name.txt",
            dest_path="/test/new_name.txt",
        )
        handler.on_any_event(event)

        assert queue.qsize() == 1
        dto = queue.get_nowait()
        assert dto.event_type == EventType.MOVED
        assert dto.path == Path("/test/new_name.txt")
        assert dto.src_path == Path("/test/old_name.txt")

    def test_directory_events_ignored(self):
        """Test that directory events are not enqueued."""
        queue = Queue(maxsize=100)
        handler = _make_handler(queue=queue)

        event = MagicMock()
        event.is_directory = True
        event.event_type = "created"
        event.__class__.__name__ = "DirCreatedEvent"

        handler.on_any_event(event)

        assert queue.qsize() == 0

    def test_paused_events_ignored(self):
        """Test that events are not enqueued when handler is paused."""
        queue = Queue(maxsize=100)
        handler = _make_handler(queue=queue, is_paused=lambda: True)

        event = _make_file_event("FileCreatedEvent", "/test/file.txt")
        handler.on_any_event(event)

        assert queue.qsize() == 0

    def test_queue_full_drops_gracefully(self):
        """Test that events are dropped gracefully when queue is full."""
        queue = Queue(maxsize=1)
        handler = _make_handler(queue=queue)

        # Fill the queue
        event1 = _make_file_event("FileCreatedEvent", "/test/file1.txt")
        handler.on_any_event(event1)
        assert queue.qsize() == 1

        # This should be dropped (queue full) without exception
        event2 = _make_file_event("FileCreatedEvent", "/test/file2.txt")
        handler.on_any_event(event2)

        # Queue should still have only 1 item
        assert queue.qsize() == 1

    def test_overflow_event_detected(self):
        """Test that overflow events trigger the on_overflow callback."""
        overflow_called = []
        handler = _make_handler(on_overflow=lambda: overflow_called.append(True))

        # Create a mock overflow event
        overflow_event = MagicMock()
        overflow_event.is_directory = False
        overflow_event.event_type = "overflow"
        overflow_event.__class__ = type("OverflowEvent", (), {"__name__": "OverflowEvent"})
        type(overflow_event).__name__ = "OverflowEvent"

        handler.on_any_event(overflow_event)

        assert len(overflow_called) == 1

    def test_overflow_event_does_not_enqueue(self):
        """Test that overflow events are not enqueued as file events."""
        queue = Queue(maxsize=100)
        handler = _make_handler(queue=queue, on_overflow=lambda: None)

        overflow_event = MagicMock()
        overflow_event.is_directory = False
        overflow_event.event_type = "overflow"
        overflow_event.__class__ = type("OverflowEvent", (), {"__name__": "OverflowEvent"})
        type(overflow_event).__name__ = "OverflowEvent"

        handler.on_any_event(overflow_event)

        assert queue.qsize() == 0
