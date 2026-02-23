"""Reconcile scanner to compensate for missing watchdog events."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from queue import Full, Queue
from typing import TYPE_CHECKING

from ..models import EventType, FileEventDTO

if TYPE_CHECKING:
    from ..pipeline.ignore_filter import IgnoreFilter

logger = logging.getLogger("filehub")


class ReconcileScanner:
    """Reconcile scanner.

    - Scans watch folders every 10 minutes
    - Compares with processed file cache to prevent duplicate enqueue
    - Triggers immediate scan on WATCHDOG_OVERFLOW
    """

    def __init__(
        self,
        watch_paths: Iterable[Path],
        queue: Queue[FileEventDTO],
        ignore_filter: IgnoreFilter,
        is_paused: Callable[[], bool],
        interval_seconds: float = 600.0,
    ):
        self._watch_paths = list(watch_paths)
        self._queue = queue
        self._ignore = ignore_filter
        self._is_paused = is_paused
        self._interval = interval_seconds

        self._processed: dict[Path, tuple[int, float]] = {}
        self._lock = threading.Lock()

        self._stop_event = threading.Event()
        self._trigger_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="ReconcileScanner", daemon=True)
        self._thread.start()
        logger.info("ReconcileScanner started")

    def stop(self) -> None:
        self._stop_event.set()
        self._trigger_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("ReconcileScanner stopped")

    def trigger_scan(self, reason: str = "overflow") -> None:
        """Trigger immediate scan (e.g., WATCHDOG_OVERFLOW)."""
        logger.warning("Reconcile scan triggered: %s", reason)
        self._trigger_event.set()

    def mark_processed(self, path: Path) -> None:
        """Update processed file cache."""
        try:
            stat = path.stat()
        except OSError:
            return
        with self._lock:
            self._processed[path] = (stat.st_size, stat.st_mtime)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            triggered = self._trigger_event.wait(timeout=self._interval)
            self._trigger_event.clear()

            if self._stop_event.is_set():
                break

            if self._is_paused():
                continue

            try:
                self.scan(triggered=triggered)
            except Exception as e:
                logger.exception("ReconcileScanner scan error: %s", e)
                time.sleep(5.0)

    def scan(self, triggered: bool = False) -> None:
        """Scan folders and add reconcile events to queue."""
        if not self._watch_paths:
            return

        logger.info("Reconcile scan started%s", " (triggered)" if triggered else "")

        seen: set[Path] = set()
        now = time.time()

        for root in self._watch_paths:
            if not root.exists() or not root.is_dir():
                continue

            def _on_walk_error(err: OSError) -> None:
                logger.warning("Reconcile scan permission error: %s", err)

            for dirpath, _dirnames, filenames in os.walk(root, onerror=_on_walk_error):
                for fname in filenames:
                    path = Path(dirpath) / fname
                    seen.add(path)

                    try:
                        stat = path.stat()
                    except OSError:
                        continue

                    # Apply ignore filter
                    event = FileEventDTO(path=path, event_type=EventType.MODIFIED, timestamp=now)
                    try:
                        if self._ignore.should_ignore(event):
                            continue
                    except Exception as e:
                        logger.warning("Ignore filter error for %s: %s", path, e)
                        continue

                    signature = (stat.st_size, stat.st_mtime)
                    with self._lock:
                        if self._processed.get(path) == signature:
                            continue
                        self._processed[path] = signature

                    try:
                        self._queue.put_nowait(event)
                    except Full:
                        logger.warning("Reconcile enqueue drop (queue full): %s", path)

        # Remove stale entries from cache
        with self._lock:
            stale = [p for p in self._processed.keys() if p not in seen]
            for p in stale:
                self._processed.pop(p, None)

        logger.info("Reconcile scan complete: %d files", len(seen))
