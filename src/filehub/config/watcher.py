"""Configuration file watcher with hot-reload support.

Monitors the configuration file for changes and reloads it
automatically, calling a user-provided callback with the new config.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .loader import load_config
from .schema import FileHubConfig

logger = logging.getLogger(__name__)


class _ConfigFileHandler(FileSystemEventHandler):
    """Watchdog handler that filters events for a specific config file."""

    def __init__(self, config_filename: str, on_change: Callable[[], None]) -> None:
        super().__init__()
        self._config_filename = config_filename
        self._on_change = on_change

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        changed_path = Path(str(event.src_path))
        if changed_path.name == self._config_filename:
            self._on_change()

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        changed_path = Path(str(event.src_path))
        if changed_path.name == self._config_filename:
            self._on_change()


class ConfigWatcher:
    """Watches a configuration file and triggers hot-reload on changes.

    Uses watchdog to monitor the parent directory of the config file.
    On detecting a change to the config file, it debounces rapid changes,
    validates the new configuration, and calls the on_reload callback
    only when the new config is valid.

    Args:
        config_path: Path to the configuration file to watch.
        on_reload: Callback invoked with the new FileHubConfig on valid reload.
        debounce_seconds: Minimum interval between reload attempts.
    """

    def __init__(
        self,
        config_path: Path,
        on_reload: Callable[[FileHubConfig], None],
        debounce_seconds: float = 1.0,
    ) -> None:
        self._config_path = Path(config_path)
        self._on_reload = on_reload
        self._debounce_seconds = debounce_seconds
        self._observer: Observer | None = None  # type: ignore
        self._lock = threading.Lock()
        self._debounce_timer: threading.Timer | None = None

    def start(self) -> None:
        """Start watching the config file in a background thread.

        If already running, this method is a no-op (idempotent).
        """
        if self.is_running:
            logger.debug("ConfigWatcher already running, ignoring start()")
            return

        watch_dir = self._config_path.parent
        handler = _ConfigFileHandler(
            config_filename=self._config_path.name,
            on_change=self._schedule_reload,
        )

        # self._observer is typed as Any/Observer
        self._observer = Observer()
        self._observer.schedule(handler, str(watch_dir), recursive=False)
        self._observer.daemon = True
        self._observer.start()
        logger.info("ConfigWatcher started, monitoring: %s", self._config_path)

    def stop(self) -> None:
        """Stop watching the config file.

        Safe to call even if not currently running.
        """
        with self._lock:
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()
                self._debounce_timer = None

        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
            logger.info("ConfigWatcher stopped")

    @property
    def is_running(self) -> bool:
        """Return True if the watcher is actively monitoring."""
        return self._observer is not None and self._observer.is_alive()

    def _schedule_reload(self) -> None:
        """Schedule a debounced config reload."""
        with self._lock:
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()
            self._debounce_timer = threading.Timer(
                self._debounce_seconds,
                self._handle_change,
            )
            self._debounce_timer.daemon = True
            self._debounce_timer.start()

    def _handle_change(self) -> None:
        """Load and validate the changed config, then invoke the callback."""
        logger.info("Config file change detected: %s", self._config_path)
        try:
            new_config = load_config(self._config_path)
        except Exception as exc:
            logger.warning(
                "Config reload failed, keeping previous config: %s",
                exc,
            )
            return

        try:
            self._on_reload(new_config)
            logger.info("Config reloaded successfully from: %s", self._config_path)
        except Exception as exc:
            logger.error(
                "on_reload callback raised an exception: %s",
                exc,
            )
