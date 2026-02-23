"""Observer factory for watchdog."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from queue import Queue

from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver
from watchdog.observers.polling import PollingObserver

from ..models import FileEventDTO
from .handlers import FileEventHandler

logger = logging.getLogger("filehub")


def is_network_path(path: Path) -> bool:
    """Check if a path is on a network drive.

    Detects UNC paths (\\\\server\\share) and mapped network drives on Windows.
    """
    path_str = str(path)

    # UNC path
    if path_str.startswith("\\\\") or path_str.startswith("//"):
        return True

    # Windows mapped drive check
    if sys.platform == "win32" and len(path_str) >= 2 and path_str[1] == ":":
        try:
            import ctypes

            drive = path_str[0].upper() + ":\\"
            drive_type = ctypes.windll.kernel32.GetDriveTypeW(drive)
            # DRIVE_REMOTE = 4
            return bool(drive_type == 4)
        except (OSError, AttributeError):
            pass

    return False


def create_observer(
    watch_paths: Sequence[Path],
    queue: Queue[FileEventDTO],
    is_paused: Callable[[], bool],
    on_overflow: Callable[[], None] | None = None,
    recursive: bool = True,
    use_polling: bool = False,
    poll_interval: float = 2.0,
) -> BaseObserver:
    """Create observer and register handlers.

    Args:
        watch_paths: Directories to watch
        queue: Event queue for file events
        is_paused: Callable returning whether watching is paused
        on_overflow: Callback for overflow events
        recursive: Watch subdirectories
        use_polling: Force polling mode (for NAS/network drives)
        poll_interval: Polling interval in seconds (only for polling mode)

    Returns:
        Observer instance (native or polling)
    """
    # Determine if polling is needed
    needs_polling = use_polling

    if not needs_polling:
        # Auto-detect network paths
        for path in watch_paths:
            if path.exists() and is_network_path(path):
                logger.info("Network path detected: %s - enabling polling mode", path)
                needs_polling = True
                break

    observer: BaseObserver
    if needs_polling:
        observer = PollingObserver(timeout=poll_interval)
        logger.info("Using PollingObserver (interval=%.1fs)", poll_interval)
    else:
        observer = Observer()
        logger.info("Using native Observer")

    handler = FileEventHandler(queue, is_paused=is_paused, on_overflow=on_overflow)

    for path in watch_paths:
        if not path.exists():
            logger.warning("Watch path does not exist: %s", path)
            continue
        if not path.is_dir():
            logger.warning("Watch path is not a directory: %s", path)
            continue
        try:
            observer.schedule(handler, str(path), recursive=recursive)
            logger.info("Watch path added: %s", path)
        except Exception as e:
            logger.exception("Failed to add watch path: %s - %s", path, e)

    return observer


def start_observer(observer: BaseObserver) -> None:
    """Start the observer."""
    if not observer.is_alive():
        observer.start()
        logger.info("File watching started")


def stop_observer(observer: BaseObserver, timeout: float = 5.0) -> None:
    """Stop the observer."""
    if observer.is_alive():
        observer.stop()
        observer.join(timeout=timeout)
        logger.info("File watching stopped")
