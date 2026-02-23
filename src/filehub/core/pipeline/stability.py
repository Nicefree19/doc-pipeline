"""Stability checker: size + mtime based.

Requirements:
- stability_timeout: 20s
- stability_check_interval: 0.5s
- stability_rounds: 2 consecutive identical
- nonzero: size 0 is unstable
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from ..models import FileState

logger = logging.getLogger("filehub")


def is_file_locked(path: Path) -> bool:
    """Check if file is locked by another process."""
    try:
        with open(path, "rb"):
            return False
    except PermissionError:
        return True
    except OSError:
        return False


class StabilityChecker:
    """File stability checker."""

    def __init__(self, timeout: float = 20.0, interval: float = 0.5, required_rounds: int = 2):
        self._timeout = timeout
        self._interval = interval
        self._required_rounds = required_rounds

    def check(self, state: FileState) -> bool:
        """Check stability.

        Returns:
            True: Stable (ready for processing)
            False: Still unstable (needs recheck)
        """
        path = state.path

        # Check file exists
        if not path.exists():
            logger.debug("File not found: %s", path)
            return False

        # Ignore directories
        if path.is_dir():
            return False

        # Check file lock
        if is_file_locked(path):
            logger.debug("File locked: %s", path)
            state.stability_rounds = 0
            return False

        try:
            stat = path.stat()
            current_size = stat.st_size
            current_mtime = stat.st_mtime
        except (OSError, PermissionError) as e:
            logger.debug("stat failed: %s - %s", path, e)
            return False

        # Nonzero check
        if current_size == 0:
            logger.debug("Size 0: %s", path)
            state.stability_rounds = 0
            return False

        # First check
        if state.last_size < 0:
            state.last_size = current_size
            state.last_mtime = current_mtime
            state.stability_rounds = 1
            return False

        # Check for changes
        if current_size != state.last_size or current_mtime != state.last_mtime:
            # Changed → reset counter
            state.last_size = current_size
            state.last_mtime = current_mtime
            state.stability_rounds = 1
            return False

        # Same → increment rounds
        state.stability_rounds += 1

        if state.stability_rounds >= self._required_rounds:
            logger.debug("Stability confirmed: %s (rounds=%d)", path, state.stability_rounds)
            return True

        return False

    @property
    def interval(self) -> float:
        """Recheck interval."""
        return self._interval

    @property
    def timeout(self) -> float:
        """Maximum wait time."""
        return self._timeout

    def is_timed_out(self, state: FileState, now: float | None = None) -> bool:
        """Check if stability check has timed out."""
        current = now if now is not None else time.time()
        return (current - state.first_seen) >= self._timeout
