"""Base action classes for file operations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("filehub")


@dataclass(frozen=True)
class ActionResult:
    """Result of a file action.

    Attributes:
        success: Whether the action completed successfully
        source: Original file path
        destination: New file path (if changed)
        message: Description of what happened
        dry_run: Whether this was a dry run (no actual changes)
    """

    success: bool
    source: Path
    destination: Path | None = None
    message: str = ""
    dry_run: bool = False


class FileAction(ABC):
    """Abstract base class for file actions."""

    def __init__(self, dry_run: bool = False):
        self._dry_run = dry_run

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @property
    @abstractmethod
    def name(self) -> str:
        """Action name for logging."""
        ...

    @abstractmethod
    def execute(self, path: Path, **context) -> ActionResult:
        """Execute the action on a file.

        Args:
            path: File path to act on
            **context: Additional context (validation_result, config, etc.)

        Returns:
            ActionResult with outcome details
        """
        ...
