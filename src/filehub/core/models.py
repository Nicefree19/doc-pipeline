"""Core data models for FileHub."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

from ..i18n import _


class EventType(Enum):
    """File system event types."""

    CREATED = auto()
    MODIFIED = auto()
    MOVED = auto()
    DELETED = auto()
    WATCHDOG_OVERFLOW = auto()

    def __str__(self) -> str:
        return self.name.lower()


class AggregatorState(Enum):
    """Aggregator state machine states."""

    DEBOUNCING = auto()  # Waiting for debounce
    STABILITY_CHECK = auto()  # Checking file stability
    READY = auto()  # Ready for processing
    COOLDOWN = auto()  # In cooldown period


@dataclass(frozen=True, slots=True)
class FileEventDTO:
    """File event DTO (immutable)."""

    path: Path
    event_type: EventType
    timestamp: float
    src_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))
        if self.src_path is not None and not isinstance(self.src_path, Path):
            object.__setattr__(self, "src_path", Path(self.src_path))

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def extension(self) -> str:
        return self.path.suffix

    def __lt__(self, other: FileEventDTO) -> bool:
        return self.timestamp < other.timestamp


@dataclass(slots=True)
class FileState:
    """Per-file state tracking (for Aggregator).

    State machine: DEBOUNCING → STABILITY_CHECK → READY
    """

    path: Path
    state: AggregatorState = AggregatorState.DEBOUNCING
    first_seen: float = field(default_factory=time.time)
    last_event_time: float = field(default_factory=time.time)
    due_time: float = 0.0  # Scheduled processing time for heap

    # Stability related
    last_size: int = -1
    last_mtime: float = -1.0
    stability_rounds: int = 0

    # Cooldown related
    last_notified: float = 0.0

    def update_event(self, timestamp: float, debounce_sec: float) -> None:
        """Update with new event (reset debounce)."""
        self.last_event_time = timestamp
        self.state = AggregatorState.DEBOUNCING
        self.due_time = timestamp + debounce_sec
        # Reset stability counter
        self.stability_rounds = 0
        self.last_size = -1
        self.last_mtime = -1.0


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Naming convention validation result (immutable)."""

    is_valid: bool
    filename: str
    message: str
    matched_groups: dict[str, str] | None = None

    @classmethod
    def valid(cls, filename: str, groups: dict[str, str]) -> ValidationResult:
        return cls(True, filename, _("Validation passed"), groups)

    @classmethod
    def invalid(cls, filename: str, reason: str) -> ValidationResult:
        return cls(False, filename, reason, None)
