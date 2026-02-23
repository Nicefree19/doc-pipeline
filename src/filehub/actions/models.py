"""Action models and configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class ActionType(str, Enum):
    """Available action types."""

    MOVE = "move"
    COPY = "copy"
    RENAME = "rename"
    DELETE = "delete"


class TriggerType(str, Enum):
    """Trigger conditions."""

    ALWAYS = "always"
    VALID = "valid"
    INVALID = "invalid"


@dataclass
class ActionRule:
    """Rule defining an action to take."""

    name: str
    action: ActionType
    trigger: TriggerType = TriggerType.VALID

    # Target path pattern (e.g., "D:/Archive/{year}")
    # Supported variables: year, month, day, project, originator, etc.
    target: str | None = None

    # Conflict resolution: skip, overwrite, rename
    conflict: Literal["skip", "overwrite", "rename"] = "rename"

    @classmethod
    def from_dict(cls, data: dict) -> ActionRule:
        """Create rule from dictionary."""
        return cls(
            name=data.get("name", "Unnamed Action"),
            action=ActionType(data.get("action", "move")),
            trigger=TriggerType(data.get("trigger", "valid")),
            target=data.get("target"),
            conflict=data.get("conflict", "rename"),
        )
