"""Naming profile system - Strategy pattern for file name validation.

Profiles define validation rules. The system supports:
1. ISO 19650 (built-in)
2. Regex-based custom profiles (YAML-defined)
3. Custom field-based profiles (via models.NamingProfile)
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ...core.models import ValidationResult
from ...i18n import _

logger = logging.getLogger("filehub")


class NamingProfile(ABC):
    """Abstract base class for naming profiles."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Profile name identifier."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable profile description."""
        ...

    @abstractmethod
    def validate(self, path: str | Path) -> ValidationResult:
        """Validate a file path against this profile's naming rules."""
        ...


@dataclass
class RegexRule:
    """A single regex-based naming rule."""

    pattern: str
    description: str = ""
    example: str = ""
    error_message: str = ""

    def __post_init__(self):
        self._compiled = re.compile(self.pattern)

    def matches(self, filename_stem: str) -> bool:
        """Check if filename matches this rule."""
        return bool(self._compiled.match(filename_stem))


@dataclass
class RegexProfileConfig:
    """Configuration for a regex-based naming profile."""

    name: str = "custom"
    description: str = "Custom naming profile"
    rules: list[RegexRule] = field(default_factory=list)
    extensions: list[str] = field(default_factory=list)
    case_sensitive: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> RegexProfileConfig:
        """Create from dictionary (YAML-parsed data)."""
        rules = [RegexRule(**rd) for rd in data.get("rules", [])]
        return cls(
            name=data.get("name", "custom"),
            description=data.get("description", "Custom naming profile"),
            rules=rules,
            extensions=data.get("extensions", []),
            case_sensitive=data.get("case_sensitive", True),
        )


class RegexProfile(NamingProfile):
    """Regex-based naming profile."""

    def __init__(self, config: RegexProfileConfig):
        self._config = config

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def description(self) -> str:
        return self._config.description

    def validate(self, path: str | Path) -> ValidationResult:
        """Validate file against regex rules."""
        path = Path(path) if isinstance(path, str) else path
        if path.is_dir():
            return ValidationResult.invalid(str(path), _("Cannot validate directory."))

        filename = path.name
        stem = path.stem

        # Check extension restrictions
        if self._config.extensions:
            ext = path.suffix.lower()
            if ext not in [e.lower() for e in self._config.extensions]:
                allowed = ", ".join(self._config.extensions)
                return ValidationResult.invalid(
                    filename,
                    _("Extension '%s' not allowed. Allowed: [%s]") % (ext, allowed),
                )

        check_stem = stem if self._config.case_sensitive else stem.lower()

        if not self._config.rules:
            return ValidationResult.valid(filename, {})

        for rule in self._config.rules:
            compiled = (
                rule._compiled
                if self._config.case_sensitive
                else re.compile(rule.pattern, re.IGNORECASE)
            )
            match = compiled.match(check_stem)
            if match:
                return ValidationResult.valid(filename, match.groupdict())

        # No rule matched
        error_parts = []
        for rule in self._config.rules:
            if rule.error_message:
                error_parts.append(rule.error_message)
            elif rule.example:
                error_parts.append(_("Expected format like: %s") % rule.example)

        message = (
            "; ".join(error_parts) if error_parts else _("Filename does not match any naming rule.")
        )
        return ValidationResult.invalid(filename, message)


class ISO19650Profile(NamingProfile):
    """ISO 19650 naming profile - wraps the existing ISO19650Validator."""

    def __init__(self, config=None):
        from ..config import ISO19650Config
        from ..iso19650 import ISO19650Validator

        self._validator = ISO19650Validator(config or ISO19650Config())

    @property
    def name(self) -> str:
        return "iso19650"

    @property
    def description(self) -> str:
        return "ISO 19650 BIM naming standard"

    def validate(self, path: str | Path) -> ValidationResult:
        result: ValidationResult = self._validator.validate(path)
        return result


def load_profile(profile_data: dict) -> NamingProfile:
    """Load a naming profile from configuration data.

    Args:
        profile_data: Profile configuration dictionary with 'type' key.

    Returns:
        NamingProfile instance

    Raises:
        ValueError: If profile type is unknown
    """
    profile_type = profile_data.get("type", "regex")

    if profile_type == "iso19650":
        from ..config import ISO19650Config

        iso_data = {k: v for k, v in profile_data.items() if k != "type"}
        return ISO19650Profile(ISO19650Config(**iso_data) if iso_data else None)
    elif profile_type == "regex":
        config = RegexProfileConfig.from_dict(profile_data)
        return RegexProfile(config)
    else:
        raise ValueError(f"Unknown profile type: {profile_type}")
