"""Naming convention configuration schemas."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ISO19650Config:
    """ISO 19650 naming convention configuration.

    Attributes:
        project: Allowed project codes
        originator: Allowed originator codes (company codes)
        volume: Allowed volume/zone codes
        level: Allowed level/floor codes
        type: Allowed document type codes
        role: Allowed discipline/role codes
        number_pattern: Regex pattern for number field (default: 4 digits)
        type_help: Human-readable labels for type codes
    """

    project: list[str] = field(default_factory=list)
    originator: list[str] = field(default_factory=list)
    volume: list[str] = field(default_factory=list)
    level: list[str] = field(default_factory=list)
    type: list[str] = field(default_factory=list)
    role: list[str] = field(default_factory=list)
    number_pattern: str = r"^\d{4}$"
    type_help: dict[str, str] = field(default_factory=dict)


@dataclass
class NamingConfig:
    """Root naming configuration.

    Supports multiple naming standards.
    """

    iso19650: ISO19650Config | None = None
    profiles: dict[str, dict] = field(default_factory=dict)
    enabled: bool = True
    active_profile: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> NamingConfig:
        """Create config from dictionary."""
        iso_data = data.get("iso19650", {})
        if iso_data:
            allowed = ISO19650Config.__dataclass_fields__.keys()
            sanitized = {k: v for k, v in iso_data.items() if k in allowed}
            iso_config = ISO19650Config(**sanitized) if sanitized else None
        else:
            iso_config = None

        return cls(
            iso19650=iso_config,
            profiles=data.get("profiles", {}),
            enabled=data.get("enabled", True),
            active_profile=data.get("active_profile"),
        )
