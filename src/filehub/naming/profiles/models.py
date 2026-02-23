"""Naming profile models."""

from dataclasses import dataclass, field
from typing import Any, Literal

from ..config import ISO19650Config


@dataclass
class NamingField:
    """Definition of a single naming field."""

    name: str
    description: str = ""
    allowed: list[str] = field(default_factory=list)
    pattern: str | None = None
    required: bool = True
    error_msg: str | None = None


@dataclass
class NamingProfile:
    """Naming convention profile."""

    name: str
    type: Literal["iso19650", "custom", "regex"] = "custom"
    description: str = ""
    separator: str = "-"
    fields: list[NamingField] = field(default_factory=list)
    regex_pattern: str | None = None

    # Legacy support
    iso_config: ISO19650Config | None = None

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "NamingProfile":
        """Create profile from dictionary."""
        profile_type = data.get("type", "custom")

        if profile_type == "iso19650":
            from ..config import ISO19650Config

            config_data = data.get("config", {})
            allowed = ISO19650Config.__dataclass_fields__.keys()
            sanitized = {k: v for k, v in config_data.items() if k in allowed}

            return cls(
                name=name,
                type="iso19650",
                iso_config=ISO19650Config(**sanitized),
            )

        fields_data = data.get("fields", [])
        fields = [NamingField(**f) for f in fields_data]

        return cls(
            name=name,
            type=profile_type,
            description=data.get("description", ""),
            separator=data.get("separator", "-"),
            fields=fields,
            regex_pattern=data.get("regex_pattern"),
        )
