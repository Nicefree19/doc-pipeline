"""Generic profile validator."""

from __future__ import annotations

import re
from pathlib import Path

from ..core.models import ValidationResult
from ..i18n import _
from .iso19650 import ISO19650Validator
from .profiles.models import NamingProfile


class ProfileValidator:
    """Validator based on NamingProfile."""

    def __init__(self, profile: NamingProfile):
        self.profile = profile
        self._iso_validator: ISO19650Validator | None = None

        if profile.type == "iso19650" and profile.iso_config:
            self._iso_validator = ISO19650Validator(profile.iso_config)

    def validate(self, path: str | Path) -> ValidationResult:
        """Validate path against profile."""
        path = Path(path) if isinstance(path, str) else path
        if path.is_dir():
            return ValidationResult.invalid(str(path), _("Cannot validate directory."))

        filename = path.name

        # 1. ISO 19650 Delegation
        if self.profile.type == "iso19650" and self._iso_validator:
            return self._iso_validator.validate(path)

        # 2. Regex Validation
        if self.profile.type == "regex" and self.profile.regex_pattern:
            if not re.match(self.profile.regex_pattern, filename):
                return ValidationResult.invalid(
                    filename,
                    _("Filename does not match required pattern: %s") % self.profile.regex_pattern,
                )
            return ValidationResult.valid(filename, {})

        # 3. Custom Field Validation
        if self.profile.type == "custom":
            return self._validate_custom(path)

        return ValidationResult.invalid(filename, _("Unknown profile type or configuration."))

    def _validate_custom(self, path: Path) -> ValidationResult:
        """Validate custom field-based profile."""
        filename = path.name
        stem = path.stem

        # Split by separator
        parts = stem.split(self.profile.separator)
        fields = self.profile.fields

        if len(parts) != len(fields):
            return ValidationResult.invalid(
                filename,
                _("Incorrect number of fields. Expected %d, got %d.") % (len(fields), len(parts)),
            )

        values = {}
        for i, field_def in enumerate(fields):
            value = parts[i]
            values[field_def.name] = value

            # Check allowed values
            if field_def.allowed and value not in field_def.allowed:
                return ValidationResult.invalid(
                    filename,
                    _("Field '%s' has invalid value '%s'. Allowed: %s")
                    % (field_def.name, value, field_def.allowed),
                )

            # Check pattern
            if field_def.pattern:
                if not re.match(field_def.pattern, value):
                    return ValidationResult.invalid(
                        filename, _("Field '%s' format invalid.") % field_def.name
                    )

        return ValidationResult.valid(filename, values)
