"""ISO 19650 file naming validator.

Format: [Project]-[Originator]-[Volume]-[Level]-[Type]-[Role]-[Number].ext
Separators: Both `-` and `_` are allowed.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..core.models import ValidationResult
from ..i18n import _
from .config import ISO19650Config


class ISO19650Validator:
    """ISO 19650 naming convention validator.

    Validates filenames against the ISO 19650 BIM naming standard.
    Supports configurable allowed codes for each field.
    """

    FIELDS = ["project", "originator", "volume", "level", "type", "role", "number"]

    def __init__(self, config: ISO19650Config | None = None):
        """Initialize validator.

        Args:
            config: Optional configuration with allowed codes.
                    If None, only format validation is performed.
        """
        self._config = config or ISO19650Config()
        pattern = self._config.number_pattern or r"^\d{4}$"
        self._number_re = re.compile(pattern)

    def validate(self, path: str | Path) -> ValidationResult:
        """Validate a file path or filename.

        Args:
            path: File path or filename to validate

        Returns:
            ValidationResult with validation status and details
        """
        path = Path(path) if isinstance(path, str) else path
        if path.is_dir():
            return ValidationResult.invalid(str(path), _("Cannot validate directory."))
        return self.validate_filename(path.name)

    def validate_filename(self, filename: str) -> ValidationResult:
        """Validate a filename against ISO 19650 format.

        Args:
            filename: Filename to validate (with extension)

        Returns:
            ValidationResult with validation status and details
        """
        if not filename:
            return ValidationResult.invalid(filename, _("Filename is empty."))

        stem = Path(filename).stem
        parts = re.split(r"[-_]", stem)

        if len(parts) != len(self.FIELDS):
            return self._length_error(parts)

        values = dict(zip(self.FIELDS, parts, strict=False))

        # Check for missing fields (empty strings)
        for field in self.FIELDS:
            if not values.get(field):
                return ValidationResult.invalid(filename, self._missing_message(field))

        # Validate individual fields
        for field in ["project", "originator", "volume", "level", "type", "role"]:
            allowed = getattr(self._config, field)
            if allowed and values[field] not in allowed:
                return ValidationResult.invalid(filename, self._invalid_message(field, allowed))

        # Validate number pattern
        number = values["number"]
        if not self._number_re.match(number):
            return ValidationResult.invalid(
                filename, _("Number format is incorrect. Example: %s") % self._example_number()
            )

        return ValidationResult.valid(filename, values)

    def _length_error(self, parts: list[str]) -> ValidationResult:
        """Generate error for incorrect field count."""
        if len(parts) < len(self.FIELDS):
            missing = self.FIELDS[len(parts)]
            return ValidationResult.invalid("-".join(parts), self._missing_message(missing))
        return ValidationResult.invalid(
            "-".join(parts),
            _("Too many fields. Format: Project-Originator-Volume-Level-Type-Role-Number"),
        )

    def _missing_message(self, field: str) -> str:
        """Generate missing field message."""
        if field == "type":
            return self._type_help_message(missing=True)
        if field == "originator":
            return self._invalid_message("originator", self._config.originator, missing=True)
        return _("%s code is missing.") % field.capitalize()

    def _invalid_message(self, field: str, allowed: list[str], missing: bool = False) -> str:
        """Generate invalid field message."""
        if field == "type":
            return self._type_help_message(missing=missing)

        if field == "originator" and allowed:
            codes = ", ".join(allowed)
            if missing:
                return _("Originator code is missing. Our company code is [%s].") % codes
            return _("Originator code is incorrect. Our company code is [%s].") % codes

        if allowed:
            codes = ", ".join(allowed)
            if missing:
                return _("%s code is missing. Allowed codes are [%s].") % (
                    field.capitalize(),
                    codes,
                )
            return _("%s code is invalid. Allowed codes are [%s].") % (field.capitalize(), codes)

        return _("%s code is invalid.") % field.capitalize()

    def _type_help_message(self, missing: bool = False) -> str:
        """Generate type field help message."""
        if self._config.type_help:
            items = [f"{label}: [{code}]" for code, label in self._config.type_help.items()]
            guide = ", ".join(items)
            if missing:
                return _("Type code is missing. Use %s.") % guide
            return _("Type code is incorrect. Use %s.") % guide

        if self._config.type:
            codes = ", ".join(self._config.type)
            if missing:
                return _("%s code is missing. Allowed codes are [%s].") % ("Type", codes)
            return _("%s code is invalid. Allowed codes are [%s].") % ("Type", codes)

        return _("%s code is invalid.") % "Type"

    def _example_number(self) -> str:
        """Return example number for error messages."""
        return "0001"
