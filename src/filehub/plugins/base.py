"""Plugin base class for FileHub."""

from __future__ import annotations

import abc
from pathlib import Path


class PluginBase(abc.ABC):
    """Abstract base class for FileHub plugins.

    Subclasses must implement the ``name`` property.
    All hook methods have default no-op implementations so plugins
    can override only the hooks they need.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique plugin name."""

    def on_file_ready(  # noqa: B027
        self, path: Path, validation_result: object
    ) -> None:
        """Called when file processing completes.

        Args:
            path: Path to the processed file.
            validation_result: The ValidationResult for the file.
        """

    def on_validation_error(  # noqa: B027
        self, path: Path, message: str
    ) -> None:
        """Called on validation failure.

        Args:
            path: Path to the file that failed validation.
            message: The validation error message.
        """

    def on_startup(self) -> None:  # noqa: B027
        """Called when the application starts."""

    def on_shutdown(self) -> None:  # noqa: B027
        """Called when the application stops."""
