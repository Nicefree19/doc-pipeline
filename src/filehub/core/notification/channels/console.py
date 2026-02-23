"""Console notification channel."""

from __future__ import annotations

import sys

from .base import NotificationChannel


class ConsoleChannel(NotificationChannel):
    """Notification channel that prints formatted messages to stdout."""

    @property
    def name(self) -> str:
        """Return channel identifier."""
        return "console"

    @property
    def is_available(self) -> bool:
        """Return True always."""
        return True

    def send(
        self,
        message: str,
        title: str = "",
        level: str = "info",
        **context: object,
    ) -> bool:
        """Print a formatted notification to stdout.

        Format: [LEVEL] [title] message
        When title is empty: [LEVEL] message

        Args:
            message: The notification message body.
            title: Optional notification title.
            level: Severity level (info, warning, error).
            **context: Additional context (unused).

        Returns:
            True always.
        """
        level_tag = f"[{level.upper()}]"
        if title:
            line = f"{level_tag} [{title}] {message}"
        else:
            line = f"{level_tag} {message}"
        print(line, file=sys.stdout)  # noqa: T201
        return True
