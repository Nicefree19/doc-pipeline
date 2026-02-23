"""Logging notification channel."""

from __future__ import annotations

import logging

from .base import NotificationChannel

logger = logging.getLogger("filehub.channels.log")

_LEVEL_MAP: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class LogChannel(NotificationChannel):
    """Notification channel that logs messages via Python logging."""

    @property
    def name(self) -> str:
        """Return channel identifier."""
        return "log"

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
        """Log a notification at the appropriate logging level.

        Maps the level string to a Python logging level. Unknown levels
        default to INFO.

        Args:
            message: The notification message body.
            title: Optional notification title (prepended if present).
            level: Severity level (info, warning, error, debug, critical).
            **context: Additional context (unused).

        Returns:
            True always.
        """
        log_level = _LEVEL_MAP.get(level.lower(), logging.INFO)
        if title:
            log_message = f"[{title}] {message}"
        else:
            log_message = message
        logger.log(log_level, log_message)
        return True
