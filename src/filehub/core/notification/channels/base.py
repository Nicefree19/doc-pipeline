"""Notification channel interface."""

from abc import ABC, abstractmethod


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return channel identifier."""
        pass

    @abstractmethod
    def send(
        self,
        message: str,
        title: str = "",
        level: str = "info",
        **context: object,
    ) -> bool:
        """Send a notification."""
        pass

    def close(self) -> None:  # noqa: B027
        """Close the channel (optional cleanup)."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the channel is available for use."""
        pass
