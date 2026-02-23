"""Channel manager for coordinating multiple notification channels."""

from __future__ import annotations

import logging

from .base import NotificationChannel

logger = logging.getLogger("filehub.channels.manager")


def _channel_name(channel: NotificationChannel) -> str:
    """Return channel name with a safe fallback for malformed channel objects."""
    name = getattr(channel, "name", "")
    if isinstance(name, str) and name:
        return name
    return channel.__class__.__name__.lower()


class ChannelManager:
    """Manages a collection of notification channels.

    Allows adding, removing, and broadcasting notifications
    to all registered channels.
    """

    def __init__(self) -> None:
        self._channels: list[NotificationChannel] = []

    def add_channel(self, channel: NotificationChannel) -> None:
        """Register a notification channel.

        Args:
            channel: The channel instance to add.
        """
        self._channels.append(channel)

    def remove_channel(self, name: str) -> bool:
        """Remove a channel by name.

        Args:
            name: The channel name to remove.

        Returns:
            True if a channel was removed, False if not found.
        """
        for i, ch in enumerate(self._channels):
            if _channel_name(ch) == name:
                self._channels.pop(i)
                return True
        return False

    def get_channel(self, name: str) -> NotificationChannel | None:
        """Look up a channel by name.

        Args:
            name: The channel name to find.

        Returns:
            The channel instance, or None if not found.
        """
        for ch in self._channels:
            if _channel_name(ch) == name:
                return ch
        return None

    def list_channels(self) -> list[str]:
        """Return the names of all registered channels."""
        return [_channel_name(ch) for ch in self._channels]

    def send_all(
        self,
        message: str,
        title: str = "",
        level: str = "info",
        **context: object,
    ) -> dict[str, bool]:
        """Send a notification to every registered channel.

        Each channel is called independently; an exception in one
        channel does not prevent delivery to the others.

        Args:
            message: The notification message body.
            title: Optional notification title.
            level: Severity level (info, warning, error).
            **context: Additional context forwarded to each channel.

        Returns:
            A dict mapping channel name to send success (True/False).
        """
        results: dict[str, bool] = {}
        for ch in self._channels:
            channel_name = _channel_name(ch)
            try:
                results[channel_name] = bool(ch.send(
                    message, title=title, level=level, **context,
                ))
            except Exception:
                logger.exception(
                    "Channel '%s' raised an exception during send", channel_name,
                )
                results[channel_name] = False
        return results

    def close_all(self) -> None:
        """Call close() on every registered channel."""
        for ch in self._channels:
            channel_name = _channel_name(ch)
            try:
                ch.close()
            except Exception:
                logger.exception(
                    "Channel '%s' raised an exception during close", channel_name,
                )
