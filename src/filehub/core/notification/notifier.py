"""Toast notifier with Explorer click callback."""

from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path

from ...i18n import _
from .channels.base import NotificationChannel
from .channels.slack import SlackNotifier
from .channels.teams import TeamsNotifier

logger = logging.getLogger("filehub")

# win10toast-click import
try:
    from win10toast_click import ToastNotifier as Win10Toast

    TOAST_AVAILABLE = True
except ImportError:
    Win10Toast = None
    TOAST_AVAILABLE = False
    logger.warning("win10toast-click unavailable")


def open_in_explorer(path: Path, select: bool = True) -> None:
    """Open file in Windows Explorer."""
    try:
        if select:
            subprocess.run(["explorer", "/select,", str(path)], check=False)
        else:
            subprocess.run(["explorer", str(path.parent)], check=False)
    except Exception as e:
        logger.error("Failed to open in Explorer: %s", e)


class Notifier:
    """Notification manager supporting multiple channels."""

    def __init__(
        self,
        title: str = "FileHub",
        duration: int = 10,
        enabled: bool = True,
        slack_webhook: str | None = None,
        teams_webhook: str | None = None,
    ):
        self._title = title
        self._duration = duration
        self._enabled = enabled and TOAST_AVAILABLE
        self._lock = threading.Lock()

        self.channels: list[NotificationChannel] = []

        # Windows Toast
        if self._enabled:
            self._toaster = Win10Toast()
        else:
            self._toaster = None

        # Slack
        if slack_webhook:
            self.channels.append(SlackNotifier(slack_webhook))

        # Teams
        if teams_webhook:
            self.channels.append(TeamsNotifier(teams_webhook))

    def notify(self, message: str, file_path: Path | None = None, title: str | None = None) -> bool:
        """Show toast notification and send to channels.

        Args:
            message: Notification message
            file_path: File path to open on click
            title: Notification title (uses default if None)

        Returns:
            Success status
        """
        success = True
        final_title = title or self._title

        # 1. Send to Windows Toast (Core)
        if self._enabled and self._toaster:
            with self._lock:
                try:
                    callback = None
                    if file_path:

                        def make_cb(p: Path):
                            def on_click():
                                open_in_explorer(p, select=True)

                            return on_click

                        callback = make_cb(file_path)

                    self._toaster.show_toast(
                        title=final_title,
                        msg=message,
                        duration=self._duration,
                        threaded=True,
                        callback_on_click=callback,
                    )
                    logger.debug("Toast notification shown: %s", message[:50])
                except Exception as e:
                    logger.exception("Toast notification failed: %s", e)
                    success = False

        # 2. Send to other channels (async to avoid blocking)
        if self.channels:
            dispatch_title = final_title
            dispatch_msg = message

            def _send_channels() -> None:
                for ch in self.channels:
                    try:
                        ch.send(
                            dispatch_msg,
                            title=dispatch_title,
                            level="info",
                        )
                    except Exception as e:
                        logger.error(
                            "Channel notification failed: %s",
                            e,
                        )

            threading.Thread(
                target=_send_channels,
                daemon=True,
                name="NotifyChannels",
            ).start()

        return success

    def notify_validation_error(
        self, filename: str, reason: str, file_path: Path | None = None
    ) -> bool:
        """Notify validation failure."""
        if len(reason) > 200:
            reason = reason[:197] + "..."

        msg = f"{filename}\n\n{reason}"
        return self.notify(
            message=msg,
            file_path=file_path,
            title=f"{self._title} - {_('Naming Convention Violation')}",
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value and TOAST_AVAILABLE
