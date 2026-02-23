"""Slack notification channel."""

import json
import logging
import urllib.error
import urllib.request

from .base import NotificationChannel

logger = logging.getLogger("filehub.notification.slack")


class SlackNotifier(NotificationChannel):
    """Sends notifications to Slack via Incoming Webhook."""

    @property
    def name(self) -> str:
        """Return channel identifier."""
        return "slack"

    @property
    def is_available(self) -> bool:
        """Return True if webhook_url is configured."""
        return bool(self.webhook_url)

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(
        self,
        message: str,
        title: str = "",
        level: str = "info",
        **context: object,
    ) -> bool:
        del level, context
        if not self.webhook_url:
            logger.warning("Slack webhook URL not configured.")
            return False

        payload = {"text": f"*{title}*\n{message}"}

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url, data=data, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status != 200:
                    logger.error("Failed to send Slack notification: %s", response.read().decode())
                    return False
            return True
        except TimeoutError:
            logger.error("Slack notification timed out")
            return False
        except urllib.error.URLError as e:
            logger.error("Slack notification error: %s", e)
            return False
