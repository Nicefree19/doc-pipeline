"""Microsoft Teams notification channel."""

import json
import logging
import urllib.error
import urllib.request

from .base import NotificationChannel

logger = logging.getLogger("filehub.notification.teams")


class TeamsNotifier(NotificationChannel):
    """Sends notifications to Microsoft Teams via Incoming Webhook."""

    @property
    def name(self) -> str:
        """Return channel identifier."""
        return "teams"

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
            logger.warning("Teams webhook URL not configured.")
            return False

        # Teams adaptive card payload (simplified)
        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": "0076D7",
            "summary": title,
            "sections": [{"activityTitle": title, "text": message}],
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url, data=data, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status != 200:
                    logger.error("Failed to send Teams notification: %s", response.read().decode())
                    return False
            return True
        except TimeoutError:
            logger.error("Teams notification timed out")
            return False
        except urllib.error.URLError as e:
            logger.error("Teams notification error: %s", e)
            return False
