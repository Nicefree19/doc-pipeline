"""Unit tests for notifier."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from filehub.core.notification.notifier import Notifier


@pytest.fixture
def mock_toast():
    with (
        patch("filehub.core.notification.notifier.TOAST_AVAILABLE", True),
        patch("filehub.core.notification.notifier.Win10Toast") as mock,
    ):
        yield mock


def test_notifier_init(mock_toast):
    notifier = Notifier(title="Test", duration=5)
    assert notifier.enabled
    mock_toast.assert_called_once()


def test_notifier_disabled_init(mock_toast):
    notifier = Notifier(enabled=False)
    assert not notifier.enabled
    mock_toast.assert_not_called()


def test_notifier_with_webhooks(mock_toast):
    """Test notifier initialization with webhook URLs."""
    with (
        patch("filehub.core.notification.notifier.SlackNotifier") as mock_slack,
        patch("filehub.core.notification.notifier.TeamsNotifier") as mock_teams,
    ):

        notifier = Notifier(slack_webhook="https://slack.com", teams_webhook="https://teams.com")

        assert len(notifier.channels) == 2
        mock_slack.assert_called_once_with("https://slack.com")
        mock_teams.assert_called_once_with("https://teams.com")


def test_notify_sends_message_first_to_channels(mock_toast):
    """Notifier should call channel.send(message, title=..., level=...)."""
    slack_channel = MagicMock()
    teams_channel = MagicMock()

    with (
        patch("filehub.core.notification.notifier.SlackNotifier", return_value=slack_channel),
        patch("filehub.core.notification.notifier.TeamsNotifier", return_value=teams_channel),
    ):
        notifier = Notifier(slack_webhook="https://slack.com", teams_webhook="https://teams.com")
        notifier.notify("hello", title="FileHub")

    slack_channel.send.assert_called_once_with("hello", title="FileHub", level="info")
    teams_channel.send.assert_called_once_with("hello", title="FileHub", level="info")


def test_notify(mock_toast):
    notifier = Notifier()
    instance = mock_toast.return_value

    result = notifier.notify("Test message")

    assert result is True
    instance.show_toast.assert_called_once()
    args, kwargs = instance.show_toast.call_args
    assert kwargs["msg"] == "Test message"
    assert kwargs["title"] == "FileHub"


def test_notify_disabled(mock_toast):
    notifier = Notifier(enabled=False)
    result = notifier.notify("Test message")

    assert result is True
    mock_toast.return_value.show_toast.assert_not_called()


def test_notify_file_path_callback(mock_toast):
    notifier = Notifier()
    instance = mock_toast.return_value
    file_path = Path("/tmp/test.txt")

    result = notifier.notify("Test", file_path=file_path)

    assert result is True
    args, kwargs = instance.show_toast.call_args
    assert kwargs["callback_on_click"] is not None

    # Verify callback calls open_in_explorer
    callback = kwargs["callback_on_click"]
    with patch("filehub.core.notification.notifier.open_in_explorer") as mock_open:
        callback()
        mock_open.assert_called_once_with(file_path, select=True)


def test_notify_validation_error(mock_toast):
    notifier = Notifier()
    instance = mock_toast.return_value

    notifier.notify_validation_error("bad_file.txt", "Invalid format")

    instance.show_toast.assert_called_once()
    args, kwargs = instance.show_toast.call_args
    assert "bad_file.txt" in kwargs["msg"]
    assert "Invalid format" in kwargs["msg"]
    assert "Naming Convention Violation" in kwargs["title"]
