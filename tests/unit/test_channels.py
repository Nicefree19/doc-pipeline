"""Tests for notification channels system."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.filehub.core.notification.channels import (
    ChannelManager,
    ConsoleChannel,
    LogChannel,
    NotificationChannel,
)
from src.filehub.core.notification.channels.slack import SlackNotifier
from src.filehub.core.notification.channels.teams import TeamsNotifier


class TestNotificationChannelABC:
    """Test that NotificationChannel cannot be instantiated directly."""

    def test_cannot_instantiate_abc(self):
        """NotificationChannel is abstract and cannot be created."""
        with pytest.raises(TypeError):
            NotificationChannel()  # type: ignore[abstract]


class TestConsoleChannel:
    """Test ConsoleChannel behaviour."""

    def test_name_returns_console(self):
        """Channel name must be 'console'."""
        channel = ConsoleChannel()
        assert channel.name == "console"

    def test_is_available_returns_true(self):
        """Console channel is always available."""
        channel = ConsoleChannel()
        assert channel.is_available is True

    def test_send_returns_true(self):
        """send() always returns True."""
        channel = ConsoleChannel()
        result = channel.send("hello")
        assert result is True

    def test_send_outputs_to_stdout(self, capsys):
        """send() prints a formatted line to stdout."""
        channel = ConsoleChannel()
        channel.send("file saved", title="FileHub", level="info")
        captured = capsys.readouterr()
        assert "[INFO] [FileHub] file saved" in captured.out

    def test_send_without_title(self, capsys):
        """send() omits title brackets when title is empty."""
        channel = ConsoleChannel()
        channel.send("plain message", level="warning")
        captured = capsys.readouterr()
        assert "[WARNING] plain message" in captured.out

    def test_send_level_uppercased(self, capsys):
        """Level string is uppercased in output."""
        channel = ConsoleChannel()
        channel.send("msg", level="error")
        captured = capsys.readouterr()
        assert "[ERROR]" in captured.out


class TestLogChannel:
    """Test LogChannel behaviour."""

    def test_name_returns_log(self):
        """Channel name must be 'log'."""
        channel = LogChannel()
        assert channel.name == "log"

    def test_is_available_returns_true(self):
        """Log channel is always available."""
        channel = LogChannel()
        assert channel.is_available is True

    def test_send_returns_true(self):
        """send() always returns True."""
        channel = LogChannel()
        result = channel.send("test message")
        assert result is True

    def test_send_logs_at_info_level(self, caplog):
        """send() with level='info' logs at INFO."""
        channel = LogChannel()
        with caplog.at_level(logging.DEBUG, logger="filehub.channels.log"):
            channel.send("info msg", level="info")
        assert any(
            r.levelno == logging.INFO and "info msg" in r.message
            for r in caplog.records
        )

    def test_send_logs_at_warning_level(self, caplog):
        """send() with level='warning' logs at WARNING."""
        channel = LogChannel()
        with caplog.at_level(logging.DEBUG, logger="filehub.channels.log"):
            channel.send("warn msg", level="warning")
        assert any(
            r.levelno == logging.WARNING and "warn msg" in r.message
            for r in caplog.records
        )

    def test_send_logs_at_error_level(self, caplog):
        """send() with level='error' logs at ERROR."""
        channel = LogChannel()
        with caplog.at_level(logging.DEBUG, logger="filehub.channels.log"):
            channel.send("err msg", level="error")
        assert any(
            r.levelno == logging.ERROR and "err msg" in r.message
            for r in caplog.records
        )

    def test_send_unknown_level_defaults_to_info(self, caplog):
        """Unknown level strings default to INFO."""
        channel = LogChannel()
        with caplog.at_level(logging.DEBUG, logger="filehub.channels.log"):
            channel.send("unknown level msg", level="banana")
        assert any(
            r.levelno == logging.INFO and "unknown level msg" in r.message
            for r in caplog.records
        )

    def test_send_with_title_prepends_title(self, caplog):
        """Title is prepended to the log message."""
        channel = LogChannel()
        with caplog.at_level(logging.DEBUG, logger="filehub.channels.log"):
            channel.send("body", title="MyTitle", level="info")
        assert any("[MyTitle] body" in r.message for r in caplog.records)


class TestChannelManager:
    """Test ChannelManager orchestration."""

    def test_add_and_list_channels(self):
        """add_channel registers channels visible via list_channels."""
        mgr = ChannelManager()
        mgr.add_channel(ConsoleChannel())
        mgr.add_channel(LogChannel())
        assert sorted(mgr.list_channels()) == ["console", "log"]

    def test_remove_channel_returns_true(self):
        """remove_channel returns True when the channel exists."""
        mgr = ChannelManager()
        mgr.add_channel(ConsoleChannel())
        assert mgr.remove_channel("console") is True
        assert mgr.list_channels() == []

    def test_remove_channel_returns_false_when_missing(self):
        """remove_channel returns False for an unknown name."""
        mgr = ChannelManager()
        assert mgr.remove_channel("nonexistent") is False

    def test_get_channel_found(self):
        """get_channel returns the channel when it exists."""
        mgr = ChannelManager()
        console = ConsoleChannel()
        mgr.add_channel(console)
        assert mgr.get_channel("console") is console

    def test_get_channel_not_found(self):
        """get_channel returns None for an unknown name."""
        mgr = ChannelManager()
        assert mgr.get_channel("missing") is None

    def test_send_all_calls_all_channels(self):
        """send_all dispatches to every registered channel."""
        ch1 = MagicMock(spec=NotificationChannel)
        ch1.name = "mock1"
        ch1.send.return_value = True

        ch2 = MagicMock(spec=NotificationChannel)
        ch2.name = "mock2"
        ch2.send.return_value = True

        mgr = ChannelManager()
        mgr.add_channel(ch1)
        mgr.add_channel(ch2)

        results = mgr.send_all("msg", title="T", level="info")

        ch1.send.assert_called_once_with("msg", title="T", level="info")
        ch2.send.assert_called_once_with("msg", title="T", level="info")
        assert results == {"mock1": True, "mock2": True}

    def test_send_all_returns_per_channel_results(self):
        """send_all returns a dict mapping channel name to success."""
        ch_ok = MagicMock(spec=NotificationChannel)
        ch_ok.name = "ok"
        ch_ok.send.return_value = True

        ch_fail = MagicMock(spec=NotificationChannel)
        ch_fail.name = "fail"
        ch_fail.send.return_value = False

        mgr = ChannelManager()
        mgr.add_channel(ch_ok)
        mgr.add_channel(ch_fail)

        results = mgr.send_all("msg")
        assert results == {"ok": True, "fail": False}

    def test_send_all_handles_channel_exception(self):
        """An exception in one channel does not block others."""
        ch_bad = MagicMock(spec=NotificationChannel)
        ch_bad.name = "bad"
        ch_bad.send.side_effect = RuntimeError("boom")

        ch_good = MagicMock(spec=NotificationChannel)
        ch_good.name = "good"
        ch_good.send.return_value = True

        mgr = ChannelManager()
        mgr.add_channel(ch_bad)
        mgr.add_channel(ch_good)

        results = mgr.send_all("msg")
        assert results["bad"] is False
        assert results["good"] is True

    def test_close_all_calls_close_on_all(self):
        """close_all invokes close() on every channel."""
        ch1 = MagicMock(spec=NotificationChannel)
        ch1.name = "a"
        ch2 = MagicMock(spec=NotificationChannel)
        ch2.name = "b"

        mgr = ChannelManager()
        mgr.add_channel(ch1)
        mgr.add_channel(ch2)

        mgr.close_all()
        ch1.close.assert_called_once()
        ch2.close.assert_called_once()

    def test_send_all_handles_missing_name_on_broken_channel(self):
        """Broken channel objects without name must not crash send_all."""

        class BrokenChannel:
            def send(self, message: str, **kwargs: object) -> bool:
                raise RuntimeError("boom")

            def close(self) -> None:
                pass

        mgr = ChannelManager()
        mgr.add_channel(BrokenChannel())  # type: ignore[arg-type]

        results = mgr.send_all("msg")
        assert results == {"brokenchannel": False}


class TestWebhookChannels:
    """Tests for Slack/Teams webhook channels."""

    @staticmethod
    def _mock_urlopen_with_status(status: int):
        response = MagicMock()
        response.__enter__.return_value.status = status
        response.__enter__.return_value.read.return_value = b"body"
        return response

    def test_slack_channel_basics(self):
        channel = SlackNotifier("https://example.com/slack")
        assert channel.name == "slack"
        assert channel.is_available is True

    def test_teams_channel_basics(self):
        channel = TeamsNotifier("https://example.com/teams")
        assert channel.name == "teams"
        assert channel.is_available is True

    def test_slack_send_success(self):
        with patch(
            "src.filehub.core.notification.channels.slack.urllib.request.urlopen",
            return_value=self._mock_urlopen_with_status(200),
        ) as mock_urlopen:
            channel = SlackNotifier("https://example.com/slack")
            assert channel.send("message", title="title") is True
            mock_urlopen.assert_called_once()

    def test_teams_send_success(self):
        with patch(
            "src.filehub.core.notification.channels.teams.urllib.request.urlopen",
            return_value=self._mock_urlopen_with_status(200),
        ) as mock_urlopen:
            channel = TeamsNotifier("https://example.com/teams")
            assert channel.send("message", title="title") is True
            mock_urlopen.assert_called_once()

    def test_manager_send_all_with_webhook_channels(self):
        with patch(
            "src.filehub.core.notification.channels.slack.urllib.request.urlopen",
            return_value=self._mock_urlopen_with_status(200),
        ) as mock_urlopen:
            mgr = ChannelManager()
            mgr.add_channel(SlackNotifier("https://example.com/slack"))
            mgr.add_channel(TeamsNotifier("https://example.com/teams"))

            results = mgr.send_all("msg", title="T", level="info")
            assert results == {"slack": True, "teams": True}
            assert mock_urlopen.call_count == 2
