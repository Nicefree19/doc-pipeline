"""Unit tests for system tray."""

from unittest.mock import MagicMock, patch

import pytest
from filehub.ui.tray import SystemTray, TrayState


@pytest.fixture
def mock_pystray():
    with (
        patch("filehub.ui.tray.TRAY_AVAILABLE", True),
        patch("filehub.ui.tray.pystray") as mock,
    ):
        yield mock


def test_tray_init():
    with patch("filehub.ui.tray.TRAY_AVAILABLE", True):
        tray = SystemTray(app_name="TestApp")
        assert tray.state == TrayState.ACTIVE
        assert tray.is_available is True


def test_tray_start_success(mock_pystray):
    tray = SystemTray()

    with patch.object(tray, "_create_icon"), patch.object(tray, "_create_menu"):
        assert tray.start() is True
        mock_pystray.Icon.assert_called_once()


def test_tray_stop(mock_pystray):
    tray = SystemTray()
    tray._icon = mock_pystray.Icon.return_value

    tray.stop()
    mock_pystray.Icon.return_value.stop.assert_called_once()


def test_toggle_pause():
    on_pause = MagicMock()
    on_resume = MagicMock()
    tray = SystemTray(on_pause=on_pause, on_resume=on_resume)

    # Initial state
    assert tray.state == TrayState.ACTIVE

    # Pause
    tray._toggle_pause()
    assert tray.state == TrayState.PAUSED
    on_pause.assert_called_once()

    # Resume
    tray._toggle_pause()
    assert tray.state == TrayState.ACTIVE
    on_resume.assert_called_once()
