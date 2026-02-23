"""System Tray UI for FileHub.

Provides system tray icon with menu for controlling the application.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from enum import Enum
from pathlib import Path

from ..i18n import _

logger = logging.getLogger(__name__)

# Try to import pystray
try:
    import pystray
    from PIL import Image

    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False
    pystray = None
    Image = None  # type: ignore[assignment]


class TrayState(Enum):
    """System tray icon states."""

    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"


class SystemTray:
    """System tray icon and menu controller.

    Provides a system tray icon with menu items for:
    - Pause/Resume monitoring
    - Open configuration
    - Open log folder
    - Exit application
    """

    def __init__(
        self,
        app_name: str = "FileHub",
        icon_path: Path | None = None,
        on_pause: Callable[[], None] | None = None,
        on_resume: Callable[[], None] | None = None,
        on_exit: Callable[[], None] | None = None,
        on_open_config: Callable[[], None] | None = None,
    ):
        """Initialize system tray.

        Args:
            app_name: Application name shown in tooltip
            icon_path: Path to icon file (PNG, ICO)
            on_pause: Callback when pause is clicked
            on_resume: Callback when resume is clicked
            on_exit: Callback when exit is clicked
            on_open_config: Callback when open config is clicked
        """
        self._app_name = app_name
        self._icon_path = icon_path
        self._on_pause = on_pause
        self._on_resume = on_resume
        self._on_exit = on_exit
        self._on_open_config = on_open_config

        self._state = TrayState.ACTIVE
        self._is_paused = False
        self._icon: pystray.Icon | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    @property
    def is_available(self) -> bool:
        """Check if system tray is available."""
        return TRAY_AVAILABLE

    @property
    def state(self) -> TrayState:
        """Get current tray state."""
        return self._state

    @state.setter
    def state(self, value: TrayState) -> None:
        """Set tray state and update icon."""
        self._state = value
        self._update_icon()

    def start(self) -> bool:
        """Start system tray in background thread."""
        if not TRAY_AVAILABLE:
            logger.warning(_("System tray not available (pystray not installed)"))
            return False

        if self._running:
            return True

        try:
            self._thread = threading.Thread(target=self.run, daemon=True, name="SystemTray")
            self._thread.start()
            return True

        except Exception as e:
            logger.error(_("Failed to start system tray: %s"), e)
            return False

    def run(self) -> None:
        """Run system tray (blocking)."""
        if not TRAY_AVAILABLE:
            return

        self._setup_icon()
        self._running = True
        logger.info(_("System tray started"))

        # Blocks until stop() is called
        if self._icon is not None:
            self._icon.run()

        self._running = False
        logger.info(_("System tray stopped"))

    def stop(self) -> None:
        """Stop system tray."""
        if self._icon:
            self._icon.stop()

    def _setup_icon(self) -> None:
        """Initialize icon instance."""
        if self._icon:
            return

        self._icon = pystray.Icon(
            self._app_name, self._create_icon(), self._app_name, self._create_menu()
        )

    def _create_icon(self) -> Image.Image:
        """Create tray icon image."""
        if self._icon_path and self._icon_path.exists():
            return Image.open(self._icon_path)

        # Create default icon (simple colored square)
        size = 64
        color = {
            TrayState.ACTIVE: (0, 180, 0),  # Green
            TrayState.PAUSED: (255, 165, 0),  # Orange
            TrayState.ERROR: (255, 0, 0),  # Red
        }.get(self._state, (128, 128, 128))

        image = Image.new("RGB", (size, size), color)
        return image

    def _create_menu(self) -> pystray.Menu:
        """Create tray context menu."""
        return pystray.Menu(
            pystray.MenuItem(self._get_pause_text, self._toggle_pause, default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(_("Open Config"), self._handle_open_config),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(_("Exit"), self._handle_exit),
        )

    def _get_pause_text(self, item) -> str:
        """Get dynamic pause/resume text."""
        return _("Resume") if self._is_paused else _("Pause")

    def _toggle_pause(self) -> None:
        """Toggle pause state."""
        if self._is_paused:
            self._is_paused = False
            self.state = TrayState.ACTIVE
            if self._on_resume:
                self._on_resume()
            logger.info(_("Monitoring resumed"))
        else:
            self._is_paused = True
            self.state = TrayState.PAUSED
            if self._on_pause:
                self._on_pause()
            logger.info(_("Monitoring paused"))

        self._update_menu()

    def _handle_open_config(self) -> None:
        """Handle open config menu item."""
        if self._on_open_config:
            self._on_open_config()

    def _handle_exit(self) -> None:
        """Handle exit menu item."""
        logger.info(_("Exit requested from tray"))
        if self._on_exit:
            self._on_exit()
        self.stop()

    def _update_icon(self) -> None:
        """Update tray icon based on state."""
        if self._icon:
            self._icon.icon = self._create_icon()

    def _update_menu(self) -> None:
        """Update tray menu."""
        if self._icon:
            self._icon.update_menu()
