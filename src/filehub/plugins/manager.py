"""Plugin manager for FileHub."""

from __future__ import annotations

import logging
from pathlib import Path

from .base import PluginBase

logger = logging.getLogger("filehub")


class PluginManager:
    """Manages plugin registration, lookup, and lifecycle notifications.

    Each notify method catches exceptions per-plugin and logs them
    so that a misbehaving plugin never crashes the host application.
    """

    def __init__(self) -> None:
        self._plugins: list[PluginBase] = []

    def register(self, plugin: PluginBase) -> None:
        """Register a plugin.

        If a plugin with the same name already exists it is replaced.

        Args:
            plugin: The plugin instance to register.
        """
        # Replace existing plugin with same name
        self._plugins = [p for p in self._plugins if p.name != plugin.name]
        self._plugins.append(plugin)
        logger.info("Plugin registered: %s", plugin.name)

    def unregister(self, name: str) -> bool:
        """Remove a plugin by name.

        Args:
            name: The plugin name to remove.

        Returns:
            True if the plugin was found and removed, False otherwise.
        """
        before = len(self._plugins)
        self._plugins = [p for p in self._plugins if p.name != name]
        removed = len(self._plugins) < before
        if removed:
            logger.info("Plugin unregistered: %s", name)
        return removed

    def get_plugin(self, name: str) -> PluginBase | None:
        """Look up a plugin by name.

        Args:
            name: The plugin name to find.

        Returns:
            The plugin instance, or None if not found.
        """
        for plugin in self._plugins:
            if plugin.name == name:
                return plugin
        return None

    def list_plugins(self) -> list[str]:
        """Return a list of registered plugin names."""
        return [p.name for p in self._plugins]

    # ------------------------------------------------------------------
    # Notification helpers
    # ------------------------------------------------------------------

    def notify_file_ready(
        self, path: Path, validation_result: object
    ) -> None:
        """Notify all plugins that a file has been processed.

        Args:
            path: Path to the processed file.
            validation_result: The ValidationResult for the file.
        """
        for plugin in self._plugins:
            try:
                plugin.on_file_ready(path, validation_result)
            except Exception:
                logger.exception(
                    "Plugin '%s' raised in on_file_ready", plugin.name
                )

    def notify_validation_error(self, path: Path, message: str) -> None:
        """Notify all plugins of a validation failure.

        Args:
            path: Path to the file that failed validation.
            message: The validation error message.
        """
        for plugin in self._plugins:
            try:
                plugin.on_validation_error(path, message)
            except Exception:
                logger.exception(
                    "Plugin '%s' raised in on_validation_error",
                    plugin.name,
                )

    def notify_startup(self) -> None:
        """Notify all plugins that the application is starting."""
        for plugin in self._plugins:
            try:
                plugin.on_startup()
            except Exception:
                logger.exception(
                    "Plugin '%s' raised in on_startup", plugin.name
                )

    def notify_shutdown(self) -> None:
        """Notify all plugins that the application is shutting down."""
        for plugin in self._plugins:
            try:
                plugin.on_shutdown()
            except Exception:
                logger.exception(
                    "Plugin '%s' raised in on_shutdown", plugin.name
                )
