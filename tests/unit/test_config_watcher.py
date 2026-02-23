"""Tests for ConfigWatcher hot-reload module."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from unittest.mock import MagicMock

import yaml

from src.filehub.config.schema import FileHubConfig
from src.filehub.config.watcher import ConfigWatcher


def _write_valid_config(path: Path, debounce: float = 1.0) -> None:
    """Write a valid YAML config to the given path."""
    data = {
        "watcher": {"paths": ["~/Documents"], "recursive": True},
        "pipeline": {"debounce_seconds": debounce},
    }
    path.write_text(yaml.dump(data), encoding="utf-8")


def _write_invalid_yaml(path: Path) -> None:
    """Write invalid YAML to the given path."""
    path.write_text(":\n  invalid: [yaml\n  broken", encoding="utf-8")


class TestConfigWatcherInit:
    """Test ConfigWatcher initialization."""

    def test_init_stores_config_path(self, tmp_path):
        """Test that constructor stores config_path correctly."""
        config_file = tmp_path / "config.yaml"
        _write_valid_config(config_file)
        callback = MagicMock()

        watcher = ConfigWatcher(config_file, on_reload=callback)

        assert watcher._config_path == config_file

    def test_init_stores_callback(self, tmp_path):
        """Test that constructor stores on_reload callback."""
        config_file = tmp_path / "config.yaml"
        _write_valid_config(config_file)
        callback = MagicMock()

        watcher = ConfigWatcher(config_file, on_reload=callback)

        assert watcher._on_reload is callback

    def test_init_default_debounce(self, tmp_path):
        """Test that default debounce is 1.0 seconds."""
        config_file = tmp_path / "config.yaml"
        _write_valid_config(config_file)

        watcher = ConfigWatcher(config_file, on_reload=MagicMock())

        assert watcher._debounce_seconds == 1.0

    def test_init_custom_debounce(self, tmp_path):
        """Test that custom debounce value is stored."""
        config_file = tmp_path / "config.yaml"
        _write_valid_config(config_file)

        watcher = ConfigWatcher(
            config_file, on_reload=MagicMock(), debounce_seconds=2.5
        )

        assert watcher._debounce_seconds == 2.5

    def test_not_running_after_init(self, tmp_path):
        """Test that watcher is not running immediately after creation."""
        config_file = tmp_path / "config.yaml"
        _write_valid_config(config_file)

        watcher = ConfigWatcher(config_file, on_reload=MagicMock())

        assert watcher.is_running is False


class TestConfigWatcherLifecycle:
    """Test ConfigWatcher start/stop lifecycle."""

    def test_start_sets_running(self, tmp_path):
        """Test that start() puts the watcher in running state."""
        config_file = tmp_path / "config.yaml"
        _write_valid_config(config_file)

        watcher = ConfigWatcher(config_file, on_reload=MagicMock())
        try:
            watcher.start()
            assert watcher.is_running is True
        finally:
            watcher.stop()

    def test_stop_clears_running(self, tmp_path):
        """Test that stop() puts the watcher in stopped state."""
        config_file = tmp_path / "config.yaml"
        _write_valid_config(config_file)

        watcher = ConfigWatcher(config_file, on_reload=MagicMock())
        watcher.start()
        watcher.stop()

        assert watcher.is_running is False

    def test_stop_while_not_running_does_not_raise(self, tmp_path):
        """Test that stop() on an idle watcher is safe."""
        config_file = tmp_path / "config.yaml"
        _write_valid_config(config_file)

        watcher = ConfigWatcher(config_file, on_reload=MagicMock())
        # Should not raise
        watcher.stop()

    def test_double_start_is_idempotent(self, tmp_path):
        """Test that calling start() twice does not create duplicate observers."""
        config_file = tmp_path / "config.yaml"
        _write_valid_config(config_file)

        watcher = ConfigWatcher(config_file, on_reload=MagicMock())
        try:
            watcher.start()
            first_observer = watcher._observer
            watcher.start()
            assert watcher._observer is first_observer
            assert watcher.is_running is True
        finally:
            watcher.stop()


class TestConfigWatcherReload:
    """Test ConfigWatcher reload behavior."""

    def test_config_change_triggers_callback(self, tmp_path):
        """Test that modifying the config file triggers on_reload."""
        config_file = tmp_path / "config.yaml"
        _write_valid_config(config_file)
        callback = MagicMock()

        watcher = ConfigWatcher(
            config_file, on_reload=callback, debounce_seconds=0.1
        )
        try:
            watcher.start()
            time.sleep(0.2)

            # Modify the config file
            _write_valid_config(config_file, debounce=5.0)
            # Wait for debounce + processing
            time.sleep(1.0)

            assert callback.call_count >= 1
            args = callback.call_args[0]
            assert isinstance(args[0], FileHubConfig)
        finally:
            watcher.stop()

    def test_invalid_config_does_not_call_callback(self, tmp_path):
        """Test that invalid config change does NOT invoke on_reload."""
        config_file = tmp_path / "config.yaml"
        _write_valid_config(config_file)
        callback = MagicMock()

        watcher = ConfigWatcher(
            config_file, on_reload=callback, debounce_seconds=0.1
        )
        try:
            watcher.start()
            time.sleep(0.2)

            _write_invalid_yaml(config_file)
            time.sleep(1.0)

            callback.assert_not_called()
        finally:
            watcher.stop()

    def test_non_config_file_changes_are_ignored(self, tmp_path):
        """Test that changes to other files do not trigger reload."""
        config_file = tmp_path / "config.yaml"
        _write_valid_config(config_file)
        callback = MagicMock()

        watcher = ConfigWatcher(
            config_file, on_reload=callback, debounce_seconds=0.1
        )
        try:
            watcher.start()
            time.sleep(0.2)

            # Modify a different file in the same directory
            other_file = tmp_path / "other.txt"
            other_file.write_text("not a config", encoding="utf-8")
            time.sleep(1.0)

            callback.assert_not_called()
        finally:
            watcher.stop()

    def test_callback_exception_is_caught_and_logged(self, tmp_path, caplog):
        """Test that an exception in on_reload is caught and logged."""
        config_file = tmp_path / "config.yaml"
        _write_valid_config(config_file)
        callback = MagicMock(side_effect=RuntimeError("callback boom"))

        watcher = ConfigWatcher(
            config_file, on_reload=callback, debounce_seconds=0.1
        )
        try:
            watcher.start()
            time.sleep(0.2)

            with caplog.at_level(logging.ERROR):
                _write_valid_config(config_file, debounce=9.0)
                time.sleep(1.0)

            assert callback.call_count >= 1
            assert any(
                "on_reload callback raised" in record.message
                for record in caplog.records
            )
        finally:
            watcher.stop()

    def test_deleted_config_file_handled_gracefully(self, tmp_path, caplog):
        """Test that deleting the config file does not crash the watcher."""
        config_file = tmp_path / "config.yaml"
        _write_valid_config(config_file)
        callback = MagicMock()

        watcher = ConfigWatcher(
            config_file, on_reload=callback, debounce_seconds=0.1
        )
        try:
            watcher.start()
            time.sleep(0.2)

            with caplog.at_level(logging.WARNING):
                config_file.unlink()
                time.sleep(1.0)

            callback.assert_not_called()
        finally:
            watcher.stop()


class TestConfigWatcherDebounce:
    """Test ConfigWatcher debounce behavior."""

    def test_debounce_prevents_rapid_reloads(self, tmp_path):
        """Test that rapid file changes result in a single reload."""
        config_file = tmp_path / "config.yaml"
        _write_valid_config(config_file)
        callback = MagicMock()

        watcher = ConfigWatcher(
            config_file, on_reload=callback, debounce_seconds=0.5
        )
        try:
            watcher.start()
            time.sleep(0.2)

            # Trigger multiple rapid changes
            for i in range(5):
                _write_valid_config(config_file, debounce=float(i + 1))
                time.sleep(0.05)

            # Wait for debounce to expire
            time.sleep(1.5)

            # Should have been called only once (or at most twice)
            # due to debounce collapsing rapid changes
            assert callback.call_count <= 2
        finally:
            watcher.stop()
