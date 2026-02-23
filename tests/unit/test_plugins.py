"""Tests for the plugin system."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.filehub.plugins.base import PluginBase
from src.filehub.plugins.manager import PluginManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _DummyPlugin(PluginBase):
    """Minimal concrete plugin for testing."""

    def __init__(self, plugin_name: str = "dummy") -> None:
        self._name = plugin_name

    @property
    def name(self) -> str:
        return self._name


class _TrackingPlugin(PluginBase):
    """Plugin that records every hook invocation."""

    def __init__(self, plugin_name: str = "tracker") -> None:
        self._name = plugin_name
        self.calls: list[tuple[str, tuple]] = []

    @property
    def name(self) -> str:
        return self._name

    def on_file_ready(self, path: Path, validation_result: object) -> None:
        self.calls.append(("on_file_ready", (path, validation_result)))

    def on_validation_error(self, path: Path, message: str) -> None:
        self.calls.append(("on_validation_error", (path, message)))

    def on_startup(self) -> None:
        self.calls.append(("on_startup", ()))

    def on_shutdown(self) -> None:
        self.calls.append(("on_shutdown", ()))


class _ExplodingPlugin(PluginBase):
    """Plugin that raises on every hook."""

    def __init__(self, plugin_name: str = "exploding") -> None:
        self._name = plugin_name

    @property
    def name(self) -> str:
        return self._name

    def on_file_ready(self, path: Path, validation_result: object) -> None:
        raise RuntimeError("boom in on_file_ready")

    def on_validation_error(self, path: Path, message: str) -> None:
        raise RuntimeError("boom in on_validation_error")

    def on_startup(self) -> None:
        raise RuntimeError("boom in on_startup")

    def on_shutdown(self) -> None:
        raise RuntimeError("boom in on_shutdown")


# ---------------------------------------------------------------------------
# PluginBase Tests
# ---------------------------------------------------------------------------

class TestPluginBase:
    """Tests for the PluginBase abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """PluginBase cannot be instantiated because name is abstract."""
        with pytest.raises(TypeError):
            PluginBase()  # type: ignore[abstract]

    def test_concrete_plugin_instantiates(self):
        """A subclass implementing name can be created."""
        plugin = _DummyPlugin("test")
        assert plugin.name == "test"

    def test_default_on_file_ready_is_noop(self):
        """Default on_file_ready does not raise."""
        plugin = _DummyPlugin()
        plugin.on_file_ready(Path("/fake"), object())

    def test_default_on_validation_error_is_noop(self):
        """Default on_validation_error does not raise."""
        plugin = _DummyPlugin()
        plugin.on_validation_error(Path("/fake"), "err")

    def test_default_on_startup_is_noop(self):
        """Default on_startup does not raise."""
        plugin = _DummyPlugin()
        plugin.on_startup()

    def test_default_on_shutdown_is_noop(self):
        """Default on_shutdown does not raise."""
        plugin = _DummyPlugin()
        plugin.on_shutdown()


# ---------------------------------------------------------------------------
# PluginManager Tests
# ---------------------------------------------------------------------------

class TestPluginManager:
    """Tests for the PluginManager class."""

    def test_register_and_list(self):
        """Registering a plugin makes it appear in list_plugins."""
        mgr = PluginManager()
        mgr.register(_DummyPlugin("alpha"))
        assert mgr.list_plugins() == ["alpha"]

    def test_register_multiple(self):
        """Multiple distinct plugins are all listed."""
        mgr = PluginManager()
        mgr.register(_DummyPlugin("a"))
        mgr.register(_DummyPlugin("b"))
        assert mgr.list_plugins() == ["a", "b"]

    def test_unregister_existing(self):
        """Unregistering an existing plugin returns True."""
        mgr = PluginManager()
        mgr.register(_DummyPlugin("x"))
        assert mgr.unregister("x") is True
        assert mgr.list_plugins() == []

    def test_unregister_nonexistent_returns_false(self):
        """Unregistering a name that was never registered returns False."""
        mgr = PluginManager()
        assert mgr.unregister("ghost") is False

    def test_get_plugin_found(self):
        """get_plugin returns the plugin when it exists."""
        mgr = PluginManager()
        plugin = _DummyPlugin("found")
        mgr.register(plugin)
        assert mgr.get_plugin("found") is plugin

    def test_get_plugin_not_found(self):
        """get_plugin returns None when no match."""
        mgr = PluginManager()
        assert mgr.get_plugin("missing") is None

    def test_duplicate_name_replaces(self):
        """Registering a plugin with an existing name replaces the old one."""
        mgr = PluginManager()
        first = _DummyPlugin("dup")
        second = _DummyPlugin("dup")
        mgr.register(first)
        mgr.register(second)

        assert mgr.list_plugins() == ["dup"]
        assert mgr.get_plugin("dup") is second

    # ------------------------------------------------------------------
    # Notification tests
    # ------------------------------------------------------------------

    def test_notify_file_ready_calls_all(self):
        """notify_file_ready invokes on_file_ready on every plugin."""
        mgr = PluginManager()
        t1 = _TrackingPlugin("t1")
        t2 = _TrackingPlugin("t2")
        mgr.register(t1)
        mgr.register(t2)

        path = Path("/some/file.txt")
        result = MagicMock()
        mgr.notify_file_ready(path, result)

        assert len(t1.calls) == 1
        assert t1.calls[0] == ("on_file_ready", (path, result))
        assert len(t2.calls) == 1
        assert t2.calls[0] == ("on_file_ready", (path, result))

    def test_notify_validation_error_calls_all(self):
        """notify_validation_error invokes on_validation_error on every plugin."""
        mgr = PluginManager()
        t1 = _TrackingPlugin("t1")
        t2 = _TrackingPlugin("t2")
        mgr.register(t1)
        mgr.register(t2)

        path = Path("/bad.txt")
        mgr.notify_validation_error(path, "bad name")

        assert len(t1.calls) == 1
        assert t1.calls[0] == ("on_validation_error", (path, "bad name"))
        assert len(t2.calls) == 1

    def test_notify_startup_calls_all(self):
        """notify_startup invokes on_startup on every plugin."""
        mgr = PluginManager()
        t1 = _TrackingPlugin("t1")
        t2 = _TrackingPlugin("t2")
        mgr.register(t1)
        mgr.register(t2)

        mgr.notify_startup()

        assert t1.calls == [("on_startup", ())]
        assert t2.calls == [("on_startup", ())]

    def test_notify_shutdown_calls_all(self):
        """notify_shutdown invokes on_shutdown on every plugin."""
        mgr = PluginManager()
        t1 = _TrackingPlugin("t1")
        mgr.register(t1)

        mgr.notify_shutdown()

        assert t1.calls == [("on_shutdown", ())]

    def test_exception_in_plugin_is_caught_and_logged(self, caplog):
        """A plugin exception is caught, logged, and does not crash the manager."""
        mgr = PluginManager()
        bad = _ExplodingPlugin("bad")
        good = _TrackingPlugin("good")
        mgr.register(bad)
        mgr.register(good)

        path = Path("/file.txt")

        with caplog.at_level(logging.ERROR, logger="filehub"):
            mgr.notify_file_ready(path, None)

        # The good plugin was still called despite the bad one exploding
        assert len(good.calls) == 1
        assert good.calls[0][0] == "on_file_ready"

        # The error was logged
        assert "bad" in caplog.text
        assert "on_file_ready" in caplog.text

    def test_exception_in_notify_validation_error_caught(self, caplog):
        """Exception in on_validation_error is caught and logged."""
        mgr = PluginManager()
        mgr.register(_ExplodingPlugin("bad"))

        with caplog.at_level(logging.ERROR, logger="filehub"):
            mgr.notify_validation_error(Path("/x"), "err")

        assert "bad" in caplog.text
        assert "on_validation_error" in caplog.text

    def test_exception_in_notify_startup_caught(self, caplog):
        """Exception in on_startup is caught and logged."""
        mgr = PluginManager()
        mgr.register(_ExplodingPlugin("bad"))

        with caplog.at_level(logging.ERROR, logger="filehub"):
            mgr.notify_startup()

        assert "bad" in caplog.text
        assert "on_startup" in caplog.text

    def test_exception_in_notify_shutdown_caught(self, caplog):
        """Exception in on_shutdown is caught and logged."""
        mgr = PluginManager()
        mgr.register(_ExplodingPlugin("bad"))

        with caplog.at_level(logging.ERROR, logger="filehub"):
            mgr.notify_shutdown()

        assert "bad" in caplog.text
        assert "on_shutdown" in caplog.text

    def test_empty_manager_notify_methods_dont_raise(self):
        """All notify methods on an empty manager complete without error."""
        mgr = PluginManager()
        mgr.notify_file_ready(Path("/x"), None)
        mgr.notify_validation_error(Path("/x"), "msg")
        mgr.notify_startup()
        mgr.notify_shutdown()
