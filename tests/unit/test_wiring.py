"""Tests for the wiring module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from src.filehub.core.notification.channels.manager import ChannelManager
from src.filehub.plugins.manager import PluginManager
from src.filehub.reporting.collector import StatsCollector
from src.filehub.wiring import (
    create_channel_manager,
    create_plugin_manager,
    create_stats_collector,
    wire_stats_to_processor,
)


class TestCreatePluginManager:
    """Tests for create_plugin_manager factory."""

    def test_returns_plugin_manager_instance(self):
        """create_plugin_manager returns a PluginManager."""
        result = create_plugin_manager()
        assert isinstance(result, PluginManager)

    def test_returns_empty_manager(self):
        """Returned PluginManager has no plugins registered."""
        manager = create_plugin_manager()
        assert manager.list_plugins() == []


class TestCreateStatsCollector:
    """Tests for create_stats_collector factory."""

    def test_returns_stats_collector_instance(self):
        """create_stats_collector returns a StatsCollector."""
        result = create_stats_collector()
        assert isinstance(result, StatsCollector)

    def test_initial_summary_is_zeroed(self):
        """Returned StatsCollector starts with all-zero counters."""
        collector = create_stats_collector()
        summary = collector.get_summary()
        assert summary["total_files_processed"] == 0
        assert summary["validation_passed"] == 0
        assert summary["validation_failed"] == 0


class TestCreateChannelManager:
    """Tests for create_channel_manager factory."""

    def test_returns_channel_manager_instance(self):
        """create_channel_manager returns a ChannelManager."""
        result = create_channel_manager()
        assert isinstance(result, ChannelManager)

    def test_has_two_default_channels(self):
        """Returned ChannelManager has exactly 2 default channels."""
        manager = create_channel_manager()
        assert len(manager.list_channels()) == 2

    def test_has_console_channel(self):
        """Returned ChannelManager includes a console channel."""
        manager = create_channel_manager()
        assert "console" in manager.list_channels()

    def test_has_log_channel(self):
        """Returned ChannelManager includes a log channel."""
        manager = create_channel_manager()
        assert "log" in manager.list_channels()


class TestWireStatsToProcessor:
    """Tests for wire_stats_to_processor."""

    def test_registers_callback_on_processor(self):
        """wire_stats_to_processor calls set_on_processed on the processor."""
        collector = StatsCollector()
        mock_processor = MagicMock()

        wire_stats_to_processor(collector, mock_processor)

        mock_processor.set_on_processed.assert_called_once()

    def test_callback_records_file_processed(self):
        """The registered callback records a processed file in the collector."""
        collector = StatsCollector()
        mock_processor = MagicMock()

        wire_stats_to_processor(collector, mock_processor)

        # Extract the callback that was passed to set_on_processed
        callback = mock_processor.set_on_processed.call_args[0][0]

        # Invoke it with a test path
        callback(Path("/test/file.txt"))

        summary = collector.get_summary()
        assert summary["total_files_processed"] == 1
