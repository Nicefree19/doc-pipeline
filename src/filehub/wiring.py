"""Integration wiring for FileHub Phase 3 modules.

Factory functions that create and connect Phase 3 components
(PluginManager, StatsCollector, ChannelManager) without requiring
a complex dependency-injection framework.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .core.notification.channels.console import ConsoleChannel
from .core.notification.channels.log import LogChannel
from .core.notification.channels.manager import ChannelManager
from .core.pipeline.processor import Processor
from .plugins.manager import PluginManager
from .reporting.collector import StatsCollector

logger = logging.getLogger("filehub.wiring")


def create_plugin_manager() -> PluginManager:
    """Create and return an empty PluginManager.

    Plugins register themselves at a later stage.

    Returns:
        A fresh PluginManager instance.
    """
    logger.debug("Creating PluginManager")
    return PluginManager()


def create_stats_collector() -> StatsCollector:
    """Create and return a StatsCollector.

    Returns:
        A fresh StatsCollector instance.
    """
    logger.debug("Creating StatsCollector")
    return StatsCollector()


def create_channel_manager() -> ChannelManager:
    """Create a ChannelManager pre-loaded with default channels.

    Default channels:
      - ConsoleChannel (prints to stdout)
      - LogChannel (logs via Python logging)

    Returns:
        A ChannelManager with console and log channels registered.
    """
    logger.debug("Creating ChannelManager with default channels")
    manager = ChannelManager()
    manager.add_channel(ConsoleChannel())
    manager.add_channel(LogChannel())
    return manager


def wire_stats_to_processor(
    collector: StatsCollector, processor: Processor
) -> None:
    """Wire a StatsCollector to a Processor's on_processed callback.

    Wraps the processor's ``set_on_processed`` callback so that every
    successfully processed file is also recorded in the stats collector.

    Args:
        collector: The StatsCollector to record into.
        processor: The Processor whose callback is being wrapped.
    """
    logger.debug("Wiring StatsCollector to Processor")
    existing = processor._on_processed

    def _on_processed(path: Path) -> None:
        collector.record_file_processed(path)
        if existing:
            existing(path)

    processor.set_on_processed(_on_processed)
