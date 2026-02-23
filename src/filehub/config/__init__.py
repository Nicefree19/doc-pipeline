"""FileHub Configuration Module.

Handles loading and validating configuration from multiple sources.
"""

from .loader import ConfigError, create_default_config, get_config_path, load_config
from .schema import (
    FileHubConfig,
    LoggingConfig,
    NotificationConfig,
    PipelineConfig,
    WatcherConfig,
)

__all__ = [
    "load_config",
    "get_config_path",
    "create_default_config",
    "ConfigError",
    "FileHubConfig",
    "WatcherConfig",
    "PipelineConfig",
    "NotificationConfig",
    "LoggingConfig",
]
