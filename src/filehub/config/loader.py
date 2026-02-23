"""Configuration loader with multi-path priority.

Priority order:
1. CLI argument path
2. Portable config (next to executable)
3. User config (AppData/~/.config)
4. Bundled default config
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from ..i18n import _
from .schema import FileHubConfig

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Configuration loading or validation error."""

    pass


def get_config_path(cli_path: str | None = None, config_name: str = "config.yaml") -> Path:
    """Find configuration file path.

    Priority:
    1. CLI argument path
    2. Portable config (next to executable)
    3. User config (AppData/~/.config)
    4. Bundled default config

    Args:
        cli_path: Path provided via command line
        config_name: Config filename

    Returns:
        Path to configuration file

    Raises:
        ConfigError: If no config file found
    """
    # 1. CLI argument path
    if cli_path:
        path = Path(cli_path)
        # Allow non-existent CLI path if we are initializing?
        # Assuming CLI path is explicit intent.
        if path.exists():
            logger.info(_("Using config from CLI: %s"), path)
            return path
        # If specified but missing, maybe we shouldn't fall back?
        # Current behavior raised error. Keep it.
        raise ConfigError(_("Config file not found: %s") % path)

    # 2. Portable config (next to executable)
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
    else:
        exe_dir = Path(__file__).parent.parent.parent.parent

    portable_config = exe_dir / config_name
    if portable_config.exists():
        logger.info(_("Using portable config: %s"), portable_config)
        return portable_config

    # 3. User config directory (Preferred for installed app)
    if sys.platform == "win32":
        app_data = Path(os.environ.get("APPDATA", "~")).expanduser()
        user_config_dir = app_data / "FileHub"
    else:
        user_config_dir = Path.home() / ".config" / "filehub"

    user_config = user_config_dir / config_name

    if user_config.exists():
        logger.info(_("Using user config: %s"), user_config)
        return user_config

    # 4. Fallback: Initialize from Bundled default if available
    bundled_config = exe_dir / "config" / "default_config.yaml"
    if bundled_config.exists():
        logger.info(_("Found bundled default config at %s"), bundled_config)
        try:
            # Create user config directory
            user_config_dir.mkdir(parents=True, exist_ok=True)
            # Copy bundled to user config
            import shutil

            shutil.copy2(bundled_config, user_config)
            logger.info(_("Initialized user config from bundled default: %s"), user_config)
            return user_config
        except Exception as e:
            logger.warning(_("Failed to initialize user config: %s"), e)
            # Fallback to reading bundled config directly (readonly mode)
            return bundled_config

    # 5. Default to user config path (even if missing, to create fresh)
    return user_config


def load_config(config_path: str | Path | None = None) -> FileHubConfig:
    """Load configuration from file.

    Args:
        config_path: Optional explicit config path

    Returns:
        FileHubConfig instance

    Raises:
        ConfigError: If loading fails
    """
    try:
        if config_path is None:
            path = get_config_path()
        elif isinstance(config_path, str):
            path = Path(config_path)
        else:
            path = config_path

        if not path.exists():
            raise ConfigError(_("Config file not found: %s") % path)

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        config = FileHubConfig.from_dict(data)
        logger.info(_("Configuration loaded from: %s"), path)
        return config

    except yaml.YAMLError as e:
        raise ConfigError(_("Invalid YAML in config file: %s") % e) from e
    except Exception as e:
        if isinstance(e, ConfigError):
            raise
        raise ConfigError(_("Failed to load config: %s") % e) from e


def create_default_config(path: Path) -> None:
    """Create default configuration file.

    Args:
        path: Path to create config at
    """
    default_content = """# FileHub Configuration

watcher:
  paths:
    - "~/Documents"
  recursive: true
  use_polling: false
  poll_interval: 2.0

pipeline:
  debounce_seconds: 1.0
  cooldown_seconds: 60.0
  stability_timeout: 20.0

notification:
  enabled: true
  duration: 10
  title: "FileHub"

logging:
  level: "INFO"
  # file: "~/.filehub/filehub.log"

ignore:
  prefixes:
    - "~$"
    - "."
  extensions:
    - ".tmp"
    - ".bak"
    - ".swp"

# ISO 19650 Naming Convention (optional)
# iso19650:
#   project: ["PROJ"]
#   originator: ["ABC", "XYZ"]
#   volume: ["ZZ"]
#   level: ["00", "01", "02"]
#   type: ["DR", "MO", "SP"]
#   role: ["A", "S", "M", "E"]
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(default_content, encoding="utf-8")
    logger.info(_("Created default config at: %s"), path)
