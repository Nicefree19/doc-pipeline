"""Configuration schema definitions.

Dataclasses for structured configuration with validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..actions.models import ActionRule
from ..naming.config import ISO19650Config, NamingConfig
from ..templates.schemas import OrganizeTemplate, ProjectTemplate

logger = logging.getLogger("filehub")


@dataclass
class WatcherConfig:
    """File watcher configuration.

    Attributes:
        paths: Directories to watch
        recursive: Watch subdirectories
        use_polling: Force polling mode (for NAS)
        poll_interval: Polling interval in seconds
    """

    paths: list[str] = field(default_factory=list)
    recursive: bool = True
    use_polling: bool = False
    poll_interval: float = 2.0

    def get_watch_paths(self) -> list[Path]:
        """Get validated watch paths."""
        result = []
        for p in self.paths:
            path = Path(p).expanduser()
            if path.exists() and path.is_dir():
                result.append(path)
        return result


@dataclass
class PipelineConfig:
    """Event processing pipeline configuration.

    Attributes:
        debounce_seconds: Time to wait for file stability
        cooldown_seconds: Time before re-notifying same file
        stability_interval: Check interval for file stability
        stability_timeout: Max wait time for file stability
        stability_rounds: Required stable checks before processing
    """

    debounce_seconds: float = 1.0
    cooldown_seconds: float = 60.0
    stability_interval: float = 0.3
    stability_timeout: float = 20.0
    stability_rounds: int = 2


@dataclass
class NotificationConfig:
    """Notification system configuration.

    Attributes:
        enabled: Enable toast notifications
        duration: Notification display duration in seconds
        title: Default notification title
    """

    enabled: bool = True
    duration: int = 10
    title: str = "FileHub"
    slack_webhook: str | None = None
    teams_webhook: str | None = None


@dataclass
class LoggingConfig:
    """Logging configuration.

    Attributes:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        file: Log file path (None for no file logging)
        max_size_mb: Max log file size before rotation
        backup_count: Number of backup files to keep
    """

    level: str = "INFO"
    file: str | None = None
    max_size_mb: int = 10
    backup_count: int = 5


@dataclass
class IgnoreConfig:
    """File ignore configuration.

    Attributes:
        prefixes: Ignore files starting with these prefixes
        extensions: Ignore files with these extensions
        globs: Ignore files matching these glob patterns
    """

    prefixes: list[str] = field(default_factory=lambda: ["~$", "."])
    extensions: list[str] = field(default_factory=lambda: [".tmp", ".bak", ".swp"])
    globs: list[str] = field(default_factory=list)


@dataclass
class TrayConfig:
    """System tray configuration.

    Attributes:
        tooltip: Tray icon tooltip text
        minimize_to_tray: Minimize window to tray instead of closing
    """

    tooltip: str = "FileHub"
    minimize_to_tray: bool = True


@dataclass
class UpdateConfig:
    """Update configuration (tufup).

    Attributes:
        enabled: Enable auto-updates
        url: Update repository URL
        channel: Update channel (stable, beta, alpha)
    """

    enabled: bool = False
    url: str | None = None
    channel: str = "stable"


@dataclass
class FileHubConfig:
    """Root FileHub configuration.

    Combines all configuration sections.
    """

    watcher: WatcherConfig = field(default_factory=WatcherConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    notification: NotificationConfig = field(default_factory=NotificationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ignore: IgnoreConfig = field(default_factory=IgnoreConfig)
    naming: NamingConfig = field(default_factory=NamingConfig)
    actions: list[ActionRule] = field(default_factory=list)
    templates: list[ProjectTemplate] = field(default_factory=list)
    organize_templates: list[OrganizeTemplate] = field(default_factory=list)
    tray: TrayConfig = field(default_factory=TrayConfig)
    update: UpdateConfig = field(default_factory=UpdateConfig)
    naming_pattern: str | None = None

    @property
    def iso19650(self) -> ISO19650Config | None:
        """Backward compatibility for ISO 19650 config."""
        return self.naming.iso19650

    @classmethod
    def from_dict(cls, data: dict) -> FileHubConfig:
        """Create config from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            FileHubConfig instance
        """

        def _make(klass, section: str):
            section_data = data.get(section, {})
            if section_data:
                allowed = klass.__dataclass_fields__.keys()
                unknown = set(section_data.keys()) - set(allowed)
                if unknown:
                    logger.warning(
                        "Unknown keys in '%s' config (ignored): %s",
                        section,
                        ", ".join(sorted(unknown)),
                    )
                sanitized = {k: v for k, v in section_data.items() if k in allowed}
                return klass(**sanitized) if sanitized else klass()
            return klass()

        # Handle NamingConfig (legacy iso19650 key support)
        naming_data = data.get("naming", {})
        iso_data = data.get("iso19650")
        if iso_data and not naming_data.get("iso19650"):
            # Migrate legacay iso19650 content to naming.iso19650
            if isinstance(iso_data, dict):
                naming_data["iso19650"] = iso_data
            naming_data["enabled"] = data.get("enabled", True)  # Should this be naming enabled?

        naming_config = NamingConfig.from_dict(naming_data)

        # Handle Actions
        actions_data = data.get("actions", [])
        actions = [ActionRule.from_dict(a) for a in actions_data]

        # Handle Templates
        templates_data = data.get("templates", [])
        templates = [ProjectTemplate.from_dict(t) for t in templates_data]

        # Handle Organize Templates
        ot_data = data.get("organize_templates", [])
        organize_templates = [OrganizeTemplate.from_dict(t) for t in ot_data]

        return cls(
            watcher=_make(WatcherConfig, "watcher"),
            pipeline=_make(PipelineConfig, "pipeline"),
            notification=_make(NotificationConfig, "notification"),
            logging=_make(LoggingConfig, "logging"),
            ignore=_make(IgnoreConfig, "ignore"),
            naming=naming_config,
            actions=actions,
            templates=templates,
            organize_templates=organize_templates,
            tray=_make(TrayConfig, "tray"),
            update=_make(UpdateConfig, "update"),
            naming_pattern=data.get("naming_pattern"),
        )
