"""Tests for configuration schema module."""



from src.filehub.config.schema import (
    FileHubConfig,
    IgnoreConfig,
    LoggingConfig,
    NotificationConfig,
    PipelineConfig,
    WatcherConfig,
)


class TestWatcherConfig:
    """Test WatcherConfig dataclass."""

    def test_default_values(self):
        """Test default WatcherConfig values."""
        cfg = WatcherConfig()
        assert cfg.paths == []
        assert cfg.recursive is True
        assert cfg.use_polling is False
        assert cfg.poll_interval == 2.0

    def test_get_watch_paths_existing_dir(self, tmp_path):
        """Test get_watch_paths returns existing directories."""
        sub = tmp_path / "docs"
        sub.mkdir()
        cfg = WatcherConfig(paths=[str(sub)])

        result = cfg.get_watch_paths()

        assert len(result) == 1
        assert result[0] == sub

    def test_get_watch_paths_nonexistent_dir(self):
        """Test get_watch_paths skips nonexistent directories."""
        cfg = WatcherConfig(paths=["/nonexistent/path/that/does/not/exist"])

        result = cfg.get_watch_paths()

        assert len(result) == 0

    def test_get_watch_paths_file_not_dir(self, tmp_path):
        """Test get_watch_paths skips paths that are files, not directories."""
        file_path = tmp_path / "somefile.txt"
        file_path.write_text("content")
        cfg = WatcherConfig(paths=[str(file_path)])

        result = cfg.get_watch_paths()

        assert len(result) == 0

    def test_get_watch_paths_mixed(self, tmp_path):
        """Test get_watch_paths with mix of existing and nonexistent paths."""
        existing = tmp_path / "existing"
        existing.mkdir()
        cfg = WatcherConfig(paths=[str(existing), "/nonexistent/path"])

        result = cfg.get_watch_paths()

        assert len(result) == 1
        assert result[0] == existing

    def test_get_watch_paths_empty(self):
        """Test get_watch_paths with empty paths list."""
        cfg = WatcherConfig(paths=[])

        result = cfg.get_watch_paths()

        assert result == []


class TestPipelineConfig:
    """Test PipelineConfig dataclass."""

    def test_default_values(self):
        """Test default PipelineConfig values."""
        cfg = PipelineConfig()
        assert cfg.debounce_seconds == 1.0
        assert cfg.cooldown_seconds == 60.0
        assert cfg.stability_interval == 0.3
        assert cfg.stability_timeout == 20.0
        assert cfg.stability_rounds == 2


class TestNotificationConfig:
    """Test NotificationConfig dataclass."""

    def test_default_values(self):
        """Test default NotificationConfig values."""
        cfg = NotificationConfig()
        assert cfg.enabled is True
        assert cfg.duration == 10
        assert cfg.title == "FileHub"


class TestLoggingConfig:
    """Test LoggingConfig dataclass."""

    def test_default_values(self):
        """Test default LoggingConfig values."""
        cfg = LoggingConfig()
        assert cfg.level == "INFO"
        assert cfg.file is None
        assert cfg.max_size_mb == 10
        assert cfg.backup_count == 5


class TestIgnoreConfig:
    """Test IgnoreConfig dataclass (schema version)."""

    def test_default_values(self):
        """Test default IgnoreConfig values have prefixes and extensions."""
        cfg = IgnoreConfig()
        assert "~$" in cfg.prefixes
        assert "." in cfg.prefixes
        assert ".tmp" in cfg.extensions
        assert ".bak" in cfg.extensions
        assert ".swp" in cfg.extensions
        assert cfg.globs == []


class TestFileHubConfig:
    """Test FileHubConfig dataclass."""

    def test_default_values(self):
        """Test default FileHubConfig values."""
        cfg = FileHubConfig()
        assert isinstance(cfg.watcher, WatcherConfig)
        assert isinstance(cfg.pipeline, PipelineConfig)
        assert isinstance(cfg.notification, NotificationConfig)
        assert isinstance(cfg.logging, LoggingConfig)
        assert isinstance(cfg.ignore, IgnoreConfig)
        assert cfg.iso19650 is None

    def test_from_dict_full(self):
        """Test from_dict with a full config dictionary."""
        data = {
            "watcher": {
                "paths": ["~/Documents"],
                "recursive": False,
                "use_polling": True,
                "poll_interval": 5.0,
            },
            "pipeline": {
                "debounce_seconds": 2.0,
                "cooldown_seconds": 120.0,
                "stability_interval": 0.5,
                "stability_timeout": 30.0,
                "stability_rounds": 3,
            },
            "notification": {
                "enabled": False,
                "duration": 5,
                "title": "TestApp",
            },
            "logging": {
                "level": "DEBUG",
                "file": "/tmp/test.log",
                "max_size_mb": 20,
                "backup_count": 3,
            },
            "ignore": {
                "prefixes": ["~$"],
                "extensions": [".tmp"],
                "globs": ["*.log"],
            },
        }

        cfg = FileHubConfig.from_dict(data)

        assert cfg.watcher.recursive is False
        assert cfg.watcher.use_polling is True
        assert cfg.watcher.poll_interval == 5.0
        assert cfg.pipeline.debounce_seconds == 2.0
        assert cfg.pipeline.cooldown_seconds == 120.0
        assert cfg.notification.enabled is False
        assert cfg.notification.title == "TestApp"
        assert cfg.logging.level == "DEBUG"
        assert cfg.ignore.prefixes == ["~$"]

    def test_from_dict_empty(self):
        """Test from_dict with empty dictionary returns defaults."""
        cfg = FileHubConfig.from_dict({})

        assert isinstance(cfg.watcher, WatcherConfig)
        assert cfg.watcher.recursive is True
        assert cfg.pipeline.debounce_seconds == 1.0
        assert cfg.notification.enabled is True
        assert cfg.iso19650 is None

    def test_from_dict_with_iso19650(self):
        """Test from_dict with ISO 19650 configuration."""
        data = {
            "iso19650": {
                "project": ["PROJ"],
                "originator": ["ABC", "XYZ"],
                "volume": ["ZZ"],
                "level": ["00", "01"],
                "type": ["DR", "MO"],
                "role": ["A", "S"],
            },
        }

        cfg = FileHubConfig.from_dict(data)

        assert cfg.iso19650 is not None
        assert "PROJ" in cfg.iso19650.project
        assert "ABC" in cfg.iso19650.originator
        assert "ZZ" in cfg.iso19650.volume

    def test_from_dict_with_legacy_iso19650_enabled_key(self):
        """Legacy iso19650.enabled key should not break config parsing."""
        data = {
            "iso19650": {
                "enabled": True,
                "project": ["PRJ"],
                "originator": ["ABC"],
            },
        }

        cfg = FileHubConfig.from_dict(data)

        assert cfg.iso19650 is not None
        assert cfg.iso19650.project == ["PRJ"]
        assert cfg.iso19650.originator == ["ABC"]

    def test_from_dict_partial_sections(self):
        """Test from_dict with only some sections provided."""
        data = {
            "pipeline": {"debounce_seconds": 3.0},
        }

        cfg = FileHubConfig.from_dict(data)

        assert cfg.pipeline.debounce_seconds == 3.0
        # Defaults for missing sections
        assert cfg.watcher.recursive is True
        assert cfg.notification.enabled is True
        assert cfg.iso19650 is None
