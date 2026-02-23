"""Unit tests for config loader."""

import os
import sys
from unittest.mock import patch

import pytest
import yaml
from filehub.config.loader import (
    ConfigError,
    create_default_config,
    get_config_path,
    load_config,
)
from filehub.config.schema import FileHubConfig


def test_get_config_path_cli(tmp_path):
    """Test getting config path from CLI argument."""
    config_file = tmp_path / "config.yaml"
    config_file.touch()
    assert get_config_path(str(config_file)) == config_file


def test_get_config_path_cli_missing():
    """Test missing CLI config file raises error."""
    with pytest.raises(ConfigError, match="Config file not found"):
        get_config_path("non_existent.yaml")


def test_get_config_path_portable(tmp_path):
    """Test portable config priority (next to executable)."""
    # Mock sys.frozen and executable path
    with (
        patch("sys.frozen", True, create=True),
        patch("sys.executable", str(tmp_path / "filehub.exe")),
    ):

        config_file = tmp_path / "config.yaml"
        config_file.touch()

        try:
            assert get_config_path() == config_file
        except ConfigError as e:
            # Should not happen with mock
            pytest.fail(f"ConfigError raised unexpectedly: {e}")


def test_get_config_path_user(tmp_path):
    """Test user config directory priority."""
    # Mock portable not found
    with (
        patch("sys.frozen", False, create=True),
        patch("filehub.config.loader.Path.home", return_value=tmp_path),
    ):

        if sys.platform == "win32":
            with patch.dict(os.environ, {"APPDATA": str(tmp_path)}):
                config_dir = tmp_path / "FileHub"
                config_dir.mkdir()
                config_file = config_dir / "config.yaml"
                config_file.touch()
                assert get_config_path() == config_file
        else:
            config_dir = tmp_path / ".config" / "filehub"
            config_dir.mkdir(parents=True)
            config_file = config_dir / "config.yaml"
            config_file.touch()
            assert get_config_path() == config_file


def test_load_config_valid(tmp_path):
    """Test loading valid configuration."""
    config_file = tmp_path / "config.yaml"
    data = {
        "watcher": {"paths": ["/tmp"], "recursive": False},
        "pipeline": {"debounce_seconds": 2.0},
        "iso19650": {"project": ["TEST"]},
    }
    with open(config_file, "w") as f:
        yaml.dump(data, f)

    config = load_config(config_file)

    assert isinstance(config, FileHubConfig)
    assert config.watcher.paths == ["/tmp"]
    assert config.watcher.recursive is False
    assert config.pipeline.debounce_seconds == 2.0
    assert config.iso19650 is not None
    assert config.iso19650.project == ["TEST"]
    assert config.logging.level == "INFO"  # Default


def test_load_config_invalid_yaml(tmp_path):
    """Test loading text file with invalid YAML."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("invalid: [yaml", encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(config_file)


def test_create_default_config(tmp_path):
    """Test creating default configuration."""
    config_file = tmp_path / "config.yaml"
    create_default_config(config_file)

    assert config_file.exists()
    content = config_file.read_text(encoding="utf-8")
    assert "watcher:" in content
    assert "pipeline:" in content

    # Verify it loads back
    config = load_config(config_file)
    assert isinstance(config, FileHubConfig)
