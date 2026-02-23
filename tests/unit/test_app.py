"""Tests for Application class and run_app function."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.filehub.config.schema import FileHubConfig

# ---------------------------------------------------------------------------
# Fixture: patch all heavy dependencies so Application.__init__ completes
# without touching the real file system, OS observer, or notification layer.
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_deps():
    """Patch every heavy dependency that Application.__init__ imports lazily.

    The imports inside __init__ are relative (``from .core.pipeline import ...``).
    At runtime they resolve to ``src.filehub.core.pipeline``, etc.  We patch the
    *source* modules so the names are replaced before ``Application.__init__``
    binds them locally.
    """
    mock_processor_cls = MagicMock(name="Processor")
    mock_processor = MagicMock(name="processor_instance")
    mock_processor_cls.return_value = mock_processor

    mock_ignore_filter_cls = MagicMock(name="IgnoreFilter")
    mock_ignore_config_cls = MagicMock(name="IgnoreConfig")

    mock_notifier_cls = MagicMock(name="Notifier")
    mock_notifier = MagicMock(name="notifier_instance")
    mock_notifier_cls.return_value = mock_notifier

    mock_observer = MagicMock(name="observer_instance")
    mock_create_observer = MagicMock(name="create_observer", return_value=mock_observer)
    mock_start_observer = MagicMock(name="start_observer")
    mock_stop_observer = MagicMock(name="stop_observer")

    mock_reconcile_cls = MagicMock(name="ReconcileScanner")
    mock_reconcile = MagicMock(name="reconcile_instance")
    mock_reconcile_cls.return_value = mock_reconcile

    mock_validator_cls = MagicMock(name="ISO19650Validator")

    patches = [
        patch("src.filehub.core.pipeline.Processor", mock_processor_cls),
        patch("src.filehub.core.pipeline.IgnoreFilter", mock_ignore_filter_cls),
        patch("src.filehub.core.pipeline.ignore_filter.IgnoreConfig", mock_ignore_config_cls),
        patch("src.filehub.core.notification.Notifier", mock_notifier_cls),
        patch("src.filehub.core.watcher.create_observer", mock_create_observer),
        patch("src.filehub.core.watcher.start_observer", mock_start_observer),
        patch("src.filehub.core.watcher.stop_observer", mock_stop_observer),
        patch("src.filehub.core.watcher.ReconcileScanner", mock_reconcile_cls),
        patch("src.filehub.naming.ISO19650Validator", mock_validator_cls),
    ]

    for p in patches:
        p.start()

    yield {
        "processor_cls": mock_processor_cls,
        "processor": mock_processor,
        "ignore_filter_cls": mock_ignore_filter_cls,
        "ignore_config_cls": mock_ignore_config_cls,
        "notifier_cls": mock_notifier_cls,
        "notifier": mock_notifier,
        "observer": mock_observer,
        "create_observer": mock_create_observer,
        "start_observer": mock_start_observer,
        "stop_observer": mock_stop_observer,
        "reconcile_cls": mock_reconcile_cls,
        "reconcile": mock_reconcile,
        "validator_cls": mock_validator_cls,
    }

    for p in patches:
        p.stop()


def _make_app(mock_deps, **kwargs):
    """Helper: import Application fresh and instantiate with given kwargs."""
    from src.filehub.app import Application

    return Application(**kwargs)


# ===================================================================
# TestApplication
# ===================================================================


class TestApplication:
    """Tests for the Application class."""

    # ---- init / config --------------------------------------------------

    def test_init_default_config(self, mock_deps):
        """Init without config_path uses FileHubConfig defaults."""
        app = _make_app(mock_deps)

        # Default config values should be present
        assert isinstance(app._config, FileHubConfig)
        assert app._config.pipeline.debounce_seconds == 1.0
        assert app._config.notification.enabled is True

    def test_init_with_valid_config_path(self, mock_deps, tmp_path):
        """Init with a valid config file loads it correctly."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "pipeline:\n  debounce_seconds: 3.5\n",
            encoding="utf-8",
        )

        app = _make_app(mock_deps, config_path=cfg_file)

        assert app._config.pipeline.debounce_seconds == 3.5

    def test_init_invalid_config_falls_back_to_defaults(self, mock_deps, tmp_path, caplog):
        """Invalid config falls back to defaults and logs a warning."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text(":\n  broken: [yaml\n  nope", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="filehub"):
            app = _make_app(mock_deps, config_path=bad_file)

        # Should have fallen back to defaults
        assert isinstance(app._config, FileHubConfig)
        assert app._config.pipeline.debounce_seconds == 1.0
        # Warning should have been logged
        assert any(
            "defaults" in r.message.lower() or "config" in r.message.lower()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )

    # ---- watch paths ----------------------------------------------------

    def test_watch_paths_from_cli_override_config(self, mock_deps, tmp_path):
        """CLI watch_paths override config and defaults."""
        dir_a = tmp_path / "a"
        dir_a.mkdir()

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "watcher:\n  paths:\n    - " + str(tmp_path / "b") + "\n",
            encoding="utf-8",
        )

        app = _make_app(mock_deps, watch_paths=[dir_a], config_path=cfg_file)

        assert app._watch_paths == [dir_a]

    def test_watch_paths_from_config(self, mock_deps, tmp_path):
        """Config paths used when no CLI paths provided."""
        watch_dir = tmp_path / "docs"
        watch_dir.mkdir()

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "watcher:\n  paths:\n    - " + str(watch_dir) + "\n",
            encoding="utf-8",
        )

        app = _make_app(mock_deps, config_path=cfg_file)

        assert watch_dir in app._watch_paths

    def test_watch_paths_default_documents(self, mock_deps):
        """Default ~/Documents used when no paths specified anywhere."""
        with patch("pathlib.Path.home") as mock_home, patch("pathlib.Path.exists") as mock_exists:

            mock_home.return_value = Path("/mock/home")
            mock_docs = Path("/mock/home/Documents")
            mock_exists.side_effect = lambda: (
                True
                if str(mock_docs) in str(Path.cwd()) or str(mock_docs) in str(mock_docs)
                else False
            )

            # Simplified: just ensure it tries to use the path if it exists
            # We mock exists for the specific path we expect
            mock_exists.return_value = True

            app = _make_app(mock_deps)

            # logical check: if it exists, it should be in there
            assert mock_docs in app._watch_paths

    # ---- logging --------------------------------------------------------

    def test_setup_logging_configures_root_logger(self, mock_deps):
        """_setup_logging calls basicConfig with INFO and StreamHandler."""
        with patch("src.filehub.app.logging.basicConfig") as mock_basic:
            _make_app(mock_deps)

        mock_basic.assert_called_once()
        kwargs = mock_basic.call_args
        assert kwargs[1]["level"] == logging.INFO

    def test_apply_logging_config_sets_level(self, mock_deps, tmp_path):
        """_apply_logging_config changes logger level from config."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("logging:\n  level: DEBUG\n", encoding="utf-8")

        _make_app(mock_deps, config_path=cfg_file)

        filehub_logger = logging.getLogger("filehub")
        assert filehub_logger.level == logging.DEBUG

        # Reset to INFO so other tests are not affected
        filehub_logger.setLevel(logging.INFO)

    def test_apply_logging_config_adds_file_handler(self, mock_deps, tmp_path):
        """When config.logging.file is set, a RotatingFileHandler is added."""
        log_file = tmp_path / "logs" / "filehub.log"
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            'logging:\n  file: "'
            + str(log_file).replace("\\", "\\\\")
            + '"\n  max_size_mb: 5\n  backup_count: 3\n',
            encoding="utf-8",
        )

        _make_app(mock_deps, config_path=cfg_file)

        filehub_logger = logging.getLogger("filehub")
        from logging.handlers import RotatingFileHandler

        rotating_handlers = [
            h for h in filehub_logger.handlers if isinstance(h, RotatingFileHandler)
        ]
        assert len(rotating_handlers) >= 1
        handler = rotating_handlers[-1]
        assert handler.maxBytes == 5 * 1024 * 1024
        assert handler.backupCount == 3

        # Cleanup: remove the handler so it doesn't leak to other tests
        for h in rotating_handlers:
            filehub_logger.removeHandler(h)
            h.close()

    # ---- callbacks & delegation -----------------------------------------

    def test_on_validation_error_calls_notifier(self, mock_deps):
        """_on_validation_error delegates to notifier.notify_validation_error."""
        app = _make_app(mock_deps)

        test_path = Path("/some/bad_file.txt")
        app._on_validation_error(test_path, "Invalid name")

        mock_deps["notifier"].notify_validation_error.assert_called_once_with(
            filename="bad_file.txt",
            reason="Invalid name",
            file_path=test_path,
        )

    def test_pause_resume_delegates_to_processor(self, mock_deps):
        """pause()/resume() call the corresponding processor methods."""
        app = _make_app(mock_deps)

        app.pause()
        mock_deps["processor"].pause.assert_called_once()

        app.resume()
        mock_deps["processor"].resume.assert_called_once()

    def test_stop_clears_running_event(self, mock_deps):
        """stop() clears the _running event when it was set."""
        app = _make_app(mock_deps)

        # Simulate a running state
        app._running.set()
        assert app._running.is_set()

        app.stop()

        assert not app._running.is_set()


# ===================================================================
# TestRunApp
# ===================================================================


class TestRunApp:
    """Tests for the run_app helper function."""

    def test_run_app_returns_zero_on_success(self, mock_deps):
        """run_app returns 0 when Application starts and stops cleanly."""
        from src.filehub.app import run_app

        mock_app_instance = MagicMock()
        with patch("src.filehub.app.Application", return_value=mock_app_instance):
            result = run_app()

        assert result == 0
        mock_app_instance.start.assert_called_once()

    def test_run_app_returns_one_on_error(self):
        """run_app returns 1 when Application init raises an exception."""
        from src.filehub.app import run_app

        with patch("src.filehub.app.Application", side_effect=RuntimeError("boom")):
            result = run_app()

        assert result == 1
