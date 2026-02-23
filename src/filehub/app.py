"""FileHub Application Core.

Thread model:
- Main Thread: pystray.Icon.run() (UI loop)
- Worker Thread 1: watchdog Observer
- Worker Thread 2: Processor.run() (Queue consumer)
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
from pathlib import Path
from queue import Queue

from . import __version__
from .i18n import _

logger = logging.getLogger("filehub")


class Application:
    """FileHub unified application."""

    def __init__(
        self,
        watch_paths: list[Path] | None = None,
        config_path: str | Path | None = None,
        target_root: Path | None = None,
    ):
        """Initialize FileHub application.

        Args:
            watch_paths: Paths to monitor for file changes
            config_path: Path to configuration file (optional)
            target_root: Root directory for action rule targets (optional)
        """
        # Setup logging first
        self._setup_logging()

        # Load configuration
        from .config import ConfigError, FileHubConfig, load_config

        try:
            self._config = load_config(config_path)
        except ConfigError as e:
            logger.warning(_("Config load failed, using defaults: %s"), e)
            self._config = FileHubConfig()

        # Apply logging configuration
        self._apply_logging_config()

        # Watch paths: CLI args > config > default
        if watch_paths:
            self._watch_paths = [Path(p) for p in watch_paths]
        elif self._config.watcher.paths:
            self._watch_paths = self._config.watcher.get_watch_paths()
        else:
            default_path = Path.home() / "Documents"
            self._watch_paths = [default_path] if default_path.exists() else []

        logger.info(_("Application initializing"))
        logger.info(_("Watch paths: %s"), [str(p) for p in self._watch_paths])

        # Queue
        self._queue: Queue = Queue(maxsize=1000)

        # Import core components
        from .core.notification import Notifier
        from .core.pipeline import IgnoreFilter, Processor
        from .core.pipeline.ignore_filter import IgnoreConfig
        from .core.watcher import ReconcileScanner, create_observer, start_observer, stop_observer

        # Create Validator (Profile-based)
        from .naming.profiles.loader import ProfileLoader
        from .naming.validator import ProfileValidator

        self._validator = None
        if self._config.naming.enabled:
            loader = ProfileLoader(self._config.naming)
            profile = loader.get_active_profile()

            if profile:
                self._validator = ProfileValidator(profile)
                logger.info(_("Validator enabled: %s (%s)"), profile.name, profile.type)

        # Action Engine
        from .actions.engine import ActionEngine

        self._action_engine = None
        if self._config.actions:
            self._action_engine = ActionEngine(
                self._config.actions,
                target_root=target_root,
            )
            logger.info(
                _("Action engine enabled with %d rules"),
                len(self._config.actions),
            )

        # --- Phase 3: Stats & Plugins ---
        from .reporting.collector import StatsCollector
        from .reporting.store import StatsStore
        from .wiring import (
            create_plugin_manager,
            wire_stats_to_processor,
        )

        try:
            db_dir = Path.home() / ".filehub"
            db_dir.mkdir(parents=True, exist_ok=True)
            self._stats_store = StatsStore(
                db_dir / "stats.db",
            )
            logger.info(
                _("Stats store initialized: %s"),
                db_dir / "stats.db",
            )
        except Exception:
            logger.warning(
                "Persistent stats unavailable, using in-memory",
            )
            self._stats_store = StatsStore(":memory:")
        self._stats_collector = StatsCollector()
        self._plugin_manager = create_plugin_manager()

        # Notifier
        self._notifier = Notifier(
            title=self._config.notification.title,
            duration=self._config.notification.duration,
            enabled=self._config.notification.enabled,
            slack_webhook=self._config.notification.slack_webhook,
            teams_webhook=self._config.notification.teams_webhook,
        )

        # Ignore config from loaded config
        ignore_config = IgnoreConfig(
            prefixes=self._config.ignore.prefixes,
            extensions=self._config.ignore.extensions,
            globs=self._config.ignore.globs,
        )

        # Processor with validator and notifier callback
        self._processor = Processor(
            queue=self._queue,
            ignore_config=ignore_config,
            debounce_seconds=self._config.pipeline.debounce_seconds,
            cooldown_seconds=self._config.pipeline.cooldown_seconds,
            stability_timeout=self._config.pipeline.stability_timeout,
            stability_interval=self._config.pipeline.stability_interval,
            stability_rounds=self._config.pipeline.stability_rounds,
            validator=self._validator,
            action_engine=self._action_engine,
            on_validation_error=self._on_validation_error,
            on_file_ready=self._on_file_ready,
        )

        # Reconcile Scanner
        self._reconcile = ReconcileScanner(
            watch_paths=self._watch_paths,
            queue=self._queue,
            ignore_filter=IgnoreFilter(ignore_config),
            is_paused=self._processor.is_paused,
            interval_seconds=600.0,
        )
        self._processor.set_on_processed(
            self._reconcile.mark_processed,
        )

        # Wire stats collector AFTER set_on_processed
        # so it wraps the reconcile callback, not the
        # other way around.
        wire_stats_to_processor(
            self._stats_collector,
            self._processor,
        )

        # Watchdog Observer
        self._observer = create_observer(
            watch_paths=self._watch_paths,
            queue=self._queue,
            is_paused=self._processor.is_paused,
            on_overflow=self._on_queue_overflow,
            recursive=self._config.watcher.recursive,
            use_polling=self._config.watcher.use_polling,
            poll_interval=self._config.watcher.poll_interval,
        )

        self._processor_thread: threading.Thread | None = None
        self._running = threading.Event()

        # Store module references for start/stop
        self._start_observer = start_observer
        self._stop_observer = stop_observer

    def _on_file_ready(
        self,
        path: Path,
        validation_result: object,
    ) -> None:
        """Handle file ready — record stats and notify plugins."""
        is_valid = True
        message = ""
        if validation_result and hasattr(validation_result, "is_valid"):
            is_valid = validation_result.is_valid
            message = getattr(validation_result, "message", "")

        # Record to persistent store
        try:
            self._stats_store.record_event(
                str(path),
                "PROCESSED",
            )
            self._stats_store.record_validation(
                str(path),
                is_valid,
                message,
            )
        except Exception:
            logger.exception("Stats recording failed: %s", path)

        # Notify plugins
        if is_valid:
            self._plugin_manager.notify_file_ready(
                path,
                validation_result,
            )
        else:
            self._plugin_manager.notify_validation_error(
                path,
                message,
            )

    def _on_validation_error(
        self,
        path: Path,
        message: str,
    ) -> None:
        """Handle validation error - send notification."""
        self._notifier.notify_validation_error(
            filename=path.name,
            reason=message,
            file_path=path,
        )

    def _on_queue_overflow(self) -> None:
        """Handle queue overflow — log and reconcile."""
        try:
            self._stats_store.record_event(
                "system",
                "QUEUE_OVERFLOW",
            )
        except Exception:
            logger.exception("Failed to record overflow")
        self._reconcile.trigger_scan()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    def _apply_logging_config(self) -> None:
        """Apply LoggingConfig settings after config is loaded."""
        log_cfg = self._config.logging
        root = logging.getLogger("filehub")
        root.setLevel(getattr(logging, log_cfg.level.upper(), logging.INFO))

        if log_cfg.file:
            from logging.handlers import RotatingFileHandler

            log_path = Path(log_cfg.file).expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=log_cfg.max_size_mb * 1024 * 1024,
                backupCount=log_cfg.backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            root.addHandler(file_handler)
            logger.info(_("File logging enabled: %s"), log_path)

    def start(self, blocking: bool = True) -> None:
        """Start the application.

        Args:
            blocking: If True, block main thread until stopped.
        """
        logger.info(_("Application starting"))
        self._running.set()

        # Notify plugins of startup
        self._plugin_manager.notify_startup()

        # Only handle signals if blocking (running on main thread)
        if blocking:
            signal.signal(signal.SIGINT, self._signal_handler)
            if sys.platform != "win32":
                signal.signal(signal.SIGTERM, self._signal_handler)

        # Start Observer
        self._start_observer(self._observer)

        # Start Processor Thread
        self._processor_thread = threading.Thread(
            target=self._processor.run, name="Processor", daemon=True
        )
        self._processor_thread.start()

        # Start Reconcile Scanner
        self._reconcile.start()

        logger.info("FileHub v%s running. Press Ctrl+C to stop.", __version__)

        if blocking:
            # Keep running
            try:
                while self._running.is_set():
                    threading.Event().wait(1.0)
            except KeyboardInterrupt:
                pass
            finally:
                self._cleanup()

    def stop(self) -> None:
        """Stop the application."""
        if not self._running.is_set():
            return

        logger.info(_("Application shutdown started"))
        self._running.clear()

        # Stop Processor
        self._processor.stop()

        # Stop Observer
        self._stop_observer(self._observer)

        # Stop Reconcile Scanner
        self._reconcile.stop()

        # Plugin & stats cleanup
        self._plugin_manager.notify_shutdown()
        self._stats_store.close()

    def pause(self) -> None:
        """Pause file watching."""
        logger.info(_("Processor paused"))
        self._processor.pause()

    def resume(self) -> None:
        """Resume file watching."""
        logger.info(_("Processor resumed"))
        self._processor.resume()

    def _cleanup(self) -> None:
        """Cleanup resources."""
        if self._processor_thread and self._processor_thread.is_alive():
            self._processor.stop()
            self._processor_thread.join(timeout=5.0)
        self._stop_observer(self._observer)
        self._reconcile.stop()

        # Plugin & stats cleanup
        self._plugin_manager.notify_shutdown()
        self._stats_store.close()

        logger.info(_("Application shutdown complete"))

    def _signal_handler(self, signum: int, frame) -> None:
        logger.info(_("Signal received: %s"), signum)
        self.stop()


def run_app(
    watch_paths: list[str] | None = None,
    config_path: str | None = None,
    target_root: Path | None = None,
) -> int:
    """Run the FileHub application.

    Args:
        watch_paths: Paths to monitor
        config_path: Configuration file path
        target_root: Root directory for action rule targets

    Returns:
        Exit code
    """
    try:
        paths = [Path(p) for p in watch_paths] if watch_paths else None
        app = Application(watch_paths=paths, config_path=config_path, target_root=target_root)

        # Check tray configuration
        use_tray = getattr(app._config, "tray", None) and app._config.tray.minimize_to_tray

        # Import SystemTray here to avoid circular dependencies
        try:
            from .ui.tray import TRAY_AVAILABLE, SystemTray
        except ImportError:
            TRAY_AVAILABLE = False

        if use_tray and TRAY_AVAILABLE:
            logger.info(_("Starting in Tray Mode"))

            def on_exit():
                logger.debug("Tray exit requested")
                app.stop()

            def on_open_config():
                # TODO: Implement config opening
                pass

            tray = SystemTray(
                app_name=app._config.tray.tooltip,
                on_exit=on_exit,
                on_pause=app.pause,
                on_resume=app.resume,
                on_open_config=on_open_config,
            )

            # Start App (Background)
            app.start(blocking=False)

            # Start Tray (Blocking Main Thread)
            tray.run()

            # Ensure app cleanup
            app.stop()
            app._cleanup()
        else:
            logger.info(_("Starting in Console Mode"))
            app.start(blocking=True)

        return 0
    except FileNotFoundError as e:
        print(_("Configuration file error: %s") % e, file=sys.stderr)
        return 1
    except Exception as e:
        print(_("Application error: %s") % e, file=sys.stderr)
        logger.exception(_("Application error: %s"), e)
        return 1
