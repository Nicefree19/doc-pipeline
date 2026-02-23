"""Event Processor: Queue Consumer.

watchdog(Producer) ??Queue ??Processor(Consumer)
Processor = Aggregator + Stability + Validator + Notifier
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any

from ...i18n import _

if TYPE_CHECKING:
    from ...actions.engine import ActionEngine
from ..models import AggregatorState, EventType, FileEventDTO
from .aggregator import Aggregator
from .ignore_filter import IgnoreConfig, IgnoreFilter
from .stability import StabilityChecker

logger = logging.getLogger("filehub")


class Processor:
    """Event processor (Consumer)."""

    def __init__(
        self,
        queue: Queue[FileEventDTO],
        ignore_config: IgnoreConfig | None = None,
        debounce_seconds: float = 1.0,
        stability_timeout: float = 20.0,
        stability_interval: float = 0.5,
        stability_rounds: int = 2,
        cooldown_seconds: float = 1800.0,
        validator: Any | None = None,
        action_engine: ActionEngine | None = None,
        on_processed: Callable[[Path], None] | None = None,
        on_validation_error: Callable[[Path, str], None] | None = None,
        on_file_ready: Callable[[Path, Any], None] | None = None,
    ):
        self._queue = queue
        self._validator = validator
        self._action_engine = action_engine
        self._on_processed = on_processed
        self._on_validation_error = on_validation_error
        self._on_file_ready = on_file_ready

        # Components
        self._ignore = IgnoreFilter(ignore_config or IgnoreConfig())
        self._aggregator = Aggregator(
            debounce_seconds=debounce_seconds, cooldown_seconds=cooldown_seconds
        )
        self._stability = StabilityChecker(
            timeout=stability_timeout, interval=stability_interval, required_rounds=stability_rounds
        )

        # Thread control
        self._running = threading.Event()
        self._paused = threading.Event()
        self._last_cleanup = 0.0

        # Health tracking
        self._healthy = True
        self._error_count = 0

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    @property
    def error_count(self) -> int:
        return self._error_count

    def run(self) -> None:
        """Processing loop (runs in separate thread)."""
        self._running.set()
        logger.info(_("Processor started"))

        while self._running.is_set():
            if self._paused.is_set():
                time.sleep(0.1)
                continue

            try:
                self._process_cycle()
            except Exception as e:
                self._error_count += 1
                logger.exception("Processor cycle error (count=%d): %s", self._error_count, e)
                time.sleep(1.0)
                if self._error_count >= 10:
                    self._healthy = False
                    logger.error("Processor unhealthy after %d errors", self._error_count)

        logger.info(_("Processor stopped"))

    def stop(self) -> None:
        """Stop processor."""
        self._running.clear()

    def pause(self) -> None:
        """Pause processor."""
        self._paused.set()
        logger.info(_("Processor paused"))

    def resume(self) -> None:
        """Resume processor."""
        self._paused.clear()
        logger.info(_("Processor resumed"))

    def is_paused(self) -> bool:
        """Check if paused."""
        return self._paused.is_set()

    def set_on_processed(self, callback: Callable[[Path], None] | None) -> None:
        """Set processed callback."""
        self._on_processed = callback

    def _process_cycle(self) -> None:
        """Process one cycle."""
        # 1. Collect events from queue
        self._collect_from_queue()

        # 2. Process due items
        self._process_due_items()

        # 3. Periodic cleanup of expired cooldown states
        now = time.time()
        if now - self._last_cleanup >= 300.0:
            cleaned = self._aggregator.cleanup_expired(now)
            if cleaned:
                logger.debug("Cleaned up %d expired aggregator states", cleaned)
            self._last_cleanup = now

        time.sleep(0.05)

    def _collect_from_queue(self) -> None:
        """Collect events from queue."""
        collected = 0
        max_per_cycle = 50

        while collected < max_per_cycle:
            try:
                event = self._queue.get_nowait()
            except Empty:
                break

            collected += 1

            # Ignore filter
            if self._ignore.should_ignore(event):
                logger.debug("Ignored: %s", event.filename)
                continue

            # Deleted event
            if event.event_type == EventType.DELETED:
                self._aggregator.remove(event.path)
                continue

            # Add to aggregator
            self._aggregator.add_event(event)

    def _process_due_items(self) -> None:
        """Process items past due_time."""
        current = time.time()

        for state in self._aggregator.get_due_items(current):
            try:
                # Check cooldown
                if self._aggregator.is_in_cooldown(state.path, current):
                    logger.debug("In cooldown: %s", state.path)
                    continue

                # Check file exists
                if not state.path.exists():
                    self._aggregator.remove(state.path)
                    continue

                # Ignore directories
                if state.path.is_dir():
                    self._aggregator.remove(state.path)
                    continue

                # Stability check
                if state.state == AggregatorState.STABILITY_CHECK:
                    if self._stability.is_timed_out(state, current):
                        logger.warning(_("Stability timeout: %s"), state.path)
                        self._aggregator.remove(state.path)
                        continue

                    if not self._stability.check(state):
                        # Reschedule
                        self._aggregator.reschedule(state, self._stability.interval)
                        continue
                    else:
                        # Stability confirmed
                        self._aggregator.mark_ready(state)

                # READY state: validate
                if state.state == AggregatorState.READY:
                    self._handle_ready(state)
            except Exception as e:
                self._error_count += 1
                logger.exception("Error processing %s: %s", state.path, e)
                try:
                    self._aggregator.remove(state.path)
                except Exception:
                    pass

    def _handle_ready(self, state) -> None:
        """Handle ready state - validate file, execute actions, and notify."""
        path = state.path

        # Run validation if validator is set
        validation_passed = True
        validation_result = None

        if self._validator:
            try:
                validation_result = self._validator.validate(path)
                if not validation_result.is_valid:
                    validation_passed = False
                    logger.warning(
                        _("Validation failed: %s - %s"), path.name, validation_result.message
                    )

                    if self._on_validation_error:
                        self._on_validation_error(path, validation_result.message)
            except Exception as e:
                logger.error(_("Validation error: %s"), e)

        # Execute Actions
        if self._action_engine:
            try:
                self._action_engine.process(path, validation_result)
            except Exception as e:
                logger.error("Action execution failed for %s: %s", path, e)

        # Mark as processed (with or without validation)
        try:
            self._aggregator.mark_notified(path)
        except Exception as e:
            logger.exception("Failed to mark notified: %s: %s", path, e)

        if not validation_passed:
            logger.info("File entered cooldown after validation failure: %s", path.name)

        # Notify file ready (stats, plugins)
        if self._on_file_ready:
            try:
                self._on_file_ready(path, validation_result)
            except Exception:
                logger.exception(
                    "on_file_ready callback failed: %s",
                    path.name,
                )

        if validation_passed and self._on_processed:
            try:
                self._on_processed(path)
            except Exception:
                logger.exception(
                    "on_processed callback failed for: %s",
                    path.name,
                )
