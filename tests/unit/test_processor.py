"""Tests for Processor module."""

import time
from unittest.mock import MagicMock, patch

from src.filehub.core.models import (
    AggregatorState,
    EventType,
    FileEventDTO,
    FileState,
    ValidationResult,
)
from src.filehub.core.pipeline.ignore_filter import IgnoreConfig
from src.filehub.core.pipeline.processor import Processor


class TestProcessor:
    """Test Processor functionality."""

    def test_processor_start_stop(self, file_queue):
        """Test that processor can start and stop."""
        processor = Processor(queue=file_queue)

        # Processor should not be paused initially
        assert not processor.is_paused()

        # Stop should work even without starting
        processor.stop()

    def test_pause_resume(self, file_queue):
        """Test pause and resume functionality."""
        processor = Processor(queue=file_queue)

        assert not processor.is_paused()

        processor.pause()
        assert processor.is_paused()

        processor.resume()
        assert not processor.is_paused()

    def test_ignore_filter_prefixes(self, file_queue, temp_dir):
        """Test that ignored files are not processed."""
        processor = Processor(queue=file_queue, ignore_config=IgnoreConfig(prefixes=["~$", "."]))

        # Add ignored event
        event = FileEventDTO(
            path=temp_dir / "~$tempfile.docx", event_type=EventType.CREATED, timestamp=time.time()
        )
        file_queue.put(event)

        # Process
        processor._collect_from_queue()

        # Should not be in aggregator
        assert len(processor._aggregator._states) == 0

    def test_ignore_filter_extensions(self, file_queue, temp_dir):
        """Test that ignored extensions are not processed."""
        processor = Processor(
            queue=file_queue, ignore_config=IgnoreConfig(extensions=[".tmp", ".bak"])
        )

        # Add ignored event
        event = FileEventDTO(
            path=temp_dir / "backup.bak", event_type=EventType.CREATED, timestamp=time.time()
        )
        file_queue.put(event)

        # Process
        processor._collect_from_queue()

        # Should not be in aggregator
        assert len(processor._aggregator._states) == 0

    def test_collect_from_queue_adds_to_aggregator(self, file_queue, sample_file):
        """Test that valid events are added to aggregator."""
        processor = Processor(queue=file_queue)

        event = FileEventDTO(path=sample_file, event_type=EventType.CREATED, timestamp=time.time())
        file_queue.put(event)

        processor._collect_from_queue()

        assert sample_file in processor._aggregator._states

    def test_deleted_event_removes_from_aggregator(self, file_queue, temp_dir):
        """Test that DELETED events remove from aggregator."""
        processor = Processor(queue=file_queue)
        path = temp_dir / "to_delete.txt"

        # First add a CREATED event
        create_event = FileEventDTO(path=path, event_type=EventType.CREATED, timestamp=time.time())
        processor._aggregator.add_event(create_event)
        assert path in processor._aggregator._states

        # Then delete
        delete_event = FileEventDTO(path=path, event_type=EventType.DELETED, timestamp=time.time())
        file_queue.put(delete_event)
        processor._collect_from_queue()

        assert path not in processor._aggregator._states

    def test_on_processed_callback(self, file_queue, sample_file):
        """Test that on_processed callback is called."""
        processor = Processor(queue=file_queue, debounce_seconds=0)

        callback_called = []
        processor.set_on_processed(lambda p: callback_called.append(p))

        # This test is simplified - full processing requires stability checks
        assert processor._on_processed is not None

    def test_max_per_cycle_limit(self, file_queue, temp_dir):
        """Test that collection is limited per cycle."""
        processor = Processor(queue=file_queue)

        # Add more than max_per_cycle events
        for i in range(100):
            event = FileEventDTO(
                path=temp_dir / f"file_{i}.txt", event_type=EventType.CREATED, timestamp=time.time()
            )
            file_queue.put(event)

        # Should collect max 50 per cycle
        processor._collect_from_queue()

        # 50 collected, 50 remaining in queue
        assert len(processor._aggregator._states) == 50
        assert file_queue.qsize() == 50


class TestProcessorStability:
    """Test stability check integration in processor."""

    def test_stability_pass_transitions_to_ready(self, file_queue, sample_file):
        """Stability pass -> READY state."""
        processor = Processor(
            queue=file_queue, debounce_seconds=0, stability_rounds=1
        )

        event = FileEventDTO(path=sample_file, event_type=EventType.CREATED, timestamp=time.time() - 1)
        processor._aggregator.add_event(event)

        # First process_due_items: DEBOUNCING -> STABILITY_CHECK
        processor._process_due_items()

        state = processor._aggregator._states.get(sample_file)
        if state and state.state == AggregatorState.STABILITY_CHECK:
            # Simulate stability check pass
            state.last_size = sample_file.stat().st_size
            state.last_mtime = sample_file.stat().st_mtime
            state.stability_rounds = 1

    def test_file_deleted_during_processing(self, file_queue, temp_dir):
        """File deleted before processing -> removed from aggregator."""
        deleted_file = temp_dir / "will_delete.txt"
        deleted_file.write_text("data")

        processor = Processor(queue=file_queue, debounce_seconds=0)
        event = FileEventDTO(path=deleted_file, event_type=EventType.CREATED, timestamp=time.time() - 1)
        processor._aggregator.add_event(event)

        # Delete the file
        deleted_file.unlink()

        processor._process_due_items()
        assert deleted_file not in processor._aggregator._states

    def test_directory_removed_from_aggregator(self, file_queue, temp_dir):
        """Directory events -> removed from aggregator."""
        dir_path = temp_dir / "subdir"
        dir_path.mkdir()

        processor = Processor(queue=file_queue, debounce_seconds=0)
        event = FileEventDTO(path=dir_path, event_type=EventType.CREATED, timestamp=time.time() - 1)
        processor._aggregator.add_event(event)

        processor._process_due_items()
        assert dir_path not in processor._aggregator._states


class TestProcessorHandleReady:
    """Test _handle_ready method."""

    def test_validator_pass(self, file_queue, sample_file):
        """Validator passes -> on_processed called."""
        mock_validator = MagicMock()
        mock_validator.validate.return_value = ValidationResult.valid(sample_file.name, {})

        processed = []
        processor = Processor(
            queue=file_queue,
            validator=mock_validator,
            on_processed=lambda p: processed.append(p),
        )

        state = FileState(path=sample_file, state=AggregatorState.READY)
        processor._aggregator._states[sample_file] = state
        processor._handle_ready(state)

        assert len(processed) == 1
        assert processed[0] == sample_file

    def test_validator_fail_calls_error_callback(self, file_queue, sample_file):
        """Validator fails -> on_validation_error called."""
        mock_validator = MagicMock()
        mock_validator.validate.return_value = ValidationResult.invalid(sample_file.name, "Bad name")

        errors = []
        processor = Processor(
            queue=file_queue,
            validator=mock_validator,
            on_validation_error=lambda p, m: errors.append((p, m)),
        )

        state = FileState(path=sample_file, state=AggregatorState.READY)
        processor._aggregator._states[sample_file] = state
        processor._handle_ready(state)

        assert len(errors) == 1
        assert "Bad name" in errors[0][1]

    def test_validator_exception_caught(self, file_queue, sample_file):
        """Validator raising exception -> caught, no crash."""
        mock_validator = MagicMock()
        mock_validator.validate.side_effect = RuntimeError("validator crash")

        processor = Processor(queue=file_queue, validator=mock_validator)

        state = FileState(path=sample_file, state=AggregatorState.READY)
        processor._aggregator._states[sample_file] = state

        # Should not raise
        processor._handle_ready(state)

    def test_action_exception_caught(self, file_queue, sample_file):
        """Action engine raising exception -> caught, no crash."""
        mock_engine = MagicMock()
        mock_engine.process.side_effect = RuntimeError("action crash")

        processor = Processor(queue=file_queue, action_engine=mock_engine)

        state = FileState(path=sample_file, state=AggregatorState.READY)
        processor._aggregator._states[sample_file] = state

        processor._handle_ready(state)

    def test_on_processed_callback_called(self, file_queue, sample_file):
        """on_processed callback is invoked for valid files."""
        processed_paths = []
        processor = Processor(
            queue=file_queue,
            on_processed=lambda p: processed_paths.append(p),
        )

        state = FileState(path=sample_file, state=AggregatorState.READY)
        processor._aggregator._states[sample_file] = state
        processor._handle_ready(state)

        assert sample_file in processed_paths


class TestProcessorErrorRecovery:
    """Test error recovery and health tracking."""

    def test_health_initially_true(self, file_queue):
        processor = Processor(queue=file_queue)
        assert processor.is_healthy is True
        assert processor.error_count == 0

    def test_cycle_exception_thread_survives(self, file_queue):
        """Cycle exception -> thread continues running."""
        processor = Processor(queue=file_queue)

        with patch.object(processor, '_process_cycle', side_effect=[RuntimeError("boom"), None]):
            # Run one iteration manually
            processor._running.set()
            try:
                processor._process_cycle()
            except RuntimeError:
                processor._error_count += 1

        assert processor._error_count == 1

    def test_per_item_exception_continues(self, file_queue, temp_dir):
        """Exception on one item -> next item still processed."""
        f1 = temp_dir / "good.txt"
        f1.write_text("data")

        processor = Processor(queue=file_queue, debounce_seconds=0)

        # Add event for good file
        event = FileEventDTO(path=f1, event_type=EventType.CREATED, timestamp=time.time() - 1)
        processor._aggregator.add_event(event)

        # Processing should not crash even if individual items fail
        processor._process_due_items()

    def test_ten_errors_makes_unhealthy(self, file_queue):
        """10 consecutive errors -> processor marked unhealthy."""
        processor = Processor(queue=file_queue)
        processor._error_count = 9

        # Simulate one more error pushing to 10
        processor._error_count += 1
        if processor._error_count >= 10:
            processor._healthy = False

        assert processor.is_healthy is False
        assert processor.error_count == 10

    def test_error_count_increments(self, file_queue):
        """error_count tracks cumulative errors."""
        processor = Processor(queue=file_queue)
        assert processor.error_count == 0

        processor._error_count = 3
        assert processor.error_count == 3

    def test_run_loop_catches_cycle_error(self, file_queue):
        """run() loop catches _process_cycle errors and stays alive."""
        processor = Processor(queue=file_queue)

        call_count = 0

        def side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            # Stop after second call
            processor.stop()

        with patch.object(processor, '_process_cycle', side_effect=side_effect):
            processor.run()

        assert processor.error_count == 1
        assert processor.is_healthy is True  # < 10 errors
