"""Integration tests for the full pipeline (Step 10).

Uses real files, real Queue, real Processor thread. Minimal mocking.
"""

import threading
import time
from pathlib import Path
from queue import Queue

import pytest
from filehub.actions.engine import ActionEngine
from filehub.actions.models import ActionRule, ActionType, TriggerType
from filehub.core.models import EventType, FileEventDTO, ValidationResult
from filehub.core.pipeline.processor import Processor


@pytest.fixture
def pipeline_workspace(tmp_path):
    """Create workspace with source/target dirs."""
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    return tmp_path, source, target


def _wait_for_file(path: Path, timeout: float = 10.0) -> bool:
    """Wait until a file exists or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        if path.exists():
            return True
        time.sleep(0.1)
    return False


def _wait_for_empty_aggregator(processor: Processor, timeout: float = 10.0) -> bool:
    """Wait until aggregator has no pending items."""
    start = time.time()
    while time.time() - start < timeout:
        if len(processor._aggregator._states) == 0:
            return True
        time.sleep(0.1)
    return False


class TestPipelineIntegration:
    """End-to-end pipeline tests."""

    def test_file_created_moved_to_target(self, pipeline_workspace):
        """File creation -> event -> queue -> processor -> action -> moved."""
        root, source, target = pipeline_workspace

        rule = ActionRule(
            name="Move All",
            action=ActionType.MOVE,
            trigger=TriggerType.ALWAYS,
            target=str(target),
        )
        engine = ActionEngine([rule])

        queue = Queue(maxsize=100)
        processor = Processor(
            queue=queue,
            debounce_seconds=0.1,
            stability_interval=0.1,
            stability_timeout=5.0,
            stability_rounds=1,
            action_engine=engine,
        )

        # Start processor
        t = threading.Thread(target=processor.run, daemon=True)
        t.start()

        try:
            # Create file and enqueue event
            test_file = source / "pipeline_test.txt"
            test_file.write_text("integration test data")

            event = FileEventDTO(
                path=test_file,
                event_type=EventType.CREATED,
                timestamp=time.time(),
            )
            queue.put(event)

            # Wait for processing
            dest = target / "pipeline_test.txt"
            assert _wait_for_file(dest, timeout=10.0), f"File not moved to {dest}"
        finally:
            processor.stop()
            t.join(timeout=5.0)

    def test_invalid_filename_triggers_callback(self, pipeline_workspace):
        """Invalid file -> validator fail -> on_validation_error called."""
        root, source, target = pipeline_workspace

        # Mock validator that always fails
        class AlwaysInvalidValidator:
            def validate(self, path):
                return ValidationResult.invalid(path.name, "Invalid naming")

        errors = []

        def on_error(path, message):
            errors.append((path, message))

        queue = Queue(maxsize=100)
        processor = Processor(
            queue=queue,
            debounce_seconds=0.1,
            stability_interval=0.1,
            stability_timeout=5.0,
            stability_rounds=1,
            validator=AlwaysInvalidValidator(),
            on_validation_error=on_error,
        )

        t = threading.Thread(target=processor.run, daemon=True)
        t.start()

        try:
            test_file = source / "bad_name.txt"
            test_file.write_text("data")

            event = FileEventDTO(
                path=test_file,
                event_type=EventType.CREATED,
                timestamp=time.time(),
            )
            queue.put(event)

            # Wait for error callback
            start = time.time()
            while time.time() - start < 10.0 and not errors:
                time.sleep(0.1)

            assert len(errors) > 0
            assert "Invalid naming" in errors[0][1]
        finally:
            processor.stop()
            t.join(timeout=5.0)

    def test_template_organize_ext_group(self, pipeline_workspace):
        """Template {ext_group} -> file placed in correct subfolder."""
        root, source, target = pipeline_workspace

        rule = ActionRule(
            name="Organize by ext",
            action=ActionType.MOVE,
            trigger=TriggerType.ALWAYS,
            target=str(target / "{ext_group}"),
        )
        engine = ActionEngine([rule])

        queue = Queue(maxsize=100)
        processor = Processor(
            queue=queue,
            debounce_seconds=0.1,
            stability_interval=0.1,
            stability_timeout=5.0,
            stability_rounds=1,
            action_engine=engine,
        )

        t = threading.Thread(target=processor.run, daemon=True)
        t.start()

        try:
            pdf_file = source / "report.pdf"
            pdf_file.write_text("pdf content")

            event = FileEventDTO(
                path=pdf_file,
                event_type=EventType.CREATED,
                timestamp=time.time(),
            )
            queue.put(event)

            dest = target / "documents" / "report.pdf"
            assert _wait_for_file(dest, timeout=10.0)
        finally:
            processor.stop()
            t.join(timeout=5.0)

    def test_multiple_files_batch(self, pipeline_workspace):
        """5 files enqueued -> all processed."""
        root, source, target = pipeline_workspace

        rule = ActionRule(
            name="Move All",
            action=ActionType.MOVE,
            trigger=TriggerType.ALWAYS,
            target=str(target),
        )
        engine = ActionEngine([rule])

        queue = Queue(maxsize=100)
        processor = Processor(
            queue=queue,
            debounce_seconds=0.1,
            stability_interval=0.1,
            stability_timeout=5.0,
            stability_rounds=1,
            action_engine=engine,
        )

        t = threading.Thread(target=processor.run, daemon=True)
        t.start()

        try:
            files = []
            for i in range(5):
                f = source / f"batch_{i}.txt"
                f.write_text(f"content {i}")
                files.append(f)
                event = FileEventDTO(
                    path=f, event_type=EventType.CREATED, timestamp=time.time()
                )
                queue.put(event)

            # Wait for all files
            for i in range(5):
                dest = target / f"batch_{i}.txt"
                assert _wait_for_file(dest, timeout=15.0), f"File batch_{i}.txt not moved"
        finally:
            processor.stop()
            t.join(timeout=5.0)

    def test_stability_recheck_on_modification(self, pipeline_workspace):
        """File modified during stability -> re-checked -> eventually processed."""
        root, source, target = pipeline_workspace

        rule = ActionRule(
            name="Move All",
            action=ActionType.MOVE,
            trigger=TriggerType.ALWAYS,
            target=str(target),
        )
        engine = ActionEngine([rule])

        queue = Queue(maxsize=100)
        processor = Processor(
            queue=queue,
            debounce_seconds=0.2,
            stability_interval=0.2,
            stability_timeout=10.0,
            stability_rounds=2,
            action_engine=engine,
        )

        t = threading.Thread(target=processor.run, daemon=True)
        t.start()

        try:
            test_file = source / "changing.txt"
            test_file.write_text("v1")

            event = FileEventDTO(
                path=test_file,
                event_type=EventType.CREATED,
                timestamp=time.time(),
            )
            queue.put(event)

            # Modify the file shortly after to trigger re-check
            time.sleep(0.3)
            test_file.write_text("v2")
            event2 = FileEventDTO(
                path=test_file,
                event_type=EventType.MODIFIED,
                timestamp=time.time(),
            )
            queue.put(event2)

            dest = target / "changing.txt"
            assert _wait_for_file(dest, timeout=15.0)
            assert dest.read_text() == "v2"
        finally:
            processor.stop()
            t.join(timeout=5.0)
