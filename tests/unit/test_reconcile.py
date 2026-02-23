"""Tests for ReconcileScanner module."""

import time
from queue import Queue

from src.filehub.core.pipeline.ignore_filter import IgnoreConfig, IgnoreFilter
from src.filehub.core.watcher.reconcile import ReconcileScanner


def _make_scanner(
    watch_paths=None,
    queue=None,
    ignore_filter=None,
    is_paused=None,
    interval_seconds=600.0,
):
    """Helper to create a ReconcileScanner."""
    if watch_paths is None:
        watch_paths = []
    if queue is None:
        queue = Queue(maxsize=1000)
    if ignore_filter is None:
        ignore_filter = IgnoreFilter(IgnoreConfig())
    if is_paused is None:

        def is_paused():
            return False
    return ReconcileScanner(
        watch_paths=watch_paths,
        queue=queue,
        ignore_filter=ignore_filter,
        is_paused=is_paused,
        interval_seconds=interval_seconds,
    )


class TestReconcileScanner:
    """Test ReconcileScanner functionality."""

    def test_scan_with_empty_paths(self):
        """Test scan with empty watch paths does nothing."""
        queue = Queue(maxsize=100)
        scanner = _make_scanner(watch_paths=[], queue=queue)

        scanner.scan()

        assert queue.qsize() == 0

    def test_scan_finds_new_files(self, tmp_path):
        """Test scan discovers new files and enqueues them."""
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()
        (watch_dir / "file1.txt").write_text("content1")
        (watch_dir / "file2.txt").write_text("content2")

        queue = Queue(maxsize=100)
        ignore_filter = IgnoreFilter(IgnoreConfig(prefixes=[], extensions=[], globs=[]))
        scanner = _make_scanner(
            watch_paths=[watch_dir],
            queue=queue,
            ignore_filter=ignore_filter,
        )

        scanner.scan()

        assert queue.qsize() == 2

    def test_scan_skips_already_processed_files(self, tmp_path):
        """Test scan skips files that have already been processed."""
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()
        test_file = watch_dir / "file.txt"
        test_file.write_text("content")

        queue = Queue(maxsize=100)
        ignore_filter = IgnoreFilter(IgnoreConfig(prefixes=[], extensions=[], globs=[]))
        scanner = _make_scanner(
            watch_paths=[watch_dir],
            queue=queue,
            ignore_filter=ignore_filter,
        )

        # First scan finds the file
        scanner.scan()
        assert queue.qsize() == 1

        # Drain the queue
        while not queue.empty():
            queue.get_nowait()

        # Second scan should skip the already-processed file
        scanner.scan()
        assert queue.qsize() == 0

    def test_scan_detects_changed_files(self, tmp_path):
        """Test scan detects files that have changed since last scan."""
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()
        test_file = watch_dir / "file.txt"
        test_file.write_text("original")

        queue = Queue(maxsize=100)
        ignore_filter = IgnoreFilter(IgnoreConfig(prefixes=[], extensions=[], globs=[]))
        scanner = _make_scanner(
            watch_paths=[watch_dir],
            queue=queue,
            ignore_filter=ignore_filter,
        )

        # First scan
        scanner.scan()
        assert queue.qsize() == 1
        while not queue.empty():
            queue.get_nowait()

        # Modify the file (change content and ensure mtime changes)
        time.sleep(0.05)
        test_file.write_text("modified content that is different")

        # Second scan should detect the change
        scanner.scan()
        assert queue.qsize() == 1

    def test_scan_applies_ignore_filter(self, tmp_path):
        """Test scan applies ignore filter to skip ignored files."""
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()
        (watch_dir / "normal.txt").write_text("content")
        (watch_dir / "ignored.tmp").write_text("temp")

        queue = Queue(maxsize=100)
        ignore_filter = IgnoreFilter(
            IgnoreConfig(prefixes=[], extensions=[".tmp"], globs=[])
        )
        scanner = _make_scanner(
            watch_paths=[watch_dir],
            queue=queue,
            ignore_filter=ignore_filter,
        )

        scanner.scan()

        # Only normal.txt should be enqueued, not ignored.tmp
        assert queue.qsize() == 1
        dto = queue.get_nowait()
        assert dto.path.name == "normal.txt"

    def test_stale_entry_removal(self, tmp_path):
        """Test that stale entries are removed from processed cache."""
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()
        test_file = watch_dir / "file.txt"
        test_file.write_text("content")

        queue = Queue(maxsize=100)
        ignore_filter = IgnoreFilter(IgnoreConfig(prefixes=[], extensions=[], globs=[]))
        scanner = _make_scanner(
            watch_paths=[watch_dir],
            queue=queue,
            ignore_filter=ignore_filter,
        )

        # First scan - file gets processed
        scanner.scan()
        assert test_file in scanner._processed

        # Delete the file
        test_file.unlink()

        # Drain queue
        while not queue.empty():
            queue.get_nowait()

        # Second scan - stale entry should be removed
        scanner.scan()
        assert test_file not in scanner._processed

    def test_mark_processed(self, tmp_path):
        """Test mark_processed updates the processed cache."""
        scanner = _make_scanner()

        test_file = tmp_path / "file.txt"
        test_file.write_text("content")

        scanner.mark_processed(test_file)

        assert test_file in scanner._processed
        size, mtime = scanner._processed[test_file]
        assert size == test_file.stat().st_size
        assert mtime == test_file.stat().st_mtime

    def test_mark_processed_nonexistent_file(self, tmp_path):
        """Test mark_processed with nonexistent file does nothing."""
        scanner = _make_scanner()

        nonexistent = tmp_path / "does_not_exist.txt"
        scanner.mark_processed(nonexistent)

        assert nonexistent not in scanner._processed

    def test_trigger_scan(self):
        """Test trigger_scan sets the trigger event."""
        scanner = _make_scanner()

        scanner.trigger_scan(reason="overflow")

        assert scanner._trigger_event.is_set()

    def test_start_stop_lifecycle(self):
        """Test start and stop lifecycle."""
        scanner = _make_scanner(interval_seconds=9999)

        scanner.start()
        assert scanner._thread is not None
        assert scanner._thread.is_alive()

        scanner.stop()
        assert not scanner._thread.is_alive()

    def test_start_idempotent(self):
        """Test that calling start twice does not create duplicate threads."""
        scanner = _make_scanner(interval_seconds=9999)

        scanner.start()
        first_thread = scanner._thread

        scanner.start()
        second_thread = scanner._thread

        assert first_thread is second_thread

        scanner.stop()

    def test_scan_skips_directories(self, tmp_path):
        """Test that scan only processes files, not directories."""
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()
        sub_dir = watch_dir / "subdir"
        sub_dir.mkdir()
        (watch_dir / "file.txt").write_text("content")

        queue = Queue(maxsize=100)
        ignore_filter = IgnoreFilter(IgnoreConfig(prefixes=[], extensions=[], globs=[]))
        scanner = _make_scanner(
            watch_paths=[watch_dir],
            queue=queue,
            ignore_filter=ignore_filter,
        )

        scanner.scan()

        # Only the file should be enqueued, not the directory
        assert queue.qsize() == 1
        dto = queue.get_nowait()
        assert dto.path.name == "file.txt"

    def test_scan_nonexistent_watch_path(self, tmp_path):
        """Test scan handles nonexistent watch paths gracefully."""
        queue = Queue(maxsize=100)
        scanner = _make_scanner(
            watch_paths=[tmp_path / "nonexistent"],
            queue=queue,
        )

        scanner.scan()

        assert queue.qsize() == 0
