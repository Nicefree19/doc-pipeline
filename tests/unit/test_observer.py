"""Tests for observer module."""

from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.filehub.core.watcher.observer import (
    create_observer,
    is_network_path,
    start_observer,
    stop_observer,
)


class TestCreateObserver:
    """Test create_observer factory function."""

    @patch("src.filehub.core.watcher.observer.Observer")
    def test_returns_observer_instance(self, MockObserver):
        """Test that create_observer returns an Observer."""
        mock_obs = MagicMock()
        MockObserver.return_value = mock_obs

        result = create_observer(
            watch_paths=[],
            queue=Queue(),
            is_paused=lambda: False,
        )

        assert result is mock_obs

    @patch("src.filehub.core.watcher.observer.Observer")
    def test_schedules_valid_paths(self, MockObserver, tmp_path):
        """Test that valid paths are scheduled on the observer."""
        mock_obs = MagicMock()
        MockObserver.return_value = mock_obs

        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()

        create_observer(
            watch_paths=[watch_dir],
            queue=Queue(),
            is_paused=lambda: False,
            recursive=True,
        )

        mock_obs.schedule.assert_called_once()
        schedule_call = mock_obs.schedule.call_args
        assert schedule_call[0][1] == str(watch_dir)
        assert schedule_call[1]["recursive"] is True

    @patch("src.filehub.core.watcher.observer.Observer")
    def test_skips_nonexistent_paths(self, MockObserver):
        """Test that nonexistent paths are skipped."""
        mock_obs = MagicMock()
        MockObserver.return_value = mock_obs

        create_observer(
            watch_paths=[Path("/nonexistent/path/does/not/exist")],
            queue=Queue(),
            is_paused=lambda: False,
        )

        mock_obs.schedule.assert_not_called()

    @patch("src.filehub.core.watcher.observer.Observer")
    def test_skips_file_paths(self, MockObserver, tmp_path):
        """Test that file paths (not directories) are skipped."""
        mock_obs = MagicMock()
        MockObserver.return_value = mock_obs

        file_path = tmp_path / "somefile.txt"
        file_path.write_text("content")

        create_observer(
            watch_paths=[file_path],
            queue=Queue(),
            is_paused=lambda: False,
        )

        mock_obs.schedule.assert_not_called()

    @patch("src.filehub.core.watcher.observer.Observer")
    def test_empty_paths_list(self, MockObserver):
        """Test create_observer with empty paths list."""
        mock_obs = MagicMock()
        MockObserver.return_value = mock_obs

        result = create_observer(
            watch_paths=[],
            queue=Queue(),
            is_paused=lambda: False,
        )

        mock_obs.schedule.assert_not_called()
        assert result is mock_obs

    @patch("src.filehub.core.watcher.observer.Observer")
    def test_multiple_valid_paths(self, MockObserver, tmp_path):
        """Test scheduling multiple valid paths."""
        mock_obs = MagicMock()
        MockObserver.return_value = mock_obs

        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        create_observer(
            watch_paths=[dir1, dir2],
            queue=Queue(),
            is_paused=lambda: False,
        )

        assert mock_obs.schedule.call_count == 2

    @patch("src.filehub.core.watcher.observer.PollingObserver")
    @patch("src.filehub.core.watcher.observer.Observer")
    def test_uses_polling_observer_when_requested(self, MockObserver, MockPolling):
        """Test that PollingObserver is used when use_polling=True."""
        mock_polling = MagicMock()
        MockPolling.return_value = mock_polling

        result = create_observer(
            watch_paths=[],
            queue=Queue(),
            is_paused=lambda: False,
            use_polling=True,
            poll_interval=3.0,
        )

        MockPolling.assert_called_once_with(timeout=3.0)
        MockObserver.assert_not_called()
        assert result is mock_polling

    @patch("src.filehub.core.watcher.observer.Observer")
    def test_uses_native_observer_by_default(self, MockObserver):
        """Test that native Observer is used when use_polling=False."""
        mock_obs = MagicMock()
        MockObserver.return_value = mock_obs

        result = create_observer(
            watch_paths=[],
            queue=Queue(),
            is_paused=lambda: False,
            use_polling=False,
        )

        MockObserver.assert_called_once()
        assert result is mock_obs

    @patch("src.filehub.core.watcher.observer.PollingObserver")
    @patch("src.filehub.core.watcher.observer.Observer")
    def test_polling_observer_with_custom_interval(self, MockObserver, MockPolling):
        """Test that PollingObserver respects custom poll_interval."""
        mock_polling = MagicMock()
        MockPolling.return_value = mock_polling

        create_observer(
            watch_paths=[],
            queue=Queue(),
            is_paused=lambda: False,
            use_polling=True,
            poll_interval=5.0,
        )

        MockPolling.assert_called_once_with(timeout=5.0)

    @patch("src.filehub.core.watcher.observer.PollingObserver")
    @patch("src.filehub.core.watcher.observer.Observer")
    def test_polling_observer_schedules_paths(self, MockObserver, MockPolling, tmp_path):
        """Test that PollingObserver schedules watch paths correctly."""
        mock_polling = MagicMock()
        MockPolling.return_value = mock_polling

        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()

        create_observer(
            watch_paths=[watch_dir],
            queue=Queue(),
            is_paused=lambda: False,
            use_polling=True,
            poll_interval=2.0,
        )

        mock_polling.schedule.assert_called_once()
        schedule_call = mock_polling.schedule.call_args
        assert schedule_call[0][1] == str(watch_dir)

    @patch("src.filehub.core.watcher.observer.PollingObserver")
    @patch("src.filehub.core.watcher.observer.Observer")
    @patch("src.filehub.core.watcher.observer.is_network_path", return_value=True)
    def test_auto_enables_polling_for_network_paths(
        self,
        _mock_is_network,
        MockObserver,
        MockPolling,
        tmp_path,
    ):
        """Auto-detection should switch to PollingObserver for network paths."""
        mock_polling = MagicMock()
        MockPolling.return_value = mock_polling

        watch_dir = tmp_path / "network_like"
        watch_dir.mkdir()

        result = create_observer(
            watch_paths=[watch_dir],
            queue=Queue(),
            is_paused=lambda: False,
            use_polling=False,
            poll_interval=4.0,
        )

        MockPolling.assert_called_once_with(timeout=4.0)
        MockObserver.assert_not_called()
        assert result is mock_polling


class TestStartObserver:
    """Test start_observer function."""

    def test_start_calls_start(self):
        """Test that start_observer calls observer.start()."""
        mock_obs = MagicMock()
        mock_obs.is_alive.return_value = False

        start_observer(mock_obs)

        mock_obs.start.assert_called_once()

    def test_start_skips_if_already_alive(self):
        """Test that start_observer skips if already running."""
        mock_obs = MagicMock()
        mock_obs.is_alive.return_value = True

        start_observer(mock_obs)

        mock_obs.start.assert_not_called()


class TestStopObserver:
    """Test stop_observer function."""

    def test_stop_calls_stop_and_join(self):
        """Test that stop_observer calls stop() and join()."""
        mock_obs = MagicMock()
        mock_obs.is_alive.return_value = True

        stop_observer(mock_obs, timeout=3.0)

        mock_obs.stop.assert_called_once()
        mock_obs.join.assert_called_once_with(timeout=3.0)

    def test_stop_skips_if_not_alive(self):
        """Test that stop_observer skips if not running."""
        mock_obs = MagicMock()
        mock_obs.is_alive.return_value = False

        stop_observer(mock_obs)

        mock_obs.stop.assert_not_called()
        mock_obs.join.assert_not_called()


class TestIsNetworkPath:
    """Test is_network_path function."""

    def test_unc_path_backslash(self):
        """Test UNC path with backslashes."""
        assert is_network_path(Path("\\\\server\\share\\folder")) is True

    def test_unc_path_forward_slash(self):
        """Test UNC path with forward slashes."""
        assert is_network_path(Path("//server/share/folder")) is True

    def test_local_path(self):
        """Test local path is not network."""
        assert is_network_path(Path("C:\\Users\\user\\Documents")) is False

    def test_relative_path(self):
        """Test relative path is not network."""
        assert is_network_path(Path("./some/folder")) is False

    @patch("src.filehub.core.watcher.observer.sys.platform", "win32")
    def test_windows_mapped_network_drive(self):
        """Mapped remote drive on Windows should be treated as network path."""
        fake_ctypes = SimpleNamespace(
            windll=SimpleNamespace(
                kernel32=SimpleNamespace(GetDriveTypeW=lambda _drive: 4),  # DRIVE_REMOTE
            )
        )
        with patch.dict("sys.modules", {"ctypes": fake_ctypes}):
            assert is_network_path(Path("Z:\\Projects\\P5")) is True

    @patch("src.filehub.core.watcher.observer.sys.platform", "win32")
    def test_windows_local_drive_not_network(self):
        """Fixed local drive on Windows should not be treated as network path."""
        fake_ctypes = SimpleNamespace(
            windll=SimpleNamespace(
                kernel32=SimpleNamespace(GetDriveTypeW=lambda _drive: 3),  # DRIVE_FIXED
            )
        )
        with patch.dict("sys.modules", {"ctypes": fake_ctypes}):
            assert is_network_path(Path("C:\\Users\\user\\Documents")) is False
