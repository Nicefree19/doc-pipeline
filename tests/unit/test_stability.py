"""Tests for StabilityChecker module."""

import time

from src.filehub.core.models import FileState
from src.filehub.core.pipeline.stability import StabilityChecker, is_file_locked


class TestStabilityChecker:
    """Test StabilityChecker functionality."""

    def test_check_nonexistent_file_returns_false(self, temp_dir):
        """Test that check returns False for nonexistent file."""
        checker = StabilityChecker()
        state = FileState(path=temp_dir / "nonexistent.txt")

        assert checker.check(state) is False

    def test_check_directory_returns_false(self, temp_dir):
        """Test that check returns False for directories."""
        checker = StabilityChecker()
        state = FileState(path=temp_dir)

        assert checker.check(state) is False

    def test_check_empty_file_returns_false(self, temp_dir):
        """Test that check returns False for empty files."""
        checker = StabilityChecker()
        empty_file = temp_dir / "empty.txt"
        empty_file.touch()

        state = FileState(path=empty_file)

        assert checker.check(state) is False

    def test_first_check_returns_false(self, sample_file):
        """Test that first check always returns False."""
        checker = StabilityChecker()
        state = FileState(path=sample_file)

        # First check should return False (needs at least 2 rounds)
        assert checker.check(state) is False
        assert state.stability_rounds == 1

    def test_stable_after_required_rounds(self, sample_file):
        """Test that file is stable after required rounds."""
        checker = StabilityChecker(required_rounds=2)
        state = FileState(path=sample_file)

        # Round 1
        assert checker.check(state) is False
        assert state.stability_rounds == 1

        # Round 2 (stable)
        assert checker.check(state) is True
        assert state.stability_rounds == 2

    def test_changes_reset_rounds(self, temp_dir):
        """Test that file changes reset stability rounds."""
        checker = StabilityChecker(required_rounds=3)
        test_file = temp_dir / "changing.txt"
        test_file.write_text("initial")

        state = FileState(path=test_file)

        # Round 1
        checker.check(state)
        assert state.stability_rounds == 1

        # Modify file
        time.sleep(0.01)
        test_file.write_text("modified")

        # Rounds reset
        checker.check(state)
        assert state.stability_rounds == 1

    def test_is_timed_out(self, sample_file):
        """Test timeout detection."""
        checker = StabilityChecker(timeout=1.0)
        state = FileState(path=sample_file)
        state.first_seen = time.time() - 2.0  # 2 seconds ago

        assert checker.is_timed_out(state) is True

    def test_not_timed_out(self, sample_file):
        """Test not timed out."""
        checker = StabilityChecker(timeout=10.0)
        state = FileState(path=sample_file)

        assert checker.is_timed_out(state) is False

    def test_interval_property(self):
        """Test interval property."""
        checker = StabilityChecker(interval=0.5)
        assert checker.interval == 0.5

    def test_timeout_property(self):
        """Test timeout property."""
        checker = StabilityChecker(timeout=30.0)
        assert checker.timeout == 30.0


class TestIsFileLocked:
    """Test is_file_locked function."""

    def test_unlocked_file(self, sample_file):
        """Test that unlocked file returns False."""
        assert is_file_locked(sample_file) is False

    def test_nonexistent_file(self, temp_dir):
        """Test that nonexistent file returns False (not locked, just missing)."""
        result = is_file_locked(temp_dir / "nonexistent.txt")
        assert result is False
