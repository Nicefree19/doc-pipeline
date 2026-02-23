"""Test fixtures for FileHub."""

import tempfile
from pathlib import Path
from queue import Queue

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def file_queue():
    """Create a queue for file events."""
    return Queue(maxsize=100)


@pytest.fixture
def sample_file(temp_dir):
    """Create a sample test file."""
    file_path = temp_dir / "test_file.txt"
    file_path.write_text("test content")
    return file_path
