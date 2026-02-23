"""Tests for IgnoreFilter module."""

import time
from pathlib import Path

from src.filehub.core.models import EventType, FileEventDTO
from src.filehub.core.pipeline.ignore_filter import IgnoreConfig, IgnoreFilter


def _make_event(path_str: str) -> FileEventDTO:
    """Helper to create a FileEventDTO from a path string."""
    return FileEventDTO(
        path=Path(path_str),
        event_type=EventType.CREATED,
        timestamp=time.time(),
    )


class TestIgnoreConfig:
    """Test IgnoreConfig dataclass."""

    def test_default_values(self):
        """Test that IgnoreConfig has empty defaults."""
        cfg = IgnoreConfig()
        assert cfg.prefixes == []
        assert cfg.extensions == []
        assert cfg.globs == []

    def test_custom_values(self):
        """Test creating IgnoreConfig with custom values."""
        cfg = IgnoreConfig(
            prefixes=["~$", "."],
            extensions=[".tmp"],
            globs=["**/desktop.ini"],
        )
        assert cfg.prefixes == ["~$", "."]
        assert cfg.extensions == [".tmp"]
        assert cfg.globs == ["**/desktop.ini"]


class TestIgnoreFilter:
    """Test IgnoreFilter functionality."""

    # --- Prefix matching ---

    def test_prefix_match_tilde_dollar(self):
        """Test that files starting with ~$ are ignored."""
        filt = IgnoreFilter(IgnoreConfig(prefixes=["~$"]))
        event = _make_event("/docs/~$document.docx")
        assert filt.should_ignore(event) is True

    def test_prefix_match_dot(self):
        """Test that files starting with . are ignored."""
        filt = IgnoreFilter(IgnoreConfig(prefixes=["."]))
        event = _make_event("/project/.gitignore")
        assert filt.should_ignore(event) is True

    def test_prefix_no_match(self):
        """Test that files not matching prefix are not ignored."""
        filt = IgnoreFilter(IgnoreConfig(prefixes=["~$", "."]))
        event = _make_event("/docs/report.docx")
        assert filt.should_ignore(event) is False

    def test_prefix_checks_filename_not_path(self):
        """Test that prefix matching is against filename, not full path."""
        filt = IgnoreFilter(IgnoreConfig(prefixes=["~$"]))
        event = _make_event("/~$folder/normalfile.txt")
        assert filt.should_ignore(event) is False

    def test_empty_prefix_list(self):
        """Test that empty prefix list ignores nothing."""
        filt = IgnoreFilter(IgnoreConfig(prefixes=[]))
        event = _make_event("/docs/~$document.docx")
        assert filt.should_ignore(event) is False

    # --- Extension matching ---

    def test_extension_match_tmp(self):
        """Test that .tmp files are ignored."""
        filt = IgnoreFilter(IgnoreConfig(extensions=[".tmp"]))
        event = _make_event("/temp/data.tmp")
        assert filt.should_ignore(event) is True

    def test_extension_case_insensitive(self):
        """Test that extension matching is case-insensitive."""
        filt = IgnoreFilter(IgnoreConfig(extensions=[".tmp"]))
        event = _make_event("/temp/data.TMP")
        assert filt.should_ignore(event) is True

    def test_extension_case_insensitive_config(self):
        """Test that config extensions in uppercase are handled."""
        filt = IgnoreFilter(IgnoreConfig(extensions=[".TMP"]))
        event = _make_event("/temp/data.tmp")
        assert filt.should_ignore(event) is True

    def test_extension_no_match(self):
        """Test that non-matching extensions are not ignored."""
        filt = IgnoreFilter(IgnoreConfig(extensions=[".tmp", ".bak"]))
        event = _make_event("/docs/report.pdf")
        assert filt.should_ignore(event) is False

    def test_empty_extension_list(self):
        """Test that empty extension list ignores nothing."""
        filt = IgnoreFilter(IgnoreConfig(extensions=[]))
        event = _make_event("/temp/data.tmp")
        assert filt.should_ignore(event) is False

    # --- Glob matching ---

    def test_glob_match_desktop_ini(self):
        """Test that **/desktop.ini glob matches."""
        filt = IgnoreFilter(IgnoreConfig(globs=["**/desktop.ini"]))
        event = _make_event("/some/path/desktop.ini")
        assert filt.should_ignore(event) is True

    def test_glob_match_filename(self):
        """Test that glob matches against filename."""
        filt = IgnoreFilter(IgnoreConfig(globs=["Thumbs.db"]))
        event = _make_event("/photos/Thumbs.db")
        assert filt.should_ignore(event) is True

    def test_glob_no_match(self):
        """Test that non-matching glob patterns are not ignored."""
        filt = IgnoreFilter(IgnoreConfig(globs=["**/desktop.ini"]))
        event = _make_event("/docs/report.docx")
        assert filt.should_ignore(event) is False

    def test_empty_glob_list(self):
        """Test that empty glob list ignores nothing."""
        filt = IgnoreFilter(IgnoreConfig(globs=[]))
        event = _make_event("/some/path/desktop.ini")
        assert filt.should_ignore(event) is False

    def test_glob_wildcard_pattern(self):
        """Test wildcard glob patterns."""
        filt = IgnoreFilter(IgnoreConfig(globs=["*.log"]))
        event = _make_event("/var/log/app.log")
        assert filt.should_ignore(event) is True

    # --- Empty config ---

    def test_empty_config_ignores_nothing(self):
        """Test that empty config ignores no files."""
        filt = IgnoreFilter(IgnoreConfig())
        event = _make_event("/docs/~$document.tmp")
        assert filt.should_ignore(event) is False

    # --- Combined filters ---

    def test_combined_prefix_and_extension(self):
        """Test that prefix match triggers ignore even if extension does not match."""
        filt = IgnoreFilter(IgnoreConfig(prefixes=["~$"], extensions=[".pdf"]))
        event = _make_event("/docs/~$tempfile.docx")
        assert filt.should_ignore(event) is True

    def test_combined_extension_matches_when_prefix_doesnt(self):
        """Test that extension match triggers ignore even if prefix does not match."""
        filt = IgnoreFilter(IgnoreConfig(prefixes=["~$"], extensions=[".tmp"]))
        event = _make_event("/temp/normalfile.tmp")
        assert filt.should_ignore(event) is True

    def test_combined_all_filters(self):
        """Test file matching none of multiple filter types."""
        filt = IgnoreFilter(
            IgnoreConfig(
                prefixes=["~$", "."],
                extensions=[".tmp", ".bak"],
                globs=["**/desktop.ini"],
            )
        )
        event = _make_event("/docs/report.pdf")
        assert filt.should_ignore(event) is False

    def test_combined_glob_match_with_others_empty(self):
        """Test glob match when prefix and extension lists are empty."""
        filt = IgnoreFilter(
            IgnoreConfig(
                prefixes=[],
                extensions=[],
                globs=["**/Thumbs.db"],
            )
        )
        event = _make_event("/photos/Thumbs.db")
        assert filt.should_ignore(event) is True
