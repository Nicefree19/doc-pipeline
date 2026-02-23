"""Ignore filter for file events."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import FileEventDTO


@dataclass
class IgnoreConfig:
    """Ignore filter configuration."""

    prefixes: list[str] = field(default_factory=list)
    extensions: list[str] = field(default_factory=list)
    globs: list[str] = field(default_factory=list)


class IgnoreFilter:
    """File ignore filter."""

    def __init__(self, config: IgnoreConfig):
        self._prefixes = tuple(config.prefixes)
        self._extensions = frozenset(e.lower() for e in config.extensions)
        self._globs = list(config.globs)

    def should_ignore(self, event: FileEventDTO) -> bool:
        """Return True if should be ignored."""
        return self._check_prefix(event) or self._check_extension(event) or self._check_glob(event)

    def _check_prefix(self, event: FileEventDTO) -> bool:
        if not self._prefixes:
            return False
        return event.filename.startswith(self._prefixes)

    def _check_extension(self, event: FileEventDTO) -> bool:
        if not self._extensions:
            return False
        return event.extension.lower() in self._extensions

    def _check_glob(self, event: FileEventDTO) -> bool:
        if not self._globs:
            return False

        path_str = str(event.path)
        filename = event.filename

        for pattern in self._globs:
            if fnmatch.fnmatch(path_str, pattern):
                return True
            if fnmatch.fnmatch(filename, pattern):
                return True
            if "**" in pattern:
                simple = pattern.replace("**/", "").replace("**", "*")
                if fnmatch.fnmatch(filename, simple):
                    return True

        return False
