"""Rename action - automatically rename files based on rules."""

from __future__ import annotations

import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

from .base import ActionResult, FileAction

logger = logging.getLogger("filehub")


class RenameAction(FileAction):
    """Rename files based on a template pattern.

    Template variables:
        {stem}: Original filename without extension
        {ext}: File extension (with dot)
        {date}: Current date (YYYYMMDD)
        {counter}: Auto-incrementing counter (4 digits)
        {lower}: Lowercase stem
        {upper}: Uppercase stem

    Example templates:
        "{date}_{stem}{ext}" -> "20260212_report.pdf"
        "{lower}{ext}" -> "my_document.pdf"
    """

    def __init__(
        self,
        template: str | None = None,
        pattern: str | None = None,
        replacement: str | None = None,
        dry_run: bool = False,
    ):
        """Initialize rename action.

        Args:
            template: Full filename template (overrides pattern/replacement)
            pattern: Regex pattern to match in filename
            replacement: Replacement string (supports regex groups)
            dry_run: If True, only log what would happen
        """
        super().__init__(dry_run=dry_run)
        self._template = template
        self._pattern = pattern
        self._replacement = replacement
        self._counter = 0

    @property
    def name(self) -> str:
        return "rename"

    def execute(self, path: Path, **context) -> ActionResult:
        """Rename a file."""
        if not path.exists():
            return ActionResult(success=False, source=path, message=f"File not found: {path}")

        try:
            new_name = self._compute_new_name(path)
        except Exception as e:
            return ActionResult(
                success=False, source=path, message=f"Failed to compute new name: {e}"
            )

        if new_name == path.name:
            return ActionResult(
                success=True,
                source=path,
                destination=path,
                message="No rename needed",
                dry_run=self._dry_run,
            )

        destination = path.parent / new_name

        # Avoid overwriting (but allow case-only renames on case-insensitive FS)
        if destination.exists() and not self._is_same_file(path, destination):
            destination = self._unique_path(destination)

        if self._dry_run:
            logger.info("[DRY RUN] Would rename: %s -> %s", path.name, destination.name)
            return ActionResult(
                success=True,
                source=path,
                destination=destination,
                message=f"Would rename to {destination.name}",
                dry_run=True,
            )

        try:
            shutil.move(str(path), str(destination))
            logger.info("Renamed: %s -> %s", path.name, destination.name)
            return ActionResult(
                success=True,
                source=path,
                destination=destination,
                message=f"Renamed to {destination.name}",
            )
        except OSError as e:
            logger.error("Rename failed: %s -> %s: %s", path.name, destination.name, e)
            return ActionResult(success=False, source=path, message=f"Rename failed: {e}")

    def _compute_new_name(self, path: Path) -> str:
        """Compute the new filename."""
        if self._template:
            return self._apply_template(path)
        elif self._pattern and self._replacement is not None:
            return self._apply_regex(path)
        return path.name

    def _apply_template(self, path: Path) -> str:
        """Apply template to generate new name."""
        self._counter += 1
        assert self._template is not None
        variables = {
            "stem": path.stem,
            "ext": path.suffix,
            "date": datetime.now().strftime("%Y%m%d"),
            "counter": f"{self._counter:04d}",
            "lower": path.stem.lower(),
            "upper": path.stem.upper(),
        }
        new_name = self._template.format(**variables)
        return new_name.replace("/", "").replace("\\", "")

    def _apply_regex(self, path: Path) -> str:
        """Apply regex pattern/replacement."""
        assert self._pattern is not None and self._replacement is not None
        new_stem = re.sub(self._pattern, self._replacement, path.stem)
        new_stem = new_stem.replace("/", "").replace("\\", "")
        return new_stem + path.suffix

    @staticmethod
    def _is_same_file(a: Path, b: Path) -> bool:
        """Check if two paths point to the same file (handles case-insensitive FS)."""
        try:
            return a.resolve() == b.resolve() or a.stat().st_ino == b.stat().st_ino
        except OSError:
            return False

    @staticmethod
    def _unique_path(path: Path) -> Path:
        """Generate a unique path by appending a counter."""
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        counter = 1
        while path.exists():
            path = parent / f"{stem}_{counter}{suffix}"
            counter += 1
        return path
