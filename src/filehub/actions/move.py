"""Move action - automatically move files to target directories."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .base import ActionResult, FileAction

logger = logging.getLogger("filehub")


class MoveAction(FileAction):
    """Move files to a target directory based on rules.

    Can organize files by extension, matched groups, or custom mapping.
    """

    def __init__(
        self,
        target_dir: str | Path,
        create_dirs: bool = True,
        organize_by: str | None = None,
        extension_map: dict[str, str] | None = None,
        dry_run: bool = False,
    ):
        """Initialize move action.

        Args:
            target_dir: Base target directory
            create_dirs: Create target directories if they don't exist
            organize_by: Organize by field from validation groups (e.g., "type", "project")
            extension_map: Map extensions to subdirectories (e.g., {".pdf": "documents"})
            dry_run: If True, only log what would happen
        """
        super().__init__(dry_run=dry_run)
        self._target_dir = Path(target_dir)
        self._create_dirs = create_dirs
        self._organize_by = organize_by
        self._extension_map = extension_map or {}

    @property
    def name(self) -> str:
        return "move"

    def execute(self, path: Path, **context) -> ActionResult:
        """Move a file to the target directory."""
        if not path.exists():
            return ActionResult(success=False, source=path, message=f"File not found: {path}")

        # Determine destination directory
        dest_dir = self._resolve_target_dir(path, context)

        destination = dest_dir / path.name

        # Already in target
        if destination == path:
            return ActionResult(
                success=True,
                source=path,
                destination=path,
                message="File already in target directory",
                dry_run=self._dry_run,
            )

        # Avoid overwriting
        if destination.exists():
            destination = self._unique_path(destination)

        if self._dry_run:
            logger.info("[DRY RUN] Would move: %s -> %s", path, destination)
            return ActionResult(
                success=True,
                source=path,
                destination=destination,
                message=f"Would move to {destination}",
                dry_run=True,
            )

        try:
            if self._create_dirs:
                dest_dir.mkdir(parents=True, exist_ok=True)

            shutil.move(str(path), str(destination))
            logger.info("Moved: %s -> %s", path, destination)
            return ActionResult(
                success=True,
                source=path,
                destination=destination,
                message=f"Moved to {destination}",
            )
        except OSError as e:
            logger.error("Move failed: %s -> %s: %s", path, destination, e)
            return ActionResult(success=False, source=path, message=f"Move failed: {e}")

    def _resolve_target_dir(self, path: Path, context: dict) -> Path:
        """Resolve the target directory for a file."""
        base = self._target_dir

        # Organize by validation group
        if self._organize_by:
            validation_result = context.get("validation_result")
            if validation_result and validation_result.matched_groups:
                group_value = validation_result.matched_groups.get(self._organize_by, "")
                if group_value:
                    safe_value = group_value.replace("/", "").replace("\\", "").replace("..", "")
                    target = base / safe_value
                    try:
                        if not target.resolve().is_relative_to(base.resolve()):
                            logger.warning("Path traversal blocked in move: '%s'", target)
                            return base
                    except (OSError, ValueError):
                        return base
                    return Path(target)

        # Organize by extension
        ext = path.suffix.lower()
        if ext in self._extension_map:
            return base / self._extension_map[ext]

        return base

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
