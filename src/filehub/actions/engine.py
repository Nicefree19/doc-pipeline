"""Action engine for executing rules."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.models import ValidationResult
from .models import ActionRule, ActionType, TriggerType

logger = logging.getLogger("filehub")


class ActionEngine:
    """Executes actions based on rules."""

    _DRAWING_EXTS = {"dwg", "dxf", "dwl", "dwl2"}
    _DOCUMENT_EXTS = {"pdf", "doc", "docx", "txt", "ppt", "pptx", "xls", "xlsx", "rtf", "hwp", "hwpx"}
    _IMAGE_EXTS = {"jpg", "jpeg", "png", "gif", "bmp", "tif", "tiff", "webp"}
    _MODEL_EXTS = {"mgb", "msr", "out", "bak", "inp", "nc1", "ifc", "rvt", "nwd", "nwc"}

    def __init__(
        self,
        rules: list[ActionRule],
        target_root: Path | None = None,
        dry_run: bool = False,
        ext_groups: dict[str, list[str]] | None = None,
        default_ext_group: str = "others",
    ):
        self._rules = rules
        self._target_root = target_root
        self._dry_run = dry_run
        self._custom_ext_groups = self._normalize_ext_groups(ext_groups)
        self._default_ext_group = default_ext_group or "others"

    def process(self, path: Path, validation_result: ValidationResult | None = None) -> bool:
        """Process file against all rules."""
        if not path.exists():
            return False

        executed = False
        for rule in self._rules:
            try:
                if self._should_run(rule, validation_result):
                    did_run = self._execute_rule(rule, path, validation_result)
                    executed = executed or did_run

                    # If file was moved/renamed/deleted, stop processing other rules for this file
                    # unless we implement chaining (which is complex).
                    # For now, assume one successful moving/renaming action per file.
                    if did_run and rule.action in (
                        ActionType.MOVE,
                        ActionType.RENAME,
                        ActionType.DELETE,
                    ):
                        break
            except Exception as e:
                logger.error("Failed to execute action '%s' on '%s': %s", rule.name, path, e)
        return executed

    def _should_run(self, rule: ActionRule, result: ValidationResult | None) -> bool:
        """Check if rule trigger matches."""
        if rule.trigger == TriggerType.ALWAYS:
            return True

        if rule.trigger == TriggerType.VALID:
            return result is not None and result.is_valid

        if rule.trigger == TriggerType.INVALID:
            return result is not None and not result.is_valid

        return False

    def _execute_rule(
        self,
        rule: ActionRule,
        path: Path,
        result: ValidationResult | None,
    ) -> bool:
        """Execute a single rule."""
        context = self._build_context(path, result)

        if rule.action == ActionType.DELETE:
            self._delete_file(path)
            return True

        # Target resolution for Move/Copy/Rename
        if not rule.target:
            logger.warning("Action '%s' missing target pattern", rule.name)
            return False

        target_path = self._resolve_target(rule.target, context, path)
        if target_path is None:
            # Missing variable or bad target pattern; try next rule.
            return False

        if rule.action in (ActionType.MOVE, ActionType.COPY):
            # For Move/Copy, target is a directory — append source filename
            target_path = target_path / path.name

        if rule.action == ActionType.MOVE:
            return self._move_file(path, target_path, rule.conflict)
        elif rule.action == ActionType.COPY:
            return self._copy_file(path, target_path, rule.conflict)
        elif rule.action == ActionType.RENAME:
            return self._move_file(path, target_path, rule.conflict)
        return False

    def _build_context(self, path: Path, result: ValidationResult | None) -> dict[str, Any]:
        """Build variable context."""
        stats = path.stat()
        mtime = datetime.fromtimestamp(stats.st_mtime)
        now = datetime.now()
        ext_no_dot = path.suffix.lower().lstrip(".")

        ctx = {
            "original_name": path.name,
            "stem": path.stem,
            "ext": path.suffix,
            "ext_no_dot": ext_no_dot,
            "ext_group": self._guess_ext_group(ext_no_dot),
            "year": mtime.strftime("%Y"),
            "month": mtime.strftime("%m"),
            "day": mtime.strftime("%d"),
            "today_year": now.strftime("%Y"),
            "today_month": now.strftime("%m"),
            "today_day": now.strftime("%d"),
        }

        if result and result.matched_groups:
            safe = {k: self._sanitize_path_value(v) if isinstance(v, str) else v
                    for k, v in result.matched_groups.items()}
            ctx.update(safe)

        return ctx

    def _resolve_target(
        self,
        pattern: str,
        context: dict[str, Any],
        source_path: Path,
    ) -> Path | None:
        """Resolve target path from pattern."""
        try:
            resolved = pattern.format(**context)
            path = Path(resolved)
            pattern_was_absolute = path.is_absolute()

            # If path is relative, make it relative to source parent
            if not pattern_was_absolute:
                if self._target_root is not None:
                    path = self._target_root / path
                else:
                    path = source_path.parent / path

            # Containment check: only for relative patterns or when target_root is set.
            # Absolute paths from config are intentional admin decisions.
            if self._target_root is not None:
                if not self._validate_containment(path, self._target_root):
                    logger.warning(
                        "Path traversal blocked: '%s' escapes boundary '%s'",
                        path, self._target_root,
                    )
                    return None
            elif not pattern_was_absolute:
                boundary = source_path.parent
                if not self._validate_containment(path, boundary):
                    logger.warning(
                        "Path traversal blocked: '%s' escapes boundary '%s'",
                        path, boundary,
                    )
                    return None

            return path
        except KeyError as e:
            logger.warning("Variable substitution failed for pattern '%s': missing %s", pattern, e)
            return None

    def _delete_file(self, path: Path) -> None:
        if self._dry_run:
            logger.info("[DRY RUN] Delete file: %s", path)
            return
        path.unlink()
        logger.info("Deleted file: %s", path)

    def _move_file(self, source: Path, dest: Path, conflict: str) -> bool:
        if source == dest:
            return False

        if dest.exists():
            if conflict == "skip":
                logger.info("Skipping move (dest exists): %s -> %s", source, dest)
                return False
            elif conflict == "rename":
                dest = self._unique_path(dest)
            # overwrite falls through

        if self._dry_run:
            logger.info("[DRY RUN] Move file: %s -> %s", source, dest)
            return True

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(dest))
        logger.info("Moved file: %s -> %s", source, dest)
        return True

    def _copy_file(self, source: Path, dest: Path, conflict: str) -> bool:
        if source == dest:
            return False

        if dest.exists():
            if conflict == "skip":
                logger.info("Skipping copy (dest exists): %s -> %s", source, dest)
                return False
            elif conflict == "rename":
                dest = self._unique_path(dest)

        if self._dry_run:
            logger.info("[DRY RUN] Copy file: %s -> %s", source, dest)
            return True

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(dest))
        logger.info("Copied file: %s -> %s", source, dest)
        return True

    def _unique_path(self, path: Path) -> Path:
        counter = 1
        stem = path.stem
        suffix = path.suffix
        parent = path.parent

        while path.exists():
            path = parent / f"{stem}_{counter}{suffix}"
            counter += 1
        return path

    @staticmethod
    def _sanitize_path_value(value: str) -> str:
        """Strip path separators and parent references from a string value."""
        return value.replace("/", "").replace("\\", "").replace("..", "")

    @staticmethod
    def _validate_containment(resolved: Path, boundary: Path) -> bool:
        """Check resolved path stays within boundary directory."""
        try:
            return resolved.resolve().is_relative_to(boundary.resolve())
        except (OSError, ValueError):
            return False

    def _guess_ext_group(self, ext: str) -> str:
        """Guess extension group name for convenience variables.

        Custom ext_groups (from organize templates) are checked first.
        Falls back to hardcoded groups for backward compatibility.
        """
        if not ext:
            return self._default_ext_group
        # Custom mapping takes priority
        if self._custom_ext_groups:
            for group, exts in self._custom_ext_groups.items():
                if ext in exts:
                    return group
            return self._default_ext_group
        # Hardcoded fallback
        if ext in self._DRAWING_EXTS:
            return "drawings"
        if ext in self._DOCUMENT_EXTS:
            return "documents"
        if ext in self._IMAGE_EXTS:
            return "images"
        if ext in self._MODEL_EXTS:
            return "models"
        return self._default_ext_group

    @staticmethod
    def _normalize_ext_groups(
        ext_groups: dict[str, list[str]] | None,
    ) -> dict[str, list[str]] | None:
        """Normalize extension mappings for robust matching.

        Input examples supported:
        - "PDF" -> "pdf"
        - ".dwg" -> "dwg"
        """
        if not ext_groups:
            return None

        normalized: dict[str, list[str]] = {}
        for group, exts in ext_groups.items():
            cleaned: list[str] = []
            for ext in exts:
                token = ext.strip().lower().lstrip(".")
                if token and token not in cleaned:
                    cleaned.append(token)
            normalized[group] = cleaned
        return normalized
