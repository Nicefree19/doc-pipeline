"""FileHub CLI - Command Line Interface."""

from __future__ import annotations

import fnmatch
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="filehub",
    help="FileHub - Unified File Management Hub",
    add_completion=False,
)


@app.command()
def watch(
    paths: Annotated[list[Path] | None, typer.Argument(help="Directories to watch")] = None,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Configuration file path")
    ] = None,
    target: Annotated[
        Path | None, typer.Option("--target", "-t", help="Root directory for action rule targets")
    ] = None,
) -> None:
    """Start file watching with naming convention validation."""
    from ..app import run_app

    str_paths = [str(p) for p in paths] if paths else None
    str_config = str(config) if config else None
    raise typer.Exit(run_app(watch_paths=str_paths, config_path=str_config, target_root=target))


@app.command()
def validate(
    files: Annotated[list[Path], typer.Argument(help="Files to validate")],
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Configuration file path")
    ] = None,
) -> None:
    """Validate file naming conventions."""
    from ..config import ConfigError, FileHubConfig, load_config
    from ..naming import ISO19650Validator

    # Load config
    try:
        cfg = load_config(config)
    except ConfigError:
        cfg = FileHubConfig()

    # Create validator
    validator = ISO19650Validator(cfg.iso19650) if cfg.iso19650 else ISO19650Validator()

    has_errors = False
    for file_path in files:
        if not file_path.exists():
            typer.echo(f"  [SKIP] {file_path} (not found)", err=True)
            continue

        result = validator.validate(file_path)
        if result.is_valid:
            typer.echo(f"  [PASS] {file_path.name}")
        else:
            typer.echo(f"  [FAIL] {file_path.name}: {result.message}")
            has_errors = True

    if has_errors:
        raise typer.Exit(1)


@app.command()
def organize(
    source: Annotated[Path, typer.Argument(help="Source folder to analyze")],
    target: Annotated[Path, typer.Option("--target", "-t", help="Destination root folder")],
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Configuration file path")
    ] = None,
    template_name: Annotated[
        str | None,
        typer.Option(
            "--template", "-T", help="Organize template name (e.g. epc_structural, general)"
        ),
    ] = None,
    recursive: Annotated[
        bool, typer.Option("--recursive/--no-recursive", help="Scan subdirectories")
    ] = True,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Preview actions without moving files")
    ] = False,
    analyze_only: Annotated[
        bool, typer.Option("--analyze-only", help="Analyze folder only")
    ] = False,
) -> None:
    """Analyze and organize files into a user-selected target path."""
    from ..actions.engine import ActionEngine
    from ..actions.models import ActionRule as ActionRuleModel
    from ..config import ConfigError, FileHubConfig, load_config
    from ..core.models import ValidationResult
    from ..naming import ProfileLoader, ProfileValidator
    from ..templates.engine import TemplateEngine

    if not source.exists() or not source.is_dir():
        typer.echo(f"Source folder not found: {source}", err=True)
        raise typer.Exit(1)

    try:
        cfg = load_config(config)
    except ConfigError as exc:
        typer.echo(f"Config load failed, using defaults: {exc}", err=True)
        cfg = FileHubConfig()

    files = [p for p in _iter_files(source, recursive) if not _should_ignore_path(p, cfg)]
    ext_counts = Counter(p.suffix.lower().lstrip(".") for p in files)

    typer.echo("Folder Analysis")
    typer.echo("---")
    typer.echo(f"Source: {source}")
    typer.echo(f"Target: {target}")
    typer.echo(f"Files scanned: {len(files)}")
    if ext_counts:
        typer.echo("Top extensions:")
        for ext, count in ext_counts.most_common(10):
            label = ext if ext else "(noext)"
            typer.echo(f"  - {label}: {count}")
    else:
        typer.echo("Top extensions: (none)")

    if analyze_only:
        return

    # Template-based or config-based rules
    ext_groups: dict[str, list[str]] | None = None
    default_ext_group = "others"
    if template_name:
        tmpl_engine = TemplateEngine(cfg.templates)
        tmpl_engine.load_organize_templates(cfg.organize_templates)
        org_tmpl = tmpl_engine.get_organize_template(template_name)
        if org_tmpl is None:
            typer.echo(f"Template not found: {template_name}", err=True)
            raise typer.Exit(1)
        typer.echo(f"Template: {org_tmpl.name} ({org_tmpl.description})")
        rules = [ActionRuleModel.from_dict(r) for r in org_tmpl.rules]
        ext_groups = org_tmpl.ext_groups or None
        default_ext_group = org_tmpl.default_group or "others"

        # Optional folder scaffolding
        if org_tmpl.folder_template:
            scaffold_tmpl = tmpl_engine.get_template(org_tmpl.folder_template)
            if scaffold_tmpl and not dry_run:
                tmpl_engine.scaffold(org_tmpl.folder_template, target)
    else:
        rules = cfg.actions or _default_organize_rules()

    validator = None
    if cfg.naming.enabled:
        profile = ProfileLoader(cfg.naming).get_active_profile()
        if profile is not None:
            validator = ProfileValidator(profile)
            typer.echo(f"Naming profile: {profile.name} ({profile.type})")
        else:
            typer.echo("Naming profile: (not configured)")
    else:
        typer.echo("Naming profile: (disabled)")

    engine = ActionEngine(
        rules,
        target_root=target,
        dry_run=dry_run,
        ext_groups=ext_groups,
        default_ext_group=default_ext_group,
    )
    applied = 0
    valid_count = 0
    invalid_count = 0

    for path in files:
        result: ValidationResult
        if validator is not None:
            result = validator.validate(path)
            if result.is_valid:
                valid_count += 1
            else:
                invalid_count += 1
        else:
            result = ValidationResult.valid(path.name, {})

        if engine.process(path, result):
            applied += 1

    typer.echo("")
    typer.echo("Organize Summary")
    typer.echo("---")
    typer.echo(f"Rules used: {len(rules)}")
    typer.echo(f"Actions applied: {applied}")
    if validator is not None:
        typer.echo(f"Validation passed: {valid_count}")
        typer.echo(f"Validation failed: {invalid_count}")
    if dry_run:
        typer.echo("Dry-run mode: no files were moved.")


@app.command("scaffold")
def scaffold_cmd(
    template: Annotated[str, typer.Argument(help="Project template name")] = "epc_standard",
    target: Annotated[Path, typer.Argument(help="Target directory path")] = Path("."),
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Configuration file path")
    ] = None,
) -> None:
    """Generate project directory structure from a template."""
    from ..config import ConfigError, FileHubConfig, load_config
    from ..templates.engine import TemplateEngine

    try:
        cfg = load_config(config)
    except ConfigError:
        cfg = FileHubConfig()

    tmpl_engine = TemplateEngine(cfg.templates)

    try:
        tmpl_engine.scaffold(template, target)
        typer.echo(f"Successfully scaffolded '{template}' at {target.resolve()}")
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        typer.echo("Available templates:", err=True)
        for name in tmpl_engine.list_templates():
            typer.echo(f"  - {name}", err=True)
        raise typer.Exit(1) from e
    except Exception as e:
        typer.echo(f"Scaffold failed: {e}", err=True)
        raise typer.Exit(1) from e


@app.command("template")
def template_cmd(
    action: Annotated[str, typer.Argument(help="Action: list, info")] = "list",
    name: Annotated[str | None, typer.Option("--name", "-n", help="Template name")] = None,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Configuration file path")
    ] = None,
) -> None:
    """Manage organize templates."""
    from ..config import ConfigError, FileHubConfig, load_config
    from ..templates.engine import TemplateEngine

    try:
        cfg = load_config(config)
    except ConfigError:
        cfg = FileHubConfig()

    tmpl_engine = TemplateEngine(cfg.templates)
    tmpl_engine.load_organize_templates(cfg.organize_templates)

    if action == "list":
        typer.echo("Organize Templates")
        typer.echo("---")
        names = tmpl_engine.list_organize_templates()
        if not names:
            typer.echo("No templates available.")
            return
        for tname in names:
            tmpl = tmpl_engine.get_organize_template(tname)
            desc = tmpl.description if tmpl else ""
            typer.echo(f"  - {tname}: {desc}")
    elif action == "info":
        if not name:
            typer.echo("Please specify --name for template info.", err=True)
            raise typer.Exit(1)
        tmpl = tmpl_engine.get_organize_template(name)
        if not tmpl:
            typer.echo(f"Template not found: {name}", err=True)
            raise typer.Exit(1)
        typer.echo(f"Template: {tmpl.name}")
        typer.echo(f"Description: {tmpl.description}")
        typer.echo(f"Default group: {tmpl.default_group}")
        if tmpl.folder_template:
            typer.echo(f"Folder template: {tmpl.folder_template}")
        typer.echo("Extension groups:")
        for group, exts in tmpl.ext_groups.items():
            typer.echo(f"  {group}: {', '.join(exts)}")
        typer.echo(f"Rules: {len(tmpl.rules)}")
        for i, rule in enumerate(tmpl.rules, 1):
            typer.echo(f"  {i}. {rule.get('name', 'Unnamed')} ({rule.get('action', '?')})")
    else:
        typer.echo(f"Unknown action: {action}. Use 'list' or 'info'.", err=True)
        raise typer.Exit(1)


@app.command("config")
def config_cmd(
    action: Annotated[str, typer.Argument(help="Action: show, init")] = "show",
    path: Annotated[Path | None, typer.Option("--path", "-p", help="Config file path")] = None,
) -> None:
    """Manage FileHub configuration."""
    if action == "show":
        _config_show(path)
    elif action == "init":
        _config_init(path)
    else:
        typer.echo(f"Unknown action: {action}. Use 'show' or 'init'.", err=True)
        raise typer.Exit(1)


def _config_show(path: Path | None = None) -> None:
    """Show current configuration."""
    from ..config import ConfigError, get_config_path

    try:
        if path:
            config_path = path
        else:
            config_path = Path(get_config_path())
    except ConfigError as exc:
        typer.echo("No configuration file found. Run 'filehub config init' to create one.")
        raise typer.Exit(1) from exc

    typer.echo(f"Config file: {config_path}")
    typer.echo("---")

    content = config_path.read_text(encoding="utf-8")
    typer.echo(content)


def _config_init(path: Path | None = None) -> None:
    """Create default configuration file."""
    from ..config import create_default_config

    if path is None:
        import os

        app_data = os.environ.get("APPDATA", "")
        if app_data:
            path = Path(app_data) / "FileHub" / "config.yaml"
        else:
            path = Path.home() / ".config" / "filehub" / "config.yaml"

    if path.exists():
        overwrite = typer.confirm(f"Config file already exists at {path}. Overwrite?")
        if not overwrite:
            raise typer.Exit(0)

    create_default_config(path)
    typer.echo(f"Configuration file created: {path}")


def _iter_files(source: Path, recursive: bool) -> list[Path]:
    """Yield file paths in source folder."""
    if recursive:
        return [p for p in source.rglob("*") if p.is_file()]
    return [p for p in source.glob("*") if p.is_file()]


def _should_ignore_path(path: Path, config) -> bool:
    """Apply ignore rules from config to a path."""
    name = path.name
    ignore = config.ignore

    if ignore.prefixes and name.startswith(tuple(ignore.prefixes)):
        return True

    extensions = {ext.lower() for ext in ignore.extensions}
    if extensions and path.suffix.lower() in extensions:
        return True

    for pattern in ignore.globs:
        if fnmatch.fnmatch(str(path), pattern) or fnmatch.fnmatch(name, pattern):
            return True

    return False


def _default_organize_rules():
    """Fallback rules used when config has no action rules."""
    from ..actions.models import ActionRule, ActionType, TriggerType

    return [
        ActionRule(
            name="Organize by extension group",
            action=ActionType.MOVE,
            trigger=TriggerType.ALWAYS,
            target="{ext_group}/{ext_no_dot}",
            conflict="rename",
        )
    ]


@app.command()
def status() -> None:
    """Show FileHub status."""
    from .. import __version__

    typer.echo(f"FileHub v{__version__}")
    typer.echo("---")

    # Show config path
    from ..config import ConfigError, get_config_path

    try:
        config_path = get_config_path()
        typer.echo(f"Config: {config_path}")
    except ConfigError:
        typer.echo("Config: (not found)")

    # Show default watch paths
    from ..config import FileHubConfig

    cfg = FileHubConfig()
    watch_paths = cfg.watcher.get_watch_paths()
    if watch_paths:
        typer.echo("Watch paths:")
        for wp in watch_paths:
            typer.echo(f"  - {wp}")
    else:
        typer.echo("Watch paths: (none configured)")


@app.command()
def stats(
    db_path: Annotated[
        Path | None,
        typer.Option("--db", help="Path to stats database"),
    ] = None,
) -> None:
    """Show recent validation statistics."""
    from ..reporting.report import ReportGenerator
    from ..reporting.store import StatsStore

    if db_path is None:
        db_path = Path.home() / ".filehub" / "stats.db"

    if not db_path.exists():
        typer.echo("FileHub Statistics")
        typer.echo("---")
        typer.echo("No statistics database found.")
        typer.echo(f"Expected: {db_path}")
        return

    store = StatsStore(db_path=db_path)
    try:
        gen = ReportGenerator(store)
        report = gen.generate_text_report()
        typer.echo(report)
    finally:
        store.close()


@app.command()
def plugins() -> None:
    """List available plugins.

    Shows plugins discovered via the 'filehub.plugins' entry-point group.
    Plugins are loaded at runtime when the 'watch' command is active.
    """
    typer.echo("FileHub Plugins")
    typer.echo("---")

    # Discover plugins via entry points
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="filehub.plugins")
    except Exception:
        eps = []  # type: ignore[assignment]

    if not eps:
        typer.echo("No plugins installed.")
        typer.echo("Install packages that provide 'filehub.plugins' entry points.")
        return

    for ep in eps:
        typer.echo(f"  - {ep.name} ({ep.value})")
