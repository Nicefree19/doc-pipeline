"""FileHub Dashboard using Flet.

A professional, corporate-ready dashboard for FileHub.
Features:
- Sidebar navigation (NavigationRail)
- Dedicated views: Dashboard, Organize, Settings
- Real-time status monitoring
- File management tools
"""

import logging
from pathlib import Path

import flet as ft

try:
    from filehub.config import FileHubConfig, get_config_path, load_config
    from filehub.templates.engine import TemplateEngine
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).parents[2]))
    from filehub.config import FileHubConfig, get_config_path, load_config
    from filehub.templates.engine import TemplateEngine

logger = logging.getLogger("filehub.ui")


class FileHubApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "FileHub Enterprise Dashboard"
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.theme = ft.Theme(color_scheme_seed="indigo")
        self.page.window_width = 1000  # type: ignore[attr-defined]
        self.page.window_height = 800  # type: ignore[attr-defined]
        self.page.padding = 0

        self.config = self._load_config()
        self.tmpl_engine = TemplateEngine(self.config.templates)
        self.tmpl_engine.load_organize_templates(self.config.organize_templates)

        self._current_view_index = 0
        self._views: list[ft.Control] = []

        self._setup_ui()

    def _load_config(self) -> FileHubConfig:
        try:
            return load_config(get_config_path())
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return FileHubConfig()

    def _setup_ui(self):
        # 0. Initialize FilePickers EARLY (fix for potential rendering issues)
        self.src_picker = ft.FilePicker()
        self.tgt_picker = ft.FilePicker()
        self.page.overlay.extend([self.src_picker, self.tgt_picker])

        # 1. Navigation Rail
        self.rail = ft.NavigationRail(
            selected_index=0,
            label_type=ft.NavigationRailLabelType.ALL,
            min_width=100,
            min_extended_width=200,
            group_alignment=-0.9,
            destinations=[
                ft.NavigationRailDestination(
                    icon="dashboard_outlined",
                    selected_icon="dashboard",
                    label="Overview",
                ),
                ft.NavigationRailDestination(
                    icon="folder_open_outlined",
                    selected_icon="folder_open",
                    label="Organize",
                ),
                ft.NavigationRailDestination(
                    icon="settings_outlined",
                    selected_icon="settings",
                    label="Settings",
                ),
            ],
            on_change=self._on_nav_change,
        )

        # 2. Content Area
        self.content_area = ft.Container(expand=True, padding=20)

        # 3. Main Layout
        self.page.add(
            ft.Row(
                [
                    self.rail,
                    ft.VerticalDivider(width=1),
                    self.content_area,
                ],
                expand=True,
            )
        )

        # Initialize Views (dependent on pickers)
        self._views = [
            self._build_dashboard_view(),
            self._build_organize_view(),
            self._build_settings_view(),
        ]

        # Render initial view
        self._update_view()

    def _on_nav_change(self, e):
        self._current_view_index = e.control.selected_index
        self._update_view()

    def _update_view(self):
        self.content_area.content = self._views[self._current_view_index]
        self.content_area.update()

    # --- VIEW BUILDERS ---

    def _build_dashboard_view(self) -> ft.Control:
        """Create the main dashboard overview."""
        # Stats cards
        watch_count = len(self.config.watcher.paths)
        notif_status = "Enabled" if self.config.notification.enabled else "Disabled"

        return ft.Column(
            [
                ft.Text("System Overview", size=28, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                ft.Row(
                    [
                        _InfoCard("Watch Paths", str(watch_count), "visibility", "blue"),
                        _InfoCard("Notifications", notif_status, "notifications", "orange"),
                        _InfoCard("Active Profile", "EPC Standard", "verified", "green"),
                    ],
                    spacing=20,
                ),
                ft.Container(height=20),
                ft.Text("Quick Access", size=20, weight=ft.FontWeight.BOLD),
                ft.ListView(
                    [
                        ft.ListTile(
                            leading=ft.Icon(ft.Icons.FOLDER),
                            title=ft.Text(p),
                            subtitle=ft.Text("Monitored directory"),
                        )
                        for p in self.config.watcher.paths
                    ],
                    height=200,
                ),
            ],
            scroll=ft.ScrollMode.AUTO,
        )

    def _build_organize_view(self) -> ft.Control:
        """Create the bulk organize tool view."""

        self.source_path = ft.TextField(label="Source Directory", read_only=True, expand=True)
        self.target_path = ft.TextField(label="Target Directory", read_only=True, expand=True)
        self.tmpl_dropdown = ft.Dropdown(
            label="Organization Template",
            options=[
                ft.dropdown.Option(tname) for tname in self.tmpl_engine.list_organize_templates()
            ],
            value="epc_structural",  # Default
        )
        self.log_view = ft.Column(scroll=ft.ScrollMode.ALWAYS, height=200)

        def pick_source(e):
            if e.path:
                self.source_path.value = e.path
                self.source_path.update()

        def pick_target(e):
            if e.path:
                self.target_path.value = e.path
                self.target_path.update()

        # Connect handlers to pre-created pickers
        self.src_picker.on_result = pick_source
        self.tgt_picker.on_result = pick_target

        def run_organize(dry_run: bool):
            if not self.source_path.value or not self.target_path.value:
                self.page.show_snack_bar(  # type: ignore[attr-defined]
                    ft.SnackBar(ft.Text("Please select both directories."))
                )
                return

            def ui_call(func):
                call_from_thread = getattr(self.page, "call_from_thread", None)
                if callable(call_from_thread):
                    call_from_thread(func)
                else:
                    func()

            def set_organize_running(running: bool):
                def _update():
                    self.btn_dry_run.disabled = running
                    self.btn_start.disabled = running
                    self.btn_dry_run.update()
                    self.btn_start.update()

                ui_call(_update)

            def append_log(
                text: str,
                color: str | None = None,
                bold: bool = False,
                size: int | None = None,
                refresh: bool = True,
            ):
                def _append():
                    kwargs: dict[str, object] = {}
                    if color is not None:
                        kwargs["color"] = color
                    if bold:
                        kwargs["weight"] = ft.FontWeight.BOLD
                    if size is not None:
                        kwargs["size"] = size
                    self.log_view.controls.append(ft.Text(text, **kwargs))
                    if refresh:
                        self.log_view.update()

                ui_call(_append)

            def reset_log():
                def _reset():
                    self.log_view.controls.clear()
                    self.log_view.controls.append(
                        ft.Text(f"Starting organization... (Dry Run: {dry_run})")
                    )
                    self.log_view.update()

                ui_call(_reset)

            def worker():
                try:
                    # Import Engine dynamically to avoid circular imports if any
                    from filehub.actions.engine import ActionEngine
                    from filehub.actions.models import ActionRule as ActionRuleModel
                    from filehub.core.models import ValidationResult

                    target_dir = Path(self.target_path.value)
                    source_dir = Path(self.source_path.value)

                    # Load selected template
                    tmpl_name = self.tmpl_dropdown.value
                    org_tmpl = self.tmpl_engine.get_organize_template(tmpl_name)

                    if not org_tmpl:
                        append_log(f"Template '{tmpl_name}' not found.", color="red")
                        return

                    rules = [ActionRuleModel.from_dict(r) for r in org_tmpl.rules]
                    ext_groups = org_tmpl.ext_groups
                    default_ext_group = org_tmpl.default_group or "others"

                    engine = ActionEngine(
                        rules,
                        target_root=target_dir,
                        dry_run=dry_run,
                        ext_groups=ext_groups,
                        default_ext_group=default_ext_group,
                    )

                    # Collect files
                    files = [p for p in source_dir.rglob("*") if p.is_file()]
                    append_log(f"Found {len(files)} files.", color="blue")

                    applied_count = 0
                    for file_path in files:
                        # Create a dummy validation result since we are just organizing based on rules
                        val_result = ValidationResult.valid(file_path.name, {})

                        if engine.process(file_path, val_result):
                            applied_count += 1
                            action_text = "Would move" if dry_run else "Moved"
                            append_log(
                                f"{action_text}: {file_path.name}",
                                size=12,
                                refresh=(applied_count % 10 == 0),
                            )

                    append_log(f"Complete. Actions: {applied_count}", color="green", bold=True)

                except Exception as e:
                    append_log(f"Error: {e}", color="red")
                    logger.error("GUI Organize Error: %s", e)
                finally:
                    set_organize_running(False)

            reset_log()
            set_organize_running(True)

            run_thread = getattr(self.page, "run_thread", None)
            if callable(run_thread):
                run_thread(worker)
            else:
                logger.warning("Page.run_thread unavailable; running organize synchronously.")
                worker()

        self.btn_dry_run = ft.ElevatedButton(
            "Simulate (Dry Run)",
            icon=ft.Icons.PREVIEW,
            on_click=lambda _: run_organize(True),
        )
        self.btn_start = ft.FilledButton(
            "Start Organize",
            icon=ft.Icons.PLAY_ARROW,
            on_click=lambda _: run_organize(False),
        )

        return ft.Column(
            [
                ft.Text("Bulk Organize", size=28, weight=ft.FontWeight.BOLD),
                ft.Text("Organize messy folders into structured project directories."),
                ft.Divider(),
                ft.Row(
                    [
                        self.source_path,
                        ft.IconButton(ft.Icons.FOLDER, on_click=self.src_picker.get_directory_path),
                    ]
                ),
                ft.Row(
                    [
                        self.target_path,
                        ft.IconButton(ft.Icons.FOLDER, on_click=self.tgt_picker.get_directory_path),
                    ]
                ),
                self.tmpl_dropdown,
                ft.Row(
                    [
                        self.btn_dry_run,
                        self.btn_start,
                    ]
                ),
                ft.Divider(),
                ft.Text("Activity Log", weight=ft.FontWeight.BOLD),
                ft.Container(
                    content=self.log_view,
                    bgcolor="grey_100",
                    padding=10,
                    border_radius=5,
                ),
            ]
        )

    def _build_settings_view(self) -> ft.Control:
        """Create the settings view."""

        self.sw_notif = ft.Switch(
            label="Enable Notifications", value=self.config.notification.enabled
        )
        self.sw_recur = ft.Switch(label="Watch recursive", value=self.config.watcher.recursive)
        self.txt_slack = ft.TextField(
            label="Slack Webhook URL", value=self.config.notification.slack_webhook or ""
        )
        self.txt_teams = ft.TextField(
            label="Teams Webhook URL", value=self.config.notification.teams_webhook or ""
        )

        def save_settings(e):
            try:
                # Update config object
                self.config.notification.enabled = self.sw_notif.value
                self.config.watcher.recursive = self.sw_recur.value
                self.config.notification.slack_webhook = self.txt_slack.value or None
                self.config.notification.teams_webhook = self.txt_teams.value or None

                # Save to file
                import yaml  # type: ignore[import-untyped]

                from filehub.config import get_config_path

                # We need a clean dict dump. For now, simple manual update or use a dumper if available.
                # Since FileHubConfig is complex, we might just update the specific fields in the YAML.
                # But for this demo, we'll assume we can dump the config or just patch it.
                # To fail gracefully if saving isn't fully implemented in Config class:

                # Minimal implementation: Load raw, update, dump
                cfg_path = get_config_path()
                if cfg_path.exists():
                    with open(cfg_path, encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}

                    data.setdefault("notification", {})["enabled"] = self.sw_notif.value
                    data["notification"]["slack_webhook"] = self.txt_slack.value
                    data["notification"]["teams_webhook"] = self.txt_teams.value

                    data.setdefault("watcher", {})["recursive"] = self.sw_recur.value

                    with open(cfg_path, "w", encoding="utf-8") as f:
                        yaml.dump(data, f, allow_unicode=True)

                self.page.show_snack_bar(ft.SnackBar(ft.Text("Settings saved successfully.")))

            except Exception as ex:
                self.page.show_snack_bar(ft.SnackBar(ft.Text(f"Failed to save: {ex}")))
                logger.error(f"Settings Save Error: {ex}")

        return ft.Column(
            [
                ft.Text("Settings", size=28, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                self.sw_notif,
                self.sw_recur,
                self.txt_slack,
                self.txt_teams,
                ft.Container(height=20),
                ft.FilledButton("Save Configuration", icon=ft.Icons.SAVE, on_click=save_settings),
            ]
        )


def _InfoCard(title: str, value: str, icon: str, color: str) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            [
                ft.Icon(icon, color=color, size=30),  # type: ignore[arg-type]
                ft.Text(value, size=24, weight=ft.FontWeight.BOLD),
                ft.Text(title, size=14, color="grey"),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        bgcolor="white",
        padding=20,
        border_radius=10,
        shadow=ft.BoxShadow(blur_radius=10, color="#1A000000"),
        expand=True,
    )


def main(page: ft.Page):
    FileHubApp(page)


if __name__ == "__main__":
    ft.app(target=main)
