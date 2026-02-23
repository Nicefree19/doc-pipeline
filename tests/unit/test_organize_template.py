"""Unit tests for organize template system."""

from __future__ import annotations

from typer.testing import CliRunner

from src.filehub.actions.engine import ActionEngine
from src.filehub.actions.models import ActionRule, ActionType, TriggerType
from src.filehub.cli.main import app
from src.filehub.config.schema import FileHubConfig
from src.filehub.templates.engine import TemplateEngine
from src.filehub.templates.schemas import OrganizeTemplate

runner = CliRunner()


# ------------------------------------------------------------------
# OrganizeTemplate dataclass
# ------------------------------------------------------------------


class TestOrganizeTemplate:
    """Tests for OrganizeTemplate dataclass."""

    def test_basic_creation(self):
        tmpl = OrganizeTemplate(name="test", description="A test template")
        assert tmpl.name == "test"
        assert tmpl.description == "A test template"
        assert tmpl.ext_groups == {}
        assert tmpl.rules == []
        assert tmpl.default_group == "others"
        assert tmpl.folder_template is None

    def test_from_dict(self):
        data = {
            "name": "custom",
            "description": "Custom template",
            "ext_groups": {"docs": ["pdf", "docx"], "images": ["jpg", "png"]},
            "rules": [
                {"name": "Sort", "action": "move", "trigger": "always",
                 "target": "{ext_group}", "conflict": "rename"}
            ],
            "default_group": "misc",
            "folder_template": "epc_standard",
        }
        tmpl = OrganizeTemplate.from_dict(data)
        assert tmpl.name == "custom"
        assert tmpl.ext_groups["docs"] == ["pdf", "docx"]
        assert len(tmpl.rules) == 1
        assert tmpl.default_group == "misc"
        assert tmpl.folder_template == "epc_standard"

    def test_from_dict_defaults(self):
        tmpl = OrganizeTemplate.from_dict({"name": "minimal"})
        assert tmpl.name == "minimal"
        assert tmpl.ext_groups == {}
        assert tmpl.rules == []
        assert tmpl.default_group == "others"
        assert tmpl.folder_template is None


# ------------------------------------------------------------------
# TemplateEngine organize template integration
# ------------------------------------------------------------------


class TestTemplateEngineOrganize:
    """Tests for TemplateEngine organize template methods."""

    def test_builtin_epc_structural_loaded(self):
        engine = TemplateEngine()
        tmpl = engine.get_organize_template("epc_structural")
        assert tmpl is not None
        assert tmpl.name == "epc_structural"
        assert "drawings" in tmpl.ext_groups
        assert "analysis" in tmpl.ext_groups
        assert "fabrication" in tmpl.ext_groups
        assert "dwg" in tmpl.ext_groups["drawings"]
        assert "mgb" in tmpl.ext_groups["analysis"]
        assert "nc1" in tmpl.ext_groups["fabrication"]
        assert tmpl.folder_template == "epc_standard"

    def test_builtin_general_loaded(self):
        engine = TemplateEngine()
        tmpl = engine.get_organize_template("general")
        assert tmpl is not None
        assert tmpl.name == "general"
        assert "drawings" in tmpl.ext_groups
        assert "documents" in tmpl.ext_groups
        assert tmpl.folder_template is None

    def test_list_organize_templates(self):
        engine = TemplateEngine()
        names = engine.list_organize_templates()
        assert "epc_structural" in names
        assert "general" in names

    def test_get_nonexistent_returns_none(self):
        engine = TemplateEngine()
        assert engine.get_organize_template("nonexistent") is None

    def test_load_user_templates(self):
        engine = TemplateEngine()
        user_tmpl = OrganizeTemplate(
            name="my_custom",
            description="My custom template",
            ext_groups={"data": ["csv", "json"]},
            rules=[{"name": "Sort data", "action": "move", "trigger": "always",
                     "target": "data_files", "conflict": "rename"}],
        )
        engine.load_organize_templates([user_tmpl])
        assert engine.get_organize_template("my_custom") is not None
        assert engine.get_organize_template("my_custom").ext_groups["data"] == ["csv", "json"]

    def test_user_template_overrides_builtin(self):
        engine = TemplateEngine()
        override = OrganizeTemplate(
            name="general",
            description="Overridden general template",
            ext_groups={"custom": ["xyz"]},
        )
        engine.load_organize_templates([override])
        tmpl = engine.get_organize_template("general")
        assert tmpl.description == "Overridden general template"
        assert tmpl.ext_groups == {"custom": ["xyz"]}


# ------------------------------------------------------------------
# ActionEngine custom ext_groups
# ------------------------------------------------------------------


class TestActionEngineExtGroups:
    """Tests for ActionEngine with custom ext_groups."""

    def test_custom_ext_groups_applied(self, tmp_path):
        """Custom ext_groups should classify files correctly."""
        rule = ActionRule(
            name="Sort",
            action=ActionType.MOVE,
            trigger=TriggerType.ALWAYS,
            target="{ext_group}/{ext_no_dot}",
            conflict="rename",
        )
        custom_groups = {
            "analysis": ["mgb", "out", "msr"],
            "fabrication": ["nc1"],
        }
        engine = ActionEngine(
            [rule],
            target_root=tmp_path / "target",
            ext_groups=custom_groups,
        )

        src = tmp_path / "source"
        src.mkdir()
        mgb_file = src / "test.mgb"
        mgb_file.touch()

        engine.process(mgb_file, None)
        assert (tmp_path / "target" / "analysis" / "mgb" / "test.mgb").exists()

    def test_custom_ext_groups_unknown_goes_to_others(self, tmp_path):
        """Extensions not in custom groups should go to 'others'."""
        rule = ActionRule(
            name="Sort",
            action=ActionType.MOVE,
            trigger=TriggerType.ALWAYS,
            target="{ext_group}/{ext_no_dot}",
            conflict="rename",
        )
        custom_groups = {"docs": ["pdf"]}
        engine = ActionEngine(
            [rule],
            target_root=tmp_path / "target",
            ext_groups=custom_groups,
        )

        src = tmp_path / "source"
        src.mkdir()
        xyz_file = src / "test.xyz"
        xyz_file.touch()

        engine.process(xyz_file, None)
        assert (tmp_path / "target" / "others" / "xyz" / "test.xyz").exists()

    def test_custom_default_group_is_applied_for_unknown_extension(self, tmp_path):
        """Custom template default group should be used for unknown extensions."""
        rule = ActionRule(
            name="Sort",
            action=ActionType.MOVE,
            trigger=TriggerType.ALWAYS,
            target="{ext_group}/{ext_no_dot}",
            conflict="rename",
        )
        engine = ActionEngine(
            [rule],
            target_root=tmp_path / "target",
            ext_groups={"docs": ["pdf"]},
            default_ext_group="misc",
        )

        src = tmp_path / "source"
        src.mkdir()
        file_path = src / "test.xyz"
        file_path.touch()

        engine.process(file_path, None)
        assert (tmp_path / "target" / "misc" / "xyz" / "test.xyz").exists()

    def test_custom_ext_groups_normalize_case_and_dot(self, tmp_path):
        """Extension mapping should accept values like '.PDF' and 'DWG'."""
        rule = ActionRule(
            name="Sort",
            action=ActionType.MOVE,
            trigger=TriggerType.ALWAYS,
            target="{ext_group}/{ext_no_dot}",
            conflict="rename",
        )
        engine = ActionEngine(
            [rule],
            target_root=tmp_path / "target",
            ext_groups={"documents": [".PDF"], "drawings": ["DWG"]},
        )

        src = tmp_path / "source"
        src.mkdir()
        pdf_file = src / "sample.PDF"
        pdf_file.touch()

        engine.process(pdf_file, None)
        assert (tmp_path / "target" / "documents" / "pdf" / "sample.PDF").exists()

    def test_no_custom_ext_groups_uses_hardcoded(self, tmp_path):
        """Without custom ext_groups, hardcoded groups should work."""
        rule = ActionRule(
            name="Sort",
            action=ActionType.MOVE,
            trigger=TriggerType.ALWAYS,
            target="{ext_group}/{ext_no_dot}",
            conflict="rename",
        )
        engine = ActionEngine(
            [rule],
            target_root=tmp_path / "target",
        )

        src = tmp_path / "source"
        src.mkdir()
        pdf_file = src / "test.pdf"
        pdf_file.touch()

        engine.process(pdf_file, None)
        assert (tmp_path / "target" / "documents" / "pdf" / "test.pdf").exists()


# ------------------------------------------------------------------
# FileHubConfig.from_dict with organize_templates
# ------------------------------------------------------------------


class TestFileHubConfigOrganizeTemplates:
    """Tests for FileHubConfig organize_templates parsing."""

    def test_empty_organize_templates(self):
        cfg = FileHubConfig.from_dict({})
        assert cfg.organize_templates == []

    def test_parse_organize_templates(self):
        data = {
            "organize_templates": [
                {
                    "name": "my_tmpl",
                    "description": "Test template",
                    "ext_groups": {"docs": ["pdf"]},
                    "rules": [{"name": "R1", "action": "move", "target": "{ext_group}"}],
                    "default_group": "other_stuff",
                }
            ]
        }
        cfg = FileHubConfig.from_dict(data)
        assert len(cfg.organize_templates) == 1
        tmpl = cfg.organize_templates[0]
        assert tmpl.name == "my_tmpl"
        assert tmpl.ext_groups == {"docs": ["pdf"]}
        assert tmpl.default_group == "other_stuff"


# ------------------------------------------------------------------
# End-to-end: template -> ext_groups -> ActionEngine -> file moved
# ------------------------------------------------------------------


class TestEndToEndOrganizeTemplate:
    """Full-flow test: template selection -> file organization."""

    def test_epc_template_flow(self, tmp_path):
        """EPC template should classify .mgb as 'analysis'."""
        engine = TemplateEngine()
        tmpl = engine.get_organize_template("epc_structural")
        assert tmpl is not None

        rules = [ActionRule.from_dict(r) for r in tmpl.rules]
        action_engine = ActionEngine(
            rules,
            target_root=tmp_path / "target",
            ext_groups=tmpl.ext_groups,
        )

        src = tmp_path / "source"
        src.mkdir()
        for name in ["calc.mgb", "drawing.dwg", "report.pdf", "photo.jpg", "model.nc1"]:
            (src / name).touch()

        for f in src.iterdir():
            action_engine.process(f, None)

        assert (tmp_path / "target" / "analysis" / "mgb" / "calc.mgb").exists()
        assert (tmp_path / "target" / "drawings" / "dwg" / "drawing.dwg").exists()
        assert (tmp_path / "target" / "documents" / "pdf" / "report.pdf").exists()
        assert (tmp_path / "target" / "images" / "jpg" / "photo.jpg").exists()
        assert (tmp_path / "target" / "fabrication" / "nc1" / "model.nc1").exists()


# ------------------------------------------------------------------
# CLI integration tests
# ------------------------------------------------------------------


def _make_base_config() -> FileHubConfig:
    cfg = FileHubConfig()
    cfg.naming.enabled = False
    cfg.actions = []
    cfg.ignore.prefixes = []
    cfg.ignore.extensions = []
    cfg.ignore.globs = []
    return cfg


class TestCLITemplateOption:
    """Tests for CLI --template option."""

    def test_template_list_command(self, monkeypatch):
        cfg = _make_base_config()
        monkeypatch.setattr("src.filehub.config.load_config", lambda _=None: cfg)

        result = runner.invoke(app, ["template", "list"])
        assert result.exit_code == 0
        assert "epc_structural" in result.output
        assert "general" in result.output

    def test_template_info_command(self, monkeypatch):
        cfg = _make_base_config()
        monkeypatch.setattr("src.filehub.config.load_config", lambda _=None: cfg)

        result = runner.invoke(app, ["template", "info", "--name", "epc_structural"])
        assert result.exit_code == 0
        assert "epc_structural" in result.output
        assert "drawings" in result.output
        assert "analysis" in result.output

    def test_template_info_not_found(self, monkeypatch):
        cfg = _make_base_config()
        monkeypatch.setattr("src.filehub.config.load_config", lambda _=None: cfg)

        result = runner.invoke(app, ["template", "info", "--name", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_organize_with_template(self, tmp_path, monkeypatch):
        """--template should use template rules and ext_groups."""
        src = tmp_path / "source"
        dst = tmp_path / "target"
        src.mkdir()
        (src / "calc.mgb").write_text("data", encoding="utf-8")
        (src / "plan.dwg").write_text("data", encoding="utf-8")

        cfg = _make_base_config()
        monkeypatch.setattr("src.filehub.config.load_config", lambda _=None: cfg)

        result = runner.invoke(
            app,
            ["organize", str(src), "--target", str(dst),
             "--template", "epc_structural"],
        )

        assert result.exit_code == 0
        assert "Template: epc_structural" in result.output
        assert (dst / "analysis" / "mgb" / "calc.mgb").exists()
        assert (dst / "drawings" / "dwg" / "plan.dwg").exists()

    def test_organize_with_unknown_template(self, tmp_path, monkeypatch):
        """Unknown --template should exit with error."""
        src = tmp_path / "source"
        dst = tmp_path / "target"
        src.mkdir()
        (src / "a.pdf").write_text("x", encoding="utf-8")

        cfg = _make_base_config()
        monkeypatch.setattr("src.filehub.config.load_config", lambda _=None: cfg)

        result = runner.invoke(
            app,
            ["organize", str(src), "--target", str(dst),
             "--template", "nonexistent"],
        )

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_organize_template_uses_default_group(self, tmp_path, monkeypatch):
        """CLI should pass template default_group to ActionEngine."""
        src = tmp_path / "source"
        dst = tmp_path / "target"
        src.mkdir()
        (src / "archive.zip").write_text("x", encoding="utf-8")

        cfg = _make_base_config()
        cfg.organize_templates = [
            OrganizeTemplate(
                name="custom_default",
                description="Uses custom fallback",
                ext_groups={"docs": ["pdf"]},
                rules=[
                    {
                        "name": "Sort",
                        "action": "move",
                        "trigger": "always",
                        "target": "{ext_group}/{ext_no_dot}",
                        "conflict": "rename",
                    }
                ],
                default_group="misc",
            )
        ]
        monkeypatch.setattr("src.filehub.config.load_config", lambda _=None: cfg)

        result = runner.invoke(
            app,
            ["organize", str(src), "--target", str(dst), "--template", "custom_default"],
        )

        assert result.exit_code == 0
        assert (dst / "misc" / "zip" / "archive.zip").exists()

    def test_organize_without_template_backward_compatible(self, tmp_path, monkeypatch):
        """Without --template, existing behavior should be preserved."""
        src = tmp_path / "source"
        dst = tmp_path / "target"
        src.mkdir()
        (src / "a.pdf").write_text("a", encoding="utf-8")
        (src / "b.dwg").write_text("b", encoding="utf-8")

        cfg = _make_base_config()
        monkeypatch.setattr("src.filehub.config.load_config", lambda _=None: cfg)

        result = runner.invoke(
            app,
            ["organize", str(src), "--target", str(dst)],
        )

        assert result.exit_code == 0
        # Should use hardcoded groups (pdf -> documents, dwg -> drawings)
        assert (dst / "documents" / "pdf" / "a.pdf").exists()
        assert (dst / "drawings" / "dwg" / "b.dwg").exists()
