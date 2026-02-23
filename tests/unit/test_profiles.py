"""Tests for naming profile system."""

from pathlib import Path

import pytest

from src.filehub.naming.profiles import (
    ISO19650Profile,
    NamingProfile,
    RegexProfile,
    RegexProfileConfig,
    RegexRule,
    load_profile,
)


class TestNamingProfileABC:
    """Test NamingProfile abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """NamingProfile cannot be instantiated directly."""
        with pytest.raises(TypeError):
            NamingProfile()

    def test_subclass_must_implement_all_methods(self):
        """A subclass missing abstract methods cannot be instantiated."""

        class IncompleteProfile(NamingProfile):
            @property
            def name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteProfile()


class TestRegexRule:
    """Test RegexRule dataclass."""

    def test_matches_simple_pattern(self):
        """Rule matches a filename that fits the pattern."""
        rule = RegexRule(pattern=r"^PROJ-\d{4}$")
        assert rule.matches("PROJ-0001")

    def test_no_match(self):
        """Rule does not match a non-conforming filename."""
        rule = RegexRule(pattern=r"^PROJ-\d{4}$")
        assert not rule.matches("OTHER-0001")

    def test_example_field_stored(self):
        """Example and description fields are stored correctly."""
        rule = RegexRule(
            pattern=r"^PROJ-\d{4}$",
            description="Project number format",
            example="PROJ-0001",
            error_message="Must be PROJ-NNNN",
        )
        assert rule.example == "PROJ-0001"
        assert rule.description == "Project number format"
        assert rule.error_message == "Must be PROJ-NNNN"

    def test_partial_match_rejected(self):
        """Rule uses match (anchored at start), partial match mid-string fails."""
        rule = RegexRule(pattern=r"^ABC$")
        assert not rule.matches("XABC")
        assert rule.matches("ABC")


class TestRegexProfileConfig:
    """Test RegexProfileConfig dataclass and from_dict."""

    def test_from_dict_full_data(self):
        """from_dict parses a complete dictionary correctly."""
        data = {
            "name": "my-profile",
            "description": "My custom profile",
            "rules": [
                {
                    "pattern": r"^DOC-\d+$",
                    "description": "Document ID",
                    "example": "DOC-123",
                    "error_message": "Must start with DOC-",
                }
            ],
            "extensions": [".pdf", ".docx"],
            "case_sensitive": False,
        }
        config = RegexProfileConfig.from_dict(data)

        assert config.name == "my-profile"
        assert config.description == "My custom profile"
        assert len(config.rules) == 1
        assert config.rules[0].pattern == r"^DOC-\d+$"
        assert config.rules[0].example == "DOC-123"
        assert config.extensions == [".pdf", ".docx"]
        assert config.case_sensitive is False

    def test_from_dict_empty_data(self):
        """from_dict with empty dict uses defaults."""
        config = RegexProfileConfig.from_dict({})

        assert config.name == "custom"
        assert config.description == "Custom naming profile"
        assert config.rules == []
        assert config.extensions == []
        assert config.case_sensitive is True

    def test_from_dict_multiple_rules(self):
        """from_dict handles multiple rules."""
        data = {
            "rules": [
                {"pattern": r"^A-\d+$"},
                {"pattern": r"^B-\d+$"},
            ],
        }
        config = RegexProfileConfig.from_dict(data)
        assert len(config.rules) == 2


class TestRegexProfile:
    """Test RegexProfile validation."""

    def test_validate_matches_simple_pattern(self, tmp_path):
        """A file matching the regex rule is valid."""
        config = RegexProfileConfig(
            rules=[RegexRule(pattern=r"^PROJ-\d{4}$")],
        )
        profile = RegexProfile(config)

        file = tmp_path / "PROJ-0001.pdf"
        file.write_text("")

        result = profile.validate(file)
        assert result.is_valid
        assert result.filename == "PROJ-0001.pdf"

    def test_validate_no_match_returns_invalid(self, tmp_path):
        """A file not matching any rule is invalid."""
        config = RegexProfileConfig(
            rules=[RegexRule(pattern=r"^PROJ-\d{4}$")],
        )
        profile = RegexProfile(config)

        file = tmp_path / "random-name.pdf"
        file.write_text("")

        result = profile.validate(file)
        assert not result.is_valid
        assert "does not match" in result.message.lower()

    def test_validate_extension_allowed(self, tmp_path):
        """File with allowed extension passes extension check."""
        config = RegexProfileConfig(
            rules=[RegexRule(pattern=r"^DOC-\d+$")],
            extensions=[".pdf", ".dwg"],
        )
        profile = RegexProfile(config)

        file = tmp_path / "DOC-001.pdf"
        file.write_text("")

        result = profile.validate(file)
        assert result.is_valid

    def test_validate_extension_disallowed(self, tmp_path):
        """File with disallowed extension is rejected."""
        config = RegexProfileConfig(
            rules=[RegexRule(pattern=r"^DOC-\d+$")],
            extensions=[".pdf", ".dwg"],
        )
        profile = RegexProfile(config)

        file = tmp_path / "DOC-001.txt"
        file.write_text("")

        result = profile.validate(file)
        assert not result.is_valid
        assert "not allowed" in result.message.lower()
        assert ".pdf" in result.message
        assert ".dwg" in result.message

    def test_validate_case_insensitive(self, tmp_path):
        """Case-insensitive mode matches regardless of case."""
        config = RegexProfileConfig(
            rules=[RegexRule(pattern=r"^PROJ-\d{4}$")],
            case_sensitive=False,
        )
        profile = RegexProfile(config)

        file = tmp_path / "proj-0001.pdf"
        file.write_text("")

        result = profile.validate(file)
        assert result.is_valid

    def test_validate_case_sensitive_rejects_wrong_case(self, tmp_path):
        """Case-sensitive mode rejects wrong case."""
        config = RegexProfileConfig(
            rules=[RegexRule(pattern=r"^PROJ-\d{4}$")],
            case_sensitive=True,
        )
        profile = RegexProfile(config)

        file = tmp_path / "proj-0001.pdf"
        file.write_text("")

        result = profile.validate(file)
        assert not result.is_valid

    def test_validate_empty_rules_returns_valid(self, tmp_path):
        """Empty rules list means all filenames are valid."""
        config = RegexProfileConfig(rules=[])
        profile = RegexProfile(config)

        file = tmp_path / "anything.pdf"
        file.write_text("")

        result = profile.validate(file)
        assert result.is_valid
        assert result.matched_groups == {}

    def test_validate_directory_returns_invalid(self, tmp_path):
        """Validating a directory returns invalid result."""
        config = RegexProfileConfig(rules=[RegexRule(pattern=r"^.*$")])
        profile = RegexProfile(config)

        result = profile.validate(tmp_path)
        assert not result.is_valid
        assert "directory" in result.message.lower()

    def test_validate_error_message_in_output(self, tmp_path):
        """Custom error_message from rule appears in validation message."""
        config = RegexProfileConfig(
            rules=[
                RegexRule(
                    pattern=r"^PROJ-\d{4}$",
                    error_message="File must follow PROJ-NNNN format",
                )
            ],
        )
        profile = RegexProfile(config)

        file = tmp_path / "bad-name.pdf"
        file.write_text("")

        result = profile.validate(file)
        assert not result.is_valid
        assert "PROJ-NNNN" in result.message

    def test_validate_example_in_output(self, tmp_path):
        """Example from rule appears in validation message when no error_message."""
        config = RegexProfileConfig(
            rules=[
                RegexRule(
                    pattern=r"^PROJ-\d{4}$",
                    example="PROJ-0001",
                )
            ],
        )
        profile = RegexProfile(config)

        file = tmp_path / "bad-name.pdf"
        file.write_text("")

        result = profile.validate(file)
        assert not result.is_valid
        assert "PROJ-0001" in result.message

    def test_validate_named_groups_returned(self, tmp_path):
        """Named regex groups are returned in matched_groups."""
        config = RegexProfileConfig(
            rules=[
                RegexRule(pattern=r"^(?P<prefix>[A-Z]+)-(?P<number>\d{4})$")
            ],
        )
        profile = RegexProfile(config)

        file = tmp_path / "PROJ-0001.pdf"
        file.write_text("")

        result = profile.validate(file)
        assert result.is_valid
        assert result.matched_groups == {"prefix": "PROJ", "number": "0001"}

    def test_validate_multiple_rules_first_match_wins(self, tmp_path):
        """With multiple rules, the first matching rule determines the result."""
        config = RegexProfileConfig(
            rules=[
                RegexRule(pattern=r"^(?P<type>PROJ)-(?P<num>\d+)$"),
                RegexRule(pattern=r"^(?P<type>DOC)-(?P<num>\d+)$"),
            ],
        )
        profile = RegexProfile(config)

        file = tmp_path / "DOC-42.pdf"
        file.write_text("")

        result = profile.validate(file)
        assert result.is_valid
        assert result.matched_groups["type"] == "DOC"

    def test_validate_string_path(self, tmp_path):
        """Validate accepts a string path."""
        config = RegexProfileConfig(
            rules=[RegexRule(pattern=r"^PROJ-\d{4}$")],
        )
        profile = RegexProfile(config)

        file = tmp_path / "PROJ-0001.pdf"
        file.write_text("")

        result = profile.validate(str(file))
        assert result.is_valid

    def test_name_and_description(self):
        """Profile name and description come from config."""
        config = RegexProfileConfig(name="my-profile", description="My profile desc")
        profile = RegexProfile(config)

        assert profile.name == "my-profile"
        assert profile.description == "My profile desc"


class TestISO19650Profile:
    """Test ISO19650Profile wrapper."""

    def test_name(self):
        """Profile name is 'iso19650'."""
        profile = ISO19650Profile()
        assert profile.name == "iso19650"

    def test_description(self):
        """Profile has a meaningful description."""
        profile = ISO19650Profile()
        assert "ISO 19650" in profile.description

    def test_validate_delegates_to_iso19650_validator(self):
        """Profile delegates validation to the underlying ISO19650Validator."""
        profile = ISO19650Profile()

        # Valid ISO 19650 filename
        result = profile.validate(Path("/some/path/PROJ-ORG-ZZ-00-DR-A-0001.pdf"))
        assert result.is_valid
        assert result.matched_groups["project"] == "PROJ"
        assert result.matched_groups["number"] == "0001"

    def test_validate_invalid_filename(self):
        """Profile correctly rejects invalid filenames."""
        profile = ISO19650Profile()

        result = profile.validate(Path("/some/path/bad-name.pdf"))
        assert not result.is_valid

    def test_validate_with_custom_config(self):
        """Profile can be initialized with a custom ISO19650Config."""
        from src.filehub.naming.config import ISO19650Config

        config = ISO19650Config(originator=["ABC"])
        profile = ISO19650Profile(config=config)

        result = profile.validate(Path("/path/PROJ-ABC-ZZ-00-DR-A-0001.pdf"))
        assert result.is_valid

        result = profile.validate(Path("/path/PROJ-XYZ-ZZ-00-DR-A-0001.pdf"))
        assert not result.is_valid

    def test_validate_directory_fails(self, tmp_path):
        """Validating a directory returns invalid."""
        profile = ISO19650Profile()
        result = profile.validate(tmp_path)
        assert not result.is_valid
        assert "directory" in result.message.lower()


class TestLoadProfile:
    """Test the load_profile factory function."""

    def test_load_iso19650_type(self):
        """Loading a profile with type='iso19650' returns ISO19650Profile."""
        profile = load_profile({"type": "iso19650"})
        assert isinstance(profile, ISO19650Profile)
        assert profile.name == "iso19650"

    def test_load_iso19650_with_config(self):
        """Loading iso19650 profile passes config to ISO19650Config."""
        profile = load_profile({
            "type": "iso19650",
            "originator": ["ABC"],
        })
        assert isinstance(profile, ISO19650Profile)
        # Verify config was applied by testing validation
        result = profile.validate(Path("/path/PROJ-ABC-ZZ-00-DR-A-0001.pdf"))
        assert result.is_valid
        result = profile.validate(Path("/path/PROJ-XYZ-ZZ-00-DR-A-0001.pdf"))
        assert not result.is_valid

    def test_load_regex_type(self):
        """Loading a profile with type='regex' returns RegexProfile."""
        profile = load_profile({
            "type": "regex",
            "name": "my-regex",
            "rules": [{"pattern": r"^TEST-\d+$"}],
        })
        assert isinstance(profile, RegexProfile)
        assert profile.name == "my-regex"

    def test_load_default_type_is_regex(self):
        """When type is omitted, default is 'regex'."""
        profile = load_profile({
            "name": "default-regex",
            "rules": [{"pattern": r"^FILE-\d+$"}],
        })
        assert isinstance(profile, RegexProfile)
        assert profile.name == "default-regex"

    def test_load_unknown_type_raises_value_error(self):
        """Unknown profile type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown profile type"):
            load_profile({"type": "unknown_type"})
