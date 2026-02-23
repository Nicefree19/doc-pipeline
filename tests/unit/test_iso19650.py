"""Tests for ISO 19650 Validator."""

from pathlib import Path

from src.filehub.naming import ISO19650Config, ISO19650Validator


class TestISO19650Validator:
    """Test ISO 19650 naming convention validation."""

    def test_valid_filename(self):
        """Test that valid filenames pass validation."""
        validator = ISO19650Validator()
        result = validator.validate_filename("PROJ-ORG-ZZ-00-DR-A-0001.pdf")

        assert result.is_valid
        assert result.matched_groups["project"] == "PROJ"
        assert result.matched_groups["originator"] == "ORG"
        assert result.matched_groups["number"] == "0001"

    def test_valid_filename_with_underscore(self):
        """Test that underscore separators work."""
        validator = ISO19650Validator()
        result = validator.validate_filename("PROJ_ORG_ZZ_00_DR_A_0001.dwg")

        assert result.is_valid

    def test_empty_filename_fails(self):
        """Test that empty filename fails."""
        validator = ISO19650Validator()
        result = validator.validate_filename("")

        assert not result.is_valid
        assert "empty" in result.message.lower()

    def test_too_few_fields_fails(self):
        """Test that filename with too few fields fails."""
        validator = ISO19650Validator()
        result = validator.validate_filename("PROJ-ORG-ZZ.pdf")

        assert not result.is_valid
        assert "missing" in result.message.lower()

    def test_too_many_fields_fails(self):
        """Test that filename with too many fields fails."""
        validator = ISO19650Validator()
        result = validator.validate_filename("PROJ-ORG-ZZ-00-DR-A-0001-EXTRA.pdf")

        assert not result.is_valid
        assert "too many" in result.message.lower()

    def test_invalid_number_format(self):
        """Test that invalid number format fails."""
        validator = ISO19650Validator()
        result = validator.validate_filename("PROJ-ORG-ZZ-00-DR-A-XXXX.pdf")

        assert not result.is_valid
        assert "number" in result.message.lower()

    def test_custom_number_pattern(self):
        """Test custom number pattern."""
        config = ISO19650Config(number_pattern=r"^[A-Z]\d{3}$")
        validator = ISO19650Validator(config)

        result = validator.validate_filename("PROJ-ORG-ZZ-00-DR-A-A001.pdf")
        assert result.is_valid

        result = validator.validate_filename("PROJ-ORG-ZZ-00-DR-A-0001.pdf")
        assert not result.is_valid

    def test_allowed_originator_codes(self):
        """Test that only allowed originator codes pass."""
        config = ISO19650Config(originator=["ABC", "XYZ"])
        validator = ISO19650Validator(config)

        result = validator.validate_filename("PROJ-ABC-ZZ-00-DR-A-0001.pdf")
        assert result.is_valid

        result = validator.validate_filename("PROJ-DEF-ZZ-00-DR-A-0001.pdf")
        assert not result.is_valid
        assert "ABC" in result.message or "XYZ" in result.message

    def test_allowed_type_codes(self):
        """Test that only allowed type codes pass."""
        config = ISO19650Config(type=["DR", "MO", "SP"])
        validator = ISO19650Validator(config)

        result = validator.validate_filename("PROJ-ORG-ZZ-00-DR-A-0001.pdf")
        assert result.is_valid

        result = validator.validate_filename("PROJ-ORG-ZZ-00-XX-A-0001.pdf")
        assert not result.is_valid

    def test_type_help_message(self):
        """Test type help message with labels."""
        config = ISO19650Config(type=["DR", "MO"], type_help={"DR": "Drawing", "MO": "Model"})
        validator = ISO19650Validator(config)

        result = validator.validate_filename("PROJ-ORG-ZZ-00-XX-A-0001.pdf")
        assert not result.is_valid
        assert "Drawing" in result.message or "Model" in result.message

    def test_validate_path(self):
        """Test validation with Path object."""
        validator = ISO19650Validator()
        result = validator.validate(Path("/some/path/PROJ-ORG-ZZ-00-DR-A-0001.pdf"))

        assert result.is_valid

    def test_validate_directory_fails(self, temp_dir):
        """Test that directory validation fails."""
        validator = ISO19650Validator()
        result = validator.validate(temp_dir)

        assert not result.is_valid
        assert "directory" in result.message.lower()
