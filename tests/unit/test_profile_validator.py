"""Unit tests for ProfileValidator."""



from filehub.naming.profiles.models import NamingField, NamingProfile
from filehub.naming.validator import ProfileValidator


def test_validator_regex():
    profile = NamingProfile(name="date_pattern", type="regex", regex_pattern=r"^\d{8}_.+")
    validator = ProfileValidator(profile)

    assert validator.validate("20231025_test.txt").is_valid
    assert not validator.validate("invalid_test.txt").is_valid


def test_validator_custom_fields():
    fields = [
        NamingField(name="proj", allowed=["P1", "P2"]),
        NamingField(name="type", pattern=r"^[A-Z]{2}$"),
    ]
    profile = NamingProfile(name="custom_proj", type="custom", separator="_", fields=fields)
    validator = ProfileValidator(profile)

    # Valid
    assert validator.validate("P1_DW.pdf").is_valid

    # Invalid value
    res = validator.validate("P3_DW.pdf")
    assert not res.is_valid
    assert "P3" in str(res.message) or "invalid value" in str(res.message)  # check exact msg later

    # Invalid pattern
    assert not validator.validate("P1_abc.pdf").is_valid

    # Invalid field count
    assert not validator.validate("P1.pdf").is_valid


def test_validator_iso19650_delegation():
    # Mock ISO config (or use default)
    from filehub.config.schema import ISO19650Config

    iso_config = ISO19650Config(project=["TEST"])
    profile = NamingProfile(name="iso", type="iso19650", iso_config=iso_config)
    validator = ProfileValidator(profile)

    # We expect ISOValidator to be called.
    # Since we use real ISOValidator, let's test a simple failure/pass

    # Invalid (too short)
    res = validator.validate("test.txt")
    assert not res.is_valid
