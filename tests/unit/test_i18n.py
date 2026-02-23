"""Unit tests for i18n."""

from unittest.mock import patch

from filehub import i18n
from filehub.i18n import _, get_system_language, init_i18n, ngettext


def test_get_system_language_env():
    with patch.dict("os.environ", {"LANG": "ko_KR.UTF-8"}):
        assert get_system_language() == "ko"


def test_get_system_language_default():
    with (
        patch("locale.getlocale", return_value=(None, None)),
        patch.dict("os.environ", {}, clear=True),
    ):
        assert get_system_language() == "en"


def test_init_i18n_fallback():
    # Mock gettext.translation to raise error (force fallback)
    with patch("gettext.translation", side_effect=FileNotFoundError):
        init_i18n("kr")  # invalid code
        assert i18n._translator is not None
        # Should return original string
        assert _("Hello") == "Hello"


def test_translation_en():
    init_i18n("en")
    assert _("FileHub") == "FileHub"


def test_ngettext():
    init_i18n("en")
    assert ngettext("file", "files", 1) == "file"
    assert ngettext("file", "files", 2) == "files"
