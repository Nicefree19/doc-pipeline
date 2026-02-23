"""FileHub i18n - Internationalization Support.

Provides multi-language support using gettext.
Default language: English
Supported languages: en, ko
"""

from __future__ import annotations

import gettext
import locale
import os
from pathlib import Path

# Module-level translator
_translator: gettext.GNUTranslations | gettext.NullTranslations | None = None
_current_lang: str = "en"


def get_locale_dir() -> Path:
    """Get the locales directory path."""
    return Path(__file__).parent / "locales"


def get_system_language() -> str:
    """Detect system language.

    Returns:
        Language code (e.g., 'ko', 'en')
    """
    try:
        # Check environment variables first
        lang = os.environ.get("LANG", "") or os.environ.get("LC_MESSAGES", "")
        if lang:
            return lang.split("_")[0].split(".")[0].lower()

        # Fall back to system locale
        system_locale = locale.getlocale()[0]
        if system_locale:
            return system_locale.split("_")[0].lower()
    except Exception:
        pass

    return "en"


def init_i18n(language: str | None = None) -> None:
    """Initialize internationalization.

    Args:
        language: Language code (e.g., 'ko', 'en').
                  If None, auto-detect from system.
    """
    global _translator, _current_lang

    lang = language or get_system_language()
    _current_lang = lang

    locale_dir = get_locale_dir()

    try:
        _translator = gettext.translation(
            domain="filehub", localedir=str(locale_dir), languages=[lang], fallback=True
        )
    except Exception:
        # Fallback to NullTranslations (returns original strings)
        _translator = gettext.NullTranslations()


def _(message: str) -> str:
    """Translate a message.

    This is the primary translation function.

    Args:
        message: Message to translate (in English)

    Returns:
        Translated message
    """
    global _translator

    if _translator is None:
        init_i18n()

    return _translator.gettext(message) if _translator else message


def ngettext(singular: str, plural: str, n: int) -> str:
    """Translate a message with pluralization.

    Args:
        singular: Singular form (in English)
        plural: Plural form (in English)
        n: Count for pluralization

    Returns:
        Translated message
    """
    global _translator

    if _translator is None:
        init_i18n()

    if _translator:
        return _translator.ngettext(singular, plural, n)
    return singular if n == 1 else plural


def get_current_language() -> str:
    """Get the current language code."""
    return _current_lang


# Initialize on import (auto-detect language)
init_i18n()
