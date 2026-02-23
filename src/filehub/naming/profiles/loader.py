"""Profile loader."""

from __future__ import annotations

from ..config import NamingConfig
from .models import NamingProfile


class ProfileLoader:
    """Loads and manages naming profiles."""

    def __init__(self, config: NamingConfig):
        self._config = config
        self._profiles: dict[str, NamingProfile] = {}
        self._load_profiles()

    def _load_profiles(self) -> None:
        """Load profiles from config."""
        # Load custom profiles
        for name, data in self._config.profiles.items():
            try:
                self._profiles[name] = NamingProfile.from_dict(name, data)
            except Exception as e:
                # Log error but continue
                print(f"Failed to load profile '{name}': {e}")

        # Register default ISO19650 profile if configured
        if self._config.iso19650:
            # Check if already defined (custom override support)
            if "iso19650" not in self._profiles:
                self._profiles["iso19650"] = NamingProfile(
                    name="iso19650",
                    type="iso19650",
                    iso_config=self._config.iso19650,
                    description="Default ISO 19650 Standard",
                )

    def get_profile(self, name: str) -> NamingProfile | None:
        """Get profile by name."""
        return self._profiles.get(name)

    def get_active_profile(self) -> NamingProfile | None:
        """Get the active profile."""
        # 1. Explicitly active
        if self._config.active_profile:
            return self.get_profile(self._config.active_profile)

        # 2. Fallback to iso19650 if it exists
        return self.get_profile("iso19650")

    @property
    def profiles(self) -> dict[str, NamingProfile]:
        """Get all loaded profiles."""
        return self._profiles
