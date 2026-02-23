"""FileHub Plugin System.

Provides extensibility through a plugin architecture.
"""

from .base import PluginBase
from .manager import PluginManager

__all__ = [
    "PluginBase",
    "PluginManager",
]
