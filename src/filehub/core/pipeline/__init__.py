"""FileHub Core Pipeline Module.

Event processing pipeline components.
"""

from .aggregator import Aggregator
from .ignore_filter import IgnoreFilter
from .processor import Processor
from .stability import StabilityChecker

__all__ = [
    "Processor",
    "Aggregator",
    "StabilityChecker",
    "IgnoreFilter",
]
