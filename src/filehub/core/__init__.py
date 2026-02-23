"""FileHub Core Module.

Common utilities shared across all features:
- File watching
- Pipeline processing
- Notifications
- Configuration
"""

from .notification import Notifier
from .pipeline import Aggregator, IgnoreFilter, Processor, StabilityChecker
from .watcher import ReconcileScanner, create_observer, start_observer, stop_observer

__all__ = [
    "create_observer",
    "start_observer",
    "stop_observer",
    "ReconcileScanner",
    "Processor",
    "Aggregator",
    "StabilityChecker",
    "IgnoreFilter",
    "Notifier",
]
