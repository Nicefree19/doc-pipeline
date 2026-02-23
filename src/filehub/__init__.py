"""FileHub - Unified File Management Hub.

A comprehensive file management solution combining:
- File Naming Convention Validator (Digital Maknae)
- Intelligent Document Storage System (AI-IDSS)

Features:
- ISO 19650 file naming validation
- Real-time file system monitoring
- System tray integration
- Toast notifications
- PDF document analysis (optional)
"""

__version__ = "0.1.0"
__author__ = "FileHub Team"

from .app import Application, run_app

__all__ = ["Application", "run_app", "__version__"]
