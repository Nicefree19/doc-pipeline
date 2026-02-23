"""FileHub entry point."""

import sys


def main() -> int:
    """Main entry point."""
    try:
        from .cli import app
        app()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0


if __name__ == "__main__":
    sys.exit(main())
