# Contributing to FileHub

Thank you for your interest in contributing to FileHub! We welcome contributions from everyone.

## Getting Started

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/filehub.git
   cd filehub
   ```
3. **Set up development environment**:
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate virtual environment
   # Windows:
   .venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate
   
   # Install dev dependencies
   pip install -e ".[dev]"
   # or
   pip install -e ".[full]"
   ```

## Development Guidelines

### Code Style

We follow PEP 8 and use strict linting.
- **Ruff** for linting
- **Black** for formatting
- **MyPy** for type checking

Run checks before committing:
```bash
ruff check src/ tests/
black --check src/ tests/
mypy src/filehub
```

### Testing

All new features must include unit tests.
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=filehub
```

### Internationalization (i18n)

If you modify user-facing strings:
1. Wrap strings with `_("...")`.
2. Update translation files (we use gettext).
3. Test with both English and Korean locales.

## Pull Request Process

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/amazing-feature
   ```
2. Commit your changes with clear messages.
3. Push to your fork and submit a Pull Request.
4. CI checks must pass (tests, linting).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
