# FileHub

Unified File Management Hub combining **Naming Convention Validator** (Digital Maknae) and **Intelligent Document Storage** (AI-IDSS).

[![Tests](https://github.com/filehub/filehub/actions/workflows/test.yml/badge.svg)](https://github.com/filehub/filehub/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Naming Convention Watcher**: Real-time validation of ISO 19650 file naming standards.
- **Intelligent Storage**: Automated organization and processing of documents (PDF, etc.).
- **System Tray Integration**: Background operation with minimal intrusion.
- **Internationalization (i18n)**: Full support for English and Korean.
- **Cross-Platform Core**: Core logic works on Windows and Linux (UI features are Windows-optimized).

## Documentation

- **[Vision & Architecture (English)](docs/vision_en.md)** — Product vision, core architecture, ecosystem positioning, and roadmap
- **[Vision & Architecture (한국어)](docs/vision_ko.md)** — 제품 비전, 핵심 아키텍처, 생태계 포지셔닝, 로드맵

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/filehub/filehub.git
cd filehub

# Install config
pip install -e "."
# For full features including PDF support
pip install -e ".[full]"
```

### Running

```bash
# Run application
filehub

# Or via python module
python -m filehub
```

### Organize Files By Rules

```bash
# Analyze only
filehub organize "E:/01.Work/01.EPC/03. PROJECT" --target "E:/Sorted" --analyze-only

# Organize with configured rules (or fallback by extension group)
filehub organize "E:/01.Work/01.EPC/03. PROJECT" --target "E:/Sorted"
```

## Configuration

Configuration file is automatically created at:
- Windows: `%APPDATA%\FileHub\config.yaml`
- Linux: `~/.config/filehub/config.yaml`

Example `config.yaml`:
```yaml
watcher:
  paths:
    - "C:/Work/Projects"
  recursive: true

pipeline:
  debounce_seconds: 1.0
  stability_timeout: 20.0

notification:
  enabled: true
  duration: 10
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
