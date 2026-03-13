"""Build a retry-candidate report for OCR/embed failures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from doc_pipeline.config import settings
from doc_pipeline.processor.ocr_ops import build_retry_report
from doc_pipeline.storage.registry import DocumentRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Report OCR retry candidates")
    parser.add_argument("--db", default=None, help="Registry DB path (default: from settings)")
    parser.add_argument(
        "--output",
        default=str(_ROOT / "evals" / "ocr_retry_report.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    registry = DocumentRegistry(db_path=args.db or settings.registry.db_path)
    report = build_retry_report(registry)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Retry report saved → {out}")


if __name__ == "__main__":
    main()
