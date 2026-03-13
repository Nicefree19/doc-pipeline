"""Export OCR/embed failure rows from the registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from doc_pipeline.config import settings
from doc_pipeline.processor.ocr_ops import collect_embed_failures
from doc_pipeline.storage.registry import DocumentRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OCR/embed failures")
    parser.add_argument("--db", default=None, help="Registry DB path (default: from settings)")
    parser.add_argument(
        "--output",
        default=str(_ROOT / "evals" / "failed_ocr_docs.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    registry = DocumentRegistry(db_path=args.db or settings.registry.db_path)
    failures = collect_embed_failures(registry)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"total_failures": len(failures), "docs": failures}, f, ensure_ascii=False, indent=2)
    print(f"Exported {len(failures)} OCR/embed failures → {out}")


if __name__ == "__main__":
    main()
