"""CLI entry point for the document pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from doc_pipeline.config import settings
from doc_pipeline.config.logging_config import setup_logging

setup_logging(level=settings.logging.level, log_dir=settings.logging.log_dir)
logger = logging.getLogger("doc_pipeline")


# ---------------------------------------------------------------------------
# Local classifier (no LLM) for B-grade documents
# ---------------------------------------------------------------------------


def classify_local(file_path: Path, text: str) -> dict[str, str]:
    """Classify document using filename, path, and text keywords (no API).

    Delegates to classify_document_chain with grade="B" (no LLM).
    """
    from doc_pipeline.processor.classifier import classify_document_chain

    result = classify_document_chain(file_path, text, grade="B")
    return {
        "doc_type": result.doc_type,
        "project_name": result.project_name,
        "year": str(result.year),
    }


# ---------------------------------------------------------------------------
# Process command — delegates to pipeline.process_document()
# ---------------------------------------------------------------------------

def cmd_process(args: argparse.Namespace) -> None:
    """Process a single document or a directory of documents."""
    from doc_pipeline.collector.adapters import SUPPORTED_EXTENSIONS
    from doc_pipeline.models.schemas import SecurityGrade
    from doc_pipeline.processor.pipeline import process_document

    target = Path(args.path)
    if target.is_file():
        doc_files = [target]
    else:
        doc_files = sorted(
            f for ext in SUPPORTED_EXTENSIONS
            for f in target.glob(f"**/*{ext}")
        )

    if not doc_files:
        logger.error("No document files found at: %s", target)
        return

    logger.info("Processing %d document files", len(doc_files))
    grade = SecurityGrade(args.grade) if args.grade else None

    for doc_path in doc_files:
        logger.info("--- Processing: %s ---", doc_path.name)
        result = process_document(
            doc_path,
            grade=grade,
            no_embed=args.no_embed,
        )
        if result.skipped:
            logger.info("Skipped: %s", result.skip_reason)
        elif result.error:
            logger.error("Error: %s", result.error)


# ---------------------------------------------------------------------------
# Batch command — directory processing with JSON output
# ---------------------------------------------------------------------------

def cmd_batch(args: argparse.Namespace) -> None:
    """Batch process a directory of documents with JSON result output."""
    import time

    from doc_pipeline.collector.adapters import SUPPORTED_EXTENSIONS
    from doc_pipeline.models.schemas import SecurityGrade
    from doc_pipeline.processor.pipeline import process_document

    input_dir = Path(args.path)
    doc_files = sorted(
        f for ext in SUPPORTED_EXTENSIONS
        for f in input_dir.glob(f"**/*{ext}")
    )
    if args.limit:
        doc_files = doc_files[: args.limit]

    if not doc_files:
        logger.error("No document files found in: %s", input_dir)
        return

    logger.info("Batch processing %d document files", len(doc_files))
    grade = SecurityGrade(args.grade) if args.grade else None

    results = []
    for idx, pdf_path in enumerate(doc_files, 1):
        logger.info("[%d/%d] %s", idx, len(doc_files), pdf_path.name)
        start = time.time()
        try:
            result = process_document(
                pdf_path,
                grade=grade,
                no_embed=args.no_embed,
            )
            elapsed = time.time() - start
            entry = {
                "file": pdf_path.name,
                "skipped": result.skipped,
                "skip_reason": result.skip_reason,
                "doc_type": result.doc.doc_type.value if result.doc else "",
                "project_name": result.doc.project_name if result.doc else "",
                "year": result.doc.year if result.doc else 0,
                "chunks_stored": result.chunks_stored,
                "summary": result.summary[:200] if result.summary else "",
                "elapsed_sec": round(elapsed, 1),
            }
        except Exception as exc:
            elapsed = time.time() - start
            logger.error("Error processing %s: %s", pdf_path.name, exc)
            entry = {
                "file": pdf_path.name,
                "error": str(exc),
                "elapsed_sec": round(elapsed, 1),
            }
        results.append(entry)

    # Save JSON results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Results saved to %s", output_path)

    # Summary
    processed = sum(1 for r in results if not r.get("skipped") and not r.get("error"))
    errors = sum(1 for r in results if r.get("error"))
    skipped = sum(1 for r in results if r.get("skipped"))
    logger.info("Batch complete: %d processed, %d skipped, %d errors", processed, skipped, errors)


# ---------------------------------------------------------------------------
# Draft command — document draft generation
# ---------------------------------------------------------------------------

def cmd_draft(args: argparse.Namespace) -> None:
    """Generate a document draft using RAG references and templates."""
    from doc_pipeline.generator.drafter import generate_draft

    draft = generate_draft(
        doc_type=args.type,
        project_name=args.project,
        issue=args.issue,
        use_llm=not args.no_llm,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(draft, encoding="utf-8")
        logger.info("Draft saved to %s", output_path)
    else:
        print(draft)


# ---------------------------------------------------------------------------
# Watch command — auto-processing daemon
# ---------------------------------------------------------------------------

def cmd_watch(args: argparse.Namespace) -> None:
    """Watch directories for new documents and process automatically."""
    from doc_pipeline.collector.watcher import FolderWatcher
    from doc_pipeline.models.schemas import SecurityGrade
    from doc_pipeline.processor.pipeline import process_document

    grade = SecurityGrade(args.grade) if args.grade else None

    def on_new_file(file_path: Path) -> None:
        result = process_document(file_path, grade=grade)
        if result.skipped:
            logger.info("Skipped: %s", result.skip_reason)
        elif result.doc:
            logger.info(
                "Auto-processed: %s -> %s",
                result.doc.file_name_original,
                result.doc.file_name_standard,
            )

    watch_dirs = [args.path] if args.path else [
        d for d in [
            settings.watch.contracts_dir,
            settings.watch.action_plans_dir,
            settings.watch.opinions_dir,
        ] if d
    ]

    if not watch_dirs:
        logger.error("No directories to watch. Provide path or set WATCH_* env vars.")
        return

    watcher = FolderWatcher(watch_dirs, on_new_file, cooldown=args.cooldown)
    logger.info("Starting watcher on: %s", ", ".join(watch_dirs))
    watcher.run_forever()


# ---------------------------------------------------------------------------
# Embed command — batch embed already-registered documents
# ---------------------------------------------------------------------------

def cmd_embed(args: argparse.Namespace) -> None:
    """Embed documents that were processed with no_embed=True."""
    import time

    from doc_pipeline.models.schemas import SecurityGrade
    from doc_pipeline.processor.pipeline import embed_document
    from doc_pipeline.storage.registry import DocumentRegistry

    registry = DocumentRegistry(db_path=settings.registry.db_path)
    grade = SecurityGrade(args.grade) if args.grade else SecurityGrade.B

    if args.doc_id:
        doc = registry.get_document(args.doc_id)
        if not doc:
            logger.error("Document not found: %s", args.doc_id)
            return
        docs = [doc]
    elif getattr(args, "force", False):
        docs = registry.list_documents(limit=args.limit or None, offset=0)
        logger.info("Force mode: targeting all %d documents", len(docs))
    else:
        # --all: query unembedded documents directly via SQL
        docs = registry.list_unembedded(limit=args.limit)

    if not docs:
        logger.info("No documents need embedding.")
        return

    if args.dry_run:
        logger.info("Dry run — %d documents would be embedded:", len(docs))
        for d in docs:
            logger.info("  %s  %s", d["doc_id"], d.get("file_name_standard") or d["file_name_original"])
        return

    logger.info("Embedding %d documents (grade=%s)", len(docs), grade.value)
    success = 0
    errors = 0

    # Create shared instances once for the entire batch (avoid per-doc overhead)
    from doc_pipeline.storage.vectordb import VectorStore
    shared_store = VectorStore(persist_dir=settings.chroma.persist_dir)

    for idx, doc in enumerate(docs, 1):
        file_path = _resolve_doc_file_path(doc)
        if not file_path:
            logger.warning("[%d/%d] %s: file not found, skipping", idx, len(docs), doc["doc_id"])
            errors += 1
            continue

        start = time.time()
        try:
            chunks = embed_document(doc, file_path, grade, store=shared_store, registry=registry)
            elapsed = time.time() - start
            logger.info(
                "[%d/%d] %s: %d chunks (%.1fs)",
                idx, len(docs), doc["doc_id"], chunks, elapsed,
            )
            success += 1
        except Exception as exc:
            elapsed = time.time() - start
            logger.error(
                "[%d/%d] %s: error (%.1fs) — %s",
                idx, len(docs), doc["doc_id"], elapsed, exc,
            )
            errors += 1

    logger.info("Embedding complete: %d success, %d errors", success, errors)


# ---------------------------------------------------------------------------
# Reclassify command — re-classify existing documents
# ---------------------------------------------------------------------------

def cmd_reclassify(args: argparse.Namespace) -> None:
    """Reclassify documents using current classification rules."""
    from doc_pipeline.models.schemas import SecurityGrade
    from doc_pipeline.processor.pipeline import reclassify_document
    from doc_pipeline.storage.registry import DocumentRegistry

    registry = DocumentRegistry(db_path=settings.registry.db_path)
    grade = SecurityGrade(args.grade) if args.grade else SecurityGrade.B

    if args.doc_id:
        doc = registry.get_document(args.doc_id)
        if not doc:
            logger.error("Document not found: %s", args.doc_id)
            return
        docs = [doc]
    else:
        docs = registry.list_documents(limit=10000, offset=0)

    if not docs:
        logger.info("No documents to reclassify.")
        return

    logger.info("Reclassifying %d documents (grade=%s, dry_run=%s)", len(docs), grade.value, args.dry_run)
    changed = 0
    unchanged = 0
    errors = 0

    for idx, doc in enumerate(docs, 1):
        file_path = _resolve_doc_file_path(doc)
        if not file_path:
            logger.warning("[%d/%d] %s: file not found, skipping", idx, len(docs), doc["doc_id"])
            errors += 1
            continue

        try:
            result = reclassify_document(doc, file_path, grade, dry_run=args.dry_run)
            if result.get("changed"):
                if args.dry_run:
                    logger.info(
                        "[%d/%d] %s: %s -> %s (dry-run)",
                        idx, len(docs), doc["doc_id"],
                        result["old_type"], result["new_type"],
                    )
                changed += 1
            else:
                unchanged += 1
        except Exception as exc:
            logger.error("[%d/%d] %s: error — %s", idx, len(docs), doc["doc_id"], exc)
            errors += 1

    logger.info("Reclassify complete: %d changed, %d unchanged, %d errors", changed, unchanged, errors)


# ---------------------------------------------------------------------------
# Report command — classification accuracy report
# ---------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> None:
    """Generate classification accuracy report by comparing folder vs doc_type_ext."""
    from doc_pipeline.storage.registry import DocumentRegistry

    registry = DocumentRegistry(db_path=settings.registry.db_path)
    docs = registry.list_documents(limit=10000, offset=0)

    fmt = getattr(args, "format", "table")

    if not docs:
        if fmt == "json":
            import json
            print(json.dumps({
                "total_documents": 0, "compared": 0, "correct": 0,
                "incorrect": 0, "accuracy_pct": 0.0, "mismatches": [],
            }, ensure_ascii=False, indent=2))
        elif fmt == "csv":
            import csv
            import sys
            writer = csv.DictWriter(sys.stdout, fieldnames=["doc_id", "file", "folder", "classified"])
            writer.writeheader()
        else:
            print("No documents in registry.")
        return

    correct = 0
    incorrect = 0
    mismatches: list[dict] = []

    for doc in docs:
        source_path = doc.get("source_path", "")
        if not source_path:
            continue

        # Extract folder name as "actual type" heuristic
        folder_name = Path(source_path).parent.name
        classified_type = doc.get("doc_type_ext") or doc.get("doc_type", "")

        if not folder_name or folder_name in (".", "", "(Nova Web Upload)"):
            continue

        # Simple substring match (folder contains type or type contains folder)
        if folder_name in classified_type or classified_type in folder_name:
            correct += 1
        else:
            incorrect += 1
            mismatches.append({
                "doc_id": doc["doc_id"],
                "file": doc.get("file_name_original", ""),
                "folder": folder_name,
                "classified": classified_type,
            })

    total_compared = correct + incorrect
    accuracy = (correct / total_compared * 100) if total_compared > 0 else 0.0

    disclaimer = (
        "NOTE: Accuracy is heuristic — based on source_path folder name vs "
        "doc_type_ext substring match. Web uploads and non-standard folder "
        "structures may skew results."
    )

    if fmt == "json":
        import json
        report = {
            "total_documents": len(docs),
            "compared": total_compared,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy_pct": round(accuracy, 1),
            "mismatches": mismatches,
            "disclaimer": disclaimer,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif fmt == "csv":
        import csv
        import sys
        writer = csv.DictWriter(sys.stdout, fieldnames=["doc_id", "file", "folder", "classified"])
        writer.writeheader()
        for m in mismatches:
            writer.writerow(m)
        print(f"\n# {disclaimer}", file=sys.stderr)
    else:
        # table format
        print(f"\n=== Classification Accuracy Report ===")
        print(f"Total documents: {len(docs)}")
        print(f"Compared (with folder info): {total_compared}")
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Accuracy: {accuracy:.1f}%")
        if mismatches:
            print(f"\n--- Mismatches ({len(mismatches)}) ---")
            for m in mismatches[:50]:
                print(f"  {m['doc_id']}  folder={m['folder']}  classified={m['classified']}  file={m['file']}")
            if len(mismatches) > 50:
                print(f"  ... and {len(mismatches) - 50} more")
        print(f"\n{disclaimer}")


# ---------------------------------------------------------------------------
# Health command — system health check
# ---------------------------------------------------------------------------

def cmd_health(args: argparse.Namespace) -> None:
    """Check system health: Gemini API, ChromaDB, Registry, OCR, watch dirs."""
    checks: list[tuple[str, str, str]] = []  # (name, status, detail)

    # 1. Gemini API key
    strict = getattr(args, "strict", False)
    api_key = settings.gemini.api_key
    if api_key:
        if strict:
            # Verify API key by making a lightweight call
            try:
                from doc_pipeline.processor.llm import create_client
                client = create_client(api_key)
                client.models.list(config={"page_size": 1})
                checks.append(("Gemini API", "OK", f"...{api_key[-4:]} (verified)"))
            except Exception as exc:
                checks.append(("Gemini API", "ERROR", f"...{api_key[-4:]} — {exc}"))
        else:
            checks.append(("Gemini API Key", "OK", f"...{api_key[-4:]}"))
    else:
        checks.append(("Gemini API Key", "MISSING", "GEMINI_API_KEY not set"))

    # 2. ChromaDB
    try:
        from doc_pipeline.storage.vectordb import VectorStore
        store = VectorStore(persist_dir=settings.chroma.persist_dir)
        count = store.count
        checks.append(("ChromaDB", "OK", f"{count} chunks"))
    except Exception as exc:
        checks.append(("ChromaDB", "ERROR", str(exc)))

    # 3. Registry DB
    try:
        from doc_pipeline.storage.registry import DocumentRegistry
        reg = DocumentRegistry(db_path=settings.registry.db_path)
        doc_count = reg.document_count
        checks.append(("Registry DB", "OK", f"{doc_count} documents"))
    except Exception as exc:
        checks.append(("Registry DB", "ERROR", str(exc)))

    # 4. Watch directories
    watch_dirs = [
        d for d in [
            settings.watch.contracts_dir,
            settings.watch.action_plans_dir,
            settings.watch.opinions_dir,
        ] if d
    ]
    if watch_dirs:
        for d in watch_dirs:
            p = Path(d)
            if p.exists() and p.is_dir():
                checks.append(("Watch Dir", "OK", str(p)))
            else:
                checks.append(("Watch Dir", "MISSING", str(p)))
    else:
        checks.append(("Watch Dirs", "NONE", "No watch directories configured"))

    # 5. OCR Engine
    try:
        from doc_pipeline.processor.ocr import get_engine
        engine = get_engine(settings.ocr_engine)
        checks.append(("OCR Engine", "OK", type(engine).__name__))
    except Exception as exc:
        checks.append(("OCR Engine", "ERROR", str(exc)))

    # Output
    print("\n=== System Health Check ===")
    all_ok = True
    for name, status, detail in checks:
        icon = "OK" if status == "OK" else "!!"
        if status not in ("OK", "NONE"):
            all_ok = False
        print(f"  [{icon}] {name}: {status} — {detail}")
    print()
    import sys
    if all_ok:
        print("All checks passed.")
        sys.exit(0)
    else:
        print("Some checks failed. Review the output above.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helper: resolve file path for a registry document
# ---------------------------------------------------------------------------

def _resolve_doc_file_path(doc: dict) -> Path | None:
    """Resolve the best available file path for a document record."""
    for key in ("managed_path", "source_path"):
        path_str = doc.get(key, "")
        if path_str and path_str != "(Nova Web Upload)":
            p = Path(path_str)
            if p.exists():
                return p
    return None


# ---------------------------------------------------------------------------
# Search command
# ---------------------------------------------------------------------------

def cmd_search(args: argparse.Namespace) -> None:
    """Search the vector DB for similar documents."""
    from doc_pipeline.processor.llm import create_client, get_embeddings
    from doc_pipeline.search import unified_search
    from doc_pipeline.storage.vectordb import VectorStore

    client = create_client(settings.gemini.api_key)
    store = VectorStore(persist_dir=settings.chroma.persist_dir)

    # Build query parser for metadata-aware search
    query_parser = None
    try:
        from doc_pipeline.config.type_registry import get_type_registry
        from doc_pipeline.search.query_parser import QueryParser

        type_keywords = get_type_registry().get_keywords_map()
        query_parser = QueryParser(type_keywords=type_keywords)
    except Exception:
        pass

    query_emb = get_embeddings(client, [args.query])[0]
    doc_results, parsed = unified_search(
        store, args.query, query_emb,
        n_results=args.top_k,
        query_parser=query_parser,
    )

    if not doc_results:
        print("No results found.")
        return

    if parsed and (parsed.project or parsed.year or parsed.doc_type):
        print(f"  [파싱] 프로젝트={parsed.project or '-'}, 연도={parsed.year or '-'}, 유형={parsed.doc_type or '-'}")

    for i, doc_res in enumerate(doc_results, 1):
        print(f"\n[{i}] {doc_res.doc_type} — {doc_res.project_name} (점수: {doc_res.doc_score:.4f}, 청크: {doc_res.chunk_count}개)")
        print(f"    {doc_res.best_chunk.text[:200]}...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Document Pipeline CLI")
    sub = parser.add_subparsers(dest="command")

    # process
    p_proc = sub.add_parser("process", help="Process document files (PDF/DOCX/PPTX/DOC/PPT)")
    p_proc.add_argument("path", help="Document file or directory path")
    p_proc.add_argument("--grade", choices=["A", "B", "C"], help="Security grade override")
    p_proc.add_argument("--no-embed", action="store_true", help="Skip vector embedding")

    # batch
    p_batch = sub.add_parser("batch", help="Batch process document directory")
    p_batch.add_argument("path", help="Directory with document files")
    p_batch.add_argument("--grade", choices=["A", "B", "C"], help="Security grade override")
    p_batch.add_argument("--limit", type=int, default=None, help="Max files to process")
    p_batch.add_argument("--no-embed", action="store_true", help="Skip vector embedding")
    p_batch.add_argument("--output", default="data/batch_results.json", help="Output JSON path")

    # draft
    p_draft = sub.add_parser("draft", help="Generate document draft")
    p_draft.add_argument("--type", required=True, choices=["의견서", "조치계획서"], help="Document type")
    p_draft.add_argument("--project", required=True, help="Project name")
    p_draft.add_argument("--issue", required=True, help="Main issue description")
    p_draft.add_argument("--output", default=None, help="Output file path (prints to stdout if omitted)")
    p_draft.add_argument("--no-llm", action="store_true", help="Skip LLM generation, use template only")

    # watch
    p_watch = sub.add_parser("watch", help="Watch directories for new documents")
    p_watch.add_argument("path", nargs="?", default=None, help="Directory to watch (or use env vars)")
    p_watch.add_argument("--grade", choices=["A", "B", "C"], help="Security grade override")
    p_watch.add_argument("--cooldown", type=float, default=5.0, help="Cooldown seconds before processing")

    # search
    p_search = sub.add_parser("search", help="Search documents")
    p_search.add_argument("query", help="Search query text")
    p_search.add_argument("--top-k", type=int, default=5, help="Number of results")

    # embed
    p_embed = sub.add_parser("embed", help="Embed registered documents into vector DB")
    p_embed.add_argument("--doc-id", default=None, help="Specific document ID")
    p_embed.add_argument("--all", action="store_true", dest="embed_all", help="Embed all unembedded documents")
    p_embed.add_argument("--grade", choices=["B", "C"], default="B", help="Embedding grade (default: B)")
    p_embed.add_argument("--limit", type=int, default=None, help="Max documents to embed")
    p_embed.add_argument("--dry-run", action="store_true", help="Show targets without embedding")
    p_embed.add_argument("--force", action="store_true", help="Re-embed all documents (ignore embedded_at)")

    # reclassify
    p_reclass = sub.add_parser("reclassify", help="Reclassify registered documents")
    p_reclass.add_argument("--doc-id", default=None, help="Specific document ID")
    p_reclass.add_argument("--all", action="store_true", dest="reclass_all", help="Reclassify all documents")
    p_reclass.add_argument("--grade", choices=["B", "C"], default="B", help="Classification grade")
    p_reclass.add_argument("--dry-run", action="store_true", help="Show changes without applying")

    # report
    p_report = sub.add_parser("report", help="Classification accuracy report")
    p_report.add_argument("--format", choices=["table", "json", "csv"], default="table", help="Output format")

    # health
    p_health = sub.add_parser("health", help="System health check")
    p_health.add_argument("--strict", action="store_true", help="Verify Gemini API with a real call")

    args = parser.parse_args()

    if args.command == "process":
        cmd_process(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "draft":
        cmd_draft(args)
    elif args.command == "watch":
        cmd_watch(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "embed":
        cmd_embed(args)
    elif args.command == "reclassify":
        cmd_reclassify(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "health":
        cmd_health(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
