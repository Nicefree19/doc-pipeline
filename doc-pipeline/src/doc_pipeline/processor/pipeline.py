"""Core pipeline logic for PDF document processing.

Extracts, classifies, and stores documents through the full pipeline.
"""

from __future__ import annotations

import logging
import multiprocessing
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from doc_pipeline.collector.extractor import extract_text
from doc_pipeline.config import settings
from doc_pipeline.models.schemas import (
    ActionPlanMeta,
    ChunkRecord,
    ContractMeta,
    DocMaster,
    DocType,
    OpinionMeta,
    ProcessStatus,
    SecurityGrade,
    SourceFormat,
)
from doc_pipeline.processor.masker import mask_text

logger = logging.getLogger(__name__)

# Type alias for optional Gemini client (avoids hard import of google.genai)
_Client = Any


def _ocr_worker(q: multiprocessing.Queue, eng_name: str, path: str) -> None:
    """Module-level OCR worker for multiprocessing (Windows pickle compat)."""
    try:
        from doc_pipeline.processor.ocr import get_engine
        engine = get_engine(eng_name)
        result = engine.process(Path(path))
        q.put(result.model_dump())
    except Exception as exc:
        q.put({"error": str(exc)})


def _run_ocr_isolated(
    engine_name: str,
    pdf_path: Path,
    timeout: int = 300,
) -> Any:
    """Run OCR in a subprocess to survive segfaults (e.g. Marker on Windows).

    Returns:
        OCRResult on success, None on crash/timeout.
    """
    q: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_ocr_worker, args=(q, engine_name, str(pdf_path)))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        logger.warning("OCR subprocess timed out after %ds for %s — killing", timeout, pdf_path.name)
        proc.kill()
        proc.join(5)
        return None

    if proc.exitcode != 0:
        logger.warning("OCR subprocess crashed (exitcode=%d) for %s", proc.exitcode, pdf_path.name)
        return None

    if q.empty():
        logger.warning("OCR subprocess produced no output for %s", pdf_path.name)
        return None

    data = q.get_nowait()
    if isinstance(data, dict) and "error" in data:
        logger.warning("OCR subprocess error for %s: %s", pdf_path.name, data["error"])
        return None

    # Reconstruct OCRResult from dict
    try:
        from doc_pipeline.processor.ocr import OCRResult
        return OCRResult(**data)
    except Exception:
        logger.warning("Failed to reconstruct OCRResult for %s", pdf_path.name, exc_info=True)
        return None


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file using streaming reads."""
    import hashlib

    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class PipelineResult:
    """Result of processing a single document through the pipeline."""

    doc: DocMaster | None = None
    chunks_stored: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    skipped: bool = False
    skip_reason: str = ""
    error: str = ""


def _chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> list[str]:
    """Split text into overlapping chunks."""
    step = max(chunk_size - chunk_overlap, 1)
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def process_pdf(
    pdf_path: Path,
    grade: SecurityGrade | None = None,
    *,
    no_embed: bool = False,
    output_dir: Path | None = None,
    save_sheets: bool = True,
    source_dir: str | None = None,
    client: _Client = None,
) -> PipelineResult:
    """Process a single PDF through the full pipeline.

    Steps:
        0. Determine security grade
        1. Extract text from PDF
        2. OCR if scanned
        3. Mask PII
        4. Classify & extract metadata (grade-aware)
        5. Generate standard filename
        6. Create DocMaster record
        7. Save to Google Sheets (optional)
        8. Store vector embeddings (B/C-grade, optional)

    Args:
        pdf_path: Path to the PDF file.
        grade: Security grade override. Uses settings default if None.
        no_embed: If True, skip vector embedding step.
        output_dir: Directory for standard naming collision check.
                    Defaults to pdf_path.parent.
        save_sheets: If True, save metadata to Google Sheets.
        source_dir: Override for file_path_nas in DocMaster record.
                    Use when pdf_path is a temporary copy (e.g. web upload).
                    Defaults to str(pdf_path.parent) if not provided.
        client: Pre-created Gemini client. If None, created once internally.

    Returns:
        PipelineResult with processing details.
    """
    result = PipelineResult()

    # -- STEP 0: Determine security grade --
    if grade is None:
        grade = SecurityGrade(settings.security.default_grade)

    if grade == SecurityGrade.A:
        result.skipped = True
        result.skip_reason = "Security grade A — skipping all AI processing"
        logger.info(result.skip_reason)
        return result

    # -- STEP 0.5: Deduplication check --
    if settings.registry.enabled:
        try:
            from doc_pipeline.storage.registry import DocumentRegistry

            file_hash = _compute_file_hash(pdf_path)
            registry = DocumentRegistry(db_path=settings.registry.db_path)
            existing = registry.find_by_hash(file_hash)
            if existing:
                result.skipped = True
                result.skip_reason = (
                    f"중복 문서: "
                    f"{existing['file_name_standard'] or existing['file_name_original']} "
                    f"(doc_id={existing['doc_id']})"
                )
                logger.info(result.skip_reason)
                return result
        except Exception:
            logger.warning("Registry dedup check failed — continuing", exc_info=True)

    # -- STEP 1: Extract text --
    extraction = extract_text(pdf_path)
    logger.info(
        "Extracted %d chars, %d pages (scanned: %s)",
        len(extraction.text),
        extraction.page_count,
        extraction.is_scanned,
    )

    # -- STEP 2: OCR if scanned --
    ocr_engine_name = "none"
    ocr_blocks: list | None = None
    text = extraction.text
    if extraction.is_scanned:
        ocr_result = _run_ocr_isolated(settings.ocr_engine, pdf_path, settings.ocr_timeout)
        if ocr_result is not None:
            text = ocr_result.text
            ocr_engine_name = ocr_result.engine
            if ocr_result.blocks:
                ocr_blocks = ocr_result.blocks
            logger.info(
                "OCR completed: %d chars, %d blocks in %.1fs",
                len(text),
                len(ocr_result.blocks),
                ocr_result.elapsed_seconds,
            )
        else:
            logger.warning(
                "OCR failed/crashed for %s — using adapter text (%d chars)",
                pdf_path.name, len(text),
            )

    # -- STEP 3: Mask PII --
    masked_text = mask_text(text)

    # -- Create Gemini client once (shared across classify + embed steps) --
    # C-grade needs client for classify + embed; B-grade uses local embedding only
    if client is None and grade == SecurityGrade.C:
        from doc_pipeline.processor.llm import create_client
        client = create_client(settings.gemini.api_key)

    # -- STEP 4: Classify & extract metadata (grade-aware) --
    doc_type, project_name, year, metadata, summary = _classify_and_extract(
        grade, pdf_path, text, masked_text, client=client,
    )

    result.metadata = metadata
    result.summary = summary

    # -- STEP 5: Standard file name --
    # Use extended type name if available, fall back to legacy
    doc_type_ext = metadata.get("_doc_type_extended", doc_type.value)
    category = _resolve_category(doc_type_ext)
    display_type = doc_type_ext if doc_type_ext != doc_type.value else doc_type.value

    target_dir = output_dir or pdf_path.parent
    file_name_standard = _next_standard_name(
        year, project_name, display_type, target_dir,
    )

    # -- STEP 6: Create master record --
    doc = DocMaster(
        file_name_original=pdf_path.name,
        file_name_standard=file_name_standard,
        file_path_nas=source_dir or str(pdf_path.parent),
        doc_type=doc_type,
        doc_type_ext=doc_type_ext,
        category=category,
        project_name=project_name,
        year=year,
        page_count=extraction.page_count,
        ocr_engine=ocr_engine_name,
        process_date=datetime.now(),
        process_status=ProcessStatus.COMPLETED,
        security_grade=grade,
        summary=summary,
        source_format=SourceFormat.PDF,
    )
    result.doc = doc
    logger.info(
        "Processed: %s -> %s [%s]",
        doc.file_name_original,
        doc.file_name_standard,
        doc.doc_type.value,
    )

    # -- STEP 6.5: Save to Document Registry --
    metadata["extracted_text"] = masked_text
    _save_to_registry(
        doc, pdf_path, metadata,
        source_path_override=source_dir or "",
    )

    # -- STEP 7: Save to Google Sheets --
    if save_sheets and settings.sheets.spreadsheet_id:
        _save_to_sheets(doc, grade, doc_type, metadata)

    # -- STEP 8: Store vector embeddings --
    if not no_embed:
        if grade == SecurityGrade.C:
            # C-grade: embed via Gemini API (external)
            result.chunks_stored = _store_embeddings(
                doc, doc_type, project_name, year, grade, masked_text,
                client=client, ocr_blocks=ocr_blocks,
            )
        elif grade == SecurityGrade.B:
            # B-grade: embed locally (no external API call)
            result.chunks_stored = _store_embeddings_local(
                doc, doc_type, project_name, year, grade, masked_text,
                ocr_blocks=ocr_blocks,
            )

    return result


def _classify_and_extract(
    grade: SecurityGrade,
    pdf_path: Path,
    raw_text: str,
    masked_text: str,
    *,
    client: _Client = None,
) -> tuple[DocType, str, int, dict[str, Any], str]:
    """Classify document and extract metadata based on security grade.

    Uses the 3-stage classifier chain (Rule -> Keyword -> LLM)
    and generic metadata extractor from YAML config.

    Returns:
        (doc_type, project_name, year, metadata, summary)
    """
    from doc_pipeline.processor.classifier import classify_document_chain

    classification = classify_document_chain(
        pdf_path, masked_text if grade == SecurityGrade.C else raw_text,
        client=client,
        grade=grade.value,
    )
    doc_type_str = classification.doc_type
    project_name = classification.project_name
    year = classification.year
    doc_type = _resolve_doc_type(doc_type_str)

    metadata: dict[str, Any] = {}
    summary = ""

    if grade == SecurityGrade.C and client is not None:
        from doc_pipeline.processor.generic_extractor import extract_metadata_generic
        from doc_pipeline.processor.llm import generate_summary

        metadata = extract_metadata_generic(client, masked_text, doc_type_str)
        summary = generate_summary(client, masked_text)
    else:
        metadata = {"note": "Grade B - local classification only"}

    # Store extended classification info in metadata
    metadata["_doc_type_extended"] = doc_type_str
    metadata["_classification_method"] = classification.method
    metadata["_classification_confidence"] = classification.confidence
    if classification.secondary_types:
        metadata["_secondary_types"] = classification.secondary_types

    return doc_type, project_name, year, metadata, summary


def _resolve_doc_type(doc_type_str: str) -> DocType:
    """Resolve doc_type string to DocType enum, defaulting to OPINION."""
    if doc_type_str in [e.value for e in DocType]:
        return DocType(doc_type_str)
    # Try resolving via TypeRegistry for extended types
    try:
        from doc_pipeline.config.type_registry import get_type_registry
        registry = get_type_registry()
        legacy_value = registry.resolve_to_legacy(doc_type_str)
        return DocType(legacy_value)
    except Exception:
        pass
    return DocType.OPINION


def _resolve_category(doc_type_ext: str) -> str:
    """Resolve extended type name to its category key."""
    try:
        from doc_pipeline.config.type_registry import get_type_registry
        registry = get_type_registry()
        td = registry.get(doc_type_ext)
        return td.category if td else ""
    except Exception:
        return ""


def _sanitize_name_component(name: str) -> str:
    """Sanitize a name component (project_name, doc_type) for use in filenames.

    Ensures the generated file_name_standard is safe for all filesystems
    and will match the managed storage basename after _sanitize_filename().
    """
    # Remove characters unsafe on Windows/Linux filesystems
    name = name.replace("/", "_").replace("\\", "_").replace("..", "")
    name = name.replace("\x00", "").replace(":", "_").replace("*", "")
    name = name.replace("?", "").replace('"', "").replace("<", "").replace(">", "").replace("|", "")
    # Collapse multiple underscores
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_. ") or "unknown"


def _next_standard_name(
    year: int,
    project_name: str,
    doc_type_str: str,
    output_dir: Path,
    extension: str = ".pdf",
) -> str:
    """Generate file_name_standard with collision-safe sequence number.

    Scans existing files in output_dir to determine the next sequence number.
    Returns a name string only — no files are created.
    """
    safe_project = _sanitize_name_component(project_name)
    safe_doc_type = _sanitize_name_component(doc_type_str)
    prefix = f"{year}-{safe_project}-{safe_doc_type}-"
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.glob(f"{prefix}*{extension}"))
    seq = len(existing) + 1

    # Check for collision and increment if needed
    for _ in range(20):
        candidate = f"{prefix}{seq:03d}{extension}"
        if not (output_dir / candidate).exists():
            return candidate
        seq += 1

    # Fallback: use timestamp suffix if all seq numbers taken
    ts = datetime.now().strftime("%H%M%S")
    return f"{prefix}{seq:03d}-{ts}{extension}"


def _save_to_sheets(
    doc: DocMaster,
    grade: SecurityGrade,
    doc_type: DocType,
    metadata: dict[str, Any],
) -> None:
    """Save document metadata to Google Sheets."""
    try:
        from doc_pipeline.storage.sheets import SheetsClient

        sheets = SheetsClient(
            credentials_path=settings.sheets.credentials_path,
            spreadsheet_id=settings.sheets.spreadsheet_id,
        )
        sheets.append_doc_master(doc)

        if grade == SecurityGrade.C and isinstance(metadata, dict):
            try:
                _save_typed_metadata(sheets, doc.doc_id, doc_type, metadata)
            except Exception:
                logger.warning(
                    "Typed metadata save failed (Pydantic validation?)",
                    exc_info=True,
                )

        logger.info("Saved to Google Sheets")
    except Exception:
        logger.warning("Sheets storage failed - continuing without", exc_info=True)


def _save_to_registry(
    doc: DocMaster,
    file_path: Path,
    metadata: dict[str, Any],
    *,
    source_path_override: str = "",
) -> None:
    """Save document record to the SQLite registry.

    Args:
        doc: The processed DocMaster record.
        file_path: Path to the actual file (used for hashing).
        metadata: Extracted metadata dict.
        source_path_override: Logical origin path. If provided, stored as
            source_path instead of file_path (useful when file_path is a
            temporary copy, e.g. web upload).
    """
    if not settings.registry.enabled:
        return
    try:
        from doc_pipeline.storage.registry import DocumentRegistry

        registry = DocumentRegistry(db_path=settings.registry.db_path)
        file_hash = _compute_file_hash(file_path)

        # Duplicate check
        existing = registry.find_by_hash(file_hash)
        if existing:
            logger.info(
                "Duplicate detected: %s (matches %s)",
                file_path.name, existing["doc_id"],
            )
            registry.add_event(
                existing["doc_id"], "duplicate_attempt",
                f"Same file uploaded again: {file_path.name}",
            )
            return

        registry.insert_document(
            doc=doc,
            source_path=source_path_override or str(file_path),
            hash_sha256=file_hash,
            metadata=metadata,
        )
        registry.add_event(
            doc.doc_id, "processed",
            f"Pipeline completed: {doc.process_status.value}",
        )
    except Exception:
        logger.warning("Registry save failed — continuing", exc_info=True)


def _save_typed_metadata(
    sheets: Any,
    doc_id: str,
    doc_type: DocType,
    metadata: dict[str, Any],
) -> None:
    """Save typed metadata to Google Sheets.

    For legacy 3 types, uses the original Pydantic models.
    For extended types, uses the generic append_typed_metadata.
    """
    # Clean metadata dict for Pydantic (remove internal keys)
    clean = {k: v for k, v in metadata.items() if not k.startswith("_")}
    clean["doc_id"] = doc_id

    # Check if this is an extended type
    ext_type = metadata.get("_doc_type_extended", doc_type.value)
    if ext_type not in (DocType.CONTRACT.value, DocType.ACTION_PLAN.value, DocType.OPINION.value):
        # Extended type: use generic method
        sheets.append_typed_metadata(doc_id, ext_type, clean)
        return

    # Legacy types: use Pydantic models
    if doc_type == DocType.CONTRACT:
        sheets.append_contract(ContractMeta(**clean))
    elif doc_type == DocType.ACTION_PLAN:
        sheets.append_action_plan(ActionPlanMeta(**clean))
    elif doc_type == DocType.OPINION:
        sheets.append_opinion(OpinionMeta(**clean))


def _build_chunks(
    doc: DocMaster,
    doc_type: DocType,
    project_name: str,
    year: int,
    grade: SecurityGrade,
    masked_text: str,
    ocr_blocks: list | None = None,
) -> list[ChunkRecord]:
    """Split text and build ChunkRecord list.

    When *ocr_blocks* are provided (from structured OCR), uses the
    block-aware chunker which respects page/block_type boundaries.
    Otherwise falls back to the legacy character-based splitter.
    """
    chunk_size = settings.chroma.chunk_size
    chunk_overlap = settings.chroma.chunk_overlap

    # Block-aware path: OCR produced structured blocks
    if ocr_blocks:
        from doc_pipeline.processor.chunker import build_chunk_records, chunk_blocks

        block_chunks = chunk_blocks(ocr_blocks, chunk_size, chunk_overlap)
        if block_chunks:
            records = build_chunk_records(
                doc.doc_id, doc_type, project_name, year, grade, block_chunks,
            )
            # Enrich with doc-level metadata
            for r in records:
                r.doc_type_ext = getattr(doc, "doc_type_ext", "")
                r.category = getattr(doc, "category", "")
            logger.info(
                "Built %d block-aware chunks (vs %d flat)",
                len(records),
                len(_chunk_text(masked_text, chunk_size, chunk_overlap)),
            )
            max_chunks = settings.chroma.max_chunks_per_doc
            if len(records) > max_chunks:
                logger.warning(
                    "Chunk guard: doc_id=%s produced %d chunks, truncating to %d",
                    doc.doc_id, len(records), max_chunks,
                )
                records = records[:max_chunks]
            return records

    # Legacy flat path
    text_chunks = _chunk_text(masked_text, chunk_size, chunk_overlap)

    chunks_list = [
        ChunkRecord(
            chunk_id=f"{doc.doc_id}_{i}",
            doc_id=doc.doc_id,
            doc_type=doc_type,
            doc_type_ext=getattr(doc, "doc_type_ext", ""),
            category=getattr(doc, "category", ""),
            project_name=project_name,
            year=year,
            chunk_index=i,
            text=chunk_text,
            security_grade=grade,
        )
        for i, chunk_text in enumerate(text_chunks)
    ]
    max_chunks = settings.chroma.max_chunks_per_doc
    if len(chunks_list) > max_chunks:
        logger.warning(
            "Chunk guard: doc_id=%s produced %d chunks, truncating to %d",
            doc.doc_id, len(chunks_list), max_chunks,
        )
        chunks_list = chunks_list[:max_chunks]
    return chunks_list


def _store_embeddings(
    doc: DocMaster,
    doc_type: DocType,
    project_name: str,
    year: int,
    grade: SecurityGrade,
    masked_text: str,
    *,
    client: _Client = None,
    ocr_blocks: list | None = None,
) -> int:
    """Chunk text, generate Gemini embeddings, and store (C-grade).

    Returns:
        Number of chunks stored.
    """
    from doc_pipeline.processor.llm import create_client, get_embeddings
    from doc_pipeline.storage.vectordb import VectorStore

    if client is None:
        client = create_client(settings.gemini.api_key)
    store = VectorStore(persist_dir=settings.chroma.persist_dir)

    chunks = _build_chunks(doc, doc_type, project_name, year, grade, masked_text, ocr_blocks)

    if chunks:
        embeddings = get_embeddings(client, [c.text for c in chunks])
        store.upsert_chunks(chunks, embeddings)
        logger.info("Stored %d C-grade chunks via Gemini embedding", len(chunks))

    return len(chunks)


def _store_embeddings_local(
    doc: DocMaster,
    doc_type: DocType,
    project_name: str,
    year: int,
    grade: SecurityGrade,
    masked_text: str,
    ocr_blocks: list | None = None,
) -> int:
    """Chunk text and store with local embedding (B-grade, no external API).

    Returns:
        Number of chunks stored.
    """
    from doc_pipeline.storage.vectordb import VectorStore

    store = VectorStore(persist_dir=settings.chroma.persist_dir)
    chunks = _build_chunks(doc, doc_type, project_name, year, grade, masked_text, ocr_blocks)

    if chunks:
        store.upsert_chunks_local(chunks)
        logger.info("Stored %d B-grade chunks via local embedding", len(chunks))

    return len(chunks)


# ---------------------------------------------------------------------------
# Batch embedding for already-registered documents
# ---------------------------------------------------------------------------


def _clear_opposite_collection(
    store: Any, doc_id: str, grade: SecurityGrade,
) -> int:
    """Remove all chunks for *doc_id* from the collection NOT used by *grade*.

    B-grade writes to ``_local_collection`` → clear ``_collection`` (API).
    C-grade writes to ``_collection`` → clear ``_local_collection`` (local).

    This enforces a single-source-of-truth: a document's chunks live in
    exactly one collection after re-embedding.
    """
    opposite = store._collection if grade == SecurityGrade.B else store._local_collection
    try:
        existing = opposite.get(where={"doc_id": doc_id})
        if existing and existing["ids"]:
            opposite.delete(ids=existing["ids"])
            removed = len(existing["ids"])
            logger.info(
                "Cleared %d chunks from opposite collection for doc_id=%s (grade=%s)",
                removed, doc_id, grade.value,
            )
            return removed
    except Exception as exc:
        logger.warning(
            "Failed to clear opposite collection for %s: %s", doc_id, exc,
        )
    return 0


def _prune_stale_chunks(store: Any, doc_id: str, new_chunk_ids: set[str]) -> int:
    """Remove chunks belonging to doc_id that are NOT in new_chunk_ids.

    Replace semantics: upsert succeeded → now prune leftover stale chunks.
    If new_chunk_ids is empty, this is a no-op to prevent accidental total deletion.
    """
    if not new_chunk_ids:
        return 0
    pruned = 0
    for col in (store._collection, store._local_collection):
        try:
            existing = col.get(where={"doc_id": doc_id})
            if not existing or not existing["ids"]:
                continue
            stale_ids = [cid for cid in existing["ids"] if cid not in new_chunk_ids]
            if stale_ids:
                col.delete(ids=stale_ids)
                pruned += len(stale_ids)
        except Exception as exc:
            logger.warning("Stale chunk prune failed for %s in %s: %s", doc_id, col.name, exc)
    if pruned:
        logger.info("Pruned %d stale chunks for doc_id=%s", pruned, doc_id)
    return pruned


def embed_document(
    doc_record: dict[str, Any],
    file_path: Path,
    grade: SecurityGrade,
    *,
    client: _Client = None,
    store: Any = None,
    registry: Any = None,
) -> int:
    """Re-extract text from a registered document and embed into vector DB.

    Skips OCR/classification — only does text extraction, PII masking,
    chunking, and embedding. Updates registry ``embedded_at`` timestamp.

    Args:
        doc_record: Dict from ``registry.get_document()``.
        file_path: Path to the document file on disk.
        grade: Security grade for embedding strategy.
        client: Gemini client (needed for C-grade only).
        store: Optional shared VectorStore instance (avoids per-call creation).
        registry: Optional shared DocumentRegistry instance.

    Returns:
        Number of chunks stored.
    """
    from doc_pipeline.collector.adapters import get_adapter
    from doc_pipeline.storage.registry import DocumentRegistry

    # 1. Extract text
    adapter = get_adapter(file_path)
    normalized = adapter.extract(file_path)
    text = normalized.text

    # Track embed failure (single record per invocation)
    embed_error: tuple[str, str] | None = None  # (error_type, error_msg)

    # 2. OCR for scanned PDFs (capture blocks for block-aware chunking)
    ocr_blocks = None
    if normalized.is_scanned and file_path.suffix.lower() == ".pdf":
        ocr_result = _run_ocr_isolated(settings.ocr_engine, file_path, settings.ocr_timeout)
        if ocr_result is not None:
            text = ocr_result.text
            ocr_blocks = getattr(ocr_result, "blocks", None) or None
        else:
            embed_error = ("ocr_timeout", f"OCR timeout {settings.ocr_timeout}s")
            logger.warning(
                "OCR failed/crashed for %s — using adapter text (%d chars)",
                file_path.name, len(text),
            )

    # 3. Fallback to cached extracted_text if no text available
    if not text.strip():
        try:
            _reg = registry if registry is not None else DocumentRegistry(db_path=settings.registry.db_path)
            cached_meta = _reg.get_metadata(doc_record["doc_id"])
            if cached_meta:
                cached_text = cached_meta.get("metadata", {}).get("extracted_text", "")
                if cached_text and cached_text.strip():
                    text = cached_text
                    logger.info("Using cached extracted_text for %s (%d chars)", file_path.name, len(text))
        except Exception:
            logger.debug("Failed to retrieve cached text for %s", file_path.name, exc_info=True)

    if not text.strip():
        # Final failure: no text from any source — record once
        if embed_error is None:
            embed_error = ("no_text", "No text after OCR+adapter+cache")
        try:
            _reg = registry if registry is not None else DocumentRegistry(db_path=settings.registry.db_path)
            _reg.update_embed_failure(doc_record["doc_id"], embed_error[0], embed_error[1])
        except Exception:
            logger.debug("Failed to record embed failure", exc_info=True)
        logger.warning("No text extracted for embedding: %s", file_path.name)
        return 0

    # 4. PII masking
    masked_text = mask_text(text)

    # 4.5 Cache extracted_text if not already stored
    try:
        _reg = registry if registry is not None else DocumentRegistry(db_path=settings.registry.db_path)
        cached_meta = _reg.get_metadata(doc_record["doc_id"])
        if cached_meta:
            existing_text = cached_meta.get("metadata", {}).get("extracted_text", "")
            if not existing_text:
                meta_dict = cached_meta.get("metadata", {})
                meta_dict["extracted_text"] = masked_text
                _reg.save_metadata(doc_record["doc_id"], meta_dict)
                logger.debug("Cached extracted_text for %s", doc_record["doc_id"])
        else:
            _reg.save_metadata(doc_record["doc_id"], {"extracted_text": masked_text})
            logger.debug("Saved initial extracted_text for %s", doc_record["doc_id"])
    except Exception:
        logger.debug("Failed to cache extracted_text for %s", doc_record["doc_id"], exc_info=True)

    # 4. Build chunks using doc_record metadata
    doc_type = _resolve_doc_type(doc_record.get("doc_type", "의견서"))
    doc = DocMaster(
        doc_id=doc_record["doc_id"],
        file_name_original=doc_record.get("file_name_original", file_path.name),
        file_name_standard=doc_record.get("file_name_standard", ""),
        doc_type=doc_type,
        doc_type_ext=doc_record.get("doc_type_ext", ""),
        project_name=doc_record.get("project_name", ""),
        year=doc_record.get("year", 0),
        security_grade=grade,
    )

    chunks = _build_chunks(
        doc, doc_type,
        doc_record.get("project_name", ""),
        doc_record.get("year", 0),
        grade, masked_text,
        ocr_blocks=ocr_blocks,
    )

    if not chunks:
        return 0

    # 5. Store embeddings (reuse injected store or create one)
    if store is None:
        from doc_pipeline.storage.vectordb import VectorStore
        store = VectorStore(persist_dir=settings.chroma.persist_dir)

    # 5a. Collect new chunk IDs for replace semantics
    doc_id = doc_record["doc_id"]
    new_chunk_ids = {c.chunk_id for c in chunks}

    # 5b. Upsert new chunks into the target collection FIRST
    # This must succeed before any cleanup — if it fails, old chunks
    # (in either collection) remain intact for search continuity.
    if grade == SecurityGrade.B:
        store.upsert_chunks_local(chunks)
    elif grade == SecurityGrade.C:
        from doc_pipeline.processor.llm import get_embeddings
        if client is None:
            from doc_pipeline.processor.llm import create_client
            client = create_client(settings.gemini.api_key)
        embeddings = get_embeddings(client, [c.text for c in chunks])
        store.upsert_chunks(chunks, embeddings)

    # 5c. Clear opposite collection (enforce single-source-of-truth)
    # Only runs after target upsert succeeded — prevents data loss
    # if embedding/upsert fails (old chunks survive in opposite collection).
    _clear_opposite_collection(store, doc_id, grade)

    # 5d. Prune stale chunks in the target collection (same-grade leftovers)
    _prune_stale_chunks(store, doc_id, new_chunk_ids)

    # 5e. Sync FTS (delete all for doc → re-upsert new)
    try:
        from doc_pipeline.storage.vectordb import ChunkFTS
        chunk_fts = ChunkFTS(db_path=settings.fts.db_path)
        chunk_fts.delete_by_doc_ids([doc_id])
        chunk_fts.upsert(chunks)
    except Exception:
        logger.debug("FTS sync skipped for %s", doc_id, exc_info=True)

    logger.info("Embedded %d chunks for doc_id=%s", len(chunks), doc_id)

    # 5f. Clear embed failure metadata on success
    try:
        _reg = registry if registry is not None else DocumentRegistry(db_path=settings.registry.db_path)
        existing_meta = _reg.get_metadata(doc_id)
        if existing_meta:
            meta = existing_meta.get("metadata", {})
            if "embed_error_type" in meta:
                for key in ("embed_error_type", "embed_error_msg",
                            "embed_attempts", "last_embed_error_at"):
                    meta.pop(key, None)
                _reg.save_metadata(
                    doc_id, meta,
                    structured=existing_meta.get("structured_fields", {}),
                )
    except Exception:
        pass  # Non-critical

    # 6. Update registry embedded_at (reuse injected registry or create one)
    if registry is None:
        registry = DocumentRegistry(db_path=settings.registry.db_path)

    embedded_at_updated = False
    for attempt in range(2):
        try:
            registry.update_document(
                doc_record["doc_id"],
                embedded_at=datetime.now().isoformat(),
                process_status="인덱싱완료",
            )
            embedded_at_updated = True
            break
        except Exception:
            if attempt == 0:
                logger.warning("Retrying embedded_at update for %s", doc_record["doc_id"])
            else:
                logger.error(
                    "Failed to update embedded_at for %s after 2 attempts. "
                    "Document may be re-embedded on next --all run.",
                    doc_record["doc_id"], exc_info=True,
                )

    if not embedded_at_updated:
        try:
            registry.add_event(
                doc_record["doc_id"], "embed_warning",
                "Chunks stored but embedded_at update failed",
            )
        except Exception:
            pass

    return len(chunks)


# ---------------------------------------------------------------------------
# Reclassify existing documents
# ---------------------------------------------------------------------------


def reclassify_document(
    doc_record: dict[str, Any],
    file_path: Path,
    grade: SecurityGrade,
    *,
    client: _Client = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Re-classify a registered document using the current classification rules.

    Re-extracts text from file, runs the classifier chain, and optionally
    updates the registry with new classification fields.

    Args:
        dry_run: If True, compute new classification but do NOT write to registry.

    Returns:
        Dict with old and new classification info.
    """
    from doc_pipeline.collector.adapters import get_adapter

    # 1. Extract text
    adapter = get_adapter(file_path)
    normalized = adapter.extract(file_path)
    text = normalized.text

    # OCR for scanned PDFs
    if normalized.is_scanned and file_path.suffix.lower() == ".pdf":
        ocr_result = _run_ocr_isolated(settings.ocr_engine, file_path, settings.ocr_timeout)
        if ocr_result is not None:
            text = ocr_result.text
        else:
            logger.warning(
                "OCR failed/crashed for %s during reclassify — using adapter text (%d chars)",
                file_path.name, len(text),
            )

    if not text.strip():
        return {"changed": False, "reason": "no_text"}

    masked_text = mask_text(text)

    # 2. Classify
    doc_type, project_name, year, metadata, summary = _classify_and_extract(
        grade, file_path, text, masked_text, client=client,
    )

    doc_type_ext = metadata.get("_doc_type_extended", doc_type.value)
    category = _resolve_category(doc_type_ext)

    old_type = doc_record.get("doc_type_ext") or doc_record.get("doc_type", "")
    new_type = doc_type_ext

    # 3. Generate new standard filename
    # Use managed_path directory for sequence uniqueness (avoid collision domain mismatch)
    managed_path = doc_record.get("managed_path", "")
    target_dir = Path(managed_path).parent if managed_path else file_path.parent
    display_type = doc_type_ext if doc_type_ext != doc_type.value else doc_type.value
    file_name_standard = _next_standard_name(
        year, project_name, display_type, target_dir,
        extension=file_path.suffix.lower(),
    )

    # 4. Build result (always returned, even in dry_run)
    result = {
        "changed": old_type != new_type,
        "old_type": old_type,
        "new_type": new_type,
        "old_project": doc_record.get("project_name", ""),
        "new_project": project_name,
        "old_year": doc_record.get("year", 0),
        "new_year": year,
        "method": metadata.get("_classification_method", ""),
        "confidence": metadata.get("_classification_confidence", 0.0),
    }

    if dry_run:
        return result

    # 5. Rename physical file if managed_path exists (keep disk & registry in sync)
    new_managed_path = ""
    if managed_path and Path(managed_path).exists():
        old_managed = Path(managed_path)
        new_managed = old_managed.parent / file_name_standard
        if old_managed != new_managed:
            try:
                old_managed.rename(new_managed)
                new_managed_path = str(new_managed)
                logger.info("Renamed managed file: %s -> %s", old_managed.name, new_managed.name)
            except OSError:
                logger.warning(
                    "Failed to rename managed file %s; registry will be updated without rename",
                    old_managed.name, exc_info=True,
                )

    # 6. Persist to registry (skipped when dry_run=True)
    try:
        from doc_pipeline.storage.registry import DocumentRegistry

        registry = DocumentRegistry(db_path=settings.registry.db_path)
        update_fields: dict[str, Any] = {
            "doc_type": doc_type.value,
            "doc_type_ext": doc_type_ext,
            "category": category,
            "classification_method": metadata.get("_classification_method", ""),
            "classification_confidence": metadata.get("_classification_confidence", 0.0),
            "project_name": project_name,
            "year": year,
            "file_name_standard": file_name_standard,
        }
        if new_managed_path:
            update_fields["managed_path"] = new_managed_path
        registry.update_document(doc_record["doc_id"], **update_fields)
        registry.add_event(
            doc_record["doc_id"], "reclassified",
            f"{old_type} -> {new_type}",
        )
    except Exception:
        logger.warning("Failed to update registry for reclassified doc %s", doc_record["doc_id"], exc_info=True)

    return result


# ---------------------------------------------------------------------------
# Multi-format document processing
# ---------------------------------------------------------------------------


def process_document(
    file_path: Path,
    grade: SecurityGrade | None = None,
    *,
    no_embed: bool = False,
    output_dir: Path | None = None,
    save_sheets: bool = True,
    source_dir: str | None = None,
    client: _Client = None,
) -> PipelineResult:
    """Process any supported document format.

    Delegates to format-specific adapter for text extraction,
    then runs the shared pipeline (mask -> classify -> embed).
    For PDF, delegates directly to process_pdf() for full backward compatibility.
    """
    from doc_pipeline.collector.adapters import SUPPORTED_EXTENSIONS, get_adapter

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return PipelineResult(error=f"지원하지 않는 파일 형식: {suffix}")

    # PDF: delegate to existing process_pdf() for 100% compatibility
    if suffix == ".pdf":
        return process_pdf(
            file_path,
            grade,
            no_embed=no_embed,
            output_dir=output_dir,
            save_sheets=save_sheets,
            source_dir=source_dir,
            client=client,
        )

    # Non-PDF: adapter-based extraction -> shared pipeline steps
    adapter = get_adapter(file_path)
    normalized = adapter.extract(file_path)

    return _run_pipeline_steps(
        file_path=file_path,
        text=normalized.text,
        page_count=normalized.page_count,
        source_format=SourceFormat(normalized.source_format),
        is_scanned=normalized.is_scanned,
        grade=grade,
        no_embed=no_embed,
        output_dir=output_dir,
        save_sheets=save_sheets,
        source_dir=source_dir,
        client=client,
    )


def _run_pipeline_steps(
    file_path: Path,
    text: str,
    page_count: int,
    source_format: SourceFormat,
    is_scanned: bool,
    grade: SecurityGrade | None,
    *,
    no_embed: bool,
    output_dir: Path | None,
    save_sheets: bool,
    source_dir: str | None,
    client: _Client,
) -> PipelineResult:
    """Shared pipeline steps for non-PDF documents (Step 3~8)."""
    result = PipelineResult()

    # Step 0: grade
    if grade is None:
        grade = SecurityGrade(settings.security.default_grade)
    if grade == SecurityGrade.A:
        result.skipped = True
        result.skip_reason = "Security grade A — skipping all AI processing"
        return result

    # Step 0.5: Deduplication check
    if settings.registry.enabled:
        try:
            from doc_pipeline.storage.registry import DocumentRegistry

            file_hash = _compute_file_hash(file_path)
            registry = DocumentRegistry(db_path=settings.registry.db_path)
            existing = registry.find_by_hash(file_hash)
            if existing:
                result.skipped = True
                result.skip_reason = (
                    f"중복 문서: "
                    f"{existing['file_name_standard'] or existing['file_name_original']} "
                    f"(doc_id={existing['doc_id']})"
                )
                logger.info(result.skip_reason)
                return result
        except Exception:
            logger.warning("Registry dedup check failed — continuing", exc_info=True)

    # Guard: empty text (image-only documents)
    if not text.strip():
        result.skipped = True
        result.skip_reason = f"추출 가능한 텍스트가 없습니다 ({source_format.value} 파일)"
        return result

    # Step 2: OCR (non-PDF scanned documents: warn only, no OCR)
    ocr_engine_name = "none"
    if is_scanned:
        logger.warning(
            "Non-PDF scanned document detected: %s (OCR not supported for %s)",
            file_path.name,
            source_format.value,
        )

    # Step 3: Mask PII
    masked_text = mask_text(text)

    # Create Gemini client if needed
    if client is None and grade == SecurityGrade.C:
        from doc_pipeline.processor.llm import create_client

        client = create_client(settings.gemini.api_key)

    # Step 4: Classify & extract metadata
    doc_type, project_name, year, metadata, summary = _classify_and_extract(
        grade,
        file_path,
        text,
        masked_text,
        client=client,
    )
    result.metadata = metadata
    result.summary = summary

    # Step 5: Standard file name (preserve original extension)
    doc_type_ext = metadata.get("_doc_type_extended", doc_type.value)
    category = _resolve_category(doc_type_ext)
    display_type = doc_type_ext if doc_type_ext != doc_type.value else doc_type.value

    target_dir = output_dir or file_path.parent
    file_name_standard = _next_standard_name(
        year, project_name, display_type, target_dir,
        extension=file_path.suffix.lower(),
    )

    # Step 6: Create master record
    doc = DocMaster(
        file_name_original=file_path.name,
        file_name_standard=file_name_standard,
        file_path_nas=source_dir or str(file_path.parent),
        doc_type=doc_type,
        doc_type_ext=doc_type_ext,
        category=category,
        project_name=project_name,
        year=year,
        page_count=page_count,
        ocr_engine=ocr_engine_name,
        process_date=datetime.now(),
        process_status=ProcessStatus.COMPLETED,
        security_grade=grade,
        summary=summary,
        source_format=source_format,
    )
    result.doc = doc
    logger.info(
        "Processed: %s -> %s [%s] (format: %s)",
        doc.file_name_original,
        doc.file_name_standard,
        doc.doc_type.value,
        source_format.value,
    )

    # Step 6.5: Save to Document Registry
    metadata["extracted_text"] = masked_text
    _save_to_registry(
        doc, file_path, metadata,
        source_path_override=source_dir or "",
    )

    # Step 7: Save to Google Sheets
    if save_sheets and settings.sheets.spreadsheet_id:
        _save_to_sheets(doc, grade, doc_type, metadata)

    # Step 8: Store vector embeddings
    if not no_embed:
        if grade == SecurityGrade.C:
            result.chunks_stored = _store_embeddings(
                doc, doc_type, project_name, year, grade, masked_text,
                client=client,
            )
        elif grade == SecurityGrade.B:
            result.chunks_stored = _store_embeddings_local(
                doc, doc_type, project_name, year, grade, masked_text,
            )

    return result
