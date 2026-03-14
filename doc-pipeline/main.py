"""Nova API Server — FastAPI backend for Doc-Pipeline React SaaS.

Standalone (no Streamlit dependency). Singleton resources via lifespan DI.
Separated thread pools for PDF/LLM workloads.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import secrets
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from google.genai import types
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from doc_pipeline.config import settings
from doc_pipeline.generator.drafter import generate_draft
from doc_pipeline.models.schemas import SecurityGrade
from doc_pipeline.processor.llm import (
    _call_with_retry,
    _rate_limit,
    create_client,
    get_embeddings,
)
from doc_pipeline.collector.adapters import SUPPORTED_EXTENSIONS
from doc_pipeline.processor.pipeline import process_document
from doc_pipeline.storage.vectordb import VectorStore

logger = logging.getLogger("nova_api")

# Upload size limit (50 MB)
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Lifespan — singleton initialization + dedicated thread pools
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared singletons at startup, cleanup on shutdown."""
    # Activate structured JSON logging (console + rotating file)
    from doc_pipeline.config.logging_config import setup_logging

    setup_logging(
        level=settings.logging.level,
        log_dir=settings.logging.log_dir,
        log_file="nova_api.jsonl",
    )

    logger.info("Starting Nova API Server...")

    # Enforce API key in production
    if os.getenv("NOVA_ENV") == "production" and not os.getenv("NOVA_API_KEY"):
        logger.critical("NOVA_API_KEY must be set in production environment")
        raise SystemExit(1)

    # Dedicated executors to prevent cross-workload starvation
    app.state.pdf_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pdf")
    app.state.llm_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm")

    # Gemini client
    try:
        app.state.gemini_client = create_client(settings.gemini.api_key)
        logger.info("Gemini client initialized")
    except Exception as exc:
        logger.error("Gemini client init failed: %s", exc)
        app.state.gemini_client = None

    # VectorStore (ChromaDB)
    try:
        app.state.vector_store = VectorStore(persist_dir=settings.chroma.persist_dir)
        logger.info("VectorStore initialized (%d chunks)", app.state.vector_store.count)
    except Exception as exc:
        logger.error("VectorStore init failed: %s", exc)
        app.state.vector_store = None

    # Document Registry (SQLite)
    if settings.registry.enabled:
        try:
            from doc_pipeline.storage.registry import DocumentRegistry

            app.state.registry = DocumentRegistry(db_path=settings.registry.db_path)
            logger.info(
                "DocumentRegistry initialized (%d docs)",
                app.state.registry.document_count,
            )
        except Exception as exc:
            logger.error("DocumentRegistry init failed: %s", exc)
            app.state.registry = None
    else:
        app.state.registry = None

    # Query parser for metadata-aware search
    try:
        from doc_pipeline.search.query_parser import QueryParser
        from doc_pipeline.config.type_registry import get_type_registry

        known_projects: set[str] = set()
        if app.state.registry:
            known_projects = set(app.state.registry.get_unique_projects())
        tr = get_type_registry()
        type_keywords = tr.get_keywords_map()
        # Build type→category reverse map
        type_category_map: dict[str, str] = {}
        for cat, type_names in tr.categories.items():
            for tn in type_names:
                type_category_map[tn] = cat
        app.state.query_parser = QueryParser(
            known_projects=known_projects,
            type_keywords=type_keywords,
            type_category_map=type_category_map,
        )
        logger.info(
            "QueryParser initialized (%d projects, %d type keywords)",
            len(known_projects),
            len(type_keywords),
        )
    except Exception as exc:
        logger.warning("QueryParser init failed: %s", exc)
        app.state.query_parser = None

    # Chunk-level FTS index (optional, for hybrid search)
    app.state.chunk_fts = None
    if settings.fts.enabled:
        try:
            from doc_pipeline.storage.vectordb import ChunkFTS

            app.state.chunk_fts = ChunkFTS(db_path=settings.fts.db_path)
            logger.info("ChunkFTS initialized (%d chunks)", app.state.chunk_fts.count)
        except Exception as exc:
            logger.warning("ChunkFTS init failed (FTS disabled): %s", exc)

    # PydanticAI agent availability check
    if settings.agents.enabled:
        try:
            import pydantic_ai  # noqa: F401

            logger.info(
                "PydanticAI agents enabled (model=%s, temp=%s)",
                settings.agents.model,
                settings.agents.temperature,
            )
        except ImportError:
            logger.critical(
                "AGENT_ENABLED=true but pydantic-ai not installed. "
                "Install: pip install doc-pipeline[agents]"
            )
            raise SystemExit(1)

    # In production, critical services must be available
    if os.getenv("NOVA_ENV") == "production":
        if app.state.gemini_client is None:
            logger.critical("Gemini client unavailable in production — aborting")
            raise SystemExit(1)
        if app.state.vector_store is None:
            logger.critical("VectorStore unavailable in production — aborting")
            raise SystemExit(1)

    # OpenTelemetry instrumentation (optional)
    if settings.observability.otel_enabled:
        try:
            from doc_pipeline.agents.instrumentation import setup_otel

            setup_otel(settings.observability)
        except Exception as exc:
            logger.warning("OTel init failed (disabled): %s", exc)

    yield

    logger.info("Shutting down Nova API Server...")
    app.state.pdf_executor.shutdown(wait=False)
    app.state.llm_executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Doc-Pipeline Nova API",
    description="Backend API for the React-based Doc-Pipeline SaaS",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Global exception handler — prevent stack trace leakage to clients
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    # Let FastAPI handle HTTPException with proper status codes
    if isinstance(exc, HTTPException):
        raise exc
    logger.error("Unhandled exception on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "서버 내부 오류가 발생했습니다."},
    )


# ---------------------------------------------------------------------------
# Prometheus metrics (optional — graceful if package not installed)
# ---------------------------------------------------------------------------

try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
except ImportError:
    pass  # prometheus-fastapi-instrumentator not installed — skip


# ---------------------------------------------------------------------------
# Rate limiter (in-memory, IP-based)
# ---------------------------------------------------------------------------

_RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))  # requests/min
_RATE_LIMIT_UPLOAD = int(os.getenv("RATE_LIMIT_UPLOAD_PER_MIN", "5"))
# Trust X-Forwarded-For only when behind a known reverse proxy (e.g. nginx)
_TRUST_PROXY = os.getenv("TRUST_PROXY", "false").lower() in ("1", "true", "yes")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple sliding-window rate limiter per client IP."""

    def __init__(self, app, default_rpm: int = 30, upload_rpm: int = 5):
        super().__init__(app)
        self.default_rpm = default_rpm
        self.upload_rpm = upload_rpm
        self._hits: dict[str, collections.deque] = {}
        self._lock = threading.Lock()
        self._request_counter_lock = threading.Lock()
        self._request_count = 0
        self._cleanup_interval = 500

    def _client_ip(self, request: Request) -> str:
        if _TRUST_PROXY:
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _is_allowed(self, key: str, rpm: int) -> bool:
        now = time.monotonic()
        window = 60.0
        with self._lock:
            dq = self._hits.setdefault(key, collections.deque())
            while dq and dq[0] < now - window:
                dq.popleft()
            if len(dq) >= rpm:
                return False
            dq.append(now)
            return True

    def _cleanup_stale(self) -> None:
        """Remove deques that have been inactive for >120 seconds."""
        now = time.monotonic()
        stale_threshold = 120.0
        with self._lock:
            stale_keys = [
                k for k, dq in self._hits.items() if not dq or dq[-1] < now - stale_threshold
            ]
            for k in stale_keys:
                del self._hits[k]

    async def dispatch(self, request: Request, call_next):
        # Periodic cleanup of stale entries
        with self._request_counter_lock:
            self._request_count += 1
            should_cleanup = self._request_count % self._cleanup_interval == 0
        if should_cleanup:
            self._cleanup_stale()

        path = request.url.path
        # Skip rate limiting for health/root
        if path in ("/", "/api/health") or not path.startswith("/api/"):
            return await call_next(request)

        ip = self._client_ip(request)
        rpm = self.upload_rpm if path == "/api/upload" else self.default_rpm
        key = f"{ip}:{path}" if path == "/api/upload" else ip

        if not self._is_allowed(key, rpm):
            return JSONResponse(
                status_code=429,
                content={"detail": "요청이 너무 많습니다. 잠시 후 다시 시도해 주세요."},
            )
        return await call_next(request)


app.add_middleware(RateLimitMiddleware, default_rpm=_RATE_LIMIT, upload_rpm=_RATE_LIMIT_UPLOAD)


# ---------------------------------------------------------------------------
# Request timeout middleware
# ---------------------------------------------------------------------------

_LONG_PATHS = frozenset({"/api/upload", "/api/search/stream", "/api/draft"})


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """Enforce request timeouts. Long operations (upload/SSE/draft) get 300s, others 60s."""

    def __init__(self, app, default_timeout: float = 60.0, long_timeout: float = 300.0):
        super().__init__(app)
        self.default_timeout = default_timeout
        self.long_timeout = long_timeout

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # SSE and upload paths use streaming responses — wrapping with wait_for
        # would truncate mid-stream. These have nginx proxy_read_timeout as backstop.
        if path in _LONG_PATHS:
            return await call_next(request)
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.default_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Request timeout: %s %s (%.0fs)", request.method, path, self.default_timeout
            )
            return JSONResponse(
                status_code=504,
                content={"detail": "요청 처리 시간이 초과되었습니다."},
            )


app.add_middleware(RequestTimeoutMiddleware)

# CORS — configurable via CORS_ORIGINS env var (comma-separated)
_default_origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
]
_cors_raw = os.getenv("CORS_ORIGINS", "").strip()
_cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()] if _cors_raw else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or _default_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


# ---------------------------------------------------------------------------
# API Key authentication
# ---------------------------------------------------------------------------

_NOVA_API_KEY = os.getenv("NOVA_API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _verify_api_key(
    api_key: str | None = Depends(_api_key_header),  # noqa: B008
    request: Request = None,  # type: ignore[assignment]
) -> None:
    """Verify API key if NOVA_API_KEY is configured. Skip if not set (dev mode)."""
    if not _NOVA_API_KEY:
        return  # Auth disabled — development mode

    # Also accept Bearer token in Authorization header
    if not api_key and request:
        auth = request.headers.get("Authorization") or ""
        if auth.startswith("Bearer "):
            api_key = auth[7:]

    # SSE fallback: EventSource can't send headers, accept query param
    # Restricted to SSE streaming endpoints only to avoid key leakage via URL logs
    if not api_key and request:
        _SSE_PATHS = ("/api/search/stream",)
        if request.url.path in _SSE_PATHS:
            api_key = request.query_params.get("api_key")

    if not api_key or not secrets.compare_digest(api_key, _NOVA_API_KEY):
        raise HTTPException(status_code=401, detail="유효하지 않은 API 키입니다.")


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def _get_client(request: Request):
    """Retrieve the shared Gemini client from app state."""
    return request.app.state.gemini_client


def _get_store(request: Request):
    """Retrieve the shared VectorStore from app state."""
    return request.app.state.vector_store


def _run_in_llm(request: Request, fn, *args, **kwargs):
    """Run a blocking function in the LLM-dedicated thread pool."""
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(request.app.state.llm_executor, lambda: fn(*args, **kwargs))


def _run_in_pdf(request: Request, fn, *args, **kwargs):
    """Run a blocking function in the PDF-dedicated thread pool."""
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(request.app.state.pdf_executor, lambda: fn(*args, **kwargs))


_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _sanitize_filename(name: str) -> str:
    """Strip path components and dangerous characters from a filename."""
    import unicodedata

    # Normalize Unicode to prevent bypass via fullwidth chars etc.
    name = unicodedata.normalize("NFKC", name)
    # Take only the final path component (no directory traversal)
    name = Path(name).name
    # Remove characters that are unsafe on Windows/Linux
    name = name.replace("..", "").replace("/", "_").replace("\\", "_")
    name = name.replace("\x00", "")
    # Collapse multiple underscores
    while "__" in name:
        name = name.replace("__", "_")
    name = name.strip("_. ") or "unknown"
    # Block Windows reserved device names
    stem = Path(name).stem.upper()
    if stem in _WINDOWS_RESERVED:
        name = f"_{name}"
    # Cap length for NTFS (255 max, leave room for counter suffix)
    if len(name) > 200:
        suffix = Path(name).suffix
        name = name[: 200 - len(suffix)] + suffix
    return name


_managed_copy_lock = threading.Lock()
_MAX_COLLISION_RETRIES = 1000


def _copy_to_managed(src: Path, standard_name: str) -> str:
    """Copy uploaded file to managed storage so it survives temp cleanup.

    Returns the managed path string, or empty string on failure.
    """
    if not settings.registry.enabled or not standard_name:
        return ""
    try:
        safe_name = _sanitize_filename(standard_name)
        managed_dir = Path(settings.registry.managed_dir)
        managed_dir.mkdir(parents=True, exist_ok=True)
        dest = managed_dir / safe_name
        # Verify dest is actually inside managed_dir (defense in depth)
        if not dest.resolve().is_relative_to(managed_dir.resolve()):
            logger.warning("Path traversal blocked for: %s", standard_name)
            return ""
        # Serialize collision check + copy to prevent TOCTOU race
        with _managed_copy_lock:
            if dest.exists():
                stem = dest.stem
                suffix = dest.suffix
                counter = 1
                while dest.exists() and counter <= _MAX_COLLISION_RETRIES:
                    dest = managed_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                if dest.exists():
                    logger.warning("Collision limit reached for: %s", safe_name)
                    return ""
            shutil.copy2(str(src), str(dest))
        return str(dest)
    except Exception:
        logger.warning("Failed to copy file to managed storage", exc_info=True)
        return ""


def _update_registry_managed_path(
    request: Request,
    doc_id: str,
    managed_path: str,
) -> None:
    """Update registry record with managed storage path."""
    registry = getattr(request.app.state, "registry", None)
    if not registry:
        return
    try:
        registry.update_document(doc_id, managed_path=managed_path)
    except Exception:
        logger.warning("Failed to update registry managed_path", exc_info=True)


# ---------------------------------------------------------------------------
# Health check (inline — no Streamlit dependency)
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health_check(request: Request):
    """Returns system health status."""
    warnings = settings.validate_for_processing()
    client = _get_client(request)
    store = _get_store(request)
    chroma_ok = store is not None
    chroma_count = store.count if chroma_ok else 0
    gemini_ok = client is not None

    return {
        "gemini_ok": gemini_ok,
        "chroma_ok": chroma_ok,
        "chroma_count": chroma_count,
        "warnings": warnings,
    }


@app.get("/")
async def root():
    return {"message": "Nova API Server is running."}


# ---------------------------------------------------------------------------
# Upload — process_document in dedicated thread pool
# ---------------------------------------------------------------------------


@app.post("/api/upload", dependencies=[Depends(_verify_api_key)])
async def upload_document(
    request: Request,
    files: list[UploadFile] = File(...),  # noqa: B008
    security_grade: str = Form("C"),  # noqa: B008
    save_embed: bool = Form(True),  # noqa: B008
    save_sheets: bool = Form(False),  # noqa: B008
):
    """Process uploaded documents (PDF, DOCX, PPTX, DOC, PPT) via the pipeline."""
    client = _get_client(request)
    output_dir = Path(settings.chroma.persist_dir).parent / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for uploaded_file in files:
        # Sanitize filename: strip path components to prevent traversal
        raw_name = uploaded_file.filename or "unknown.pdf"
        file_name = Path(raw_name).name or "unknown.pdf"

        # Validate file extension
        ext = Path(file_name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            results.append(
                {
                    "filename": file_name,
                    "error": "지원하지 않는 파일 형식입니다. (PDF, DOCX, PPTX, DOC, PPT)",
                    "skipped": True,
                    "skip_reason": "invalid_type",
                    "chunks_stored": 0,
                    "summary": "",
                    "doc_info": None,
                }
            )
            continue

        tmp_dir = tempfile.mkdtemp(prefix="docpipe_nova_")
        tmp_path = os.path.join(tmp_dir, file_name)

        try:
            # Stream to disk with size check
            total = 0
            with open(tmp_path, "wb") as f:
                while chunk := await uploaded_file.read(1024 * 64):
                    total += len(chunk)
                    if total > _MAX_UPLOAD_BYTES:
                        max_mb = _MAX_UPLOAD_BYTES // (1024 * 1024)
                        raise HTTPException(
                            status_code=413,
                            detail=f"파일 크기가 {max_mb}MB를 초과합니다.",
                        )
                    f.write(chunk)

            # Run blocking pipeline in dedicated PDF executor
            result = await _run_in_pdf(
                request,
                process_document,
                file_path=Path(tmp_path),
                grade=SecurityGrade(security_grade),
                no_embed=not save_embed,
                save_sheets=save_sheets,
                output_dir=output_dir,
                source_dir="(Nova Web Upload)",
                client=client,
            )

            doc_info = None
            if result.doc:
                # Copy file to managed storage so it survives temp cleanup
                managed_path = _copy_to_managed(
                    Path(tmp_path),
                    result.doc.file_name_standard,
                )
                doc_info = {
                    "doc_id": result.doc.doc_id,
                    "doc_type": result.doc.doc_type.value,
                    "doc_type_ext": getattr(result.doc, "doc_type_ext", ""),
                    "category": getattr(result.doc, "category", ""),
                    "project_name": result.doc.project_name,
                    "year": result.doc.year,
                    "file_name_standard": result.doc.file_name_standard,
                    "page_count": result.doc.page_count,
                    "source_format": result.doc.source_format.value,
                    "managed_path": managed_path,
                }
                # Update registry with managed_path
                if managed_path:
                    _update_registry_managed_path(
                        request,
                        result.doc.doc_id,
                        managed_path,
                    )
                # Update query parser with new project name
                qp = getattr(request.app.state, "query_parser", None)
                if qp and result.doc.project_name:
                    qp._known_projects.add(result.doc.project_name)

            results.append(
                {
                    "filename": file_name,
                    "error": result.error,
                    "skipped": result.skipped,
                    "skip_reason": result.skip_reason,
                    "chunks_stored": result.chunks_stored,
                    "summary": result.summary,
                    "doc_info": doc_info,
                }
            )

        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Error processing %s: %s", file_name, exc)
            results.append(
                {
                    "filename": file_name,
                    "error": "문서 처리 중 오류가 발생했습니다.",
                    "skipped": False,
                    "skip_reason": "",
                    "chunks_stored": 0,
                    "summary": "",
                    "doc_info": None,
                }
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return {"status": "success", "results": results}


# ---------------------------------------------------------------------------
# Search — batch (POST, backward compatible)
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    n_results: int = Field(default=5, ge=1, le=20)
    doc_type_filter: str = "전체"
    category_filter: str = "전체"
    doc_type_ext_filter: str = "전체"
    search_profile: Literal[
        "auto", "technical_qa", "project_lookup", "contract_lookup", "method_docs"
    ] = "auto"


_MAX_CONTEXT_CHARS = 24_000  # Budget for RAG context to limit Gemini token usage


def _build_rag_context(
    doc_results,
    query: str,
    registry=None,
    category_filter: str | None = None,
) -> tuple[list[dict], str]:
    """Build references list and RAG prompt from document-level results.

    Accepts pre-aggregated ``DocumentResult`` list (from ``unified_search``).
    Registry data is batch-fetched to avoid N+1 queries.
    Documents with ``exclude_from_search=1`` are filtered out,
    and remaining references are boosted by quality score.

    Top chunks from each document are concatenated (with page labels)
    to provide richer evidence to the LLM.
    """
    # Batch-fetch registry data (single query instead of N+1)
    doc_records: dict[str, dict] = {}
    if registry and doc_results:
        try:
            doc_ids = [d.doc_id for d in doc_results]
            doc_records = registry.get_documents_batch(doc_ids)
        except Exception:
            logger.debug("Registry batch fetch failed", exc_info=True)

    context_parts: list[str] = []
    references: list[dict] = []
    budget = _MAX_CONTEXT_CHARS

    for i, doc_res in enumerate(doc_results, 1):
        doc_record = doc_records.get(doc_res.doc_id)

        # Filter excluded documents
        if doc_record and doc_record.get("exclude_from_search"):
            continue
        # Defense-in-depth: category_filter is also applied at ChromaDB level
        # via _build_where_clause(). This second check guards against stale
        # ChromaDB metadata when a document's category was updated in the
        # registry but not yet re-indexed in ChromaDB.
        if doc_record and category_filter and doc_record.get("category") != category_filter:
            continue

        # Quality-weighted doc score
        quality = 50.0
        if doc_record:
            quality = doc_record.get("quality_score", 50.0)
        quality_bonus = (quality - 50.0) / 50.0 * 0.15
        weighted_score = doc_res.doc_score * (1.0 + quality_bonus)

        # Build evidence text from top_chunks (not just best_chunk[:500])
        max_per_doc = max(1500, budget // max(len(doc_results), 1))
        evidence_parts: list[str] = []
        for chunk in doc_res.top_chunks:
            page_label = f"[p.{chunk.page_number}] " if chunk.page_number else ""
            evidence_parts.append(f"{page_label}{chunk.text}")
        evidence_text = "\n".join(evidence_parts)
        if len(evidence_text) > max_per_doc:
            evidence_text = evidence_text[:max_per_doc] + "..."

        ref = {
            "id": f"문서 {len(references) + 1}",
            "doc_type": doc_res.doc_type,
            "project_name": doc_res.project_name,
            "text": evidence_text,
            "similarity": f"{doc_res.doc_score:.4f}",
            "doc_id": doc_res.doc_id,
            "file_name_standard": "",
            "source_path": "",
            "managed_path": "",
            "quality_score": quality,
            "source_collection": doc_res.best_chunk.source_collection,
            "page_number": doc_res.best_chunk.page_number,
            "category": doc_res.category,
            "chunk_count": doc_res.chunk_count,
            "_weighted_score": weighted_score,
            "method_name": "",
            "file_name_original": "",
        }

        # Enrich with registry data
        if doc_record:
            ref["file_name_standard"] = doc_record.get("file_name_standard", "")
            ref["file_name_original"] = doc_record.get("file_name_original", "")
            ref["source_path"] = doc_record.get("source_path", doc_record.get("file_path_nas", ""))
            ref["managed_path"] = doc_record.get("managed_path", "")

        # Enrich with metadata json (method_name, etc.)
        if registry:
            meta_record = registry.get_metadata(doc_res.doc_id)
            if meta_record and "structured_fields" in meta_record:
                ref["method_name"] = meta_record["structured_fields"].get("method_name", "")

        references.append(ref)

    # Sort by quality-weighted doc score (higher=better)
    references.sort(key=lambda r: r.get("_weighted_score", 0.0), reverse=True)

    # Rebuild context from ranked document references
    for idx, ref in enumerate(references, 1):
        ref["id"] = f"문서 {idx}"
        part = (
            f"[문서 {idx}] 유형: {ref['doc_type']} | 프로젝트: {ref['project_name']}"
            f" | 청크: {ref.get('chunk_count', 1)}개\n{ref['text']}"
        )
        if budget - len(part) < 0:
            break
        budget -= len(part)
        context_parts.append(part)

    # Remove internal fields before returning
    for ref in references:
        ref.pop("_weighted_score", None)

    context = "\n\n---\n\n".join(context_parts)
    rag_prompt = (
        f"다음은 건축·구조 엔지니어링 문서 검색 결과입니다.\n\n"
        f"{context}\n\n---\n\n"
        f"위 문서들을 참고하여 다음 질문에 답변하세요: {query}\n"
        f"답변 시 어떤 문서를 참고했는지 [문서 N] 형식으로 출처를 표시하세요."
    )
    return references, rag_prompt


@app.get("/api/doc-types", dependencies=[Depends(_verify_api_key)])
async def get_doc_types(request: Request):
    """Retrieve document categories and doc_type_ext values.

    Policy: categories are derived from **corpus data only** — only categories
    that have at least one document in the registry are returned.  This prevents
    the UI from showing filter options that yield 0 results.

    ``doc_type_exts`` and ``all_types`` come from the corpus and type_registry
    respectively, for template/draft support.
    """
    from doc_pipeline.config.type_registry import get_type_registry

    type_registry = get_type_registry()
    # type_registry provides the canonical label lookup for known categories
    base_cats_list = type_registry.get_category_types()
    label_lookup = {c["category"]: c for c in base_cats_list}

    registry = getattr(request.app.state, "registry", None)
    categories: dict[str, dict] = {}
    exts: list[str] = []

    if registry:
        try:
            exts = registry.get_unique_doc_type_exts()
            for cat in registry.get_unique_categories():
                if cat in label_lookup:
                    categories[cat] = label_lookup[cat]
                else:
                    # Corpus has a category not in YAML — include with raw label
                    categories[cat] = {"category": cat, "label": cat, "types": []}
        except Exception as e:
            logger.error("Failed to retrieve doc-types from registry: %s", e)

    return {
        "status": "success",
        "categories": categories,
        "doc_type_exts": exts,
        "all_types": type_registry.all_type_names,
        "default_type": type_registry.default_type,
    }


@app.get("/api/suggest", dependencies=[Depends(_verify_api_key)])
async def suggest(
    q: str = "",
    limit: int = Query(default=8, ge=1, le=20),  # noqa: B008
    request: Request = None,  # type: ignore[assignment]
):
    """Lightweight autocomplete suggestions from corpus metadata.

    Returns project names and doc_type_ext values matching the query substring.
    Designed for debounced typeahead — fast, no AI calls, no vector search.
    """
    if len(q.strip()) < 2:
        return {"suggestions": []}
    registry = getattr(request.app.state, "registry", None)
    if not registry:
        return {"suggestions": []}
    results = registry.suggest(q.strip(), limit=limit)
    return {"suggestions": results}


@app.post("/api/search", dependencies=[Depends(_verify_api_key)])
async def search_documents(req: SearchRequest, request: Request):
    """Searches documents and generates an AI answer using RAG (batch response)."""
    client = _get_client(request)
    store = _get_store(request)

    if not client or not store:
        raise HTTPException(status_code=503, detail="서비스가 준비되지 않았습니다.")

    if store.count == 0:
        return {
            "answer": "벡터 DB에 저장된 문서가 없습니다. 문서를 먼저 업로드해 주세요.",
            "references": [],
        }

    try:
        from doc_pipeline.search import unified_search

        filter_type = req.doc_type_filter if req.doc_type_filter != "전체" else None
        filter_category = req.category_filter if req.category_filter != "전체" else None
        filter_ext = (
            req.doc_type_ext_filter
            if getattr(req, "doc_type_ext_filter", "전체") != "전체"
            else None
        )

        # Pre-filter excluded docs at the vector DB level
        registry = getattr(request.app.state, "registry", None)
        excluded_ids: list[str] = []
        if registry:
            try:
                excluded_ids = registry.get_excluded_doc_ids()
            except Exception:
                pass

        qp = getattr(request.app.state, "query_parser", None)
        chunk_fts = getattr(request.app.state, "chunk_fts", None)
        query_emb = await _run_in_llm(request, get_embeddings, client, [req.query])
        doc_results, _ = await _run_in_llm(
            request,
            unified_search,
            store,
            req.query,
            query_emb[0],
            n_results=req.n_results,
            doc_type_filter=filter_type,
            category_filter=filter_category,
            doc_type_ext_filter=filter_ext,
            search_profile=req.search_profile,
            exclude_doc_ids=excluded_ids or None,
            query_parser=qp,
            registry=registry,
            chunk_fts=chunk_fts,
        )

        if not doc_results:
            return {
                "answer": "관련 문서를 찾지 못했습니다. 다른 키워드로 검색해 보세요.",
                "references": [],
            }

        references, rag_prompt = _build_rag_context(
            doc_results, req.query, registry=registry, category_filter=filter_category
        )

        # --- PydanticAI agent path (feature toggle) ---
        if settings.agents.enabled:
            try:
                from doc_pipeline.agents.deps import SearchDeps
                from doc_pipeline.agents.search_agent import get_search_agent

                deps = SearchDeps(
                    query=req.query,
                    rag_prompt=rag_prompt,
                    references=references,
                    search_profile=req.search_profile,
                )
                agent = get_search_agent()
                await _run_in_llm(request, _rate_limit)
                result = await agent.run(req.query, deps=deps)
                answer_obj = result.output
                return {
                    "answer": answer_obj.answer,
                    "references": references,
                    "search_profile": req.search_profile,
                    "citations": [c.model_dump() for c in answer_obj.citations],
                    "confidence": answer_obj.confidence,
                    "follow_up": answer_obj.follow_up,
                }
            except Exception:
                logger.warning("Search agent failed, falling back to legacy", exc_info=True)

        # --- Legacy Gemini direct call ---
        response = await _run_in_llm(
            request,
            _call_with_retry,
            client.models.generate_content,
            model=settings.gemini.model_name,
            contents=rag_prompt,
            config=types.GenerateContentConfig(temperature=0.3),
        )

        return {
            "answer": response.text,
            "references": references,
            "search_profile": req.search_profile,
        }

    except Exception as exc:
        logger.error("Search API Error: %s", exc)
        raise HTTPException(status_code=500, detail="검색 처리 중 오류가 발생했습니다.") from exc


# ---------------------------------------------------------------------------
# Search SSE — streaming (GET, for React EventSource)
# ---------------------------------------------------------------------------


@app.get("/api/search/stream", dependencies=[Depends(_verify_api_key)])
async def search_stream(
    request: Request,
    query: str = Query(min_length=1, max_length=1000),  # noqa: B008
    n_results: int = Query(default=5, ge=1, le=20),  # noqa: B008
    doc_type_filter: str = "전체",
    category_filter: str = "전체",
    doc_type_ext_filter: str = "전체",
    search_profile: Literal[
        "auto", "technical_qa", "project_lookup", "contract_lookup", "method_docs"
    ] = "auto",
):
    """SSE streaming search: sends references, then token-by-token LLM answer."""
    client = _get_client(request)
    store = _get_store(request)

    if not client or not store:
        raise HTTPException(status_code=503, detail="서비스가 준비되지 않았습니다.")

    if store.count == 0:

        async def empty_gen():
            yield {
                "event": "server_error",
                "data": "벡터 DB에 저장된 문서가 없습니다. 문서를 먼저 업로드해 주세요.",
            }

        return EventSourceResponse(empty_gen())

    async def event_generator():
        try:
            from doc_pipeline.search import unified_search as _unified_search

            # 1. Embedding + search in LLM thread pool
            filter_type = doc_type_filter if doc_type_filter != "전체" else None
            cat_filter = category_filter if category_filter != "전체" else None
            ext_filter = doc_type_ext_filter if doc_type_ext_filter != "전체" else None

            # Pre-filter excluded docs at the vector DB level
            registry = getattr(request.app.state, "registry", None)
            excluded_ids: list[str] = []
            if registry:
                try:
                    excluded_ids = registry.get_excluded_doc_ids()
                except Exception:
                    pass

            qp = getattr(request.app.state, "query_parser", None)
            chunk_fts = getattr(request.app.state, "chunk_fts", None)
            query_emb = await _run_in_llm(request, get_embeddings, client, [query])
            doc_results, _ = await _run_in_llm(
                request,
                _unified_search,
                store,
                query,
                query_emb[0],
                n_results=n_results,
                doc_type_filter=filter_type,
                category_filter=cat_filter,
                doc_type_ext_filter=ext_filter,
                search_profile=search_profile,
                exclude_doc_ids=excluded_ids or None,
                query_parser=qp,
                registry=registry,
                chunk_fts=chunk_fts,
            )

            if not doc_results:
                yield {
                    "event": "server_error",
                    "data": "관련 문서를 찾지 못했습니다. 다른 키워드로 검색해 보세요.",
                }
                return

            # 2. Send references
            references, rag_prompt = _build_rag_context(
                doc_results, query, registry=registry, category_filter=cat_filter
            )
            yield {"event": "references", "data": json.dumps(references, ensure_ascii=False)}

            # 3. LLM answer generation (streaming)
            _use_legacy = True

            # --- PydanticAI agent streaming path ---
            if settings.agents.enabled:
                try:
                    from doc_pipeline.agents.deps import SearchDeps
                    from doc_pipeline.agents.search_agent import get_search_agent

                    deps = SearchDeps(
                        query=query, rag_prompt=rag_prompt,
                        references=references, search_profile=search_profile,
                    )
                    agent = get_search_agent()
                    await _run_in_llm(request, _rate_limit)
                    async with agent.run_stream(query, deps=deps) as stream_result:
                        async for token in stream_result.stream_text(delta=True):
                            if await request.is_disconnected():
                                logger.info("SSE client disconnected, cancelling stream")
                                return
                            yield {"event": "token", "data": token}

                        # Emit structured metadata after streaming completes
                        try:
                            final = stream_result.output
                            if hasattr(final, "citations"):
                                meta = {
                                    "citations": [c.model_dump() for c in final.citations],
                                    "confidence": final.confidence,
                                    "follow_up": final.follow_up,
                                }
                                yield {
                                    "event": "answer_meta",
                                    "data": json.dumps(meta, ensure_ascii=False),
                                }
                        except Exception:
                            logger.debug(
                                "Could not extract structured output from stream",
                                exc_info=True,
                            )

                    yield {"event": "done", "data": ""}
                    _use_legacy = False
                except Exception:
                    logger.warning(
                        "Stream agent failed, falling back to legacy", exc_info=True,
                    )

            if _use_legacy:
                # --- Legacy Gemini streaming via asyncio.Queue bridge ---
                await _run_in_llm(request, _rate_limit)

                stream = await _run_in_llm(
                    request,
                    client.models.generate_content_stream,
                    model=settings.gemini.model_name,
                    contents=rag_prompt,
                    config=types.GenerateContentConfig(temperature=0.3),
                )

                queue: asyncio.Queue[str | None | Exception] = asyncio.Queue()
                loop = asyncio.get_running_loop()
                cancel = threading.Event()

                def _consume_stream():
                    try:
                        for chunk in stream:
                            if cancel.is_set():
                                break
                            text = getattr(chunk, "text", "") or ""
                            if text:
                                asyncio.run_coroutine_threadsafe(queue.put(text), loop)
                    except Exception as exc:
                        asyncio.run_coroutine_threadsafe(queue.put(exc), loop)
                    finally:
                        asyncio.run_coroutine_threadsafe(queue.put(None), loop)

                request.app.state.llm_executor.submit(_consume_stream)

                while True:
                    if await request.is_disconnected():
                        cancel.set()
                        logger.info("SSE client disconnected, cancelling stream")
                        return

                    try:
                        token = await asyncio.wait_for(queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue

                    if token is None:
                        break
                    if isinstance(token, Exception):
                        yield {
                            "event": "server_error",
                            "data": "AI 응답 생성 중 오류가 발생했습니다.",
                        }
                        return
                    yield {"event": "token", "data": token}

                yield {"event": "done", "data": ""}

        except Exception as exc:
            logger.error("SSE stream error: %s", exc)
            yield {"event": "server_error", "data": "스트리밍 처리 중 오류가 발생했습니다."}

    return EventSourceResponse(event_generator(), ping=15)


# ---------------------------------------------------------------------------
# Document Registry API
# ---------------------------------------------------------------------------


# [Removed doc-types duplicate route]


_SORT_COLUMNS = frozenset(
    {
        "file_name_standard",
        "doc_type",
        "project_name",
        "year",
        "process_status",
        "quality_score",
        "ingested_at",
    }
)


@app.get("/api/documents", dependencies=[Depends(_verify_api_key)])
async def list_documents(
    request: Request,
    doc_type: str | None = Query(default=None),
    doc_type_ext: str | None = Query(default=None),
    project: str | None = Query(default=None),
    year: int | None = Query(default=None),
    status: str | None = Query(default=None),
    category: str | None = Query(default=None),
    needs_review: bool | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    sort_by: str = Query(default="ingested_at"),
    sort_order: str = Query(default="desc"),
):
    """List documents from the registry with optional filters."""
    registry = getattr(request.app.state, "registry", None)
    if not registry:
        raise HTTPException(503, "Registry not available")
    # Validate sort params against whitelist to prevent SQL injection
    col = sort_by if sort_by in _SORT_COLUMNS else "ingested_at"
    direction = "ASC" if sort_order.lower() == "asc" else "DESC"
    order_by = f"{col} {direction}"
    docs = registry.list_documents(
        doc_type=doc_type,
        doc_type_ext=doc_type_ext,
        project=project,
        year=year,
        status=status,
        category=category,
        needs_review=needs_review,
        limit=limit,
        offset=offset,
        order_by=order_by,
    )
    total = registry.count_documents(
        doc_type=doc_type,
        doc_type_ext=doc_type_ext,
        project=project,
        year=year,
        status=status,
        category=category,
        needs_review=needs_review,
    )
    return {"documents": docs, "total": total, "limit": limit, "offset": offset}


@app.get("/api/documents/stats", dependencies=[Depends(_verify_api_key)])
async def document_stats(request: Request):
    """Return registry statistics for the dashboard."""
    registry = getattr(request.app.state, "registry", None)
    if not registry:
        raise HTTPException(503, "Registry not available")
    return registry.get_stats()


@app.get("/api/documents/export", dependencies=[Depends(_verify_api_key)])
async def export_documents(
    request: Request,
    format: str = Query(default="csv"),
    doc_type: str | None = Query(default=None),
    doc_type_ext: str | None = Query(default=None),
    category: str | None = Query(default=None),
    needs_review: bool = Query(default=False),
):
    """Export documents as CSV."""
    import csv
    import io

    from starlette.responses import StreamingResponse

    registry = getattr(request.app.state, "registry", None)
    if not registry:
        raise HTTPException(503, "Registry not available")

    docs = registry.list_documents(
        doc_type=doc_type,
        doc_type_ext=doc_type_ext,
        category=category,
        needs_review=needs_review,
        limit=None,
    )

    columns = [
        "doc_id",
        "file_name_original",
        "file_name_standard",
        "doc_type",
        "doc_type_ext",
        "category",
        "project_name",
        "year",
        "process_status",
        "quality_grade",
        "quality_score",
        "ingested_at",
    ]

    output = io.StringIO()
    # Write BOM for Excel UTF-8 compatibility
    output.write("\ufeff")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for doc in docs:
        writer.writerow({col: doc.get(col, "") for col in columns})

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=documents.csv"},
    )


@app.get("/api/activity/recent", dependencies=[Depends(_verify_api_key)])
async def recent_activity(
    request: Request, limit: int = Query(default=20, ge=1, le=100)
):  # noqa: B008
    """Return recent document events for the dashboard activity feed."""
    registry = getattr(request.app.state, "registry", None)
    if not registry:
        raise HTTPException(503, "Registry not available")
    return {"events": registry.get_recent_events(limit=limit)}


@app.get("/api/documents/{doc_id}", dependencies=[Depends(_verify_api_key)])
async def get_document(doc_id: str, request: Request):
    """Get a single document with metadata and events."""
    registry = getattr(request.app.state, "registry", None)
    if not registry:
        raise HTTPException(503, "Registry not available")
    doc = registry.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    meta = registry.get_metadata(doc_id)
    events = registry.get_events(doc_id, limit=20)
    return {"document": doc, "metadata": meta, "events": events}


@app.get("/api/documents/{doc_id}/preview", dependencies=[Depends(_verify_api_key)])
async def get_document_preview(
    doc_id: str,
    page: int = Query(1, ge=1),
    request: Request = None,  # type: ignore[assignment]
):
    """Return extracted text chunks for a document, paginated by page_number."""
    registry = getattr(request.app.state, "registry", None)
    if not registry:
        raise HTTPException(503, "Registry not available")
    doc = registry.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    vector_store = getattr(request.app.state, "vector_store", None)
    if not vector_store:
        raise HTTPException(503, "Vector store not available")

    all_chunks = vector_store.get_chunks_by_doc_id(doc_id)
    if not all_chunks:
        return {"doc_id": doc_id, "current_page": 1, "total_pages": 0, "content": "", "chunks": []}

    # Group by page_number
    pages: dict[int, list[dict]] = {}
    for chunk in all_chunks:
        pg = chunk.get("page_number") or 1
        pages.setdefault(pg, []).append(chunk)

    sorted_page_nums = sorted(pages.keys())
    total_pages = len(sorted_page_nums)
    clamped_page = min(page, total_pages)
    page_num = sorted_page_nums[clamped_page - 1] if sorted_page_nums else 1
    page_chunks = pages.get(page_num, [])

    content = "\n\n".join(c["text"] for c in page_chunks)
    return {
        "doc_id": doc_id,
        "current_page": clamped_page,
        "total_pages": total_pages,
        "content": content,
        "chunks": [
            {"text": c["text"], "page_number": c["page_number"], "chunk_index": c["chunk_index"]}
            for c in page_chunks
        ],
    }


# ---------------------------------------------------------------------------
# Feedback API
# ---------------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    doc_id: str
    rating: str  # "positive" or "negative"
    reason: str = ""  # "incorrect", "outdated", "duplicate", "exclude", ""
    comment: str = ""


_VALID_FEEDBACK_REASONS = frozenset({"incorrect", "outdated", "duplicate", "exclude", ""})


@app.post("/api/feedback", dependencies=[Depends(_verify_api_key)])
async def submit_feedback(req: FeedbackRequest, request: Request):
    """Submit document-level feedback (positive/negative)."""
    registry = getattr(request.app.state, "registry", None)
    if not registry:
        raise HTTPException(503, "Registry not available")
    doc = registry.get_document(req.doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    if req.rating not in ("positive", "negative"):
        raise HTTPException(422, "rating must be 'positive' or 'negative'")
    if req.reason and req.reason not in _VALID_FEEDBACK_REASONS:
        raise HTTPException(
            422,
            f"reason must be one of: {', '.join(sorted(_VALID_FEEDBACK_REASONS - {''}))} (or empty)",
        )
    registry.add_feedback(req.doc_id, req.rating, req.comment, reason=req.reason)
    return {"status": "ok", "doc_id": req.doc_id}


@app.get("/api/documents/{doc_id}/feedback", dependencies=[Depends(_verify_api_key)])
async def get_document_feedback(doc_id: str, request: Request):
    """Get feedback entries and quality info for a document."""
    registry = getattr(request.app.state, "registry", None)
    if not registry:
        raise HTTPException(503, "Registry not available")
    feedback = registry.get_feedback(doc_id)
    doc = registry.get_document(doc_id)
    return {
        "feedback": feedback,
        "quality_score": doc["quality_score"] if doc else None,
        "quality_grade": doc["quality_grade"] if doc else None,
    }


# ---------------------------------------------------------------------------
# Draft
# ---------------------------------------------------------------------------


class DraftRequest(BaseModel):
    doc_type: str
    project_name: str
    issue: str
    use_llm: bool = True
    extra_fields: dict[str, str] | None = None


@app.post("/api/draft", dependencies=[Depends(_verify_api_key)])
async def create_draft(req: DraftRequest, request: Request):
    """Generates a document draft using templates and RAG."""
    try:
        draft, ref_data = await asyncio.to_thread(
            generate_draft,
            doc_type=req.doc_type,
            project_name=req.project_name,
            issue=req.issue,
            use_llm=req.use_llm,
            extra_fields=req.extra_fields,
        )

        # Enrich references with registry data (managed_path, file_name_standard)
        registry = getattr(request.app.state, "registry", None)
        if registry:
            for ref in ref_data:
                doc_id = ref.get("doc_id")
                if doc_id:
                    try:
                        doc_record = registry.get_document(doc_id)
                        if doc_record:
                            ref["file_name_standard"] = doc_record.get("file_name_standard", "")
                            ref["managed_path"] = doc_record.get("managed_path", "")
                            ref["source_path"] = doc_record.get("source_path", "")
                    except Exception:
                        pass

        return {"status": "success", "draft": draft, "references": ref_data}
    except Exception as exc:
        logger.error("Draft API Error: %s", exc)
        raise HTTPException(status_code=500, detail="초안 생성 중 오류가 발생했습니다.") from exc


# ---------------------------------------------------------------------------
# Manual reclassification API
# ---------------------------------------------------------------------------


class ReclassifyRequest(BaseModel):
    doc_type: str
    project_name: str | None = None
    year: int | None = None


@app.patch("/api/documents/{doc_id}/classify", dependencies=[Depends(_verify_api_key)])
async def reclassify_document(doc_id: str, req: ReclassifyRequest, request: Request):
    """Manually reclassify a document to a different type."""
    from doc_pipeline.config.type_registry import get_type_registry

    registry = getattr(request.app.state, "registry", None)
    if not registry:
        raise HTTPException(503, "Registry not available")

    doc = registry.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    # Validate doc_type against TypeRegistry
    type_reg = get_type_registry()
    td = type_reg.get(req.doc_type)
    if not td:
        raise HTTPException(422, f"Unknown document type: {req.doc_type}")

    old_type = doc.get("doc_type_ext") or doc.get("doc_type", "")

    # Resolve project_name and year (use request values or fall back to existing)
    project_name = req.project_name if req.project_name is not None else doc.get("project_name", "")
    year = req.year if req.year is not None else doc.get("year", 0)

    # Recompute standard filename with new classification
    from doc_pipeline.processor.pipeline import _next_standard_name

    managed_path = doc.get("managed_path", "")
    source_path = doc.get("source_path", "")
    ref_path = managed_path or source_path or ""
    extension = Path(ref_path).suffix.lower() if ref_path else ".pdf"
    # Use managed_path's parent for sequence uniqueness; fall back to source_path's parent
    if managed_path:
        target_dir = Path(managed_path).parent
    elif source_path and source_path != "(Nova Web Upload)":
        target_dir = Path(source_path).parent
    else:
        target_dir = Path(settings.registry.managed_storage_dir)
    file_name_standard = _next_standard_name(
        year,
        project_name,
        req.doc_type,
        target_dir,
        extension=extension,
    )

    # Rename physical managed file to keep disk & registry in sync
    new_managed_path = ""
    if managed_path and Path(managed_path).exists():
        old_managed = Path(managed_path)
        new_managed = old_managed.parent / file_name_standard
        if old_managed != new_managed:
            try:
                old_managed.rename(new_managed)
                new_managed_path = str(new_managed)
            except OSError:
                logging.getLogger(__name__).warning(
                    "Failed to rename managed file %s",
                    old_managed.name,
                    exc_info=True,
                )

    # Build update fields
    update_fields: dict = {
        "doc_type": td.resolve_to_legacy_value(),
        "doc_type_ext": req.doc_type,
        "category": td.category,
        "classification_method": "manual",
        "classification_confidence": 1.0,
        "file_name_standard": file_name_standard,
    }
    if new_managed_path:
        update_fields["managed_path"] = new_managed_path
    if req.project_name is not None:
        update_fields["project_name"] = req.project_name
    if req.year is not None:
        update_fields["year"] = req.year

    registry.update_document(doc_id, **update_fields)
    registry.add_event(doc_id, "manual_reclassify", f"{old_type} -> {req.doc_type}")

    return {"status": "ok", "old_type": old_type, "new_type": req.doc_type}


# ---------------------------------------------------------------------------
# Document file download
# ---------------------------------------------------------------------------


@app.get("/api/documents/{doc_id}/download", dependencies=[Depends(_verify_api_key)])
async def download_document(doc_id: str, request: Request):
    """Download the managed file for a document."""
    from fastapi.responses import FileResponse

    registry = getattr(request.app.state, "registry", None)
    if not registry:
        raise HTTPException(503, "Registry not available")
    doc = registry.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    managed_path = doc.get("managed_path", "")
    if not managed_path:
        raise HTTPException(404, "파일을 다운로드할 수 없습니다. (managed 경로 없음)")
    file_path = Path(managed_path)
    if not file_path.exists():
        raise HTTPException(404, "파일이 디스크에 존재하지 않습니다.")
    # Security: ensure file is within managed directory
    managed_dir = Path(settings.registry.managed_dir).resolve()
    if not file_path.resolve().is_relative_to(managed_dir):
        raise HTTPException(403, "접근이 거부되었습니다.")
    return FileResponse(
        path=str(file_path),
        filename=doc.get("file_name_standard") or file_path.name,
        media_type="application/octet-stream",
    )


if __name__ == "__main__":
    import uvicorn

    _is_production = os.getenv("NOVA_ENV") == "production"
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=2 if _is_production else 1,
        reload=not _is_production,
        log_level="info",
    )
