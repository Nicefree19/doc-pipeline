"""SQLite-backed document registry.

Provides persistent storage for all processed documents with metadata,
event logging, deduplication via SHA-256 hash, and query support.
Uses Python stdlib sqlite3 — no external dependencies.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from doc_pipeline.models.schemas import DocMaster

_ALLOWED_ORDER = frozenset(
    {
        "ingested_at DESC",
        "ingested_at ASC",
        "processed_at DESC",
        "processed_at ASC",
        "year DESC",
        "year ASC",
        "doc_type ASC",
        "doc_type DESC",
        "project_name ASC",
        "project_name DESC",
    }
)

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    source_path     TEXT NOT NULL DEFAULT '',
    managed_path    TEXT DEFAULT '',
    file_name_original TEXT NOT NULL,
    file_name_standard TEXT DEFAULT '',
    source_format   TEXT NOT NULL DEFAULT 'pdf',
    doc_type        TEXT NOT NULL,
    project_name    TEXT DEFAULT '',
    year            INTEGER DEFAULT 0,
    page_count      INTEGER DEFAULT 0,
    security_grade  TEXT NOT NULL DEFAULT 'C',
    ocr_engine      TEXT DEFAULT 'none',
    process_status  TEXT NOT NULL DEFAULT '대기',
    ingested_at     TEXT NOT NULL,
    processed_at    TEXT DEFAULT '',
    hash_sha256     TEXT DEFAULT '',
    summary         TEXT DEFAULT '',
    quality_score   REAL DEFAULT 50.0,
    quality_grade   TEXT DEFAULT 'B',
    exclude_from_search INTEGER DEFAULT 0,
    is_duplicate    INTEGER DEFAULT 0,
    superseded_by   TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS document_metadata (
    doc_id              TEXT PRIMARY KEY REFERENCES documents(doc_id),
    metadata_json       TEXT DEFAULT '{}',
    structured_fields   TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS document_events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id),
    event_type  TEXT NOT NULL,
    message     TEXT DEFAULT '',
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_doc_project ON documents(project_name);
CREATE INDEX IF NOT EXISTS idx_doc_year ON documents(year);
CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(hash_sha256);
CREATE INDEX IF NOT EXISTS idx_doc_status ON documents(process_status);
CREATE INDEX IF NOT EXISTS idx_events_doc ON document_events(doc_id);

CREATE TABLE IF NOT EXISTS document_feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id),
    rating      TEXT NOT NULL,
    comment     TEXT DEFAULT '',
    source      TEXT DEFAULT 'web',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_doc ON document_feedback(doc_id);
"""


class DocumentRegistry:
    """SQLite-backed document registry."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str = "data/registry.db") -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(path)
        self._write_lock = threading.Lock()
        self._init_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_tables(self) -> None:
        with self._write_lock:
            conn = self._connect()
            try:
                conn.executescript(_SCHEMA_SQL)
                conn.commit()
            finally:
                conn.close()
        self._ensure_extended_columns()
        self._init_fts5()

    def _ensure_extended_columns(self) -> None:
        """Non-destructive migration: add extended classifier columns if missing."""
        migrations = [
            ("managed_path", "TEXT DEFAULT ''"),
            ("doc_type_ext", "TEXT DEFAULT ''"),
            ("category", "TEXT DEFAULT ''"),
            ("classification_method", "TEXT DEFAULT ''"),
            ("classification_confidence", "REAL DEFAULT 0.0"),
            ("embedded_at", "TEXT DEFAULT ''"),
        ]
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_doc_type_ext ON documents(doc_type_ext)",
            "CREATE INDEX IF NOT EXISTS idx_doc_category ON documents(category)",
        ]
        # Feedback table migrations
        feedback_migrations = [
            ("reason", "TEXT DEFAULT ''"),
        ]
        with self._write_lock:
            conn = self._connect()
            try:
                # Check existing columns on documents table
                cursor = conn.execute("PRAGMA table_info(documents)")
                existing = {row["name"] for row in cursor.fetchall()}

                for col_name, col_def in migrations:
                    if col_name not in existing:
                        conn.execute(f"ALTER TABLE documents ADD COLUMN {col_name} {col_def}")
                        logger.info("Added column documents.%s", col_name)

                for idx_sql in indexes:
                    conn.execute(idx_sql)

                # Check existing columns on document_feedback table
                cursor = conn.execute("PRAGMA table_info(document_feedback)")
                fb_existing = {row["name"] for row in cursor.fetchall()}

                for col_name, col_def in feedback_migrations:
                    if col_name not in fb_existing:
                        conn.execute(
                            f"ALTER TABLE document_feedback ADD COLUMN {col_name} {col_def}",
                        )
                        logger.info("Added column document_feedback.%s", col_name)

                conn.commit()
            except Exception:
                logger.warning("Extended column migration failed", exc_info=True)
            finally:
                conn.close()

    def _init_fts5(self) -> None:
        """Create doc-level FTS5 table and bulk-populate from existing documents."""
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                    doc_id UNINDEXED,
                    summary,
                    file_name_standard,
                    project_name,
                    doc_type_ext,
                    tokenize='unicode61 remove_diacritics 2'
                )
            """
            )
            # Bulk populate if empty
            fts_count = conn.execute("SELECT COUNT(*) FROM documents_fts").fetchone()[0]
            if fts_count == 0:
                doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                if doc_count > 0:
                    conn.execute(
                        """
                        INSERT INTO documents_fts (doc_id, summary, file_name_standard, project_name, doc_type_ext)
                        SELECT doc_id, COALESCE(summary, ''), COALESCE(file_name_standard, ''),
                               COALESCE(project_name, ''), COALESCE(doc_type_ext, '')
                        FROM documents
                    """
                    )
                    logger.info("FTS5 bulk-populated %d documents", doc_count)
            conn.commit()
        except Exception:
            logger.warning("FTS5 init failed (may not be supported)", exc_info=True)
        finally:
            conn.close()

    def _sync_fts5(self, doc_id: str) -> None:
        """Sync a single document to the FTS5 index."""
        conn = self._connect()
        try:
            conn.execute("DELETE FROM documents_fts WHERE doc_id = ?", (doc_id,))
            conn.execute(
                """
                INSERT INTO documents_fts (doc_id, summary, file_name_standard, project_name, doc_type_ext)
                SELECT doc_id, COALESCE(summary, ''), COALESCE(file_name_standard, ''),
                       COALESCE(project_name, ''), COALESCE(doc_type_ext, '')
                FROM documents WHERE doc_id = ?
            """,
                (doc_id,),
            )
            conn.commit()
        except Exception:
            logger.debug("FTS5 sync failed for %s", doc_id, exc_info=True)
        finally:
            conn.close()

    @staticmethod
    def _retry_on_lock(
        fn: Any,
        *,
        max_retries: int = 3,
        base_wait: float = 0.5,
    ) -> Any:
        for attempt in range(max_retries):
            try:
                return fn()
            except sqlite3.OperationalError as exc:
                if "database is locked" in str(exc) and attempt < max_retries - 1:
                    wait = base_wait * (attempt + 1)
                    logger.warning(
                        "SQLite locked, retrying in %.1fs (%d/%d)",
                        wait,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(wait)
                else:
                    raise

    # --- CRUD ---

    def insert_document(
        self,
        doc: DocMaster,
        source_path: str,
        hash_sha256: str = "",
        metadata: dict[str, Any] | None = None,
        managed_path: str = "",
    ) -> str:
        """Insert a document record. Returns doc_id."""
        now = datetime.now().isoformat()
        processed_at = doc.process_date.isoformat() if doc.process_date else now

        def _insert() -> None:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO documents (
                        doc_id, source_path, managed_path,
                        file_name_original, file_name_standard,
                        source_format, doc_type, project_name, year, page_count,
                        security_grade, ocr_engine, process_status,
                        ingested_at, processed_at, hash_sha256, summary,
                        doc_type_ext, category,
                        classification_method, classification_confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        doc.doc_id,
                        source_path,
                        managed_path,
                        doc.file_name_original,
                        doc.file_name_standard,
                        doc.source_format.value,
                        doc.doc_type.value,
                        doc.project_name,
                        doc.year,
                        doc.page_count,
                        doc.security_grade.value,
                        doc.ocr_engine,
                        doc.process_status.value,
                        now,
                        processed_at,
                        hash_sha256,
                        doc.summary,
                        getattr(doc, "doc_type_ext", ""),
                        getattr(doc, "category", ""),
                        (metadata or {}).get("_classification_method", ""),
                        (metadata or {}).get("_classification_confidence", 0.0),
                    ),
                )
                if metadata:
                    conn.execute(
                        """INSERT OR REPLACE INTO document_metadata
                           (doc_id, metadata_json) VALUES (?, ?)""",
                        (doc.doc_id, json.dumps(metadata, ensure_ascii=False)),
                    )
                conn.commit()
            finally:
                conn.close()

        with self._write_lock:
            self._retry_on_lock(_insert)
        self._sync_fts5(doc.doc_id)
        return doc.doc_id

    def update_document(self, doc_id: str, **fields: Any) -> None:
        """Update specific fields of a document."""
        if not fields:
            return
        allowed = {
            "source_path",
            "managed_path",
            "file_name_standard",
            "process_status",
            "summary",
            "quality_score",
            "quality_grade",
            "exclude_from_search",
            "is_duplicate",
            "superseded_by",
            "hash_sha256",
            "processed_at",
            "embedded_at",
            "doc_type",
            "doc_type_ext",
            "category",
            "classification_method",
            "classification_confidence",
            "project_name",
            "year",
        }
        invalid = set(fields) - allowed
        if invalid:
            raise ValueError(f"Cannot update fields: {invalid}")

        set_clause = ", ".join(f'"{k}" = ?' for k in fields)
        values = list(fields.values()) + [doc_id]

        def _update() -> None:
            conn = self._connect()
            try:
                conn.execute(
                    f"UPDATE documents SET {set_clause} WHERE doc_id = ?",  # noqa: S608
                    values,
                )
                conn.commit()
            finally:
                conn.close()

        with self._write_lock:
            self._retry_on_lock(_update)
        self._sync_fts5(doc_id)

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Get a single document by doc_id."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_documents_batch(self, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch multiple documents by ID in a single query.

        Returns:
            Dict mapping doc_id → document record. Missing IDs are omitted.
        """
        if not doc_ids:
            return {}
        conn = self._connect()
        try:
            placeholders = ",".join("?" for _ in doc_ids)
            rows = conn.execute(
                f"SELECT * FROM documents WHERE doc_id IN ({placeholders})",  # noqa: S608
                doc_ids,
            ).fetchall()
            return {row["doc_id"]: dict(row) for row in rows}
        finally:
            conn.close()

    # --- List / Search ---

    def list_documents(
        self,
        *,
        doc_type: str | None = None,
        doc_type_ext: str | None = None,
        project: str | None = None,
        year: int | None = None,
        status: str | None = None,
        category: str | None = None,
        exclude_search: bool | None = None,
        needs_review: bool | None = None,
        embedded_only: bool | None = None,
        limit: int | None = 50,
        offset: int = 0,
        order_by: str = "ingested_at DESC",
    ) -> list[dict[str, Any]]:
        """List documents with optional filters.

        Args:
            limit: Max rows to return.  ``None`` returns all matching rows.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if doc_type is not None:
            clauses.append("doc_type = ?")
            params.append(doc_type)
        if doc_type_ext is not None:
            clauses.append("doc_type_ext = ?")
            params.append(doc_type_ext)
        if project is not None:
            clauses.append("project_name = ?")
            params.append(project)
        if year is not None:
            clauses.append("year = ?")
            params.append(year)
        if status is not None:
            clauses.append("process_status = ?")
            params.append(status)
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        if exclude_search is not None:
            clauses.append("exclude_from_search = ?")
            params.append(int(exclude_search))
        if needs_review is True:
            clauses.append("quality_score < 30 AND quality_score >= 0")
        if embedded_only is True:
            clauses.append("embedded_at IS NOT NULL AND length(embedded_at) > 0")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        # Whitelist order_by to prevent SQL injection
        if order_by not in _ALLOWED_ORDER:
            order_by = "ingested_at DESC"

        # Build LIMIT/OFFSET clause (omit LIMIT when limit is None for full export)
        if limit is not None:
            limit_clause = "LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        else:
            limit_clause = ""

        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT * FROM documents {where} ORDER BY {order_by} {limit_clause}",  # noqa: S608
                params,
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_unique_doc_type_exts(self) -> list[str]:
        """Return a list of unique doc_type_ext values currently in the DB."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT DISTINCT doc_type_ext FROM documents WHERE doc_type_ext IS NOT NULL AND doc_type_ext != '' ORDER BY doc_type_ext"
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    def get_unique_categories(self) -> list[str]:
        """Return a list of unique category values currently in the DB."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT DISTINCT category FROM documents WHERE category IS NOT NULL AND category != '' ORDER BY category"
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    def count_documents(self, **filters: Any) -> int:
        """Count documents with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []

        filter_map = {
            "doc_type": "doc_type",
            "doc_type_ext": "doc_type_ext",
            "project": "project_name",
            "year": "year",
            "status": "process_status",
            "category": "category",
        }
        for key, col in filter_map.items():
            val = filters.get(key)
            if val is not None:
                clauses.append(f"{col} = ?")
                params.append(val)

        exclude_search = filters.get("exclude_search")
        if exclude_search is not None:
            clauses.append("exclude_from_search = ?")
            params.append(int(exclude_search))

        needs_review = filters.get("needs_review")
        if needs_review is True:
            clauses.append("quality_score < 30 AND quality_score >= 0")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        conn = self._connect()
        try:
            row = conn.execute(
                f"SELECT COUNT(*) as cnt FROM documents {where}",  # noqa: S608
                params,
            ).fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()

    def find_by_hash(self, hash_sha256: str) -> dict[str, Any] | None:
        """Find a document by its SHA-256 hash."""
        if not hash_sha256:
            return None
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM documents WHERE hash_sha256 = ? LIMIT 1",
                (hash_sha256,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    # --- Embed failure tracking ---

    def update_embed_failure(self, doc_id: str, error_type: str, error_msg: str) -> None:
        """Record embedding failure, preserving existing metadata/structured_fields.

        Uses read-modify-write to prevent INSERT OR REPLACE from clobbering
        structured_fields.
        """
        existing = self.get_metadata(doc_id)
        if existing:
            meta = existing.get("metadata", {})
            structured = existing.get("structured_fields", {})
        else:
            meta = {}
            structured = {}

        meta["embed_error_type"] = error_type
        meta["embed_error_msg"] = error_msg[:500]
        meta["embed_attempts"] = meta.get("embed_attempts", 0) + 1
        meta["last_embed_error_at"] = datetime.now().isoformat()

        self.save_metadata(doc_id, meta, structured=structured)

    # --- FTS5 Search ---

    @staticmethod
    def _build_fts_queries(tokens: list[str]) -> list[str]:
        """Build FTS5 queries in order: phrase -> AND -> OR."""
        queries: list[str] = []
        if len(tokens) >= 2:
            queries.append('"' + " ".join(tokens) + '"')  # phrase
        queries.append(" AND ".join(f'"{t}"' for t in tokens))  # AND
        queries.append(" OR ".join(f'"{t}"' for t in tokens))   # OR
        return queries

    def search_fts(
        self,
        query: str,
        *,
        limit: int = 20,
        project_name: str | None = None,
        year: int | None = None,
    ) -> list[dict]:
        """Search documents using FTS5 full-text search with phrase-first strategy.

        Returns list of {doc_id, rank, project_name, doc_type_ext, summary_snippet}.
        """
        if not query or not query.strip():
            return []

        tokens = [t.strip() for t in query.split() if len(t.strip()) >= 2]
        if not tokens:
            return []

        # Build WHERE clause for metadata filters (joined from documents table)
        extra_joins = ""
        extra_where = ""
        extra_params: list[Any] = []
        if project_name or (year and year > 0):
            extra_joins = " JOIN documents d ON documents_fts.doc_id = d.doc_id"
            parts: list[str] = []
            if project_name:
                parts.append("d.project_name = ?")
                extra_params.append(project_name)
            if year and year > 0:
                parts.append("d.year = ?")
                extra_params.append(year)
            extra_where = " AND " + " AND ".join(parts)

        conn = self._connect()
        try:
            for fts_query in self._build_fts_queries(tokens):
                rows = conn.execute(
                    "SELECT documents_fts.doc_id, rank, "
                    "documents_fts.project_name, documents_fts.doc_type_ext, "
                    "snippet(documents_fts, 1, '<b>', '</b>', '...', 32) "
                    f"FROM documents_fts{extra_joins} "
                    f"WHERE documents_fts MATCH ?{extra_where} "
                    "ORDER BY rank "
                    "LIMIT ?",
                    (fts_query, *extra_params, limit),
                ).fetchall()
                if rows:
                    return [
                        {
                            "doc_id": r[0],
                            "rank": r[1],
                            "project_name": r[2],
                            "doc_type_ext": r[3],
                            "summary_snippet": r[4],
                        }
                        for r in rows
                    ]
            return []
        except Exception:
            logger.warning("FTS5 search failed", exc_info=True)
            return []
        finally:
            conn.close()

    # --- Metadata ---

    def save_metadata(
        self,
        doc_id: str,
        metadata: dict[str, Any],
        structured: dict[str, Any] | None = None,
    ) -> None:
        """Save or update metadata for a document."""

        def _save() -> None:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO document_metadata
                       (doc_id, metadata_json, structured_fields) VALUES (?, ?, ?)""",
                    (
                        doc_id,
                        json.dumps(metadata, ensure_ascii=False),
                        json.dumps(structured or {}, ensure_ascii=False),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        with self._write_lock:
            self._retry_on_lock(_save)

    def get_metadata(self, doc_id: str) -> dict[str, Any] | None:
        """Get metadata for a document."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM document_metadata WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "doc_id": row["doc_id"],
                "metadata": json.loads(row["metadata_json"] or "{}"),
                "structured_fields": json.loads(row["structured_fields"] or "{}"),
            }
        finally:
            conn.close()

    # --- Event Log ---

    def add_event(
        self,
        doc_id: str,
        event_type: str,
        message: str = "",
    ) -> None:
        """Add an event to the document event log."""
        now = datetime.now().isoformat()

        def _add() -> None:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO document_events
                       (doc_id, event_type, message, created_at) VALUES (?, ?, ?, ?)""",
                    (doc_id, event_type, message, now),
                )
                conn.commit()
            finally:
                conn.close()

        with self._write_lock:
            self._retry_on_lock(_add)

    def get_events(
        self,
        doc_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get events for a document, newest first."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT * FROM document_events
                   WHERE doc_id = ? ORDER BY created_at DESC LIMIT ?""",
                (doc_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # --- Feedback ---

    def add_feedback(
        self,
        doc_id: str,
        rating: str,
        comment: str = "",
        source: str = "web",
        *,
        reason: str = "",
    ) -> None:
        """Add feedback for a document and recompute its quality score.

        Args:
            doc_id: Document identifier.
            rating: ``"positive"`` or ``"negative"``.
            comment: Optional free-text comment.
            source: Feedback source (default ``"web"``).
            reason: Optional structured reason — one of
                ``"incorrect"``, ``"outdated"``, ``"duplicate"``, ``"exclude"``,
                or empty string.
        """
        now = datetime.now().isoformat()

        def _add() -> None:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO document_feedback
                       (doc_id, rating, comment, source, created_at, reason)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (doc_id, rating, comment, source, now, reason),
                )
                conn.commit()
            finally:
                conn.close()

        with self._write_lock:
            self._retry_on_lock(_add)

        # Auto-recompute quality after new feedback
        self.recompute_quality(doc_id)

    def get_feedback(
        self,
        doc_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get feedback for a document, newest first."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT * FROM document_feedback
                   WHERE doc_id = ? ORDER BY created_at DESC LIMIT ?""",
                (doc_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def recompute_quality(self, doc_id: str) -> None:
        """Recompute quality_score/grade from feedback and update document.

        Reason-weighted scoring:
        - ``"incorrect"`` / ``"outdated"`` negative feedback counts as 1.5x
        - Regular negative feedback counts as 1x
        - Positive feedback counts as 1x
        - ``"exclude"`` reason immediately sets ``exclude_from_search = 1``
        - ``"duplicate"`` reason immediately sets ``is_duplicate = 1``
        """
        conn = self._connect()
        try:
            row = conn.execute(
                """SELECT
                       SUM(CASE WHEN rating = 'positive' THEN 1.0 ELSE 0 END) AS positives,
                       SUM(CASE WHEN rating = 'negative' AND reason IN ('incorrect', 'outdated') THEN 1.5
                                WHEN rating = 'negative' THEN 1.0 ELSE 0 END) AS negatives,
                       SUM(CASE WHEN reason = 'exclude' THEN 1 ELSE 0 END) AS exclude_count,
                       SUM(CASE WHEN reason = 'duplicate' THEN 1 ELSE 0 END) AS duplicate_count
                   FROM document_feedback WHERE doc_id = ?""",
                (doc_id,),
            ).fetchone()
        finally:
            conn.close()

        positives = float(row["positives"] or 0) if row else 0.0
        negatives = float(row["negatives"] or 0) if row else 0.0
        exclude_count = int(row["exclude_count"] or 0) if row else 0
        duplicate_count = int(row["duplicate_count"] or 0) if row else 0
        total = positives + negatives

        if total == 0:
            score = 50.0
        else:
            score = (positives / total) * 100

        if score >= 80:
            grade = "A"
        elif score >= 50:
            grade = "B"
        elif score >= 30:
            grade = "C"
        else:
            grade = "D"

        exclude = 1 if (score < 20 and total >= 3) else 0

        # Immediate exclusion/duplicate flags from feedback reasons
        if exclude_count > 0:
            exclude = 1
        if duplicate_count > 0:
            self.update_document(doc_id, is_duplicate=1)

        self.update_document(
            doc_id,
            quality_score=score,
            quality_grade=grade,
            exclude_from_search=exclude,
        )

    # --- Exclusion helpers ---

    def get_excluded_doc_ids(self) -> list[str]:
        """Return doc_ids where exclude_from_search = 1."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT doc_id FROM documents WHERE exclude_from_search = 1",
            ).fetchall()
            return [r["doc_id"] for r in rows]
        except Exception:
            return []
        finally:
            conn.close()

    # --- Stats ---

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics for dashboard.

        All queries run inside a single DEFERRED transaction so counters
        are consistent even when concurrent writes happen.
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN DEFERRED")
            total = conn.execute("SELECT COUNT(*) as cnt FROM documents").fetchone()
            by_type = conn.execute(
                "SELECT doc_type, COUNT(*) as cnt FROM documents GROUP BY doc_type",
            ).fetchall()
            by_status = conn.execute(
                "SELECT process_status, COUNT(*) as cnt FROM documents GROUP BY process_status",
            ).fetchall()
            latest = conn.execute(
                "SELECT MAX(processed_at) as latest FROM documents",
            ).fetchone()
            by_grade = conn.execute(
                "SELECT quality_grade, COUNT(*) as cnt FROM documents GROUP BY quality_grade",
            ).fetchall()
            excluded = conn.execute(
                "SELECT COUNT(*) as cnt FROM documents WHERE exclude_from_search = 1",
            ).fetchone()
            needs_review = conn.execute(
                "SELECT COUNT(*) as cnt FROM documents WHERE quality_score < 30 AND quality_score >= 0",
            ).fetchone()
            by_category = conn.execute(
                "SELECT category, COUNT(*) as cnt FROM documents WHERE category != '' GROUP BY category",
            ).fetchall()
            embedded = conn.execute(
                "SELECT COUNT(*) as cnt FROM documents WHERE embedded_at != ''",
            ).fetchone()
            not_embedded = conn.execute(
                "SELECT COUNT(*) as cnt FROM documents WHERE embedded_at = '' OR embedded_at IS NULL",
            ).fetchone()
            by_format = conn.execute(
                "SELECT source_format, COUNT(*) as cnt FROM documents GROUP BY source_format",
            ).fetchall()
            by_year = conn.execute(
                "SELECT year, COUNT(*) as cnt FROM documents WHERE year > 0 GROUP BY year ORDER BY year DESC",
            ).fetchall()
            # Use Python local time to match ingested_at (stored as datetime.now().isoformat())
            from datetime import timedelta

            cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
            recent_24h = conn.execute(
                "SELECT COUNT(*) as cnt FROM documents WHERE ingested_at >= ?",
                (cutoff,),
            ).fetchone()

            conn.execute("COMMIT")

            return {
                "total_documents": total["cnt"] if total else 0,
                "by_type": {r["doc_type"]: r["cnt"] for r in by_type},
                "by_status": {r["process_status"]: r["cnt"] for r in by_status},
                "by_grade": {r["quality_grade"]: r["cnt"] for r in by_grade},
                "by_category": {r["category"]: r["cnt"] for r in by_category},
                "excluded_count": excluded["cnt"] if excluded else 0,
                "needs_review_count": needs_review["cnt"] if needs_review else 0,
                "latest_processed": latest["latest"] if latest else None,
                "embedded_count": embedded["cnt"] if embedded else 0,
                "not_embedded_count": not_embedded["cnt"] if not_embedded else 0,
                "by_format": {r["source_format"]: r["cnt"] for r in by_format},
                "by_year": {str(r["year"]): r["cnt"] for r in by_year},
                "recent_24h": recent_24h["cnt"] if recent_24h else 0,
            }
        finally:
            conn.close()

    def get_recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent document events with document info."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT e.event_type, e.message, e.created_at, "
                "d.doc_id, d.file_name_standard, d.doc_type, d.project_name "
                "FROM document_events e "
                "JOIN documents d ON e.doc_id = d.doc_id "
                "ORDER BY e.created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def list_unembedded(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Return documents that have not been embedded yet."""
        conn = self._connect()
        try:
            sql = "SELECT * FROM documents WHERE embedded_at = '' OR embedded_at IS NULL"
            params: list[Any] = []
            if limit is not None:
                sql += " LIMIT ?"
                params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def reset_embedded_at(self, *, doc_ids: list[str] | None = None) -> int:
        """Reset embedded_at to empty string for specified documents.

        Args:
            doc_ids: List of doc IDs to reset. If None, resets all documents.

        Returns:
            Number of rows updated.
        """
        result_count = 0

        def _reset() -> None:
            nonlocal result_count
            conn = self._connect()
            try:
                if doc_ids is None:
                    cursor = conn.execute(
                        "UPDATE documents SET embedded_at = '' WHERE embedded_at != '' AND embedded_at IS NOT NULL",
                    )
                else:
                    placeholders = ",".join("?" for _ in doc_ids)
                    cursor = conn.execute(
                        f"UPDATE documents SET embedded_at = '' WHERE doc_id IN ({placeholders})",  # noqa: S608
                        doc_ids,
                    )
                result_count = cursor.rowcount
                conn.commit()
            finally:
                conn.close()

        with self._write_lock:
            self._retry_on_lock(_reset)
        return result_count

    def delete_documents(self, doc_ids: list[str]) -> int:
        """Delete documents and all related records. Returns count deleted."""
        if not doc_ids:
            return 0
        result_count = 0

        def _delete() -> None:
            nonlocal result_count
            conn = self._connect()
            try:
                placeholders = ",".join("?" for _ in doc_ids)
                for table in (
                    "document_feedback",
                    "document_events",
                    "document_metadata",
                ):
                    conn.execute(
                        f"DELETE FROM {table} WHERE doc_id IN ({placeholders})",  # noqa: S608
                        doc_ids,
                    )
                # Also remove from doc-level FTS
                try:
                    conn.execute(
                        f"DELETE FROM documents_fts WHERE doc_id IN ({placeholders})",
                        doc_ids,
                    )
                except Exception:
                    pass  # FTS may not exist
                cursor = conn.execute(
                    f"DELETE FROM documents WHERE doc_id IN ({placeholders})",  # noqa: S608
                    doc_ids,
                )
                result_count = cursor.rowcount
                conn.commit()
            finally:
                conn.close()

        with self._write_lock:
            self._retry_on_lock(_delete)
        return result_count

    def get_unique_projects(self) -> list[str]:
        """Return all distinct non-empty project names."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT DISTINCT project_name FROM documents "
                "WHERE project_name IS NOT NULL AND project_name != '' "
                "ORDER BY project_name",
            ).fetchall()
            return [r["project_name"] for r in rows]
        finally:
            conn.close()

    def get_unique_years(self) -> list[int]:
        """Return all distinct years > 0, sorted descending."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT DISTINCT year FROM documents "
                "WHERE year IS NOT NULL AND year > 0 "
                "ORDER BY year DESC",
            ).fetchall()
            return [r["year"] for r in rows]
        finally:
            conn.close()

    def suggest(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        """Return autocomplete suggestions from corpus metadata.

        Searches project_name and doc_type_ext for substring matches.
        Returns list of {text, type, count} sorted by frequency.
        """
        if not query or len(query.strip()) < 2:
            return []
        if limit < 1:
            return []
        q = query.strip()
        pattern = f"%{q}%"
        conn = self._connect()
        try:
            results: list[dict[str, Any]] = []
            # Only suggest from searchable documents
            _searchable = "AND (exclude_from_search IS NULL OR exclude_from_search = 0)"
            # Project names containing query
            rows = conn.execute(
                "SELECT project_name, COUNT(*) as cnt FROM documents "
                "WHERE project_name IS NOT NULL AND project_name != '' "
                f"AND project_name LIKE ? {_searchable} "
                "GROUP BY project_name ORDER BY cnt DESC LIMIT ?",
                (pattern, limit),
            ).fetchall()
            for r in rows:
                results.append(
                    {"text": r["project_name"], "type": "project", "count": r["cnt"]}
                )
            # Doc type ext containing query
            rows = conn.execute(
                "SELECT doc_type_ext, COUNT(*) as cnt FROM documents "
                "WHERE doc_type_ext IS NOT NULL AND doc_type_ext != '' "
                f"AND doc_type_ext LIKE ? {_searchable} "
                "GROUP BY doc_type_ext ORDER BY cnt DESC LIMIT ?",
                (pattern, limit),
            ).fetchall()
            for r in rows:
                results.append(
                    {"text": r["doc_type_ext"], "type": "doc_type", "count": r["cnt"]}
                )
            # Sort by count descending, deduplicate, and trim to limit
            seen: set[str] = set()
            deduped: list[dict[str, Any]] = []
            for item in sorted(results, key=lambda x: x["count"], reverse=True):
                if item["text"] not in seen:
                    seen.add(item["text"])
                    deduped.append(item)
            return deduped[:limit]
        except Exception:
            logger.warning("suggest() query failed", exc_info=True)
            return []
        finally:
            conn.close()

    @property
    def document_count(self) -> int:
        """Total number of documents in the registry."""
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM documents").fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()
