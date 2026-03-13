"""ChromaDB vector storage for RAG search.

Uses PersistentClient (SQLite-backed) — suitable for single-user/small-team.
For multi-user deployments with concurrent writes, switch to HttpClient
connecting to a dedicated Chroma server (docker: chromadb/chroma).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import chromadb

from doc_pipeline.models.schemas import ChunkRecord

logger = logging.getLogger(__name__)

# Local collection name for B-grade docs (embedded locally, not via external API)
_LOCAL_COLLECTION = "doc_chunks_local"


@dataclass
class SearchResult:
    """A single search result from vector DB."""

    doc_id: str
    doc_type: str
    project_name: str
    text: str
    distance: float
    # Extended fields (all have defaults for backward compat)
    source_collection: str = ""
    doc_type_ext: str = ""
    category: str = ""
    page_number: int | None = None
    block_type: str | None = None
    chunk_index: int = 0
    ocr_confidence: float | None = None
    year: int = 0
    rrf_score: float = 0.0


class VectorStore:
    """ChromaDB-based vector store for document chunks."""

    def __init__(self, persist_dir: str, collection_name: str = "doc_chunks") -> None:
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        try:
            self._client = chromadb.PersistentClient(path=str(persist_path))
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            # Local collection for B-grade docs (uses ChromaDB default embedding)
            self._local_collection = self._client.get_or_create_collection(
                name=_LOCAL_COLLECTION,
            )
        except Exception as exc:
            logger.error("Failed to initialize ChromaDB at %s: %s", persist_dir, exc)
            raise RuntimeError(f"ChromaDB init failed: {exc}") from exc
        logger.info(
            "VectorStore ready: %s (%d chunks, %d local)",
            collection_name,
            self._collection.count(),
            self._local_collection.count(),
        )

    @staticmethod
    def _retry_on_lock(fn, *, max_retries: int = 3, base_wait: float = 0.5):  # type: ignore[no-untyped-def]
        """Retry a ChromaDB write operation on SQLite locking errors."""
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as exc:
                if "database is locked" in str(exc) and attempt < max_retries - 1:
                    wait = base_wait * (attempt + 1)
                    logger.warning(
                        "SQLite locked, retrying in %.1fs (%d/%d)", wait, attempt + 1, max_retries
                    )
                    time.sleep(wait)
                else:
                    raise

    def _validate_inputs(
        self,
        chunks: list[ChunkRecord],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Validate chunk/embedding list alignment."""
        if not chunks:
            raise ValueError("chunks list is empty")
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )

    def _build_metadatas(self, chunks: list[ChunkRecord]) -> list[dict[str, Any]]:
        """Build metadata dicts for ChromaDB from chunk records."""
        metadatas = []
        for c in chunks:
            meta: dict[str, Any] = {
                "doc_id": c.doc_id,
                "doc_type": c.doc_type.value,
                "project_name": c.project_name,
                "year": c.year,
                "chunk_index": c.chunk_index,
                "security_grade": c.security_grade.value,
            }
            if c.doc_type_ext:
                meta["doc_type_ext"] = c.doc_type_ext
            if c.category:
                meta["category"] = c.category
            if c.page_number is not None:
                meta["page_number"] = c.page_number
            if c.block_type is not None:
                meta["block_type"] = c.block_type.value
            if c.ocr_confidence is not None:
                meta["ocr_confidence"] = c.ocr_confidence
            metadatas.append(meta)
        return metadatas

    def add_chunks(
        self,
        chunks: list[ChunkRecord],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Add document chunks with their embeddings."""
        self._validate_inputs(chunks, embeddings)
        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = self._build_metadatas(chunks)
        with self._write_lock:
            self._retry_on_lock(
                lambda: self._collection.add(
                    ids=ids,
                    embeddings=cast(Any, embeddings),
                    documents=docs,
                    metadatas=cast(Any, metas),
                )
            )
        logger.info("Added %d chunks to vector store", len(chunks))

    def upsert_chunks(
        self,
        chunks: list[ChunkRecord],
        embeddings: Sequence[Sequence[float]],
        *,
        chunk_fts: "ChunkFTS | None" = None,
    ) -> None:
        """Upsert document chunks (safe for re-processing)."""
        self._validate_inputs(chunks, embeddings)
        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = self._build_metadatas(chunks)
        with self._write_lock:
            self._retry_on_lock(
                lambda: self._collection.upsert(
                    ids=ids,
                    embeddings=cast(Any, embeddings),
                    documents=docs,
                    metadatas=cast(Any, metas),
                )
            )
        logger.info("Upserted %d chunks to vector store", len(chunks))
        if chunk_fts:
            chunk_fts.upsert(chunks)

    def upsert_chunks_local(
        self, chunks: list[ChunkRecord], *, chunk_fts: "ChunkFTS | None" = None
    ) -> None:
        """Upsert B-grade chunks using ChromaDB's built-in local embedding.

        No external API call is made — embedding is computed locally.
        """
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = self._build_metadatas(chunks)
        with self._write_lock:
            self._retry_on_lock(
                lambda: self._local_collection.upsert(
                    ids=ids,
                    documents=docs,
                    metadatas=cast(Any, metas),
                )
            )
        logger.info("Upserted %d chunks to local collection (B-grade)", len(chunks))
        if chunk_fts:
            chunk_fts.upsert(chunks)

    @staticmethod
    def _parse_query_results(
        results: dict,
        *,
        source_collection: str = "",
    ) -> list[SearchResult]:
        """Parse ChromaDB query results into SearchResult list."""
        search_results: list[SearchResult] = []
        documents = results.get("documents")
        metadatas = results.get("metadatas")
        distances = results.get("distances")

        if documents and documents[0] and metadatas and distances:
            for i, doc_text in enumerate(documents[0]):
                meta = metadatas[0][i]
                dist = distances[0][i]
                page = meta.get("page_number")
                ocr_conf = meta.get("ocr_confidence")
                search_results.append(
                    SearchResult(
                        doc_id=str(meta.get("doc_id", "")),
                        doc_type=str(meta.get("doc_type", "")),
                        project_name=str(meta.get("project_name", "")),
                        text=doc_text or "",
                        distance=float(dist),
                        source_collection=source_collection,
                        doc_type_ext=str(meta.get("doc_type_ext", "")),
                        category=str(meta.get("category", "")),
                        page_number=int(page) if page is not None else None,
                        block_type=str(meta.get("block_type", "")) or None,
                        chunk_index=int(meta.get("chunk_index", 0)),
                        ocr_confidence=float(ocr_conf) if ocr_conf is not None else None,
                        year=int(meta.get("year", 0)),
                    )
                )
        return search_results

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        doc_type_filter: str | None = None,
        *,
        category_filter: str | None = None,
        doc_type_ext_filter: str | None = None,
        query_text: str = "",
        exclude_doc_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents by embedding.

        Searches both the API-embedded collection (C-grade) and the
        locally-embedded collection (B-grade), merging results by distance.

        Args:
            query_embedding: Gemini embedding vector for the query.
            n_results: Max results to return.
            doc_type_filter: Optional doc_type to filter by.
            category_filter: Optional category to filter by.
            doc_type_ext_filter: Optional doc_type_ext to filter by.
            query_text: Raw query text (used for local collection search).
            exclude_doc_ids: Optional list of doc_ids to exclude from results.
        """
        if not query_embedding:
            logger.warning("search called with empty query_embedding")
            return []

        where = self._build_where_clause(
            doc_type_filter,
            category_filter,
            doc_type_ext_filter,
            exclude_doc_ids,
        )

        all_results: list[SearchResult] = []

        # 1. Search API-embedded collection (C-grade docs)
        api_total = self._collection.count()
        if api_total > 0:
            try:
                results = self._collection.query(
                    query_embeddings=cast(Any, [query_embedding]),
                    n_results=min(n_results, api_total),
                    where=cast(Any, where),
                    include=["documents", "metadatas", "distances"],
                )
                all_results.extend(
                    self._parse_query_results(results, source_collection="api"),
                )
            except Exception as exc:
                logger.error("ChromaDB API collection search failed: %s", exc)

        # 2. Search locally-embedded collection (B-grade docs)
        local_total = self._local_collection.count()
        if local_total > 0 and query_text:
            try:
                results = self._local_collection.query(
                    query_texts=[query_text],
                    n_results=min(n_results, local_total),
                    where=cast(Any, where),
                    include=["documents", "metadatas", "distances"],
                )
                all_results.extend(
                    self._parse_query_results(results, source_collection="local"),
                )
            except Exception as exc:
                logger.error("ChromaDB local collection search failed: %s", exc)

        # Merge by distance (lower = more similar for cosine)
        all_results.sort(key=lambda r: r.distance)
        return all_results[:n_results]

    @staticmethod
    def _build_where_clause(
        doc_type_filter: str | None,
        category_filter: str | None,
        doc_type_ext_filter: str | None,
        exclude_doc_ids: list[str] | None,
        year_filter: int | None = None,
        project_name_filter: str | None = None,
    ) -> dict | None:
        """Build ChromaDB where clause from filter parameters."""
        conditions: list[dict[str, Any]] = []
        if doc_type_filter:
            conditions.append({"doc_type": doc_type_filter})
        if category_filter:
            conditions.append({"category": category_filter})
        if doc_type_ext_filter:
            conditions.append({"doc_type_ext": doc_type_ext_filter})
        if exclude_doc_ids:
            conditions.append({"doc_id": {"$nin": exclude_doc_ids}})
        if year_filter and year_filter > 0:
            conditions.append({"year": year_filter})
        if project_name_filter:
            conditions.append({"project_name": project_name_filter})

        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"$and": conditions}
        return None

    def search_rrf(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        *,
        query_text: str = "",
        doc_type_filter: str | None = None,
        category_filter: str | None = None,
        doc_type_ext_filter: str | None = None,
        exclude_doc_ids: list[str] | None = None,
        rrf_k: int = 60,
        year_filter: int | None = None,
        project_name_filter: str | None = None,
    ) -> list[SearchResult]:
        """Search using Reciprocal Rank Fusion across collections.

        Instead of merging by raw distance (which is incomparable between
        L2 and cosine spaces), this method ranks results from each
        collection independently and assigns RRF scores:

            ``rrf_score = 1.0 / (rrf_k + rank)``   (rank is 1-indexed)

        When a chunk appears in both collections, scores are summed.

        If metadata filters (year/project) produce fewer than *n_results*
        hits, a fallback unfiltered search is performed and the two result
        sets are merged via RRF (duplicates deduplicated).

        Args:
            query_embedding: Gemini embedding vector.
            n_results: Max results to return.
            query_text: Raw query text (for local collection).
            doc_type_filter: Optional doc_type filter.
            category_filter: Optional category filter.
            doc_type_ext_filter: Optional doc_type_ext filter.
            exclude_doc_ids: Doc IDs to exclude.
            rrf_k: RRF smoothing constant (default 60).
            year_filter: Optional year to pre-filter in ChromaDB.
            project_name_filter: Optional project name to pre-filter.
        """
        if not query_embedding:
            logger.warning("search_rrf called with empty query_embedding")
            return []

        where_full = self._build_where_clause(
            doc_type_filter,
            category_filter,
            doc_type_ext_filter,
            exclude_doc_ids,
            year_filter,
            project_name_filter,
        )
        has_meta_filter = bool(year_filter or project_name_filter)
        where_base = (
            self._build_where_clause(
                doc_type_filter,
                category_filter,
                doc_type_ext_filter,
                exclude_doc_ids,
            )
            if has_meta_filter
            else None
        )

        # Fetch 5x candidates for better recall
        fetch_n = n_results * 5

        def _query_collections(
            where: dict | None,
        ) -> tuple[list[SearchResult], list[SearchResult]]:
            api_results: list[SearchResult] = []
            local_results: list[SearchResult] = []

            api_total = self._collection.count()
            if api_total > 0:
                try:
                    raw = self._collection.query(
                        query_embeddings=cast(Any, [query_embedding]),
                        n_results=min(fetch_n, api_total),
                        where=cast(Any, where),
                        include=["documents", "metadatas", "distances"],
                    )
                    api_results = self._parse_query_results(
                        raw,
                        source_collection="api",
                    )
                except Exception as exc:
                    logger.error("ChromaDB API search failed: %s", exc)

            local_total = self._local_collection.count()
            if local_total > 0 and query_text:
                try:
                    raw = self._local_collection.query(
                        query_texts=[query_text],
                        n_results=min(fetch_n, local_total),
                        where=cast(Any, where),
                        include=["documents", "metadatas", "distances"],
                    )
                    local_results = self._parse_query_results(
                        raw,
                        source_collection="local",
                    )
                except Exception as exc:
                    logger.error("ChromaDB local search failed: %s", exc)

            return api_results, local_results

        api_results, local_results = _query_collections(where_full)

        # Fallback: if metadata filter produced too few results, search without it
        api_fallback: list[SearchResult] = []
        local_fallback: list[SearchResult] = []
        if has_meta_filter and (len(api_results) + len(local_results)) < n_results:
            api_fallback, local_fallback = _query_collections(where_base)

        # 3. RRF scoring
        # Key: (doc_id, chunk_index) → SearchResult with accumulated rrf_score
        merged: dict[tuple[str, int], SearchResult] = {}

        for rank, r in enumerate(api_results, 1):
            key = (r.doc_id, r.chunk_index)
            score = 1.0 / (rrf_k + rank)
            if key in merged:
                merged[key].rrf_score += score
            else:
                r.rrf_score = score
                merged[key] = r

        for rank, r in enumerate(local_results, 1):
            key = (r.doc_id, r.chunk_index)
            score = 1.0 / (rrf_k + rank)
            if key in merged:
                merged[key].rrf_score += score
            else:
                r.rrf_score = score
                r.source_collection = "local"
                merged[key] = r

        # Merge fallback results (lower base score since they didn't match metadata)
        for rank, r in enumerate(api_fallback, 1):
            key = (r.doc_id, r.chunk_index)
            score = 1.0 / (rrf_k + rank)
            if key in merged:
                merged[key].rrf_score += score * 0.5  # Penalize unfiltered
            else:
                r.rrf_score = score * 0.5
                merged[key] = r

        for rank, r in enumerate(local_fallback, 1):
            key = (r.doc_id, r.chunk_index)
            score = 1.0 / (rrf_k + rank)
            if key in merged:
                merged[key].rrf_score += score * 0.5
            else:
                r.rrf_score = score * 0.5
                r.source_collection = "local"
                merged[key] = r

        # 4. Sort by RRF score (higher = better)
        ranked = sorted(merged.values(), key=lambda r: r.rrf_score, reverse=True)
        return ranked[:n_results]

    def search_hybrid(
        self,
        query_embedding: list[float],
        *,
        query_text: str = "",
        n_results: int = 5,
        doc_type_filter: str | None = None,
        category_filter: str | None = None,
        doc_type_ext_filter: str | None = None,
        exclude_doc_ids: list[str] | None = None,
        year_filter: int | None = None,
        project_name_filter: str | None = None,
        chunk_fts: "ChunkFTS | None" = None,
        fts_weight: float = 0.3,
        vector_weight: float = 0.7,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """Hybrid search combining vector RRF and chunk-level FTS5.

        1. Run search_rrf() for vector results.
        2. Run chunk_fts.search() for FTS results.
        3. Merge at chunk level using weighted RRF.

        Falls back to pure vector search if chunk_fts is None or FTS fails.
        """
        # Vector search (always runs)
        vector_results = self.search_rrf(
            query_embedding,
            n_results=n_results,
            query_text=query_text,
            doc_type_filter=doc_type_filter,
            category_filter=category_filter,
            doc_type_ext_filter=doc_type_ext_filter,
            exclude_doc_ids=exclude_doc_ids,
            rrf_k=rrf_k,
            year_filter=year_filter,
            project_name_filter=project_name_filter,
        )

        # FTS search (optional)
        if not chunk_fts or not query_text:
            return vector_results

        try:
            fts_hits = chunk_fts.search(
                query_text,
                limit=n_results * 5,
                project_name_filter=project_name_filter,
                doc_type_ext_filter=doc_type_ext_filter,
            )
        except Exception:
            logger.warning("FTS5 search failed, using vector-only", exc_info=True)
            return vector_results

        if not fts_hits:
            return vector_results

        # Merge: key by (doc_id, chunk_index derived from chunk_id)
        merged: dict[str, SearchResult] = {}

        # Add vector results with weighted score
        for r in vector_results:
            key = f"{r.doc_id}::{r.chunk_index}"
            r.rrf_score *= vector_weight
            merged[key] = r

        # Add FTS results with weighted RRF score
        for rank, hit in enumerate(fts_hits, 1):
            fts_score = (1.0 / (rrf_k + rank)) * fts_weight
            chunk_id = hit["chunk_id"]
            doc_id = hit["doc_id"]

            # Try to parse chunk_index from chunk_id (format: "doc_id_N")
            chunk_index = 0
            if "_" in chunk_id:
                try:
                    chunk_index = int(chunk_id.rsplit("_", 1)[1])
                except (ValueError, IndexError):
                    pass

            key = f"{doc_id}::{chunk_index}"
            if key in merged:
                merged[key].rrf_score += fts_score
            else:
                # Create a minimal SearchResult for FTS-only hits (with metadata from FTS)
                merged[key] = SearchResult(
                    doc_id=doc_id,
                    doc_type="",
                    doc_type_ext=hit.get("doc_type_ext", ""),
                    project_name=hit.get("project_name", ""),
                    text="",
                    distance=1.0,
                    chunk_index=chunk_index,
                    rrf_score=fts_score,
                    source_collection="fts",
                )

        # Post-filter: if year_filter is set, drop FTS-only hits that lack year
        # (ChunkFTS has no year column, so FTS-only hits always have year=0)
        if year_filter:
            merged = {
                k: v for k, v in merged.items()
                if v.source_collection != "fts" or v.year == year_filter
            }

        ranked = sorted(merged.values(), key=lambda r: r.rrf_score, reverse=True)
        return ranked[:n_results]

    def get_chunks_by_doc_id(self, doc_id: str) -> list[dict[str, Any]]:
        """Return all chunks for a given doc_id, sorted by page_number then chunk_index."""
        chunks: list[dict[str, Any]] = []
        for col in (self._collection, self._local_collection):
            total = col.count()
            if total == 0:
                continue
            try:
                result = col.get(
                    where={"doc_id": doc_id},
                    include=["documents", "metadatas"],
                )
                docs = result.get("documents") or []
                metas = result.get("metadatas") or []
                for text, meta in zip(docs, metas):
                    if not text:
                        continue
                    chunks.append(
                        {
                            "text": text,
                            "page_number": meta.get("page_number") if meta else None,
                            "chunk_index": meta.get("chunk_index", 0) if meta else 0,
                        }
                    )
            except Exception as exc:
                logger.warning("get_chunks_by_doc_id failed for %s: %s", doc_id, exc)
        # Sort by page_number (None last), then chunk_index
        chunks.sort(key=lambda c: (c["page_number"] or 9999, c["chunk_index"]))
        return chunks

    def delete_by_doc_ids(self, doc_ids: list[str]) -> int:
        """Delete all chunks for given doc_ids from both collections."""
        if not doc_ids:
            return 0
        deleted = 0
        with self._write_lock:
            for col in (self._collection, self._local_collection):
                try:
                    existing = col.get(where={"doc_id": {"$in": doc_ids}})
                    if existing and existing["ids"]:
                        col.delete(ids=existing["ids"])
                        deleted += len(existing["ids"])
                except Exception as exc:
                    logger.warning("Failed to delete from %s: %s", col.name, exc)
        return deleted

    @property
    def count(self) -> int:
        """Return total chunks across all collections."""
        return self._collection.count() + self._local_collection.count()

    @property
    def doc_count(self) -> int:
        """Return number of unique documents across all collections."""
        unique_ids: set[str] = set()
        for col in (self._collection, self._local_collection):
            total = col.count()
            if total == 0:
                continue
            result = col.get(include=["metadatas"])
            metadatas = result.get("metadatas") or []
            unique_ids.update(m.get("doc_id") for m in metadatas if m)
        return len(unique_ids)


class ChunkFTS:
    """SQLite FTS5 index for chunk text search.

    Stored in a separate SQLite database from ChromaDB.
    Enables exact keyword matching that complements vector similarity search.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._init_fts5()

    def _connect(self) -> Any:
        import sqlite3

        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_fts5(self) -> None:
        """Create FTS5 table and populate from existing data if needed."""
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    doc_id UNINDEXED,
                    text,
                    project_name,
                    doc_type_ext,
                    tokenize='unicode61 remove_diacritics 2'
                )
            """
            )
            conn.commit()
        finally:
            conn.close()

    def upsert(self, chunks: list[ChunkRecord]) -> None:
        """Insert or replace chunks in the FTS index."""
        if not chunks:
            return
        conn = self._connect()
        try:
            with self._write_lock:
                for c in chunks:
                    # DELETE then INSERT (FTS5 doesn't support true UPSERT)
                    conn.execute(
                        "DELETE FROM chunks_fts WHERE chunk_id = ?",
                        (c.chunk_id,),
                    )
                    conn.execute(
                        "INSERT INTO chunks_fts (chunk_id, doc_id, text, project_name, doc_type_ext) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            c.chunk_id,
                            c.doc_id,
                            c.text,
                            c.project_name,
                            getattr(c, "doc_type_ext", ""),
                        ),
                    )
                conn.commit()
            logger.info("FTS5 upserted %d chunks", len(chunks))
        except Exception:
            logger.warning("FTS5 upsert failed", exc_info=True)
        finally:
            conn.close()

    @staticmethod
    def _build_fts_queries(tokens: list[str]) -> list[str]:
        """Build FTS5 queries in order: phrase -> AND -> OR."""
        queries: list[str] = []
        if len(tokens) >= 2:
            queries.append('"' + " ".join(tokens) + '"')  # phrase
        queries.append(" AND ".join(f'"{t}"' for t in tokens))  # AND
        queries.append(" OR ".join(f'"{t}"' for t in tokens))   # OR
        return queries

    def search(
        self,
        query: str,
        *,
        limit: int = 50,
        project_name_filter: str | None = None,
        doc_type_ext_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search FTS5 index with phrase-first strategy and metadata filters.

        Returns list of {chunk_id, doc_id, rank, project_name, doc_type_ext}.
        """
        if not query or not query.strip():
            return []

        tokens = [t.strip() for t in query.split() if len(t.strip()) >= 2]
        if not tokens:
            return []

        # Build WHERE clause for metadata filters
        where_parts: list[str] = []
        where_params: list[Any] = []
        if project_name_filter:
            where_parts.append("project_name = ?")
            where_params.append(project_name_filter)
        if doc_type_ext_filter:
            where_parts.append("doc_type_ext = ?")
            where_params.append(doc_type_ext_filter)
        extra_where = (" AND " + " AND ".join(where_parts)) if where_parts else ""

        conn = self._connect()
        try:
            # Strategy: phrase -> AND -> OR
            for fts_query in self._build_fts_queries(tokens):
                rows = conn.execute(
                    f"SELECT chunk_id, doc_id, rank, project_name, doc_type_ext "
                    f"FROM chunks_fts WHERE chunks_fts MATCH ?{extra_where} "
                    f"ORDER BY rank LIMIT ?",
                    (fts_query, *where_params, limit),
                ).fetchall()
                if rows:
                    return [
                        {
                            "chunk_id": r[0],
                            "doc_id": r[1],
                            "rank": r[2],
                            "project_name": r[3],
                            "doc_type_ext": r[4],
                        }
                        for r in rows
                    ]
            return []
        except Exception:
            logger.warning("FTS5 search failed for query: %s", query, exc_info=True)
            return []
        finally:
            conn.close()

    def delete_by_doc_ids(self, doc_ids: list[str]) -> int:
        """Delete all FTS entries for given doc_ids."""
        if not doc_ids:
            return 0
        conn = self._connect()
        try:
            with self._write_lock:
                placeholders = ",".join("?" for _ in doc_ids)
                cursor = conn.execute(
                    f"DELETE FROM chunks_fts WHERE doc_id IN ({placeholders})",
                    doc_ids,
                )
                deleted = cursor.rowcount
                conn.commit()
            return deleted
        except Exception:
            logger.warning("FTS5 delete_by_doc_ids failed", exc_info=True)
            return 0
        finally:
            conn.close()

    @property
    def count(self) -> int:
        """Return number of rows in FTS index."""
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()
            return row[0] if row else 0
        except Exception:
            return 0
        finally:
            conn.close()
