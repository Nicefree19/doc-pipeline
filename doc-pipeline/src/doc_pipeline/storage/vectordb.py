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
                    logger.warning("SQLite locked, retrying in %.1fs (%d/%d)", wait, attempt + 1, max_retries)
                    time.sleep(wait)
                else:
                    raise

    def _validate_inputs(
        self, chunks: list[ChunkRecord], embeddings: Sequence[Sequence[float]],
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
            self._retry_on_lock(lambda: self._collection.add(
                ids=ids, embeddings=cast(Any, embeddings),
                documents=docs, metadatas=cast(Any, metas),
            ))
        logger.info("Added %d chunks to vector store", len(chunks))

    def upsert_chunks(
        self,
        chunks: list[ChunkRecord],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Upsert document chunks (safe for re-processing)."""
        self._validate_inputs(chunks, embeddings)
        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = self._build_metadatas(chunks)
        with self._write_lock:
            self._retry_on_lock(lambda: self._collection.upsert(
                ids=ids, embeddings=cast(Any, embeddings),
                documents=docs, metadatas=cast(Any, metas),
            ))
        logger.info("Upserted %d chunks to vector store", len(chunks))

    def upsert_chunks_local(self, chunks: list[ChunkRecord]) -> None:
        """Upsert B-grade chunks using ChromaDB's built-in local embedding.

        No external API call is made — embedding is computed locally.
        """
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = self._build_metadatas(chunks)
        with self._write_lock:
            self._retry_on_lock(lambda: self._local_collection.upsert(
                ids=ids, documents=docs, metadatas=cast(Any, metas),
            ))
        logger.info("Upserted %d chunks to local collection (B-grade)", len(chunks))

    @staticmethod
    def _parse_query_results(
        results: dict, *, source_collection: str = "",
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
                search_results.append(SearchResult(
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
                ))
        return search_results

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        doc_type_filter: str | None = None,
        *,
        category_filter: str | None = None,
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
            query_text: Raw query text (used for local collection search).
            exclude_doc_ids: Optional list of doc_ids to exclude from results.
        """
        if not query_embedding:
            logger.warning("search called with empty query_embedding")
            return []

        where = self._build_where_clause(
            doc_type_filter, category_filter, exclude_doc_ids,
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
            exclude_doc_ids: Doc IDs to exclude.
            rrf_k: RRF smoothing constant (default 60).
            year_filter: Optional year to pre-filter in ChromaDB.
            project_name_filter: Optional project name to pre-filter.
        """
        if not query_embedding:
            logger.warning("search_rrf called with empty query_embedding")
            return []

        where_full = self._build_where_clause(
            doc_type_filter, category_filter, exclude_doc_ids,
            year_filter, project_name_filter,
        )
        has_meta_filter = bool(year_filter or project_name_filter)
        where_base = self._build_where_clause(
            doc_type_filter, category_filter, exclude_doc_ids,
        ) if has_meta_filter else None

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
                        raw, source_collection="api",
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
                        raw, source_collection="local",
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
                    chunks.append({
                        "text": text,
                        "page_number": meta.get("page_number") if meta else None,
                        "chunk_index": meta.get("chunk_index", 0) if meta else 0,
                    })
            except Exception as exc:
                logger.warning("get_chunks_by_doc_id failed for %s: %s", doc_id, exc)
        # Sort by page_number (None last), then chunk_index
        chunks.sort(key=lambda c: (c["page_number"] or 9999, c["chunk_index"]))
        return chunks

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
