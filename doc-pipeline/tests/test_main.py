"""Integration tests for the Nova FastAPI server (main.py).

Uses TestClient (synchronous) with mocked Gemini/VectorStore singletons.
"""

from __future__ import annotations

import collections
import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_client():
    """Return a mock Gemini client."""
    client = MagicMock()
    # generate_content returns response with .text
    resp = MagicMock()
    resp.text = "테스트 AI 답변입니다."
    client.models.generate_content.return_value = resp
    # embed_content returns embeddings
    emb = MagicMock()
    emb.values = [0.1] * 768
    embed_result = MagicMock()
    embed_result.embeddings = [emb]
    client.models.embed_content.return_value = embed_result
    # generate_content_stream returns iterable chunks
    chunk = MagicMock()
    chunk.text = "스트리밍 토큰"
    client.models.generate_content_stream.return_value = [chunk]
    return client


@pytest.fixture()
def mock_store():
    """Return a mock VectorStore with sample search results."""
    from doc_pipeline.storage.vectordb import SearchResult

    store = MagicMock()
    store.count = 5
    _default_results = [
        SearchResult(
            doc_id="doc1", doc_type="의견서",
            project_name="테스트 프로젝트", text="샘플 문서 텍스트입니다.",
            distance=0.2, rrf_score=0.0164,
        ),
    ]
    store.search.return_value = _default_results
    store.search_rrf.return_value = _default_results
    return store


@pytest.fixture()
def empty_store():
    """Return a mock VectorStore with zero documents."""
    store = MagicMock()
    store.count = 0
    return store


@pytest.fixture()
def mock_registry(tmp_path):
    """Return a real DocumentRegistry backed by a temp DB."""
    from doc_pipeline.storage.registry import DocumentRegistry

    return DocumentRegistry(db_path=str(tmp_path / "test_reg.db"))


def _reset_rate_limiter(app):
    """Clear rate limiter state so tests don't interfere with each other."""
    for _middleware in app.user_middleware:
        pass
    # Access the actual middleware instance from the middleware stack
    from main import RateLimitMiddleware

    # Walk the middleware stack to find the RateLimitMiddleware
    handler = app.middleware_stack
    while handler is not None:
        if isinstance(handler, RateLimitMiddleware):
            handler._hits.clear()
            break
        handler = getattr(handler, "app", None)


@pytest.fixture()
def app_client(mock_client, mock_store, mock_registry):
    """TestClient with mocked singletons injected via lifespan."""
    from concurrent.futures import ThreadPoolExecutor

    from main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        # Override state AFTER lifespan runs so mocks aren't clobbered
        # by real singletons (e.g. real VectorStore from disk).
        app.state.gemini_client = mock_client
        app.state.vector_store = mock_store
        app.state.registry = mock_registry
        app.state.query_parser = None  # Tests don't need real parser
        app.state.pdf_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-pdf")
        app.state.llm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-llm")
        _reset_rate_limiter(app)
        yield client

    app.state.pdf_executor.shutdown(wait=False)
    app.state.llm_executor.shutdown(wait=False)


@pytest.fixture()
def empty_app_client(mock_client, empty_store, mock_registry):
    """TestClient with empty VectorStore."""
    from concurrent.futures import ThreadPoolExecutor

    from main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.gemini_client = mock_client
        app.state.vector_store = empty_store
        app.state.registry = mock_registry
        app.state.query_parser = None
        app.state.pdf_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-pdf")
        app.state.llm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-llm")
        yield client

    app.state.pdf_executor.shutdown(wait=False)
    app.state.llm_executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Root & Health
# ---------------------------------------------------------------------------

class TestRootAndHealth:
    def test_root(self, app_client):
        resp = app_client.get("/")
        assert resp.status_code == 200
        assert "running" in resp.json()["message"].lower()

    def test_health_ok(self, app_client):
        resp = app_client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["gemini_ok"] is True
        assert data["chroma_ok"] is True
        assert data["chroma_count"] == 5

    def test_health_degraded(self, empty_app_client):
        """Health endpoint returns count=0 for empty store."""
        resp = empty_app_client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["chroma_count"] == 0


# ---------------------------------------------------------------------------
# Search (POST batch)
# ---------------------------------------------------------------------------

class TestSearchBatch:
    def test_search_success(self, app_client):
        resp = app_client.post("/api/search", json={
            "query": "슬래브 균열 보강",
            "n_results": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert len(data["references"]) == 1
        assert data["references"][0]["doc_type"] == "의견서"

    def test_search_empty_db(self, empty_app_client):
        resp = empty_app_client.post("/api/search", json={"query": "테스트"})
        assert resp.status_code == 200
        assert "업로드" in resp.json()["answer"]

    def test_search_no_results(self, app_client, mock_store):
        mock_store.search.return_value = []
        mock_store.search_rrf.return_value = []
        resp = app_client.post("/api/search", json={"query": "존재하지않는검색어"})
        assert resp.status_code == 200
        assert "찾지 못했습니다" in resp.json()["answer"]

    def test_search_query_validation(self, app_client):
        """Empty query should be rejected by Pydantic validation."""
        resp = app_client.post("/api/search", json={"query": ""})
        assert resp.status_code == 422

    def test_search_n_results_bounds(self, app_client):
        """n_results > 20 should be rejected."""
        resp = app_client.post("/api/search", json={"query": "테스트", "n_results": 100})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Search SSE (GET stream)
# ---------------------------------------------------------------------------

class TestSearchSSE:
    def test_sse_stream_events(self, app_client):
        """SSE should yield references, token(s), and done events."""
        resp = app_client.get("/api/search/stream", params={"query": "테스트 쿼리"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        # Parse SSE events
        events = _parse_sse(resp.text)
        event_types = [e["event"] for e in events]
        assert "references" in event_types
        assert "token" in event_types
        assert "done" in event_types

        # Verify references are valid JSON
        refs_event = next(e for e in events if e["event"] == "references")
        refs = json.loads(refs_event["data"])
        assert len(refs) >= 1
        assert refs[0]["doc_type"] == "의견서"

    def test_sse_empty_db(self, empty_app_client):
        """SSE on empty DB should return error event."""
        resp = empty_app_client.get("/api/search/stream", params={"query": "테스트"})
        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        assert any(e["event"] == "server_error" for e in events)

    def test_sse_query_validation(self, app_client):
        """Empty query param should be rejected."""
        resp = app_client.get("/api/search/stream", params={"query": ""})
        assert resp.status_code == 422

    def test_sse_n_results_bounds(self, app_client):
        """n_results > 20 should be rejected."""
        resp = app_client.get(
            "/api/search/stream", params={"query": "테스트", "n_results": 50}
        )
        assert resp.status_code == 422

    def test_sse_no_results(self, app_client, mock_store):
        mock_store.search.return_value = []
        mock_store.search_rrf.return_value = []
        resp = app_client.get("/api/search/stream", params={"query": "빈결과"})
        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        assert any(e["event"] == "server_error" for e in events)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

class TestUpload:
    def test_upload_invalid_type(self, app_client):
        """Unsupported file extension should be rejected per-file."""
        resp = app_client.post(
            "/api/upload",
            files=[("files", ("test.txt", b"hello", "text/plain"))],
            data={"security_grade": "C"},
        )
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["skipped"] is True
        assert "지원하지 않는" in result["error"]

    def test_upload_success(self, app_client):
        """PDF upload with mocked pipeline should succeed."""
        # Mock process_document to return a result
        mock_result = MagicMock()
        mock_result.doc = None
        mock_result.error = ""
        mock_result.skipped = False
        mock_result.skip_reason = ""
        mock_result.chunks_stored = 3
        mock_result.summary = "테스트 요약"

        with patch("main.process_document", return_value=mock_result):
            resp = app_client.post(
                "/api/upload",
                files=[("files", ("test.pdf", b"%PDF-1.4 fake", "application/pdf"))],
                data={"security_grade": "C", "save_embed": "true"},
            )

        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["chunks_stored"] == 3
        assert result["error"] == ""

    def test_upload_with_managed_path(self, app_client, tmp_path):
        """Upload with doc result should copy file to managed storage."""
        from doc_pipeline.models.schemas import (
            DocMaster,
            DocType,
            ProcessStatus,
            SourceFormat,
        )

        doc = DocMaster(
            doc_id="upload001",
            file_name_original="원본.pdf",
            file_name_standard="2024-테스트-의견서-001.pdf",
            doc_type=DocType.OPINION,
            process_status=ProcessStatus.COMPLETED,
            source_format=SourceFormat.PDF,
        )
        mock_result = MagicMock()
        mock_result.doc = doc
        mock_result.error = ""
        mock_result.skipped = False
        mock_result.skip_reason = ""
        mock_result.chunks_stored = 5
        mock_result.summary = "테스트 요약"

        managed_dir = str(tmp_path / "managed")
        with (
            patch("main.process_document", return_value=mock_result),
            patch("main.settings") as mock_settings,
        ):
            mock_settings.registry.enabled = True
            mock_settings.registry.managed_dir = managed_dir
            mock_settings.chroma.persist_dir = str(tmp_path / "chroma")

            resp = app_client.post(
                "/api/upload",
                files=[("files", ("test.pdf", b"%PDF-1.4 fake", "application/pdf"))],
                data={"security_grade": "C", "save_embed": "true"},
            )

        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["doc_info"] is not None
        assert result["doc_info"]["doc_id"] == "upload001"
        # managed_path should be present (may be "" if copy was mocked out)
        assert "managed_path" in result["doc_info"]

    def test_upload_pipeline_error(self, app_client):
        """Pipeline exception should be caught and return sanitized error."""
        with patch("main.process_document", side_effect=RuntimeError("internal boom")):
            resp = app_client.post(
                "/api/upload",
                files=[("files", ("err.pdf", b"%PDF-1.4", "application/pdf"))],
                data={"security_grade": "C"},
            )

        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["error"]
        # Error message should be sanitized (not expose internal details)
        assert "internal boom" not in result["error"]


# ---------------------------------------------------------------------------
# Filename Sanitization
# ---------------------------------------------------------------------------

class TestSanitizeFilename:
    def test_strips_directory_traversal(self):
        from main import _sanitize_filename

        assert ".." not in _sanitize_filename("../../etc/passwd")
        assert "/" not in _sanitize_filename("sub/dir/file.pdf")
        assert "\\" not in _sanitize_filename("sub\\dir\\file.pdf")

    def test_preserves_korean_filename(self):
        from main import _sanitize_filename

        result = _sanitize_filename("2024-테스트-의견서-001.pdf")
        assert result == "2024-테스트-의견서-001.pdf"

    def test_empty_returns_unknown(self):
        from main import _sanitize_filename

        assert _sanitize_filename("") == "unknown"
        assert _sanitize_filename("...") == "unknown"

    def test_blocks_windows_reserved_names(self):
        from main import _sanitize_filename

        for name in ("CON.pdf", "NUL.txt", "COM1.docx", "LPT3.pptx"):
            result = _sanitize_filename(name)
            # Should prefix with underscore to avoid Windows device collision
            assert not result.upper().startswith(("CON.", "NUL.", "COM1.", "LPT3.")), (
                f"{name} -> {result}"
            )

    def test_null_byte_removed(self):
        from main import _sanitize_filename

        assert "\x00" not in _sanitize_filename("file\x00name.pdf")

    def test_long_filename_truncated(self):
        from main import _sanitize_filename

        long_name = "A" * 300 + ".pdf"
        result = _sanitize_filename(long_name)
        assert len(result) <= 200

    def test_unicode_normalization(self):
        from main import _sanitize_filename

        # Fullwidth period (U+FF0E U+FF0E) should be normalized
        result = _sanitize_filename("\uff0e\uff0e/etc/passwd")
        assert ".." not in result


# ---------------------------------------------------------------------------
# Draft
# ---------------------------------------------------------------------------

class TestDraft:
    def test_draft_success(self, app_client):
        ref_data = [{"index": 1, "doc_id": "d001", "doc_type": "의견서",
                      "project_name": "참고프로젝트", "similarity": 0.85,
                      "text_preview": "참고 텍스트"}]
        with patch("main.generate_draft", return_value=("# 테스트 초안", ref_data)):
            resp = app_client.post("/api/draft", json={
                "doc_type": "의견서",
                "project_name": "테스트 프로젝트",
                "issue": "슬래브 균열",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["draft"] == "# 테스트 초안"
        assert len(data["references"]) == 1
        assert data["references"][0]["doc_type"] == "의견서"

    def test_draft_references_enriched_by_registry(self, app_client):
        """Draft references are enriched with registry data when available."""
        ref_data = [{"index": 1, "doc_id": "d001", "doc_type": "의견서",
                      "project_name": "P", "similarity": 0.9, "text_preview": "t"}]
        mock_registry = MagicMock()
        mock_registry.get_document.return_value = {
            "file_name_standard": "2024-P-의견서-001.pdf",
            "managed_path": "/data/managed/2024-P-의견서-001.pdf",
            "source_path": "/uploads/original.pdf",
        }
        app_client.app.state.registry = mock_registry
        with patch("main.generate_draft", return_value=("draft", ref_data)):
            resp = app_client.post("/api/draft", json={
                "doc_type": "의견서", "project_name": "P", "issue": "i",
            })
        data = resp.json()
        assert data["references"][0]["file_name_standard"] == "2024-P-의견서-001.pdf"
        assert data["references"][0]["managed_path"] == "/data/managed/2024-P-의견서-001.pdf"

    def test_draft_no_references(self, app_client):
        """Draft with no RAG references returns empty list."""
        with patch("main.generate_draft", return_value=("# 빈 초안", [])):
            resp = app_client.post("/api/draft", json={
                "doc_type": "의견서", "project_name": "P", "issue": "i",
            })
        assert resp.json()["references"] == []

    def test_draft_error_sanitized(self, app_client):
        """Draft errors should not expose internal details."""
        with patch("main.generate_draft", side_effect=RuntimeError("secret path")):
            resp = app_client.post("/api/draft", json={
                "doc_type": "의견서",
                "project_name": "프로젝트",
                "issue": "이슈",
            })
        assert resp.status_code == 500
        assert "secret path" not in resp.json()["detail"]


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

class TestCORS:
    def test_cors_allowed_origin(self, app_client):
        resp = app_client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_cors_blocked_origin(self, app_client):
        resp = app_client.options(
            "/api/health",
            headers={
                "Origin": "http://evil.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") != "http://evil.com"


# ---------------------------------------------------------------------------
# API Key Authentication
# ---------------------------------------------------------------------------

class TestAPIKeyAuth:
    """Test API key authentication when NOVA_API_KEY is set."""

    @pytest.fixture()
    def auth_client(self, mock_client, mock_store):
        """TestClient with API key authentication enabled."""
        from concurrent.futures import ThreadPoolExecutor

        import main
        from main import app

        original_key = main._NOVA_API_KEY
        main._NOVA_API_KEY = "test-secret-key-123"
        try:
            with TestClient(app, raise_server_exceptions=False) as client:
                app.state.gemini_client = mock_client
                app.state.vector_store = mock_store
                app.state.pdf_executor = ThreadPoolExecutor(max_workers=1)
                app.state.llm_executor = ThreadPoolExecutor(max_workers=1)
                yield client
        finally:
            main._NOVA_API_KEY = original_key

    def test_protected_endpoint_rejects_no_key(self, auth_client):
        """POST /api/search without API key should return 401."""
        resp = auth_client.post("/api/search", json={"query": "test"})
        assert resp.status_code == 401

    def test_protected_endpoint_accepts_header(self, auth_client):
        """POST /api/search with X-API-Key header should succeed."""
        resp = auth_client.post(
            "/api/search",
            json={"query": "test"},
            headers={"X-API-Key": "test-secret-key-123"},
        )
        assert resp.status_code == 200

    def test_protected_endpoint_accepts_bearer(self, auth_client):
        """POST /api/search with Bearer token should succeed."""
        resp = auth_client.post(
            "/api/search",
            json={"query": "test"},
            headers={"Authorization": "Bearer test-secret-key-123"},
        )
        assert resp.status_code == 200

    def test_protected_endpoint_rejects_wrong_key(self, auth_client):
        """POST /api/search with wrong key should return 401."""
        resp = auth_client.post(
            "/api/search",
            json={"query": "test"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_health_is_public(self, auth_client):
        """GET /api/health should work without API key."""
        resp = auth_client.get("/api/health")
        assert resp.status_code == 200

    def test_root_is_public(self, auth_client):
        """GET / should work without API key."""
        resp = auth_client.get("/")
        assert resp.status_code == 200

    def test_sse_accepts_query_param(self, auth_client):
        """GET /api/search/stream with api_key query param should work."""
        resp = auth_client.get(
            "/api/search/stream",
            params={"query": "test", "api_key": "test-secret-key-123"},
        )
        assert resp.status_code == 200

    def test_rejects_empty_string_key(self, auth_client):
        """Empty string API key should be rejected."""
        resp = auth_client.post(
            "/api/search",
            json={"query": "test"},
            headers={"X-API-Key": ""},
        )
        assert resp.status_code == 401

    def test_rejects_whitespace_key(self, auth_client):
        """Whitespace-only API key should be rejected."""
        resp = auth_client.post(
            "/api/search",
            json={"query": "test"},
            headers={"X-API-Key": "   "},
        )
        assert resp.status_code == 401

    def test_rejects_bearer_without_token(self, auth_client):
        """'Bearer ' with no actual token should be rejected."""
        resp = auth_client.post(
            "/api/search",
            json={"query": "test"},
            headers={"Authorization": "Bearer "},
        )
        assert resp.status_code == 401

    def test_upload_requires_auth(self, auth_client):
        """POST /api/upload without key should return 401."""
        resp = auth_client.post(
            "/api/upload",
            files=[("files", ("t.txt", b"hello", "text/plain"))],
        )
        assert resp.status_code == 401

    def test_draft_requires_auth(self, auth_client):
        """POST /api/draft without key should return 401."""
        resp = auth_client.post("/api/draft", json={
            "doc_type": "의견서", "project_name": "p", "issue": "i",
        })
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Document Registry API
# ---------------------------------------------------------------------------

class TestDocumentsAPI:
    """Test /api/documents endpoints."""

    def _insert_sample_doc(self, app_client, doc_id="sample01"):
        """Insert a sample document into the registry via app state."""
        from doc_pipeline.models.schemas import DocMaster, DocType
        from main import app

        doc = DocMaster(
            doc_id=doc_id,
            file_name_original="test.pdf",
            file_name_standard="2024-테스트-의견서-001.pdf",
            doc_type=DocType.OPINION,
            project_name="테스트",
            year=2024,
        )
        app.state.registry.insert_document(doc, source_path="/test/path.pdf")

    def test_list_documents_empty(self, app_client):
        resp = app_client.get("/api/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["documents"] == []
        assert data["total"] == 0

    def test_list_documents_with_data(self, app_client):
        self._insert_sample_doc(app_client)
        resp = app_client.get("/api/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["documents"][0]["doc_id"] == "sample01"

    def test_list_documents_with_filter(self, app_client):
        self._insert_sample_doc(app_client, doc_id="s1")
        resp = app_client.get("/api/documents", params={"doc_type": "의견서"})
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

        resp = app_client.get("/api/documents", params={"doc_type": "계약서"})
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_document_stats(self, app_client):
        self._insert_sample_doc(app_client)
        resp = app_client.get("/api/documents/stats")
        assert resp.status_code == 200
        stats = resp.json()
        assert stats["total_documents"] == 1
        assert "의견서" in stats["by_type"]
        # Full response shape validation
        assert "by_status" in stats
        assert "latest_processed" in stats
        assert isinstance(stats["by_type"], dict)
        assert isinstance(stats["by_status"], dict)

    def test_get_document_detail(self, app_client):
        self._insert_sample_doc(app_client)
        resp = app_client.get("/api/documents/sample01")
        assert resp.status_code == 200
        data = resp.json()
        assert data["document"]["doc_id"] == "sample01"
        assert data["document"]["file_name_standard"] == "2024-테스트-의견서-001.pdf"

    def test_get_document_not_found(self, app_client):
        resp = app_client.get("/api/documents/nonexistent")
        assert resp.status_code == 404

    def test_documents_api_no_registry_list(self, app_client):
        """When registry is None, /api/documents returns 503."""
        from main import app

        original = app.state.registry
        app.state.registry = None
        try:
            resp = app_client.get("/api/documents")
            assert resp.status_code == 503
        finally:
            app.state.registry = original

    def test_documents_api_no_registry_stats(self, app_client):
        """When registry is None, /api/documents/stats returns 503."""
        from main import app

        original = app.state.registry
        app.state.registry = None
        try:
            resp = app_client.get("/api/documents/stats")
            assert resp.status_code == 503
        finally:
            app.state.registry = original

    def test_documents_api_no_registry_detail(self, app_client):
        """When registry is None, /api/documents/{id} returns 503."""
        from main import app

        original = app.state.registry
        app.state.registry = None
        try:
            resp = app_client.get("/api/documents/any_id")
            assert resp.status_code == 503
        finally:
            app.state.registry = original


# ---------------------------------------------------------------------------
# Feedback API
# ---------------------------------------------------------------------------

class TestFeedbackAPI:
    """Test /api/feedback and /api/documents/{doc_id}/feedback endpoints."""

    def _insert_sample_doc(self, app_client, doc_id="fb_doc01"):
        from doc_pipeline.models.schemas import DocMaster, DocType
        from main import app

        doc = DocMaster(
            doc_id=doc_id,
            file_name_original="test.pdf",
            file_name_standard="2024-테스트-의견서-001.pdf",
            doc_type=DocType.OPINION,
            project_name="테스트",
            year=2024,
        )
        app.state.registry.insert_document(doc, source_path="/test/path.pdf")

    def test_submit_feedback_positive(self, app_client):
        self._insert_sample_doc(app_client)
        resp = app_client.post("/api/feedback", json={
            "doc_id": "fb_doc01",
            "rating": "positive",
            "comment": "좋은 문서",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["doc_id"] == "fb_doc01"

    def test_submit_feedback_updates_quality(self, app_client):
        self._insert_sample_doc(app_client)
        from main import app

        # Submit 3 positive, 1 negative → score = 75.0
        for _ in range(3):
            app_client.post("/api/feedback", json={
                "doc_id": "fb_doc01", "rating": "positive",
            })
        app_client.post("/api/feedback", json={
            "doc_id": "fb_doc01", "rating": "negative",
        })

        doc = app.state.registry.get_document("fb_doc01")
        assert doc is not None
        assert doc["quality_score"] == 75.0
        assert doc["quality_grade"] == "B"

    def test_submit_feedback_invalid_rating(self, app_client):
        self._insert_sample_doc(app_client)
        resp = app_client.post("/api/feedback", json={
            "doc_id": "fb_doc01", "rating": "neutral",
        })
        assert resp.status_code == 422

    def test_submit_feedback_unknown_doc(self, app_client):
        resp = app_client.post("/api/feedback", json={
            "doc_id": "nonexistent", "rating": "positive",
        })
        assert resp.status_code == 404

    def test_get_document_feedback(self, app_client):
        self._insert_sample_doc(app_client)
        app_client.post("/api/feedback", json={
            "doc_id": "fb_doc01", "rating": "positive", "comment": "유용",
        })
        app_client.post("/api/feedback", json={
            "doc_id": "fb_doc01", "rating": "negative", "comment": "개선 필요",
        })

        resp = app_client.get("/api/documents/fb_doc01/feedback")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["feedback"]) == 2
        assert data["quality_score"] is not None
        assert data["quality_grade"] is not None

    def test_feedback_api_no_registry(self, app_client):
        from main import app

        original = app.state.registry
        app.state.registry = None
        try:
            resp = app_client.post("/api/feedback", json={
                "doc_id": "any", "rating": "positive",
            })
            assert resp.status_code == 503

            resp = app_client.get("/api/documents/any/feedback")
            assert resp.status_code == 503
        finally:
            app.state.registry = original


# ---------------------------------------------------------------------------
# Managed Storage & Registry Helpers
# ---------------------------------------------------------------------------

class TestCopyToManaged:
    """Tests for _copy_to_managed helper."""

    def test_copy_basic(self, tmp_path, monkeypatch):
        from main import _copy_to_managed

        # Create source file
        src = tmp_path / "src" / "test.pdf"
        src.parent.mkdir()
        src.write_bytes(b"%PDF-test")

        managed_dir = tmp_path / "managed"
        monkeypatch.setattr("main.settings.registry.enabled", True)
        monkeypatch.setattr("main.settings.registry.managed_dir", str(managed_dir))

        result = _copy_to_managed(src, "2024-test.pdf")
        assert result != ""
        assert Path(result).exists()
        assert Path(result).read_bytes() == b"%PDF-test"

    def test_copy_collision_renames(self, tmp_path, monkeypatch):
        from main import _copy_to_managed

        managed_dir = tmp_path / "managed"
        managed_dir.mkdir()
        # Pre-create file to trigger collision
        (managed_dir / "test.pdf").write_bytes(b"existing")

        src = tmp_path / "new.pdf"
        src.write_bytes(b"%PDF-new")

        monkeypatch.setattr("main.settings.registry.enabled", True)
        monkeypatch.setattr("main.settings.registry.managed_dir", str(managed_dir))

        result = _copy_to_managed(src, "test.pdf")
        assert result != ""
        assert "test_1.pdf" in result

    def test_copy_disabled_returns_empty(self, tmp_path, monkeypatch):
        from main import _copy_to_managed

        src = tmp_path / "test.pdf"
        src.write_bytes(b"data")
        monkeypatch.setattr("main.settings.registry.enabled", False)

        assert _copy_to_managed(src, "test.pdf") == ""

    def test_copy_empty_name_returns_empty(self, tmp_path, monkeypatch):
        from main import _copy_to_managed

        src = tmp_path / "test.pdf"
        src.write_bytes(b"data")
        monkeypatch.setattr("main.settings.registry.enabled", True)

        assert _copy_to_managed(src, "") == ""


class TestUpdateRegistryManagedPath:
    """Tests for _update_registry_managed_path helper."""

    def test_updates_managed_path(self, app_client):
        from doc_pipeline.models.schemas import DocMaster, DocType
        from main import _update_registry_managed_path, app

        doc = DocMaster(
            doc_id="reg_mp_001",
            file_name_original="test.pdf",
            doc_type=DocType.OPINION,
        )
        app.state.registry.insert_document(doc, source_path="/test")

        # Create a mock request with app reference
        mock_req = MagicMock()
        mock_req.app = app

        _update_registry_managed_path(mock_req, "reg_mp_001", "/managed/test.pdf")

        doc_record = app.state.registry.get_document("reg_mp_001")
        assert doc_record is not None
        assert doc_record["managed_path"] == "/managed/test.pdf"

    def test_noop_when_no_registry(self):
        from main import _update_registry_managed_path

        mock_req = MagicMock()
        mock_req.app.state = MagicMock(spec=[])  # no 'registry' attr
        # Should not raise
        _update_registry_managed_path(mock_req, "any_id", "/path")


class TestBuildRagContextRegistry:
    """Tests for _build_rag_context with registry enrichment."""

    @staticmethod
    def _make_doc_result(doc_id="d1", doc_type="의견서", project_name="프로젝트", text="테스트 텍스트", n_chunks=1):
        from doc_pipeline.search.aggregator import DocumentResult
        from doc_pipeline.storage.vectordb import SearchResult

        chunks = [
            SearchResult(
                doc_id=doc_id, doc_type=doc_type,
                project_name=project_name, text=f"{text} 청크{i+1}",
                distance=0.1 + i * 0.1, rrf_score=0.016 - i * 0.001,
                page_number=i + 1,
            )
            for i in range(n_chunks)
        ]
        return DocumentResult(
            doc_id=doc_id, doc_type=doc_type, doc_type_ext="",
            category="", project_name=project_name, year=2024,
            doc_score=0.5, best_chunk=chunks[0],
            top_chunks=chunks[:3], chunk_count=n_chunks,
        )

    def test_enriches_with_registry(self, app_client):
        from doc_pipeline.models.schemas import DocMaster, DocType
        from main import _build_rag_context, app

        # Insert a doc into registry
        doc = DocMaster(
            doc_id="rag_test01",
            file_name_original="원본.pdf",
            file_name_standard="2024-표준-의견서-001.pdf",
            doc_type=DocType.OPINION,
        )
        app.state.registry.insert_document(
            doc, source_path="/test/path.pdf",
            managed_path="/managed/2024-표준-의견서-001.pdf",
        )

        doc_results = [self._make_doc_result(doc_id="rag_test01")]

        refs, _prompt = _build_rag_context(doc_results, "테스트", registry=app.state.registry)
        assert len(refs) == 1
        assert refs[0]["file_name_standard"] == "2024-표준-의견서-001.pdf"
        assert refs[0]["managed_path"] == "/managed/2024-표준-의견서-001.pdf"
        assert refs[0]["source_path"] == "/test/path.pdf"

    def test_defaults_without_registry(self):
        from main import _build_rag_context

        doc_results = [self._make_doc_result(doc_id="no_reg01")]

        refs, _prompt = _build_rag_context(doc_results, "테스트", registry=None)
        assert refs[0]["file_name_standard"] == ""
        assert refs[0]["managed_path"] == ""
        assert refs[0]["source_path"] == ""

    def test_rag_context_uses_top_chunks(self):
        """RAG context includes text from multiple top_chunks, not just best_chunk[:500]."""
        from main import _build_rag_context

        doc_results = [self._make_doc_result(doc_id="multi_chunk", text="상세 내용", n_chunks=3)]

        refs, prompt = _build_rag_context(doc_results, "테스트", registry=None)
        assert len(refs) == 1
        # All 3 chunk texts should be present
        assert "청크1" in refs[0]["text"]
        assert "청크2" in refs[0]["text"]
        assert "청크3" in refs[0]["text"]
        # Page labels should be present
        assert "[p.1]" in refs[0]["text"]

    def test_rag_context_respects_budget(self):
        """RAG context should not exceed _MAX_CONTEXT_CHARS."""
        from main import _MAX_CONTEXT_CHARS, _build_rag_context

        # Create a doc with very long text
        doc_results = [
            self._make_doc_result(
                doc_id=f"big_{i}",
                text="X" * 5000,
                n_chunks=3,
            )
            for i in range(20)
        ]

        refs, prompt = _build_rag_context(doc_results, "테스트", registry=None)
        # The prompt context should not exceed budget
        # (Some refs may be truncated or excluded)
        context_section = prompt.split("다음은 건축·구조 엔지니어링")[1].split("위 문서들을 참고하여")[0]
        assert len(context_section) <= _MAX_CONTEXT_CHARS + 1000  # Allow slack for delimiters


# ---------------------------------------------------------------------------
# Document Download
# ---------------------------------------------------------------------------

class TestDocumentDownload:
    def test_download_success(self, app_client, tmp_path, monkeypatch):
        """Download returns file when managed_path exists."""
        managed_dir = tmp_path / "managed"
        managed_dir.mkdir()
        test_file = managed_dir / "2024-테스트-의견서-001.pdf"
        test_file.write_bytes(b"%PDF-1.4 test content")

        mock_registry = MagicMock()
        mock_registry.get_document.return_value = {
            "managed_path": str(test_file),
            "file_name_standard": "2024-테스트-의견서-001.pdf",
        }
        app_client.app.state.registry = mock_registry
        monkeypatch.setattr("main.settings.registry.managed_dir", str(managed_dir))

        resp = app_client.get("/api/documents/d001/download")
        assert resp.status_code == 200
        assert b"%PDF-1.4 test content" in resp.content

    def test_download_no_managed_path(self, app_client):
        """Download returns 404 when no managed_path."""
        mock_registry = MagicMock()
        mock_registry.get_document.return_value = {
            "managed_path": "",
            "file_name_standard": "test.pdf",
        }
        app_client.app.state.registry = mock_registry
        resp = app_client.get("/api/documents/d001/download")
        assert resp.status_code == 404

    def test_download_file_not_found(self, app_client, tmp_path, monkeypatch):
        """Download returns 404 when file doesn't exist on disk."""
        managed_dir = tmp_path / "managed"
        managed_dir.mkdir()
        monkeypatch.setattr("main.settings.registry.managed_dir", str(managed_dir))

        mock_registry = MagicMock()
        mock_registry.get_document.return_value = {
            "managed_path": str(managed_dir / "nonexistent.pdf"),
            "file_name_standard": "test.pdf",
        }
        app_client.app.state.registry = mock_registry
        resp = app_client.get("/api/documents/d001/download")
        assert resp.status_code == 404

    def test_download_doc_not_found(self, app_client):
        """Download returns 404 when document doesn't exist."""
        mock_registry = MagicMock()
        mock_registry.get_document.return_value = None
        app_client.app.state.registry = mock_registry
        resp = app_client.get("/api/documents/d001/download")
        assert resp.status_code == 404

    def test_download_path_traversal_blocked(self, app_client, tmp_path, monkeypatch):
        """Download blocks path traversal attempts."""
        managed_dir = tmp_path / "managed"
        managed_dir.mkdir()
        monkeypatch.setattr("main.settings.registry.managed_dir", str(managed_dir))

        # Create file outside managed_dir
        outside = tmp_path / "secret.txt"
        outside.write_text("secret")

        mock_registry = MagicMock()
        mock_registry.get_document.return_value = {
            "managed_path": str(outside),
            "file_name_standard": "secret.txt",
        }
        app_client.app.state.registry = mock_registry
        resp = app_client.get("/api/documents/d001/download")
        assert resp.status_code == 403

    def test_download_no_registry(self, app_client):
        """Download returns 503 when registry is not available."""
        app_client.app.state.registry = None
        resp = app_client.get("/api/documents/d001/download")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_sse(raw: str) -> list[dict[str, str]]:
    """Parse raw SSE text into a list of {event, data} dicts."""
    events = []
    current_event = "message"
    current_data = []

    for line in raw.split("\n"):
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current_data.append(line[len("data:"):].strip())
        elif line.strip() == "" and current_data:
            events.append({
                "event": current_event,
                "data": "\n".join(current_data),
            })
            current_event = "message"
            current_data = []
        elif line.startswith(":"):
            # Comment line (ping), skip
            continue

    # Capture last event if stream didn't end with blank line
    if current_data:
        events.append({
            "event": current_event,
            "data": "\n".join(current_data),
        })

    return events


# ---------------------------------------------------------------------------
# Manual Reclassify API (PATCH /api/documents/{doc_id}/classify)
# ---------------------------------------------------------------------------


class TestReclassifyAPI:
    def _insert_doc(self, app_client):
        """Helper: insert a document into the test registry."""
        from doc_pipeline.models.schemas import DocMaster, DocType, ProcessStatus, SecurityGrade

        registry = app_client.app.state.registry
        doc = DocMaster(
            doc_id="reclass001",
            file_name_original="test.pdf",
            doc_type=DocType.OPINION,
            doc_type_ext="의견서",
            category="감리",
            process_status=ProcessStatus.COMPLETED,
            security_grade=SecurityGrade.B,
        )
        registry.insert_document(doc, source_path="/test/path.pdf")

    def test_reclassify_success(self, app_client):
        self._insert_doc(app_client)
        resp = app_client.patch("/api/documents/reclass001/classify", json={
            "doc_type": "의견서",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        # Verify file_name_standard was recalculated
        doc = app_client.app.state.registry.get_document("reclass001")
        assert doc["file_name_standard"]  # should be non-empty

    def test_reclassify_not_found(self, app_client):
        resp = app_client.patch("/api/documents/nonexistent/classify", json={
            "doc_type": "의견서",
        })
        assert resp.status_code == 404

    def test_reclassify_invalid_type(self, app_client):
        self._insert_doc(app_client)
        resp = app_client.patch("/api/documents/reclass001/classify", json={
            "doc_type": "존재하지않는유형",
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# CSV Export API (GET /api/documents/export)
# ---------------------------------------------------------------------------


class TestExportAPI:
    def _insert_docs(self, app_client):
        from doc_pipeline.models.schemas import DocMaster, DocType, ProcessStatus, SecurityGrade

        registry = app_client.app.state.registry
        for i in range(3):
            doc = DocMaster(
                doc_id=f"exp{i:03d}",
                file_name_original=f"test{i}.pdf",
                doc_type=DocType.OPINION,
                process_status=ProcessStatus.COMPLETED,
                security_grade=SecurityGrade.B,
            )
            registry.insert_document(doc, source_path=f"/test/{i}")

    def test_export_csv(self, app_client):
        self._insert_docs(app_client)
        resp = app_client.get("/api/documents/export")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        body = resp.text
        assert "doc_id" in body
        assert "exp000" in body

    def test_export_csv_empty(self, app_client):
        resp = app_client.get("/api/documents/export")
        assert resp.status_code == 200
        # Should have header row but no data rows
        lines = resp.text.strip().split("\n")
        assert len(lines) >= 1  # at least header (may have BOM)

    def test_export_csv_with_filter(self, app_client):
        self._insert_docs(app_client)
        resp = app_client.get("/api/documents/export?doc_type=의견서")
        assert resp.status_code == 200
        assert "exp000" in resp.text

    def test_export_csv_with_doc_type_ext_filter(self, app_client):
        self._insert_docs(app_client)
        resp = app_client.get("/api/documents/export?doc_type_ext=의견서")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# Production API Key Enforcement (B2)
# ---------------------------------------------------------------------------


class TestProductionAPIKeyEnforcement:
    """NOVA_ENV=production without NOVA_API_KEY should abort startup."""

    def test_production_without_api_key_raises(self):
        """Lifespan should raise SystemExit(1) when NOVA_API_KEY is missing in production."""
        import main

        with (
            patch.dict(os.environ, {"NOVA_ENV": "production"}, clear=False),
            patch.object(main, "create_client", return_value=MagicMock()),
        ):
            os.environ.pop("NOVA_API_KEY", None)
            with pytest.raises((SystemExit, Exception)):
                with TestClient(main.app, raise_server_exceptions=True):
                    pass

    def test_production_with_api_key_ok(self, mock_client, mock_store, mock_registry):
        """Lifespan should not raise when NOVA_API_KEY is set in production."""
        import main

        with (
            patch.dict(os.environ, {"NOVA_ENV": "production", "NOVA_API_KEY": "test-key"}, clear=False),
            patch.object(main, "create_client", return_value=mock_client),
        ):
            with TestClient(main.app, raise_server_exceptions=True) as client:
                main.app.state.gemini_client = mock_client
                main.app.state.vector_store = mock_store
                resp = client.get("/")
                assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Production Service Validation (M3)
# ---------------------------------------------------------------------------


class TestProductionServiceValidation:
    """Critical services must be available in production."""

    def test_production_gemini_unavailable_raises(self):
        """Lifespan should SystemExit when Gemini client fails in production."""
        import main

        with (
            patch.dict(os.environ, {"NOVA_ENV": "production", "NOVA_API_KEY": "test"}, clear=False),
            patch.object(main, "create_client", side_effect=RuntimeError("fail")),
        ):
            with pytest.raises((SystemExit, Exception)):
                with TestClient(main.app, raise_server_exceptions=True):
                    pass


# ---------------------------------------------------------------------------
# Global Exception Handler (H3)
# ---------------------------------------------------------------------------


class TestGlobalExceptionHandler:
    """Unhandled exceptions should return sanitized 500 response."""

    def test_unhandled_exception_returns_500(self, app_client):
        """An endpoint that raises should return sanitized error."""
        from main import app

        @app.get("/api/_test_crash")
        async def _crash():
            raise RuntimeError("sensitive internal detail")

        resp = app_client.get("/api/_test_crash")
        assert resp.status_code == 500
        body = resp.json()
        assert "서버 내부 오류" in body["detail"]
        assert "sensitive" not in body["detail"]

    def test_http_exception_not_swallowed(self, app_client):
        """HTTPException should return proper status code, not 500."""
        # 404 from non-existent document
        resp = app_client.get("/api/documents/nonexistent_doc_id_12345")
        assert resp.status_code == 404

    def test_422_not_swallowed(self, app_client):
        """Pydantic validation error should return 422, not 500."""
        resp = app_client.post("/api/search", json={"query": ""})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Request Timeout Middleware (H5)
# ---------------------------------------------------------------------------


class TestRequestTimeoutMiddleware:
    """Timeout middleware is registered and uses correct paths."""

    def test_timeout_middleware_registered(self):
        from main import RequestTimeoutMiddleware, app

        handler = app.middleware_stack
        found = False
        while handler is not None:
            if isinstance(handler, RequestTimeoutMiddleware):
                found = True
                break
            handler = getattr(handler, "app", None)
        assert found, "RequestTimeoutMiddleware not found in middleware stack"

    def test_long_paths_skip_timeout(self):
        """Long paths (upload/SSE/draft) should bypass timeout (nginx backstop)."""
        from main import _LONG_PATHS

        assert "/api/upload" in _LONG_PATHS
        assert "/api/search/stream" in _LONG_PATHS
        assert "/api/draft" in _LONG_PATHS


# ---------------------------------------------------------------------------
# Rate Limiter Cleanup (M1)
# ---------------------------------------------------------------------------


class TestRateLimiterCleanup:
    """RateLimitMiddleware should clean up stale entries."""

    def test_cleanup_stale_removes_old_entries(self):
        from main import RateLimitMiddleware

        mw = RateLimitMiddleware.__new__(RateLimitMiddleware)
        mw._hits = {}
        mw._lock = threading.Lock()

        # Add a stale entry (old timestamp)
        stale_dq = collections.deque([time.monotonic() - 300])
        mw._hits["stale_ip"] = stale_dq

        # Add a fresh entry
        fresh_dq = collections.deque([time.monotonic()])
        mw._hits["fresh_ip"] = fresh_dq

        mw._cleanup_stale()

        assert "stale_ip" not in mw._hits
        assert "fresh_ip" in mw._hits
