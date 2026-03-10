"""Tests for doc_pipeline.storage.registry module."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from doc_pipeline.models.schemas import (
    DocMaster,
    DocType,
    ProcessStatus,
    SecurityGrade,
    SourceFormat,
)
from doc_pipeline.storage.registry import DocumentRegistry


@pytest.fixture()
def registry(tmp_path: Path) -> DocumentRegistry:
    """Create a fresh registry in a temp directory."""
    return DocumentRegistry(db_path=str(tmp_path / "test_registry.db"))


@pytest.fixture()
def sample_doc() -> DocMaster:
    """Create a sample DocMaster for testing."""
    return DocMaster(
        doc_id="test001",
        file_name_original="원본파일.pdf",
        file_name_standard="2024-테스트-의견서-001.pdf",
        doc_type=DocType.OPINION,
        project_name="테스트프로젝트",
        year=2024,
        page_count=5,
        process_status=ProcessStatus.COMPLETED,
        security_grade=SecurityGrade.C,
        summary="테스트 요약",
        source_format=SourceFormat.PDF,
    )


class TestDocumentRegistry:
    def test_init_creates_tables(self, registry: DocumentRegistry) -> None:
        import sqlite3

        conn = sqlite3.connect(registry._db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        table_names = {t[0] for t in tables}
        assert "documents" in table_names
        assert "document_metadata" in table_names
        assert "document_events" in table_names

    def test_insert_and_get_document(
        self, registry: DocumentRegistry, sample_doc: DocMaster,
    ) -> None:
        doc_id = registry.insert_document(
            sample_doc, source_path="/test/path.pdf", hash_sha256="abc123",
        )
        assert doc_id == "test001"

        doc = registry.get_document("test001")
        assert doc is not None
        assert doc["file_name_original"] == "원본파일.pdf"
        assert doc["file_name_standard"] == "2024-테스트-의견서-001.pdf"
        assert doc["doc_type"] == "의견서"
        assert doc["project_name"] == "테스트프로젝트"
        assert doc["year"] == 2024
        assert doc["hash_sha256"] == "abc123"
        assert doc["source_path"] == "/test/path.pdf"

    def test_insert_duplicate_doc_id_raises(
        self, registry: DocumentRegistry, sample_doc: DocMaster,
    ) -> None:
        registry.insert_document(sample_doc, source_path="/path1")
        with pytest.raises(sqlite3.IntegrityError):
            registry.insert_document(sample_doc, source_path="/path2")

    def test_update_document_fields(
        self, registry: DocumentRegistry, sample_doc: DocMaster,
    ) -> None:
        registry.insert_document(sample_doc, source_path="/test")
        registry.update_document("test001", summary="새 요약", quality_score=85.0)

        doc = registry.get_document("test001")
        assert doc is not None
        assert doc["summary"] == "새 요약"
        assert doc["quality_score"] == 85.0

    def test_update_invalid_field_raises(
        self, registry: DocumentRegistry, sample_doc: DocMaster,
    ) -> None:
        registry.insert_document(sample_doc, source_path="/test")
        with pytest.raises(ValueError, match="Cannot update"):
            registry.update_document("test001", file_name_original="hack")

    def test_list_documents_no_filter(
        self, registry: DocumentRegistry, sample_doc: DocMaster,
    ) -> None:
        registry.insert_document(sample_doc, source_path="/test")
        docs = registry.list_documents()
        assert len(docs) == 1
        assert docs[0]["doc_id"] == "test001"

    def test_list_documents_with_filters(
        self, registry: DocumentRegistry,
    ) -> None:
        for i, dtype in enumerate(["의견서", "계약서", "의견서"]):
            doc = DocMaster(
                doc_id=f"doc{i:03d}",
                file_name_original=f"file{i}.pdf",
                doc_type=DocType(dtype),
                year=2024,
            )
            registry.insert_document(doc, source_path=f"/path/{i}")

        opinions = registry.list_documents(doc_type="의견서")
        assert len(opinions) == 2

        contracts = registry.list_documents(doc_type="계약서")
        assert len(contracts) == 1

    def test_list_documents_pagination(
        self, registry: DocumentRegistry,
    ) -> None:
        for i in range(10):
            doc = DocMaster(
                doc_id=f"doc{i:03d}",
                file_name_original=f"file{i}.pdf",
                doc_type=DocType.OPINION,
            )
            registry.insert_document(doc, source_path=f"/path/{i}")

        page1 = registry.list_documents(limit=3, offset=0)
        assert len(page1) == 3

        page2 = registry.list_documents(limit=3, offset=3)
        assert len(page2) == 3

        # No overlap
        ids1 = {d["doc_id"] for d in page1}
        ids2 = {d["doc_id"] for d in page2}
        assert ids1.isdisjoint(ids2)

    def test_count_documents(
        self, registry: DocumentRegistry,
    ) -> None:
        for i in range(5):
            doc = DocMaster(
                doc_id=f"doc{i:03d}",
                file_name_original=f"file{i}.pdf",
                doc_type=DocType.OPINION if i < 3 else DocType.CONTRACT,
            )
            registry.insert_document(doc, source_path=f"/p/{i}")

        assert registry.count_documents() == 5
        assert registry.count_documents(doc_type="의견서") == 3
        assert registry.count_documents(doc_type="계약서") == 2

    def test_count_documents_category_filter(
        self, registry: DocumentRegistry,
    ) -> None:
        """count_documents with category filter must match list_documents."""
        for i, cat in enumerate(["설계", "설계", "시공", "감리"]):
            doc = DocMaster(
                doc_id=f"cat{i:03d}",
                file_name_original=f"cat{i}.pdf",
                doc_type=DocType.OPINION,
                category=cat,
            )
            registry.insert_document(doc, source_path=f"/p/{i}")

        assert registry.count_documents(category="설계") == 2
        assert registry.count_documents(category="시공") == 1
        assert registry.count_documents(category="감리") == 1
        assert registry.count_documents(category="없는카테고리") == 0
        # count must match list length
        listed = registry.list_documents(category="설계")
        assert registry.count_documents(category="설계") == len(listed)

    def test_count_documents_exclude_search(
        self, registry: DocumentRegistry,
    ) -> None:
        for i in range(3):
            doc = DocMaster(
                doc_id=f"ex{i:03d}",
                file_name_original=f"ex{i}.pdf",
                doc_type=DocType.OPINION,
            )
            registry.insert_document(doc, source_path=f"/p/{i}")
        # Mark one as excluded
        registry.update_document("ex001", exclude_from_search=True)
        assert registry.count_documents() == 3
        assert registry.count_documents(exclude_search=False) == 2
        assert registry.count_documents(exclude_search=True) == 1

    def test_find_by_hash(
        self, registry: DocumentRegistry, sample_doc: DocMaster,
    ) -> None:
        registry.insert_document(
            sample_doc, source_path="/test", hash_sha256="deadbeef",
        )
        found = registry.find_by_hash("deadbeef")
        assert found is not None
        assert found["doc_id"] == "test001"

    def test_find_by_hash_not_found(self, registry: DocumentRegistry) -> None:
        assert registry.find_by_hash("nonexistent") is None

    def test_find_by_hash_empty_string(self, registry: DocumentRegistry) -> None:
        assert registry.find_by_hash("") is None

    def test_save_and_get_metadata(
        self, registry: DocumentRegistry, sample_doc: DocMaster,
    ) -> None:
        registry.insert_document(sample_doc, source_path="/test")
        registry.save_metadata(
            "test001",
            metadata={"site_name": "화성동탄"},
            structured={"type": "opinion"},
        )
        meta = registry.get_metadata("test001")
        assert meta is not None
        assert meta["metadata"]["site_name"] == "화성동탄"
        assert meta["structured_fields"]["type"] == "opinion"

    def test_get_metadata_not_found(self, registry: DocumentRegistry) -> None:
        assert registry.get_metadata("nonexistent") is None

    def test_add_and_get_events(
        self, registry: DocumentRegistry, sample_doc: DocMaster,
    ) -> None:
        registry.insert_document(sample_doc, source_path="/test")
        registry.add_event("test001", "processed", "Pipeline completed")
        registry.add_event("test001", "indexed", "Vector stored")

        events = registry.get_events("test001")
        assert len(events) == 2
        assert events[0]["event_type"] == "indexed"  # newest first
        assert events[1]["event_type"] == "processed"

    def test_get_stats(
        self, registry: DocumentRegistry,
    ) -> None:
        for i, dtype in enumerate(["의견서", "계약서", "의견서"]):
            doc = DocMaster(
                doc_id=f"doc{i:03d}",
                file_name_original=f"file{i}.pdf",
                doc_type=DocType(dtype),
                process_status=ProcessStatus.COMPLETED,
            )
            registry.insert_document(doc, source_path=f"/p/{i}")

        stats = registry.get_stats()
        assert stats["total_documents"] == 3
        assert stats["by_type"]["의견서"] == 2
        assert stats["by_type"]["계약서"] == 1
        assert stats["by_status"]["완료"] == 3
        assert "by_grade" in stats
        assert "excluded_count" in stats
        assert "needs_review_count" in stats
        assert stats["excluded_count"] == 0
        assert stats["needs_review_count"] == 0

    def test_document_count_property(
        self, registry: DocumentRegistry, sample_doc: DocMaster,
    ) -> None:
        assert registry.document_count == 0
        registry.insert_document(sample_doc, source_path="/test")
        assert registry.document_count == 1

    def test_insert_with_metadata(
        self, registry: DocumentRegistry, sample_doc: DocMaster,
    ) -> None:
        registry.insert_document(
            sample_doc, source_path="/test",
            metadata={"client": "테스트사"},
        )
        meta = registry.get_metadata("test001")
        assert meta is not None
        assert meta["metadata"]["client"] == "테스트사"

    def test_insert_with_managed_path(
        self, registry: DocumentRegistry, sample_doc: DocMaster,
    ) -> None:
        registry.insert_document(
            sample_doc, source_path="(Nova Web Upload)",
            managed_path="/data/managed/2024-테스트-의견서-001.pdf",
        )
        doc = registry.get_document("test001")
        assert doc is not None
        assert doc["source_path"] == "(Nova Web Upload)"
        assert doc["managed_path"] == "/data/managed/2024-테스트-의견서-001.pdf"

    def test_update_managed_path(
        self, registry: DocumentRegistry, sample_doc: DocMaster,
    ) -> None:
        registry.insert_document(sample_doc, source_path="(Nova Web Upload)")
        doc = registry.get_document("test001")
        assert doc is not None
        assert doc["managed_path"] == ""

        registry.update_document(
            "test001", managed_path="/data/managed/test.pdf",
        )
        doc = registry.get_document("test001")
        assert doc is not None
        assert doc["managed_path"] == "/data/managed/test.pdf"

    def test_concurrent_writes(
        self, registry: DocumentRegistry,
    ) -> None:
        """Verify concurrent inserts don't corrupt the database."""
        errors: list[Exception] = []

        def insert_doc(idx: int) -> None:
            try:
                doc = DocMaster(
                    doc_id=f"concurrent{idx:03d}",
                    file_name_original=f"concurrent{idx}.pdf",
                    doc_type=DocType.OPINION,
                )
                registry.insert_document(doc, source_path=f"/concurrent/{idx}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=insert_doc, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent insert errors: {errors}"
        assert registry.document_count == 10

    def test_add_and_get_feedback(
        self, registry: DocumentRegistry, sample_doc: DocMaster,
    ) -> None:
        """Feedback can be added and retrieved for a document."""
        registry.insert_document(sample_doc, source_path="/test")
        registry.add_feedback("test001", "positive", comment="유용한 문서")
        registry.add_feedback("test001", "negative", comment="오래된 정보")

        feedback = registry.get_feedback("test001")
        assert len(feedback) == 2
        assert feedback[0]["rating"] == "negative"  # newest first
        assert feedback[1]["rating"] == "positive"
        assert feedback[1]["comment"] == "유용한 문서"

    def test_recompute_quality_positive(
        self, registry: DocumentRegistry, sample_doc: DocMaster,
    ) -> None:
        """Mostly positive feedback should result in high quality score."""
        registry.insert_document(sample_doc, source_path="/test")
        for _ in range(4):
            registry.add_feedback("test001", "positive")
        registry.add_feedback("test001", "negative")

        doc = registry.get_document("test001")
        assert doc is not None
        assert doc["quality_score"] == 80.0  # 4/5 * 100
        assert doc["quality_grade"] == "A"
        assert doc["exclude_from_search"] == 0

    def test_recompute_quality_negative_excludes(
        self, registry: DocumentRegistry, sample_doc: DocMaster,
    ) -> None:
        """Mostly negative feedback (score<20, total>=3) should exclude from search."""
        registry.insert_document(sample_doc, source_path="/test")
        for _ in range(4):
            registry.add_feedback("test001", "negative")
        registry.add_feedback("test001", "positive")

        doc = registry.get_document("test001")
        assert doc is not None
        assert doc["quality_score"] == 20.0  # 1/5 * 100
        assert doc["quality_grade"] == "D"  # 20 < 30 → grade D
        assert doc["exclude_from_search"] == 0  # score=20, not < 20

        # Push it below 20 (1 positive out of 6 = ~16.67)
        registry.add_feedback("test001", "negative")
        doc = registry.get_document("test001")
        assert doc is not None
        assert doc["quality_score"] < 20.0
        assert doc["quality_grade"] == "D"
        assert doc["exclude_from_search"] == 1

    def test_recompute_quality_no_feedback_default(
        self, registry: DocumentRegistry, sample_doc: DocMaster,
    ) -> None:
        """No feedback should result in default score of 50."""
        registry.insert_document(sample_doc, source_path="/test")
        registry.recompute_quality("test001")

        doc = registry.get_document("test001")
        assert doc is not None
        assert doc["quality_score"] == 50.0
        assert doc["quality_grade"] == "B"
        assert doc["exclude_from_search"] == 0

    def test_retry_on_lock_succeeds_after_retry(self) -> None:
        """_retry_on_lock should retry on 'database is locked' errors."""
        call_count = 0

        def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise sqlite3.OperationalError("database is locked")
            return "ok"

        result = DocumentRegistry._retry_on_lock(flaky_fn, max_retries=3, base_wait=0.01)
        assert result == "ok"
        assert call_count == 3

    def test_retry_on_lock_raises_after_exhaustion(self) -> None:
        """_retry_on_lock should raise after max retries exhausted."""
        def always_locked():
            raise sqlite3.OperationalError("database is locked")

        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            DocumentRegistry._retry_on_lock(always_locked, max_retries=2, base_wait=0.01)

    def test_retry_on_lock_non_lock_error_raises_immediately(self) -> None:
        """Non-lock OperationalError should not be retried."""
        call_count = 0

        def bad_sql():
            nonlocal call_count
            call_count += 1
            raise sqlite3.OperationalError("no such table: foo")

        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            DocumentRegistry._retry_on_lock(bad_sql, max_retries=3, base_wait=0.01)
        assert call_count == 1

    # --- Phase 0/2C: embedded_at + extended stats ---

    def test_update_embedded_at(
        self, registry: DocumentRegistry, sample_doc: DocMaster,
    ) -> None:
        """embedded_at field can be updated via update_document."""
        registry.insert_document(sample_doc, source_path="/test")
        registry.update_document("test001", embedded_at="2024-01-01T00:00:00")
        doc = registry.get_document("test001")
        assert doc is not None
        assert doc["embedded_at"] == "2024-01-01T00:00:00"

    def test_update_classification_fields(
        self, registry: DocumentRegistry, sample_doc: DocMaster,
    ) -> None:
        """Classification-related fields can be updated."""
        registry.insert_document(sample_doc, source_path="/test")
        registry.update_document(
            "test001",
            doc_type="계약서",
            doc_type_ext="용역계약서",
            category="계약",
            classification_method="manual",
            classification_confidence=1.0,
            project_name="새프로젝트",
            year=2025,
        )
        doc = registry.get_document("test001")
        assert doc is not None
        assert doc["doc_type"] == "계약서"
        assert doc["doc_type_ext"] == "용역계약서"
        assert doc["category"] == "계약"
        assert doc["classification_method"] == "manual"
        assert doc["classification_confidence"] == 1.0
        assert doc["project_name"] == "새프로젝트"
        assert doc["year"] == 2025

    def test_get_stats_extended(
        self, registry: DocumentRegistry,
    ) -> None:
        """get_stats includes embedded_count, by_format, by_year, recent_24h."""
        doc1 = DocMaster(
            doc_id="ext001",
            file_name_original="test1.pdf",
            doc_type=DocType.OPINION,
            year=2024,
            process_status=ProcessStatus.COMPLETED,
        )
        doc2 = DocMaster(
            doc_id="ext002",
            file_name_original="test2.docx",
            doc_type=DocType.CONTRACT,
            year=2023,
            source_format=SourceFormat.DOCX,
            process_status=ProcessStatus.COMPLETED,
        )
        registry.insert_document(doc1, source_path="/p/1")
        registry.insert_document(doc2, source_path="/p/2")
        registry.update_document("ext001", embedded_at="2024-01-01T00:00:00")

        stats = registry.get_stats()
        assert stats["embedded_count"] == 1
        assert stats["not_embedded_count"] == 1
        assert "pdf" in stats["by_format"]
        assert "docx" in stats["by_format"]
        assert "2024" in stats["by_year"]
        assert "2023" in stats["by_year"]
        assert "recent_24h" in stats

    def test_list_unembedded(self, registry):
        """list_unembedded() returns only documents with no embedded_at."""
        doc1 = DocMaster(
            doc_id="unemb001",
            file_name_original="a.pdf",
            doc_type=DocType.OPINION,
            process_status=ProcessStatus.COMPLETED,
        )
        doc2 = DocMaster(
            doc_id="unemb002",
            file_name_original="b.pdf",
            doc_type=DocType.OPINION,
            process_status=ProcessStatus.COMPLETED,
        )
        registry.insert_document(doc1, source_path="/p/1")
        registry.insert_document(doc2, source_path="/p/2")
        registry.update_document("unemb001", embedded_at="2024-01-01T00:00:00")

        unembedded = registry.list_unembedded()
        ids = [d["doc_id"] for d in unembedded]
        assert "unemb002" in ids
        assert "unemb001" not in ids

    def test_list_unembedded_with_limit(self, registry):
        """list_unembedded(limit=1) returns at most 1 document."""
        for i in range(3):
            doc = DocMaster(
                doc_id=f"lim{i:03d}",
                file_name_original=f"{i}.pdf",
                doc_type=DocType.OPINION,
                process_status=ProcessStatus.COMPLETED,
            )
            registry.insert_document(doc, source_path=f"/p/{i}")

        result = registry.list_unembedded(limit=1)
        assert len(result) == 1

    def test_list_documents_limit_none(self, registry):
        """list_documents(limit=None) returns all matching docs."""
        for i in range(5):
            doc = DocMaster(
                doc_id=f"all{i:03d}",
                file_name_original=f"{i}.pdf",
                doc_type=DocType.OPINION,
                process_status=ProcessStatus.COMPLETED,
            )
            registry.insert_document(doc, source_path=f"/p/{i}")

        result = registry.list_documents(limit=None)
        assert len(result) >= 5

    # --- Phase 4: Unique helpers for QueryParser ---

    def test_get_unique_projects(self, registry: DocumentRegistry) -> None:
        """get_unique_projects returns distinct non-empty project names."""
        for i, proj in enumerate(["화성동탄", "판교신도시", "화성동탄", ""]):
            doc = DocMaster(
                doc_id=f"proj{i:03d}",
                file_name_original=f"proj{i}.pdf",
                doc_type=DocType.OPINION,
                project_name=proj,
            )
            registry.insert_document(doc, source_path=f"/p/{i}")

        projects = registry.get_unique_projects()
        assert "화성동탄" in projects
        assert "판교신도시" in projects
        assert "" not in projects
        assert len(projects) == 2

    def test_reset_embedded_at_all(self, registry: DocumentRegistry) -> None:
        """reset_embedded_at(doc_ids=None) resets all documents."""
        for i in range(3):
            doc = DocMaster(
                doc_id=f"rst{i:03d}",
                file_name_original=f"rst{i}.pdf",
                doc_type=DocType.OPINION,
                process_status=ProcessStatus.COMPLETED,
            )
            registry.insert_document(doc, source_path=f"/p/{i}")
            registry.update_document(f"rst{i:03d}", embedded_at="2024-01-01T00:00:00")

        # Verify all are embedded
        assert registry.list_unembedded() == []

        # Reset all
        count = registry.reset_embedded_at()
        assert count == 3

        # Verify all are now unembedded
        unembedded = registry.list_unembedded()
        assert len(unembedded) == 3

    def test_reset_embedded_at_selective(self, registry: DocumentRegistry) -> None:
        """reset_embedded_at(doc_ids=[...]) resets only specified documents."""
        for i in range(3):
            doc = DocMaster(
                doc_id=f"rsel{i:03d}",
                file_name_original=f"rsel{i}.pdf",
                doc_type=DocType.OPINION,
                process_status=ProcessStatus.COMPLETED,
            )
            registry.insert_document(doc, source_path=f"/p/{i}")
            registry.update_document(f"rsel{i:03d}", embedded_at="2024-01-01T00:00:00")

        # Reset only first two
        count = registry.reset_embedded_at(doc_ids=["rsel000", "rsel001"])
        assert count == 2

        # Verify selective reset
        unembedded = registry.list_unembedded()
        unembedded_ids = [d["doc_id"] for d in unembedded]
        assert "rsel000" in unembedded_ids
        assert "rsel001" in unembedded_ids
        assert "rsel002" not in unembedded_ids

    def test_get_unique_years(self, registry: DocumentRegistry) -> None:
        """get_unique_years returns distinct years > 0, sorted descending."""
        for i, yr in enumerate([2024, 2023, 2024, 0]):
            doc = DocMaster(
                doc_id=f"yr{i:03d}",
                file_name_original=f"yr{i}.pdf",
                doc_type=DocType.OPINION,
                year=yr,
            )
            registry.insert_document(doc, source_path=f"/p/{i}")

        years = registry.get_unique_years()
        assert years == [2024, 2023]
        assert 0 not in years
