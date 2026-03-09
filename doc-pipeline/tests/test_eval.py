"""Tests for evaluation framework (evals/ and scripts/eval_search.py)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Import evaluation modules
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.generate_eval_set import generate_eval_set, read_jsonl, write_jsonl
from scripts.eval_search import (
    EvalReport,
    hit_at_k,
    ndcg_at_k,
    reciprocal_rank,
)


# ---------------------------------------------------------------------------
# Metric calculation tests
# ---------------------------------------------------------------------------


class TestHitAtK:
    def test_hit_at_1_found(self) -> None:
        assert hit_at_k(["a", "b", "c"], ["a"], 1) == 1.0

    def test_hit_at_1_not_found(self) -> None:
        assert hit_at_k(["b", "c", "d"], ["a"], 1) == 0.0

    def test_hit_at_3_found_at_position_3(self) -> None:
        assert hit_at_k(["b", "c", "a"], ["a"], 3) == 1.0

    def test_hit_at_5_found_at_position_4(self) -> None:
        assert hit_at_k(["x", "y", "z", "a", "b"], ["a"], 5) == 1.0

    def test_hit_at_k_empty_expected(self) -> None:
        assert hit_at_k(["a", "b"], [], 3) == 0.0

    def test_hit_at_k_empty_retrieved(self) -> None:
        assert hit_at_k([], ["a"], 3) == 0.0

    def test_hit_at_k_multiple_expected(self) -> None:
        assert hit_at_k(["x", "b", "c"], ["a", "b"], 3) == 1.0


class TestReciprocalRank:
    def test_first_position(self) -> None:
        assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0

    def test_second_position(self) -> None:
        assert reciprocal_rank(["b", "a", "c"], ["a"]) == 0.5

    def test_third_position(self) -> None:
        assert reciprocal_rank(["b", "c", "a"], ["a"]) == pytest.approx(1 / 3)

    def test_not_found(self) -> None:
        assert reciprocal_rank(["b", "c", "d"], ["a"]) == 0.0

    def test_empty_expected(self) -> None:
        assert reciprocal_rank(["a", "b"], []) == 0.0

    def test_multiple_expected_first_wins(self) -> None:
        # "b" at position 2 is found first
        assert reciprocal_rank(["x", "b", "a"], ["a", "b"]) == 0.5


class TestNdcgAtK:
    def test_perfect_single_result(self) -> None:
        assert ndcg_at_k(["a"], ["a"], 5) == 1.0

    def test_perfect_two_results(self) -> None:
        assert ndcg_at_k(["a", "b"], ["a", "b"], 5) == 1.0

    def test_no_relevant(self) -> None:
        assert ndcg_at_k(["x", "y", "z"], ["a"], 5) == 0.0

    def test_empty_expected(self) -> None:
        assert ndcg_at_k(["a"], [], 5) == 0.0

    def test_relevant_at_position_2(self) -> None:
        # Only "a" is relevant, at position 2
        # DCG = 0 + 1/log2(3) = 0.6309
        # IDCG = 1/log2(2) = 1.0
        # nDCG = 0.6309
        result = ndcg_at_k(["x", "a"], ["a"], 5)
        assert result == pytest.approx(0.6309, abs=0.001)


# ---------------------------------------------------------------------------
# EvalReport tests
# ---------------------------------------------------------------------------


class TestEvalReport:
    def test_empty_report(self) -> None:
        report = EvalReport()
        s = report.summary()
        assert s["total_queries"] == 0
        assert s["Hit@1"] == 0.0

    def test_single_perfect_query(self) -> None:
        report = EvalReport()
        report.add("test query", ["doc1", "doc2"], ["doc1"], "synthetic", ["tag1"])
        s = report.summary()
        assert s["total_queries"] == 1
        assert s["Hit@1"] == 1.0
        assert s["MRR"] == 1.0

    def test_mixed_results(self) -> None:
        report = EvalReport()
        report.add("q1", ["doc1"], ["doc1"])  # hit
        report.add("q2", ["doc2"], ["doc3"])  # miss
        s = report.summary()
        assert s["Hit@1"] == 0.5
        assert s["MRR"] == 0.5

    def test_to_dict_has_details(self) -> None:
        report = EvalReport()
        report.add("test", ["a"], ["a"])
        d = report.to_dict()
        assert "summary" in d
        assert "details" in d
        assert len(d["details"]) == 1
        assert d["details"][0]["query"] == "test"


# ---------------------------------------------------------------------------
# JSONL read/write tests
# ---------------------------------------------------------------------------


class TestJsonl:
    def test_write_and_read(self, tmp_path: Path) -> None:
        queries = [
            {"query": "test", "expected_doc_ids": ["d1"], "category": "synthetic", "tags": []},
        ]
        path = tmp_path / "test.jsonl"
        count = write_jsonl(queries, path)
        assert count == 1

        loaded = read_jsonl(path)
        assert len(loaded) == 1
        assert loaded[0]["query"] == "test"

    def test_read_skips_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"query": "ok", "expected_doc_ids": ["x"]}\nnot json\n', encoding="utf-8")
        loaded = read_jsonl(path)
        assert len(loaded) == 1

    def test_read_skips_missing_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "incomplete.jsonl"
        path.write_text('{"query": "ok", "expected_doc_ids": ["x"]}\n{"foo": "bar"}\n', encoding="utf-8")
        loaded = read_jsonl(path)
        assert len(loaded) == 1

    def test_read_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        loaded = read_jsonl(path)
        assert loaded == []


# ---------------------------------------------------------------------------
# Generate eval set tests
# ---------------------------------------------------------------------------


class TestGenerateEvalSet:
    def _make_mock_registry(self, docs: list[dict]) -> MagicMock:
        registry = MagicMock()
        registry.list_documents.return_value = docs
        return registry

    def test_generates_project_type_query(self) -> None:
        docs = [{
            "doc_id": "d1",
            "project_name": "테스트프로젝트",
            "doc_type_ext": "구조검토의견서",
            "doc_type": "의견서",
            "summary": "이것은 테스트 요약입니다.",
        }]
        registry = self._make_mock_registry(docs)
        queries = generate_eval_set(registry, include_curated=False)

        project_type = [q for q in queries if "project_type_match" in q.get("tags", [])]
        assert len(project_type) == 1
        assert "테스트프로젝트" in project_type[0]["query"]
        assert "구조검토의견서" in project_type[0]["query"]

    def test_generates_summary_query(self) -> None:
        docs = [{
            "doc_id": "d1",
            "project_name": "프로젝트A",
            "doc_type_ext": "의견서",
            "doc_type": "의견서",
            "summary": "슬래브 균열에 대한 구조 검토 의견서입니다.",
        }]
        registry = self._make_mock_registry(docs)
        queries = generate_eval_set(registry, include_curated=False)

        summary_q = [q for q in queries if "summary_match" in q.get("tags", [])]
        assert len(summary_q) == 1
        assert summary_q[0]["expected_doc_ids"] == ["d1"]

    def test_generates_project_only_query(self) -> None:
        docs = [
            {"doc_id": "d1", "project_name": "프로젝트A", "doc_type_ext": "", "doc_type": "의견서", "summary": "요약1"},
            {"doc_id": "d2", "project_name": "프로젝트A", "doc_type_ext": "", "doc_type": "계약서", "summary": "요약2"},
        ]
        registry = self._make_mock_registry(docs)
        queries = generate_eval_set(registry, include_curated=False)

        project_q = [q for q in queries if "project_match" in q.get("tags", [])]
        assert len(project_q) == 1
        assert set(project_q[0]["expected_doc_ids"]) == {"d1", "d2"}

    def test_skips_docs_without_project_and_summary(self) -> None:
        docs = [{"doc_id": "d1", "project_name": "", "doc_type_ext": "", "doc_type": "", "summary": ""}]
        registry = self._make_mock_registry(docs)
        queries = generate_eval_set(registry, include_curated=False)
        assert len(queries) == 0

    def test_includes_curated_by_default(self) -> None:
        registry = self._make_mock_registry([])
        queries = generate_eval_set(registry, include_curated=True)
        curated = [q for q in queries if q.get("category") == "curated"]
        assert len(curated) > 0

    def test_excludes_curated_when_requested(self) -> None:
        registry = self._make_mock_registry([])
        queries = generate_eval_set(registry, include_curated=False)
        assert len(queries) == 0

    def test_topic_project_query_generation(self) -> None:
        """Type 4: topic+project query when summary contains a domain topic."""
        docs = [{
            "doc_id": "d1",
            "project_name": "화성동탄",
            "doc_type_ext": "의견서",
            "doc_type": "의견서",
            "summary": "슬래브 균열에 대한 구조 검토 보고서입니다.",
            "year": 2024,
        }]
        registry = self._make_mock_registry(docs)
        queries = generate_eval_set(registry, include_curated=False)

        topic_proj = [q for q in queries if "topic_project_match" in q.get("tags", [])]
        assert len(topic_proj) >= 1
        # Should contain project name + a domain topic
        q = topic_proj[0]["query"]
        assert "화성동탄" in q

    def test_year_type_query_generation(self) -> None:
        """Type 5: year+type query."""
        docs = [{
            "doc_id": "d1",
            "project_name": "테스트",
            "doc_type_ext": "구조검토의견서",
            "doc_type": "의견서",
            "summary": "테스트 요약",
            "year": 2024,
        }]
        registry = self._make_mock_registry(docs)
        queries = generate_eval_set(registry, include_curated=False)

        year_type = [q for q in queries if "year_type_match" in q.get("tags", [])]
        assert len(year_type) >= 1
        assert "2024년" in year_type[0]["query"]
        assert "구조검토의견서" in year_type[0]["query"]

    def test_curated_jsonl_loading(self, tmp_path: Path) -> None:
        """Curated queries load from JSONL file when provided."""
        import json

        jsonl_path = tmp_path / "curated.jsonl"
        lines = [
            {"query": "테스트 쿼리", "expected_doc_ids": ["d99"], "category": "curated", "tags": ["test"]},
        ]
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        registry = self._make_mock_registry([])
        queries = generate_eval_set(registry, include_curated=True, curated_path=jsonl_path)
        curated = [q for q in queries if q.get("category") == "curated"]
        assert len(curated) == 1
        assert curated[0]["query"] == "테스트 쿼리"
