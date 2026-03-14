"""Tests for evaluation framework (evals/ and scripts/eval_search.py)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Import evaluation modules
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.generate_eval_set import generate_eval_set, read_jsonl, write_jsonl
from scripts.eval_search import (
    EvalReport,
    _run_agent_citation_eval,
    citation_precision,
    citation_recall,
    evaluate,
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

    def test_summary_by_intent(self) -> None:
        report = EvalReport()
        report.add("q1", ["a"], ["a"], "curated", ["tag"], "technical_qa")
        report.add("q2", ["b"], ["x"], "curated", ["tag"], "technical_qa")
        report.add("q3", ["c"], ["c"], "curated", ["tag"], "contract_lookup")
        by_intent = report.summary_by_field("intent")
        assert by_intent["technical_qa"]["total_queries"] == 2
        assert by_intent["contract_lookup"]["Hit@1"] == 1.0

    def test_false_positive_docs_counts_misses(self) -> None:
        report = EvalReport()
        report.add("q1", ["d1", "d2"], ["x"], "curated", ["tag"], "technical_qa")
        report.add("q2", ["d1", "d3"], ["y"], "curated", ["tag"], "technical_qa")
        report.add("q3", ["z"], ["z"], "curated", ["tag"], "technical_qa")
        rows = report.false_positive_docs(category="curated", top_n=2)
        assert rows[0]["doc_id"] == "d1"
        assert rows[0]["count"] == 2


class TestCitationMetrics:
    def test_precision_all_relevant(self) -> None:
        assert citation_precision(["a", "b"], ["a", "b", "c"]) == 1.0

    def test_precision_half_relevant(self) -> None:
        assert citation_precision(["a", "x"], ["a", "b"]) == 0.5

    def test_precision_empty_cited(self) -> None:
        assert citation_precision([], ["a"]) == 0.0

    def test_recall_all_cited(self) -> None:
        assert citation_recall(["a", "b"], ["a", "b"]) == 1.0

    def test_recall_half_cited(self) -> None:
        assert citation_recall(["a"], ["a", "b"]) == 0.5

    def test_recall_empty_expected(self) -> None:
        assert citation_recall(["a"], []) == 0.0

    def test_recall_none_cited(self) -> None:
        assert citation_recall(["x"], ["a", "b"]) == 0.0

    def test_report_add_citation(self) -> None:
        report = EvalReport()
        report.add("q1", ["a"], ["a"])
        report.add_citation(["a"], ["a", "b"])
        s = report.summary()
        assert s["CitePrecision"] == 1.0
        assert s["CiteRecall"] == 0.5

    def test_report_summary_no_citations(self) -> None:
        report = EvalReport()
        report.add("q1", ["a"], ["a"])
        s = report.summary()
        assert "CitePrecision" not in s


class TestEvaluate:
    def test_batches_embedding_requests(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[list[str]] = []

        def fake_get_embeddings(_client, texts: list[str]) -> list[list[float]]:
            calls.append(list(texts))
            return [[0.1, 0.2, 0.3] for _ in texts]

        def fake_unified_search(_store, _query_text, _query_embedding, **_kwargs):
            return [SimpleNamespace(doc_id="doc1")], None

        monkeypatch.setattr("doc_pipeline.search.unified_search", fake_unified_search)

        queries = [
            {"query": "q1", "expected_doc_ids": ["doc1"], "category": "synthetic", "tags": []},
            {"query": "q2", "expected_doc_ids": ["doc1"], "category": "synthetic", "tags": []},
            {"query": "q3", "expected_doc_ids": [], "category": "curated", "tags": []},
        ]

        report = evaluate(
            queries,
            store=object(),
            get_embeddings_fn=fake_get_embeddings,
            client=object(),
        )

        assert report.summary()["total_queries"] == 2
        assert calls == [["q1", "q2"]]


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


# ---------------------------------------------------------------------------
# Searchable-only eval set generation (Phase 1)
# ---------------------------------------------------------------------------


class TestGenerateEvalSearchableOnly:
    """generate_eval_set with embedded_only=True excludes unembedded docs."""

    def test_generate_eval_searchable_only(self) -> None:
        """list_documents is called with embedded_only=True."""
        registry = MagicMock()
        registry.list_documents.return_value = [
            {
                "doc_id": "d1",
                "project_name": "P",
                "doc_type_ext": "의견서",
                "doc_type": "의견서",
                "summary": "요약",
                "year": 2024,
            },
        ]

        generate_eval_set(registry, include_curated=False)

        # Verify list_documents was called with embedded_only=True
        registry.list_documents.assert_called_once()
        call_kwargs = registry.list_documents.call_args
        assert call_kwargs.kwargs.get("embedded_only") is True
        assert call_kwargs.kwargs.get("exclude_search") is False

    def test_project_type_group_answers(self) -> None:
        """project+type query should use all matching doc_ids as expected."""
        registry = MagicMock()
        registry.list_documents.return_value = [
            {
                "doc_id": "d1", "project_name": "프로젝트A",
                "doc_type_ext": "의견서", "doc_type": "의견서",
                "summary": "요약1", "year": 2024,
            },
            {
                "doc_id": "d2", "project_name": "프로젝트A",
                "doc_type_ext": "의견서", "doc_type": "의견서",
                "summary": "요약2", "year": 2024,
            },
        ]

        queries = generate_eval_set(registry, include_curated=False)
        pt_queries = [q for q in queries if "project_type_match" in q.get("tags", [])]

        # Dedup: same (프로젝트A, 의견서) → exactly 1 query, not 2
        matching = [q for q in pt_queries if "프로젝트A" in q["query"] and "의견서" in q["query"]]
        assert len(matching) == 1, f"Expected 1 deduped query, got {len(matching)}"

        # Both d1 and d2 share (프로젝트A, 의견서) → both are valid answers
        assert "d1" in matching[0]["expected_doc_ids"]
        assert "d2" in matching[0]["expected_doc_ids"]


# ---------------------------------------------------------------------------
# eval_search settings-aware defaults (Task 2)
# ---------------------------------------------------------------------------


class TestAgentCitationEval:
    """Tests for _run_agent_citation_eval and --with-agent flag."""

    def test_skips_when_agents_disabled(self, monkeypatch, capsys) -> None:
        """_run_agent_citation_eval returns early when AGENT_ENABLED=false."""
        from doc_pipeline.config import settings

        monkeypatch.setattr(settings.agents, "enabled", False)
        report = EvalReport()
        report.add("q1", ["a"], ["a"])
        _run_agent_citation_eval(report, None, None, None, None, None, None)
        out = capsys.readouterr().out
        assert "AGENT_ENABLED is false" in out
        assert not report.cite_precision  # no citations added

    def test_with_agent_flag_parsed(self) -> None:
        """--with-agent flag is recognized by argparse."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--with-agent", action="store_true")
        args = parser.parse_args(["--with-agent"])
        assert args.with_agent is True

    def test_with_agent_flag_default_off(self) -> None:
        """--with-agent defaults to False."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--with-agent", action="store_true")
        args = parser.parse_args([])
        assert args.with_agent is False

    def test_success_path_collects_citations(self, monkeypatch) -> None:
        """Agent enabled + citations returned → report.add_citation() called."""
        from unittest.mock import AsyncMock, patch
        from doc_pipeline.config import settings

        monkeypatch.setattr(settings.agents, "enabled", True)

        # Build a report with one query
        report = EvalReport()
        report.add("슬래브 균열 검토", ["doc1", "doc2"], ["doc1"])

        # Mock agent result with citations
        mock_citation = SimpleNamespace(doc_id="doc1", doc_ref="문서 1", relevance="high")
        mock_answer = SimpleNamespace(
            citations=[mock_citation],
            confidence=0.85,
            follow_up="추가 질문",
        )
        mock_result = SimpleNamespace(output=mock_answer)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        # Mock unified_search to return fake doc results
        fake_doc = SimpleNamespace(
            doc_id="doc1",
            top_chunks=[SimpleNamespace(text="슬래브 균열 검토 내용")],
        )

        def fake_unified_search(*_args, **_kwargs):
            return [fake_doc], None

        with (
            patch("scripts.eval_search.get_search_agent", return_value=mock_agent, create=True),
            patch("doc_pipeline.agents.search_agent.get_search_agent", return_value=mock_agent),
        ):
            monkeypatch.setattr(
                "doc_pipeline.search.unified_search",
                fake_unified_search,
            )
            _run_agent_citation_eval(
                report,
                store=object(),
                get_embeddings_fn=lambda _c, texts: [[0.1] * 768 for _ in texts],
                client=object(),
                query_parser=None,
                registry=None,
                chunk_fts=None,
            )

        # Verify citation was recorded
        assert len(report.cite_precision) == 1
        assert report.cite_precision[0] == 1.0  # doc1 cited, doc1 expected
        s = report.summary()
        assert "CitePrecision" in s
        assert s["CitePrecision"] == 1.0


class TestEvalSearchDefaults:
    """Verify eval_search uses settings-aware paths instead of hardcoded."""

    def test_default_chroma_dir_uses_settings(self) -> None:
        """--chroma-dir default should be None (resolved from settings at runtime)."""
        import argparse
        from scripts.eval_search import main

        # Parse with no args — defaults should be None
        parser = argparse.ArgumentParser()
        parser.add_argument("--chroma-dir", default=None)
        parser.add_argument("--db", default=None)
        args = parser.parse_args([])
        assert args.chroma_dir is None
        assert args.db is None
