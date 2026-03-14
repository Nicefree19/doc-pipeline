"""Evaluate search quality using Hit@K, MRR, and nDCG metrics.

Runs each query from the eval set against the vector store using the
``unified_search()`` pipeline (RRF + aggregation + metadata filtering).
No LLM calls are made — only embedding + ChromaDB search.

Paths default to settings (``settings.chroma.persist_dir``, ``settings.registry.db_path``)
unless overridden via CLI flags.

Usage:
    python scripts/eval_search.py --baseline          # Save current state
    python scripts/eval_search.py --compare           # Compare to baseline
    python scripts/eval_search.py --queries evals/search_queries.jsonl
    python scripts/eval_search.py --chroma-dir path/  # Override ChromaDB path
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import sys
from pathlib import Path

# Ensure src is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))


def hit_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str], k: int) -> float:
    """1.0 if any expected doc_id appears in top-k results, else 0.0."""
    if not expected_doc_ids:
        return 0.0
    expected = set(expected_doc_ids)
    for doc_id in retrieved_doc_ids[:k]:
        if doc_id in expected:
            return 1.0
    return 0.0


def reciprocal_rank(retrieved_doc_ids: list[str], expected_doc_ids: list[str]) -> float:
    """Reciprocal of the rank of the first relevant result."""
    if not expected_doc_ids:
        return 0.0
    expected = set(expected_doc_ids)
    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in expected:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Binary relevance: 1 if doc_id in expected, 0 otherwise.
    """
    if not expected_doc_ids:
        return 0.0

    expected = set(expected_doc_ids)

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_doc_ids[:k]):
        rel = 1.0 if doc_id in expected else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0

    # Ideal DCG: all relevant docs at top positions
    n_relevant = min(len(expected), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))

    return dcg / idcg if idcg > 0 else 0.0


def citation_precision(cited_ids: list[str], expected_ids: list[str]) -> float:
    """Fraction of cited docs that are actually relevant."""
    if not cited_ids:
        return 0.0
    expected = set(expected_ids)
    return sum(1 for d in cited_ids if d in expected) / len(cited_ids)


def citation_recall(cited_ids: list[str], expected_ids: list[str]) -> float:
    """Fraction of relevant docs that were cited."""
    if not expected_ids:
        return 0.0
    cited = set(cited_ids)
    return sum(1 for d in expected_ids if d in cited) / len(expected_ids)


class EvalReport:
    """Aggregated evaluation metrics."""

    def __init__(self) -> None:
        self.hit_1: list[float] = []
        self.hit_3: list[float] = []
        self.hit_5: list[float] = []
        self.mrr: list[float] = []
        self.ndcg_5: list[float] = []
        self.cite_precision: list[float] = []
        self.cite_recall: list[float] = []
        self.details: list[dict] = []

    def add(
        self,
        query: str,
        retrieved_doc_ids: list[str],
        expected_doc_ids: list[str],
        category: str = "",
        tags: list[str] | None = None,
        intent: str = "",
    ) -> None:
        h1 = hit_at_k(retrieved_doc_ids, expected_doc_ids, 1)
        h3 = hit_at_k(retrieved_doc_ids, expected_doc_ids, 3)
        h5 = hit_at_k(retrieved_doc_ids, expected_doc_ids, 5)
        rr = reciprocal_rank(retrieved_doc_ids, expected_doc_ids)
        ndcg = ndcg_at_k(retrieved_doc_ids, expected_doc_ids, 5)

        self.hit_1.append(h1)
        self.hit_3.append(h3)
        self.hit_5.append(h5)
        self.mrr.append(rr)
        self.ndcg_5.append(ndcg)

        self.details.append({
            "query": query,
            "expected": expected_doc_ids,
            "retrieved": retrieved_doc_ids[:5],
            "hit_1": h1,
            "hit_5": h5,
            "mrr": rr,
            "ndcg_5": ndcg,
            "category": category,
            "tags": tags or [],
            "intent": intent,
        })

    def add_citation(self, cited_ids: list[str], expected_ids: list[str]) -> None:
        """Record citation precision/recall for a single query."""
        self.cite_precision.append(citation_precision(cited_ids, expected_ids))
        self.cite_recall.append(citation_recall(cited_ids, expected_ids))

    def _mean(self, values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def summary(self) -> dict:
        base = {
            "total_queries": len(self.hit_1),
            "Hit@1": round(self._mean(self.hit_1), 4),
            "Hit@3": round(self._mean(self.hit_3), 4),
            "Hit@5": round(self._mean(self.hit_5), 4),
            "MRR": round(self._mean(self.mrr), 4),
            "nDCG@5": round(self._mean(self.ndcg_5), 4),
        }
        if self.cite_precision:
            base["CitePrecision"] = round(self._mean(self.cite_precision), 4)
            base["CiteRecall"] = round(self._mean(self.cite_recall), 4)
        return base

    def summary_by_category(self) -> dict[str, dict]:
        """Return separate metric summaries grouped by category."""
        return self.summary_by_field("category")

    def summary_by_field(self, field_name: str) -> dict[str, dict]:
        """Return separate metric summaries grouped by a detail field."""
        grouped: dict[str, EvalReport] = {}
        for d in self.details:
            key = d.get(field_name) or "unknown"
            if key not in grouped:
                grouped[key] = EvalReport()
            grouped[key].add(
                d["query"],
                d.get("retrieved", []),
                d.get("expected", []),
                d.get("category", ""),
                d.get("tags", []),
                d.get("intent", ""),
            )
        return {key: sub.summary() for key, sub in grouped.items()}

    def filtered(
        self,
        *,
        category: str | None = None,
        intent: str | None = None,
    ) -> EvalReport:
        """Return a new report filtered by category and/or intent."""
        filtered = EvalReport()
        for d in self.details:
            if category is not None and d.get("category") != category:
                continue
            if intent is not None and d.get("intent") != intent:
                continue
            filtered.add(
                d["query"],
                d.get("retrieved", []),
                d.get("expected", []),
                d.get("category", ""),
                d.get("tags", []),
                d.get("intent", ""),
            )
        return filtered

    def false_positive_docs(
        self,
        *,
        category: str | None = None,
        top_n: int = 10,
    ) -> list[dict[str, int]]:
        """Return the most frequent top-k false-positive doc IDs for misses."""
        counts: Counter[str] = Counter()
        for d in self.details:
            if category is not None and d.get("category") != category:
                continue
            if d.get("hit_5"):
                continue
            for doc_id in d.get("retrieved", [])[:5]:
                counts[doc_id] += 1
        return [
            {"doc_id": doc_id, "count": count}
            for doc_id, count in counts.most_common(top_n)
        ]

    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "details": self.details,
        }


def evaluate(
    queries: list[dict],
    store,
    *,
    get_embeddings_fn=None,
    client=None,
    n_results: int = 10,
    query_parser=None,
    registry=None,
    chunk_fts=None,
) -> EvalReport:
    """Run evaluation queries against the vector store.

    Uses ``unified_search()`` to go through the same improved pipeline
    (RRF + aggregation + metadata filtering) as the API and CLI.

    Args:
        queries: List of eval query dicts.
        store: VectorStore instance.
        get_embeddings_fn: Callable to get embeddings (for C-grade search).
        client: Gemini client for embeddings.
        n_results: Number of results to retrieve per query.
        query_parser: Optional QueryParser for metadata-aware search.
        registry: Optional DocumentRegistry for doc-level FTS bonus.
        chunk_fts: Optional ChunkFTS for hybrid search.
    """
    from doc_pipeline.search import unified_search

    report = EvalReport()
    query_embeddings: dict[str, list[float]] = {}

    if get_embeddings_fn and client:
        judged_texts = [q["query"] for q in queries if q.get("expected_doc_ids")]
        embedded: list[list[float]] = []
        try:
            for i in range(0, len(judged_texts), 100):
                embedded.extend(get_embeddings_fn(client, judged_texts[i:i + 100]))
            query_embeddings = dict(zip(judged_texts, embedded, strict=False))
        except Exception:
            query_embeddings = {}

    for q in queries:
        query_text = q["query"]
        expected = q.get("expected_doc_ids", [])
        category = q.get("category", "")
        tags = q.get("tags", [])
        intent = q.get("intent", "")

        # Skip curated queries with no expected doc_ids (can't score them)
        if not expected:
            continue

        # Get search results via unified pipeline
        retrieved_doc_ids: list[str] = []
        try:
            if query_text in query_embeddings:
                query_emb = query_embeddings[query_text]
            elif get_embeddings_fn and client:
                embs = get_embeddings_fn(client, [query_text])
                query_emb = embs[0]
            else:
                query_emb = [0.0] * 768

            doc_results, _ = unified_search(
                store, query_text, query_emb,
                n_results=n_results,
                query_parser=query_parser,
                registry=registry,
                chunk_fts=chunk_fts,
            )

            retrieved_doc_ids = [d.doc_id for d in doc_results]
        except Exception:
            pass

        report.add(query_text, retrieved_doc_ids, expected, category, tags, intent)

    return report


def summarize_query_pool(queries: list[dict], *, field_name: str) -> dict[str, int]:
    """Count queries by a field name for ungrounded/judged reporting."""
    counts: Counter[str] = Counter()
    for q in queries:
        key = q.get(field_name) or "unknown"
        counts[key] += 1
    return dict(sorted(counts.items()))


def enrich_false_positive_docs(
    false_positives: list[dict[str, int]],
    registry=None,
) -> list[dict]:
    """Attach doc metadata to false-positive frequency rows."""
    if not false_positives:
        return []
    if not registry:
        return false_positives

    try:
        doc_map = registry.get_documents_batch([row["doc_id"] for row in false_positives])
    except Exception:
        return false_positives

    enriched: list[dict] = []
    for row in false_positives:
        doc = doc_map.get(row["doc_id"], {})
        enriched.append({
            **row,
            "doc_type": doc.get("doc_type", ""),
            "doc_type_ext": doc.get("doc_type_ext", ""),
            "project_name": doc.get("project_name", ""),
            "year": doc.get("year", 0),
            "file_name_original": doc.get("file_name_original", ""),
        })
    return enriched


def _run_agent_citation_eval(
    report: EvalReport,
    store,
    get_embeddings_fn,
    client,
    query_parser,
    registry,
    chunk_fts,
) -> None:
    """Run the search agent on each query in report.details and collect citation metrics.

    Mutates ``report`` in-place by calling ``report.add_citation()`` for each query.
    Requires pydantic-ai and AGENT_ENABLED=true.
    """
    try:
        from doc_pipeline.agents.deps import SearchDeps
        from doc_pipeline.agents.search_agent import get_search_agent
        from doc_pipeline.search import unified_search
    except ImportError as exc:
        print(f"--with-agent: pydantic-ai not installed ({exc}). Skipping citation eval.")
        return

    from doc_pipeline.config import settings

    if not settings.agents.enabled:
        print("--with-agent: AGENT_ENABLED is false. Set AGENT_ENABLED=true to run agent eval.")
        return

    agent = get_search_agent()
    total = len(report.details)
    success = 0
    errors = 0

    print(f"\n--- Agent Citation Eval ({total} queries) ---")

    for i, detail in enumerate(report.details):
        query_text = detail["query"]
        expected = detail.get("expected", [])

        try:
            # Recompute search context (same as API path)
            if get_embeddings_fn and client:
                query_emb = get_embeddings_fn(client, [query_text])[0]
            else:
                query_emb = [0.0] * 768

            doc_results, _ = unified_search(
                store, query_text, query_emb,
                n_results=10,
                query_parser=query_parser,
                registry=registry,
                chunk_fts=chunk_fts,
            )

            if not doc_results:
                report.add_citation([], expected)
                continue

            # Build RAG context (lightweight inline version)
            references: list[dict] = []
            context_parts: list[str] = []
            for j, doc_res in enumerate(doc_results[:5], 1):
                chunk_text = doc_res.top_chunks[0].text[:500] if doc_res.top_chunks else ""
                context_parts.append(f"[문서 {j}] (doc_id={doc_res.doc_id}) {chunk_text}")
                references.append({"doc_id": doc_res.doc_id, "doc_type": ""})
            rag_prompt = f"질문: {query_text}\n\n검색 결과:\n" + "\n\n".join(context_parts)

            deps = SearchDeps(
                query=query_text,
                rag_prompt=rag_prompt,
                references=references,
            )

            import asyncio
            result = asyncio.get_event_loop().run_until_complete(agent.run(query_text, deps=deps))
            answer_obj = result.output

            # Extract cited doc_ids from agent citations
            cited_ids = [c.doc_id for c in answer_obj.citations if c.doc_id]
            report.add_citation(cited_ids, expected)
            success += 1

            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{total}] processed...")

        except Exception as exc:
            errors += 1
            report.add_citation([], expected)
            if errors <= 3:
                print(f"  Warning: query {i + 1} failed: {exc}")

    print(f"  Done: {success} ok, {errors} errors out of {total}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate search quality")
    parser.add_argument(
        "--queries", default=None, help="Query JSONL file (default: evals/search_queries.jsonl)",
    )
    parser.add_argument("--db", default=None, help="Registry DB path (default: from settings)")
    parser.add_argument("--chroma-dir", default=None, help="ChromaDB dir (default: from settings)")
    parser.add_argument("--baseline", action="store_true", help="Save as baseline")
    parser.add_argument("--compare", action="store_true", help="Compare to baseline")
    parser.add_argument("--output", default=None, help="Report path (default: evals/eval_report.json)")
    parser.add_argument("--baseline-path", default=None, help="Baseline path (default: evals/baseline.json)")
    parser.add_argument(
        "--with-agent", action="store_true",
        help="Run search agent for each query to collect citation metrics (requires AGENT_ENABLED, makes LLM calls)",
    )
    args = parser.parse_args()

    from doc_pipeline.config import settings
    from evals.generate_eval_set import read_jsonl
    from doc_pipeline.storage.vectordb import VectorStore

    # Resolve paths: explicit > settings > _ROOT-relative defaults
    chroma_dir = args.chroma_dir or settings.chroma.persist_dir
    db_path = args.db or settings.registry.db_path
    queries_path = args.queries or str(_ROOT / "evals" / "search_queries.jsonl")
    output_path = args.output or str(_ROOT / "evals" / "eval_report.json")
    baseline_path = args.baseline_path or str(_ROOT / "evals" / "baseline.json")

    queries = read_jsonl(queries_path)
    if not queries:
        print("No queries found. Run generate_eval_set.py first.")
        sys.exit(1)

    # Load relevance judgments if available
    judgments_path = _ROOT / "evals" / "relevance_judgments.jsonl"
    if judgments_path.exists():
        judgments = []
        with open(judgments_path, encoding="utf-8") as jf:
            for line in jf:
                line = line.strip()
                if line:
                    try:
                        judgments.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        # Build lookup: query_id -> list of doc_ids with grade >= 1
        judgment_map: dict[int, list[str]] = {}
        for j in judgments:
            qid = j.get("query_id")
            grade = j.get("relevance_grade", 0)
            if qid is not None and grade and grade >= 1:
                judgment_map.setdefault(qid, []).append(j["doc_id"])

        # Merge into curated queries that have empty expected_doc_ids
        curated_idx = 0
        for q in queries:
            if q.get("category") == "curated":
                if not q.get("expected_doc_ids"):
                    qid = q.get("query_id", curated_idx)
                    if qid in judgment_map:
                        q["expected_doc_ids"] = judgment_map[qid]
                curated_idx += 1

    # Preflight validation (paths are now absolute via settings)
    print(f"\n--- Preflight ---")
    print(f"  Registry:  {db_path} (exists: {Path(db_path).exists()})")
    print(f"  ChromaDB:  {chroma_dir} (exists: {Path(chroma_dir).exists()})")
    fts_path = settings.fts.db_path
    print(f"  FTS:       {fts_path} (exists: {Path(fts_path).exists()})")

    store = VectorStore(persist_dir=chroma_dir)

    if store.count == 0:
        print("FATAL: ChromaDB empty. Aborting.")
        sys.exit(1)

    # Try to get embeddings function
    embeddings_fn = None
    client = None
    try:
        from doc_pipeline.processor.llm import create_client, get_embeddings
        client = create_client(settings.gemini.api_key)
        embeddings_fn = get_embeddings
    except Exception:
        print("Warning: No Gemini client available. Using local-only search.")

    # Build registry for metadata-aware search and FTS doc bonus
    registry = None
    try:
        from doc_pipeline.storage.registry import DocumentRegistry

        registry = DocumentRegistry(db_path=db_path)
    except Exception:
        print("Warning: Registry not available. Metadata features disabled.")

    # Build ChunkFTS for hybrid search
    chunk_fts = None
    try:
        from doc_pipeline.storage.vectordb import ChunkFTS

        chunk_fts = ChunkFTS(db_path=settings.fts.db_path)
        if chunk_fts.count > 0:
            print(f"FTS5 enabled: {chunk_fts.count} entries")
        else:
            print("WARNING: FTS5 empty. Hybrid search disabled.")
            chunk_fts = None
    except Exception:
        pass

    # Build query parser for metadata-aware search
    qp = None
    try:
        from doc_pipeline.config.type_registry import get_type_registry
        from doc_pipeline.search.query_parser import QueryParser

        tr = get_type_registry()
        type_keywords = tr.get_keywords_map()
        # Build type→category reverse map (same as main.py)
        type_category_map: dict[str, str] = {}
        for cat, type_names in tr.categories.items():
            for tn in type_names:
                type_category_map[tn] = cat

        # Load known projects/years from registry for accurate parsing
        known_projects: set[str] = set()
        if registry:
            try:
                known_projects = set(registry.get_unique_projects())
            except Exception:
                pass

        qp = QueryParser(
            type_keywords=type_keywords,
            type_category_map=type_category_map,
            known_projects=known_projects,
        )
    except Exception:
        pass

    judged_queries = [q for q in queries if q.get("expected_doc_ids")]
    ungrounded_queries = [q for q in queries if not q.get("expected_doc_ids")]

    report = evaluate(
        judged_queries, store,
        get_embeddings_fn=embeddings_fn, client=client,
        query_parser=qp, registry=registry, chunk_fts=chunk_fts,
    )

    # --with-agent: run search agent for citation quality metrics
    if args.with_agent:
        _run_agent_citation_eval(report, store, embeddings_fn, client, qp, registry, chunk_fts)

    summary = report.summary()
    judged_count = len(judged_queries)
    skipped_count = len(ungrounded_queries)
    synthetic_summary = report.filtered(category="synthetic").summary()
    curated_summary = report.filtered(category="curated").summary()
    by_category = report.summary_by_category()
    by_intent = report.summary_by_field("intent")
    curated_by_intent = report.filtered(category="curated").summary_by_field("intent")
    false_positive_docs = enrich_false_positive_docs(
        report.false_positive_docs(category="curated", top_n=10),
        registry=registry,
    )
    ungrounded_summary = {
        "total_queries": skipped_count,
        "by_category": summarize_query_pool(ungrounded_queries, field_name="category"),
        "by_intent": summarize_query_pool(ungrounded_queries, field_name="intent"),
    }

    # Warn if all curated queries lack ground truth
    curated_queries = [q for q in queries if q.get("category") == "curated"]
    curated_with_gt = [q for q in curated_queries if q.get("expected_doc_ids")]
    if curated_queries and not curated_with_gt:
        print("\nWARNING: All curated queries have empty expected_doc_ids.")
        print("  → Run annotate_eval.py and fill relevance_judgments.jsonl first.")

    # Print summary
    print("\n=== Search Evaluation Report ===")
    print(f"  Input:    {len(queries)}")
    print(f"  Queries:  {summary['total_queries']}")
    print(f"  Judged:   {judged_count}  (with ground truth)")
    print(f"  Skipped:  {skipped_count}  (ungrounded / no expected_doc_ids)")
    print(f"  Hit@1:    {summary['Hit@1']:.1%}")
    print(f"  Hit@3:    {summary['Hit@3']:.1%}")
    print(f"  Hit@5:    {summary['Hit@5']:.1%}")
    print(f"  MRR:      {summary['MRR']:.4f}")
    print(f"  nDCG@5:   {summary['nDCG@5']:.4f}")
    if "CitePrecision" in summary:
        print(f"\n--- Citation Quality (--with-agent) ---")
        print(f"  CitePrecision: {summary['CitePrecision']:.1%}")
        print(f"  CiteRecall:    {summary['CiteRecall']:.1%}")
    if curated_summary["total_queries"] > 0:
        print("\n--- Curated ---")
        print(f"  n={curated_summary['total_queries']}  "
              f"Hit@1={curated_summary['Hit@1']:.1%}  "
              f"Hit@5={curated_summary['Hit@5']:.1%}  "
              f"MRR={curated_summary['MRR']:.4f}")
    if skipped_count:
        print("\n--- Ungrounded ---")
        print(f"  Total: {skipped_count}")
        print(f"  By intent: {ungrounded_summary['by_intent']}")

    # Category breakdown
    if len(by_category) > 1:
        print("\n--- By Category ---")
        for cat, metrics in sorted(by_category.items()):
            print(f"  [{cat}] n={metrics['total_queries']}  "
                  f"Hit@1={metrics['Hit@1']:.1%}  Hit@5={metrics['Hit@5']:.1%}  "
                  f"MRR={metrics['MRR']:.4f}")
    if by_intent:
        print("\n--- By Intent ---")
        for intent, metrics in sorted(by_intent.items()):
            print(f"  [{intent}] n={metrics['total_queries']}  "
                  f"Hit@1={metrics['Hit@1']:.1%}  Hit@5={metrics['Hit@5']:.1%}  "
                  f"MRR={metrics['MRR']:.4f}")
    if false_positive_docs:
        print("\n--- Top Curated False Positives ---")
        for row in false_positive_docs[:5]:
            print(f"  {row['doc_id']} x{row['count']}  {row.get('doc_type','')}  "
                  f"{row.get('project_name','')[:40]}")

    # Save report
    report_payload = {
        "summary": summary,
        "sections": {
            "overall": {
                **summary,
                "input_queries": len(queries),
                "judged_queries": judged_count,
                "ungrounded_queries": skipped_count,
            },
            "synthetic": synthetic_summary,
            "curated": curated_summary,
            "judged_only": summary,
            "ungrounded": ungrounded_summary,
        },
        "by_category": by_category,
        "by_intent": by_intent,
        "curated_by_intent": curated_by_intent,
        "false_positives": {
            "curated_top_docs": false_positive_docs,
        },
        "details": report.details,
    }
    report_out = Path(output_path)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    print(f"\nFull report → {report_out}")

    # Baseline handling
    baseline_out = Path(baseline_path)
    if args.baseline:
        baseline_data = {
            **summary,
            "judged_query_count": judged_count,
            "skipped_query_count": skipped_count,
            "by_category": by_category,
            "by_intent": by_intent,
            "sections": report_payload["sections"],
            "false_positives": report_payload["false_positives"],
        }
        with open(baseline_out, "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, ensure_ascii=False, indent=2)
        print(f"Baseline saved → {baseline_out}")

    if args.compare and baseline_out.exists():
        with open(baseline_out, encoding="utf-8") as f:
            baseline = json.load(f)
        print("\n=== Comparison vs Baseline ===")
        metrics_to_compare = ["Hit@1", "Hit@3", "Hit@5", "MRR", "nDCG@5"]
        if "CitePrecision" in summary:
            metrics_to_compare.extend(["CitePrecision", "CiteRecall"])
        for metric in metrics_to_compare:
            old = baseline.get(metric, 0)
            new = summary.get(metric, 0)
            delta = new - old
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
            print(f"  {metric:8s}: {old:.4f} → {new:.4f}  ({arrow} {abs(delta):+.4f})")


if __name__ == "__main__":
    main()
