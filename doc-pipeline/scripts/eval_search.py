"""Evaluate search quality using Hit@K, MRR, and nDCG metrics.

Runs each query from the eval set against the vector store (no LLM calls)
and measures retrieval accuracy.

Usage:
    python scripts/eval_search.py --baseline          # Save current state
    python scripts/eval_search.py --compare           # Compare to baseline
    python scripts/eval_search.py --queries evals/search_queries.jsonl
"""

from __future__ import annotations

import argparse
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


class EvalReport:
    """Aggregated evaluation metrics."""

    def __init__(self) -> None:
        self.hit_1: list[float] = []
        self.hit_3: list[float] = []
        self.hit_5: list[float] = []
        self.mrr: list[float] = []
        self.ndcg_5: list[float] = []
        self.details: list[dict] = []

    def add(
        self,
        query: str,
        retrieved_doc_ids: list[str],
        expected_doc_ids: list[str],
        category: str = "",
        tags: list[str] | None = None,
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
        })

    def _mean(self, values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def summary(self) -> dict:
        return {
            "total_queries": len(self.hit_1),
            "Hit@1": round(self._mean(self.hit_1), 4),
            "Hit@3": round(self._mean(self.hit_3), 4),
            "Hit@5": round(self._mean(self.hit_5), 4),
            "MRR": round(self._mean(self.mrr), 4),
            "nDCG@5": round(self._mean(self.ndcg_5), 4),
        }

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
    """
    from doc_pipeline.search import unified_search

    report = EvalReport()

    for q in queries:
        query_text = q["query"]
        expected = q.get("expected_doc_ids", [])
        category = q.get("category", "")
        tags = q.get("tags", [])

        # Skip curated queries with no expected doc_ids (can't score them)
        if not expected:
            continue

        # Get search results via unified pipeline
        retrieved_doc_ids: list[str] = []
        try:
            if get_embeddings_fn and client:
                embs = get_embeddings_fn(client, [query_text])
                query_emb = embs[0]
            else:
                query_emb = [0.0] * 768

            doc_results, _ = unified_search(
                store, query_text, query_emb,
                n_results=n_results,
                query_parser=query_parser,
            )

            retrieved_doc_ids = [d.doc_id for d in doc_results]
        except Exception:
            pass

        report.add(query_text, retrieved_doc_ids, expected, category, tags)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate search quality")
    parser.add_argument(
        "--queries", default="evals/search_queries.jsonl", help="Query JSONL file",
    )
    parser.add_argument("--db", default="data/registry.db", help="Registry DB path")
    parser.add_argument("--chroma-dir", default="data/chroma_db", help="ChromaDB dir")
    parser.add_argument("--baseline", action="store_true", help="Save as baseline")
    parser.add_argument("--compare", action="store_true", help="Compare to baseline")
    parser.add_argument("--output", default="evals/eval_report.json", help="Report path")
    parser.add_argument("--baseline-path", default="evals/baseline.json", help="Baseline path")
    args = parser.parse_args()

    from evals.generate_eval_set import read_jsonl
    from doc_pipeline.storage.vectordb import VectorStore

    queries = read_jsonl(args.queries)
    if not queries:
        print("No queries found. Run generate_eval_set.py first.")
        sys.exit(1)

    store = VectorStore(persist_dir=args.chroma_dir)

    # Try to get embeddings function
    embeddings_fn = None
    client = None
    try:
        from doc_pipeline.config import settings
        from doc_pipeline.processor.llm import create_client, get_embeddings
        client = create_client(settings.gemini.api_key)
        embeddings_fn = get_embeddings
    except Exception:
        print("Warning: No Gemini client available. Using local-only search.")

    # Build query parser for metadata-aware search
    qp = None
    try:
        from doc_pipeline.config.type_registry import get_type_registry
        from doc_pipeline.search.query_parser import QueryParser

        type_keywords = get_type_registry().get_keywords_map()
        qp = QueryParser(type_keywords=type_keywords)
    except Exception:
        pass

    report = evaluate(queries, store, get_embeddings_fn=embeddings_fn, client=client, query_parser=qp)
    summary = report.summary()

    # Print summary
    print("\n=== Search Evaluation Report ===")
    print(f"  Queries:  {summary['total_queries']}")
    print(f"  Hit@1:    {summary['Hit@1']:.1%}")
    print(f"  Hit@3:    {summary['Hit@3']:.1%}")
    print(f"  Hit@5:    {summary['Hit@5']:.1%}")
    print(f"  MRR:      {summary['MRR']:.4f}")
    print(f"  nDCG@5:   {summary['nDCG@5']:.4f}")

    # Save report
    report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"\nFull report → {report_path}")

    # Baseline handling
    baseline_path = Path(args.baseline_path)
    if args.baseline:
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Baseline saved → {baseline_path}")

    if args.compare and baseline_path.exists():
        with open(baseline_path, encoding="utf-8") as f:
            baseline = json.load(f)
        print("\n=== Comparison vs Baseline ===")
        for metric in ("Hit@1", "Hit@3", "Hit@5", "MRR", "nDCG@5"):
            old = baseline.get(metric, 0)
            new = summary.get(metric, 0)
            delta = new - old
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
            print(f"  {metric:8s}: {old:.4f} → {new:.4f}  ({arrow} {abs(delta):+.4f})")


if __name__ == "__main__":
    main()
