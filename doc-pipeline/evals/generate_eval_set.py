"""Generate evaluation query set from document registry.

Produces synthetic queries from existing document metadata and
optionally merges manually curated queries from a JSONL file.

Usage:
    python -m evals.generate_eval_set --db data/registry.db --output evals/search_queries.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Curated domain-expert queries — extend as needed
CURATED_QUERIES: list[dict] = [
    {
        "query": "슬래브 균열 보강 방법",
        "expected_doc_ids": [],
        "category": "curated",
        "tags": ["topic_match"],
    },
    {
        "query": "기둥 내진 보강 설계",
        "expected_doc_ids": [],
        "category": "curated",
        "tags": ["topic_match"],
    },
    {
        "query": "구조 안전성 검토 의견서",
        "expected_doc_ids": [],
        "category": "curated",
        "tags": ["type_match"],
    },
    {
        "query": "철골 접합부 상세",
        "expected_doc_ids": [],
        "category": "curated",
        "tags": ["topic_match"],
    },
    {
        "query": "기초 구조 검토",
        "expected_doc_ids": [],
        "category": "curated",
        "tags": ["topic_match"],
    },
    {
        "query": "내진 성능 평가 보고서",
        "expected_doc_ids": [],
        "category": "curated",
        "tags": ["type_match"],
    },
    {
        "query": "콘크리트 강도 시험 결과",
        "expected_doc_ids": [],
        "category": "curated",
        "tags": ["topic_match"],
    },
    {
        "query": "구조설계 변경 사유",
        "expected_doc_ids": [],
        "category": "curated",
        "tags": ["topic_match"],
    },
    {
        "query": "심의 지적사항 조치계획",
        "expected_doc_ids": [],
        "category": "curated",
        "tags": ["type_match"],
    },
    {
        "query": "프리스트레스 긴장력 계산",
        "expected_doc_ids": [],
        "category": "curated",
        "tags": ["topic_match"],
    },
    {
        "query": "보 단면 검토",
        "expected_doc_ids": [],
        "category": "curated",
        "tags": ["topic_match"],
    },
    {
        "query": "화성동탄 아파트 구조 검토",
        "expected_doc_ids": [],
        "category": "curated",
        "tags": ["project_match"],
    },
    {
        "query": "용역 계약서 계약금액",
        "expected_doc_ids": [],
        "category": "curated",
        "tags": ["type_match"],
    },
    {
        "query": "지하주차장 슬래브 처짐",
        "expected_doc_ids": [],
        "category": "curated",
        "tags": ["topic_match"],
    },
    {
        "query": "탄소섬유 보강 시공",
        "expected_doc_ids": [],
        "category": "curated",
        "tags": ["topic_match"],
    },
]


def _load_curated_jsonl(path: str | Path | None = None) -> list[dict]:
    """Load curated queries from external JSONL file.

    Falls back to the embedded ``CURATED_QUERIES`` if file not found.
    """
    if path is None:
        path = Path(__file__).parent / "curated_queries.jsonl"
    p = Path(path)
    if not p.exists():
        return list(CURATED_QUERIES)
    try:
        return read_jsonl(p)
    except Exception:
        logger.warning("Failed to load curated JSONL from %s, using defaults", p)
        return list(CURATED_QUERIES)


# Domain topics for synthetic query generation (type 4)
_DOMAIN_TOPICS = [
    "균열", "보강", "내진", "처짐", "슬래브", "기둥", "기초",
    "전단", "철골", "콘크리트", "앵커", "용접", "합성보",
]


def generate_eval_set(
    registry,
    *,
    max_docs: int = 200,
    include_curated: bool = True,
    curated_path: str | Path | None = None,
) -> list[dict]:
    """Generate evaluation queries from registry documents.

    For each document with a ``summary`` and ``project_name``, up to six
    synthetic queries are created:

    1. **project+type**: ``"{project_name} {doc_type_ext}"``
    2. **summary keyword**: first sentence of the summary
    3. **project only**: ``"{project_name}"`` — all docs sharing that project
    4. **topic+project**: ``"{project_name} {topic}"`` — project + domain topic
    5. **year+type**: ``"{year}년 {doc_type_ext}"`` — year + type matching
    6. **keyword extract**: 2-3 key nouns from summary

    Args:
        registry: A ``DocumentRegistry`` instance.
        max_docs: Cap on documents to sample (avoids huge eval sets).
        include_curated: Whether to append curated queries.
        curated_path: Path to curated JSONL (default: evals/curated_queries.jsonl).

    Returns:
        List of query dicts ready for JSONL serialisation.
    """
    docs = registry.list_documents(limit=max_docs, order_by="ingested_at DESC")

    # Index project→doc_ids for project-only queries
    project_doc_ids: dict[str, list[str]] = {}
    for d in docs:
        pname = d.get("project_name", "")
        if pname:
            project_doc_ids.setdefault(pname, []).append(d["doc_id"])

    # Index year+type→doc_ids for year+type queries
    year_type_doc_ids: dict[tuple[int, str], list[str]] = {}
    for d in docs:
        y = d.get("year", 0)
        dt = d.get("doc_type_ext") or d.get("doc_type", "")
        if y and dt:
            year_type_doc_ids.setdefault((y, dt), []).append(d["doc_id"])

    queries: list[dict] = []
    seen_projects: set[str] = set()
    seen_year_types: set[tuple[int, str]] = set()

    for d in docs:
        doc_id = d["doc_id"]
        project = d.get("project_name", "")
        doc_type_ext = d.get("doc_type_ext") or d.get("doc_type", "")
        summary = d.get("summary", "")
        year = d.get("year", 0)

        if not project and not summary:
            continue

        # 1. project + type query
        if project and doc_type_ext:
            queries.append({
                "query": f"{project} {doc_type_ext}",
                "expected_doc_ids": [doc_id],
                "category": "synthetic",
                "tags": ["project_type_match"],
            })

        # 2. summary first sentence
        if summary:
            first_sentence = summary.split(".")[0].split("。")[0].strip()
            if len(first_sentence) >= 5:
                queries.append({
                    "query": first_sentence,
                    "expected_doc_ids": [doc_id],
                    "category": "synthetic",
                    "tags": ["summary_match"],
                })

        # 3. project-only query (once per project)
        if project and project not in seen_projects:
            seen_projects.add(project)
            queries.append({
                "query": project,
                "expected_doc_ids": project_doc_ids.get(project, [doc_id]),
                "category": "synthetic",
                "tags": ["project_match"],
            })

        # 4. topic+project query — match a domain topic found in summary
        if project and summary:
            for topic in _DOMAIN_TOPICS:
                if topic in summary:
                    queries.append({
                        "query": f"{project} {topic}",
                        "expected_doc_ids": [doc_id],
                        "category": "synthetic",
                        "tags": ["topic_project_match"],
                    })
                    break  # One per doc

        # 5. year+type query (once per unique year+type combo)
        if year and doc_type_ext:
            key = (year, doc_type_ext)
            if key not in seen_year_types:
                seen_year_types.add(key)
                queries.append({
                    "query": f"{year}년 {doc_type_ext}",
                    "expected_doc_ids": year_type_doc_ids.get(key, [doc_id]),
                    "category": "synthetic",
                    "tags": ["year_type_match"],
                })

        # 6. keyword extract — 2-3 key nouns from summary
        if summary and len(summary) >= 10:
            # Extract words that match domain topics
            keywords = [t for t in _DOMAIN_TOPICS if t in summary]
            if len(keywords) >= 2:
                queries.append({
                    "query": " ".join(keywords[:3]),
                    "expected_doc_ids": [doc_id],
                    "category": "synthetic",
                    "tags": ["keyword_extract"],
                })

    if include_curated:
        curated = _load_curated_jsonl(curated_path)
        queries.extend(curated)

    return queries


def write_jsonl(queries: list[dict], path: str | Path) -> int:
    """Write queries to a JSONL file. Returns count written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return len(queries)


def read_jsonl(path: str | Path) -> list[dict]:
    """Read queries from a JSONL file."""
    result: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "query" not in obj or "expected_doc_ids" not in obj:
                    logger.warning("Line %d missing required fields, skipping", lineno)
                    continue
                result.append(obj)
            except json.JSONDecodeError:
                logger.warning("Line %d invalid JSON, skipping", lineno)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate search evaluation set")
    parser.add_argument("--db", default="data/registry.db", help="Registry DB path")
    parser.add_argument("--output", default="evals/search_queries.jsonl", help="Output JSONL")
    parser.add_argument("--max-docs", type=int, default=200, help="Max documents to sample")
    parser.add_argument("--no-curated", action="store_true", help="Exclude curated queries")
    args = parser.parse_args()

    # Lazy import to avoid loading full pipeline in eval scripts
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from doc_pipeline.storage.registry import DocumentRegistry

    registry = DocumentRegistry(db_path=args.db)
    queries = generate_eval_set(
        registry, max_docs=args.max_docs, include_curated=not args.no_curated,
    )
    count = write_jsonl(queries, args.output)
    print(f"Generated {count} evaluation queries → {args.output}")


if __name__ == "__main__":
    main()
