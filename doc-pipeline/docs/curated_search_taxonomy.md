# Curated Search Taxonomy

Curated search is the primary product-relevance KPI for `doc-pipeline`.

## Intents

| Intent | Purpose | Examples | Retrieval expectation |
|---|---|---|---|
| `technical_qa` | Technical issue / structural reasoning queries | `슬래브 안전성 검토`, `기초 구조 검토` | Prefer technical docs; contracts may appear as fallback if they carry the only project evidence |
| `project_lookup` | Project or site lookup | `홍천읍 갈마곡리 슬래브 변경 구조검토` | Prefer exact project-name matches and metadata-rich docs |
| `contract_lookup` | Contract / scope / amount lookup | `구조용역 계약서`, `용역 계약서 계약금액` | Prefer `계약서` strongly |
| `method_docs_lookup` | Method / VE / proposal corpus lookup | `TSC 공법소개 요약자료`, `내화패널 소개자료` | Prefer `공법자료` and method-doc references |

## Evaluation Rules

- Curated queries are stored in [curated_queries.jsonl](/mnt/d/00.Work_AI_Tool/15.Filename/doc-pipeline/evals/curated_queries.jsonl).
- Ground truth is resolved through [relevance_judgments.jsonl](/mnt/d/00.Work_AI_Tool/15.Filename/doc-pipeline/evals/relevance_judgments.jsonl).
- Every curated query must have:
  - stable `query_id`
  - `intent`
  - non-empty judged ground truth after merge

## Reporting

`scripts/eval_search.py` reports:

- `overall`
- `synthetic`
- `curated`
- `judged_only`
- `ungrounded`
- `by_intent`
- curated false-positive frequency

Curated metrics are the main acceptance gate for relevance work. Synthetic metrics are secondary regression guards.
