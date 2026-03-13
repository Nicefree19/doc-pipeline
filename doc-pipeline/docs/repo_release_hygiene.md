# Repo And Release Hygiene

## Repository Boundary

- Git repository root: `/mnt/d/00.Work_AI_Tool/15.Filename`
- Product subproject analyzed here: `/mnt/d/00.Work_AI_Tool/15.Filename/doc-pipeline`
- `doc-pipeline` is not a separate Git repository in this workspace snapshot.

When working on search or eval, always state whether the change belongs to:

- root `filehub`
- subproject `doc-pipeline`

## Generated Artifact Policy

Generated eval artifacts are committed only when they are intentionally used as the current baseline:

- [search_queries.jsonl](/mnt/d/00.Work_AI_Tool/15.Filename/doc-pipeline/evals/search_queries.jsonl)
- [baseline.json](/mnt/d/00.Work_AI_Tool/15.Filename/doc-pipeline/evals/baseline.json)
- [eval_report.json](/mnt/d/00.Work_AI_Tool/15.Filename/doc-pipeline/evals/eval_report.json)

Rules:

1. Regenerate all three together.
2. Do not commit only one of them.
3. Note DB/corpus assumptions in the same change.
4. Treat stale eval artifacts as invalid evidence.

## Search Relevance Workflow

1. Update search code and tests.
2. Regenerate curated judgments or verify existing judgments are still valid.
3. Regenerate eval queries.
4. Run eval and update baseline only if it is an intentional new reference point.
5. Record top curated false positives before and after the change.

## OCR Backlog Workflow

Use:

- `scripts/export_failed_ocr.py`
- `scripts/report_ocr_retry_candidates.py`
- `scripts/report_ocr_engine_stats.py`

These reports are operational artifacts and should usually stay out of unrelated code-only commits.
