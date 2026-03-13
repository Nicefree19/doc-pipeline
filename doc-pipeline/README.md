# doc-pipeline

건축·구조 엔지니어링 문서를 수집, 분류, 검색, RAG 응답, 드래프트 작성까지 연결하는 React + FastAPI 기반 문서 운영 시스템입니다.

## 현재 기준점

- 백엔드: FastAPI + SQLite Registry + ChromaDB + FTS5
- 프런트엔드: React + Vite
- 공법자료 파이프라인: `scan -> extract -> index -> /api/search`
- 운영 기준 문서:
  - [공법자료 운영 Runbook](docs/method_docs_runbook.md)
  - [배포 체크리스트](docs/deployment_checklist.md)
  - [Docker 배포 가이드](DEPLOYMENT.md)
  - [Curated Search Taxonomy](docs/curated_search_taxonomy.md)
  - [Repo And Release Hygiene](docs/repo_release_hygiene.md)

## 설치

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows
pip install -e ".[api,dev]"
cd frontend && npm ci
```

## 로컬 실행

```bash
# 백엔드
.venv/Scripts/python.exe main.py

# 프런트엔드
cd frontend
npm run dev
```

## 운영용 공법자료 처리

```bash
# 1. 신규 자료 스캔
scripts/scan_methods_prod.bat

# 2. 텍스트 추출
scripts/extract_methods_prod.bat

# 3. 인덱싱
scripts/index_methods_prod.bat

# 4. smoke test
.venv/Scripts/python.exe scripts/smoke_deploy.py --base-url http://localhost:8000 --api-key YOUR_KEY
```

## 프로젝트 구조

```text
doc-pipeline/
  src/doc_pipeline/        # API, 검색, 파이프라인 코어
  frontend/                # React UI
  scripts/                 # 운영/실험/배치 스크립트
  docs/                    # runbook, 계획서, 체크리스트
  tests/                   # Python tests
```

## 저장소 경계

- Git 저장소 루트는 `15.Filename/` 입니다.
- `doc-pipeline/`은 이 저장소 안의 서브프로젝트입니다.
- root `filehub` 변경과 `doc-pipeline` 변경은 커밋/리뷰/릴리스에서 분리해서 관리해야 합니다.

## Eval Artifact 정책

다음 파일은 baseline 갱신이 목적일 때만 함께 커밋합니다.

- `evals/search_queries.jsonl`
- `evals/baseline.json`
- `evals/eval_report.json`

세 파일 중 하나만 갱신된 상태는 신뢰 가능한 기준점으로 보지 않습니다.

## 보안 등급

| 등급 | 대상 | AI 처리 |
|------|------|---------|
| A | 계약서 원본(금액) | AI 처리 제외 |
| B | 내부 문서 | 로컬 OCR/로컬 검색 우선 |
| C | 의견서, 조치계획서 | API 기반 처리 가능 |

## 주의

- 운영 배포 전에는 반드시 `docs/deployment_checklist.md`를 기준으로 백업과 smoke test를 수행해야 합니다.
- SSE 인증은 아직 query-string `api_key` fallback을 사용하므로, 외부 노출 배포 전에는 추가 보안 정리가 필요합니다.
