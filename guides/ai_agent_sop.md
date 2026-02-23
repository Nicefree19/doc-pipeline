# AI Agent Standard Operating Procedure (SOP)

이 문서는 AI 코딩 에이전트(사용자 협업 시)의 핵심 작동 규칙(SOP)을 정의합니다.
이 SOP는 에이전트가 단독으로 코드를 작성하고 "완료"라고 선언하기 전에 거쳐야 하는 필수 공정과 제약을 설명합니다.

## 1. 지침 분할 및 조건부 호출 (자동 매뉴얼 시스템)

에이전트는 사용자의 프롬프트 내용이나 수정 중인 파일의 종류에 따라 아래 명시된 **가이드 문서**를 읽어야 합니다. (view_file 권장)

- **조건 1 (키워드/경로):** `backend`, `api`, `server`, `src/backend`, `.py` 수정 시
  👉 `guides/backend_guideline.md` 를 우선 숙독한다.
- **조건 2 (키워드/경로):** `frontend`, `ui`, `react`, `src/components`, `.ts`, `.tsx` 수정 시
  👉 `guides/frontend_guideline.md` 를 우선 숙독한다.
- **조건 3 (키워드/경로):** `test`, `pytest`, `jest`, `tests/` 수정 시
  👉 `guides/testing_guideline.md` 를 우선 숙독한다.

> **에이전트 제약:** 위 조건에 해당될 경우, 코드를 작성하기 전에 반드시 해당 가이드의 내용을 읽고 컨텍스트에 포함시켜야 합니다.

## 2. 작업 기억 시스템 유지

어떤 새로운 피처 구현이나 버그 수정을 요청받든 **즉시 코딩을 시작하지 않습니다.**
반드시 `_agents/workflows/start-task.md` 에 명시된 절차를 밟아, `implementation_plan.md`와 `task.md`를 작성하고 승인을 받아야 합니다.

## 3. 자동 품질 검사 및 셀프 리뷰

코드 작성이 완료되었다고 판단되면 곧바로 사용자에게 보고하기 전, `_agents/workflows/quality-check.md` 에 정의된 자체 품질 검사를 먼저 실행해야 합니다.

## 4. 전문 에이전트 분업 및 완료 보고

모든 구현 및 품질 검증이 끝나면 `_agents/workflows/finish-task.md` 에 따라 최종 문서화(`walkthrough.md`)를 진행한 후 알립니다.
