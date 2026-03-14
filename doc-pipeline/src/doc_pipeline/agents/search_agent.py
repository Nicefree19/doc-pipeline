"""Search answer agent — generates RAG answers from pre-computed context.

The agent receives a ``SearchDeps`` with pre-built ``rag_prompt`` from
``_build_rag_context()``.  It does NOT perform retrieval — that is
deterministic and happens before the agent runs.

Phase 1: output_type=str  (plain text answer)
Phase 2: output_type=SearchAnswer  (structured with citations)
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from doc_pipeline.config import settings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pydantic_ai import Agent, RunContext

    from .deps import SearchDeps

# Lazy-initialized module-level agent (avoids import errors when pydantic-ai
# is not installed and agents are disabled).
_search_agent: Agent | None = None
_lock = threading.Lock()


def get_search_agent() -> Agent:
    """Return the singleton search agent, creating it on first call.

    Thread-safe via double-checked locking.
    """
    global _search_agent  # noqa: PLW0603
    if _search_agent is not None:
        return _search_agent

    with _lock:
        if _search_agent is not None:
            return _search_agent

        from pydantic_ai import Agent as _Agent
        from pydantic_ai.settings import ModelSettings

        from .deps import SearchDeps as _SearchDeps
        from .schemas import SearchAnswer

        agent = _Agent(
            settings.agents.model,
            deps_type=_SearchDeps,
            output_type=SearchAnswer,
            system_prompt=(
                "당신은 건축·구조 엔지니어링 문서 전문 검색 어시스턴트입니다.\n"
                "제공된 검색 결과를 기반으로 정확하고 유용한 답변을 생성하세요.\n"
                "답변 시 어떤 문서를 참고했는지 [문서 N] 형식으로 출처를 표시하세요.\n"
                "검색 결과에 없는 내용은 추측하지 마세요."
            ),
            retries=settings.agents.max_retries,
            model_settings=ModelSettings(temperature=settings.agents.temperature),
        )

        @agent.system_prompt
        def _add_context(ctx: RunContext[SearchDeps]) -> str:
            """Inject RAG context as dynamic system prompt."""
            return ctx.deps.rag_prompt

        _search_agent = agent

    return _search_agent
