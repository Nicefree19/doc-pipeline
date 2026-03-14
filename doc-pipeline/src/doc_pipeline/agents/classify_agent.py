"""Classification agent — replaces LLM stage in classifier chain.

Only the LLM stage (Stage 3) is replaced; Rule and Keyword stages
remain deterministic.  The agent receives ``ClassifyDeps`` with
text preview and type registry info, and returns ``ClassifyOutput``.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from doc_pipeline.config import settings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pydantic_ai import Agent, RunContext

    from .deps import ClassifyDeps

_classify_agent: Agent | None = None
_lock = threading.Lock()


def get_classify_agent() -> Agent:
    """Return the singleton classify agent, creating it on first call.

    Thread-safe via double-checked locking.
    """
    global _classify_agent  # noqa: PLW0603
    if _classify_agent is not None:
        return _classify_agent

    with _lock:
        if _classify_agent is not None:
            return _classify_agent

        from pydantic_ai import Agent as _Agent
        from pydantic_ai.settings import ModelSettings

        from .deps import ClassifyDeps as _ClassifyDeps
        from .schemas import ClassifyOutput

        agent = _Agent(
            settings.agents.model,
            deps_type=_ClassifyDeps,
            output_type=ClassifyOutput,
            system_prompt=(
                "건축·구조 엔지니어링 문서 유형을 분류하는 전문 분류 어시스턴트입니다.\n"
                "파일명과 텍스트 내용을 분석하여 가장 적합한 문서 유형을 선택하세요.\n"
                "분류 근거를 간결하게 설명하세요."
            ),
            retries=settings.agents.max_retries,
            model_settings=ModelSettings(temperature=settings.agents.temperature),
        )

        @agent.system_prompt
        def _add_context(ctx: RunContext[ClassifyDeps]) -> str:
            """Inject classification context as dynamic system prompt."""
            parts = [f"분류 유형 목록:\n{ctx.deps.types_detail}"]
            if ctx.deps.filename:
                parts.append(f"파일명: {ctx.deps.filename}")
            parts.append(f"---\n{ctx.deps.text_preview}\n---")
            return "\n\n".join(parts)

        _classify_agent = agent

    return _classify_agent
