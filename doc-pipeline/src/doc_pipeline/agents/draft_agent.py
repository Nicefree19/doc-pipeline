"""Draft generation agent — produces structured document sections.

Replaces ``_generate_sections_batch()`` in ``drafter.py`` when agents
are enabled.  The agent receives pre-searched references via ``DraftDeps``
and returns a ``DraftOutput`` with validated sections.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from doc_pipeline.config import settings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pydantic_ai import Agent, RunContext

    from .deps import DraftDeps

_draft_agent: Agent | None = None
_lock = threading.Lock()


def get_draft_agent() -> Agent:
    """Return the singleton draft agent, creating it on first call.

    Thread-safe via double-checked locking.
    """
    global _draft_agent  # noqa: PLW0603
    if _draft_agent is not None:
        return _draft_agent

    with _lock:
        if _draft_agent is not None:
            return _draft_agent

        from pydantic_ai import Agent as _Agent
        from pydantic_ai.settings import ModelSettings

        from .deps import DraftDeps as _DraftDeps
        from .schemas import DraftOutput

        agent = _Agent(
            settings.agents.model,
            deps_type=_DraftDeps,
            output_type=DraftOutput,
            system_prompt=(
                "건축·구조 엔지니어링 문서 초안을 작성하는 전문 작성 어시스턴트입니다.\n"
                "제공된 프로젝트 정보와 참고 사례를 바탕으로 각 섹션을 한국어로 "
                "전문적이고 구체적으로 2-3문단 이내로 작성하세요.\n"
                "참고 사례에 없는 내용은 일반적인 엔지니어링 관행에 따라 작성하되, "
                "추측인 경우 명시하세요."
            ),
            retries=settings.agents.max_retries,
            model_settings=ModelSettings(temperature=settings.agents.temperature),
        )

        @agent.system_prompt
        def _add_context(ctx: RunContext[DraftDeps]) -> str:
            """Inject draft context as dynamic system prompt."""
            sections_list = "\n".join(f"- {s}" for s in ctx.deps.sections)
            return (
                f"## 프로젝트 정보\n"
                f"문서 유형: {ctx.deps.template_type}\n"
                f"프로젝트: {ctx.deps.project_name}\n\n"
                f"## 참고 사례\n{ctx.deps.references_text}\n\n"
                f"## 작성할 섹션\n{sections_list}"
            )

        _draft_agent = agent

    return _draft_agent
