"""Structured output models for PydanticAI agents.

PydanticAI validates agent output against these models and auto-retries
on validation failure (up to ``retries`` times).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Search agent output (Phase 2)
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A single document citation in a search answer."""

    doc_ref: str = Field(description="문서 N 형식의 참조 ID")
    doc_id: str = Field(default="", description="문서 고유 ID")
    relevance: str = Field(default="", description="해당 문서가 관련된 이유 (1문장)")


class SearchAnswer(BaseModel):
    """Structured search answer with citations and confidence."""

    answer: str = Field(description="질문에 대한 답변")
    citations: list[Citation] = Field(default_factory=list, description="참조 문서 목록")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="답변 확신도")
    follow_up: str | None = Field(default=None, description="추가 검색 제안")


# ---------------------------------------------------------------------------
# Draft agent output (Phase 2)
# ---------------------------------------------------------------------------


class DraftSection(BaseModel):
    """A single section in a generated draft."""

    title: str = Field(description="섹션 제목")
    content: str = Field(description="섹션 내용")


class DraftOutput(BaseModel):
    """Structured draft generation output."""

    sections: list[DraftSection] = Field(description="생성된 섹션 목록")


# ---------------------------------------------------------------------------
# Classify agent output (Phase 3)
# ---------------------------------------------------------------------------


class ClassifyOutput(BaseModel):
    """Structured document classification output."""

    doc_type_ext: str = Field(description="확장 문서 유형 (YAML 기반)")
    category: str = Field(default="", description="문서 카테고리")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0, description="분류 확신도")
    project_name: str = Field(default="미분류", description="프로젝트명")
    year: int = Field(default=0, description="연도")
    reasoning: str = Field(default="", description="분류 근거 (1문장)")
