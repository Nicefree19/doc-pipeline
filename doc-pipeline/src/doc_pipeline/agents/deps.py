"""Dependency injection containers for PydanticAI agents.

Each ``*Deps`` dataclass carries pre-computed context into its agent.
Retrieval happens *before* the agent runs — agents only generate answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchDeps:
    """Pre-computed search context injected into the search agent."""

    query: str
    rag_prompt: str  # built by _build_rag_context()
    references: list[dict[str, Any]] = field(default_factory=list)
    search_profile: str = "auto"


@dataclass
class DraftDeps:
    """Pre-computed draft context injected into the draft agent."""

    template_type: str
    project_name: str
    references_text: str  # from _search_references()
    references: list[dict[str, Any]] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)


@dataclass
class ClassifyDeps:
    """Pre-computed classification context injected into the classify agent."""

    filename: str
    text_preview: str  # first ~2000 chars
    rule_result: str | None = None
    keyword_result: str | None = None
    type_list: str = ""  # available doc types
    types_detail: str = ""  # detailed type descriptions for prompt
