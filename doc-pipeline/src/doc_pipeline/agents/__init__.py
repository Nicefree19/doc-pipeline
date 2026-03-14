"""PydanticAI agent definitions for doc-pipeline.

Agents wrap LLM answer-generation only — retrieval is deterministic
and happens before agents run (injected via ``deps``).

Feature toggle: ``settings.agents.enabled`` (default: False).
"""
