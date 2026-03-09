"""Document draft auto-generation using RAG references and templates."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from doc_pipeline.config import settings
from doc_pipeline.generator.templates import TEMPLATES, get_templates

logger = logging.getLogger(__name__)


def _search_references(
    query: str,
    doc_type: str | None = None,
    # 3건이면 유사 사례 참고에 충분 (너무 많으면 프롬프트 토큰 낭비)
    n_results: int = 3,
    *,
    client: Any = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Search vector DB for similar documents and format as references.

    Uses ``unified_search()`` to go through the same improved pipeline
    (RRF + aggregation) as the API and CLI.

    Returns:
        Tuple of (formatted_text_for_llm, structured_references).
    """
    from doc_pipeline.processor.llm import create_client, get_embeddings
    from doc_pipeline.search import unified_search
    from doc_pipeline.storage.vectordb import VectorStore

    try:
        if client is None:
            client = create_client(settings.gemini.api_key)
        store = VectorStore(persist_dir=settings.chroma.persist_dir)

        if store.count == 0:
            return "(참고 사례 없음 — 벡터 DB가 비어있습니다)", []

        query_emb = get_embeddings(client, [query])[0]
        doc_results, _ = unified_search(
            store, query, query_emb,
            n_results=n_results,
            doc_type_filter=doc_type,
        )

        if not doc_results:
            return "(관련 참고 사례를 찾지 못했습니다)", []

        parts: list[str] = []
        structured: list[dict[str, Any]] = []
        for i, doc_res in enumerate(doc_results, 1):
            parts.append(
                f"**[참고 {i}]** {doc_res.doc_type} — {doc_res.project_name} "
                f"(점수: {doc_res.doc_score:.4f})\n> {doc_res.best_chunk.text[:300]}..."
            )
            structured.append({
                "index": i,
                "doc_id": doc_res.doc_id or "",
                "doc_type": doc_res.doc_type,
                "project_name": doc_res.project_name,
                "similarity": round(doc_res.doc_score, 4),
                "text_preview": doc_res.best_chunk.text[:300],
            })
        return "\n\n".join(parts), structured

    except Exception as exc:
        logger.warning("RAG search failed: %s", exc)
        return "(참고 사례 검색 실패)", []


def _generate_sections_batch(
    client: Any,
    section_keys: list[str],
    context: str,
    references: str,
) -> dict[str, str] | None:
    """Generate all missing sections in a single LLM call.

    Returns dict of section_key -> content, or None on failure.
    """
    from google.genai import types

    from doc_pipeline.processor.llm import _call_with_retry

    keys_desc = {
        "structural_judgment": "구조적 판단 (안전성 평가, 하중 검토 결과)",
        "reinforcement_method": "보강 방안 (구체적 보강 공법, 재료, 시공 방법)",
        "conclusion": "결론 및 권고사항",
        "damage_status": "손상 현황 (균열, 변형, 부식 등)",
        "action_content": "조치 내용 (구체적 대응 방안)",
        "action_type": "조치 유형 (설계변경/보완설명/반영불가 중 택일 후 근거 설명)",
        "structure_type": "구조형식 (RC, SRC, S조, PC 등)",
        "municipality": "인허가기관명",
        "attachments": "첨부 자료 목록",
    }
    keys_list = "\n".join(
        f'- "{k}": {keys_desc.get(k, k)}' for k in section_keys
    )
    prompt = (
        "건축·구조 엔지니어링 문서의 섹션들을 작성합니다.\n\n"
        f"## 프로젝트 정보\n{context}\n\n"
        f"## 참고 사례\n{references}\n\n"
        f"## 작성할 섹션\n{keys_list}\n\n"
        "위 섹션들을 JSON 객체로 한 번에 작성하세요.\n"
        "각 섹션은 한국어로 전문적이고 구체적으로 2-3문단 이내로 작성합니다.\n"
        "키는 반드시 위에 명시된 영문 키를 사용하세요."
    )

    try:
        response = _call_with_retry(
            client.models.generate_content,
            model=settings.gemini.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json",
            ),
        )
        import json

        raw = getattr(response, "text", None) or ""
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {k: str(v) for k, v in parsed.items() if k in section_keys}
    except Exception as exc:
        logger.warning("Batch section generation failed: %s — falling back", exc)

    return None


def _generate_section(
    client: Any,
    section_name: str,
    context: str,
    references: str,
) -> str:
    """Use LLM to generate content for a specific section."""
    from google.genai import types

    from doc_pipeline.processor.llm import _call_with_retry

    prompt = (
        f"다음은 건축·구조 엔지니어링 문서의 '{section_name}' 섹션을 작성하기 위한 정보입니다.\n\n"
        f"컨텍스트: {context}\n\n"
        f"참고 사례:\n{references}\n\n"
        f"위 정보를 바탕으로 '{section_name}' 섹션 내용을 한국어로 작성하세요. "
        f"전문적이고 구체적으로 작성하되, 2-3문단 이내로 간결하게 작성하세요."
    )

    try:
        response = _call_with_retry(
            client.models.generate_content,
            model=settings.gemini.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3),
        )
        return (getattr(response, "text", None) or "").strip()
    except Exception as exc:
        logger.warning("Section generation failed for '%s': %s", section_name, exc)
        return f"('{section_name}' 내용을 자동 생성하지 못했습니다)"


def generate_draft(
    doc_type: str,
    project_name: str,
    issue: str,
    *,
    use_llm: bool = True,
    extra_fields: dict[str, str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Generate a document draft using template + RAG references + optional LLM.

    Args:
        doc_type: Document type ('의견서' or '조치계획서').
        project_name: Project name for the document.
        issue: Main issue or topic description.
        use_llm: If True, use LLM to generate section content.
        extra_fields: Additional template field values.

    Returns:
        Tuple of (draft_markdown, structured_references).
    """
    all_templates = get_templates()
    template = all_templates.get(doc_type)
    if not template:
        return f"지원되지 않는 문서 유형: {doc_type}\n지원 유형: {', '.join(all_templates.keys())}", []

    # Create client once for all LLM operations in this draft
    from doc_pipeline.processor.llm import create_client

    llm_client = None
    if use_llm:
        try:
            llm_client = create_client(settings.gemini.api_key)
        except Exception as exc:
            logger.warning("Failed to create Gemini client: %s", exc)

    # Search for references (reuses the same client)
    query = f"{project_name} {issue}"
    references_text, ref_data = _search_references(query, doc_type=doc_type, client=llm_client)

    # Build field values
    fields: dict[str, str] = {
        "project_name": project_name,
        "date": date.today().isoformat(),
        "issue_description": issue,
        "rag_references": references_text,
    }

    if extra_fields:
        fields.update(extra_fields)

    # Generate remaining sections with LLM if enabled
    if use_llm and llm_client is not None:
        context = f"프로젝트: {project_name}, 이슈: {issue}"

        # Collect missing keys and generate all at once
        all_section_keys = (
            "structural_judgment", "reinforcement_method", "conclusion",
            "damage_status", "action_content", "action_type",
            "structure_type", "municipality", "attachments",
        )
        missing_keys = [k for k in all_section_keys if k not in fields]

        if missing_keys:
            batch_result = _generate_sections_batch(
                llm_client, missing_keys, context, references_text,
            )
            if batch_result:
                fields.update(batch_result)

            # Check for any keys still missing after batch
            still_missing = [k for k in missing_keys if k not in fields]
            if still_missing:
                # One more batch attempt for remaining keys only
                retry_result = _generate_sections_batch(
                    llm_client, still_missing, context, references_text,
                )
                if retry_result:
                    fields.update(retry_result)
                else:
                    # Final fallback: placeholder text
                    for key in still_missing:
                        fields[key] = f"[{key} — 자동 생성 실패, 직접 작성 필요]"

    # Fill template with available fields (leave placeholders for missing)
    try:
        draft = template.format_map(
            {k: fields.get(k, f"[{k} 입력 필요]") for k in _extract_keys(template)}
        )
    except (KeyError, IndexError):
        draft = template
        for key, value in fields.items():
            draft = draft.replace(f"{{{key}}}", value)

    return draft, ref_data


def _extract_keys(template: str) -> list[str]:
    """Extract format keys from a template string."""
    import re
    return re.findall(r"\{(\w+)\}", template)
