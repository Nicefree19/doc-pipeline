"""Tests for doc_pipeline.search.query_parser module."""

from __future__ import annotations

import pytest

from doc_pipeline.search.query_parser import ParsedQuery, QueryParser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def known_projects() -> set[str]:
    return {"화성동탄", "판교신도시", "세종시 2-4생활권", "송도국제도시"}


@pytest.fixture()
def type_keywords() -> dict[str, list[str]]:
    return {
        "의견서": ["의견", "검토", "보강", "균열", "구조안전"],
        "계약서": ["계약", "contract", "용역", "위탁"],
        "조치계획서": ["조치", "계획", "지적사항", "심의"],
        "구조계산서": ["구조계산", "구조설계"],
        "도면": ["도면", "배치도", "평면도"],
    }


@pytest.fixture()
def parser(known_projects, type_keywords) -> QueryParser:
    return QueryParser(known_projects=known_projects, type_keywords=type_keywords)


# ---------------------------------------------------------------------------
# Year extraction
# ---------------------------------------------------------------------------

class TestYearExtraction:
    def test_extract_year_4digit(self, parser: QueryParser) -> None:
        result = parser.parse("2024 화성동탄 의견서")
        assert result.year == 2024

    def test_extract_year_with_suffix(self, parser: QueryParser) -> None:
        result = parser.parse("2024년 화성동탄 의견서")
        assert result.year == 2024

    def test_extract_year_removes_from_cleaned(self, parser: QueryParser) -> None:
        result = parser.parse("2024년 화성동탄 의견서")
        assert "2024" not in result.cleaned_query
        assert "화성동탄" in result.cleaned_query

    def test_no_year(self, parser: QueryParser) -> None:
        result = parser.parse("화성동탄 의견서")
        assert result.year == 0

    def test_year_boundary_1900(self, parser: QueryParser) -> None:
        result = parser.parse("1999년 구조검토")
        assert result.year == 1999

    def test_non_year_number_ignored(self, parser: QueryParser) -> None:
        """Numbers outside 1900-2099 should not be treated as years."""
        result = parser.parse("1800년대 건물")
        assert result.year == 0


# ---------------------------------------------------------------------------
# Project matching
# ---------------------------------------------------------------------------

class TestProjectMatching:
    def test_exact_project_match(self, parser: QueryParser) -> None:
        result = parser.parse("화성동탄 구조검토")
        assert result.project == "화성동탄"

    def test_substring_project_match(self, parser: QueryParser) -> None:
        """Query contains known project as substring."""
        result = parser.parse("화성동탄 아파트 구조검토")
        assert result.project == "화성동탄"

    def test_reverse_containment_match(self, parser: QueryParser) -> None:
        """Query token is substring of a known project (reverse direction)."""
        result = parser.parse("화성 구조검토")
        assert result.project == "화성동탄"

    def test_fuzzy_project_match(self, parser: QueryParser) -> None:
        """Fuzzy matching with SequenceMatcher ratio >= 0.7."""
        result = parser.parse("판교신도 구조검토")
        assert result.project == "판교신도시"

    def test_no_project_match(self, parser: QueryParser) -> None:
        result = parser.parse("일반적인 구조검토")
        assert result.project == ""

    def test_longest_project_match(self) -> None:
        """When multiple projects match, prefer the longest."""
        p = QueryParser(known_projects={"송도", "송도국제도시"})
        result = p.parse("송도국제도시 의견서")
        assert result.project == "송도국제도시"

    def test_empty_projects_set(self) -> None:
        p = QueryParser(known_projects=set())
        result = p.parse("화성동탄 구조검토")
        assert result.project == ""

    def test_reverse_containment_prefers_meaningful_tokens(self) -> None:
        """Generic technical tokens should not outweigh unique project tokens."""
        p = QueryParser(
            known_projects={
                "국립충주박물관 DECK PLATE 구조검토 용역",
                "여주시 강천면 걸은리 DECK PLATE 구조검토 용역",
            },
        )
        result = p.parse("국립충주박물관 데크플레이트")
        assert result.project == "국립충주박물관 DECK PLATE 구조검토 용역"

    def test_generic_technical_query_does_not_become_project(self) -> None:
        """Topic-only queries should not be coerced into a project name."""
        p = QueryParser(
            known_projects={
                "안양동 비구조요소 내진설계 구조검토 용역",
                "화성동탄 구조검토",
            },
        )
        result = p.parse("비구조요소 내진설계 구조검토")
        assert result.project == ""


# ---------------------------------------------------------------------------
# Document type matching
# ---------------------------------------------------------------------------

class TestDocTypeMatching:
    def test_doc_type_keyword_match(self, parser: QueryParser) -> None:
        result = parser.parse("구조검토 의견서")
        # "의견" keyword should match "의견서" type
        assert result.doc_type == "의견서"

    def test_doc_type_contract_match(self, parser: QueryParser) -> None:
        result = parser.parse("계약금액 확인")
        assert result.doc_type == "계약서"

    def test_doc_type_longest_match(self, parser: QueryParser) -> None:
        """Longest keyword should match first (specificity)."""
        result = parser.parse("구조계산 검토")
        # "구조계산" (4 chars) should match before "검토" (2 chars)
        assert result.doc_type == "구조계산서"

    def test_doc_type_no_match(self, parser: QueryParser) -> None:
        result = parser.parse("일반적인 질문입니다")
        assert result.doc_type == ""

    def test_doc_type_from_yaml_keywords(self) -> None:
        """Keyword map from TypeRegistry should work."""
        p = QueryParser(type_keywords={"감리보고서": ["감리", "검측"]})
        result = p.parse("감리 보고서 확인")
        assert result.doc_type == "감리보고서"


# ---------------------------------------------------------------------------
# Domain topics
# ---------------------------------------------------------------------------

class TestTopicExtraction:
    def test_topic_extraction(self, parser: QueryParser) -> None:
        result = parser.parse("균열 보강 방법")
        assert "균열" in result.topics
        assert "보강" in result.topics

    def test_multiple_topics(self, parser: QueryParser) -> None:
        result = parser.parse("슬래브 처짐 및 철근 배근")
        assert "슬래브" in result.topics
        assert "처짐" in result.topics
        assert "철근" in result.topics
        assert "배근" in result.topics

    def test_no_topics(self, parser: QueryParser) -> None:
        result = parser.parse("일반적인 질문")
        assert result.topics == []

    def test_synonym_topic(self, parser: QueryParser) -> None:
        """크랙 should map to canonical topic 균열."""
        result = parser.parse("크랙 발생 원인")
        assert "균열" in result.topics


# ---------------------------------------------------------------------------
# Integration / full parse
# ---------------------------------------------------------------------------

class TestFullParse:
    def test_full_parse(self, parser: QueryParser) -> None:
        result = parser.parse("2024년 화성동탄 구조검토 의견서")
        assert result.year == 2024
        assert result.project == "화성동탄"
        assert result.doc_type in ("의견서", "구조계산서")  # depends on keyword priority
        assert result.raw_query == "2024년 화성동탄 구조검토 의견서"
        assert "2024" not in result.cleaned_query

    def test_cleaned_query_content(self, parser: QueryParser) -> None:
        result = parser.parse("2024년 화성동탄 의견서")
        assert "화성동탄" in result.cleaned_query
        assert "의견서" in result.cleaned_query

    def test_empty_query(self, parser: QueryParser) -> None:
        result = parser.parse("")
        assert result.raw_query == ""
        assert result.cleaned_query == ""
        assert result.year == 0
        assert result.project == ""
        assert result.doc_type == ""
        assert result.topics == []

    def test_whitespace_only_query(self, parser: QueryParser) -> None:
        result = parser.parse("   ")
        assert result.cleaned_query == ""

    def test_no_config(self) -> None:
        """Parser with no known_projects or type_keywords should still work."""
        p = QueryParser()
        result = p.parse("2024년 화성동탄 의견서")
        assert result.year == 2024
        assert result.project == ""
        assert result.doc_type == ""


class TestCategoryReturn:
    def test_doc_type_returns_category(self) -> None:
        """_extract_doc_type should return category from type_category_map."""
        parser = QueryParser(
            type_keywords={"구조검토의견서": ["구조검토"]},
            type_category_map={"구조검토의견서": "구조"},
        )
        result = parser.parse("구조검토 결과")
        assert result.doc_type == "구조검토의견서"
        assert result.category == "구조"

    def test_doc_type_no_category_map(self) -> None:
        """Without category map, category should be empty string."""
        parser = QueryParser(
            type_keywords={"구조검토의견서": ["구조검토"]},
        )
        result = parser.parse("구조검토 결과")
        assert result.doc_type == "구조검토의견서"
        assert result.category == ""


class TestMutableProjects:
    def test_known_projects_mutable(self) -> None:
        """_known_projects set can be mutated externally."""
        parser = QueryParser(known_projects={"화성동탄"})
        # Initially finds 화성동탄
        result = parser.parse("화성동탄 검토")
        assert result.project == "화성동탄"

        # Add new project dynamically
        parser._known_projects.add("세종시 행복도시")
        result2 = parser.parse("세종시 행복도시 의견서")
        assert result2.project == "세종시 행복도시"
