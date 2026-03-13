"""Tests for doc_pipeline.config.settings module."""

from __future__ import annotations

from doc_pipeline.config.settings import (
    ChromaSettings,
    GeminiSettings,
    LoggingSettings,
    OCRGatewaySettings,
    RegistrySettings,
    SecuritySettings,
    Settings,
)


class TestSettings:
    def test_default_settings_created(self) -> None:
        s = Settings()
        assert s.gemini is not None
        assert s.sheets is not None
        assert s.chroma is not None
        assert s.security is not None
        assert s.logging is not None

    def test_gemini_defaults(self) -> None:
        g = GeminiSettings()
        assert g.model_name == "gemini-2.0-flash"
        assert g.embedding_model == "models/gemini-embedding-001"
        assert g.max_rpm == 14

    def test_chroma_defaults(self) -> None:
        c = ChromaSettings()
        assert c.chunk_size == 800
        assert c.chunk_overlap == 200
        assert c.collection_name == "doc_chunks"

    def test_security_default_grade(self) -> None:
        s = SecuritySettings()
        assert s.default_grade in ("A", "B", "C")

    def test_logging_defaults(self) -> None:
        lo = LoggingSettings()
        assert lo.level == "INFO"
        assert lo.log_dir == "logs"

    def test_validate_for_processing_no_key(self) -> None:
        s = Settings()
        s.gemini.api_key = ""
        warnings = s.validate_for_processing()
        assert any("GEMINI_API_KEY" in w for w in warnings)

    def test_validate_for_processing_bad_grade(self) -> None:
        s = Settings()
        s.security.default_grade = "X"
        warnings = s.validate_for_processing()
        assert any("Invalid" in w for w in warnings)

    def test_ocr_gateway_defaults(self) -> None:
        g = OCRGatewaySettings()
        assert g.url == "http://localhost:8010"
        assert g.timeout == 120
        assert g.fallback_engine == "marker"
        assert g.quality_threshold == 0.6
        assert g.enabled is False

    def test_settings_has_ocr_gateway(self) -> None:
        s = Settings()
        assert s.ocr_gateway is not None
        assert s.ocr_gateway.enabled is False

    def test_registry_defaults(self) -> None:
        r = RegistrySettings()
        # db_path is now resolved to absolute path via _resolve_path()
        assert r.db_path.endswith("registry.db")
        assert "data" in r.db_path
        # managed_dir is now resolved to absolute path via _resolve_path()
        assert r.managed_dir.endswith("managed")
        assert "data" in r.managed_dir
        assert r.enabled is True

    def test_settings_has_registry(self) -> None:
        s = Settings()
        assert s.registry is not None
        assert s.registry.enabled is True
