"""Application settings loaded from environment and config file."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Only load .env in non-CI environments to prevent real keys leaking in CI
if not os.getenv("CI") and os.getenv("NOVA_ENV") != "ci":
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass  # python-dotenv is optional in production/CI


def _resolve_path(raw: str) -> str:
    """Resolve relative path against PROJECT_ROOT."""
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    return str(PROJECT_ROOT / p)


class GeminiSettings(BaseModel):
    api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model_name: str = "gemini-2.0-flash"
    embedding_model: str = "models/gemini-embedding-001"
    max_rpm: int = int(os.getenv("MAX_GEMINI_RPM", "14"))


class GoogleSheetsSettings(BaseModel):
    credentials_path: str = Field(
        default_factory=lambda: os.getenv(
            "GOOGLE_CREDENTIALS_PATH", "config/service_account.json"
        )
    )
    spreadsheet_id: str = Field(
        default_factory=lambda: os.getenv("SPREADSHEET_ID", "")
    )


class WatchSettings(BaseModel):
    nas_root: str = Field(default_factory=lambda: os.getenv("NAS_ROOT", ""))
    contracts_dir: str = Field(default_factory=lambda: os.getenv("WATCH_CONTRACTS", ""))
    action_plans_dir: str = Field(default_factory=lambda: os.getenv("WATCH_ACTION_PLANS", ""))
    opinions_dir: str = Field(default_factory=lambda: os.getenv("WATCH_OPINIONS", ""))


class ChromaSettings(BaseModel):
    persist_dir: str = Field(
        default_factory=lambda: _resolve_path(
            os.getenv("CHROMA_PERSIST_DIR", "data/chromadb")
        )
    )
    collection_name: str = "doc_chunks"
    chunk_size: int = 800
    chunk_overlap: int = 200
    max_chunks_per_doc: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CHUNKS_PER_DOC", "500"))
    )


class LoggingSettings(BaseModel):
    level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_dir: str = Field(default_factory=lambda: os.getenv("LOG_DIR", "logs"))


class TelegramSettings(BaseModel):
    bot_token: str = Field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))


class SecuritySettings(BaseModel):
    default_grade: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_SECURITY_GRADE", "C")
    )


class OCRGatewaySettings(BaseModel):
    url: str = Field(default_factory=lambda: os.getenv("OCR_GATEWAY_URL", "http://localhost:8010"))
    timeout: int = int(os.getenv("OCR_GATEWAY_TIMEOUT", "120"))
    fallback_engine: str = Field(
        default_factory=lambda: os.getenv("OCR_FALLBACK_ENGINE", "marker")
    )
    quality_threshold: float = float(os.getenv("OCR_QUALITY_THRESHOLD", "0.6"))
    enabled: bool = os.getenv("OCR_GATEWAY_ENABLED", "false").lower() == "true"


class RegistrySettings(BaseModel):
    db_path: str = Field(
        default_factory=lambda: _resolve_path(
            os.getenv("REGISTRY_DB_PATH", "data/registry.db")
        )
    )
    managed_dir: str = Field(
        default_factory=lambda: _resolve_path(
            os.getenv("MANAGED_STORAGE_DIR", "data/managed")
        )
    )
    enabled: bool = os.getenv("REGISTRY_ENABLED", "true").lower() == "true"


class ClassifierSettings(BaseModel):
    doc_types_config: str = Field(
        default_factory=lambda: os.getenv(
            "DOC_TYPES_CONFIG",
            str(Path(__file__).parent / "doc_types.yaml"),
        )
    )
    keyword_confidence_threshold: float = float(
        os.getenv("CLASSIFIER_KEYWORD_THRESHOLD", "0.7")
    )
    llm_confidence_threshold: float = float(
        os.getenv("CLASSIFIER_LLM_THRESHOLD", "0.5")
    )


class FTSSettings(BaseModel):
    enabled: bool = os.getenv("FTS_ENABLED", "true").lower() == "true"
    fts_weight: float = float(os.getenv("FTS_WEIGHT", "0.3"))
    vector_weight: float = float(os.getenv("VECTOR_WEIGHT", "0.7"))
    db_path: str = Field(
        default_factory=lambda: _resolve_path(
            os.getenv("FTS_DB_PATH", "data/chunks_fts.db")
        )
    )


class AgentSettings(BaseModel):
    """PydanticAI agent configuration. Disabled by default (feature toggle)."""

    enabled: bool = Field(
        default_factory=lambda: os.getenv("AGENT_ENABLED", "false").lower() == "true"
    )
    model: str = Field(
        default_factory=lambda: os.getenv("AGENT_MODEL", "google-gla:gemini-2.0-flash")
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("AGENT_TEMPERATURE", "0.3"))
    )
    max_retries: int = Field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_RETRIES", "2"))
    )


class ObservabilitySettings(BaseModel):
    """OpenTelemetry observability configuration."""

    otel_enabled: bool = Field(
        default_factory=lambda: os.getenv("OTEL_ENABLED", "false").lower() == "true"
    )
    otel_endpoint: str = Field(
        default_factory=lambda: os.getenv("OTEL_ENDPOINT", "http://localhost:4317")
    )
    service_name: str = "doc-pipeline"


class Settings(BaseModel):
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    sheets: GoogleSheetsSettings = Field(default_factory=GoogleSheetsSettings)
    watch: WatchSettings = Field(default_factory=WatchSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    ocr_engine: str = Field(default_factory=lambda: os.getenv("OCR_ENGINE", "marker"))
    ocr_timeout: int = Field(default_factory=lambda: int(os.getenv("OCR_TIMEOUT", "300")))
    ocr_gateway: OCRGatewaySettings = Field(default_factory=OCRGatewaySettings)
    registry: RegistrySettings = Field(default_factory=RegistrySettings)
    classifier: ClassifierSettings = Field(default_factory=ClassifierSettings)
    fts: FTSSettings = Field(default_factory=FTSSettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    def validate_for_processing(self) -> list[str]:
        """Check required settings and return list of warnings (empty = all OK)."""
        warnings: list[str] = []
        if not self.gemini.api_key:
            warnings.append("GEMINI_API_KEY is not set — LLM features will fail")
        if self.security.default_grade not in ("A", "B", "C"):
            warnings.append(f"Invalid DEFAULT_SECURITY_GRADE: {self.security.default_grade}")
        return warnings


settings = Settings()
