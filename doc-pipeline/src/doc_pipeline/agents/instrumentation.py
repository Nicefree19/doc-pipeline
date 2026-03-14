"""OpenTelemetry instrumentation setup for PydanticAI agents.

PydanticAI has built-in logfire integration — when OTel is configured,
each agent run is automatically recorded as a span with model, tokens,
latency, and retry count.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from doc_pipeline.config.settings import ObservabilitySettings

logger = logging.getLogger(__name__)


def setup_otel(obs_settings: ObservabilitySettings) -> None:
    """Initialize OpenTelemetry tracing with OTLP exporter.

    Safe to call when opentelemetry is not installed — logs a warning
    and returns silently.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "opentelemetry packages not installed — OTel disabled. "
            "Install with: pip install doc-pipeline[observability]"
        )
        return

    resource = Resource.create({"service.name": obs_settings.service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=obs_settings.otel_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    logger.info(
        "OTel tracing enabled → %s (service: %s)",
        obs_settings.otel_endpoint,
        obs_settings.service_name,
    )
