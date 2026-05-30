from __future__ import annotations

"""OpenTelemetry tracing setup.

Zero-overhead when unconfigured: until a provider is installed, OTel returns a
no-op tracer, so spans cost nothing in production unless an exporter is wired
(via OTEL_EXPORTER_OTLP_ENDPOINT, or InMemorySpanExporter in tests). Completes
the observability triad alongside the Prometheus /metrics endpoint.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

_provider: TracerProvider | None = None


def setup_tracing(exporter=None) -> TracerProvider:
    """Install a TracerProvider (once) and optionally attach a span exporter."""
    global _provider
    if _provider is None:
        _provider = TracerProvider()
        trace.set_tracer_provider(_provider)
    if exporter is not None:
        _provider.add_span_processor(SimpleSpanProcessor(exporter))
    return _provider


def get_tracer():
    return trace.get_tracer("llm-inference-gateway")
