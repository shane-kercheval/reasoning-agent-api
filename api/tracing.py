"""
OpenTelemetry tracing configuration for the Reasoning Agent API.

This module provides utilities for setting up distributed tracing with Phoenix Arize.
"""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry import trace
from phoenix.otel import register


def setup_tracing(
    enabled: bool = True,
    project_name: str = 'reasoning-agent',
    endpoint: str | None = None,
    enable_console_export: bool = False,
) -> TracerProvider:
    """
    Initialize OpenTelemetry tracing with Phoenix backend.

    This uses the Phoenix-specific configuration for optimal integration
    with the Phoenix UI and evaluation capabilities.

    Args:
        enabled:
            Whether to enable tracing. If False, returns a no-op provider.
        project_name:
            Name of the project for organizing traces in Phoenix.
        endpoint:
            Phoenix OTLP endpoint. If None, uses PHOENIX_COLLECTOR_ENDPOINT env var
            or defaults to http://localhost:4317.
        enable_console_export:
            Whether to also export spans to console for debugging.

    Returns:
        Configured TracerProvider instance (or no-op if disabled).

    Raises:
        Exception: If tracing is enabled but initialization fails.

    Example:
        >>> from api.config import settings
        >>> tracer_provider = setup_tracing(
        ...     enabled=settings.enable_tracing,
        ...     project_name=settings.phoenix_project_name,
        ...     endpoint=settings.phoenix_collector_endpoint
        ... )
    """
    if not enabled:
        # Create no-op provider and set it as global to override any previous Phoenix providers
        no_op_provider = TracerProvider()
        trace.set_tracer_provider(no_op_provider)
        return no_op_provider

    # Use Phoenix's register function for optimal configuration
    tracer_provider = register(
        project_name=project_name,
        endpoint=endpoint,
        # Use simple processor for immediate export to Phoenix
        # We were originally getting errors with BatchSpanProcessor and Phoenix and had to
        # set this to False to get spans to appear in the UI.
        batch=False,
        auto_instrument=True,  # Auto-detect and instrument known libraries (e.g. OpenAI)
    )

    if enable_console_export:
        # Use SimpleSpanProcessor for console export
        console_exporter = ConsoleSpanExporter()
        console_processor = SimpleSpanProcessor(console_exporter)
        tracer_provider.add_span_processor(console_processor)

    return tracer_provider
