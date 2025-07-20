"""
OpenTelemetry tracing configuration for the Reasoning Agent API.

This module provides utilities for setting up distributed tracing with Phoenix Arize.
"""

import logging

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from phoenix.otel import register

logger = logging.getLogger(__name__)


def setup_tracing(
    enabled: bool = True,
    project_name: str = 'reasoning-agent',
    endpoint: str | None = None,
    enable_console_export: bool = False,
    auto_instrument: bool | None = None,
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
        auto_instrument:
            Whether to enable automatic instrumentation of known libraries.
            If None, defaults to the same value as enabled.

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
    # DEBUG: Log tracing setup details for CI debugging
    import os  # noqa: PLC0415
    print(f"🔍 TRACING DEBUG: enabled={enabled}, project={project_name}, auto_instrument={auto_instrument}")  # noqa: E501
    print(f"🔍 TRACING DEBUG: ENABLE_TRACING env var = {os.environ.get('ENABLE_TRACING', 'NOT_SET')}")  # noqa: E501
    print(f"🔍 TRACING DEBUG: CI env var = {os.environ.get('CI', 'NOT_SET')}")
    logger.error(f"TRACING DEBUG: enabled={enabled}, project={project_name}, auto_instrument={auto_instrument}")  # noqa: E501
    logger.error(f"TRACING DEBUG: ENABLE_TRACING env var = {os.environ.get('ENABLE_TRACING', 'NOT_SET')}")  # noqa: E501
    logger.error(f"TRACING DEBUG: CI env var = {os.environ.get('CI', 'NOT_SET')}")

    if not enabled:
        logger.info("Tracing is disabled via configuration")
        # Return no-op provider that makes all tracing calls safe but doesn't record
        return TracerProvider()

    try:
        # Default auto_instrument to same as enabled if not specified
        if auto_instrument is None:
            auto_instrument = enabled

        # Use Phoenix's register function for optimal configuration
        tracer_provider = register(
            project_name=project_name,
            endpoint=endpoint,
            batch=False,  # Use simple processor for immediate export
            auto_instrument=auto_instrument,  # Control auto-instrumentation
        )

        if enable_console_export:
            # Add console exporter for debugging
            console_exporter = ConsoleSpanExporter()
            console_processor = BatchSpanProcessor(console_exporter)
            tracer_provider.add_span_processor(console_processor)

        logger.info(
            f"Phoenix tracing initialized for project '{project_name}' "
            f"with endpoint: {endpoint}",
        )
        return tracer_provider

    except Exception as e:
        logger.error(f"Failed to initialize Phoenix tracing: {e}")
        raise Exception(f"Tracing initialization failed: {e}") from e


