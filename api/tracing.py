"""
OpenTelemetry tracing configuration for the Reasoning Agent API.

This module provides utilities for setting up distributed tracing with Phoenix Arize.
"""

import logging
import os

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry import trace
from phoenix.otel import register

logger = logging.getLogger(__name__)


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
        logger.info("Tracing is disabled via configuration")
        # Create no-op provider and set it as global to override any previous Phoenix providers
        no_op_provider = TracerProvider()
        trace.set_tracer_provider(no_op_provider)
        return no_op_provider

    try:
        # Set timeout environment variable to prevent hanging in tests/CI
        # This affects gRPC client timeout for span exports
        if 'OTEL_EXPORTER_OTLP_TIMEOUT' not in os.environ:
            os.environ['OTEL_EXPORTER_OTLP_TIMEOUT'] = '2'  # 2 second timeout

        # Use Phoenix's register function for optimal configuration
        # Use batch=False for immediate span visibility in Phoenix (as user requires)
        tracer_provider = register(
            project_name=project_name,
            endpoint=endpoint,
            batch=False,  # Use simple processor for immediate export to Phoenix
            auto_instrument=True,  # Auto-detect and instrument known libraries
        )

        if enable_console_export:
            # Use SimpleSpanProcessor for console export
            console_exporter = ConsoleSpanExporter()
            console_processor = SimpleSpanProcessor(console_exporter)
            tracer_provider.add_span_processor(console_processor)

        logger.info(
            f"Phoenix tracing initialized for project '{project_name}' "
            f"with endpoint: {endpoint}",
        )
        return tracer_provider

    except Exception as e:
        logger.error(f"Failed to initialize Phoenix tracing: {e}")
        raise Exception(f"Tracing initialization failed: {e}") from e


