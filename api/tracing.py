"""
OpenTelemetry tracing configuration for the Reasoning Agent API.

This module provides utilities for setting up distributed tracing with Phoenix Arize,
including span creation, context management, and custom instrumentation.
"""

import asyncio
import logging
from typing import Any, TypeVar
from collections.abc import Callable

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Status, StatusCode, Tracer
from phoenix.otel import register

F = TypeVar('F', bound=Callable[..., Any])

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
        # Return no-op provider that makes all tracing calls safe but doesn't record
        return TracerProvider()

    try:
        # Use Phoenix's register function for optimal configuration
        tracer_provider = register(
            project_name=project_name,
            endpoint=endpoint,
            batch=True,
            auto_instrument=True,  # Auto-detect and instrument known libraries
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


def get_tracer(name: str) -> Tracer:
    """
    Get a tracer instance for creating spans.

    Args:
        name:
            Name of the tracer, typically __name__ of the module.

    Returns:
        Tracer instance for creating spans.

    Example:
        >>> tracer = get_tracer(__name__)
        >>> with tracer.start_as_current_span('my-operation') as span:
        ...     span.set_attribute('user.id', '12345')
        ...     # Your code here
    """
    return trace.get_tracer(name)


def trace_function(
    name: str | None = None, attributes: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """
    Decorator for tracing function execution.

    Args:
        name:
            Custom span name. If None, uses function name.
        attributes:
            Additional attributes to add to the span.

    Example:
        >>> @trace_function(attributes={'operation.type': 'api_call'})
        ... async def fetch_data(user_id: str):
        ...     return await db.get_user(user_id)
    """
    def decorator(func: F) -> F:
        tracer = get_tracer(func.__module__)
        span_name = name or func.__name__

        async def async_wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(
                        Status(StatusCode.ERROR, str(e)),
                    )
                    raise

        def sync_wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(
                        Status(StatusCode.ERROR, str(e)),
                    )
                    raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator


def add_span_attributes(span: trace.Span, attributes: dict[str, Any]) -> None:
    """
    Add multiple attributes to a span with type safety.

    Args:
        span:
            The span to add attributes to.
        attributes:
            Dictionary of attributes to add.

    Example:
        >>> span = tracer.start_span('operation')
        >>> add_span_attributes(span, {
        ...     'http.method': 'POST',
        ...     'http.url': '/api/chat',
        ...     'user.id': 12345
        ... })
    """
    for key, value in attributes.items():
        if value is not None:
            # Convert to appropriate type for OTEL
            if isinstance(value, str | int | float | bool):
                span.set_attribute(key, value)
            elif isinstance(value, list | tuple):
                # Lists must be homogeneous in OTEL
                if all(isinstance(v, str | int | float | bool) for v in value):
                    span.set_attribute(key, list(value))
                else:
                    span.set_attribute(key, str(value))
            else:
                span.set_attribute(key, str(value))

def create_span_context(  # noqa: ANN201
        span_name: str,
        attributes: dict[str, Any] | None = None,
        tracer_name: str | None = None,
    ):
    """
    Context manager for creating spans with automatic error handling.

    Args:
        span_name:
            Name of the span to create.
        attributes:
            Optional attributes to add to the span.
        tracer_name:
            Name of tracer to use. If None, uses calling module name.

    Example:
        >>> async with create_span_context('process_request', {'request.id': '123'}):
        ...     result = await process_data()
        ...     return result
    """
    tracer = get_tracer(tracer_name or __name__)

    class SpanContext:
        def __enter__(self):
            self.span = tracer.start_as_current_span(span_name).__enter__()
            if attributes:
                add_span_attributes(self.span, attributes)
            return self.span

        def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
            if exc_val:
                self.span.record_exception(exc_val)
                self.span.set_status(
                    Status(StatusCode.ERROR, str(exc_val)),
                )
            else:
                self.span.set_status(Status(StatusCode.OK))
            return self.span.__exit__(exc_type, exc_val, exc_tb)

        async def __aenter__(self):
            self.span = tracer.start_as_current_span(span_name).__enter__()
            if attributes:
                add_span_attributes(self.span, attributes)
            return self.span

        async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
            if exc_val:
                self.span.record_exception(exc_val)
                self.span.set_status(
                    Status(StatusCode.ERROR, str(exc_val)),
                )
            else:
                self.span.set_status(Status(StatusCode.OK))
            return self.span.__exit__(exc_type, exc_val, exc_tb)

    return SpanContext()
