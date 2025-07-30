"""
FastAPI application for the Reasoning Agent API.

This module provides an OpenAI-compatible chat completion API that enhances requests
with reasoning capabilities. The API supports both streaming and non-streaming chat
completions through a clean dependency injection architecture.
"""


import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress

import httpx
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace, context
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
from opentelemetry.trace import set_span_in_context
from .openai_protocol import (
    OpenAIChatRequest,
    OpenAIChatResponse,
    ModelsResponse,
    ModelInfo,
    ErrorResponse,
)
from .dependencies import (
    service_container,
    ReasoningAgentDependency,
    ToolsDependency,
)
from .auth import verify_token
from .config import settings
from .tracing import setup_tracing


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:  # noqa: ARG001
    """
    Manage application lifespan events.

    NOTE: This runs once when the server starts and after it stops (yields control).
    """
    # Always initialize tracing (will be no-op if disabled)
    setup_tracing(
        enabled=settings.enable_tracing,
        project_name=settings.phoenix_project_name,
        endpoint=settings.phoenix_collector_endpoint,
        enable_console_export=settings.enable_console_tracing,
    )

    await service_container.initialize()
    try:
        yield
    finally:
        # SHUTDOWN: This runs once when server stops
        await service_container.cleanup()


app = FastAPI(
    title="Reasoning Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get tracer for request instrumentation
tracer = trace.get_tracer(__name__)


def safe_detach_context(token: object) -> None:
    """
    Safely detach OpenTelemetry context token.

    Handles the case where the token was created in a different asyncio context,
    which can happen during concurrent request cancellation.
    """
    with suppress(ValueError):
        context.detach(token)


def span_cleanup(span: trace.Span, token: object) -> None:
    """
    End span and safely detach context token.

    This helper ensures we always clean up both the span and context together,
    preventing resource leaks and maintaining proper tracing hygiene.

    Only performs cleanup if the span is still recording, preventing
    duplicate cleanup in cases where the span was already ended.
    """
    if span.is_recording():
        span.end()
        safe_detach_context(token)


@app.get("/v1/models")
async def list_models(
    _: bool = Depends(verify_token),
) -> ModelsResponse:
    """
    List available models.

    Requires authentication via bearer token.
    """
    return ModelsResponse(
        data=[
            ModelInfo(
                id="gpt-4o",
                created=int(time.time()),
                owned_by="openai",
            ),
            ModelInfo(
                id="gpt-4o-mini",
                created=int(time.time()),
                owned_by="openai",
            ),
        ],
    )


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: OpenAIChatRequest,
    reasoning_agent: ReasoningAgentDependency,
    http_request: Request,
    _: bool = Depends(verify_token),
) -> OpenAIChatResponse | StreamingResponse:
    """
    OpenAI-compatible chat completions endpoint with streaming-aware tracing.

    TRACING ARCHITECTURE:
    This endpoint manages its own tracing instead of using middleware because:

    1. **Context Manager Incompatibility**: Standard `with tracer.start_as_current_span()`
       automatically ends spans when the context manager exits. For streaming responses,
       this happens immediately when StreamingResponse is returned, before the actual
       streaming completes.

    2. **Asynchronous Generator Consumption**: FastAPI's StreamingResponse consumes
       the generator asynchronously AFTER the endpoint function returns. Middleware-based
       tracing ends too early, causing "Setting attribute on ended span" warnings.

    3. **Manual Span Lifecycle Control**: We need the span to stay alive for the entire
       duration of streaming (potentially 10+ seconds), not just the function execution
       time (milliseconds). Only manual span management provides this control.

    4. **Output Attribute Timing**: The reasoning agent sets output attributes on the
       parent span during streaming. The span must remain active to receive these
       attributes properly.

    Uses dependency injection to get the reasoning agent instance.
    This provides better testability, type safety, and cleaner architecture.
    Requires authentication via bearer token.
    """
    # Extract session ID from headers for tracing correlation
    session_id = http_request.headers.get("X-Session-ID")

    # Create span attributes
    span_attributes = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
        "http.method": "POST",
        "http.url": str(http_request.url),
        "http.route": "/v1/chat/completions",
        "http.user_agent": http_request.headers.get("user-agent", ""),
    }

    # Add session ID to span if provided
    if session_id:
        span_attributes[SpanAttributes.SESSION_ID] = session_id

    # Use manual span management for both streaming and non-streaming
    span = tracer.start_span("POST /v1/chat/completions", attributes=span_attributes)
    ctx = set_span_in_context(span)
    token = context.attach(ctx)

    # Set default success status - will only be overridden by errors
    span.set_attribute("http.status_code", 200)
    span.set_status(trace.Status(trace.StatusCode.OK))

    try:
        if request.stream:
            async def span_aware_stream():  # noqa: ANN202
                """
                Critical wrapper for streaming span lifecycle management with cancellation support.

                WHY THIS WRAPPER IS ESSENTIAL:

                1. **Span Lifecycle Synchronization**: Without this wrapper, the span would
                   end immediately when the endpoint function returns, but FastAPI's
                   StreamingResponse consumes the generator asynchronously afterward.

                2. **Generator Consumption Timing**: This wrapper ensures the span only
                   ends in the `finally` block AFTER the generator is fully consumed,
                   which happens when the client finishes reading all chunks.

                3. **Attribute Setting Window**: The reasoning agent sets output attributes
                   on the parent span during generation. The span must stay alive until
                   after `parent_span.set_attribute(SpanAttributes.OUTPUT_VALUE, ...)`
                   is called, which happens during chunk generation.

                4. **Context Cleanup**: Properly detaches the OpenTelemetry context to
                   prevent memory leaks and context pollution.

                5. **Cancellation Support**: Checks for client disconnection before each yield
                   and raises CancelledError to stop the stream gracefully.

                Without this wrapper:
                - Span duration would be ~10ms (function execution time)
                - Output attributes would fail with "Setting attribute on ended span"
                - Tracing would not reflect actual streaming duration (~10+ seconds)

                With this wrapper:
                - Span duration matches actual streaming time
                - All attributes (input, metadata, output) appear on the same span
                - Clean context management and proper resource cleanup
                - Cancellation when client disconnects
                """
                try:
                    async for chunk in reasoning_agent.execute_stream(request, parent_span=span):
                        # Check for client disconnection before yielding
                        if await http_request.is_disconnected():
                            raise asyncio.CancelledError("Client disconnected")
                        yield chunk
                except asyncio.CancelledError:
                    # Cancellation is expected behavior - keep default OK status
                    span.set_attribute("http.cancelled", True)
                    span.set_attribute("cancellation.reason", "Client disconnected")
                    span_cleanup(span, token)
                    # Don't re-raise - let stream end gracefully
                    return
                finally:
                    # End span after streaming is complete (if not already ended)
                    span_cleanup(span, token)

            return StreamingResponse(
                span_aware_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        # For non-streaming, end span immediately after getting result
        result = await reasoning_agent.execute(request, parent_span=span)

        span_cleanup(span, token)

        return result

    except httpx.HTTPStatusError as e:
        # Forward OpenAI API errors directly
        span.record_exception(e)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
        span_cleanup(span, token)

        content_type = e.response.headers.get('content-type', '')
        if content_type.startswith('application/json'):
            # Return the OpenAI error format directly
            openai_error = e.response.json()
            raise HTTPException(
                status_code=e.response.status_code,
                detail=openai_error,
            )
        raise HTTPException(
            status_code=e.response.status_code,
            detail={"error": {"message": str(e), "type": "http_error"}},
        )
    except Exception as e:
        # Handle other internal errors
        span.record_exception(e)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
        span_cleanup(span, token)

        error_response = ErrorResponse(
            error={
                "message": str(e),
                "type": "internal_server_error",
                "code": "500",
            },
        )
        raise HTTPException(
            status_code=500,
            detail=error_response.model_dump(),
        )


@app.get("/health")
async def health_check() -> dict[str, object]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/tools")
async def list_tools(
    tools: ToolsDependency,
    _: bool = Depends(verify_token),
) -> dict[str, list[str]]:
    """
    List available tools.

    Uses dependency injection to get available tools from MCP servers.
    Returns tool names grouped for compatibility.
    Requires authentication via bearer token.
    """
    try:
        if not tools:
            return {"tools": []}

        # Return list of tool names for compatibility
        tool_names = [tool.name for tool in tools]
        return {"tools": tool_names}

    except Exception as e:
        # Log error and re-raise for proper error response
        logger = logging.getLogger(__name__)
        logger.error(f"Error in list_tools: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
