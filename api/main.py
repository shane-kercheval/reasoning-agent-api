"""
FastAPI application for the Reasoning Agent API.

This module provides an OpenAI-compatible chat completion API that enhances requests
with reasoning capabilities. The API supports both streaming and non-streaming chat
completions through a clean dependency injection architecture.
"""


import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace, context
from openinference.semconv.trace import SpanAttributes
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

@app.middleware("http")
async def basic_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Basic middleware for non-chat requests."""
    # Only handle tracing for non-chat endpoints here
    if request.url.path == "/v1/chat/completions":
        # Let the endpoint handle its own tracing
        return await call_next(request)

    # Simple tracing for other endpoints
    if request.url.path not in ["/health", "/docs", "/openapi.json", "/favicon.ico"]:
        with tracer.start_as_current_span(f"{request.method} {request.url.path}"):
            return await call_next(request)
    else:
        return await call_next(request)


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
    http_request: Request = None,
    _: bool = Depends(verify_token),
) -> OpenAIChatResponse | StreamingResponse:
    """
    OpenAI-compatible chat completions endpoint.

    Uses dependency injection to get the reasoning agent instance.
    This provides better testability, type safety, and cleaner architecture.
    Requires authentication via bearer token.
    """
    # Extract session ID from headers for tracing correlation
    session_id = http_request.headers.get("X-Session-ID") if http_request else None

    # Create span attributes
    span_attributes = {
        "http.method": "POST",
        "http.url": str(http_request.url) if http_request else "/v1/chat/completions",
        "http.route": "/v1/chat/completions",
        "http.user_agent": http_request.headers.get("user-agent", "") if http_request else "",
    }

    # Add session ID to span if provided
    if session_id:
        span_attributes[SpanAttributes.SESSION_ID] = session_id

    try:
        if request.stream:
            # For streaming responses, we need manual span management
            span = tracer.start_span("POST /v1/chat/completions", attributes=span_attributes)

            # Set span as current in context
            ctx = set_span_in_context(span)
            token = context.attach(ctx)

            start_time = time.time()

            try:
                # Get the generator from reasoning agent
                stream_generator = reasoning_agent.execute_stream(request, parent_span=span)

                async def span_aware_stream():  # noqa: ANN202
                    try:
                        async for chunk in stream_generator:
                            yield chunk
                    finally:
                        # End span after streaming is complete
                        duration_ms = (time.time() - start_time) * 1000
                        span.set_attribute("http.duration_ms", duration_ms)
                        span.set_attribute("http.status_code", 200)
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        span.end()
                        context.detach(token)

                return StreamingResponse(
                    span_aware_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )
            except Exception as e:
                # Handle errors in streaming setup
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.end()
                context.detach(token)
                raise
        else:
            # For non-streaming, use context manager
            with tracer.start_as_current_span("POST /v1/chat/completions", attributes=span_attributes) as span:  # noqa: E501
                start_time = time.time()
                result = await reasoning_agent.execute(request, parent_span=span)

                # Add timing information
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("http.duration_ms", duration_ms)
                span.set_attribute("http.status_code", 200)

                return result

    except httpx.HTTPStatusError as e:
        # Forward OpenAI API errors directly
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
