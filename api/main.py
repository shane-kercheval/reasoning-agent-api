"""
FastAPI application for the Reasoning Agent API.

This module provides an OpenAI-compatible chat completion API that enhances requests
with reasoning capabilities. The API supports both streaming and non-streaming chat
completions through a clean dependency injection architecture.
"""


import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelsResponse,
    ModelInfo,
    ErrorResponse,
)
from .dependencies import service_container, ReasoningAgentDependency, ToolsDependency
from .auth import verify_token


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:  # noqa: ARG001
    """Manage application lifespan events."""
    # STARTUP: This runs once when server starts
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
    request: ChatCompletionRequest,
    reasoning_agent: ReasoningAgentDependency,
    _: bool = Depends(verify_token),
) -> ChatCompletionResponse | StreamingResponse:
    """
    OpenAI-compatible chat completions endpoint.

    Uses dependency injection to get the reasoning agent instance.
    This provides better testability, type safety, and cleaner architecture.
    Requires authentication via bearer token.
    """
    try:
        if request.stream:
            return StreamingResponse(
                reasoning_agent.execute_stream(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        return await reasoning_agent.execute(request)

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
