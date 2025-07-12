"""
FastAPI application for the Reasoning Agent API.

This module provides an OpenAI-compatible chat completion API that enhances requests
with reasoning capabilities. The API supports both streaming and non-streaming chat
completions through a clean ReasoningAgent architecture.
"""

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelsResponse,
    ModelInfo,
    ErrorResponse,
)
from .reasoning_agent import ReasoningAgent
from .mcp_client import MCPClient, DEFAULT_MCP_CONFIG
from .config import settings

# Global instances
http_client = httpx.AsyncClient(timeout=60.0)
mcp_client = MCPClient(DEFAULT_MCP_CONFIG) if settings.openai_api_key else None
reasoning_agent = ReasoningAgent(
    base_url=settings.reasoning_agent_base_url,
    api_key=settings.openai_api_key,
    http_client=http_client,
    mcp_client=mcp_client,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:  # noqa: ARG001
    """Manage application lifespan events."""
    # Startup
    print("🚀 Starting Reasoning Agent API")
    yield
    # Shutdown
    if mcp_client:
        await mcp_client.close()
    await http_client.aclose()


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
async def list_models() -> ModelsResponse:
    """List available models."""
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
) -> ChatCompletionResponse | StreamingResponse:
    """OpenAI-compatible chat completions endpoint."""
    try:
        if request.stream:
            return StreamingResponse(
                reasoning_agent.process_chat_completion_stream(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        return await reasoning_agent.process_chat_completion(request)

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
async def list_tools() -> dict[str, list[str]]:
    """List available MCP tools."""
    if mcp_client:
        return await mcp_client.list_tools()
    return {"tools": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
