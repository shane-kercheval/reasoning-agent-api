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

# Global instances - initialized during lifespan
http_client: httpx.AsyncClient | None = None
mcp_client: MCPClient | None = None
reasoning_agent: ReasoningAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:  # noqa: ARG001
    """Manage application lifespan events."""
    global http_client, mcp_client, reasoning_agent  # noqa: PLW0603

    # Startup
    print("🚀 Starting Reasoning Agent API")
    http_client = httpx.AsyncClient(timeout=60.0)
    mcp_client = MCPClient(DEFAULT_MCP_CONFIG) if settings.openai_api_key else None
    reasoning_agent = ReasoningAgent(
        base_url=settings.reasoning_agent_base_url,
        api_key=settings.openai_api_key,
        http_client=http_client,
        mcp_client=mcp_client,
    )

    try:
        yield
    finally:
        # Shutdown
        if mcp_client:
            await mcp_client.close()
        if http_client:
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


def get_reasoning_agent() -> ReasoningAgent:
    """Get the reasoning agent instance, creating a test instance if needed."""
    global reasoning_agent  # noqa: PLW0603
    if reasoning_agent is None:
        # For testing with TestClient (which doesn't run lifespan)
        test_http_client = httpx.AsyncClient(timeout=60.0)
        test_mcp_client = MCPClient(DEFAULT_MCP_CONFIG) if settings.openai_api_key else None
        reasoning_agent = ReasoningAgent(
            base_url=settings.reasoning_agent_base_url,
            api_key=settings.openai_api_key,
            http_client=test_http_client,
            mcp_client=test_mcp_client,
        )
    return reasoning_agent


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """OpenAI-compatible chat completions endpoint."""
    agent = get_reasoning_agent()

    try:
        if request.stream:
            return StreamingResponse(
                agent.process_chat_completion_stream(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        return await agent.process_chat_completion(request)

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
    agent = get_reasoning_agent()
    if agent.mcp_client:
        return await agent.mcp_client.list_tools()
    return {"tools": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
