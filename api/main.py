"""
FastAPI application for the Reasoning Agent API.

This module provides an OpenAI-compatible chat completion API that enhances requests
with reasoning capabilities and tool usage through MCP (Model Context Protocol).
The API supports both streaming and non-streaming chat completions.
"""

import json
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ModelsResponse,
    ModelInfo,
    StreamChoice,
    Delta,
    ErrorResponse,
)
from .reasoning_agent import ReasoningAgent
from .mcp_client import MCPClient, DEFAULT_MCP_CONFIG
from .config import settings

# Global instances
mcp_client = MCPClient(DEFAULT_MCP_CONFIG)
reasoning_agent = ReasoningAgent(mcp_client)
openai_client = httpx.AsyncClient(
    base_url="https://api.openai.com/v1",
    headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
    timeout=60.0,
)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:  # noqa: ARG001
    """Manage application lifespan events."""
    # Startup
    print("🚀 Starting Reasoning Agent API")
    yield
    # Shutdown
    await mcp_client.close()
    await openai_client.aclose()

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
            ModelInfo(
                id="gpt-3.5-turbo",
                created=int(time.time()),
                owned_by="openai",
            ),
        ],
    )


@app.post("/v1/chat/completions")
async def chat_completions(
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse | StreamingResponse:
    """OpenAI-compatible chat completions endpoint."""
    try:
        if request.stream:
            return StreamingResponse(
                stream_chat_completion(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        return await complete_chat_completion(request)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error={
                    "message": str(e),
                    "type": "internal_server_error",
                    "code": "500",
                },
            ).model_dump(),
        )


async def stream_chat_completion(request: ChatCompletionRequest) -> AsyncGenerator[str]:
    """Handle streaming chat completion."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(time.time())

    # Process request through reasoning agent
    async for update in reasoning_agent.process_streaming_request(request):

        if update["type"] == "reasoning_step":
            # Stream reasoning progress
            chunk = ChatCompletionStreamResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=Delta(content=f"\n{update['content']}"),
                        finish_reason=None,
                    ),
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        elif update["type"] == "enhanced_request":
            # Stream the actual OpenAI response
            enhanced_request = update["request"]

            # Add separator between reasoning and response
            separator_chunk = ChatCompletionStreamResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=Delta(content="\n\n---\n\n"),
                        finish_reason=None,
                    ),
                ],
            )
            yield f"data: {separator_chunk.model_dump_json()}\n\n"

            # Stream OpenAI response
            async for openai_chunk in call_openai_streaming(enhanced_request, completion_id, created):  # noqa: E501
                yield f"data: {openai_chunk}\n\n"

    yield "data: [DONE]\n\n"


async def complete_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Handle non-streaming chat completion."""
    # Process through reasoning agent
    enhanced_request = await reasoning_agent.process_request(request)

    # Call OpenAI with enhanced request
    return await call_openai_complete(enhanced_request)


async def call_openai_streaming(
        request: ChatCompletionRequest,
        completion_id: str,
        created: int,
    ) -> AsyncGenerator[str]:
    """Call OpenAI streaming API."""
    payload = request.model_dump(exclude_unset=True)
    payload["stream"] = True

    async with openai_client.stream("POST", "/chat/completions", json=payload) as response:
        response.raise_for_status()

        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]  # Remove "data: " prefix
                if data == "[DONE]":
                    break
                try:
                    # Parse and re-emit with our completion_id
                    chunk_data = json.loads(data)
                    chunk_data["id"] = completion_id
                    chunk_data["created"] = created
                    yield json.dumps(chunk_data)
                except json.JSONDecodeError:
                    continue


async def call_openai_complete(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Call OpenAI completion API."""
    payload = request.model_dump(exclude_unset=True)
    payload["stream"] = False

    response = await openai_client.post("/chat/completions", json=payload)
    response.raise_for_status()

    return ChatCompletionResponse(**response.json())


@app.get("/health")
async def health_check() -> dict[str, object]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/tools")
async def list_tools() -> dict[str, list[str]]:
    """List available MCP tools."""
    return await mcp_client.list_tools()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
