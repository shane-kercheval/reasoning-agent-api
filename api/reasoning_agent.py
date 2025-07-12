"""
Reasoning agent that enhances OpenAI chat completions with reasoning steps.

This agent acts as a proxy to OpenAI's API while injecting reasoning steps
in the proper OpenAI streaming format. It supports dependency injection
for HTTP client and optional MCP integration.
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator

import httpx

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    StreamChoice,
    Delta,
)
from .mcp_client import MCPClient


class ReasoningAgent:
    """Reasoning agent that processes chat completions with reasoning steps."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        http_client: httpx.AsyncClient,
        mcp_client: MCPClient | None = None,
    ):
        """
        Initialize the reasoning agent.

        Args:
            base_url: Base URL for the OpenAI-compatible API (e.g., "https://api.openai.com/v1")
            api_key: API key for authentication
            http_client: HTTP client for making requests
            mcp_client: Optional MCP client for tool integration
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.http_client = http_client
        self.mcp_client = mcp_client

        # Ensure auth header is set
        if 'Authorization' not in self.http_client.headers:
            self.http_client.headers['Authorization'] = f"Bearer {api_key}"

    async def process_chat_completion(
        self, request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """
        Process a non-streaming chat completion request.

        Args:
            request: The chat completion request

        Returns:
            OpenAI-compatible chat completion response

        Raises:
            httpx.HTTPStatusError: If the OpenAI API returns an error
        """
        # For non-streaming, we don't inject reasoning steps
        # Just enhance the request if MCP is available and forward to OpenAI
        enhanced_request = await self._enhance_request(request)

        # Convert to OpenAI payload
        payload = enhanced_request.model_dump(exclude_unset=True)
        payload['stream'] = False

        # Call OpenAI API
        response = await self.http_client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()

        return ChatCompletionResponse.model_validate(response.json())

    async def process_chat_completion_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[str]:
        """
        Process a streaming chat completion request with reasoning steps.

        Args:
            request: The chat completion request

        Yields:
            Server-sent event formatted strings compatible with OpenAI streaming API

        Raises:
            httpx.HTTPStatusError: If the OpenAI API returns an error
        """
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        created = int(time.time())

        # Emit reasoning steps first
        async for reasoning_chunk in self._generate_reasoning_steps(
            request, completion_id, created,
        ):
            yield f"data: {reasoning_chunk}\n\n"

        # Enhance request if MCP is available
        enhanced_request = await self._enhance_request(request)

        # Stream the actual OpenAI response
        async for openai_chunk in self._stream_openai_response(
            enhanced_request, completion_id, created,
        ):
            yield f"data: {openai_chunk}\n\n"

        yield "data: [DONE]\n\n"

    async def _generate_reasoning_steps(
        self,
        request: ChatCompletionRequest,
        completion_id: str,
        created: int,
    ) -> AsyncGenerator[str]:
        """
        Generate fake reasoning steps in OpenAI streaming format.

        Args:
            request: The original chat completion request
            completion_id: Unique completion ID
            created: Timestamp for the completion

        Yields:
            JSON strings representing reasoning step chunks
        """
        reasoning_steps = [
            "🔍 Analyzing request and gathering context...",
            "🤔 Considering multiple approaches and perspectives...",
            "✅ Formulating comprehensive response strategy...",
        ]

        for step in reasoning_steps:
            chunk = ChatCompletionStreamResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=Delta(content=f"\n{step}\n"),
                        finish_reason=None,
                    ),
                ],
            )
            yield chunk.model_dump_json()

            # Brief pause between reasoning steps
            await asyncio.sleep(0.1)

        # Add separator between reasoning and response
        separator = ChatCompletionStreamResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                StreamChoice(
                    index=0,
                    delta=Delta(content="\n---\n\n"),
                    finish_reason=None,
                ),
            ],
        )
        yield separator.model_dump_json()

    async def _stream_openai_response(
        self,
        request: ChatCompletionRequest,
        completion_id: str,
        created: int,
    ) -> AsyncGenerator[str]:
        """
        Stream response from OpenAI API.

        Args:
            request: The chat completion request
            completion_id: Unique completion ID to use
            created: Timestamp to use

        Yields:
            JSON strings representing OpenAI response chunks
        """
        payload = request.model_dump(exclude_unset=True)
        payload['stream'] = True

        async with self.http_client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break
                    try:
                        # Parse and update with our completion_id and created timestamp
                        chunk_data = json.loads(data)
                        chunk_data["id"] = completion_id
                        chunk_data["created"] = created
                        yield json.dumps(chunk_data)
                    except json.JSONDecodeError:
                        # Skip malformed JSON
                        continue

    async def _enhance_request(
        self, request: ChatCompletionRequest,
    ) -> ChatCompletionRequest:
        """
        Enhance request with MCP tools if available.

        Args:
            request: Original chat completion request

        Returns:
            Enhanced request (currently just returns original)
        """
        # Placeholder for future MCP integration
        # For now, just return the original request
        return request
