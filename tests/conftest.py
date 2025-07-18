"""
Test configuration and fixtures for the reasoning agent API.

Provides common test fixtures and mock configurations for testing
the FastAPI application and ReasoningAgent with proper HTTP mocking.
"""

from typing import Any
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
import httpx
from unittest.mock import AsyncMock

from api.models import (
    ChatCompletionRequest,
    ChatMessage,
    MessageRole,
)
from api.reasoning_agent import ReasoningAgent
from api.prompt_manager import PromptManager
from api.tools import function_to_tool

OPENAI_TEST_MODEL = "gpt-4o-mini"


# Mock tools
def get_weather(location: str) -> dict[str, Any]:
    """Get weather information for a location."""
    return {
        "location": location,
        "temperature": "22°C",
        "condition": "Partly cloudy",
        "humidity": "65%",
        "source": "mock_weather_api",
    }


def search_web(query: str, num_results: int = 5) -> dict[str, Any]:
    """Search the web for information."""
    return {
        "query": query,
        "results": [
            {
                "title": f"Result {i+1} for {query}",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"Mock search result {i+1} containing information about {query}",
            }
            for i in range(num_results)
        ],
        "total_results": num_results,
    }


@pytest.fixture
def sample_chat_request() -> ChatCompletionRequest:
    """Sample OpenAI-compatible chat request."""
    return ChatCompletionRequest(
        model="gpt-4o",
        messages=[
            ChatMessage(role=MessageRole.USER, content="What's the weather in Paris?"),
        ],
        temperature=0.7,
        max_tokens=150,
    )


@pytest.fixture
def sample_streaming_request() -> ChatCompletionRequest:
    """Sample streaming chat request."""
    return ChatCompletionRequest(
        model="gpt-4o",
        messages=[
            ChatMessage(role=MessageRole.USER, content="Search for recent news about AI"),
        ],
        stream=True,
        temperature=0.7,
    )


@pytest.fixture
def mock_openai_response() -> dict[str, Any]:
    """Mock OpenAI API response."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from OpenAI.",
                },
                "finish_reason": "stop",
            },
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    }


@pytest.fixture
def mock_openai_streaming_chunks() -> list[str]:
    """Mock OpenAI streaming response chunks."""
    return [
        'data: {"id":"chatcmpl-test123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}',  # noqa: E501
        'data: {"id":"chatcmpl-test123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"This"},"finish_reason":null}]}',  # noqa: E501
        'data: {"id":"chatcmpl-test123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" is"},"finish_reason":null}]}',  # noqa: E501
        'data: {"id":"chatcmpl-test123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" a"},"finish_reason":null}]}',  # noqa: E501
        'data: {"id":"chatcmpl-test123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" test"},"finish_reason":null}]}',  # noqa: E501
        'data: {"id":"chatcmpl-test123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',  # noqa: E501
        "data: [DONE]",
    ]


@pytest.fixture
def http_client() -> httpx.AsyncClient:
    """HTTP client for testing."""
    return httpx.AsyncClient()


@pytest_asyncio.fixture
async def reasoning_agent() -> AsyncGenerator[ReasoningAgent]:
    """ReasoningAgent instance for testing with mock tools."""
    async with httpx.AsyncClient() as client:
        # Create mock tools
        tools = [
            function_to_tool(get_weather),
            function_to_tool(search_web),
        ]

        # Create mock prompt manager
        mock_prompt_manager = AsyncMock(spec=PromptManager)
        mock_prompt_manager.get_prompt.return_value = "Test system prompt"

        yield ReasoningAgent(
            base_url="https://api.openai.com/v1",
            api_key="test-api-key",
            http_client=client,
            tools=tools,
            prompt_manager=mock_prompt_manager,
        )

@pytest_asyncio.fixture
async def reasoning_agent_no_tools() -> AsyncGenerator[ReasoningAgent]:
    """ReasoningAgent instance without any tools."""
    async with httpx.AsyncClient() as client:
        # Create mock prompt manager
        mock_prompt_manager = AsyncMock(spec=PromptManager)
        mock_prompt_manager.get_prompt.return_value = "Test system prompt"

        yield ReasoningAgent(
            base_url="https://api.openai.com/v1",
            api_key="test-api-key",
            http_client=client,
            tools=[],  # No tools
            prompt_manager=mock_prompt_manager,
        )


@pytest.fixture
def mock_openai_error_response() -> dict[str, Any]:
    """Mock OpenAI API error response."""
    return {
        "error": {
            "message": "Invalid API key provided",
            "type": "invalid_request_error",
            "param": None,
            "code": "invalid_api_key",
        },
    }
