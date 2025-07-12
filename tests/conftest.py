"""
Test configuration and fixtures for the reasoning agent API.

Provides common test fixtures and mock configurations for testing
the FastAPI application and MCP integration.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch

from api.main import app
from api.models import ChatCompletionRequest, ChatMessage, MessageRole
from api.mcp_client import MCPClient
from api.reasoning_agent import ReasoningAgent

@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)

@pytest.fixture
async def async_client():
    """Async HTTP client for testing."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def sample_chat_request():
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
def sample_streaming_request():
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
def mock_mcp_client():
    """Mock MCP client."""
    mock_client = AsyncMock(spec=MCPClient)

    # Mock tool responses
    mock_client.call_tool.return_value = {
        "results": [
            {"title": "Test Result", "snippet": "Test content"},
        ],
    }

    mock_client.list_tools.return_value = {
        "test_server": ["web_search", "weather_api"],
    }

    return mock_client

@pytest.fixture
def mock_reasoning_agent(mock_mcp_client: MCPClient):
    """Mock reasoning agent."""
    return ReasoningAgent(mock_mcp_client)

@pytest.fixture
def mock_openai_response():
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
def mock_openai_streaming_response():
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

@pytest.fixture(autouse=True)
def mock_openai_api():
    """Mock OpenAI API calls."""
    with patch("api.main.openai_client") as mock_client:
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Mocked OpenAI response",
                    },
                    "finish_reason": "stop",
                },
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response

        # Mock streaming response
        mock_stream_response = AsyncMock()
        mock_stream_response.aiter_lines.return_value = [
            "data: " + '{"id":"chatcmpl-test123","choices":[{"delta":{"content":"Test"}}]}',
            "data: [DONE]",
        ].__aiter__()
        mock_stream_response.raise_for_status.return_value = None
        mock_client.stream.return_value.__aenter__.return_value = mock_stream_response

        yield mock_client

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# Pytest configuration
pytest_plugins = ["pytest_asyncio"]
