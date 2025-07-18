"""
Test configuration and fixtures for the reasoning agent API.

Provides common test fixtures and mock configurations for testing
the FastAPI application and ReasoningAgent with proper HTTP mocking.

This module now imports centralized fixtures from the fixtures/ package
while maintaining backward compatibility with existing tests.
"""

from typing import Any
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
import httpx
from unittest.mock import AsyncMock

from api.openai_protocol import (
    OpenAIChatRequest,
    OpenAIChatResponse,
    ErrorResponse,
)
from api.reasoning_agent import ReasoningAgent
from api.prompt_manager import PromptManager
from api.tools import function_to_tool

# Import all centralized fixtures to make them available globally
from tests.fixtures.tools import *  # noqa: F403
from tests.fixtures.agents import *  # noqa: F403
from tests.fixtures.models import *  # noqa: F403
from tests.fixtures.requests import *  # noqa: F403
from tests.fixtures.responses import *  # noqa: F403

OPENAI_TEST_MODEL = "gpt-4o-mini"


# =============================================================================
# LEGACY FIXTURES - Marked for deprecation after migration is complete
# =============================================================================
# These fixtures remain for backward compatibility during the test refactoring.
# New tests should use the centralized fixtures from tests.fixtures.*
# TODO: Remove these after all tests are migrated to centralized fixtures

# Legacy mock tools - use tests.fixtures.tools instead
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
def sample_chat_request() -> OpenAIChatRequest:
    """Sample OpenAI-compatible chat request."""
    return OpenAIChatRequest(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "What's the weather in Paris?"},
        ],
        temperature=0.7,
        max_tokens=150,
    )


@pytest.fixture
def sample_streaming_request() -> OpenAIChatRequest:
    """Sample streaming chat request."""
    return OpenAIChatRequest(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Search for recent news about AI"},
        ],
        stream=True,
        temperature=0.7,
    )


@pytest.fixture
def mock_openai_response() -> OpenAIChatResponse:
    """
    Mock OpenAI API response - DEPRECATED: Use simple_openai_response from
    fixtures.responses.
    """
    return create_simple_response("This is a test response from OpenAI.", "chatcmpl-test123")  # noqa: F405


@pytest.fixture
def mock_openai_streaming_chunks() -> list[str]:
    """
    Mock OpenAI streaming response chunks - DEPRECATED: Use streaming_chunks from
    fixtures.responses.
    """
    return create_streaming_response("This is a test", "chatcmpl-test123").split('\n\n')[:-1]  # noqa: F405


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
def mock_openai_error_response() -> ErrorResponse:
    """Mock OpenAI API error response - DEPRECATED: Use error_response from fixtures.responses."""
    return ErrorResponse(
        error=ErrorDetail(  # noqa: F405
            message="Invalid API key provided",
            type="invalid_request_error",
            code="invalid_api_key",
        ),
    )
