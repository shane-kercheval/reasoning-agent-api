"""
Standardized ReasoningAgent fixtures for testing.

This module provides centralized ReasoningAgent fixture definitions
to eliminate duplication and ensure consistent agent configurations.
"""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
import httpx

from api.reasoning_agent import ReasoningAgent
from api.prompt_manager import PromptManager
from api.tools import Tool

from .tools import (
    BASIC_TOOLS,
    FULL_TOOLS,
    ERROR_TOOLS,
    ASYNC_TOOLS,
    EXTENDED_TOOLS,
    STATEFUL_TOOLS,
    SLEEP_TOOLS,
    weather_tool,
    search_tool,
    failing_tool,
    async_delay_tool,
    sentiment_analysis_tool,
    calculator_tool,
    function_to_tool,
)



def create_mock_prompt_manager(system_prompt: str = "Test system prompt") -> AsyncMock:
    """
    Create a mock PromptManager for testing.

    Args:
        system_prompt: The system prompt to return from get_prompt.

    Returns:
        Mock PromptManager configured for testing.
    """
    mock_prompt_manager = AsyncMock(spec=PromptManager)
    mock_prompt_manager.get_prompt.return_value = system_prompt
    return mock_prompt_manager


def create_reasoning_agent(
    tools: list[Tool] | None = None,
    prompt_manager: AsyncMock | None = None,
    base_url: str = "https://api.openai.com/v1",
    api_key: str = "test-api-key",
    http_client: httpx.AsyncClient | None = None,
) -> ReasoningAgent:
    """
    Create a ReasoningAgent with specified configuration.

    Args:
        tools: List of tools to attach to the agent.
        prompt_manager: Mock prompt manager instance.
        base_url: OpenAI API base URL.
        api_key: OpenAI API key.
        http_client: HTTP client instance.

    Returns:
        Configured ReasoningAgent instance.
    """
    if tools is None:
        tools = []
    if prompt_manager is None:
        prompt_manager = create_mock_prompt_manager()
    if http_client is None:
        http_client = httpx.AsyncClient()

    return ReasoningAgent(
        base_url=base_url,
        api_key=api_key,
        http_client=http_client,
        tools=tools,
        prompt_manager=prompt_manager,
    )


# =============================================================================
# Basic Agent Fixtures
# =============================================================================

@pytest_asyncio.fixture
async def reasoning_agent_no_tools() -> AsyncGenerator[ReasoningAgent]:
    """ReasoningAgent instance without any tools for basic testing."""
    async with httpx.AsyncClient() as client:
        agent = create_reasoning_agent(
            tools=[],
            http_client=client,
        )
        yield agent


@pytest_asyncio.fixture
async def basic_reasoning_agent() -> AsyncGenerator[ReasoningAgent]:
    """ReasoningAgent with basic tools (weather only) for simple tests."""
    async with httpx.AsyncClient() as client:
        agent = create_reasoning_agent(
            tools=BASIC_TOOLS,
            http_client=client,
        )
        yield agent


@pytest_asyncio.fixture
async def full_reasoning_agent() -> AsyncGenerator[ReasoningAgent]:
    """ReasoningAgent with full tool set (weather + search) for complex workflows."""
    async with httpx.AsyncClient() as client:
        agent = create_reasoning_agent(
            tools=FULL_TOOLS,
            http_client=client,
        )
        yield agent


@pytest_asyncio.fixture
async def error_testing_agent() -> AsyncGenerator[ReasoningAgent]:
    """ReasoningAgent with error-prone tools for error handling tests."""
    async with httpx.AsyncClient() as client:
        agent = create_reasoning_agent(
            tools=ERROR_TOOLS,
            http_client=client,
        )
        yield agent


@pytest_asyncio.fixture
async def async_testing_agent() -> AsyncGenerator[ReasoningAgent]:
    """ReasoningAgent with async tools for concurrency testing."""
    async with httpx.AsyncClient() as client:
        agent = create_reasoning_agent(
            tools=ASYNC_TOOLS,
            http_client=client,
        )
        yield agent


# =============================================================================
# Specialized Agent Fixtures
# =============================================================================

@pytest_asyncio.fixture
async def extended_reasoning_agent() -> AsyncGenerator[ReasoningAgent]:
    """ReasoningAgent with extended tool set for comprehensive testing."""
    async with httpx.AsyncClient() as client:
        agent = create_reasoning_agent(
            tools=EXTENDED_TOOLS,
            http_client=client,
        )
        yield agent


@pytest_asyncio.fixture
async def stateful_reasoning_agent() -> AsyncGenerator[ReasoningAgent]:
    """ReasoningAgent with stateful tools for memory/persistence testing."""
    async with httpx.AsyncClient() as client:
        agent = create_reasoning_agent(
            tools=STATEFUL_TOOLS,
            http_client=client,
        )
        yield agent


@pytest_asyncio.fixture
async def sleep_testing_agent() -> AsyncGenerator[ReasoningAgent]:
    """ReasoningAgent with sleep tools for timeout testing."""
    async with httpx.AsyncClient() as client:
        agent = create_reasoning_agent(
            tools=SLEEP_TOOLS,
            http_client=client,
        )
        yield agent


# =============================================================================
# Custom Configuration Fixtures
# =============================================================================

@pytest_asyncio.fixture
async def custom_prompt_agent() -> AsyncGenerator[ReasoningAgent]:
    """ReasoningAgent with custom system prompt for prompt testing."""
    async with httpx.AsyncClient() as client:
        custom_prompt_manager = create_mock_prompt_manager(
            "You are a specialized assistant for testing custom prompts.",
        )
        agent = create_reasoning_agent(
            tools=FULL_TOOLS,
            prompt_manager=custom_prompt_manager,
            http_client=client,
        )
        yield agent


@pytest.fixture
def mock_reasoning_agent_factory():
    """
    Factory function for creating ReasoningAgent instances with custom config.

    Returns:
        Function that creates ReasoningAgent with specified parameters.
    """
    def _create_agent(
        tools: list[Tool] | None = None,
        system_prompt: str = "Test system prompt",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "test-api-key",
    ) -> ReasoningAgent:
        return create_reasoning_agent(
            tools=tools,
            prompt_manager=create_mock_prompt_manager(system_prompt),
            base_url=base_url,
            api_key=api_key,
        )

    return _create_agent


# =============================================================================
# Backwards Compatibility Fixtures
# =============================================================================

@pytest_asyncio.fixture
async def reasoning_agent() -> AsyncGenerator[ReasoningAgent]:
    """
    Default ReasoningAgent fixture for backwards compatibility.

    This maintains compatibility with existing tests that use the generic
    'reasoning_agent' fixture name. Provides weather + search tools.
    """
    async with httpx.AsyncClient() as client:
        agent = create_reasoning_agent(
            tools=FULL_TOOLS,
            http_client=client,
        )
        yield agent


@pytest_asyncio.fixture
async def mock_reasoning_agent() -> AsyncGenerator[ReasoningAgent]:
    """
    Mock ReasoningAgent fixture for backwards compatibility.

    This maintains compatibility with tests that use 'mock_reasoning_agent'.
    Provides the same configuration as the standard reasoning_agent.
    """
    async with httpx.AsyncClient() as client:
        agent = create_reasoning_agent(
            tools=FULL_TOOLS,
            http_client=client,
        )
        yield agent


# =============================================================================
# Utility Functions for Dynamic Agent Creation
# =============================================================================

async def create_agent_with_tools(tool_names: list[str]) -> ReasoningAgent:
    """
    Create an agent with specific tools by name.

    Args:
        tool_names: List of tool names to include ('weather', 'search', 'failing', etc.).

    Returns:
        ReasoningAgent configured with the specified tools.

    Raises:
        ValueError: If an unknown tool name is provided.
    """
    tool_mapping = {
        'weather': weather_tool,
        'search': search_tool,
        'failing': failing_tool,
        'async_delay': async_delay_tool,
        'sentiment': sentiment_analysis_tool,
        'calculator': calculator_tool,
    }

    tools = []
    for tool_name in tool_names:
        if tool_name not in tool_mapping:
            raise ValueError(f"Unknown tool name: {tool_name}")
        tools.append(function_to_tool(tool_mapping[tool_name]))

    return create_reasoning_agent(tools=tools)


def create_agent_for_test_scenario(scenario: str) -> ReasoningAgent:
    """
    Create an agent configured for specific test scenarios.

    Args:
        scenario: Test scenario name ('basic', 'error', 'async', 'extended').

    Returns:
        ReasoningAgent configured for the scenario.

    Raises:
        ValueError: If an unknown scenario is provided.
    """
    scenario_tools = {
        'basic': BASIC_TOOLS,
        'full': FULL_TOOLS,
        'error': ERROR_TOOLS,
        'async': ASYNC_TOOLS,
        'extended': EXTENDED_TOOLS,
        'stateful': STATEFUL_TOOLS,
        'sleep': SLEEP_TOOLS,
    }

    if scenario not in scenario_tools:
        raise ValueError(f"Unknown test scenario: {scenario}")

    return create_reasoning_agent(tools=scenario_tools[scenario])
