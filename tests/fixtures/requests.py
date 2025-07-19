"""
OpenAI request builders and standard request fixtures.

This module provides centralized request builders and fixtures
to eliminate duplication in test request setup.
"""

import pytest
from api.openai_protocol import OpenAIChatRequest, OpenAIRequestBuilder


# =============================================================================
# Standard Request Fixtures
# =============================================================================

@pytest.fixture
def simple_chat_request() -> OpenAIChatRequest:
    """Basic chat request asking about weather."""
    return (OpenAIRequestBuilder()
            .model("gpt-4o")
            .message("user", "What's the weather in Paris?")
            .temperature(0.7)
            .max_tokens(150)
            .build())


@pytest.fixture
def streaming_chat_request() -> OpenAIChatRequest:
    """Streaming chat request for search query."""
    return (OpenAIRequestBuilder()
            .model("gpt-4o")
            .message("user", "Search for recent news about AI")
            .streaming()
            .temperature(0.7)
            .build())


@pytest.fixture
def json_mode_request() -> OpenAIChatRequest:
    """Request configured for JSON mode response."""
    return (OpenAIRequestBuilder()
            .model("gpt-4o")
            .message("user", "Please provide structured data about Tokyo weather")
            .json_mode()
            .build())


@pytest.fixture
def multi_turn_request() -> OpenAIChatRequest:
    """Multi-turn conversation request."""
    return (OpenAIRequestBuilder()
            .model("gpt-4o")
            .messages([
                {"role": "user", "content": "What's the weather in Tokyo?"},
                {"role": "assistant", "content": "I'll help you get weather information for Tokyo."},  # noqa: E501
                {"role": "user", "content": "Also tell me about New York weather."},
            ])
            .build())


@pytest.fixture
def weather_analysis_request() -> OpenAIChatRequest:
    """Request for weather analysis that should trigger tool usage."""
    return (OpenAIRequestBuilder()
            .model("gpt-4o")
            .message("user", "Compare the weather between Tokyo and London today")
            .temperature(0.1)
            .build())


@pytest.fixture
def search_request() -> OpenAIChatRequest:
    """Request that should trigger search tool usage."""
    return (OpenAIRequestBuilder()
            .model("gpt-4o")
            .message("user", "Find recent articles about machine learning advances")
            .build())


@pytest.fixture
def complex_reasoning_request() -> OpenAIChatRequest:
    """Request that should trigger complex multi-step reasoning."""
    return (
        OpenAIRequestBuilder()
        .model("gpt-4o")
        .message("user", "I'm planning a trip to Tokyo next week. Can you help me understand the weather forecast and find some recent travel recommendations?")  # noqa: E501
        .temperature(0.3)
        .build()
    )


@pytest.fixture
def error_prone_request() -> OpenAIChatRequest:
    """Request designed to test error handling."""
    return (OpenAIRequestBuilder()
            .model("gpt-4o")
            .message("user", "Use the failing tool to test error handling")
            .build())


# =============================================================================
# Request Builder Factory Functions
# =============================================================================

def create_weather_request(location: str, model: str = "gpt-4o") -> OpenAIChatRequest:
    """
    Create a request asking for weather in a specific location.

    Args:
        location: Location to ask about.
        model: OpenAI model to use.

    Returns:
        OpenAIChatRequest asking for weather information.
    """
    return (OpenAIRequestBuilder()
            .model(model)
            .message("user", f"What's the weather in {location}?")
            .build())


def create_search_request(query: str, model: str = "gpt-4o") -> OpenAIChatRequest:
    """
    Create a request for searching information.

    Args:
        query: Search query.
        model: OpenAI model to use.

    Returns:
        OpenAIChatRequest for search query.
    """
    return (OpenAIRequestBuilder()
            .model(model)
            .message("user", f"Search for information about: {query}")
            .build())


def create_comparison_request(item1: str, item2: str, model: str = "gpt-4o") -> OpenAIChatRequest:
    """
    Create a request comparing two items.

    Args:
        item1: First item to compare.
        item2: Second item to compare.
        model: OpenAI model to use.

    Returns:
        OpenAIChatRequest for comparison.
    """
    return (OpenAIRequestBuilder()
            .model(model)
            .message("user", f"Compare {item1} and {item2}")
            .build())


def create_streaming_request(
        user_message: str,
        model: str = "gpt-4o",
        include_usage: bool = False,
    ) -> OpenAIChatRequest:
    """
    Create a streaming request with specified message.

    Args:
        user_message: Message from user.
        model: OpenAI model to use.
        include_usage: Whether to include usage in stream.

    Returns:
        OpenAIChatRequest configured for streaming.
    """
    builder = (OpenAIRequestBuilder()
               .model(model)
               .message("user", user_message)
               .streaming(include_usage))
    return builder.build()


def create_json_request(user_message: str, model: str = "gpt-4o") -> OpenAIChatRequest:
    """
    Create a JSON mode request.

    Args:
        user_message: Message from user.
        model: OpenAI model to use.

    Returns:
        OpenAIChatRequest configured for JSON mode.
    """
    return (OpenAIRequestBuilder()
            .model(model)
            .message("user", user_message)
            .json_mode()
            .build())


def create_conversation_request(
        turns: list[tuple[str, str]],
        model: str = "gpt-4o",
    ) -> OpenAIChatRequest:
    """
    Create a multi-turn conversation request.

    Args:
        turns: List of (role, content) tuples for conversation.
        model: OpenAI model to use.

    Returns:
        OpenAIChatRequest with conversation history.
    """
    builder = OpenAIRequestBuilder().model(model)

    for role, content in turns:
        builder.message(role, content)

    return builder.build()


def create_temperature_request(
        user_message: str,
        temperature: float,
        model: str = "gpt-4o",
    ) -> OpenAIChatRequest:
    """
    Create a request with specific temperature.

    Args:
        user_message: Message from user.
        temperature: Temperature setting.
        model: OpenAI model to use.

    Returns:
        OpenAIChatRequest with specified temperature.
    """
    return (OpenAIRequestBuilder()
            .model(model)
            .message("user", user_message)
            .temperature(temperature)
            .build())


def create_max_tokens_request(
        user_message: str,
        max_tokens: int,
        model: str = "gpt-4o",
    ) -> OpenAIChatRequest:
    """
    Create a request with token limit.

    Args:
        user_message: Message from user.
        max_tokens: Maximum tokens for response.
        model: OpenAI model to use.

    Returns:
        OpenAIChatRequest with token limit.
    """
    return (OpenAIRequestBuilder()
            .model(model)
            .message("user", user_message)
            .max_tokens(max_tokens)
            .build())


# =============================================================================
# Scenario-Based Request Builders
# =============================================================================

def create_reasoning_test_request(scenario: str) -> OpenAIChatRequest:
    """
    Create a request designed for specific reasoning test scenarios.

    Args:
        scenario: Test scenario name.

    Returns:
        OpenAIChatRequest configured for the scenario.

    Raises:
        ValueError: If unknown scenario is provided.
    """
    scenarios = {
        'simple_weather': create_weather_request("Paris"),
        'multi_location': (OpenAIRequestBuilder()
                          .model("gpt-4o")
                          .message("user", "Compare weather in Tokyo, London, and New York")
                          .build()),
        'search_and_weather': (OpenAIRequestBuilder()
                              .model("gpt-4o")
                              .message("user", "Search for travel tips and get weather for Tokyo")
                              .build()),
        'error_handling': (OpenAIRequestBuilder()
                          .model("gpt-4o")
                          .message("user", "Test the failing tool functionality")
                          .build()),
        'concurrent_tools': (OpenAIRequestBuilder()
                            .model("gpt-4o")
                            .message("user", "Get weather for multiple cities simultaneously")
                            .build()),
    }

    if scenario not in scenarios:
        raise ValueError(f"Unknown reasoning test scenario: {scenario}")

    return scenarios[scenario]


# =============================================================================
# Request Validation Helpers
# =============================================================================

def validate_request_structure(request: OpenAIChatRequest) -> bool:
    """
    Validate that a request has proper structure.

    Args:
        request: Request to validate.

    Returns:
        True if request is valid.

    Raises:
        ValueError: If request structure is invalid.
    """
    if not request.model:
        raise ValueError("Request must have a model")

    if not request.messages:
        raise ValueError("Request must have messages")

    if len(request.messages) == 0:
        raise ValueError("Request must have at least one message")

    return True


def get_user_message_content(request: OpenAIChatRequest) -> str:
    """
    Extract user message content from request.

    Args:
        request: Request to extract from.

    Returns:
        Content of the first user message.

    Raises:
        ValueError: If no user message found.
    """
    for message in request.messages:
        if message.get("role") == "user":
            return message.get("content", "")

    raise ValueError("No user message found in request")


def is_streaming_request(request: OpenAIChatRequest) -> bool:
    """
    Check if request is configured for streaming.

    Args:
        request: Request to check.

    Returns:
        True if request has streaming enabled.
    """
    return bool(request.stream)


def is_json_mode_request(request: OpenAIChatRequest) -> bool:
    """
    Check if request is configured for JSON mode.

    Args:
        request: Request to check.

    Returns:
        True if request has JSON mode enabled.
    """
    return (request.response_format is not None
            and request.response_format.get("type") == "json_object")
