"""
OpenAI response builders and standard response fixtures.

This module provides centralized response builders and fixtures
to eliminate duplication in test response setup and mocking.
"""

import json
import time
import pytest
import httpx

from unittest.mock import Mock

from reasoning_api.openai_protocol import (
    OpenAIChatResponse,
    OpenAIResponseBuilder,
    OpenAIStreamingResponseBuilder,
    ErrorResponse,
    ErrorDetail,
)
from reasoning_api.reasoning_models import ReasoningStep, ReasoningAction, ToolPrediction


# =============================================================================
# Standard Response Fixtures
# =============================================================================

@pytest.fixture
def simple_openai_response() -> OpenAIChatResponse:
    """Basic OpenAI response with assistant message."""
    return (OpenAIResponseBuilder()
            .id("chatcmpl-test123")
            .model("gpt-4o")
            .created(int(time.time()))
            .choice(0, "assistant", "This is a test response from OpenAI.")
            .usage(10, 8)
            .build())


@pytest.fixture
def weather_response() -> OpenAIChatResponse:
    """OpenAI response about weather information."""
    return (OpenAIResponseBuilder()
            .id("chatcmpl-weather123")
            .model("gpt-4o")
            .created(int(time.time()))
            .choice(0, "assistant", "The weather in Paris is currently 22°C and partly cloudy.")
            .usage(15, 12)
            .build())


@pytest.fixture
def search_response() -> OpenAIChatResponse:
    """OpenAI response with search results."""
    return (OpenAIResponseBuilder()
            .id("chatcmpl-search123")
            .model("gpt-4o")
            .created(int(time.time()))
            .choice(0, "assistant", "I found several relevant articles about machine learning advances...")  # noqa: E501
            .usage(25, 50)
            .build())


@pytest.fixture
def reasoning_step_response() -> OpenAIChatResponse:
    """OpenAI response containing a reasoning step in JSON format."""
    step = ReasoningStep(
        thought="I need to get weather information for the user's request.",
        next_action=ReasoningAction.USE_TOOLS,
        tools_to_use=[
            ToolPrediction(
                tool_name="weather_tool",
                arguments={"location": "Paris"},
                reasoning="User asked for weather information about Paris",
            ),
        ],
        concurrent_execution=False,
    )

    return (OpenAIResponseBuilder()
            .id("chatcmpl-reasoning123")
            .model("gpt-4o")
            .created(int(time.time()))
            .choice(0, "assistant", step.model_dump_json())
            .usage(50, 100)
            .build())


@pytest.fixture
def error_response() -> ErrorResponse:
    """Standard error response for API failures."""
    return ErrorResponse(
        error=ErrorDetail(
            message="Invalid API key provided",
            type="invalid_request_error",
            code="invalid_api_key",
        ),
    )


@pytest.fixture
def streaming_chunks() -> list[str]:
    """List of streaming response chunks."""
    streaming_response = (
        OpenAIStreamingResponseBuilder()
        .chunk("chatcmpl-test123", "gpt-4o", delta_role="assistant", delta_content="")
        .chunk("chatcmpl-test123", "gpt-4o", delta_content="This")
        .chunk("chatcmpl-test123", "gpt-4o", delta_content=" is")
        .chunk("chatcmpl-test123", "gpt-4o", delta_content=" a")
        .chunk("chatcmpl-test123", "gpt-4o", delta_content=" test")
        .chunk("chatcmpl-test123", "gpt-4o", finish_reason="stop")
        .done()
        .build()
    )
    return streaming_response.split('\n\n')[:-1]  # Remove empty string at end


# =============================================================================
# Response Builder Factory Functions
# =============================================================================

def create_simple_response(
        content: str,
        completion_id: str = "chatcmpl-test",
    ) -> OpenAIChatResponse:
    """
    Create a simple response with specified content.

    Args:
        content: Response content.
        completion_id: Unique completion ID.

    Returns:
        OpenAIChatResponse with the content.
    """
    return (
        OpenAIResponseBuilder()
        .id(completion_id)
        .model("gpt-4o")
        .created(int(time.time()))
        .choice(0, "assistant", content)
        .usage(10, len(content.split()))
        .build()
    )


def create_reasoning_response(
        step: ReasoningStep,
        completion_id: str = "chatcmpl-reasoning",
    ) -> OpenAIChatResponse:
    """
    Create a response containing a reasoning step.

    Args:
        step: ReasoningStep to include.
        completion_id: Unique completion ID.

    Returns:
        OpenAIChatResponse with reasoning step JSON.
    """
    return (OpenAIResponseBuilder()
            .id(completion_id)
            .model("gpt-4o")
            .created(int(time.time()))
            .choice(0, "assistant", step.model_dump_json())
            .usage(50, 100)
            .build())


def create_tool_usage_response(
        tools_used: list[str],
        results: list[str],
        completion_id: str = "chatcmpl-tools",
    ) -> OpenAIChatResponse:
    """
    Create a response showing tool usage results.

    Args:
        tools_used: List of tool names used.
        results: List of tool results.
        completion_id: Unique completion ID.

    Returns:
        OpenAIChatResponse describing tool usage.
    """
    content = f"I used the following tools: {', '.join(tools_used)}. Results: {'; '.join(results)}"

    return (
        OpenAIResponseBuilder()
        .id(completion_id)
        .model("gpt-4o")
        .created(int(time.time()))
        .choice(0, "assistant", content)
        .usage(30, len(content.split()))
        .build()
    )


def create_error_response(
        error_message: str,
        error_type: str = "api_error",
        completion_id: str = "chatcmpl-error",
    ) -> OpenAIChatResponse:
    """
    Create a response indicating an error occurred.

    Args:
        error_message: Error description.
        error_type: Type of error.
        completion_id: Unique completion ID.

    Returns:
        OpenAIChatResponse with error information.
    """
    content = f"Error ({error_type}): {error_message}"

    return (OpenAIResponseBuilder()
            .id(completion_id)
            .model("gpt-4o")
            .created(int(time.time()))
            .choice(0, "assistant", content)
            .usage(5, 10)
            .build())


def create_weather_response(
        location: str,
        temperature: str = "22°C",
        condition: str = "Partly cloudy",
    ) -> OpenAIChatResponse:
    """
    Create a response with weather information.

    Args:
        location: Location name.
        temperature: Temperature reading.
        condition: Weather condition.

    Returns:
        OpenAIChatResponse with weather information.
    """
    content = f"The weather in {location} is currently {temperature} and {condition.lower()}."

    return (OpenAIResponseBuilder()
            .id("chatcmpl-weather")
            .model("gpt-4o")
            .created(int(time.time()))
            .choice(0, "assistant", content)
            .usage(20, 15)
            .build())


def create_search_response(query: str, num_results: int = 3) -> OpenAIChatResponse:
    """
    Create a response with search results.

    Args:
        query: Search query.
        num_results: Number of results to mention.

    Returns:
        OpenAIChatResponse with search results.
    """
    content = f"I found {num_results} relevant results for '{query}'. Here are the top findings..."

    return (OpenAIResponseBuilder()
            .id("chatcmpl-search")
            .model("gpt-4o")
            .created(int(time.time()))
            .choice(0, "assistant", content)
            .usage(25, 30)
            .build())


# =============================================================================
# Streaming Response Factories
# =============================================================================

def create_streaming_response(content: str, completion_id: str = "chatcmpl-stream") -> str:
    """
    Create a complete streaming response.

    Args:
        content: Content to stream.
        completion_id: Unique completion ID.

    Returns:
        Complete SSE stream as string.
    """
    builder = OpenAIStreamingResponseBuilder()

    # Initial role chunk
    builder.chunk(completion_id, "gpt-4o", delta_role="assistant")

    # Stream content word by word
    words = content.split()
    for i, word in enumerate(words):
        chunk_content = word if i == 0 else f" {word}"
        builder.chunk(completion_id, "gpt-4o", delta_content=chunk_content)

    # Final chunk
    builder.chunk(completion_id, "gpt-4o", finish_reason="stop")
    builder.done()

    return builder.build()


def create_reasoning_streaming_response(steps: list[ReasoningStep], final_content: str) -> str:
    """
    Create a streaming response with reasoning steps.

    Args:
        steps: List of reasoning steps.
        final_content: Final response content.

    Returns:
        Complete SSE stream with reasoning.
    """
    builder = OpenAIStreamingResponseBuilder()
    completion_id = "chatcmpl-reasoning-stream"

    # Initial role chunk
    builder.chunk(completion_id, "gpt-4o", delta_role="assistant")

    # Stream reasoning steps (simplified - would include reasoning events in real implementation)
    for i, step in enumerate(steps):
        step_content = f"[Step {i+1}] {step.thought}"
        builder.chunk(completion_id, "gpt-4o", delta_content=step_content)

    # Stream final content
    words = final_content.split()
    for i, word in enumerate(words):
        chunk_content = f" {word}" if i > 0 or steps else word
        builder.chunk(completion_id, "gpt-4o", delta_content=chunk_content)

    # Final chunk
    builder.chunk(completion_id, "gpt-4o", finish_reason="stop")
    builder.done()

    return builder.build()


# =============================================================================
# Response Sequence Factories for Multi-Step Tests
# =============================================================================

def create_reasoning_sequence(num_steps: int = 3) -> list[OpenAIChatResponse]:
    """
    Create a sequence of responses for multi-step reasoning.

    Args:
        num_steps: Number of reasoning steps to create.

    Returns:
        List of OpenAI responses representing reasoning sequence.
    """
    responses = []

    # Create reasoning steps
    for i in range(num_steps - 1):
        step = ReasoningStep(
            thought=f"Step {i+1}: Analyzing the problem and determining next action",
            next_action=ReasoningAction.CONTINUE_THINKING if i < num_steps - 2 else ReasoningAction.USE_TOOLS,  # noqa: E501
            tools_to_use=[
                ToolPrediction(
                    tool_name="weather_tool",
                    arguments={"location": "Paris"},
                    reasoning="Need weather data for analysis",
                ),
            ] if i == num_steps - 2 else [],
        )

        response = create_reasoning_response(step, f"chatcmpl-step{i+1}")
        responses.append(response)

    # Final synthesis response
    final_response = create_simple_response(
        "Based on my analysis, here is the complete answer to your question.",
        "chatcmpl-final",
    )
    responses.append(final_response)

    return responses


def create_tool_execution_sequence() -> list[OpenAIChatResponse]:
    """
    Create a sequence showing tool execution flow.

    Returns:
        List of responses showing reasoning -> tool usage -> synthesis.
    """
    # Step 1: Reasoning step that decides to use tools
    step1 = ReasoningStep(
        thought="I need to gather weather information to answer the user's question",
        next_action=ReasoningAction.USE_TOOLS,
        tools_to_use=[
            ToolPrediction(
                tool_name="weather_tool",
                arguments={"location": "Paris"},
                reasoning="User asked about Paris weather",
            ),
        ],
    )

    # Step 2: Tool execution results (simulated)
    tool_response = create_simple_response(
        "Tool execution completed. Weather data retrieved successfully.",
        "chatcmpl-tools",
    )

    # Step 3: Final synthesis
    final_response = create_weather_response("Paris")

    return [
        create_reasoning_response(step1, "chatcmpl-reasoning"),
        tool_response,
        final_response,
    ]


def create_error_recovery_sequence() -> list[OpenAIChatResponse]:
    """
    Create a sequence showing error recovery.

    Returns:
        List of responses showing failed tool -> retry -> success.
    """
    # Step 1: Initial tool attempt
    step1 = ReasoningStep(
        thought="I'll try to get the weather information",
        next_action=ReasoningAction.USE_TOOLS,
        tools_to_use=[
            ToolPrediction(
                tool_name="failing_tool",
                arguments={"should_fail": True},
                reasoning="Testing error handling",
            ),
        ],
    )

    # Step 2: Error occurred
    error_response = create_error_response("Tool execution failed", "tool_error")

    # Step 3: Recovery attempt
    step3 = ReasoningStep(
        thought="The previous tool failed, let me try an alternative approach",
        next_action=ReasoningAction.USE_TOOLS,
        tools_to_use=[
            ToolPrediction(
                tool_name="weather_tool",
                arguments={"location": "Paris"},
                reasoning="Using reliable weather tool as fallback",
            ),
        ],
    )

    # Step 4: Success
    success_response = create_weather_response("Paris")

    return [
        create_reasoning_response(step1, "chatcmpl-attempt1"),
        error_response,
        create_reasoning_response(step3, "chatcmpl-retry"),
        success_response,
    ]


# =============================================================================
# HTTP Response Builders for Integration Tests
# =============================================================================

def create_http_response(response: OpenAIChatResponse, status_code: int = 200) -> httpx.Response:
    """
    Create an HTTP response containing an OpenAI response.

    Args:
        response: OpenAI response to wrap.
        status_code: HTTP status code.

    Returns:
        httpx.Response with the OpenAI response as JSON.
    """
    return httpx.Response(
        status_code=status_code,
        json=response.model_dump(),
        headers={"content-type": "application/json"},
    )


def create_streaming_http_response(stream_content: str, status_code: int = 200) -> httpx.Response:
    """
    Create an HTTP response for streaming content.

    Args:
        stream_content: SSE stream content.
        status_code: HTTP status code.

    Returns:
        httpx.Response with streaming content.
    """
    return httpx.Response(
        status_code=status_code,
        content=stream_content.encode(),
        headers={"content-type": "text/event-stream"},
    )


def create_error_http_response(error: ErrorResponse, status_code: int = 400) -> httpx.Response:
    """
    Create an HTTP error response.

    Args:
        error: Error response to wrap.
        status_code: HTTP status code.

    Returns:
        httpx.Response with error information.
    """
    return httpx.Response(
        status_code=status_code,
        json=error.model_dump(),
        headers={"content-type": "application/json"},
    )


# =============================================================================
# LiteLLM Chunk Mocks (Real Data Validated)
# =============================================================================

def create_mock_litellm_chunk(
    content: str | None = "Hello",
    role: str | None = None,
    finish_reason: str | None = None,
    usage: dict | None = None,
) -> Mock:
    """
    Create mock matching real LiteLLM ModelResponseStream structure.

    Validated against scripts/litellm_chunks_captured.json. This mock accurately
    represents the structure returned by litellm.acompletion() in streaming mode.

    Args:
        content: Delta content (None for finish/usage chunks)
        role: Delta role (only in first chunk, None thereafter)
        finish_reason: Finish reason (only in finish chunk)
        usage: Usage statistics dict (only in final usage chunk)

    Returns:
        Mock object with model_dump() method matching LiteLLM structure
    """
    mock_chunk = Mock()
    mock_chunk.model_dump.return_value = {
        "id": "chatcmpl-test123",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "object": "chat.completion.chunk",
        "system_fingerprint": "fp_test",
        "choices": [{
            "index": 0,
            "delta": {
                "content": content,
                "role": role,
                "tool_calls": None,
                "function_call": None,
                "audio": None,
                "refusal": None,
                "provider_specific_fields": None,
            },
            "finish_reason": finish_reason,
            "logprobs": None,
        }],
        # Extra LiteLLM-specific fields (preserved via extra='allow')
        "provider_specific_fields": None,
        "citations": None,
        "service_tier": "default",
        "obfuscation": "test123",
        "usage": usage,
    }
    return mock_chunk


# Standard LiteLLM chunk types (validated against real captures)
MOCK_LITELLM_CONTENT_CHUNK = create_mock_litellm_chunk(
    content="Hello",
    role="assistant",  # Role only appears in first chunk
)

MOCK_LITELLM_FINISH_CHUNK = create_mock_litellm_chunk(
    content=None,  # No content in finish chunk
    finish_reason="stop",
)

MOCK_LITELLM_USAGE_CHUNK = create_mock_litellm_chunk(
    content=None,  # No content in usage chunk
    usage={
        "completion_tokens": 8,
        "prompt_tokens": 18,
        "total_tokens": 26,
    },
)


# =============================================================================
# Response Validation Helpers
# =============================================================================

def validate_response_structure(response: OpenAIChatResponse) -> bool:
    """
    Validate that a response has proper OpenAI structure.

    Args:
        response: Response to validate.

    Returns:
        True if response is valid.

    Raises:
        ValueError: If response structure is invalid.
    """
    if not response.id:
        raise ValueError("Response must have an ID")

    if not response.choices:
        raise ValueError("Response must have choices")

    if len(response.choices) == 0:
        raise ValueError("Response must have at least one choice")

    return True


def extract_response_content(response: OpenAIChatResponse) -> str:
    """
    Extract the content from the first choice.

    Args:
        response: Response to extract from.

    Returns:
        Content of the first choice.

    Raises:
        ValueError: If no content found.
    """
    if not response.choices or len(response.choices) == 0:
        raise ValueError("No choices in response")

    choice = response.choices[0]
    if not choice.message or not choice.message.content:
        raise ValueError("No content in first choice")

    return choice.message.content


def is_reasoning_step_response(response: OpenAIChatResponse) -> bool:
    """
    Check if response contains a reasoning step.

    Args:
        response: Response to check.

    Returns:
        True if response contains valid reasoning step JSON.
    """
    try:
        content = extract_response_content(response)
        data = json.loads(content)

        # Check for reasoning step fields
        required_fields = ["thought", "next_action"]
        return all(field in data for field in required_fields)

    except (json.JSONDecodeError, ValueError):
        return False
