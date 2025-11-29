"""
LiteLLM mock builders for integration tests.

Provides explicit, test-specific mock configuration instead of magic keyword detection.
This makes tests clearer, more maintainable, and easier to understand.

Usage:
    # Simple weather query test
    mock = mock_weather_query("Tokyo", "22°C", "Sunny")
    with patch('reasoning_api.executors.reasoning_agent.litellm.acompletion', side_effect=mock):
        # ... test code ...

    # Custom multi-step reasoning
    mock = (
        MockLiteLLMBuilder()
        .reasoning_step_with_tool("weather_api", {"location": "Paris"})
        .reasoning_step_finished("Data retrieved")
        .streaming_response("The weather in Paris is 20°C.")
        .build()
    )
"""

import json
from dataclasses import dataclass, field
from typing import Any
from collections.abc import AsyncGenerator

from litellm import ModelResponse
from litellm.types.utils import StreamingChoices, Delta, Choices, Message, Usage


@dataclass
class MockLiteLLMBuilder:
    """
    Builder for creating test-specific LiteLLM mocks.

    Provides explicit configuration for LiteLLM responses instead of magic
    keyword detection. Each test can configure exactly what the mock LLM
    should return, making tests clearer and more maintainable.

    Example:
        mock = (
            MockLiteLLMBuilder()
            .reasoning_step_with_tool(
                tool_name="get_weather",
                arguments={"location": "Tokyo"},
                thought="I need weather data for Tokyo",
            )
            .streaming_response("The weather in Tokyo is 22°C and Sunny.")
            .build()
        )

        with patch('reasoning_api.executors.reasoning_agent.litellm.acompletion', side_effect=mock):
            # Test code that uses the configured mock
            pass
    """

    _reasoning_steps: list[dict] = field(default_factory=list)
    _streaming_response: str | None = None
    _call_count: list[int] = field(default_factory=lambda: [0])

    def reasoning_step_with_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        thought: str = "I need to use a tool to answer this question",
        reasoning: str | None = None,
    ) -> "MockLiteLLMBuilder":
        """
        Add a reasoning step that uses a tool.

        Args:
            tool_name: Name of the tool to use (e.g., "get_weather", "search_web")
            arguments: Arguments to pass to the tool
            thought: The LLM's thought process for this step
            reasoning: Specific reasoning for why this tool is needed (defaults to thought)

        Returns:
            Self for method chaining
        """
        self._reasoning_steps.append({
            "thought": thought,
            "next_action": "use_tools",
            "tools_to_use": [{
                "tool_name": tool_name,
                "arguments": arguments,
                "reasoning": reasoning or thought,
            }],
            "concurrent_execution": False,
        })
        return self

    def reasoning_step_with_multiple_tools(
        self,
        tools: list[dict[str, Any]],
        thought: str = "I need to use multiple tools",
        concurrent: bool = False,
    ) -> "MockLiteLLMBuilder":
        """
        Add a reasoning step that uses multiple tools.

        Args:
            tools: List of tool dicts with 'tool_name', 'arguments', 'reasoning' keys
            thought: The LLM's thought process for this step
            concurrent: Whether tools should execute concurrently

        Returns:
            Self for method chaining

        Example:
            .reasoning_step_with_multiple_tools([
                {
                    "tool_name": "get_weather",
                    "arguments": {"location": "Tokyo"},
                    "reasoning": "Need Tokyo weather",
                },
                {
                    "tool_name": "get_weather",
                    "arguments": {"location": "Paris"},
                    "reasoning": "Need Paris weather",
                },
            ])
        """
        self._reasoning_steps.append({
            "thought": thought,
            "next_action": "use_tools",
            "tools_to_use": tools,
            "concurrent_execution": concurrent,
        })
        return self

    def reasoning_step_finished(
        self,
        thought: str = "I can answer the question directly without tools",
    ) -> "MockLiteLLMBuilder":
        """
        Add a reasoning step that finishes without using tools.

        Args:
            thought: The LLM's thought process for finishing

        Returns:
            Self for method chaining
        """
        self._reasoning_steps.append({
            "thought": thought,
            "next_action": "finished",
            "tools_to_use": [],
            "concurrent_execution": False,
        })
        return self

    def reasoning_step_continue_thinking(
        self,
        thought: str = "I need to think more about this problem",
    ) -> "MockLiteLLMBuilder":
        """
        Add a reasoning step that continues thinking without taking action yet.

        Args:
            thought: The LLM's thought process

        Returns:
            Self for method chaining
        """
        self._reasoning_steps.append({
            "thought": thought,
            "next_action": "continue_thinking",
            "tools_to_use": [],
            "concurrent_execution": False,
        })
        return self

    def streaming_response(self, content: str) -> "MockLiteLLMBuilder":
        """
        Set the streaming response content.

        This is the final answer the LLM streams back to the user after
        reasoning and tool execution.

        Args:
            content: The response content to stream

        Returns:
            Self for method chaining
        """
        self._streaming_response = content
        return self

    def build(self):
        """
        Build the mock function for use with unittest.mock.patch.

        Returns:
            A callable that can be used with patch() to mock litellm.acompletion

        Example:
            mock_fn = builder.build()
            with patch('reasoning_api.executors.reasoning_agent.litellm.acompletion', side_effect=mock_fn):
                # Test code
        """
        def mock_acompletion(*args: Any, **kwargs: Any) -> ModelResponse | AsyncGenerator[ModelResponse]:  # noqa
            # Streaming call - return streaming response
            if kwargs.get('stream', False):
                content = self._streaming_response or "Here is the response to your question."
                return self._create_mock_stream(content)

            # Check if this is a JSON mode or structured output call
            response_format = kwargs.get('response_format')
            is_json_mode = (
                isinstance(response_format, dict) and response_format.get('type') == 'json_object'
            )
            is_structured_output = (
                response_format is not None and not isinstance(response_format, dict)
            )

            # Non-streaming JSON mode or structured output call - return reasoning step
            if is_json_mode or is_structured_output:
                # Return next reasoning step in sequence
                if self._call_count[0] < len(self._reasoning_steps):
                    step = self._reasoning_steps[self._call_count[0]]
                    self._call_count[0] += 1
                    return self._create_mock_response(json.dumps(step))

                # No more steps configured - default to finished
                default_step = {
                    "thought": "Processing complete",
                    "next_action": "finished",
                    "tools_to_use": [],
                    "concurrent_execution": False,
                }
                return self._create_mock_response(json.dumps(default_step))

            # Fallback for other call types
            content = self._streaming_response or "Response"
            return self._create_mock_response(content)

        return mock_acompletion

    @staticmethod
    def _create_mock_response(content: str) -> ModelResponse:
        """Create a non-streaming LiteLLM response."""
        return ModelResponse(
            id="test-id",
            choices=[Choices(
                index=0,
                message=Message(
                    role="assistant",
                    content=content,
                ),
                finish_reason="stop",
            )],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion",
            usage=Usage(
                prompt_tokens=50,
                completion_tokens=100,
                total_tokens=150,
            ),
        )

    @staticmethod
    async def _create_mock_stream(content: str) -> AsyncGenerator[ModelResponse]:
        """Create a streaming LiteLLM response."""
        # First chunk - role
        yield ModelResponse(
            id="test-id",
            choices=[StreamingChoices(
                index=0,
                delta=Delta(role="assistant"),
                finish_reason=None,
            )],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        )

        # Second chunk - content
        yield ModelResponse(
            id="test-id",
            choices=[StreamingChoices(
                index=0,
                delta=Delta(content=content),
                finish_reason=None,
            )],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        )

        # Final chunk - finish
        yield ModelResponse(
            id="test-id",
            choices=[StreamingChoices(
                index=0,
                delta=Delta(),
                finish_reason="stop",
            )],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        )


# =============================================================================
# Convenience Helpers for Common Test Scenarios
# =============================================================================

def mock_weather_query(
    location: str,
    temperature: str = "22°C",
    condition: str = "Sunny",
    tool_name: str = "get_weather",
) -> callable:
    """
    Quick helper for weather tool tests.

    Args:
        location: Location to query weather for
        temperature: Temperature to return
        condition: Weather condition to return
        tool_name: Name of weather tool ("get_weather" or "weather_api")

    Returns:
        Mock function for use with patch()

    Example:
        with patch('reasoning_api.executors.reasoning_agent.litellm.acompletion',
                   side_effect=mock_weather_query("Tokyo", "25°C", "Cloudy")):
            # Test code
    """
    return (
        MockLiteLLMBuilder()
        .reasoning_step_with_tool(
            tool_name=tool_name,
            arguments={"location": location},
            thought=f"User asked about weather in {location}, I'll use the {tool_name} tool",
        )
        .streaming_response(
            f"Based on the tool results, the weather in {location} is {temperature} and {condition}.",
        )
        .build()
    )


def mock_search_query(
    query: str,
    num_results: int = 3,
    tool_name: str = "search_web",
) -> callable:
    """
    Quick helper for search tool tests.

    Args:
        query: Search query
        num_results: Number of results to mention
        tool_name: Name of search tool ("search_web" or "search_database")

    Returns:
        Mock function for use with patch()
    """
    return (
        MockLiteLLMBuilder()
        .reasoning_step_with_tool(
            tool_name=tool_name,
            arguments={"query": query, "limit": num_results},
            thought=f"I need to search for information about '{query}'",
        )
        .streaming_response(
            f"Based on the search results for '{query}', I found {num_results} relevant resources.",
        )
        .build()
    )


def mock_calculator_query(
    operation: str = "add",
    a: float = 5,
    b: float = 3,
) -> callable:
    """
    Quick helper for calculator tool tests.

    Args:
        operation: Operation to perform ("add", "subtract", "multiply", "divide")
        a: First operand
        b: Second operand

    Returns:
        Mock function for use with patch()
    """
    results = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "Error",
    }
    result = results.get(operation, "Unknown operation")

    return (
        MockLiteLLMBuilder()
        .reasoning_step_with_tool(
            tool_name="calculator",
            arguments={"operation": operation, "a": a, "b": b},
            thought=f"I need to calculate {a} {operation} {b}",
        )
        .streaming_response(
            f"Based on the calculation, {a} {operation} {b} equals {result}.",
        )
        .build()
    )


def mock_direct_answer(content: str) -> callable:
    """
    Quick helper for tests where LLM answers directly without tools.

    Args:
        content: The direct answer content

    Returns:
        Mock function for use with patch()

    Example:
        with patch('reasoning_api.executors.reasoning_agent.litellm.acompletion',
                   side_effect=mock_direct_answer("The answer is 42")):
            # Test code
    """
    return (
        MockLiteLLMBuilder()
        .reasoning_step_finished("I can answer this directly without tools")
        .streaming_response(content)
        .build()
    )


def mock_error_scenario(
    error_in_reasoning: bool = False,
    malformed_json: bool = False,
    invalid_tool_args: bool = False,
) -> callable:
    """
    Helper for testing error scenarios.

    Args:
        error_in_reasoning: LLM reasoning step fails (raises exception during call)
        malformed_json: LLM returns malformed JSON
        invalid_tool_args: LLM provides invalid tool arguments

    Returns:
        Mock function for use with patch()

    Example:
        with patch('reasoning_api.executors.reasoning_agent.litellm.acompletion',
                   side_effect=mock_error_scenario(invalid_tool_args=True)):
            # Test error handling
    """
    if error_in_reasoning:
        # Raise an exception during LLM call to simulate reasoning step failure
        async def mock_fn(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
            raise Exception("LLM reasoning step failed")
        return mock_fn

    if malformed_json:
        # Return a non-JSON response when JSON is expected
        def mock_fn(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
            response_format = kwargs.get('response_format')
            is_json_mode = (
                isinstance(response_format, dict) and response_format.get('type') == 'json_object'
            )
            is_structured_output = (
                response_format is not None and not isinstance(response_format, dict)
            )
            if is_json_mode or is_structured_output:
                return MockLiteLLMBuilder._create_mock_response("{invalid json}}")
            return MockLiteLLMBuilder._create_mock_response("Error response")
        return mock_fn

    if invalid_tool_args:
        # Return tool call with missing required arguments
        return (
            MockLiteLLMBuilder()
            .reasoning_step_with_tool(
                tool_name="get_weather",
                arguments={"invalid_field": "oops"},  # Missing 'location'
                thought="Attempting to use tool with invalid args",
            )
            .streaming_response("I encountered an error with the tool.")
            .build()
        )

    # Default error scenario
    return (
        MockLiteLLMBuilder()
        .streaming_response("An error occurred during processing.")
        .build()
    )
