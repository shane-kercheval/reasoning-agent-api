"""
Mock OpenAI client that uses the exact same protocol as the real OpenAI API.

This ensures our mocks cannot drift from reality because they use the same
builders and parsers as real OpenAI interactions.
"""

from typing import Any
from collections.abc import AsyncIterator
import asyncio

from reasoning_api.openai_protocol import (
    OpenAIResponseParser,
    OpenAIChatResponse,
)


class MockOpenAIClient:
    """
    Mock OpenAI client that enforces the same protocol as real OpenAI.

    Usage:
        mock_client = MockOpenAIClient()
        mock_client.configure_response(
            OpenAIResponseBuilder()
            .id("test-123")
            .model("gpt-4o")
            .choice(0, "assistant", "Hello!")
            .build()
        )

        # Now when ReasoningAgent calls this client, it gets the configured response
    """

    def __init__(self):
        self.chat = MockChatCompletions(self)
        self.recorded_requests: list[dict[str, Any]] = []
        self.configured_responses: list[dict[str, Any]] = []
        self.configured_streaming_responses: list[str] = []
        # Removed: request_validator - use OpenAIChatRequest directly

    def configure_response(self, response_data: dict[str, Any]) -> None:
        """Configure what this mock should return for non-streaming requests."""
        # Validate the response using our Pydantic model to ensure it's correct
        try:
            OpenAIChatResponse(**response_data)
        except Exception as e:
            msg = f"Configured mock response is invalid - would fail with real OpenAI: {e}"
            raise ValueError(msg)

        self.configured_responses.append(response_data)

    def configure_streaming_response(self, stream_data: str) -> None:
        """Configure what this mock should return for streaming requests."""
        # Validate the streaming response
        lines = stream_data.strip().split('\n\n')
        for line in lines:
            if line.startswith('data: ') and not line.endswith('[DONE]'):
                try:
                    OpenAIResponseParser.parse_streaming_chunk(line + '\n\n')
                except ValueError as e:
                    raise ValueError(f"Configured mock streaming response is invalid: {e}")

        self.configured_streaming_responses.append(stream_data)

    def get_last_request(self) -> dict[str, Any] | None:
        """Get the last request made to this mock client."""
        return self.recorded_requests[-1] if self.recorded_requests else None

    def clear_history(self) -> None:
        """Clear recorded requests and configured responses."""
        self.recorded_requests.clear()
        self.configured_responses.clear()
        self.configured_streaming_responses.clear()


class MockChatCompletions:
    """Mock chat completions endpoint."""

    def __init__(self, parent_client: MockOpenAIClient):
        self._client = parent_client

    async def create(self, **kwargs) -> 'MockChatResponse':  # type: ignore[misc]  # noqa: ANN003
        """Create a chat completion (mock implementation)."""
        # Validate the request using our validator
        try:
            self._client.request_validator(kwargs)
        except ValueError as e:
            raise ValueError(f"Invalid request to mock OpenAI client: {e}")

        # Record the request for verification in tests
        self._client.recorded_requests.append(kwargs.copy())

        if kwargs.get("stream"):
            return self._create_streaming_response()
        return self._create_response()

    def _create_response(self) -> 'MockChatResponse':
        """Create a non-streaming response."""
        if not self._client.configured_responses:
            raise ValueError(
                "No configured responses available. Use mock_client.configure_response() "
                "to set up responses before calling the mock.",
            )

        response_data = self._client.configured_responses.pop(0)
        return MockChatResponse(response_data)

    def _create_streaming_response(self) -> 'MockStreamingResponse':
        """Create a streaming response."""
        if not self._client.configured_streaming_responses:
            raise ValueError(
                "No configured streaming responses available. Use "
                "mock_client.configure_streaming_response() to set up responses.",
            )

        stream_data = self._client.configured_streaming_responses.pop(0)
        return MockStreamingResponse(stream_data)


class MockChatResponse:
    """Mock response that matches OpenAI's response structure exactly."""

    def __init__(self, response_data: dict[str, Any]):
        self._data = response_data

        # Parse using our Pydantic model to ensure structure is correct
        parsed = OpenAIChatResponse(**response_data)

        # Expose the same attributes as real OpenAI response
        self.id = parsed.id
        self.object = parsed.object
        self.created = parsed.created
        self.model = parsed.model
        self.choices = [MockChoice(choice) for choice in parsed.choices]
        self.usage = MockUsage(parsed.usage) if parsed.usage else None
        self.system_fingerprint = parsed.system_fingerprint
        self.service_tier = parsed.service_tier

    def model_dump(self) -> dict[str, Any]:
        """Return the raw response data (for compatibility with OpenAI client)."""
        return self._data.copy()


class MockChoice:
    """Mock choice that matches OpenAI's choice structure."""

    def __init__(self, choice_data: dict[str, Any]):
        self.index = choice_data.index
        self.message = MockMessage(choice_data.message) if choice_data.message else None
        self.finish_reason = choice_data.finish_reason
        self.logprobs = choice_data.logprobs


class MockMessage:
    """Mock message that matches OpenAI's message structure."""

    def __init__(self, message_data: dict[str, Any]):
        self.role = message_data.role
        self.content = message_data.content
        self.name = message_data.name
        self.tool_calls = message_data.tool_calls
        self.tool_call_id = message_data.tool_call_id
        self.refusal = message_data.refusal


class MockUsage:
    """Mock usage that matches OpenAI's usage structure."""

    def __init__(self, usage_data: dict[str, Any] | None):
        if usage_data:
            self.prompt_tokens = usage_data.prompt_tokens
            self.completion_tokens = usage_data.completion_tokens
            self.total_tokens = usage_data.total_tokens
            self.prompt_tokens_details = usage_data.prompt_tokens_details
            self.completion_tokens_details = usage_data.completion_tokens_details


class MockStreamingResponse:
    """Mock streaming response that yields chunks like real OpenAI."""

    def __init__(self, stream_data: str):
        self._stream_data = stream_data

    def __aiter__(self) -> AsyncIterator['MockStreamingChunk']:
        return self._stream_chunks()

    async def _stream_chunks(self) -> AsyncIterator['MockStreamingChunk']:
        """Yield chunks like a real streaming response."""
        lines = self._stream_data.strip().split('\n\n')

        for line in lines:
            if line.startswith('data: '):
                if line.endswith('[DONE]'):
                    break

                try:
                    chunk_data = OpenAIResponseParser.parse_streaming_chunk(line + '\n\n')
                    if chunk_data and not chunk_data.get('done'):
                        yield MockStreamingChunk(chunk_data)
                except ValueError:
                    # Skip invalid chunks
                    continue

                # Small delay to simulate network latency
                await asyncio.sleep(0.001)


class MockStreamingChunk:
    """Mock streaming chunk that matches OpenAI's chunk structure."""

    def __init__(self, chunk_data: dict[str, Any]):
        self._data = chunk_data
        self.id = chunk_data["id"]
        self.object = chunk_data["object"]
        self.created = chunk_data["created"]
        self.model = chunk_data["model"]
        self.choices = [MockStreamingChoice(choice) for choice in chunk_data["choices"]]
        self.usage = chunk_data.get("usage")

    def model_dump(self) -> dict[str, Any]:
        """Return the raw chunk data."""
        return self._data.copy()


class MockStreamingChoice:
    """Mock streaming choice."""

    def __init__(self, choice_data: dict[str, Any]):
        self.index = choice_data["index"]
        self.delta = choice_data.get("delta", {})
        self.finish_reason = choice_data.get("finish_reason")
