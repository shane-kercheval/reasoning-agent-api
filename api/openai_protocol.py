"""
OpenAI API Protocol Abstraction.

This module provides Pydantic models that match OpenAI's exact API specification.
All mocks and real API calls must use these to ensure consistency and prevent duplication.

These models are validated against real OpenAI API responses in integration tests,
ensuring our mocks cannot drift from reality.

Based on OpenAI API documentation:
- https://platform.openai.com/docs/api-reference/chat
- https://platform.openai.com/docs/guides/structured-outputs
- https://platform.openai.com/docs/api-reference/chat-streaming
"""
import json
import time
from typing import Any, Literal, TYPE_CHECKING
from enum import Enum
from pydantic import BaseModel, ConfigDict, field_validator
from functools import singledispatch
from .reasoning_models import ReasoningEvent
if TYPE_CHECKING:
    from litellm.types.utils import ModelResponseStream


SSE_DONE = "data: [DONE]\n\n"


@singledispatch
def create_sse(data: object) -> str:  # noqa: ARG001
    """
    Convert a string or dictionary to a Server-Sent Events (SSE) chunk.

    Args:
        data: string/dict to convert.

    Returns:
        SSE formatted string.
    """
    raise TypeError("Unsupported type")

@create_sse.register
def _(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"

@create_sse.register
def _(data: str) -> str:
    return f"data: {data}\n\n"

@create_sse.register
def _(data: BaseModel) -> str:
    return f"data: {json.dumps(data.model_dump())}\n\n"

def is_sse(value: str) -> bool:
    """
    Check if a string is in Server-Sent Events (SSE) format.

    Args:
        value: The string to check.

    Returns:
        True if the string is in SSE format, False otherwise.
    """
    return value.startswith("data: ") and value.endswith("\n\n")

def is_sse_done(value: str) -> bool:
    """
    Check if a string is the SSE [DONE] marker.

    Args:
        value: The string to check.

    Returns:
        True if the string is the SSE [DONE] marker, False otherwise.
    """
    return value.lstrip() == SSE_DONE.lstrip()

def parse_sse(sse_chunk: str) -> dict[str, Any]:
    """
    Extract data from a Server-Sent Events (SSE) chunk.

    Args:
        sse_chunk: The SSE formatted string.

    Returns:
        Parsed data as a dictionary.
    """
    if sse_chunk.startswith("data: "):
        data_part = sse_chunk[6:].strip()
        if data_part == "[DONE]":
            return {"done": True}
        return json.loads(data_part)
    raise ValueError("Invalid SSE format")

class OpenAIObjectType(Enum):
    """OpenAI response object types."""

    CHAT_COMPLETION = "chat.completion"
    CHAT_COMPLETION_CHUNK = "chat.completion.chunk"


class OpenAIFinishReason(Enum):
    """OpenAI finish reasons."""

    STOP = "stop"
    LENGTH = "length"
    FUNCTION_CALL = "function_call"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"


class MessageRole(str, Enum):
    """OpenAI message roles - validated against real OpenAI API."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class OpenAIMessage(BaseModel):
    """OpenAI message structure - validated against real OpenAI API."""

    model_config = ConfigDict(extra='allow')

    role: MessageRole
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    refusal: str | None = None


class OpenAIChoice(BaseModel):
    """OpenAI choice structure - validated against real OpenAI API."""

    model_config = ConfigDict(extra='allow')

    index: int
    message: OpenAIMessage | None = None
    delta: dict[str, Any] | None = None
    finish_reason: Literal[
        "stop", "length", "function_call", "content_filter", "tool_calls",
    ] | None = None
    logprobs: dict[str, Any] | None = None


class OpenAIUsage(BaseModel):
    """OpenAI usage statistics - validated against real OpenAI API."""

    model_config = ConfigDict(extra='allow')

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: dict[str, Any] | None = None
    completion_tokens_details: dict[str, Any] | None = None

    # Cost extensions (optional, not in OpenAI spec)
    prompt_cost: float | None = None
    completion_cost: float | None = None
    total_cost: float | None = None


class OpenAIChatResponse(BaseModel):
    """Complete OpenAI chat response structure - validated against real OpenAI API."""

    model_config = ConfigDict(extra='allow')

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None


class OpenAIChatRequest(BaseModel):
    """OpenAI chat completion request - validated against real OpenAI API."""

    model_config = ConfigDict(extra='allow')

    model: str
    messages: list[dict[str, Any]]  # Will be validated as OpenAIMessage when needed
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool | None = False
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    seed: int | None = None
    service_tier: str | None = None
    stream_options: dict[str, Any] | None = None
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None

    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate that all messages have valid structure and roles."""
        for msg in v:
            # Validate each message as OpenAIMessage
            OpenAIMessage(**msg)
        return v

    @field_validator('stream')
    @classmethod
    def validate_streaming_only(cls, v: bool | None) -> bool:
        """Validate that streaming is enabled (streaming-only architecture)."""
        if v is False:
            raise ValueError(
                "Non-streaming requests are not supported. "
                "This API uses a streaming-only architecture. "
                "Set stream=true in your request.",
            )
        if v is None:
            raise ValueError(
                "The 'stream' parameter is required. "
                "Set stream=true (streaming-only architecture).",
            )
        return v


# Streaming models
class OpenAIDelta(BaseModel):
    """
    OpenAI streaming delta - validated against real OpenAI API.

    Enhanced with reasoning_event field to provide structured metadata
    about reasoning progress for smart clients while maintaining
    compatibility with standard OpenAI clients.
    """

    model_config = ConfigDict(extra='allow')

    role: MessageRole | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    reasoning_event: ReasoningEvent | None = None


class OpenAIStreamChoice(BaseModel):
    """OpenAI streaming choice - validated against real OpenAI API."""

    model_config = ConfigDict(extra='allow')

    index: int
    delta: OpenAIDelta
    finish_reason: Literal[
        "stop", "length", "function_call", "content_filter", "tool_calls",
    ] | None = None


class OpenAIStreamResponse(BaseModel):
    """OpenAI streaming response - validated against real OpenAI API."""

    model_config = ConfigDict(extra='allow')

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[OpenAIStreamChoice]
    usage: OpenAIUsage | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None


def convert_litellm_to_stream_response(
    chunk: "ModelResponseStream",
    completion_id: str | None = None,
    created: int | None = None,
) -> OpenAIStreamResponse:
    """
    Convert LiteLLM ModelResponseStream to OpenAIStreamResponse.

    Uses chunk.model_dump() + override pattern for clean conversion.
    Extra LiteLLM fields (citations, obfuscation, service_tier) are preserved
    via OpenAIStreamResponse's extra='allow' configuration.

    Validated against real LiteLLM chunks captured in scripts/litellm_chunks_captured.json.

    Args:
        chunk: LiteLLM ModelResponseStream object from litellm.acompletion()
        completion_id: Optional override for chunk.id (used by ReasoningAgent for consistent IDs)
        created: Optional override for chunk.created (used by ReasoningAgent for consistent
            timestamps)

    Returns:
        OpenAIStreamResponse with optional field overrides

    Example:
        # PassthroughExecutor (no overrides)
        response = convert_litellm_to_stream_response(chunk)

        # ReasoningAgent (consistent ID across chunks)
        response = convert_litellm_to_stream_response(
            chunk,
            completion_id="chatcmpl-reasoning123",
            created=1234567890,
        )
    """
    data = chunk.model_dump()

    # Override fields if provided (ReasoningAgent needs consistent IDs across all chunks)
    if completion_id is not None:
        data['id'] = completion_id
    if created is not None:
        data['created'] = created

    return OpenAIStreamResponse(**data)


# API metadata models
class ModelInfo(BaseModel):
    """Model information for OpenAI-compatible models."""

    model_config = ConfigDict(extra='allow')

    id: str
    object: str = "model"
    created: int
    owned_by: str

    # Context limits
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None

    # Pricing
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None

    # Capabilities
    supports_reasoning: bool | None = None
    supports_response_schema: bool | None = None
    supports_vision: bool | None = None
    supports_function_calling: bool | None = None
    supports_web_search: bool | None = None


class ModelsResponse(BaseModel):
    """Response model for listing available models."""

    model_config = ConfigDict(extra='allow')

    object: str = "list"
    data: list[ModelInfo]


class ErrorDetail(BaseModel):
    """Model for error details in API responses."""

    model_config = ConfigDict(extra='allow')

    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    """Model for error responses in the API."""

    model_config = ConfigDict(extra='allow')

    error: ErrorDetail


class OpenAIRequestBuilder:
    """
    Builds OpenAI chat completion requests that match their exact API specification.

    Usage:
        request = (OpenAIRequestBuilder()
                  .model("gpt-4o")
                  .message("user", "Hello")
                  .temperature(0.7)
                  .build())
    """

    def __init__(self):
        self.reset()

    def reset(self) -> 'OpenAIRequestBuilder':
        """Reset builder to initial state."""
        self._model = "gpt-4"
        self._messages = []
        self._max_tokens = None
        self._max_completion_tokens = None
        self._temperature = None
        self._top_p = None
        self._stream = False
        self._stream_options = None
        self._response_format = None
        self._tools = None
        self._tool_choice = None
        self._parallel_tool_calls = None
        self._user = None
        self._frequency_penalty = None
        self._presence_penalty = None
        self._logit_bias = None
        self._logprobs = None
        self._top_logprobs = None
        self._n = None
        self._seed = None
        self._service_tier = None
        self._stop = None
        return self

    def model(self, model: str) -> 'OpenAIRequestBuilder':
        """Set the model to use."""
        self._model = model
        return self

    def message(self, role: str, content: str, name: str | None = None) -> 'OpenAIRequestBuilder':
        """Add a message to the conversation."""
        message = {"role": role, "content": content}
        if name:
            message["name"] = name
        self._messages.append(message)
        return self

    def messages(self, messages: list[dict[str, Any]]) -> 'OpenAIRequestBuilder':
        """Set all messages at once."""
        self._messages = messages
        return self

    def max_tokens(self, tokens: int) -> 'OpenAIRequestBuilder':
        """Set maximum tokens in response (legacy parameter)."""
        self._max_tokens = tokens
        return self

    def max_completion_tokens(self, tokens: int) -> 'OpenAIRequestBuilder':
        """Set maximum completion tokens (new parameter)."""
        self._max_completion_tokens = tokens
        return self

    def temperature(self, temp: float) -> 'OpenAIRequestBuilder':
        """Set temperature (0.0 to 2.0)."""
        self._temperature = temp
        return self

    def top_p(self, p: float) -> 'OpenAIRequestBuilder':
        """Set top_p (nucleus sampling)."""
        self._top_p = p
        return self

    def streaming(self, include_usage: bool = False) -> 'OpenAIRequestBuilder':
        """Enable streaming responses."""
        self._stream = True
        if include_usage:
            self._stream_options = {"include_usage": True}
        return self

    def json_mode(self) -> 'OpenAIRequestBuilder':
        """Enable JSON mode response format."""
        self._response_format = {"type": "json_object"}
        return self

    def structured_output(self, schema: dict[str, Any]) -> 'OpenAIRequestBuilder':
        """Enable structured output with JSON schema."""
        self._response_format = {
            "type": "json_schema",
            "json_schema": schema,
        }
        return self

    def tools(
        self,
        tools: list[dict[str, Any]],
        choice: str | None = None,
    ) -> 'OpenAIRequestBuilder':
        """Add tools for function calling."""
        self._tools = tools
        if choice:
            self._tool_choice = choice
        return self

    def parallel_tool_calls(self, enabled: bool) -> 'OpenAIRequestBuilder':
        """Enable/disable parallel tool calls."""
        self._parallel_tool_calls = enabled
        return self

    def user(self, user_id: str) -> 'OpenAIRequestBuilder':
        """Set user identifier for abuse monitoring."""
        self._user = user_id
        return self

    def frequency_penalty(self, penalty: float) -> 'OpenAIRequestBuilder':
        """Set frequency penalty (-2.0 to 2.0)."""
        self._frequency_penalty = penalty
        return self

    def presence_penalty(self, penalty: float) -> 'OpenAIRequestBuilder':
        """Set presence penalty (-2.0 to 2.0)."""
        self._presence_penalty = penalty
        return self

    def seed(self, seed: int) -> 'OpenAIRequestBuilder':
        """Set seed for deterministic outputs."""
        self._seed = seed
        return self

    def stop(self, stop_sequences: str | list[str]) -> 'OpenAIRequestBuilder':
        """Set stop sequences."""
        self._stop = stop_sequences
        return self

    def build(self) -> OpenAIChatRequest:  # noqa: PLR0912
        """Build a validated OpenAI request using Pydantic model."""
        request_data = {
            "model": self._model,
            "messages": self._messages,
        }

        # Add optional parameters only if they were set
        if self._max_tokens is not None:
            request_data["max_tokens"] = self._max_tokens
        if self._max_completion_tokens is not None:
            request_data["max_completion_tokens"] = self._max_completion_tokens
        if self._temperature is not None:
            request_data["temperature"] = self._temperature
        if self._top_p is not None:
            request_data["top_p"] = self._top_p
        if self._stream:
            request_data["stream"] = True
            if self._stream_options:
                request_data["stream_options"] = self._stream_options
        if self._response_format:
            request_data["response_format"] = self._response_format
        if self._tools:
            request_data["tools"] = self._tools
            if self._tool_choice:
                request_data["tool_choice"] = self._tool_choice
        if self._parallel_tool_calls is not None:
            request_data["parallel_tool_calls"] = self._parallel_tool_calls
        if self._user:
            request_data["user"] = self._user
        if self._frequency_penalty is not None:
            request_data["frequency_penalty"] = self._frequency_penalty
        if self._presence_penalty is not None:
            request_data["presence_penalty"] = self._presence_penalty
        if self._seed is not None:
            request_data["seed"] = self._seed
        if self._stop is not None:
            request_data["stop"] = self._stop

        return OpenAIChatRequest(**request_data)


class OpenAIResponseBuilder:
    """
    Builds OpenAI chat completion responses that match their exact API specification.

    Usage:
        response = (
            OpenAIResponseBuilder()
            .id("chatcmpl-123")
            .model("gpt-4o")
            .choice(0, "assistant", "Hello!")
            .usage(10, 5)
            .build()
        )
    """

    def __init__(self):
        self.reset()

    def reset(self) -> 'OpenAIResponseBuilder':
        """Reset builder to initial state."""
        self._id = None
        self._object = OpenAIObjectType.CHAT_COMPLETION.value
        self._created = None
        self._model = None
        self._choices = []
        self._usage = None
        self._system_fingerprint = None
        self._service_tier = None
        return self

    def id(self, completion_id: str) -> 'OpenAIResponseBuilder':
        """Set the completion ID."""
        self._id = completion_id
        return self

    def model(self, model: str) -> 'OpenAIResponseBuilder':
        """Set the model name."""
        self._model = model
        return self

    def created(self, timestamp: int) -> 'OpenAIResponseBuilder':
        """Set creation timestamp."""
        self._created = timestamp
        return self

    def choice(self, index: int, role: str, content: str,
               finish_reason: str = "stop", refusal: str | None = None) -> 'OpenAIResponseBuilder':
        """Add a choice to the response."""
        message = {
            "role": role,
            "content": content,
        }
        if refusal:
            message["refusal"] = refusal

        choice = {
            "index": index,
            "message": message,
            "finish_reason": finish_reason,
            "logprobs": None,
        }
        self._choices.append(choice)
        return self

    def choice_with_tool_calls(self, index: int, role: str, content: str | None,
                              tool_calls: list[dict[str, Any]],
                              finish_reason: str = "tool_calls") -> 'OpenAIResponseBuilder':
        """Add a choice with tool calls."""
        message = {
            "role": role,
            "content": content,
            "tool_calls": tool_calls,
        }

        choice = {
            "index": index,
            "message": message,
            "finish_reason": finish_reason,
            "logprobs": None,
        }
        self._choices.append(choice)
        return self

    def usage(self, prompt_tokens: int, completion_tokens: int) -> 'OpenAIResponseBuilder':
        """Add usage information."""
        self._usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": None,
            "completion_tokens_details": None,
        }
        return self

    def system_fingerprint(self, fingerprint: str) -> 'OpenAIResponseBuilder':
        """Set system fingerprint."""
        self._system_fingerprint = fingerprint
        return self

    def service_tier(self, tier: str) -> 'OpenAIResponseBuilder':
        """Set service tier."""
        self._service_tier = tier
        return self

    def build(self) -> OpenAIChatResponse:
        """Build a validated OpenAI response using Pydantic model."""
        response_data = {
            "id": self._id,
            "object": self._object,
            "created": self._created,
            "model": self._model,
            "choices": self._choices,
        }

        if self._usage:
            response_data["usage"] = self._usage
        if self._system_fingerprint:
            response_data["system_fingerprint"] = self._system_fingerprint
        if self._service_tier:
            response_data["service_tier"] = self._service_tier

        return OpenAIChatResponse(**response_data)


class OpenAIStreamingResponseBuilder:
    """
    Builds OpenAI streaming responses that match their exact SSE format.

    Usage:
        stream = (
            OpenAIStreamingResponseBuilder()
            .chunk("chatcmpl-123", "gpt-4o", delta_content="Hello")
            .chunk("chatcmpl-123", "gpt-4o", delta_content=" world")
            .chunk("chatcmpl-123", "gpt-4o", finish_reason="stop")
            .done()
            .build()
        )
    """

    def __init__(self):
        self._chunks = []

    def chunk(self, completion_id: str, model: str, index: int = 0,
              delta_role: str | None = None,
              delta_content: str | None = None,
              delta_tool_calls: list[dict[str, Any]] | None = None,
              finish_reason: str | None = None,
              usage: dict[str, Any] | None = None) -> 'OpenAIStreamingResponseBuilder':
        """Add a streaming chunk."""
        delta = {}
        if delta_role:
            delta["role"] = delta_role
        if delta_content is not None:
            delta["content"] = delta_content
        if delta_tool_calls:
            delta["tool_calls"] = delta_tool_calls

        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": index,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
            "system_fingerprint": None,
            "service_tier": None,
            "usage": usage,
        }

        self._chunks.append(create_sse(chunk))
        return self

    def done(self) -> 'OpenAIStreamingResponseBuilder':
        """Add the [DONE] marker."""
        self._chunks.append(SSE_DONE)
        return self

    def build(self) -> str:
        """Build the complete SSE stream."""
        return "".join(self._chunks)


class OpenAIResponseParser:
    """Parse and validate OpenAI responses to ensure they match the expected format."""

    @staticmethod
    def parse_streaming_chunk(chunk_line: str) -> dict[str, Any] | None:
        """Parse a single SSE chunk from OpenAI streaming."""
        if not chunk_line.startswith("data: "):
            return None

        data_part = chunk_line[6:].strip()
        if data_part == "[DONE]":
            return {"done": True}

        try:
            chunk_data = json.loads(data_part)

            # Validate chunk structure
            required_fields = ["id", "object", "created", "model", "choices"]
            for field in required_fields:
                if field not in chunk_data:
                    raise ValueError(f"Missing required field in chunk: {field}")

            return chunk_data

        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Invalid streaming chunk format: {e}")


def extract_system_message(messages: list[dict[str, object]]) -> str | None:
    """
    Extract the first system message from the messages list.

    Args:
        messages: List of message dictionaries with 'role' and 'content' fields

    Returns:
        System message content if found, None otherwise

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "Hello"}
        ... ]
        >>> extract_system_message(messages)
        'You are a helpful assistant.'
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    if not system_msgs:
        return None
    # Get content from first system message
    content = system_msgs[0].get("content")
    return str(content) if content is not None else None


def generate_title_from_messages(
    messages: list[dict[str, object]],
    max_length: int = 100,
) -> str | None:
    r"""
    Generate a conversation title from the first user message.

    Extracts content from the first user message, strips whitespace and newlines,
    and truncates to max_length characters with ellipsis if needed.

    Args:
        messages: List of message dictionaries with 'role' and 'content' fields
        max_length: Maximum length for the title (default: 100)

    Returns:
        Generated title string, or None if no user message found or content is empty

    Examples:
        >>> messages = [{"role": "user", "content": "What is the weather?"}]
        >>> generate_title_from_messages(messages)
        'What is the weather?'

        >>> messages = [{"role": "user", "content": "This is a very long message " * 10}]
        >>> title = generate_title_from_messages(messages, max_length=50)
        >>> len(title) <= 50
        True
        >>> title.endswith("...")
        True

        >>> messages = [{"role": "system", "content": "You are helpful."}]
        >>> generate_title_from_messages(messages)
        None

        >>> messages = [{"role": "user", "content": "  \n  Hello  \n  "}]
        >>> generate_title_from_messages(messages)
        'Hello'
    """
    # Find first user message
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return None

    # Get content from first user message
    content = user_msgs[0].get("content")
    if content is None:
        return None

    # Convert to string and clean up whitespace
    title = str(content).strip()

    # Replace newlines and multiple spaces with single space
    title = " ".join(title.split())

    # Return None if empty after cleanup
    if not title:
        return None

    # Truncate if needed
    if len(title) > max_length:
        title = title[: max_length - 3] + "..."

    return title
