"""
OpenAI API Protocol Abstraction.

This module provides builders and parsers that match OpenAI's exact API specification.
All mocks and real API calls must use these to ensure consistency.

Based on OpenAI API documentation:
- https://platform.openai.com/docs/api-reference/chat
- https://platform.openai.com/docs/guides/structured-outputs
- https://platform.openai.com/docs/api-reference/chat-streaming
"""

import json
import time
from typing import Any
from dataclasses import dataclass
from enum import Enum


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


class OpenAIRole(Enum):
    """OpenAI message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class OpenAIMessage:
    """OpenAI message structure."""

    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    refusal: str | None = None


@dataclass
class OpenAIChoice:
    """OpenAI choice structure."""

    index: int
    message: OpenAIMessage | None = None
    delta: dict[str, Any] | None = None
    finish_reason: str | None = None
    logprobs: dict[str, Any] | None = None


@dataclass
class OpenAIUsage:
    """OpenAI usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: dict[str, Any] | None = None
    completion_tokens_details: dict[str, Any] | None = None


@dataclass
class OpenAIChatResponse:
    """Complete OpenAI chat response structure."""

    id: str
    object: str
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None


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

    def build(self) -> dict[str, Any]:  # noqa: PLR0912
        """Build the actual request dict that OpenAI expects."""
        request = {
            "model": self._model,
            "messages": self._messages,
        }

        # Add optional parameters only if they were set
        if self._max_tokens is not None:
            request["max_tokens"] = self._max_tokens
        if self._max_completion_tokens is not None:
            request["max_completion_tokens"] = self._max_completion_tokens
        if self._temperature is not None:
            request["temperature"] = self._temperature
        if self._top_p is not None:
            request["top_p"] = self._top_p
        if self._stream:
            request["stream"] = True
            if self._stream_options:
                request["stream_options"] = self._stream_options
        if self._response_format:
            request["response_format"] = self._response_format
        if self._tools:
            request["tools"] = self._tools
            if self._tool_choice:
                request["tool_choice"] = self._tool_choice
        if self._parallel_tool_calls is not None:
            request["parallel_tool_calls"] = self._parallel_tool_calls
        if self._user:
            request["user"] = self._user
        if self._frequency_penalty is not None:
            request["frequency_penalty"] = self._frequency_penalty
        if self._presence_penalty is not None:
            request["presence_penalty"] = self._presence_penalty
        if self._seed is not None:
            request["seed"] = self._seed
        if self._stop is not None:
            request["stop"] = self._stop

        return request


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

    def build(self) -> dict[str, Any]:
        """Build the actual response dict that matches OpenAI's format."""
        response = {
            "id": self._id,
            "object": self._object,
            "created": self._created,
            "model": self._model,
            "choices": self._choices,
        }

        if self._usage:
            response["usage"] = self._usage
        if self._system_fingerprint:
            response["system_fingerprint"] = self._system_fingerprint
        if self._service_tier:
            response["service_tier"] = self._service_tier

        return response


class OpenAIStreamingResponseBuilder:
    """
    Builds OpenAI streaming responses that match their exact SSE format.

    Usage:
        stream = (OpenAIStreamingResponseBuilder()
                 .chunk("chatcmpl-123", "gpt-4o", delta_content="Hello")
                 .chunk("chatcmpl-123", "gpt-4o", delta_content=" world")
                 .chunk("chatcmpl-123", "gpt-4o", finish_reason="stop")
                 .done()
                 .build())
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

        self._chunks.append(f"data: {json.dumps(chunk)}\n\n")
        return self

    def done(self) -> 'OpenAIStreamingResponseBuilder':
        """Add the [DONE] marker."""
        self._chunks.append("data: [DONE]\n\n")
        return self

    def build(self) -> str:
        """Build the complete SSE stream."""
        return "".join(self._chunks)


class OpenAIResponseParser:
    """Parse and validate OpenAI responses to ensure they match the expected format."""

    @staticmethod
    def parse_chat_response(response_data: dict[str, Any]) -> OpenAIChatResponse:
        """Parse and validate a chat completion response."""
        try:
            # Validate required fields exist
            required_fields = ["id", "object", "created", "model", "choices"]
            for field in required_fields:
                if field not in response_data:
                    raise ValueError(f"Missing required field: {field}")

            # Parse choices
            choices = []
            for choice_data in response_data["choices"]:
                message = None
                if "message" in choice_data:
                    msg_data = choice_data["message"]
                    message = OpenAIMessage(
                        role=msg_data["role"],
                        content=msg_data.get("content"),
                        name=msg_data.get("name"),
                        tool_calls=msg_data.get("tool_calls"),
                        tool_call_id=msg_data.get("tool_call_id"),
                        refusal=msg_data.get("refusal"),
                    )

                choice = OpenAIChoice(
                    index=choice_data["index"],
                    message=message,
                    delta=choice_data.get("delta"),
                    finish_reason=choice_data.get("finish_reason"),
                    logprobs=choice_data.get("logprobs"),
                )
                choices.append(choice)

            # Parse usage if present
            usage = None
            if "usage" in response_data:
                usage_data = response_data["usage"]
                usage = OpenAIUsage(
                    prompt_tokens=usage_data["prompt_tokens"],
                    completion_tokens=usage_data["completion_tokens"],
                    total_tokens=usage_data["total_tokens"],
                    prompt_tokens_details=usage_data.get("prompt_tokens_details"),
                    completion_tokens_details=usage_data.get("completion_tokens_details"),
                )

            return OpenAIChatResponse(
                id=response_data["id"],
                object=response_data["object"],
                created=response_data["created"],
                model=response_data["model"],
                choices=choices,
                usage=usage,
                system_fingerprint=response_data.get("system_fingerprint"),
                service_tier=response_data.get("service_tier"),
            )

        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid OpenAI response format: {e}")

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

    @staticmethod
    def validate_request(request_data: dict[str, Any]) -> dict[str, Any]:
        """Validate a request matches OpenAI's expected format."""
        # Required fields
        if "model" not in request_data:
            raise ValueError("Missing required field: model")
        if "messages" not in request_data:
            raise ValueError("Missing required field: messages")

        # Validate messages format
        for i, message in enumerate(request_data["messages"]):
            if not isinstance(message, dict):
                raise ValueError(f"Message {i} must be a dict")
            if "role" not in message:
                raise ValueError(f"Message {i} missing required field: role")
            if message["role"] not in ["system", "user", "assistant", "tool"]:
                raise ValueError(f"Message {i} has invalid role: {message['role']}")

        # Validate optional fields have correct types
        optional_validations = {
            "max_tokens": int,
            "max_completion_tokens": int,
            "temperature": (int, float),
            "top_p": (int, float),
            "stream": bool,
            "n": int,
            "frequency_penalty": (int, float),
            "presence_penalty": (int, float),
            "seed": int,
            "user": str,
        }

        for field, expected_type in optional_validations.items():
            if field in request_data and not isinstance(request_data[field], expected_type):
                raise ValueError(f"Field {field} must be of type {expected_type}")

        return request_data
