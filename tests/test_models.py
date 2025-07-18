"""
Comprehensive tests for Pydantic models.

Tests serialization, validation, OpenAI compatibility, and Pydantic v2 features.
"""


import pytest
from pydantic import ValidationError

from api.openai_protocol import (
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIStreamResponse,
    OpenAIMessage,
    MessageRole,
    OpenAIStreamChoice,
    OpenAIDelta,
    ModelInfo,
    ModelsResponse,
    ErrorResponse,
    ErrorDetail,
)
from tests.conftest import OPENAI_TEST_MODEL


class TestOpenAIMessage:
    """Test OpenAIMessage model."""

    def test__valid_chat_message__creates_successfully(self) -> None:
        """Test that valid chat message is created successfully."""
        message = OpenAIMessage(role=MessageRole.USER, content="Hello, world!")

        assert message.role == MessageRole.USER
        assert message.content == "Hello, world!"

    def test__all_message_roles__are_supported(self) -> None:
        """Test that all message roles are supported."""
        roles = [MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT]

        for role in roles:
            message = OpenAIMessage(role=role, content="Test content")
            assert message.role == role

    def test__message_serialization__produces_correct_json(self) -> None:
        """Test that message serialization produces correct JSON."""
        message = OpenAIMessage(role=MessageRole.ASSISTANT, content="AI response")
        serialized = message.model_dump()

        # Check that the essential fields are correct
        assert serialized["role"] == "assistant"
        assert serialized["content"] == "AI response"
        # OpenAI protocol includes optional fields in serialization
        assert "name" in serialized
        assert "tool_calls" in serialized
        assert "tool_call_id" in serialized
        assert "refusal" in serialized

    def test__message_deserialization__from_dict(self) -> None:
        """Test that message can be deserialized from dict."""
        data = {
            "role": "user",
            "content": "User message",
        }

        message = OpenAIMessage.model_validate(data)
        assert message.role == MessageRole.USER
        assert message.content == "User message"

    def test__invalid_role__raises_validation_error(self) -> None:
        """Test that invalid role raises validation error."""
        with pytest.raises(ValidationError):
            OpenAIMessage(role="invalid_role", content="Test")

    def test__extra_fields_allowed__works_correctly(self) -> None:
        """Test that extra fields are allowed for OpenAI compatibility."""
        # Should not raise an error now that we allow extra fields
        message = OpenAIMessage.model_validate({
            "role": "user",
            "content": "Test",
            "extra_field": "allowed for OpenAI compatibility",
        })
        assert message.role == MessageRole.USER
        assert message.content == "Test"


class TestOpenAIChatRequest:
    """Test OpenAIChatRequest model."""

    def test__minimal_request__creates_successfully(self) -> None:
        """Test that minimal request creates successfully."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert request.model == "gpt-4o"
        assert len(request.messages) == 1
        assert request.temperature is None  # No default value in OpenAI protocol
        assert request.stream is False  # Default value

    def test__full_request__creates_successfully(self) -> None:
        """Test that request with all fields creates successfully."""
        request = OpenAIChatRequest(
            model=OPENAI_TEST_MODEL,
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ],
            max_tokens=150,
            temperature=0.7,
            top_p=0.9,
            n=2,
            stream=True,
            stop=["END"],
            presence_penalty=0.1,
            frequency_penalty=0.2,
            logit_bias={"-50256": -100},
            user="test-user",
        )

        assert request.model == OPENAI_TEST_MODEL
        assert len(request.messages) == 2
        assert request.max_tokens == 150
        assert request.temperature == 0.7
        assert request.stream is True
        assert request.stop == ["END"]

    def test__request_serialization__excludes_unset_fields(self) -> None:
        """Test that serialization excludes unset fields."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )

        serialized = request.model_dump(exclude_unset=True)

        # Should only include set fields
        expected_keys = {"model", "messages"}
        assert set(serialized.keys()) == expected_keys

    def test__request_serialization__includes_all_when_not_excluding_unset(self) -> None:
        """Test that serialization includes all fields when not excluding unset."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )

        serialized = request.model_dump()

        # Should include all fields with their default values
        assert "temperature" in serialized
        assert "stream" in serialized
        assert serialized["temperature"] is None
        assert serialized["stream"] is False

    def test__openai_compatible_serialization(self) -> None:
        """Test that serialization is compatible with OpenAI API format."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What's 2+2?"}],
            temperature=0.5,
            max_tokens=100,
        )

        serialized = request.model_dump(exclude_unset=True)

        # Should match OpenAI API format
        expected = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "What's 2+2?"}],
            "temperature": 0.5,
            "max_tokens": 100,
        }
        assert serialized == expected


class TestOpenAIChatResponse:
    """Test OpenAIChatResponse model."""

    def test__valid_response__creates_successfully(self) -> None:
        """Test that valid response creates successfully."""
        response_data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello there!"},
                    "finish_reason": "stop",
                },
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        response = OpenAIChatResponse.model_validate(response_data)

        assert response.id == "chatcmpl-123"
        assert response.model == "gpt-4o"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello there!"
        assert response.usage.total_tokens == 15

    def test__real_openai_response_format__parses_correctly(self) -> None:
        """Test that real OpenAI response format parses correctly."""
        # This is based on actual OpenAI API response format
        openai_response = {
            "id": "chatcmpl-8ZrXqYqABTyQSPjmz5HN7TVyJKP8p",
            "object": "chat.completion",
            "created": 1703097234,
            "model": "gpt-4-0613",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I'm Claude, an AI assistant created by Anthropic. How can I help you today?",  # noqa: E501
                    },
                    "finish_reason": "stop",
                },
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 17,
                "total_tokens": 29,
            },
        }

        response = OpenAIChatResponse.model_validate(openai_response)
        assert response.id == "chatcmpl-8ZrXqYqABTyQSPjmz5HN7TVyJKP8p"
        assert response.model == "gpt-4-0613"


class TestStreamingModels:
    """Test streaming-related models."""

    def test__delta__creates_with_partial_content(self) -> None:
        """Test that OpenAIDelta creates with partial content."""
        delta = OpenAIDelta(content="Hello")
        assert delta.content == "Hello"
        assert delta.role is None

    def test__delta__creates_with_role_only(self) -> None:
        """Test that OpenAIDelta creates with role only."""
        delta = OpenAIDelta(role=MessageRole.ASSISTANT)
        assert delta.role == MessageRole.ASSISTANT
        assert delta.content is None

    def test__stream_choice__creates_successfully(self) -> None:
        """Test that OpenAIStreamChoice creates successfully."""
        choice = OpenAIStreamChoice(
            index=0,
            delta=OpenAIDelta(content="test"),
            finish_reason=None,
        )

        assert choice.index == 0
        assert choice.delta.content == "test"
        assert choice.finish_reason is None

    def test__stream_response__matches_openai_format(self) -> None:
        """Test that streaming response matches OpenAI format."""
        response = OpenAIStreamResponse(
            id="chatcmpl-test",
            created=1234567890,
            model="gpt-4o",
            choices=[
                OpenAIStreamChoice(
                    index=0,
                    delta=OpenAIDelta(content="Hello"),
                    finish_reason=None,
                ),
            ],
        )

        serialized = response.model_dump()

        assert serialized["object"] == "chat.completion.chunk"
        assert "choices" in serialized
        assert serialized["choices"][0]["delta"]["content"] == "Hello"

    def test__real_openai_streaming_chunk__parses_correctly(self) -> None:
        """Test that real OpenAI streaming chunk parses correctly."""
        # Based on actual OpenAI streaming format
        chunk_data = {
            "id": "chatcmpl-8ZrXqYqABTyQSPjmz5HN7TVyJKP8p",
            "object": "chat.completion.chunk",
            "created": 1703097234,
            "model": "gpt-4-0613",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": None,
                },
            ],
        }

        chunk = OpenAIStreamResponse.model_validate(chunk_data)
        assert chunk.id == "chatcmpl-8ZrXqYqABTyQSPjmz5HN7TVyJKP8p"
        assert chunk.choices[0].delta.content == "Hello"


class TestModelsResponse:
    """Test ModelsResponse model."""

    def test__models_response__creates_successfully(self) -> None:
        """Test that models response creates successfully."""
        response = ModelsResponse(
            data=[
                ModelInfo(
                    id="gpt-4o",
                    created=1234567890,
                    owned_by="openai",
                ),
                ModelInfo(
                    id=OPENAI_TEST_MODEL,
                    created=1234567890,
                    owned_by="openai",
                ),
            ],
        )

        assert response.object == "list"
        assert len(response.data) == 2
        assert response.data[0].id == "gpt-4o"

    def test__models_response__matches_openai_format(self) -> None:
        """Test that models response matches OpenAI format."""
        response = ModelsResponse(
            data=[ModelInfo(id="gpt-4o", created=1234567890, owned_by="openai")],
        )

        serialized = response.model_dump()
        expected = {
            "object": "list",
            "data": [
                {
                    "id": "gpt-4o",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "openai",
                },
            ],
        }

        assert serialized == expected


class TestErrorModels:
    """Test error-related models."""

    def test__error_detail__creates_successfully(self) -> None:
        """Test that error detail creates successfully."""
        error = ErrorDetail(
            message="Invalid API key",
            type="invalid_request_error",
            code="invalid_api_key",
        )

        assert error.message == "Invalid API key"
        assert error.type == "invalid_request_error"
        assert error.code == "invalid_api_key"

    def test__error_response__creates_successfully(self) -> None:
        """Test that error response creates successfully."""
        error_response = ErrorResponse(
            error=ErrorDetail(
                message="Rate limit exceeded",
                type="rate_limit_error",
            ),
        )

        assert error_response.error.message == "Rate limit exceeded"
        assert error_response.error.type == "rate_limit_error"

    def test__error_response__matches_openai_format(self) -> None:
        """Test that error response matches OpenAI format."""
        error_response = ErrorResponse(
            error=ErrorDetail(
                message="Invalid API key provided",
                type="invalid_request_error",
                param=None,
                code="invalid_api_key",
            ),
        )

        serialized = error_response.model_dump()

        # Should match OpenAI error format
        assert "error" in serialized
        assert serialized["error"]["message"] == "Invalid API key provided"
        assert serialized["error"]["type"] == "invalid_request_error"


class TestPydanticV2Features:
    """Test Pydantic v2 specific features."""

    def test__model_config__allows_extra_fields_for_openai_compatibility(self) -> None:
        """Test that model config allows extra fields for OpenAI compatibility."""
        # Should not raise an error - extra fields are now allowed
        message = OpenAIMessage.model_validate({
            "role": "user",
            "content": "test",
            "extra_field": "allowed for OpenAI API compatibility",
        })
        assert message.role == MessageRole.USER
        assert message.content == "test"

    def test__model_dump_excludes_unset__works_correctly(self) -> None:
        """Test that exclude_unset works correctly in Pydantic v2."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.5,
        )

        # With exclude_unset=True, should only include explicitly set fields
        with_exclude = request.model_dump(exclude_unset=True)
        assert "temperature" in with_exclude
        assert "max_tokens" not in with_exclude  # Not explicitly set

        # Without exclude_unset, should include all fields with defaults
        without_exclude = request.model_dump()
        assert "max_tokens" in without_exclude

    def test__model_validate__works_with_pydantic_v2(self) -> None:
        """Test that model_validate works correctly with Pydantic v2."""
        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "test"}],
        }

        request = OpenAIChatRequest.model_validate(data)
        assert request.model == "gpt-4o"
        assert len(request.messages) == 1
