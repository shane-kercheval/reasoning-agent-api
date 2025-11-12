"""Unit tests for OpenAI protocol utilities."""

from unittest.mock import Mock
import pytest
from pydantic import ValidationError

from api.openai_protocol import (
    extract_system_message,
    convert_litellm_to_stream_response,
    generate_title_from_messages,
    OpenAIChatRequest,
)
from api.reasoning_models import ReasoningEventType, ReasoningEvent


class TestExtractSystemMessage:
    """Tests for extract_system_message function."""

    def test__extract_system_message__returns_content_when_present(self) -> None:
        """Test extracting system message when present."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        result = extract_system_message(messages)

        assert result == "You are a helpful assistant."

    def test__extract_system_message__returns_none_when_absent(self) -> None:
        """Test extracting system message when not present."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = extract_system_message(messages)

        assert result is None

    def test__extract_system_message__returns_first_when_multiple(self) -> None:
        """Test extracting first system message when multiple exist."""
        messages = [
            {"role": "system", "content": "First system message"},
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Second system message"},
        ]

        result = extract_system_message(messages)

        assert result == "First system message"

    def test__extract_system_message__handles_empty_list(self) -> None:
        """Test extracting system message from empty list."""
        messages = []

        result = extract_system_message(messages)

        assert result is None

    def test__extract_system_message__handles_none_content(self) -> None:
        """Test extracting system message when content is None."""
        messages = [
            {"role": "system", "content": None},
            {"role": "user", "content": "Hello"},
        ]

        result = extract_system_message(messages)

        assert result is None

    def test__extract_system_message__converts_to_string(self) -> None:
        """Test that non-string content is converted to string."""
        messages = [
            {"role": "system", "content": 123},  # Non-string content
            {"role": "user", "content": "Hello"},
        ]

        result = extract_system_message(messages)

        assert result == "123"
        assert isinstance(result, str)

    def test__extract_system_message__handles_missing_content_field(self) -> None:
        """Test extracting system message when content field is missing."""
        messages = [
            {"role": "system"},  # No content field
            {"role": "user", "content": "Hello"},
        ]

        result = extract_system_message(messages)

        assert result is None

    def test__extract_system_message__ignores_non_system_roles(self) -> None:
        """Test that only system role messages are extracted."""
        messages = [
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant message"},
            {"role": "tool", "content": "Tool message"},
        ]

        result = extract_system_message(messages)

        assert result is None


class TestConvertLitellmToStreamResponse:
    """
    Tests for convert_litellm_to_stream_response function.

    These tests use real LiteLLM chunk structure captured from
    scripts/litellm_chunks_captured.json to ensure accuracy.
    """

    def test__convert__content_chunk(self) -> None:
        """Test converting LiteLLM content chunk."""
        # Mock LiteLLM ModelResponseStream with real structure
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
                    "content": "Hello",
                    "role": "assistant",
                    "tool_calls": None,
                },
                "finish_reason": None,
            }],
            "usage": None,
        }

        result = convert_litellm_to_stream_response(mock_chunk)

        assert result.id == "chatcmpl-test123"
        assert result.created == 1234567890
        assert result.model == "gpt-4o-mini"
        assert result.choices[0].delta.content == "Hello"
        assert result.choices[0].delta.role == "assistant"
        assert result.usage is None

    def test__convert__finish_chunk_with_none_content(self) -> None:
        """Test converting finish chunk where delta.content is None."""
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
                    "content": None,  # No content in finish chunk
                    "role": None,
                },
                "finish_reason": "stop",
            }],
            "usage": None,
        }

        result = convert_litellm_to_stream_response(mock_chunk)

        assert result.choices[0].delta.content is None
        assert result.choices[0].finish_reason == "stop"
        assert result.usage is None

    def test__convert__usage_chunk(self) -> None:
        """Test converting usage chunk with usage data."""
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
                    "content": None,  # No content in usage chunk
                    "role": None,
                },
                "finish_reason": None,
            }],
            "usage": {
                "completion_tokens": 8,
                "prompt_tokens": 18,
                "total_tokens": 26,
            },
        }

        result = convert_litellm_to_stream_response(mock_chunk)

        assert result.usage is not None
        assert result.usage.completion_tokens == 8
        assert result.usage.prompt_tokens == 18
        assert result.usage.total_tokens == 26
        assert result.choices[0].delta.content is None

    def test__convert__with_id_override(self) -> None:
        """Test overriding chunk ID (for ReasoningAgent consistency)."""
        mock_chunk = Mock()
        mock_chunk.model_dump.return_value = {
            "id": "chatcmpl-original",
            "created": 1234567890,
            "model": "gpt-4o-mini",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello", "role": "assistant"},
                "finish_reason": None,
            }],
        }

        result = convert_litellm_to_stream_response(
            mock_chunk,
            completion_id="chatcmpl-reasoning123",
        )

        assert result.id == "chatcmpl-reasoning123"  # Overridden
        assert result.created == 1234567890  # Not overridden

    def test__convert__with_created_override(self) -> None:
        """Test overriding chunk timestamp (for ReasoningAgent consistency)."""
        mock_chunk = Mock()
        mock_chunk.model_dump.return_value = {
            "id": "chatcmpl-test",
            "created": 1111111111,
            "model": "gpt-4o-mini",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello", "role": "assistant"},
                "finish_reason": None,
            }],
        }

        result = convert_litellm_to_stream_response(
            mock_chunk,
            created=9999999999,
        )

        assert result.id == "chatcmpl-test"  # Not overridden
        assert result.created == 9999999999  # Overridden

    def test__convert__with_both_overrides(self) -> None:
        """Test overriding both ID and timestamp."""
        mock_chunk = Mock()
        mock_chunk.model_dump.return_value = {
            "id": "chatcmpl-original",
            "created": 1111111111,
            "model": "gpt-4o-mini",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello", "role": "assistant"},
                "finish_reason": None,
            }],
        }

        result = convert_litellm_to_stream_response(
            mock_chunk,
            completion_id="chatcmpl-override",
            created=9999999999,
        )

        assert result.id == "chatcmpl-override"
        assert result.created == 9999999999

    def test__convert__preserves_extra_litellm_fields(self) -> None:
        """Test that extra LiteLLM fields are preserved via extra='allow'."""
        mock_chunk = Mock()
        mock_chunk.model_dump.return_value = {
            "id": "chatcmpl-test",
            "created": 1234567890,
            "model": "gpt-4o-mini",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": None,
            }],
            # Extra LiteLLM-specific fields
            "citations": None,
            "service_tier": "default",
            "obfuscation": "test123",
            "provider_specific_fields": None,
        }

        result = convert_litellm_to_stream_response(mock_chunk)

        # Extra fields should be preserved (not lost in conversion)
        assert result.id == "chatcmpl-test"
        # Note: Extra fields are stored in model's __pydantic_extra__ due to extra='allow'
        # but not directly accessible - this confirms no error is raised


class TestGenerateTitleFromMessages:
    """Tests for generate_title_from_messages function."""

    def test__generate_title__returns_content_from_user_message(self) -> None:
        """Test generating title from simple user message."""
        messages = [
            {"role": "user", "content": "What is the weather today?"},
        ]

        result = generate_title_from_messages(messages)

        assert result == "What is the weather today?"

    def test__generate_title__uses_first_user_message(self) -> None:
        """Test that only the first user message is used for title."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "First user message"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second user message"},
        ]

        result = generate_title_from_messages(messages)

        assert result == "First user message"

    def test__generate_title__returns_none_when_no_user_message(self) -> None:
        """Test that None is returned when no user message exists."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "assistant", "content": "Hello"},
        ]

        result = generate_title_from_messages(messages)

        assert result is None

    def test__generate_title__returns_none_for_empty_list(self) -> None:
        """Test that None is returned for empty message list."""
        messages = []

        result = generate_title_from_messages(messages)

        assert result is None

    def test__generate_title__returns_none_when_content_is_none(self) -> None:
        """Test that None is returned when user message content is None."""
        messages = [
            {"role": "user", "content": None},
        ]

        result = generate_title_from_messages(messages)

        assert result is None

    def test__generate_title__returns_none_when_content_is_empty_string(self) -> None:
        """Test that None is returned when content is empty string."""
        messages = [
            {"role": "user", "content": ""},
        ]

        result = generate_title_from_messages(messages)

        assert result is None

    def test__generate_title__returns_none_when_content_is_whitespace_only(self) -> None:
        """Test that None is returned when content is only whitespace."""
        messages = [
            {"role": "user", "content": "   \n\n   \t  "},
        ]

        result = generate_title_from_messages(messages)

        assert result is None

    def test__generate_title__strips_leading_and_trailing_whitespace(self) -> None:
        """Test that leading and trailing whitespace is stripped."""
        messages = [
            {"role": "user", "content": "  Hello world  "},
        ]

        result = generate_title_from_messages(messages)

        assert result == "Hello world"

    def test__generate_title__replaces_newlines_with_spaces(self) -> None:
        """Test that newlines are replaced with single spaces."""
        messages = [
            {"role": "user", "content": "First line\nSecond line\nThird line"},
        ]

        result = generate_title_from_messages(messages)

        assert result == "First line Second line Third line"
        assert "\n" not in result

    def test__generate_title__normalizes_multiple_spaces(self) -> None:
        """Test that multiple consecutive spaces are normalized to single space."""
        messages = [
            {"role": "user", "content": "Hello    world     test"},
        ]

        result = generate_title_from_messages(messages)

        assert result == "Hello world test"

    def test__generate_title__handles_mixed_whitespace(self) -> None:
        """Test that mixed whitespace (tabs, newlines, spaces) is normalized."""
        messages = [
            {"role": "user", "content": "Hello\t\nworld  \n  test"},
        ]

        result = generate_title_from_messages(messages)

        assert result == "Hello world test"

    def test__generate_title__truncates_long_messages_default(self) -> None:
        """Test that long messages are truncated to 100 chars with ellipsis by default."""
        long_message = "A" * 150  # 150 characters
        messages = [
            {"role": "user", "content": long_message},
        ]

        result = generate_title_from_messages(messages)

        assert result is not None
        assert len(result) == 100
        assert result.endswith("...")
        assert result.startswith("A" * 97)  # 97 A's + "..." = 100 chars

    def test__generate_title__truncates_with_custom_max_length(self) -> None:
        """Test that custom max_length parameter is respected."""
        long_message = "This is a longer message that should be truncated"
        messages = [
            {"role": "user", "content": long_message},
        ]

        result = generate_title_from_messages(messages, max_length=20)

        assert result is not None
        assert len(result) == 20
        assert result.endswith("...")
        # First 17 chars of "This is a longer message..." + "..." = 20 chars
        assert result == "This is a longer ..."

    def test__generate_title__does_not_truncate_short_messages(self) -> None:
        """Test that messages shorter than max_length are not truncated."""
        messages = [
            {"role": "user", "content": "Short message"},
        ]

        result = generate_title_from_messages(messages, max_length=100)

        assert result == "Short message"
        assert not result.endswith("...")

    def test__generate_title__handles_message_exactly_at_max_length(self) -> None:
        """Test that messages exactly at max_length are not truncated."""
        message = "A" * 100  # Exactly 100 characters
        messages = [
            {"role": "user", "content": message},
        ]

        result = generate_title_from_messages(messages, max_length=100)

        assert result == message
        assert not result.endswith("...")

    def test__generate_title__converts_non_string_content_to_string(self) -> None:
        """Test that non-string content is converted to string."""
        messages = [
            {"role": "user", "content": 12345},
        ]

        result = generate_title_from_messages(messages)

        assert result == "12345"
        assert isinstance(result, str)

    def test__generate_title__handles_missing_content_field(self) -> None:
        """Test that missing content field returns None."""
        messages = [
            {"role": "user"},  # No content field
        ]

        result = generate_title_from_messages(messages)

        assert result is None

    def test__generate_title__ignores_system_and_assistant_messages(self) -> None:
        """Test that only user messages are considered for title."""
        messages = [
            {"role": "system", "content": "System instruction"},
            {"role": "assistant", "content": "Assistant message"},
            {"role": "tool", "content": "Tool output"},
        ]

        result = generate_title_from_messages(messages)

        assert result is None

    def test__generate_title__complex_real_world_example(self) -> None:
        """Test with a realistic complex message."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    "  Can you help me understand\n\n"
                    "how to implement\n  auto-title generation  "
                ),
            },
        ]

        result = generate_title_from_messages(messages)

        expected = "Can you help me understand how to implement auto-title generation"
        assert result == expected
        assert "\n" not in result
        assert not result.startswith(" ")
        assert not result.endswith(" ")


class TestOpenAIChatRequestReasoningEffort:
    """Tests for OpenAIChatRequest reasoning_effort parameter."""

    def test__reasoning_effort__accepts_minimal(self) -> None:
        """Test that reasoning_effort accepts 'minimal' value."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            reasoning_effort="minimal",
        )

        assert request.reasoning_effort == "minimal"

    def test__reasoning_effort__accepts_low(self) -> None:
        """Test that reasoning_effort accepts 'low' value."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            reasoning_effort="low",
        )

        assert request.reasoning_effort == "low"

    def test__reasoning_effort__accepts_medium(self) -> None:
        """Test that reasoning_effort accepts 'medium' value."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            reasoning_effort="medium",
        )

        assert request.reasoning_effort == "medium"

    def test__reasoning_effort__accepts_high(self) -> None:
        """Test that reasoning_effort accepts 'high' value."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            reasoning_effort="high",
        )

        assert request.reasoning_effort == "high"

    def test__reasoning_effort__accepts_none(self) -> None:
        """Test that reasoning_effort is optional (None is allowed)."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            reasoning_effort=None,
        )

        assert request.reasoning_effort is None

    def test__reasoning_effort__defaults_to_none(self) -> None:
        """Test that reasoning_effort defaults to None when not provided."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

        assert request.reasoning_effort is None

    def test__reasoning_effort__rejects_invalid_value(self) -> None:
        """Test that reasoning_effort rejects invalid values."""
        with pytest.raises(ValidationError) as exc_info:
            OpenAIChatRequest(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
                reasoning_effort="invalid",  # Invalid value
            )

        # Verify the error is about reasoning_effort
        errors = exc_info.value.errors()
        assert any(
            error["loc"] == ("reasoning_effort",) for error in errors
        ), "Expected validation error for reasoning_effort field"


class TestReasoningEventType:
    """Tests for ReasoningEventType enum."""

    def test__external_reasoning__event_type_exists(self) -> None:
        """Test that EXTERNAL_REASONING event type exists in enum."""
        assert hasattr(ReasoningEventType, "EXTERNAL_REASONING")
        assert ReasoningEventType.EXTERNAL_REASONING == "external_reasoning"

    def test__external_reasoning__can_create_event(self) -> None:
        """Test that we can create a ReasoningEvent with EXTERNAL_REASONING type."""
        event = ReasoningEvent(
            type=ReasoningEventType.EXTERNAL_REASONING,
            step_iteration=1,
            metadata={
                "thought": "Let me think about this problem step by step...",
                "provider": "anthropic",
            },
        )

        assert event.type == ReasoningEventType.EXTERNAL_REASONING
        assert event.step_iteration == 1
        expected_text = "Let me think about this problem step by step..."
        assert event.metadata["thought"] == expected_text
        assert event.metadata["provider"] == "anthropic"
        assert event.error is None

    def test__external_reasoning__serializes_correctly(self) -> None:
        """Test that EXTERNAL_REASONING event serializes to correct JSON."""
        event = ReasoningEvent(
            type=ReasoningEventType.EXTERNAL_REASONING,
            step_iteration=1,
            metadata={"thought": "Step by step analysis", "provider": "deepseek"},
        )

        serialized = event.model_dump()

        assert serialized["type"] == "external_reasoning"
        assert serialized["step_iteration"] == 1
        assert serialized["metadata"]["thought"] == "Step by step analysis"
        assert serialized["metadata"]["provider"] == "deepseek"
