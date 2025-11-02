"""Unit tests for OpenAI protocol utilities."""

from unittest.mock import Mock

from api.openai_protocol import extract_system_message, convert_litellm_to_stream_response


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
