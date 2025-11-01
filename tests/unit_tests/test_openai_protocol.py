"""Unit tests for OpenAI protocol utilities."""

import pytest
from api.openai_protocol import extract_system_message


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
