"""
Tests for MCP to Prompt conversion utilities.

This module tests the conversion from MCP prompts to generic Prompt objects,
ensuring the integration works correctly between FastMCP and our Prompt abstraction.

MOCK ARCHITECTURE:
==================
These tests mock the FastMCP Client and MCP Prompt objects to simulate external
MCP servers without requiring actual server connections.

What's Being Mocked:
- FastMCP Client: The client that connects to MCP servers
- MCP Prompt Objects: Prompt metadata returned by client.list_prompts()
- MCP Prompt Execution: The client.get_prompt() method that retrieves prompts
- MCP PromptMessage Objects: Message objects returned by get_prompt()

Mock Flow:
1. Create mock FastMCP Client with async context manager support
2. Create mock MCP Prompt objects with name, description, and arguments
3. Mock client.list_prompts() to return our fake MCP prompts
4. Mock client.get_prompt() to return predictable test messages
5. Call to_prompts(mock_client) to convert MCP prompts → Prompt objects
6. Verify the conversion preserved metadata and created working wrapper functions
"""

import pytest
from unittest.mock import Mock, AsyncMock

from api.mcp import to_prompts
from api.prompts import Prompt
from api.config import settings


def create_mock_client():
    """Create a mock FastMCP client with async context manager support."""
    mock_client = Mock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


class TestMCPToPromptConversion:
    """Test conversion from MCP prompts to generic Prompt objects."""

    @pytest.mark.asyncio
    async def test_to_prompt_basic_conversion(self) -> None:
        """Test basic conversion of MCP prompts to Prompt objects."""
        # Mock FastMCP client with async context manager support
        mock_client = create_mock_client()

        # Mock MCP prompt
        mock_mcp_prompt = Mock()
        mock_mcp_prompt.name = "ask_question"
        mock_mcp_prompt.description = "Generate a question about a topic"

        # Mock MCP prompt arguments
        mock_arg = Mock()
        mock_arg.name = "topic"
        mock_arg.required = True
        mock_arg.description = "Topic to ask about"
        mock_mcp_prompt.arguments = [mock_arg]

        # Mock prompt result with messages
        mock_prompt_result = Mock()
        mock_message = Mock()
        mock_message.role = "user"
        mock_message.content = Mock(text="What is Python?")
        mock_prompt_result.messages = [mock_message]

        mock_client.list_prompts = AsyncMock(return_value=[mock_mcp_prompt])
        mock_client.get_prompt = AsyncMock(return_value=mock_prompt_result)

        # Convert to prompts
        prompts = await to_prompts(mock_client)

        # Verify conversion
        assert len(prompts) == 1
        prompt = prompts[0]

        assert isinstance(prompt, Prompt)
        assert prompt.name == "ask_question"
        assert prompt.description == "Generate a question about a topic"
        assert len(prompt.arguments) == 1
        assert prompt.arguments[0]["name"] == "topic"
        assert prompt.arguments[0]["required"] is True
        assert prompt.arguments[0]["description"] == "Topic to ask about"

        # Verify the wrapped function works
        result = await prompt(topic="Python")
        assert result.success is True
        assert result.messages is not None
        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "user"
        assert result.messages[0]["content"] == "What is Python?"

        # Verify the MCP client was called correctly
        mock_client.get_prompt.assert_called_once_with("ask_question", {"topic": "Python"})

    @pytest.mark.asyncio
    async def test_to_prompt_multiple_prompts(self) -> None:
        """Test conversion of multiple MCP prompts."""
        mock_client = create_mock_client()

        # Mock multiple MCP prompts
        mock_prompt1 = Mock()
        mock_prompt1.name = "summarize"
        mock_prompt1.description = "Summarize text"
        mock_arg1 = Mock()
        mock_arg1.name = "text"
        mock_arg1.required = True
        mock_arg1.description = "Text to summarize"
        mock_prompt1.arguments = [mock_arg1]

        mock_prompt2 = Mock()
        mock_prompt2.name = "translate"
        mock_prompt2.description = "Translate text"
        mock_arg2a = Mock()
        mock_arg2a.name = "text"
        mock_arg2a.required = True
        mock_arg2a.description = "Text to translate"
        mock_arg2b = Mock()
        mock_arg2b.name = "language"
        mock_arg2b.required = True
        mock_arg2b.description = "Target language"
        mock_prompt2.arguments = [mock_arg2a, mock_arg2b]

        # Mock prompt results
        def mock_get_prompt(name: str, args: object) -> object:  # noqa: ARG001
            result = Mock()
            message = Mock()
            message.role = "user"
            message.content = Mock(text=f"prompt_for_{name}")
            result.messages = [message]
            return result

        mock_client.list_prompts = AsyncMock(return_value=[mock_prompt1, mock_prompt2])
        mock_client.get_prompt = AsyncMock(side_effect=mock_get_prompt)

        # Convert to prompts
        prompts = await to_prompts(mock_client)

        # Verify conversion
        assert len(prompts) == 2

        summarize_prompt = next(p for p in prompts if p.name == "summarize")
        translate_prompt = next(p for p in prompts if p.name == "translate")

        assert summarize_prompt.description == "Summarize text"
        assert len(summarize_prompt.arguments) == 1

        assert translate_prompt.description == "Translate text"
        assert len(translate_prompt.arguments) == 2

        # Test both prompts work independently
        summarize_result = await summarize_prompt(text="Some text")
        translate_result = await translate_prompt(text="Hello", language="Spanish")

        assert summarize_result.success is True
        assert summarize_result.messages[0]["content"] == "prompt_for_summarize"

        assert translate_result.success is True
        assert translate_result.messages[0]["content"] == "prompt_for_translate"

    @pytest.mark.asyncio
    async def test_to_prompt_with_missing_description(self) -> None:
        """Test conversion handles missing description gracefully."""
        mock_client = create_mock_client()

        mock_mcp_prompt = Mock()
        mock_mcp_prompt.name = "no_desc_prompt"
        mock_mcp_prompt.description = None  # Missing description
        mock_mcp_prompt.arguments = []

        mock_prompt_result = Mock()
        mock_message = Mock()
        mock_message.role = "user"
        mock_message.content = Mock(text="Test")
        mock_prompt_result.messages = [mock_message]

        mock_client.list_prompts = AsyncMock(return_value=[mock_mcp_prompt])
        mock_client.get_prompt = AsyncMock(return_value=mock_prompt_result)

        prompts = await to_prompts(mock_client)

        assert len(prompts) == 1
        prompt = prompts[0]
        assert prompt.description == "No description available"

    @pytest.mark.asyncio
    async def test_to_prompt_with_missing_arguments(self) -> None:
        """Test conversion handles missing arguments gracefully."""
        mock_client = create_mock_client()

        mock_mcp_prompt = Mock()
        mock_mcp_prompt.name = "no_args_prompt"
        mock_mcp_prompt.description = "Prompt without arguments"
        mock_mcp_prompt.arguments = None  # Missing arguments

        mock_prompt_result = Mock()
        mock_message = Mock()
        mock_message.role = "user"
        mock_message.content = Mock(text="Test")
        mock_prompt_result.messages = [mock_message]

        mock_client.list_prompts = AsyncMock(return_value=[mock_mcp_prompt])
        mock_client.get_prompt = AsyncMock(return_value=mock_prompt_result)

        prompts = await to_prompts(mock_client)

        assert len(prompts) == 1
        prompt = prompts[0]
        assert prompt.arguments == []

    @pytest.mark.asyncio
    async def test_to_prompt_error_handling(self) -> None:
        """Test that MCP prompt execution errors are handled properly."""
        mock_client = create_mock_client()

        mock_mcp_prompt = Mock()
        mock_mcp_prompt.name = "failing_prompt"
        mock_mcp_prompt.description = "Prompt that fails"
        mock_mcp_prompt.arguments = []

        mock_client.list_prompts = AsyncMock(return_value=[mock_mcp_prompt])
        # Make the MCP client get_prompt raise an exception
        mock_client.get_prompt = AsyncMock(side_effect=Exception("MCP prompt failed"))

        prompts = await to_prompts(mock_client)

        assert len(prompts) == 1
        prompt = prompts[0]

        # When we call the prompt, it should handle the error gracefully
        result = await prompt()
        assert result.success is False
        assert "MCP prompt failed" in result.error

    @pytest.mark.asyncio
    async def test_to_prompt_with_multiple_messages(self) -> None:
        """Test prompt wrapper correctly handles multiple messages."""
        mock_client = create_mock_client()

        mock_mcp_prompt = Mock()
        mock_mcp_prompt.name = "multi_message_prompt"
        mock_mcp_prompt.description = "Prompt with multiple messages"
        mock_mcp_prompt.arguments = []

        # Mock result with multiple messages
        mock_result = Mock()
        mock_msg1 = Mock()
        mock_msg1.role = "system"
        mock_msg1.content = Mock(text="You are a helpful assistant")
        mock_msg2 = Mock()
        mock_msg2.role = "user"
        mock_msg2.content = Mock(text="Hello")
        mock_result.messages = [mock_msg1, mock_msg2]

        mock_client.list_prompts = AsyncMock(return_value=[mock_mcp_prompt])
        mock_client.get_prompt = AsyncMock(return_value=mock_result)

        prompts = await to_prompts(mock_client)
        prompt = prompts[0]

        result = await prompt()
        assert result.success is True
        assert len(result.messages) == 2
        assert result.messages[0]["role"] == "system"
        assert result.messages[0]["content"] == "You are a helpful assistant"
        assert result.messages[1]["role"] == "user"
        assert result.messages[1]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_to_prompt_with_string_content(self) -> None:
        """Test prompt wrapper handles direct string content."""
        mock_client = create_mock_client()

        mock_mcp_prompt = Mock()
        mock_mcp_prompt.name = "string_content_prompt"
        mock_mcp_prompt.description = "Prompt with string content"
        mock_mcp_prompt.arguments = []

        # Mock result with string content (not TextContent object)
        mock_result = Mock()
        mock_message = Mock()
        mock_message.role = "user"
        mock_message.content = "Direct string content"  # String instead of Mock(text=...)
        mock_result.messages = [mock_message]

        mock_client.list_prompts = AsyncMock(return_value=[mock_mcp_prompt])
        mock_client.get_prompt = AsyncMock(return_value=mock_result)

        prompts = await to_prompts(mock_client)
        prompt = prompts[0]

        result = await prompt()
        assert result.success is True
        assert result.messages[0]["content"] == "Direct string content"

    @pytest.mark.asyncio
    async def test_to_prompt_empty_prompts_list(self) -> None:
        """Test conversion with no available MCP prompts."""
        mock_client = create_mock_client()
        mock_client.list_prompts = AsyncMock(return_value=[])

        prompts = await to_prompts(mock_client)

        assert len(prompts) == 0
        assert prompts == []

    @pytest.mark.asyncio
    async def test_to_prompt_client_list_prompts_error(self) -> None:
        """Test error handling when client.list_prompts() fails."""
        mock_client = create_mock_client()
        mock_client.list_prompts = AsyncMock(side_effect=Exception("Failed to list prompts"))

        with pytest.raises(Exception, match="Failed to list prompts"):
            await to_prompts(mock_client)

    @pytest.mark.asyncio
    async def test_to_prompt_preserves_complex_arguments(self) -> None:
        """Test that complex argument specifications are preserved correctly."""
        mock_client = create_mock_client()

        mock_mcp_prompt = Mock()
        mock_mcp_prompt.name = "complex_prompt"
        mock_mcp_prompt.description = "Prompt with complex arguments"

        # Multiple arguments with different specifications
        mock_arg1 = Mock()
        mock_arg1.name = "required_arg"
        mock_arg1.required = True
        mock_arg1.description = "A required argument"

        mock_arg2 = Mock()
        mock_arg2.name = "optional_arg"
        mock_arg2.required = False
        mock_arg2.description = "An optional argument"

        mock_arg3 = Mock()
        mock_arg3.name = "no_desc_arg"
        mock_arg3.required = True
        # Create mock without description attribute by using spec
        del mock_arg3.description

        mock_mcp_prompt.arguments = [mock_arg1, mock_arg2, mock_arg3]

        mock_result = Mock()
        mock_message = Mock()
        mock_message.role = "user"
        mock_message.content = Mock(text="Complex prompt result")
        mock_result.messages = [mock_message]

        mock_client.list_prompts = AsyncMock(return_value=[mock_mcp_prompt])
        mock_client.get_prompt = AsyncMock(return_value=mock_result)

        prompts = await to_prompts(mock_client)
        prompt = prompts[0]

        # Verify complex arguments are preserved
        assert len(prompt.arguments) == 3

        # Check first argument
        assert prompt.arguments[0]["name"] == "required_arg"
        assert prompt.arguments[0]["required"] is True
        assert prompt.arguments[0]["description"] == "A required argument"

        # Check second argument
        assert prompt.arguments[1]["name"] == "optional_arg"
        assert prompt.arguments[1]["required"] is False
        assert prompt.arguments[1]["description"] == "An optional argument"

        # Check third argument (missing description)
        assert prompt.arguments[2]["name"] == "no_desc_arg"
        assert prompt.arguments[2]["required"] is True
        assert prompt.arguments[2]["description"] == ""

        # Verify prompt still works with complex arguments
        complex_args = {
            "required_arg": "value1",
            "optional_arg": "value2",
            "no_desc_arg": "value3",
        }

        result = await prompt(**complex_args)
        assert result.success is True

        # Verify arguments were passed correctly to MCP client
        mock_client.get_prompt.assert_called_once_with("complex_prompt", complex_args)

    @pytest.mark.asyncio
    async def test_to_prompt_name_parsing(self) -> None:
        """Test that prompt names are parsed and cleaned with new naming conventions."""
        mock_client = create_mock_client()

        # Simulate FastMCP's automatic prompt name prefixing with server names
        mock_prompt1 = Mock()
        mock_prompt1.name = "server_a__summarize"  # Already prefixed by FastMCP
        mock_prompt1.description = "Summarize from server A"
        mock_prompt1.arguments = []

        mock_prompt2 = Mock()
        mock_prompt2.name = "server_b__translate"  # Already prefixed by FastMCP
        mock_prompt2.description = "Translate from server B"
        mock_prompt2.arguments = []

        mock_result = Mock()
        mock_message = Mock()
        mock_message.role = "user"
        mock_message.content = Mock(text="Test")
        mock_result.messages = [mock_message]

        mock_client.list_prompts = AsyncMock(return_value=[mock_prompt1, mock_prompt2])
        mock_client.get_prompt = AsyncMock(return_value=mock_result)

        prompts = await to_prompts(mock_client)

        # Verify names are cleaned (server prefix removed)
        prompt_names = [p.name for p in prompts]
        assert "summarize" in prompt_names
        assert "translate" in prompt_names

        # Verify metadata fields are populated
        summarize_prompt = next(p for p in prompts if p.name == "summarize")
        assert summarize_prompt.server_name == "server-a"
        assert summarize_prompt.mcp_name == "server_a__summarize"

        translate_prompt = next(p for p in prompts if p.name == "translate")
        assert translate_prompt.server_name == "server-b"
        assert translate_prompt.mcp_name == "server_b__translate"

    @pytest.mark.asyncio
    async def test_to_prompt_optional_arguments_not_required(self) -> None:
        """Test that optional arguments can be omitted when calling prompts."""
        mock_client = create_mock_client()

        mock_mcp_prompt = Mock()
        mock_mcp_prompt.name = "flexible_prompt"
        mock_mcp_prompt.description = "Prompt with optional arguments"

        mock_arg1 = Mock()
        mock_arg1.name = "required"
        mock_arg1.required = True
        mock_arg1.description = "Required argument"

        mock_arg2 = Mock()
        mock_arg2.name = "optional"
        mock_arg2.required = False
        mock_arg2.description = "Optional argument"

        mock_mcp_prompt.arguments = [mock_arg1, mock_arg2]

        mock_result = Mock()
        mock_message = Mock()
        mock_message.role = "user"
        mock_message.content = Mock(text="Response")
        mock_result.messages = [mock_message]

        mock_client.list_prompts = AsyncMock(return_value=[mock_mcp_prompt])
        mock_client.get_prompt = AsyncMock(return_value=mock_result)

        prompts = await to_prompts(mock_client)
        prompt = prompts[0]

        # Should work with only required argument
        result = await prompt(required="value")
        assert result.success is True

        # Verify only required argument was passed
        mock_client.get_prompt.assert_called_with("flexible_prompt", {"required": "value"})


class TestDisabledPromptFiltering:
    """Test filtering of disabled prompts via override configuration."""

    @pytest.mark.asyncio
    async def test_disabled_prompts_are_filtered_out(self, tmp_path):  # noqa: ANN001
        """Test that prompts with disable: true in override are excluded."""
        mock_client = create_mock_client()

        # Create prompts
        enabled_prompt = Mock()
        enabled_prompt.name = "meta__generate"
        enabled_prompt.description = "Generate playbook"
        enabled_prompt.arguments = []

        disabled_prompt = Mock()
        disabled_prompt.name = "meta__unwanted"
        disabled_prompt.description = "Unwanted prompt"
        disabled_prompt.arguments = []

        mock_client.list_prompts = AsyncMock(return_value=[enabled_prompt, disabled_prompt])

        # Create override config with disabled prompt
        config_content = """
prompts:
  "meta__unwanted":
    disable: true
"""
        config_file = tmp_path / "mcp_overrides.yaml"
        config_file.write_text(config_content)

        # Temporarily override settings
        original_path = settings.mcp_overrides_path
        settings.mcp_overrides_path = str(config_file)

        try:
            prompts = await to_prompts(mock_client)

            # Should only have enabled prompt
            assert len(prompts) == 1
            assert prompts[0].name == "generate"
        finally:
            settings.mcp_overrides_path = original_path

    @pytest.mark.asyncio
    async def test_disable_without_rename_prompt(self, tmp_path):  # noqa: ANN001
        """Test that you can disable a prompt without providing name override."""
        mock_client = create_mock_client()

        prompt1 = Mock()
        prompt1.name = "unwanted_prompt"
        prompt1.description = "Prompt to disable"
        prompt1.arguments = []

        mock_client.list_prompts = AsyncMock(return_value=[prompt1])

        # Create override with only disable field
        config_content = """
prompts:
  "unwanted_prompt":
    disable: true
"""
        config_file = tmp_path / "mcp_overrides.yaml"
        config_file.write_text(config_content)
        original_path = settings.mcp_overrides_path
        settings.mcp_overrides_path = str(config_file)

        try:
            prompts = await to_prompts(mock_client)

            # No prompts should be returned
            assert len(prompts) == 0
        finally:
            settings.mcp_overrides_path = original_path


if __name__ == "__main__":
    pytest.main([__file__])
