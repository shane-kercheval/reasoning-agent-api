"""
Unit tests for MCP duplicate name detection.

Tests that duplicate tool/prompt names are detected and fail with helpful errors.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from api.mcp import to_tools, to_prompts


class TestDuplicateToolDetection:
    """Test duplicate tool name detection and error handling."""

    @pytest.mark.asyncio
    async def test_duplicate_tools_raises_error(self) -> None:
        """Test that duplicate tool names raise ValueError."""
        # Create mock MCP tools that will parse to same name
        mock_tool1 = Mock()
        mock_tool1.name = "filesystem__search"
        mock_tool1.description = "Search filesystem"
        mock_tool1.inputSchema = {}

        mock_tool2 = Mock()
        mock_tool2.name = "brave_search__search"  # Will also parse to "search"
        mock_tool2.description = "Web search"
        mock_tool2.inputSchema = {}

        # Mock client
        mock_client = Mock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_tools = AsyncMock(return_value=[mock_tool1, mock_tool2])

        # Should raise ValueError about duplicate
        with pytest.raises(ValueError, match="Duplicate tool name") as exc_info:
            await to_tools(mock_client)

        error_msg = str(exc_info.value)
        assert "search" in error_msg
        assert "filesystem__search" in error_msg
        assert "brave_search__search" in error_msg

    @pytest.mark.asyncio
    async def test_duplicate_error_message_format(self) -> None:
        """Test that error message includes helpful resolution info."""
        # Create mock tools with duplicate names
        mock_tool1 = Mock()
        mock_tool1.name = "server_a__my_tool"
        mock_tool1.description = "Tool from server A"
        mock_tool1.inputSchema = {}

        mock_tool2 = Mock()
        mock_tool2.name = "server_b__my_tool"
        mock_tool2.description = "Tool from server B"
        mock_tool2.inputSchema = {}

        mock_client = Mock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_tools = AsyncMock(return_value=[mock_tool1, mock_tool2])

        with pytest.raises(ValueError, match="Duplicate tool name") as exc_info:
            await to_tools(mock_client)

        error_msg = str(exc_info.value)
        # Verify error message shows both conflicting tools
        assert "server_a__my_tool" in error_msg
        assert "server_b__my_tool" in error_msg

        # Verify error suggests override config
        assert "mcp_overrides.yaml" in error_msg or "overrides" in error_msg.lower()
        assert "tools:" in error_msg

    @pytest.mark.asyncio
    async def test_no_duplicates_succeeds(self) -> None:
        """Test that unique names process successfully."""
        # Create mock tools with unique names
        mock_tool1 = Mock()
        mock_tool1.name = "github__get_pr"
        mock_tool1.description = "Get PR info"
        mock_tool1.inputSchema = {}

        mock_tool2 = Mock()
        mock_tool2.name = "filesystem__read_file"
        mock_tool2.description = "Read a file"
        mock_tool2.inputSchema = {}

        mock_client = Mock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_tools = AsyncMock(return_value=[mock_tool1, mock_tool2])

        # Should complete without error
        tools = await to_tools(mock_client)

        assert len(tools) == 2
        # Verify clean names
        tool_names = {t.name for t in tools}
        assert "get_pr" in tool_names
        assert "read_file" in tool_names



class TestDuplicatePromptDetection:
    """Test duplicate prompt name detection and error handling."""

    @pytest.mark.asyncio
    async def test_duplicate_prompts_raises_error(self) -> None:
        """Test that duplicate prompt names raise ValueError."""
        # Create mock prompts that parse to same name
        mock_prompt1 = Mock()
        mock_prompt1.name = "meta__generate"
        mock_prompt1.description = "Generate from meta"
        mock_prompt1.arguments = []

        mock_prompt2 = Mock()
        mock_prompt2.name = "thinking__generate"  # Same base name
        mock_prompt2.description = "Generate from thinking"
        mock_prompt2.arguments = []

        mock_client = Mock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_prompts = AsyncMock(return_value=[mock_prompt1, mock_prompt2])

        with pytest.raises(ValueError, match="Duplicate prompt name"):
            await to_prompts(mock_client)

    @pytest.mark.asyncio
    async def test_no_duplicate_prompts_succeeds(self) -> None:
        """Test that unique prompt names process successfully."""
        mock_prompt1 = Mock()
        mock_prompt1.name = "meta__playbook"
        mock_prompt1.description = "Generate playbook"
        mock_prompt1.arguments = []

        mock_prompt2 = Mock()
        mock_prompt2.name = "thinking__analyze"
        mock_prompt2.description = "Analyze deeply"
        mock_prompt2.arguments = []

        mock_client = Mock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_prompts = AsyncMock(return_value=[mock_prompt1, mock_prompt2])

        prompts = await to_prompts(mock_client)

        assert len(prompts) == 2
        prompt_names = {p.name for p in prompts}
        assert "playbook" in prompt_names
        assert "analyze" in prompt_names
