"""
Tests for MCP to Tool conversion utilities.

This module tests the conversion from MCP tools to generic Tool objects,
ensuring the integration works correctly between FastMCP and our Tool abstraction.

MOCK ARCHITECTURE:
==================
These tests mock the FastMCP Client and MCP Tool objects to simulate external
MCP servers without requiring actual server connections.

What's Being Mocked:
- FastMCP Client: The client that connects to MCP servers
- MCP Tool Objects: Tool metadata returned by client.list_tools()
- MCP Tool Execution: The client.call_tool() method that executes tools

Mock Flow:
1. Create mock FastMCP Client with async context manager support
2. Create mock MCP Tool objects with name, description, and inputSchema
3. Mock client.list_tools() to return our fake MCP tools
4. Mock client.call_tool() to return predictable test results
5. Call to_tools(mock_client) to convert MCP tools → Tool objects
6. Verify the conversion preserved metadata and created working wrapper functions
"""

import pytest
from unittest.mock import Mock, AsyncMock

from api.mcp import to_tools
from api.tools import Tool


def create_mock_client():
    """Create a mock FastMCP client with async context manager support."""
    mock_client = Mock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


class TestMCPToToolConversion:
    """Test conversion from MCP tools to generic Tool objects."""

    @pytest.mark.asyncio
    async def test_to_tool_basic_conversion(self):
        """Test basic conversion of MCP tools to Tool objects."""
        # Mock FastMCP client with async context manager support
        mock_client = create_mock_client()

        # Mock MCP tool
        mock_mcp_tool = Mock()
        mock_mcp_tool.name = "test_tool"
        mock_mcp_tool.description = "A test tool"
        mock_mcp_tool.inputSchema = {
            "type": "object",
            "properties": {"param": {"type": "string"}},
            "required": ["param"],
        }

        # Mock tool result with data attribute
        mock_tool_result = Mock()
        mock_tool_result.data = "test_result"

        mock_client.list_tools = AsyncMock(return_value=[mock_mcp_tool])
        mock_client.call_tool = AsyncMock(return_value=mock_tool_result)

        # Convert to tools
        tools = await to_tools(mock_client)

        # Verify conversion
        assert len(tools) == 1
        tool = tools[0]

        assert isinstance(tool, Tool)
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.input_schema == mock_mcp_tool.inputSchema

        # Verify the wrapped function works
        result = await tool(param="test_value")
        assert result.success is True
        assert result.result == "test_result"

        # Verify the MCP client was called correctly
        mock_client.call_tool.assert_called_once_with("test_tool", {"param": "test_value"})

    @pytest.mark.asyncio
    async def test_to_tool_multiple_tools(self):
        """Test conversion of multiple MCP tools."""
        # Mock FastMCP client with async context manager support
        mock_client = create_mock_client()

        # Mock multiple MCP tools
        mock_tool1 = Mock()
        mock_tool1.name = "weather"
        mock_tool1.description = "Get weather"
        mock_tool1.inputSchema = {"type": "object", "properties": {"location": {"type": "string"}}}

        mock_tool2 = Mock()
        mock_tool2.name = "search"
        mock_tool2.description = "Search web"
        mock_tool2.inputSchema = {"type": "object", "properties": {"query": {"type": "string"}}}

        # Mock tool results with data attribute
        def mock_call_tool(name: str, args: object) -> object:  # noqa: ARG001
            result = Mock()
            result.data = f"result_for_{name}"
            return result

        mock_client.list_tools = AsyncMock(return_value=[mock_tool1, mock_tool2])
        mock_client.call_tool = AsyncMock(side_effect=mock_call_tool)

        # Convert to tools
        tools = await to_tools(mock_client)

        # Verify conversion
        assert len(tools) == 2

        weather_tool = next(tool for tool in tools if tool.name == "weather")
        search_tool = next(tool for tool in tools if tool.name == "search")

        assert weather_tool.description == "Get weather"
        assert search_tool.description == "Search web"

        # Test both tools work independently
        weather_result = await weather_tool(location="Paris")
        search_result = await search_tool(query="test")

        assert weather_result.success is True
        assert weather_result.result == "result_for_weather"

        assert search_result.success is True
        assert search_result.result == "result_for_search"

    @pytest.mark.asyncio
    async def test_to_tool_with_missing_description(self):
        """Test conversion handles missing description gracefully."""
        mock_client = create_mock_client()

        mock_mcp_tool = Mock()
        mock_mcp_tool.name = "no_desc_tool"
        mock_mcp_tool.description = None  # Missing description
        mock_mcp_tool.inputSchema = {}

        mock_tool_result = Mock()
        mock_tool_result.data = "result"

        mock_client.list_tools = AsyncMock(return_value=[mock_mcp_tool])
        mock_client.call_tool = AsyncMock(return_value=mock_tool_result)

        tools = await to_tools(mock_client)

        assert len(tools) == 1
        tool = tools[0]
        assert tool.description == "No description available"

    @pytest.mark.asyncio
    async def test_to_tool_with_missing_schema(self):
        """Test conversion handles missing input schema gracefully."""
        mock_client = create_mock_client()

        mock_mcp_tool = Mock()
        mock_mcp_tool.name = "no_schema_tool"
        mock_mcp_tool.description = "Tool without schema"
        mock_mcp_tool.inputSchema = None  # Missing schema

        mock_tool_result = Mock()
        mock_tool_result.data = "result"

        mock_client.list_tools = AsyncMock(return_value=[mock_mcp_tool])
        mock_client.call_tool = AsyncMock(return_value=mock_tool_result)

        tools = await to_tools(mock_client)

        assert len(tools) == 1
        tool = tools[0]
        assert tool.input_schema == {}

    @pytest.mark.asyncio
    async def test_to_tool_error_handling(self):
        """Test that MCP tool execution errors are handled properly."""
        mock_client = create_mock_client()

        mock_mcp_tool = Mock()
        mock_mcp_tool.name = "failing_tool"
        mock_mcp_tool.description = "Tool that fails"
        mock_mcp_tool.inputSchema = {}

        mock_client.list_tools = AsyncMock(return_value=[mock_mcp_tool])
        # Make the MCP client call_tool raise an exception
        mock_client.call_tool = AsyncMock(side_effect=Exception("MCP tool failed"))

        tools = await to_tools(mock_client)

        assert len(tools) == 1
        tool = tools[0]

        # When we call the tool, it should handle the error gracefully
        result = await tool()
        assert result.success is False
        assert "MCP tool failed" in result.error

    @pytest.mark.asyncio
    async def test_to_tool_with_result_data_attribute(self):
        """Test tool wrapper correctly extracts data from MCP result objects."""
        mock_client = create_mock_client()

        mock_mcp_tool = Mock()
        mock_mcp_tool.name = "data_tool"
        mock_mcp_tool.description = "Tool with result.data"
        mock_mcp_tool.inputSchema = {}

        # Mock result with .data attribute
        mock_result = Mock()
        mock_result.data = {"structured": "data", "value": 42}

        mock_client.list_tools = AsyncMock(return_value=[mock_mcp_tool])
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        tools = await to_tools(mock_client)
        tool = tools[0]

        result = await tool()
        assert result.success is True
        assert result.result == {"structured": "data", "value": 42}

    @pytest.mark.asyncio
    async def test_to_tool_with_result_no_data_attribute(self):
        """Test tool wrapper handles MCP results without .data attribute."""
        mock_client = create_mock_client()

        mock_mcp_tool = Mock()
        mock_mcp_tool.name = "no_data_tool"
        mock_mcp_tool.description = "Tool without result.data"
        mock_mcp_tool.inputSchema = {}

        # Mock result WITHOUT .data attribute
        mock_result = "plain_result"

        mock_client.list_tools = AsyncMock(return_value=[mock_mcp_tool])
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        tools = await to_tools(mock_client)
        tool = tools[0]

        result = await tool()
        assert result.success is True
        assert result.result == "plain_result"

    @pytest.mark.asyncio
    async def test_to_tool_empty_tools_list(self):
        """Test conversion with no available MCP tools."""
        mock_client = create_mock_client()
        mock_client.list_tools = AsyncMock(return_value=[])

        tools = await to_tools(mock_client)

        assert len(tools) == 0
        assert tools == []

    @pytest.mark.asyncio
    async def test_to_tool_client_list_tools_error(self):
        """Test error handling when client.list_tools() fails."""
        mock_client = create_mock_client()
        mock_client.list_tools = AsyncMock(side_effect=Exception("Failed to list tools"))

        with pytest.raises(Exception, match="Failed to list tools"):
            await to_tools(mock_client)

    @pytest.mark.asyncio
    async def test_to_tool_preserves_complex_schemas(self):
        """Test that complex input schemas are preserved correctly."""
        mock_client = create_mock_client()

        mock_mcp_tool = Mock()
        mock_mcp_tool.name = "complex_tool"
        mock_mcp_tool.description = "Tool with complex schema"
        mock_mcp_tool.inputSchema = {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "enum": ["fast", "thorough"]},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["mode"],
                },
                "metadata": {
                    "type": "object",
                    "additionalProperties": True,
                },
            },
            "required": ["config"],
        }

        mock_result = Mock()
        mock_result.data = "complex_result"

        mock_client.list_tools = AsyncMock(return_value=[mock_mcp_tool])
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        tools = await to_tools(mock_client)
        tool = tools[0]

        # Verify complex schema is preserved exactly
        assert tool.input_schema == mock_mcp_tool.inputSchema

        # Verify tool still works with complex arguments
        complex_args = {
            "config": {"mode": "fast", "options": ["opt1", "opt2"]},
            "metadata": {"user": "test", "session": "123"},
        }

        result = await tool(**complex_args)
        assert result.success is True

        # Verify arguments were passed correctly to MCP client
        mock_client.call_tool.assert_called_once_with("complex_tool", complex_args)

    @pytest.mark.asyncio
    async def test_to_tool_name_prefixing(self):
        """Test that tool names are preserved as returned by FastMCP (already prefixed)."""
        mock_client = create_mock_client()

        # Simulate FastMCP's automatic tool name prefixing with server names
        mock_tool1 = Mock()
        mock_tool1.name = "server_a_weather"  # Already prefixed by FastMCP
        mock_tool1.description = "Weather from server A"
        mock_tool1.inputSchema = {}

        mock_tool2 = Mock()
        mock_tool2.name = "server_b_search"  # Already prefixed by FastMCP
        mock_tool2.description = "Search from server B"
        mock_tool2.inputSchema = {}

        mock_client.list_tools = AsyncMock(return_value=[mock_tool1, mock_tool2])
        mock_client.call_tool = AsyncMock(return_value=Mock(data="result"))

        tools = await to_tools(mock_client)

        # Verify prefixed names are preserved
        tool_names = [tool.name for tool in tools]
        assert "server_a_weather" in tool_names
        assert "server_b_search" in tool_names


if __name__ == "__main__":
    pytest.main([__file__])
