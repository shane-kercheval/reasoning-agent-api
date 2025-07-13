"""
Tests for the MCP client implementation using FastMCP 2.0.

Tests the client's ability to communicate with FastMCP servers and
handle various response types.
"""


import pytest
from unittest.mock import AsyncMock, patch

from api.mcp_client import MCPClient, MCPServerConfig


@pytest.fixture
def sample_server_configs() -> list[MCPServerConfig]:
    """Sample MCP server configurations for testing."""
    return [
        MCPServerConfig(
            name="tools",
            url="http://localhost:8001",
            tools=["web_search", "weather_api", "filesystem"],
        ),
    ]


@pytest.fixture
def mcp_client(sample_server_configs: list[MCPServerConfig]) -> MCPClient:
    """MCP client instance for testing."""
    return MCPClient(sample_server_configs)


@pytest.mark.asyncio
async def test_list_tools_with_server_available(mcp_client: MCPClient) -> None:
    """Test listing tools when server is available."""
    # Mock the FastMCP Client
    mock_tool = AsyncMock()
    mock_tool.name = "web_search"

    with patch("api.mcp_client.Client") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_tools.return_value = [mock_tool]
        mock_client_class.return_value = mock_client

        result = await mcp_client.list_tools()

        assert "tools" in result
        assert "web_search" in result["tools"]


@pytest.mark.asyncio
async def test_list_tools_with_server_unavailable(mcp_client: MCPClient) -> None:
    """Test listing tools when server is unavailable."""
    with patch("api.mcp_client.Client") as mock_client_class:
        # Simulate connection failure
        mock_client_class.side_effect = Exception("Connection failed")

        result = await mcp_client.list_tools()

        # Should fallback to configured tools
        assert "tools" in result
        assert "web_search" in result["tools"]
        assert "weather_api" in result["tools"]


@pytest.mark.asyncio
async def test_call_tool_success(mcp_client: MCPClient) -> None:
    """Test successful tool call."""
    # Mock FastMCP 2.0 CallToolResult
    mock_result = AsyncMock()
    mock_result.is_error = False
    mock_result.data = {"result": "success"}
    mock_result.structured_content = None
    mock_result.content = None

    with patch("api.mcp_client.Client") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.call_tool.return_value = mock_result
        mock_client_class.return_value = mock_client

        result = await mcp_client.call_tool("web_search", {"query": "test"})

        assert result == {"result": "success"}


@pytest.mark.asyncio
async def test_call_tool_server_error(mcp_client: MCPClient) -> None:
    """Test tool call error handling when server is unavailable."""
    with patch("api.mcp_client.Client") as mock_client_class:
        # Simulate tool call failure
        mock_client_class.side_effect = Exception("Server error")

        result = await mcp_client.call_tool("web_search", {"query": "test"})

        # Should get error response with details
        assert "error" in result
        assert "Server error" in result["error"]
        assert result["server_name"] == "tools"
        assert result["tool_name"] == "web_search"
        assert "localhost:8001" in result["server_url"]


@pytest.mark.asyncio
async def test_call_tool_not_found(mcp_client: MCPClient) -> None:
    """Test calling a tool that doesn't exist."""
    result = await mcp_client.call_tool("nonexistent_tool", {"query": "test"})

    assert "error" in result
    assert "not found" in result["error"]


@pytest.mark.asyncio
async def test_find_server_for_tool(mcp_client: MCPClient) -> None:
    """Test finding which server provides a tool."""
    config = mcp_client._find_server_for_tool("web_search")
    assert config is not None
    assert config.name == "tools"

    config = mcp_client._find_server_for_tool("nonexistent")
    assert config is None


@pytest.mark.asyncio
async def test_close_client(mcp_client: MCPClient) -> None:
    """Test client cleanup."""
    # Should not raise any errors
    await mcp_client.close()
