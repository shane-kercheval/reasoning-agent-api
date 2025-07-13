"""
Simplified MCP tests that work with manually started servers.

Run the test servers manually before running these tests:
  Terminal 1: uv run python tests/mcp_servers/server_a.py
  Terminal 2: uv run python tests/mcp_servers/server_b.py
"""

import pytest
import pytest_asyncio

from api.mcp import (
    MCPClient,
    MCPManager,
    MCPServerConfig,
    ToolRequest,
)


class TestMCPClient:
    """Test MCPClient functionality against server A."""

    @pytest_asyncio.fixture
    async def server_a_config(self) -> MCPServerConfig:
        """Configuration for test server A."""
        return MCPServerConfig(
            name="server_a",
            url="http://localhost:8001/mcp/",
            enabled=True,
        )

    @pytest.mark.asyncio
    async def test__validate_connection__success(self, server_a_config: MCPServerConfig):
        """Test successful connection validation."""
        client = MCPClient(server_a_config)
        await client.validate_connection()
        # No exception means success

    @pytest.mark.asyncio
    async def test__list_tools__success(self, server_a_config: MCPServerConfig):
        """Test listing tools from server A."""
        client = MCPClient(server_a_config)
        tools = await client.list_tools()

        assert len(tools) == 2
        tool_names = [tool.tool_name for tool in tools]
        assert "weather_api" in tool_names
        assert "web_search" in tool_names

        # Verify tool details
        weather_tool = next(tool for tool in tools if tool.tool_name == "weather_api")
        assert weather_tool.server_name == "server_a"
        assert "weather" in weather_tool.description.lower()

    @pytest.mark.asyncio
    async def test__call_tool__weather_api_success(self, server_a_config: MCPServerConfig):
        """Test calling weather_api tool successfully."""
        client = MCPClient(server_a_config)
        result = await client.call_tool("weather_api", {"location": "Tokyo"})

        assert isinstance(result, dict)
        assert result["location"] == "Tokyo"
        assert "current" in result
        assert "forecast" in result
        assert result["server"] == "test-server-a"


class TestMCPManager:
    """Test MCPManager functionality against both servers."""

    @pytest_asyncio.fixture
    async def both_server_configs(self) -> list[MCPServerConfig]:
        """Configuration for both test servers."""
        return [
            MCPServerConfig(name="server_a", url="http://localhost:8001/mcp/", enabled=True),
            MCPServerConfig(name="server_b", url="http://localhost:8002/mcp/", enabled=True),
        ]

    @pytest.mark.asyncio
    async def test__initialize__success_all_servers(self, both_server_configs: list[MCPServerConfig]):  # noqa: E501
        """Test successful initialization with all servers available."""
        manager = MCPManager(both_server_configs)
        await manager.initialize()

        connected_servers = manager.get_connected_servers()
        assert len(connected_servers) == 2
        assert "server_a" in connected_servers
        assert "server_b" in connected_servers

    @pytest.mark.asyncio
    async def test__get_available_tools__all_servers(self, both_server_configs: list[MCPServerConfig]):  # noqa: E501
        """Test getting tools from all connected servers."""
        manager = MCPManager(both_server_configs)
        await manager.initialize()

        tools = await manager.get_available_tools()

        assert len(tools) == 4  # 2 tools from each server
        tool_names = [tool.tool_name for tool in tools]
        assert "weather_api" in tool_names
        assert "web_search" in tool_names
        assert "filesystem" in tool_names
        assert "search_news" in tool_names

    @pytest.mark.asyncio
    async def test__execute_tool__server_a_weather(self, both_server_configs: list[MCPServerConfig]):  # noqa: E501
        """Test executing weather tool on server A."""
        manager = MCPManager(both_server_configs)
        await manager.initialize()

        request = ToolRequest(
            server_name="server_a",
            tool_name="weather_api",
            arguments={"location": "Paris"},
        )

        result = await manager.execute_tool(request)

        assert result.success is True
        assert result.server_name == "server_a"
        assert result.tool_name == "weather_api"
        assert result.error is None
        assert result.execution_time_ms > 0

        # Verify result data
        assert result.result["location"] == "Paris"
        assert result.result["server"] == "test-server-a"

    @pytest.mark.asyncio
    async def test__execute_tools_parallel__multiple_servers(self, both_server_configs: list[MCPServerConfig]):  # noqa: E501
        """Test executing tools in parallel across multiple servers."""
        manager = MCPManager(both_server_configs)
        await manager.initialize()

        requests = [
            ToolRequest(
                server_name="server_a",
                tool_name="weather_api",
                arguments={"location": "Tokyo"},
            ),
            ToolRequest(
                server_name="server_b",
                tool_name="search_news",
                arguments={"query": "AI"},
            ),
        ]

        results = await manager.execute_tools_parallel(requests)

        assert len(results) == 2

        # All should succeed
        for result in results:
            assert result.success is True
            assert result.error is None

        # Verify specific results
        weather_result = results[0]
        assert weather_result.server_name == "server_a"
        assert weather_result.result["location"] == "Tokyo"

        news_result = results[1]
        assert news_result.server_name == "server_b"
        assert news_result.result["query"] == "AI"
