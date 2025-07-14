"""
Comprehensive tests for MCP module (MCPClient and MCPManager).

These tests automatically start/stop real MCP test servers to ensure proper functionality.

Test Infrastructure:
- MCPTestServerManager: Manages MCP test server lifecycle via subprocess
- Server A (port 8001): Provides weather_api and web_search tools
- Server B (port 8002): Provides filesystem and search_news tools
- Tests cover both single-server (MCPClient) and multi-server (MCPManager) scenarios
- Servers are started once per test session and cleaned up automatically

Run with: uv run python -m pytest tests/test_mcp.py -v
"""

import asyncio
import os
import pytest
import pytest_asyncio
import subprocess
import time
from collections.abc import AsyncGenerator

from api.mcp import (
    MCPClient,
    MCPManager,
    MCPServerConfig,
    ToolRequest,
    ToolExecutionError,
)
from tests.mcp_servers.server_a import get_server_instance as get_server_a
from tests.mcp_servers.server_b import get_server_instance as get_server_b


class MCPTestServerManager:
    """Test harness that manages MCP test servers."""

    def __init__(self):
        self.server_processes = {}

    async def start_server(self, name: str, port: int, script_path: str) -> None:
        """Start an MCP test server."""
        cmd = ["uv", "run", "python", script_path]
        # Get project root directory relative to this test file
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        process = subprocess.Popen(  # noqa: ASYNC220
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=project_root,
        )

        self.server_processes[name] = process

        # Wait for server to start up
        await asyncio.sleep(3)

        # Verify server is responding
        max_retries = 15
        for i in range(max_retries):
            try:
                config = MCPServerConfig(name=name, url=f"http://localhost:{port}/mcp/")
                client = MCPClient(config)
                await client.validate_connection()
                break
            except Exception as e:
                if i == max_retries - 1:
                    raise RuntimeError(f"Server {name} failed to start after {max_retries} retries. Last error: {e}")  # noqa: E501
                await asyncio.sleep(0.5)

    async def stop_server(self, name: str) -> None:
        """Stop an MCP test server."""
        if name in self.server_processes:
            process = self.server_processes[name]
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            del self.server_processes[name]

    async def cleanup(self) -> None:
        """Stop all test servers."""
        for name in list(self.server_processes.keys()):
            await self.stop_server(name)


@pytest_asyncio.fixture(scope="module")
async def test_servers() -> AsyncGenerator[MCPTestServerManager]:
    """Fixture that provides MCP test servers."""
    servers = MCPTestServerManager()

    # Start both test servers
    await servers.start_server(
        "server_a",
        8001,
        "tests/mcp_servers/server_a.py",
    )
    await servers.start_server(
        "server_b",
        8002,
        "tests/mcp_servers/server_b.py",
    )

    yield servers

    # Cleanup
    await servers.cleanup()


@pytest.fixture
def server_a_config() -> MCPServerConfig:
    """Configuration for test server A."""
    return MCPServerConfig(
        name="server_a",
        url="http://localhost:8001/mcp/",
        enabled=True,
    )


@pytest.fixture
def server_b_config() -> MCPServerConfig:
    """Configuration for test server B."""
    return MCPServerConfig(
        name="server_b",
        url="http://localhost:8002/mcp/",
        enabled=True,
    )


@pytest.fixture
def both_server_configs(server_a_config: MCPServerConfig, server_b_config: MCPServerConfig) -> list[MCPServerConfig]:  # noqa: E501
    """Configuration for both test servers."""
    return [server_a_config, server_b_config]


@pytest.fixture
def in_memory_server_a() -> MCPServerConfig:
    """In-memory configuration for test server A."""
    return MCPServerConfig(
        name="in_memory_server_a",
        url=None,  # Will use direct server instance
        enabled=True,
    )


@pytest.fixture
def in_memory_server_b() -> MCPServerConfig:
    """In-memory configuration for test server B."""
    return MCPServerConfig(
        name="in_memory_server_b",
        url=None,  # Will use direct server instance
        enabled=True,
    )


class TestMCPClient:
    """Test MCPClient functionality against a single server."""

    @pytest.mark.asyncio
    async def test__validate_connection__success(
        self,
        test_servers: MCPTestServerManager,  # noqa: ARG002
        server_a_config: MCPServerConfig,
    ):
        """Test successful connection validation."""
        client = MCPClient(server_a_config)
        await client.validate_connection()
        # No exception means success

    @pytest.mark.asyncio
    async def test__validate_connection__invalid_url(self):
        """Test connection validation with invalid URL."""
        config = MCPServerConfig(name="invalid", url="http://localhost:9999/mcp/")
        client = MCPClient(config)

        with pytest.raises(Exception):  # noqa: PT011
            await client.validate_connection()

    @pytest.mark.asyncio
    async def test__list_tools__success(
        self,
        test_servers: MCPTestServerManager,  # noqa: ARG002
        server_a_config: MCPServerConfig,
    ):
        """Test listing tools from server A."""
        client = MCPClient(server_a_config)
        tools = await client.list_tools()

        assert len(tools) == 3  # server_a test server has 3 tools
        tool_names = [tool.tool_name for tool in tools]
        assert "weather_api" in tool_names
        assert "web_search" in tool_names
        assert "failing_tool" in tool_names

        # Verify tool details
        weather_tool = next(tool for tool in tools if tool.tool_name == "weather_api")
        assert weather_tool.server_name == "server_a"
        assert "weather" in weather_tool.description.lower()

    @pytest.mark.asyncio
    async def test__call_tool__weather_api_success(
        self,
        test_servers: MCPTestServerManager,  # noqa: ARG002
        server_a_config: MCPServerConfig,
    ):
        """Test calling weather_api tool successfully."""
        client = MCPClient(server_a_config)
        result = await client.call_tool("weather_api", {"location": "Tokyo"})

        assert isinstance(result, dict)
        assert result["location"] == "Tokyo"
        assert "current" in result
        assert "forecast" in result
        assert result["server"] == "test-server-a"

        # Verify current weather structure
        current = result["current"]
        assert "temperature" in current
        assert "condition" in current
        assert "humidity" in current

    @pytest.mark.asyncio
    async def test__call_tool__web_search_success(
        self,
        test_servers: MCPTestServerManager,  # noqa: ARG002
        server_a_config: MCPServerConfig,
    ):
        """Test calling web_search tool successfully."""
        client = MCPClient(server_a_config)
        result = await client.call_tool("web_search", {"query": "machine learning"})

        assert isinstance(result, dict)
        assert result["query"] == "machine learning"
        assert "results" in result
        assert result["server"] == "test-server-a"
        assert len(result["results"]) > 0

        # Verify search result structure
        first_result = result["results"][0]
        assert "title" in first_result
        assert "url" in first_result
        assert "snippet" in first_result

    @pytest.mark.asyncio
    async def test__call_tool__nonexistent_tool(
        self,
        test_servers: MCPTestServerManager,  # noqa: ARG002
        server_a_config: MCPServerConfig,
    ):
        """Test calling a tool that doesn't exist raises ToolExecutionError."""
        client = MCPClient(server_a_config)

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.call_tool("nonexistent_tool", {})

        assert "nonexistent_tool" in str(exc_info.value)
        assert exc_info.value.tool_name == "nonexistent_tool"
        assert exc_info.value.server_name == "server_a"

    @pytest.mark.asyncio
    async def test__call_tool__invalid_arguments(
        self,
        test_servers: MCPTestServerManager,  # noqa: ARG002
        server_a_config: MCPServerConfig,
    ):
        """Test calling tool with invalid arguments raises ToolExecutionError."""
        client = MCPClient(server_a_config)

        # Test with arguments that would cause validation error (wrong type)
        # Pass an integer instead of string which should cause a validation error
        with pytest.raises(ToolExecutionError) as exc_info:
            await client.call_tool("weather_api", {"location": 12345})

        assert "weather_api" in str(exc_info.value)
        assert exc_info.value.tool_name == "weather_api"
        assert exc_info.value.server_name == "server_a"

    @pytest.mark.asyncio
    async def test__call_tool__failing_tool_raises_exception(
        self,
        test_servers: MCPTestServerManager,  # noqa: ARG002
        server_a_config: MCPServerConfig,
    ):
        """Test calling a tool that fails raises ToolExecutionError."""
        client = MCPClient(server_a_config)

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.call_tool("failing_tool", {"should_fail": True})

        assert "failing_tool" in str(exc_info.value)
        assert exc_info.value.tool_name == "failing_tool"
        assert exc_info.value.server_name == "server_a"
        assert "intentionally failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test__call_tool__failing_tool_success_when_should_fail_false(
        self,
        test_servers: MCPTestServerManager,  # noqa: ARG002
        server_a_config: MCPServerConfig,
    ):
        """Test calling failing_tool with should_fail=False succeeds."""
        client = MCPClient(server_a_config)
        result = await client.call_tool("failing_tool", {"should_fail": False})

        assert isinstance(result, dict)
        assert not result.get("error")  # Should not have error field or should be False
        assert result["success"] is True
        assert result["message"] == "Tool executed successfully"
        assert result["server"] == "test-server-a"


class TestMCPManager:
    """Test MCPManager functionality against multiple servers."""

    @pytest.mark.asyncio
    async def test__initialize__success_all_servers(
        self,
        test_servers: MCPTestServerManager,  # noqa: ARG002
        both_server_configs: list[MCPServerConfig],
    ):
        """Test successful initialization with all servers available."""
        manager = MCPManager(both_server_configs)
        await manager.initialize()

        connected_servers = manager.get_connected_servers()
        assert len(connected_servers) == 2
        assert "server_a" in connected_servers
        assert "server_b" in connected_servers

    @pytest.mark.asyncio
    async def test__initialize__fail_fast_on_unavailable_server(
        self,
        test_servers: MCPTestServerManager,  # noqa: ARG002
    ):
        """Test that initialization fails fast when any server is unavailable."""
        configs = [
            MCPServerConfig(name="server_a", url="http://localhost:8001/mcp/", enabled=True),
            MCPServerConfig(name="invalid", url="http://localhost:9999/mcp/", enabled=True),
        ]

        manager = MCPManager(configs)

        with pytest.raises(RuntimeError) as exc_info:
            await manager.initialize()

        assert "Failed to connect to MCP servers" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test__initialize__skip_disabled_servers(
        self,
        test_servers: MCPTestServerManager,  # noqa: ARG002
    ):
        """Test that disabled servers are skipped during initialization."""
        configs = [
            MCPServerConfig(name="server_a", url="http://localhost:8001/mcp/", enabled=True),
            MCPServerConfig(name="server_b", url="http://localhost:8002/mcp/", enabled=False),
        ]

        manager = MCPManager(configs)
        await manager.initialize()

        connected_servers = manager.get_connected_servers()
        assert len(connected_servers) == 1
        assert "server_a" in connected_servers
        assert "server_b" not in connected_servers

    @pytest.mark.asyncio
    async def test__get_available_tools__all_servers(
        self,
        test_servers: MCPTestServerManager,  # noqa: ARG002
        both_server_configs: list[MCPServerConfig],
    ):
        """Test getting tools from all connected servers."""
        manager = MCPManager(both_server_configs)
        await manager.initialize()

        tools = await manager.get_available_tools()

        assert len(tools) == 5  # 3 tools from server_a, 2 from server_b
        tool_names = [tool.tool_name for tool in tools]
        assert "weather_api" in tool_names
        assert "web_search" in tool_names
        assert "failing_tool" in tool_names
        assert "filesystem" in tool_names
        assert "search_news" in tool_names

        # Verify server attribution
        server_a_tools = [tool for tool in tools if tool.server_name == "server_a"]
        server_b_tools = [tool for tool in tools if tool.server_name == "server_b"]
        assert len(server_a_tools) == 3
        assert len(server_b_tools) == 2

    @pytest.mark.asyncio
    async def test__get_available_tools__caching(
        self,
        test_servers: MCPTestServerManager,  # noqa: ARG002
        both_server_configs: list[MCPServerConfig],
    ):
        """Test that tool discovery results are cached."""
        manager = MCPManager(both_server_configs)
        await manager.initialize()

        # First call - should discover tools
        start_time = time.time()
        tools1 = await manager.get_available_tools()
        first_call_time = time.time() - start_time

        # Second call - should use cache (faster)
        start_time = time.time()
        tools2 = await manager.get_available_tools()
        second_call_time = time.time() - start_time

        assert tools1 == tools2
        assert second_call_time < first_call_time  # Cache should be faster

    @pytest.mark.asyncio
    async def test__execute_tool__server_a_weather(
        self,
        test_servers: MCPTestServerManager,  # noqa: ARG002
        both_server_configs: list[MCPServerConfig],
    ):
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
    async def test__execute_tool__server_b_filesystem(
        self,
        test_servers: MCPTestServerManager,  # noqa: ARG002
        both_server_configs: list[MCPServerConfig],
    ):
        """Test executing filesystem tool on server B."""
        manager = MCPManager(both_server_configs)
        await manager.initialize()

        request = ToolRequest(
            server_name="server_b",
            tool_name="filesystem",
            arguments={"query": "python", "path": "/tmp"},
        )

        result = await manager.execute_tool(request)

        assert result.success is True
        assert result.server_name == "server_b"
        assert result.tool_name == "filesystem"
        assert result.error is None

        # Verify result data
        assert result.result["query"] == "python"
        assert result.result["search_path"] == "/tmp"
        assert result.result["server"] == "test-server-b"

    @pytest.mark.asyncio
    async def test__execute_tool__nonexistent_server(
        self,
        test_servers: MCPTestServerManager,  # noqa: ARG002
        both_server_configs: list[MCPServerConfig],
    ):
        """Test executing tool on a server that doesn't exist."""
        manager = MCPManager(both_server_configs)
        await manager.initialize()

        request = ToolRequest(
            server_name="nonexistent_server",
            tool_name="some_tool",
            arguments={},
        )

        result = await manager.execute_tool(request)

        assert result.success is False
        assert result.server_name == "nonexistent_server"
        assert result.tool_name == "some_tool"
        assert "not found or not connected" in result.error

    @pytest.mark.asyncio
    async def test__execute_tools_parallel__multiple_servers(
        self,
        test_servers: MCPTestServerManager,  # noqa: ARG002
        both_server_configs: list[MCPServerConfig],
    ):
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
            ToolRequest(
                server_name="server_a",
                tool_name="web_search",
                arguments={"query": "python"},
            ),
        ]

        start_time = time.time()
        results = await manager.execute_tools_parallel(requests)
        execution_time = time.time() - start_time

        assert len(results) == 3

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

        search_result = results[2]
        assert search_result.server_name == "server_a"
        assert search_result.result["query"] == "python"

        # Parallel execution should be faster than sequential
        # (though this is a rough check since test servers are fast)
        assert execution_time < 2.0  # Should complete quickly in parallel

    @pytest.mark.asyncio
    async def test__health_check__all_connected(
        self,
        test_servers: MCPTestServerManager,  # noqa: ARG002
        both_server_configs: list[MCPServerConfig],
    ):
        """Test health check with all servers connected."""
        manager = MCPManager(both_server_configs)
        await manager.initialize()

        # Populate tool cache
        await manager.get_available_tools()

        health = await manager.health_check()

        assert health["total_servers"] == 2
        assert health["connected_servers"] == 2

        assert "server_a" in health["servers"]
        assert "server_b" in health["servers"]

        server_a_health = health["servers"]["server_a"]
        assert server_a_health["enabled"] is True
        assert server_a_health["connected"] is True
        assert server_a_health["tools_cached"] is True
        assert server_a_health["tool_count"] == 3  # weather_api, web_search, failing_tool


class TestMCPInMemory:
    """Test MCP functionality using in-memory servers for fast unit tests."""

    @pytest.mark.asyncio
    async def test__in_memory_client__weather_api_success(
        self,
        in_memory_server_a: MCPServerConfig,
    ):
        """Test calling weather_api tool on in-memory server A."""
        client = MCPClient(in_memory_server_a)
        client.set_server_instance(get_server_a())

        result = await client.call_tool("weather_api", {"location": "Tokyo"})

        assert isinstance(result, dict)
        assert result["location"] == "Tokyo"
        assert "current" in result
        assert "forecast" in result
        assert result["server"] == "test-server-a"

    @pytest.mark.asyncio
    async def test__in_memory_client__web_search_success(
        self,
        in_memory_server_a: MCPServerConfig,
    ):
        """Test calling web_search tool on in-memory server A."""
        client = MCPClient(in_memory_server_a)
        client.set_server_instance(get_server_a())

        result = await client.call_tool("web_search", {"query": "machine learning"})

        assert isinstance(result, dict)
        assert result["query"] == "machine learning"
        assert "results" in result
        assert result["server"] == "test-server-a"
        assert len(result["results"]) > 0

    @pytest.mark.asyncio
    async def test__in_memory_client__failing_tool_exception(
        self,
        in_memory_server_a: MCPServerConfig,
    ):
        """Test that failing tool raises ToolExecutionError in in-memory mode."""
        client = MCPClient(in_memory_server_a)
        client.set_server_instance(get_server_a())

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.call_tool("failing_tool", {"should_fail": True})

        assert "failing_tool" in str(exc_info.value)
        assert "intentionally failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test__in_memory_client__filesystem_tool_success(
        self,
        in_memory_server_b: MCPServerConfig,
    ):
        """Test calling filesystem tool on in-memory server B."""
        client = MCPClient(in_memory_server_b)
        client.set_server_instance(get_server_b())

        result = await client.call_tool("filesystem", {"query": "python", "path": "/tmp"})

        assert isinstance(result, dict)
        assert result["query"] == "python"
        assert result["search_path"] == "/tmp"
        assert result["server"] == "test-server-b"
        assert "files_found" in result

    @pytest.mark.asyncio
    async def test__in_memory_client__list_tools(
        self,
        in_memory_server_a: MCPServerConfig,
    ):
        """Test listing tools from in-memory server."""
        client = MCPClient(in_memory_server_a)
        client.set_server_instance(get_server_a())

        tools = await client.list_tools()

        assert len(tools) == 3
        tool_names = [tool.tool_name for tool in tools]
        assert "weather_api" in tool_names
        assert "web_search" in tool_names
        assert "failing_tool" in tool_names


    @pytest.mark.asyncio
    async def test__in_memory_manager__mixed_servers(
        self,
        in_memory_server_a: MCPServerConfig,
        in_memory_server_b: MCPServerConfig,
    ):
        """Test MCPManager with in-memory servers."""
        # Create manager with in-memory servers
        manager = MCPManager([in_memory_server_a, in_memory_server_b])

        # Manually set up clients with server instances (since initialize() won't work with
        # in-memory)
        client_a = MCPClient(in_memory_server_a)
        client_a.set_server_instance(get_server_a())
        client_b = MCPClient(in_memory_server_b)
        client_b.set_server_instance(get_server_b())

        # Manually set clients in manager
        manager._clients["in_memory_server_a"] = client_a
        manager._clients["in_memory_server_b"] = client_b

        # Test tool execution
        request = ToolRequest(
            server_name="in_memory_server_a",
            tool_name="weather_api",
            arguments={"location": "London"},
        )

        result = await manager.execute_tool(request)

        assert result.success is True
        assert result.result["location"] == "London"
        assert result.server_name == "in_memory_server_a"


class TestMCPManagerNoServers:
    """Test MCPManager functionality when no servers are configured."""

    @pytest.mark.asyncio
    async def test__initialize__no_servers_configured(self):
        """Test MCPManager initialization with empty server list."""
        manager = MCPManager([])
        await manager.initialize()

        connected_servers = manager.get_connected_servers()
        assert len(connected_servers) == 0

    @pytest.mark.asyncio
    async def test__initialize__all_servers_disabled(self):
        """Test MCPManager initialization with all servers disabled."""
        configs = [
            MCPServerConfig(name="server_a", url="http://localhost:8001/mcp/", enabled=False),
            MCPServerConfig(name="server_b", url="http://localhost:8002/mcp/", enabled=False),
        ]

        manager = MCPManager(configs)
        await manager.initialize()

        connected_servers = manager.get_connected_servers()
        assert len(connected_servers) == 0

    @pytest.mark.asyncio
    async def test__get_available_tools__no_servers(self):
        """Test getting tools when no servers are configured."""
        manager = MCPManager([])
        await manager.initialize()

        tools = await manager.get_available_tools()
        assert len(tools) == 0
        assert tools == []

    @pytest.mark.asyncio
    async def test__get_available_tools__all_servers_disabled(self):
        """Test getting tools when all servers are disabled."""
        configs = [
            MCPServerConfig(name="server_a", url="http://localhost:8001/mcp/", enabled=False),
            MCPServerConfig(name="server_b", url="http://localhost:8002/mcp/", enabled=False),
        ]

        manager = MCPManager(configs)
        await manager.initialize()

        tools = await manager.get_available_tools()
        assert len(tools) == 0
        assert tools == []

    @pytest.mark.asyncio
    async def test__execute_tool__no_servers_returns_error(self):
        """Test executing tool when no servers are available."""
        manager = MCPManager([])
        await manager.initialize()

        request = ToolRequest(
            server_name="nonexistent_server",
            tool_name="some_tool",
            arguments={},
        )

        result = await manager.execute_tool(request)

        assert result.success is False
        assert result.server_name == "nonexistent_server"
        assert result.tool_name == "some_tool"
        assert "not found or not connected" in result.error

    @pytest.mark.asyncio
    async def test__execute_tools_parallel__no_servers(self):
        """Test parallel tool execution when no servers are available."""
        manager = MCPManager([])
        await manager.initialize()

        requests = [
            ToolRequest(
                server_name="server_a",
                tool_name="tool1",
                arguments={},
            ),
            ToolRequest(
                server_name="server_b",
                tool_name="tool2",
                arguments={},
            ),
        ]

        results = await manager.execute_tools_parallel(requests)

        assert len(results) == 2
        for result in results:
            assert result.success is False
            assert "not found or not connected" in result.error

    @pytest.mark.asyncio
    async def test__health_check__no_servers(self):
        """Test health check when no servers are configured."""
        manager = MCPManager([])
        await manager.initialize()

        health = await manager.health_check()

        assert health["total_servers"] == 0
        assert health["connected_servers"] == 0
        assert health["servers"] == {}

    @pytest.mark.asyncio
    async def test__health_check__all_servers_disabled(self):
        """Test health check when all servers are disabled."""
        configs = [
            MCPServerConfig(name="server_a", url="http://localhost:8001/mcp/", enabled=False),
            MCPServerConfig(name="server_b", url="http://localhost:8002/mcp/", enabled=False),
        ]

        manager = MCPManager(configs)
        await manager.initialize()

        health = await manager.health_check()

        assert health["total_servers"] == 2
        assert health["connected_servers"] == 0

        # Both servers should appear in health status but not be connected
        assert "server_a" in health["servers"]
        assert "server_b" in health["servers"]

        for server_name in ["server_a", "server_b"]:
            server_health = health["servers"][server_name]
            assert server_health["enabled"] is False
            assert server_health["connected"] is False
            assert server_health["tools_cached"] is False
