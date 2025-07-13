"""Comprehensive tests for MCPServerManager."""
import asyncio
from unittest.mock import patch
import pytest
from pydantic import ValidationError
import mcp.types

from api.mcp_manager import MCPServerManager
from api.reasoning_models import MCPServerConfig, ToolRequest


class MockMCPClient:
    """Mock MCP client that mimics fastmcp.Client behavior."""

    def __init__(self, name: str, should_fail: bool = False):
        self.name = name
        self.should_fail = should_fail
        self.connected = False

    async def __aenter__(self):
        if self.should_fail:
            raise Exception("Client failed to connect: All connection attempts failed")
        self.connected = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa
        self.connected = False

    async def list_tools(self):
        if not self.connected:
            raise Exception("Client not connected")

        # Return proper mcp.types.Tool objects
        tool = mcp.types.Tool(
            name=f"test_tool_{self.name}",
            description=f"Test tool from {self.name}",
            inputSchema={"type": "object", "properties": {}},
        )
        return [tool]

    async def call_tool(self, name: str, arguments: dict):
        if not self.connected:
            raise Exception("Client not connected")

        if name == "failing_tool":
            raise Exception("Tool execution failed")

        # Return proper mcp.types.CallToolResult object
        # Content should be a list of content items, so we wrap the result in TextContent
        content_data = {"result": f"success from {self.name}", "arguments": arguments}
        text_content = mcp.types.TextContent(
            type="text",
            text=str(content_data),
        )

        return mcp.types.CallToolResult(
            content=[text_content],
            structuredContent=content_data,
            isError=False,
        )


@pytest.fixture
def http_config():
    """HTTP server configuration for testing."""
    return MCPServerConfig(
        name="test_http",
        url="http://localhost:8001",
    )


@pytest.fixture
def https_config():
    """HTTPS server configuration for testing."""
    return MCPServerConfig(
        name="test_https",
        url="https://test.example.com/mcp",
        auth_env_var="TEST_API_KEY",
    )


@pytest.fixture
def disabled_config():
    """Disabled server configuration for testing."""
    return MCPServerConfig(
        name="disabled_server",
        url="https://disabled.example.com",
        enabled=False,
    )


class TestMCPServerManagerInitialization:
    """Test MCP server manager initialization and configuration."""

    @pytest.mark.asyncio
    async def test_empty_configuration(self):
        """Test manager with no servers configured."""
        manager = MCPServerManager([])
        await manager.initialize()

        assert len(manager.get_connected_servers()) == 0
        tools = await manager.get_available_tools()
        assert len(tools) == 0

    @pytest.mark.asyncio
    async def test_disabled_servers_ignored(self, disabled_config: MCPServerConfig):
        """Test that disabled servers are not initialized."""
        manager = MCPServerManager([disabled_config])
        await manager.initialize()

        assert len(manager.get_connected_servers()) == 0
        assert not manager.is_server_connected("disabled_server")

    @pytest.mark.asyncio
    @patch("api.mcp_manager.Client")
    async def test_successful_http_connection(self, mock_client_class, http_config):  # noqa
        """Test successful HTTP server connection."""
        # Make the mock return a new MockMCPClient for each call
        mock_client_class.side_effect = lambda url: MockMCPClient("test_http", should_fail=False)  # noqa: ARG005

        manager = MCPServerManager([http_config])
        await manager.initialize()

        assert manager.is_server_connected("test_http")
        assert "test_http" in manager.get_connected_servers()

        # Verify client was created with correct URL
        mock_client_class.assert_called_with("http://localhost:8001")

    @pytest.mark.asyncio
    @patch("api.mcp_manager.Client")
    async def test_connection_failure_graceful_handling(self, mock_client_class, https_config):  # noqa
        """Test graceful handling of connection failures."""
        mock_client = MockMCPClient("test_https", should_fail=True)
        mock_client_class.return_value = mock_client

        manager = MCPServerManager([https_config])
        await manager.initialize()

        assert not manager.is_server_connected("test_https")
        assert len(manager.get_connected_servers()) == 0

    @pytest.mark.asyncio
    @patch("api.mcp_manager.Client")
    async def test_multiple_servers_mixed_success(self, mock_client_class):  # noqa: ANN001
        """Test initialization with multiple servers where some fail."""
        configs = [
            MCPServerConfig(name="server1", url="http://localhost:8001"),
            MCPServerConfig(name="server2", url="http://localhost:8002"),
            MCPServerConfig(name="server3", url="http://localhost:8003"),
        ]

        # Mock servers with different behaviors
        mock_servers = [
            MockMCPClient("server1", should_fail=False),
            MockMCPClient("server2", should_fail=True),
            MockMCPClient("server3", should_fail=False),
        ]
        mock_client_class.side_effect = mock_servers

        manager = MCPServerManager(configs)
        await manager.initialize()

        connected_servers = manager.get_connected_servers()
        assert len(connected_servers) == 2
        assert "server1" in connected_servers
        assert "server3" in connected_servers
        assert "server2" not in connected_servers


class TestToolDiscovery:
    """Test tool discovery functionality."""

    @pytest.mark.asyncio
    @patch("api.mcp_manager.Client")
    async def test_tool_discovery_single_server(self, mock_client_class, http_config):  # noqa
        """Test tool discovery from a single server."""
        mock_client = MockMCPClient("test_http")
        mock_client_class.return_value = mock_client

        manager = MCPServerManager([http_config])
        await manager.initialize()

        tools = await manager.get_available_tools()
        assert len(tools) == 1
        assert tools[0].server_name == "test_http"
        assert tools[0].tool_name == "test_tool_test_http"
        assert tools[0].description == "Test tool from test_http"

    @pytest.mark.asyncio
    @patch("api.mcp_manager.Client")
    async def test_tool_discovery_multiple_servers(self, mock_client_class):  # noqa: ANN001
        """Test tool discovery from multiple servers."""
        configs = [
            MCPServerConfig(name="server1", url="http://localhost:8001"),
            MCPServerConfig(name="server2", url="http://localhost:8002"),
        ]

        # Create a function that returns appropriate mock based on URL
        def create_mock_client(url):  # noqa
            if url == "http://localhost:8001":
                return MockMCPClient("server1")
            if url == "http://localhost:8002":
                return MockMCPClient("server2")
            return MockMCPClient("unknown")

        mock_client_class.side_effect = create_mock_client

        manager = MCPServerManager(configs)
        await manager.initialize()

        tools = await manager.get_available_tools()
        assert len(tools) == 2

        tool_names = {tool.tool_name for tool in tools}
        assert "test_tool_server1" in tool_names
        assert "test_tool_server2" in tool_names

    @pytest.mark.asyncio
    @patch("api.mcp_manager.Client")
    async def test_tool_discovery_with_failed_server(self, mock_client_class):  # noqa: ANN001
        """Test tool discovery when one server fails."""
        configs = [
            MCPServerConfig(name="working", url="http://localhost:8001"),
            MCPServerConfig(name="failing", url="http://localhost:8002"),
        ]

        def create_mock_client(url):  # noqa
            if url == "http://localhost:8001":
                return MockMCPClient("working")
            if url == "http://localhost:8002":
                return MockMCPClient("failing", should_fail=True)
            return MockMCPClient("unknown")

        mock_client_class.side_effect = create_mock_client

        manager = MCPServerManager(configs)
        await manager.initialize()

        tools = await manager.get_available_tools()
        assert len(tools) == 1
        assert tools[0].server_name == "working"

    @pytest.mark.asyncio
    @patch("api.mcp_manager.Client")
    async def test_tool_cache_behavior(self, mock_client_class, http_config):  # noqa
        """Test tool caching and refresh behavior."""
        mock_client = MockMCPClient("test_http")
        mock_client_class.return_value = mock_client

        manager = MCPServerManager([http_config])
        manager._cache_timeout = 0.1  # Short timeout for testing
        await manager.initialize()

        # First call should discover tools
        tools1 = await manager.get_available_tools()
        assert len(tools1) == 1

        # Second call should use cache
        tools2 = await manager.get_available_tools()
        assert len(tools2) == 1

        # Wait for cache timeout and force refresh
        await asyncio.sleep(0.2)
        tools3 = await manager.get_available_tools(force_refresh=True)
        assert len(tools3) == 1


class TestToolExecution:
    """Test tool execution functionality."""

    @pytest.mark.asyncio
    @patch("api.mcp_manager.Client")
    async def test_successful_tool_execution(self, mock_client_class, http_config):  # noqa
        """Test successful execution of a single tool."""
        mock_client = MockMCPClient("test_http")
        mock_client_class.return_value = mock_client

        manager = MCPServerManager([http_config])
        await manager.initialize()

        request = ToolRequest(
            server_name="test_http",
            tool_name="test_tool",
            arguments={"param": "value"},
            reasoning="Test execution",
        )

        result = await manager.execute_tool(request)

        assert result.success is True
        assert result.server_name == "test_http"
        assert result.tool_name == "test_tool"
        assert result.result["result"] == "success from test_http"
        assert result.result["arguments"] == {"param": "value"}
        assert result.execution_time_ms > 0
        assert result.error is None

    @pytest.mark.asyncio
    @patch("api.mcp_manager.Client")
    async def test_tool_execution_failure(self, mock_client_class, http_config):  # noqa
        """Test handling of tool execution failures."""
        mock_client = MockMCPClient("test_http")
        mock_client_class.return_value = mock_client

        manager = MCPServerManager([http_config])
        await manager.initialize()

        request = ToolRequest(
            server_name="test_http",
            tool_name="failing_tool",
            arguments={},
            reasoning="Test failure",
        )

        result = await manager.execute_tool(request)

        assert result.success is False
        assert result.error == "Tool execution failed"
        assert result.execution_time_ms > 0
        assert result.result is None

    @pytest.mark.asyncio
    async def test_tool_execution_server_not_connected(self):
        """Test tool execution when server is not connected."""
        manager = MCPServerManager([])
        await manager.initialize()

        request = ToolRequest(
            server_name="nonexistent",
            tool_name="test_tool",
            arguments={},
            reasoning="Test missing server",
        )

        result = await manager.execute_tool(request)

        assert result.success is False
        assert "not found or disabled" in result.error
        assert result.server_name == "nonexistent"
        assert result.tool_name == "test_tool"

    @pytest.mark.asyncio
    @patch("api.mcp_manager.Client")
    async def test_parallel_tool_execution(self, mock_client_class):  # noqa
        """Test parallel execution of multiple tools."""
        configs = [
            MCPServerConfig(name="server1", url="http://localhost:8001"),
            MCPServerConfig(name="server2", url="http://localhost:8002"),
        ]

        def create_mock_client(url):  # noqa
            if url == "http://localhost:8001":
                return MockMCPClient("server1")
            if url == "http://localhost:8002":
                return MockMCPClient("server2")
            return MockMCPClient("unknown")

        mock_client_class.side_effect = create_mock_client

        manager = MCPServerManager(configs)
        await manager.initialize()

        requests = [
            ToolRequest(
                server_name="server1",
                tool_name="tool1",
                arguments={"param": "value1"},
                reasoning="First tool",
            ),
            ToolRequest(
                server_name="server2",
                tool_name="tool2",
                arguments={"param": "value2"},
                reasoning="Second tool",
            ),
        ]

        results = await manager.execute_tools_parallel(requests)

        assert len(results) == 2
        assert all(result.success for result in results)
        assert results[0].server_name == "server1"
        assert results[1].server_name == "server2"
        assert results[0].result["arguments"]["param"] == "value1"
        assert results[1].result["arguments"]["param"] == "value2"

    @pytest.mark.asyncio
    @patch("api.mcp_manager.Client")
    async def test_parallel_execution_with_failures(self, mock_client_class):  # noqa: ANN001
        """Test parallel execution when some tools fail."""
        mock_client = MockMCPClient("test_server")
        mock_client_class.return_value = mock_client

        config = MCPServerConfig(name="test_server", url="http://localhost:8001")
        manager = MCPServerManager([config])
        await manager.initialize()

        requests = [
            ToolRequest(
                server_name="test_server",
                tool_name="working_tool",
                arguments={},
                reasoning="Working tool",
            ),
            ToolRequest(
                server_name="test_server",
                tool_name="failing_tool",
                arguments={},
                reasoning="Failing tool",
            ),
        ]

        results = await manager.execute_tools_parallel(requests)

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False
        assert results[1].error == "Tool execution failed"

    @pytest.mark.asyncio
    async def test_parallel_execution_empty_list(self):
        """Test parallel execution with empty request list."""
        manager = MCPServerManager([])
        await manager.initialize()

        results = await manager.execute_tools_parallel([])
        assert len(results) == 0


class TestHealthAndManagement:
    """Test health monitoring and management functionality."""

    @pytest.mark.asyncio
    @patch("api.mcp_manager.Client")
    async def test_health_check(self, mock_client_class):  # noqa: ANN001
        """Test health check functionality."""
        configs = [
            MCPServerConfig(name="working", url="http://localhost:8001"),
            MCPServerConfig(name="failing", url="http://localhost:8002"),
            MCPServerConfig(name="disabled", url="http://localhost:8003", enabled=False),
        ]

        def create_mock_client(url):  # noqa
            if url == "http://localhost:8001":
                return MockMCPClient("working")
            if url == "http://localhost:8002":
                return MockMCPClient("failing", should_fail=True)
            return MockMCPClient("unknown")

        mock_client_class.side_effect = create_mock_client

        manager = MCPServerManager(configs)
        await manager.initialize()

        # Discover tools to populate cache
        await manager.get_available_tools()

        health = await manager.health_check()

        assert health["total_servers"] == 3
        assert health["connected_servers"] == 1

        assert health["servers"]["working"]["enabled"] is True
        assert health["servers"]["working"]["connected"] is True
        assert health["servers"]["working"]["tool_count"] == 1

        assert health["servers"]["failing"]["enabled"] is True
        assert health["servers"]["failing"]["connected"] is False

        assert health["servers"]["disabled"]["enabled"] is False
        assert health["servers"]["disabled"]["connected"] is False

    @pytest.mark.asyncio
    @patch("api.mcp_manager.Client")
    async def test_cleanup(self, mock_client_class, http_config):  # noqa
        """Test proper cleanup of resources."""
        mock_client = MockMCPClient("test_http")
        mock_client_class.return_value = mock_client

        manager = MCPServerManager([http_config])
        await manager.initialize()

        assert len(manager.get_connected_servers()) == 1

        await manager.cleanup()

        assert len(manager.get_connected_servers()) == 0


class TestConfigurationValidation:
    """Test configuration validation and edge cases."""

    def test_missing_url(self):
        """Test configuration without URL."""
        # This should be caught by Pydantic validation before reaching MCPServerManager
        with pytest.raises(ValidationError):
            MCPServerConfig(name="no_url")

    def test_client_creation_with_invalid_url(self):
        """Test client creation with missing URL."""
        config = MCPServerConfig(
            name="test",
            url="http://localhost:8001",
        )

        manager = MCPServerManager([config])

        # Valid config should work
        client = manager._create_client(config)
        assert client is not None

        # Test invalid config (manually set url to None to test error handling)
        config.url = None
        with pytest.raises(ValueError, match="requires a URL"):
            manager._create_client(config)
