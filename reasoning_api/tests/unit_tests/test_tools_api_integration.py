"""
Unit tests for tools-api client integration in dependencies.

Tests the integration between ToolsAPIClient and the dependency injection system,
following the pattern established in test_dependencies.py.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

from reasoning_api.dependencies import (
    ServiceContainer,
    get_tools_from_tools_api,
)
from reasoning_api.tools import Tool
from reasoning_api.tools_client import ToolsAPIClient, ToolDefinition
from reasoning_api.config import settings


class TestToolsAPIClientIntegration:
    """Test tools-api client integration with dependency system."""

    @pytest_asyncio.fixture(scope="function")
    async def clean_container(self):
        """Provide a clean service container for testing."""
        container = ServiceContainer()
        yield container
        # Cleanup after test
        try:
            await container.cleanup()
        except RuntimeError as e:
            # Ignore "Event loop is closed" errors during cleanup
            if "Event loop is closed" not in str(e):
                raise

    @pytest.mark.asyncio
    async def test__get_tools_from_tools_api__creates_tools_with_client(self) -> None:
        """Test that get_tools_from_tools_api creates Tool objects with tools_api_client set."""
        # Create mock tools-api client
        mock_client = AsyncMock(spec=ToolsAPIClient)
        mock_client.list_tools.return_value = [
            ToolDefinition(
                name="read_text_file",
                description="Read a text file",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                tags=["filesystem", "read"],
            ),
            ToolDefinition(
                name="write_file",
                description="Write a file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
                tags=["filesystem", "write"],
            ),
        ]

        # Patch service_container to return our mock client
        with patch('reasoning_api.dependencies.service_container') as mock_container:
            mock_container.tools_api_client = mock_client

            # Call get_tools_from_tools_api
            tools = await get_tools_from_tools_api()

            # Verify tools were created correctly
            assert len(tools) == 2

            # Verify first tool
            assert tools[0].name == "read_text_file"
            assert tools[0].description == "Read a text file"
            assert tools[0].tags == ["filesystem", "read"]
            assert tools[0].tools_api_client is mock_client
            assert tools[0].function is None  # No direct callable

            # Verify second tool
            assert tools[1].name == "write_file"
            assert tools[1].description == "Write a file"
            assert tools[1].tags == ["filesystem", "write"]
            assert tools[1].tools_api_client is mock_client

    @pytest.mark.asyncio
    async def test__get_tools_from_tools_api__handles_no_client_gracefully(self) -> None:
        """Test that get_tools_from_tools_api returns empty list when client unavailable."""
        # Patch service_container to return None client
        with patch('reasoning_api.dependencies.service_container') as mock_container:
            mock_container.tools_api_client = None

            # Call get_tools_from_tools_api
            tools = await get_tools_from_tools_api()

            # Should return empty list, not raise
            assert tools == []

    @pytest.mark.asyncio
    async def test__get_tools_from_tools_api__handles_api_errors_gracefully(self) -> None:
        """Test that get_tools_from_tools_api handles API errors gracefully."""
        import httpx

        # Create mock client that raises error
        mock_client = AsyncMock(spec=ToolsAPIClient)
        mock_client.list_tools.side_effect = httpx.ConnectError("Connection refused")

        # Patch service_container to return error-prone client
        with patch('reasoning_api.dependencies.service_container') as mock_container:
            mock_container.tools_api_client = mock_client

            # Call get_tools_from_tools_api - should handle error gracefully
            tools = await get_tools_from_tools_api()

            # Should return empty list, not raise
            assert tools == []

    @pytest.mark.asyncio
    async def test__service_container__initializes_tools_api_client(
        self,
        clean_container: ServiceContainer,
    ) -> None:
        """Test that ServiceContainer initializes tools-api client during startup."""
        # Mock the health check to succeed
        with patch.object(ToolsAPIClient, 'health_check', new_callable=AsyncMock) as mock_health:
            mock_health.return_value = {"status": "healthy"}

            await clean_container.initialize()

            # Verify tools_api_client was created
            assert clean_container.tools_api_client is not None
            assert isinstance(clean_container.tools_api_client, ToolsAPIClient)

            # Verify it used settings.tools_api_url
            assert clean_container.tools_api_client.base_url == settings.tools_api_url.rstrip("/")

            # Verify health check was called
            mock_health.assert_called_once()

    @pytest.mark.asyncio
    async def test__service_container__handles_tools_api_unavailable(
        self,
        clean_container: ServiceContainer,
    ) -> None:
        """Test that ServiceContainer continues when tools-api is unavailable."""
        import httpx

        # Mock the health check to fail
        with patch.object(
            ToolsAPIClient,
            'health_check',
            new_callable=AsyncMock,
        ) as mock_health:
            mock_health.side_effect = httpx.ConnectError("Connection refused")

            await clean_container.initialize()

            # Should not raise - tools_api_client should be None
            assert clean_container.tools_api_client is None

    @pytest.mark.asyncio
    async def test__service_container__cleanup_closes_tools_api_client(
        self,
        clean_container: ServiceContainer,
    ) -> None:
        """Test that ServiceContainer cleanup properly closes tools-api client."""
        # Mock successful initialization
        with patch.object(ToolsAPIClient, 'health_check', new_callable=AsyncMock):
            await clean_container.initialize()

            # Mock the close method
            close_mock = AsyncMock()
            clean_container.tools_api_client.close = close_mock

            # Cleanup
            await clean_container.cleanup()

            # Verify close was called
            close_mock.assert_called_once()


class TestToolExecutionViaToolsAPIClient:
    """Test Tool class execution via tools-api client (unit level)."""

    @pytest.mark.asyncio
    async def test__tool_executes_via_tools_api_client(self) -> None:
        """Test that Tool.__call__() uses tools_api_client when available."""
        # Create mock tools-api client
        from reasoning_api.tools import ToolResult

        mock_client = AsyncMock(spec=ToolsAPIClient)
        mock_client.execute_tool.return_value = ToolResult(
            tool_name="read_text_file",
            success=True,
            result={"content": "test content", "size_bytes": 12},
            execution_time_ms=10.5,
        )

        # Create tool with tools-api client
        tool = Tool(
            name="read_text_file",
            description="Read a text file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            tools_api_client=mock_client,
        )

        # Execute tool
        result = await tool(path="/test/file.txt")

        # Verify client was called
        mock_client.execute_tool.assert_called_once_with(
            "read_text_file",
            {"path": "/test/file.txt"},
        )

        # Verify result
        assert result.success is True
        assert result.result["content"] == "test content"

    @pytest.mark.asyncio
    async def test__tool_prefers_tools_api_client_over_function(self) -> None:
        """Test that Tool prefers tools_api_client over direct function."""
        from reasoning_api.tools import ToolResult

        # Create mock tools-api client
        mock_client = AsyncMock(spec=ToolsAPIClient)
        mock_client.execute_tool.return_value = ToolResult(
            tool_name="test_tool",
            success=True,
            result="from tools-api",
            execution_time_ms=10.0,
        )

        # Create mock function that should NOT be called
        mock_function = AsyncMock(return_value="from function")

        # Create tool with BOTH client and function
        tool = Tool(
            name="test_tool",
            description="Test tool",
            input_schema={"type": "object", "properties": {}},
            tools_api_client=mock_client,
            function=mock_function,
        )

        # Execute tool
        result = await tool()

        # Verify client was used, function was not
        mock_client.execute_tool.assert_called_once()
        mock_function.assert_not_called()
        assert result.result == "from tools-api"
