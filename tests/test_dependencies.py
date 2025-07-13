"""
Tests for dependency injection system.

Tests the FastAPI dependency injection system, service container lifecycle,
and error handling when dependencies are not properly initialized.
"""

import pytest
import pytest_asyncio
import httpx
from unittest.mock import AsyncMock, patch

from api.dependencies import (
    ServiceContainer,
    service_container,
    get_http_client,
    get_mcp_manager,
    get_prompt_manager,
    get_reasoning_agent,
    create_production_http_client,
)
from api.config import settings


class TestHTTPClientConfiguration:
    """Test HTTP client production configuration."""

    def test__create_production_http_client__has_proper_timeouts(self):
        """Test that production HTTP client has proper timeout configuration."""
        client = create_production_http_client()

        # Verify timeout configuration
        assert client.timeout.connect == settings.http_connect_timeout
        assert client.timeout.read == settings.http_read_timeout
        assert client.timeout.write == settings.http_write_timeout

        # Verify connection limits through the transport pool
        pool = client._transport._pool
        assert pool._max_connections == settings.http_max_connections
        assert pool._max_keepalive_connections == settings.http_max_keepalive_connections
        assert pool._keepalive_expiry == settings.http_keepalive_expiry

    def test__create_production_http_client__uses_settings_values(self):
        """Test that HTTP client respects custom settings values."""
        # Save original values
        original_connect = settings.http_connect_timeout
        original_read = settings.http_read_timeout
        original_max_conn = settings.http_max_connections

        try:
            # Temporarily modify settings
            settings.http_connect_timeout = 10.0
            settings.http_read_timeout = 60.0
            settings.http_max_connections = 50

            client = create_production_http_client()

            # Verify client uses modified values
            assert client.timeout.connect == 10.0
            assert client.timeout.read == 60.0
            assert client._transport._pool._max_connections == 50
        finally:
            # Restore original values
            settings.http_connect_timeout = original_connect
            settings.http_read_timeout = original_read
            settings.http_max_connections = original_max_conn

    @pytest.mark.asyncio
    async def test__production_http_client__is_properly_configured(self):
        """Test that production HTTP client is properly configured for real use."""
        client = create_production_http_client()

        try:
            # Verify it's an AsyncClient
            assert isinstance(client, httpx.AsyncClient)

            # Verify timeout is not the old 60.0 default
            assert client.timeout.connect != 60.0
            assert client.timeout.connect == 5.0  # Fast failure

            # Verify connection pooling is enabled
            pool = client._transport._pool
            assert pool._max_connections > 0
            assert pool._max_keepalive_connections > 0
        finally:
            await client.aclose()


class TestServiceContainer:
    """Test ServiceContainer lifecycle management."""

    @pytest_asyncio.fixture
    async def clean_container(self):
        """Provide a clean service container for testing."""
        container = ServiceContainer()
        yield container
        # Cleanup after test
        await container.cleanup()

    @pytest.mark.asyncio
    async def test__initialize__sets_up_services_correctly(self, clean_container: ServiceContainer):  # noqa: E501
        """Test that initialize sets up all services correctly."""
        await clean_container.initialize()

        # HTTP client should be created with production configuration
        assert clean_container.http_client is not None
        assert clean_container.http_client.timeout.connect == settings.http_connect_timeout
        assert clean_container.http_client.timeout.read == settings.http_read_timeout
        assert clean_container.http_client.timeout.write == settings.http_write_timeout

        # MCP manager should be initialized
        assert clean_container.mcp_manager is not None

    @pytest.mark.asyncio
    async def test__cleanup__closes_services_properly(self, clean_container: ServiceContainer):
        """Test that cleanup properly closes all services."""
        await clean_container.initialize()

        # Mock the close methods to verify they're called
        http_close_mock = AsyncMock()
        clean_container.http_client.aclose = http_close_mock

        await clean_container.cleanup()

        # Verify cleanup was called
        http_close_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test__cleanup__handles_none_services_gracefully(self, clean_container: ServiceContainer):  # noqa: E501
        """Test that cleanup handles None services without errors."""
        # Don't initialize, so services remain None
        assert clean_container.http_client is None
        assert clean_container.mcp_manager is None

        # Should not raise any errors
        await clean_container.cleanup()


class TestDependencyInjection:
    """Test dependency injection functions."""

    @pytest.mark.asyncio
    async def test__get_http_client__returns_client_when_initialized(self):
        """Test that get_http_client returns client when container is initialized."""
        # Save original state
        original_client = service_container.http_client

        try:
            # Mock an initialized client
            mock_client = AsyncMock()
            service_container.http_client = mock_client

            result = await get_http_client()

            assert result is mock_client
        finally:
            # Restore original state
            service_container.http_client = original_client

    @pytest.mark.asyncio
    async def test__get_http_client__raises_error_when_not_initialized(self):
        """Test that get_http_client raises RuntimeError when container not initialized."""
        # Save original state
        original_client = service_container.http_client

        try:
            # Set to None to simulate uninitialized state
            service_container.http_client = None

            with pytest.raises(RuntimeError) as exc_info:
                await get_http_client()

            error_message = str(exc_info.value)
            assert "Service container not initialized" in error_message
            assert "HTTP client should be available after app startup" in error_message
            assert "If testing, ensure service_container.initialize() is called" in error_message
        finally:
            # Restore original state
            service_container.http_client = original_client

    @pytest.mark.asyncio
    async def test__get_mcp_manager__returns_manager_when_initialized(self):
        """Test that get_mcp_manager returns the MCP manager when initialized."""
        # Save original state
        original_manager = service_container.mcp_manager

        try:
            # Test with mock manager
            mock_manager = AsyncMock()
            service_container.mcp_manager = mock_manager

            result = await get_mcp_manager()
            assert result is mock_manager
        finally:
            # Restore original state
            service_container.mcp_manager = original_manager

    @pytest.mark.asyncio
    async def test__get_reasoning_agent__creates_agent_with_dependencies(self):
        """Test that get_reasoning_agent creates agent with proper dependencies."""
        # Save original state
        original_http = service_container.http_client
        original_mcp_manager = service_container.mcp_manager
        original_prompt_initialized = service_container.prompt_manager_initialized

        try:
            # Mock dependencies - use real HTTP client for OpenAI compatibility
            real_http_client = httpx.AsyncClient()
            mock_mcp_manager = AsyncMock()

            service_container.http_client = real_http_client
            service_container.mcp_manager = mock_mcp_manager
            service_container.prompt_manager_initialized = True

            # Mock settings to avoid real values
            with patch('api.dependencies.settings') as mock_settings:
                mock_settings.reasoning_agent_base_url = "https://test.api.com/v1"
                mock_settings.openai_api_key = "test-key"

                # Get reasoning agent through dependency injection
                http_client = await get_http_client()
                mcp_manager = await get_mcp_manager()
                prompt_manager = await get_prompt_manager()
                agent = await get_reasoning_agent(
                    http_client,
                    mcp_manager,
                    prompt_manager,
                )

                # Verify agent was created with correct dependencies
                assert agent is not None
                assert agent.http_client is real_http_client
                assert agent.mcp_manager is mock_mcp_manager
                assert agent.prompt_manager is not None

                # Clean up the HTTP client
                await real_http_client.aclose()
        finally:
            # Restore original state
            service_container.http_client = original_http
            service_container.mcp_manager = original_mcp_manager
            service_container.prompt_manager_initialized = original_prompt_initialized

    @pytest.mark.asyncio
    async def test__get_reasoning_agent__fails_when_http_client_not_initialized(self):
        """Test that get_reasoning_agent fails when HTTP client is not initialized."""
        # Save original state
        original_http = service_container.http_client

        try:
            # Set HTTP client to None
            service_container.http_client = None

            # Should raise error when trying to get HTTP client dependency
            with pytest.raises(RuntimeError) as exc_info:
                await get_http_client()

            assert "Service container not initialized" in str(exc_info.value)
        finally:
            # Restore original state
            service_container.http_client = original_http


class TestResourceLeakPrevention:
    """Test that the fix prevents resource leaks."""

    @pytest.mark.asyncio
    async def test__no_unmanaged_httpx_clients_created(self):
        """Test that unmanaged httpx clients are never created."""
        # Save original state
        original_client = service_container.http_client

        try:
            # Set to None to simulate uninitialized state
            service_container.http_client = None

            # Track httpx.AsyncClient creation
            with patch('httpx.AsyncClient') as mock_async_client:
                mock_async_client.return_value = AsyncMock()

                # Should raise error without creating any clients
                with pytest.raises(RuntimeError):
                    await get_http_client()

                # Verify no httpx.AsyncClient was created
                mock_async_client.assert_not_called()
        finally:
            # Restore original state
            service_container.http_client = original_client

    @pytest.mark.asyncio
    async def test__service_container_lifecycle_prevents_leaks(self):
        """Test that proper service container lifecycle prevents resource leaks."""
        container = ServiceContainer()

        # Track httpx.AsyncClient creation and closure
        with patch('httpx.AsyncClient') as mock_async_client_class, \
             patch('api.mcp.Client') as mock_mcp_client:

            # Create properly configured mocks
            mock_client = AsyncMock()
            mock_client.aclose = AsyncMock()
            mock_client.stream = AsyncMock()
            mock_async_client_class.return_value = mock_client

            # Mock the MCP client to avoid internal HTTP client creation
            mock_mcp_instance = AsyncMock()
            mock_mcp_instance.__aenter__ = AsyncMock(return_value=mock_mcp_instance)
            mock_mcp_instance.__aexit__ = AsyncMock(return_value=None)
            mock_mcp_client.return_value = mock_mcp_instance

            # Initialize - creates multiple clients:
            # 1. Main production HTTP client
            # 2. FastMCP clients for each MCP server (from config)
            await container.initialize()

            # Should create at least the main client
            assert mock_async_client_class.call_count >= 1

            # Cleanup - should close all clients that were created
            await container.cleanup()

            # Verify that aclose was called at least once
            assert mock_client.aclose.call_count >= 1


class TestMCPConfigurationPath:
    """Test MCP configuration path functionality."""

    @pytest.mark.asyncio
    async def test__service_container_uses_settings_mcp_config_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path):  # noqa: ANN001, E501
        """Test that ServiceContainer uses settings.mcp_config_path."""
        # Create a test config file
        test_config = tmp_path / "test_mcp_config.yaml"
        test_config.write_text("""
servers:
  - name: test_server
    url: http://localhost:8001/mcp/
    enabled: true
""")

        # Set the MCP_CONFIG_PATH environment variable
        monkeypatch.setenv("MCP_CONFIG_PATH", str(test_config))

        # Create a new ServiceContainer to test
        container = ServiceContainer()

        try:
            # Initialize should use the custom config path
            await container.initialize()

            # Verify MCP manager was created (even if connection fails)
            assert container.mcp_manager is not None

        finally:
            await container.cleanup()

    @pytest.mark.asyncio
    async def test__service_container_handles_nonexistent_config_file(self, monkeypatch):  # noqa: ANN001
        """Test ServiceContainer gracefully handles nonexistent config file."""
        # Set path to nonexistent file
        monkeypatch.setenv("MCP_CONFIG_PATH", "nonexistent/path/config.yaml")

        # Create a new ServiceContainer
        container = ServiceContainer()

        try:
            # Should not raise exception, should create empty manager
            await container.initialize()

            # Should create empty MCP manager
            assert container.mcp_manager is not None

        finally:
            await container.cleanup()

    @pytest.mark.asyncio
    async def test__service_container_supports_json_config(self, monkeypatch, tmp_path):  # noqa: ANN001
        """Test ServiceContainer supports JSON config files."""
        # Create a test JSON config file
        test_config = tmp_path / "test_mcp_config.json"
        test_config.write_text("""
{
  "servers": [
    {
      "name": "test_json_server",
      "url": "http://localhost:8001/mcp/",
      "enabled": true
    }
  ]
}
""")

        # Set the MCP_CONFIG_PATH environment variable
        monkeypatch.setenv("MCP_CONFIG_PATH", str(test_config))

        # Create a new ServiceContainer
        container = ServiceContainer()

        try:
            # Initialize should use the JSON config
            await container.initialize()

            # Verify MCP manager was created
            assert container.mcp_manager is not None

        finally:
            await container.cleanup()
