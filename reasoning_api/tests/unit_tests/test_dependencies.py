"""
Tests for dependency injection system.

Tests the FastAPI dependency injection system, service container lifecycle,
and error handling when dependencies are not properly initialized.
"""

import pytest
import pytest_asyncio
import httpx
from unittest.mock import AsyncMock, patch

from reasoning_api.dependencies import (
    ServiceContainer,
    service_container,
    get_http_client,
    create_production_http_client,
)
from reasoning_api.config import settings


class TestHTTPClientConfiguration:
    """Test HTTP client production configuration."""

    def test__create_production_http_client__has_proper_timeouts(self) -> None:
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

    def test__create_production_http_client__uses_settings_values(self) -> None:
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
    async def test__production_http_client__is_properly_configured(self) -> None:
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
            # This can occur when pytest-asyncio closes the event loop
            # before the fixture teardown runs
            if "Event loop is closed" not in str(e):
                raise

    @pytest.mark.asyncio
    async def test__initialize__sets_up_services_correctly(self, clean_container: ServiceContainer, tmp_path: any) -> None:  # noqa: E501
        """Test that initialize sets up all services correctly."""
        await clean_container.initialize()

        # HTTP client should be created with production configuration
        assert clean_container.http_client is not None
        assert clean_container.http_client.timeout.connect == settings.http_connect_timeout
        assert clean_container.http_client.timeout.read == settings.http_read_timeout
        assert clean_container.http_client.timeout.write == settings.http_write_timeout

    @pytest.mark.asyncio
    async def test__cleanup__closes_services_properly(self, clean_container: ServiceContainer) -> None:
        """Test that cleanup properly closes all services."""
        await clean_container.initialize()

        # Mock the close methods to verify they're called
        http_close_mock = AsyncMock()
        clean_container.http_client.aclose = http_close_mock

        await clean_container.cleanup()

        # Verify cleanup was called
        http_close_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test__cleanup__handles_none_services_gracefully(self, clean_container: ServiceContainer) -> None:  # noqa: E501
        """Test that cleanup handles None services without errors."""
        # Don't initialize, so services remain None
        assert clean_container.http_client is None

        # Should not raise any errors
        await clean_container.cleanup()


class TestDependencyInjection:
    """Test dependency injection functions."""

    @pytest.mark.asyncio
    async def test__get_http_client__returns_client_when_initialized(self) -> None:
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
    async def test__get_http_client__raises_error_when_not_initialized(self) -> None:
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


class TestResourceLeakPrevention:
    """Test that the fix prevents resource leaks."""

    @pytest.mark.asyncio
    async def test__no_unmanaged_httpx_clients_created(self) -> None:
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
    async def test__service_container_lifecycle_prevents_leaks(self) -> None:
        """Test that proper service container lifecycle prevents resource leaks."""
        container = ServiceContainer()

        # Track httpx.AsyncClient creation and closure
        with patch('httpx.AsyncClient') as mock_async_client_class, \
             patch('reasoning_api.mcp.Client') as mock_mcp_client:

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


