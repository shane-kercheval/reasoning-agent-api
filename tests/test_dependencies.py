"""
Tests for dependency injection system.

Tests the FastAPI dependency injection system, service container lifecycle,
and error handling when dependencies are not properly initialized.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

from api.dependencies import (
    ServiceContainer,
    service_container,
    get_http_client,
    get_mcp_client,
    get_reasoning_agent,
)
from api.config import settings


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

        # HTTP client should be created
        assert clean_container.http_client is not None
        assert clean_container.http_client.timeout.read == 60.0

        # MCP client setup depends on API key availability
        if settings.openai_api_key:
            assert clean_container.mcp_client is not None
        else:
            assert clean_container.mcp_client is None

    @pytest.mark.asyncio
    async def test__cleanup__closes_services_properly(self, clean_container: ServiceContainer):
        """Test that cleanup properly closes all services."""
        await clean_container.initialize()

        # Mock the close methods to verify they're called
        http_close_mock = AsyncMock()
        mcp_close_mock = AsyncMock()

        clean_container.http_client.aclose = http_close_mock
        if clean_container.mcp_client:
            clean_container.mcp_client.close = mcp_close_mock

        await clean_container.cleanup()

        # Verify cleanup was called
        http_close_mock.assert_called_once()
        if clean_container.mcp_client:
            mcp_close_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test__cleanup__handles_none_services_gracefully(self, clean_container: ServiceContainer):  # noqa: E501
        """Test that cleanup handles None services without errors."""
        # Don't initialize, so services remain None
        assert clean_container.http_client is None
        assert clean_container.mcp_client is None

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
    async def test__get_mcp_client__returns_client_or_none(self):
        """Test that get_mcp_client returns the MCP client or None."""
        # Save original state
        original_client = service_container.mcp_client

        try:
            # Test with mock client
            mock_client = AsyncMock()
            service_container.mcp_client = mock_client

            result = await get_mcp_client()
            assert result is mock_client

            # Test with None
            service_container.mcp_client = None

            result = await get_mcp_client()
            assert result is None
        finally:
            # Restore original state
            service_container.mcp_client = original_client

    @pytest.mark.asyncio
    async def test__get_reasoning_agent__creates_agent_with_dependencies(self):
        """Test that get_reasoning_agent creates agent with proper dependencies."""
        # Save original state
        original_http = service_container.http_client
        original_mcp = service_container.mcp_client

        try:
            # Mock dependencies
            mock_http_client = AsyncMock()
            mock_mcp_client = AsyncMock()
            service_container.http_client = mock_http_client
            service_container.mcp_client = mock_mcp_client

            # Mock settings to avoid real values
            with patch('api.dependencies.settings') as mock_settings:
                mock_settings.reasoning_agent_base_url = "https://test.api.com/v1"
                mock_settings.openai_api_key = "test-key"

                # Get reasoning agent through dependency injection
                http_client = await get_http_client()
                mcp_client = await get_mcp_client()
                agent = await get_reasoning_agent(http_client, mcp_client)

                # Verify agent was created with correct dependencies
                assert agent is not None
                assert agent.http_client is mock_http_client
                assert agent.mcp_client is mock_mcp_client
        finally:
            # Restore original state
            service_container.http_client = original_http
            service_container.mcp_client = original_mcp

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
        with patch('httpx.AsyncClient') as mock_async_client_class:
            mock_client = AsyncMock()
            mock_async_client_class.return_value = mock_client

            # Initialize - should create one client
            await container.initialize()
            mock_async_client_class.assert_called_once()

            # Cleanup - should close the client
            await container.cleanup()
            mock_client.aclose.assert_called_once()

        # Verify the pattern: one creation, one cleanup
        assert mock_async_client_class.call_count == 1
        assert mock_client.aclose.call_count == 1
