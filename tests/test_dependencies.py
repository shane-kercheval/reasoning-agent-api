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
    get_mcp_client,
    get_tools,
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
    async def test__initialize__sets_up_services_correctly(self, clean_container: ServiceContainer, tmp_path: any):  # noqa: E501
        """Test that initialize sets up all services correctly."""
        # Create a test MCP config file with at least one server
        test_config = tmp_path / "test_mcp_config.json"
        test_config.write_text(
            '{"mcpServers": {"test_server": {"command": "test", "args": [], "env": {}}}}',
        )

        # Patch the settings to use our test config
        with patch.object(settings, 'mcp_config_path', str(test_config)):

            await clean_container.initialize()

            # HTTP client should be created with production configuration
            assert clean_container.http_client is not None
            assert clean_container.http_client.timeout.connect == settings.http_connect_timeout
            assert clean_container.http_client.timeout.read == settings.http_read_timeout
            assert clean_container.http_client.timeout.write == settings.http_write_timeout

            # MCP client should be initialized
            assert clean_container.mcp_client is not None

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
    async def test__get_mcp_client__returns_client_when_initialized(self):
        """Test that get_mcp_client returns the MCP client when initialized."""
        # Save original state
        original_client = service_container.mcp_client

        try:
            # Test with mock client
            mock_client = AsyncMock()
            service_container.mcp_client = mock_client

            result = await get_mcp_client()
            assert result is mock_client
        finally:
            # Restore original state
            service_container.mcp_client = original_client

    @pytest.mark.asyncio
    async def test__get_reasoning_agent__creates_agent_with_dependencies(self):
        """Test that get_reasoning_agent creates agent with proper dependencies."""
        # Save original state
        original_http = service_container.http_client
        original_mcp_client = service_container.mcp_client
        original_prompt_initialized = service_container.prompt_manager_initialized

        try:
            # Mock dependencies - use real HTTP client for OpenAI compatibility
            real_http_client = httpx.AsyncClient()
            mock_mcp_client = AsyncMock()

            service_container.http_client = real_http_client
            service_container.mcp_client = mock_mcp_client
            service_container.prompt_manager_initialized = True

            # Mock settings to avoid real values
            with patch('api.dependencies.settings') as mock_settings:
                mock_settings.reasoning_agent_base_url = "https://test.api.com/v1"
                mock_settings.openai_api_key = "test-key"

                # Get reasoning agent through dependency injection
                http_client = await get_http_client()
                tools = await get_tools()
                prompt_manager = await get_prompt_manager()
                agent = await get_reasoning_agent(
                    http_client,
                    tools,
                    prompt_manager,
                )

                # Verify agent was created with correct dependencies
                assert agent is not None
                assert agent.http_client is real_http_client
                assert agent.tools is not None
                assert isinstance(agent.tools, dict)
                assert agent.prompt_manager is not None

                # Clean up the HTTP client
                await real_http_client.aclose()
        finally:
            # Restore original state
            service_container.http_client = original_http
            service_container.mcp_client = original_mcp_client
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
    async def test__service_container_uses_settings_mcp_config_path(self, tmp_path):  # noqa: ANN001
        """Test that ServiceContainer uses settings.mcp_config_path."""
        # Create a test config file in JSON format
        test_config = tmp_path / "test_mcp_config.json"
        test_config.write_text("""
{
  "mcpServers": {
    "test_server": {
      "command": "test",
      "args": [],
      "env": {}
    }
  }
}
""")

        # Create a new ServiceContainer to test
        container = ServiceContainer()

        try:
            # Patch the settings to use our test config path
            with patch.object(settings, 'mcp_config_path', str(test_config)):

                # Initialize should use the custom config path
                await container.initialize()

                # Verify MCP client was created (even if connection fails)
                assert container.mcp_client is not None

        finally:
            await container.cleanup()

    @pytest.mark.asyncio
    async def test__service_container_handles_nonexistent_config_file(self):
        """Test ServiceContainer gracefully handles nonexistent config file."""
        # Create a new ServiceContainer
        container = ServiceContainer()

        try:
            # Patch the settings to use nonexistent path
            with patch.object(settings, 'mcp_config_path', "nonexistent/path/config.json"):

                # Should not raise exception, should set mcp_client to None
                await container.initialize()

                # Should have no MCP client due to missing config
                assert container.mcp_client is None

        finally:
            await container.cleanup()

    @pytest.mark.asyncio
    async def test__service_container_supports_json_config(self, tmp_path):  # noqa: ANN001
        """Test ServiceContainer supports JSON config files."""
        # Create a test JSON config file in correct FastMCP format
        test_config = tmp_path / "test_mcp_config.json"
        test_config.write_text("""
{
  "mcpServers": {
    "test_json_server": {
      "command": "test",
      "args": [],
      "env": {}
    }
  }
}
""")

        # Create a new ServiceContainer
        container = ServiceContainer()

        try:
            # Patch the settings to use our JSON config
            with patch.object(settings, 'mcp_config_path', str(test_config)):

                # Initialize should use the JSON config
                await container.initialize()

                # Verify MCP client was created
                assert container.mcp_client is not None

        finally:
            await container.cleanup()


class TestReasoningAgentInstanceIsolation:
    """
    Test that ReasoningAgent instances are properly isolated per request.

    This is critical for preventing state contamination between requests.
    Each request must get a fresh ReasoningAgent instance with clean
    reasoning_context to avoid mixing reasoning steps and tool results
    between different user requests.
    """

    @pytest.mark.asyncio
    async def test__get_reasoning_agent__creates_new_instance_per_call(self):
        """
        Test that get_reasoning_agent creates a new instance for each call.

        This test verifies that FastAPI dependency injection creates a new
        ReasoningAgent instance for each request, preventing state contamination
        between requests. Each agent should have its own fresh reasoning_context.
        """
        # Save original state
        original_http = service_container.http_client
        original_mcp_client = service_container.mcp_client
        original_prompt_initialized = service_container.prompt_manager_initialized

        try:
            # Mock dependencies - use real HTTP client for OpenAI compatibility
            real_http_client = httpx.AsyncClient()
            mock_mcp_client = AsyncMock()

            service_container.http_client = real_http_client
            service_container.mcp_client = mock_mcp_client
            service_container.prompt_manager_initialized = True

            # Mock settings to avoid real values
            with patch('api.dependencies.settings') as mock_settings:
                mock_settings.reasoning_agent_base_url = "https://test.api.com/v1"
                mock_settings.openai_api_key = "test-key"

                # Get dependencies (these are shared)
                http_client = await get_http_client()
                tools = await get_tools()
                prompt_manager = await get_prompt_manager()

                # Create multiple reasoning agents (simulating multiple requests)
                agent1 = await get_reasoning_agent(http_client, tools, prompt_manager)
                agent2 = await get_reasoning_agent(http_client, tools, prompt_manager)
                agent3 = await get_reasoning_agent(http_client, tools, prompt_manager)

                # Verify that different instances are created
                assert agent1 is not agent2, "First and second agents should be different instances"  # noqa: E501
                assert agent2 is not agent3, "Second and third agents should be different instances"  # noqa: E501
                assert agent1 is not agent3, "First and third agents should be different instances"

                # Verify each has different reasoning context objects (not shared references)
                assert agent1.reasoning_context is not agent2.reasoning_context, "Agents should have different reasoning_context objects"  # noqa: E501
                assert agent1.reasoning_context == {'steps': [], 'tool_results': [], 'final_thoughts': '', 'user_request': None}, "Agent1 should have clean reasoning context"  # noqa: E501
                assert agent2.reasoning_context == {'steps': [], 'tool_results': [], 'final_thoughts': '', 'user_request': None}, "Agent2 should have clean reasoning context"  # noqa: E501
                assert agent3.reasoning_context == {'steps': [], 'tool_results': [], 'final_thoughts': '', 'user_request': None}, "Agent3 should have clean reasoning context"  # noqa: E501

                # Verify shared resources are the same (good for performance)
                assert agent1.http_client is agent2.http_client, "HTTP client should be shared between agents"  # noqa: E501
                assert agent2.http_client is agent3.http_client, "HTTP client should be shared between agents"  # noqa: E501
                # Note: tools dict is recreated for each agent but contains same tool objects
                assert agent1.tools == agent2.tools, "Tools content should be the same between agents"  # noqa: E501
                assert agent2.tools == agent3.tools, "Tools content should be the same between agents"  # noqa: E501

                # Clean up the HTTP client
                await real_http_client.aclose()

        finally:
            # Restore original state
            service_container.http_client = original_http
            service_container.mcp_client = original_mcp_client
            service_container.prompt_manager_initialized = original_prompt_initialized

    @pytest.mark.asyncio
    async def test__reasoning_context_isolation__state_contamination_prevention(self):
        """
        Test that modifying one agent's context doesn't affect another agent.

        This test simulates the scenario where one request modifies its reasoning
        context and verifies that other concurrent requests are not affected.
        This is critical for preventing data leakage between user requests.
        """
        # Save original state
        original_http = service_container.http_client
        original_mcp_client = service_container.mcp_client
        original_prompt_initialized = service_container.prompt_manager_initialized

        try:
            # Mock dependencies
            real_http_client = httpx.AsyncClient()
            mock_mcp_client = AsyncMock()

            service_container.http_client = real_http_client
            service_container.mcp_client = mock_mcp_client
            service_container.prompt_manager_initialized = True

            with patch('api.dependencies.settings') as mock_settings:
                mock_settings.reasoning_agent_base_url = "https://test.api.com/v1"
                mock_settings.openai_api_key = "test-key"

                # Get dependencies
                http_client = await get_http_client()
                tools = await get_tools()
                prompt_manager = await get_prompt_manager()

                # Create two agents (simulating two concurrent requests)
                agent1 = await get_reasoning_agent(http_client, tools, prompt_manager)
                agent2 = await get_reasoning_agent(http_client, tools, prompt_manager)

                # Simulate request 1 processing and modifying its context
                agent1.reasoning_context['steps'].append({"thought": "User 1's reasoning"})
                agent1.reasoning_context['tool_results'].append({"tool": "weather", "result": "sunny"})  # noqa: E501
                agent1.reasoning_context['final_thoughts'] = "User 1's conclusion"
                agent1.reasoning_context['user_request'] = "What's the weather?"

                # Verify that agent2's context remains clean (not contaminated)
                assert agent2.reasoning_context['steps'] == [], "Agent2 steps should remain empty"
                assert agent2.reasoning_context['tool_results'] == [], "Agent2 tool results should remain empty"  # noqa: E501
                assert agent2.reasoning_context['final_thoughts'] == '', "Agent2 final thoughts should remain empty"  # noqa: E501
                assert agent2.reasoning_context['user_request'] is None, "Agent2 user request should remain None"  # noqa: E501

                # Simulate request 2 processing independently
                agent2.reasoning_context['steps'].append({"thought": "User 2's different reasoning"})  # noqa: E501
                agent2.reasoning_context['tool_results'].append({"tool": "calculator", "result": "42"})  # noqa: E501

                # Verify agent1's context is unchanged by agent2's modifications
                assert len(agent1.reasoning_context['steps']) == 1, "Agent1 should still have exactly 1 step"  # noqa: E501
                assert agent1.reasoning_context['steps'][0]['thought'] == "User 1's reasoning", "Agent1's step should be unchanged"  # noqa: E501
                assert agent1.reasoning_context['tool_results'][0]['tool'] == "weather", "Agent1's tool result should be unchanged"  # noqa: E501
                assert agent1.reasoning_context['final_thoughts'] == "User 1's conclusion", "Agent1's final thoughts should be unchanged"  # noqa: E501

                # Verify agent2 has its own independent state
                assert len(agent2.reasoning_context['steps']) == 1, "Agent2 should have exactly 1 step"  # noqa: E501
                assert agent2.reasoning_context['steps'][0]['thought'] == "User 2's different reasoning", "Agent2 should have its own reasoning"  # noqa: E501
                assert agent2.reasoning_context['tool_results'][0]['tool'] == "calculator", "Agent2 should have its own tool results"  # noqa: E501

                # Clean up
                await real_http_client.aclose()

        finally:
            # Restore original state
            service_container.http_client = original_http
            service_container.mcp_client = original_mcp_client
            service_container.prompt_manager_initialized = original_prompt_initialized
