"""
Dependency injection for FastAPI application.

This module provides dependency injection using FastAPI's built-in DI system.
Dependencies are created once per request and properly cleaned up.
"""

from typing import Annotated

import httpx
from fastapi import Depends

from .config import settings
from .reasoning_agent import ReasoningAgent
from .mcp_client import MCPClient, DEFAULT_MCP_CONFIG
from .mcp_manager import MCPServerManager
from .config_loader import load_mcp_config
from .prompt_manager import prompt_manager, PromptManager


def create_production_http_client() -> httpx.AsyncClient:
    """
    Create HTTP client with production-ready configuration.

    Configures timeouts and connection pooling for optimal performance
    and reliability when communicating with external APIs like OpenAI.

    Returns:
        Configured httpx.AsyncClient instance.

    Example:
        >>> client = create_production_http_client()
        >>> # Client has 5s connect, 30s read timeouts
        >>> # and connection pooling for performance
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=settings.http_connect_timeout,
            read=settings.http_read_timeout,
            write=settings.http_write_timeout,
            pool=settings.http_read_timeout,  # Pool timeout same as read timeout
        ),
        limits=httpx.Limits(
            max_connections=settings.http_max_connections,
            max_keepalive_connections=settings.http_max_keepalive_connections,
            keepalive_expiry=settings.http_keepalive_expiry,
        ),
    )


class ServiceContainer:
    """Container for application services with proper lifecycle management."""

    def __init__(self):
        self.http_client: httpx.AsyncClient | None = None
        self.mcp_client: MCPClient | None = None
        self.mcp_manager: MCPServerManager | None = None
        self.prompt_manager_initialized: bool = False

    async def initialize(self) -> None:
        """
        Initialize services during app startup.

        Creates production-ready HTTP client with proper timeouts and
        connection pooling, and initializes MCP and prompt services.
        """
         # Create ONE http client for the entire app lifetime
        self.http_client = create_production_http_client()

        # Initialize prompt manager
        await prompt_manager.initialize()
        self.prompt_manager_initialized = True

        # Initialize MCP manager with loaded configuration
        mcp_config = load_mcp_config()
        self.mcp_manager = MCPServerManager(mcp_config.servers)
        await self.mcp_manager.initialize()

        # Keep existing MCP client for backward compatibility
        # TODO: Remove this once all functionality is migrated to MCPServerManager
        self.mcp_client = MCPClient(DEFAULT_MCP_CONFIG) if settings.openai_api_key else None

    async def cleanup(self) -> None:
        """Cleanup services during app shutdown."""
        # Properly close connections when app shuts down
        if self.mcp_manager:
            await self.mcp_manager.cleanup()
        if self.mcp_client:
            await self.mcp_client.close()
        if self.http_client:
            await self.http_client.aclose()
        if self.prompt_manager_initialized:
            await prompt_manager.cleanup()


# Global service container - initialized during app lifespan
service_container = ServiceContainer()


async def get_http_client() -> httpx.AsyncClient:
    """Get HTTP client dependency."""
    if service_container.http_client is None:
        raise RuntimeError(
            "Service container not initialized. "
            "HTTP client should be available after app startup. "
            "If testing, ensure service_container.initialize() is called.",
        )
    return service_container.http_client


async def get_mcp_client() -> MCPClient | None:
    """Get MCP client dependency."""
    # TODO: are we sure we want to return the same MCP client instance per request?
    return service_container.mcp_client


async def get_mcp_manager() -> MCPServerManager:
    """Get MCP server manager dependency."""
    if service_container.mcp_manager is None:
        raise RuntimeError(
            "Service container not initialized. "
            "MCP manager should be available after app startup. "
            "If testing, ensure service_container.initialize() is called.",
        )
    return service_container.mcp_manager


async def get_prompt_manager() -> PromptManager:
    """Get prompt manager dependency."""
    if not service_container.prompt_manager_initialized:
        raise RuntimeError(
            "Service container not initialized. "
            "Prompt manager should be available after app startup. "
            "If testing, ensure service_container.initialize() is called.",
        )
    return prompt_manager


async def get_reasoning_agent(
    http_client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    mcp_manager: Annotated[MCPServerManager, Depends(get_mcp_manager)],
    prompt_manager: Annotated[PromptManager, Depends(get_prompt_manager)],
    mcp_client: Annotated[MCPClient | None, Depends(get_mcp_client)] = None,
) -> ReasoningAgent:
    """Get reasoning agent dependency with injected dependencies."""
    # this returns a new ReasoningAgent instance for each request, while reusing the same HTTP and
    # MCP clients
    return ReasoningAgent(
        base_url=settings.reasoning_agent_base_url,
        api_key=settings.openai_api_key,
        http_client=http_client,
        mcp_manager=mcp_manager,
        prompt_manager=prompt_manager,
        mcp_client=mcp_client,
    )


# Type aliases for cleaner endpoint signatures
MCPClientDependency = Annotated[MCPClient | None, Depends(get_mcp_client)]
MCPManagerDependency = Annotated[MCPServerManager, Depends(get_mcp_manager)]
PromptManagerDependency = Annotated[object, Depends(get_prompt_manager)]
ReasoningAgentDependency = Annotated[ReasoningAgent, Depends(get_reasoning_agent)]
