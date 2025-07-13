"""
Dependency injection for FastAPI application.

This module provides dependency injection using FastAPI's built-in DI system.
Dependencies are created once per request and properly cleaned up.
"""

import logging
from typing import Annotated

import httpx
from fastapi import Depends

from .config import settings
from .reasoning_agent import ReasoningAgent
from .mcp import MCPManager, MCPServersConfig
from .mcp import load_yaml_config, load_mcp_json_config
from pathlib import Path
from .prompt_manager import prompt_manager, PromptManager

logger = logging.getLogger(__name__)


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
        self.mcp_manager: MCPManager | None = None
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
        # Handle initialization failures gracefully for testing scenarios
        try:
            # Load MCP configuration (prefer JSON over YAML)
            json_config_path = Path("config/mcp_servers.json")
            yaml_config_path = Path("config/mcp_servers.yaml")

            if json_config_path.exists():
                mcp_config = load_mcp_json_config(json_config_path)
            elif yaml_config_path.exists():
                mcp_config = load_yaml_config(yaml_config_path)
            else:
                logger.warning("No MCP configuration found, starting without MCP tools")
                mcp_config = MCPServersConfig(servers=[])

            self.mcp_manager = MCPManager(mcp_config.servers)
            await self.mcp_manager.initialize()
            logger.info("MCP manager initialized successfully")
        except Exception as e:
            logger.warning(f"MCP manager initialization failed (continuing without MCP): {e}")
            # Create empty manager for graceful degradation
            self.mcp_manager = MCPManager([])

    async def cleanup(self) -> None:
        """Cleanup services during app shutdown."""
        # Properly close connections when app shuts down
        if self.mcp_manager:
            await self.mcp_manager.health_check()  # MCPManager doesn't need explicit cleanup
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


async def get_mcp_manager() -> MCPManager:
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
    mcp_manager: Annotated[MCPManager, Depends(get_mcp_manager)],
    prompt_manager: Annotated[PromptManager, Depends(get_prompt_manager)],
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
    )


# Type aliases for cleaner endpoint signatures
MCPManagerDependency = Annotated[MCPManager, Depends(get_mcp_manager)]
PromptManagerDependency = Annotated[object, Depends(get_prompt_manager)]
ReasoningAgentDependency = Annotated[ReasoningAgent, Depends(get_reasoning_agent)]
