"""
Dependency injection for FastAPI application.

This module provides dependency injection using FastAPI's built-in DI system.
Dependencies are created once per request and properly cleaned up.
"""

import logging
from contextvars import ContextVar
from typing import Annotated

import httpx
from fastapi import Depends
from opentelemetry import trace

from .config import settings
from .reasoning_agent import ReasoningAgent
from .tools import Tool
from .mcp import create_mcp_client, to_tools
from pathlib import Path
from .prompt_manager import prompt_manager, PromptManager

# Global context variable for storing current span
current_span: ContextVar[trace.Span | None] = ContextVar('current_span', default=None)

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
        self.mcp_client = None
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

        # Initialize MCP client with FastMCP
        try:
            config_path = Path(settings.mcp_config_path)
            logger.info(f"Loading MCP configuration from: {config_path}")

            if config_path.exists():
                self.mcp_client = create_mcp_client(config_path)
                logger.info(f"MCP client created from {config_path}")
            else:
                logger.warning(f"MCP configuration file not found: {config_path}")
                logger.warning("Starting without MCP tools")
                self.mcp_client = None
        except Exception as e:
            logger.warning(f"MCP client initialization failed (continuing without MCP): {e}")
            self.mcp_client = None

    async def cleanup(self) -> None:
        """Cleanup services during app shutdown."""
        # Properly close connections when app shuts down
        # Note: FastMCP Client doesn't have explicit close method
        # It's designed to be used with context managers and cleans up automatically
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


async def get_mcp_client() -> object | None:
    """Get MCP client dependency."""
    return service_container.mcp_client  # Can be None if no MCP config


async def get_prompt_manager() -> PromptManager:
    """Get prompt manager dependency."""
    if not service_container.prompt_manager_initialized:
        raise RuntimeError(
            "Service container not initialized. "
            "Prompt manager should be available after app startup. "
            "If testing, ensure service_container.initialize() is called.",
        )
    return prompt_manager


async def get_tools() -> list[Tool]:
    """Get available tools from MCP servers."""
    mcp_client = service_container.mcp_client

    if mcp_client is None:
        logger.info("No MCP client available, returning empty tools list")
        return []

    try:
        async with mcp_client:
            tools = await to_tools(mcp_client)
            logger.info(f"Loaded {len(tools)} tools from MCP servers")
            return tools
    except Exception as e:
        logger.error(f"Failed to load MCP tools: {e}")
        return []


async def get_reasoning_agent(
    http_client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    tools: Annotated[list[Tool], Depends(get_tools)],
    prompt_manager: Annotated[PromptManager, Depends(get_prompt_manager)],
) -> ReasoningAgent:
    """Get reasoning agent dependency with injected dependencies."""
    # this returns a new ReasoningAgent instance for each request, while reusing the same HTTP and
    # MCP clients
    return ReasoningAgent(
        base_url=settings.reasoning_agent_base_url,
        api_key=settings.openai_api_key,
        http_client=http_client,
        tools=tools,
        prompt_manager=prompt_manager,
    )


# Type aliases for cleaner endpoint signatures
MCPClientDependency = Annotated[object, Depends(get_mcp_client)]
ToolsDependency = Annotated[list[Tool], Depends(get_tools)]
PromptManagerDependency = Annotated[object, Depends(get_prompt_manager)]
ReasoningAgentDependency = Annotated[ReasoningAgent, Depends(get_reasoning_agent)]
