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


class ServiceContainer:
    """Container for application services with proper lifecycle management."""

    def __init__(self):
        self.http_client: httpx.AsyncClient | None = None
        self.mcp_client: MCPClient | None = None

    async def initialize(self) -> None:
        """Initialize services during app startup."""
        self.http_client = httpx.AsyncClient(timeout=60.0)
        self.mcp_client = MCPClient(DEFAULT_MCP_CONFIG) if settings.openai_api_key else None

    async def cleanup(self) -> None:
        """Cleanup services during app shutdown."""
        if self.mcp_client:
            await self.mcp_client.close()
        if self.http_client:
            await self.http_client.aclose()


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
    return service_container.mcp_client


async def get_reasoning_agent(
    http_client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    mcp_client: Annotated[MCPClient | None, Depends(get_mcp_client)],
) -> ReasoningAgent:
    """Get reasoning agent dependency with injected dependencies."""
    return ReasoningAgent(
        base_url=settings.reasoning_agent_base_url,
        api_key=settings.openai_api_key,
        http_client=http_client,
        mcp_client=mcp_client,
    )


# Type aliases for cleaner endpoint signatures
HTTPClientDep = Annotated[httpx.AsyncClient, Depends(get_http_client)]
MCPClientDep = Annotated[MCPClient | None, Depends(get_mcp_client)]
ReasoningAgentDep = Annotated[ReasoningAgent, Depends(get_reasoning_agent)]
