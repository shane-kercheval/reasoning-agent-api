"""
Dependency injection for FastAPI application.

This module provides dependency injection using FastAPI's built-in DI system.
Dependencies are created once per request and properly cleaned up.
"""

import logging
from typing import Annotated

import httpx
from fastapi import Depends

from pathlib import Path

from .config import settings
from .tools import Tool
from .mcp import create_mcp_client, to_tools
from .prompt_manager import prompt_manager, PromptManager
from .database import ConversationDB

# Note: current_span ContextVar removed - no longer needed with endpoint-based tracing

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
    """
    Container for application services with proper lifecycle management.

    Manages shared resources (HTTP client for MCP, MCP client, prompt manager) with
    proper async initialization and cleanup.

    Usage in production (via FastAPI lifespan):
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await service_container.initialize()

    Yield:
            await service_container.cleanup()

    Usage in tests (as async context manager):
        async with ServiceContainer() as container:
            # Use container for testing
            pass

    IMPORTANT - Test Limitation:
    ============================
    When using ServiceContainer as an async context manager with FastAPI's
    synchronous TestClient, you may see "Event loop is closed" errors in
    cleanup (__aexit__). This is EXPECTED and HARMLESS.

    WHY: TestClient creates its own event loop and closes it when the 'with'
    block exits. If ServiceContainer is used as 'async with', its __aexit__
    cleanup runs AFTER TestClient has already closed the loop.

    SOLUTION: The __aexit__ method catches and ignores "Event loop is closed"
    errors. This is safe because:
    1. It only affects test code (production uses lifespan, not TestClient)
    2. Resources are still cleaned up properly by TestClient's loop shutdown
    3. The error indicates cleanup was attempted, just in a closed loop

    This is a known limitation of mixing sync TestClient with async context
    managers. Alternative would be to use async test client, but that requires
    more complex test setup.
    """

    def __init__(self):
        self.http_client: httpx.AsyncClient | None = None
        self.mcp_client = None
        self.prompt_manager_initialized: bool = False
        self.conversation_db: ConversationDB | None = None

    async def initialize(self) -> None:
        """
        Initialize services during app startup.

        Creates production-ready HTTP client for MCP with proper timeouts and
        connection pooling, and initializes MCP, prompt, and database services.
        """
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

        # Initialize conversation database
        try:
            self.conversation_db = ConversationDB(settings.reasoning_database_url)
            await self.conversation_db.connect()
            logger.info("Conversation database initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize conversation database (continuing without conversation storage): {e}")  # noqa: E501
            self.conversation_db = None

    async def cleanup(self) -> None:
        """Cleanup services during app shutdown."""
        # Properly close connections when app shuts down
        # Note: FastMCP Client doesn't have explicit close method
        # It's designed to be used with context managers and cleans up automatically
        if self.http_client:
            await self.http_client.aclose()
        if self.prompt_manager_initialized:
            await prompt_manager.cleanup()
        if self.conversation_db:
            await self.conversation_db.disconnect()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        """Async context manager exit."""
        try:
            await self.cleanup()
        except RuntimeError as e:
            # Ignore "Event loop is closed" errors during cleanup
            # This can occur in tests when TestClient closes its event loop
            # before the async context manager exits
            if "Event loop is closed" not in str(e):
                raise
        return False  # Don't suppress exceptions


# Global service container - initialized during app lifespan
service_container = ServiceContainer()


async def get_http_client() -> httpx.AsyncClient:
    """Get HTTP client dependency (used for MCP)."""
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


async def get_conversation_db() -> ConversationDB | None:
    """
    Get conversation database dependency.

    Returns None if database is not available (e.g., in tests without database).
    The endpoint will handle None gracefully by rejecting stateful requests.
    """
    return service_container.conversation_db


# Type aliases for cleaner endpoint signatures
MCPClientDependency = Annotated[object, Depends(get_mcp_client)]
ToolsDependency = Annotated[list[Tool], Depends(get_tools)]
PromptManagerDependency = Annotated[object, Depends(get_prompt_manager)]
ConversationDBDependency = Annotated[ConversationDB | None, Depends(get_conversation_db)]
