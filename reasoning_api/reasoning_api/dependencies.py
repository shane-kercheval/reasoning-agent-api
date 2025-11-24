"""
Dependency injection for FastAPI application.

This module provides dependency injection using FastAPI's built-in DI system.
Dependencies are created once per request and properly cleaned up.
"""

import logging
from typing import Annotated

import httpx
from fastapi import Depends, Request

from pathlib import Path

from .config import settings
from .tools import Tool
from .prompts import Prompt
from .prompt_manager import prompt_manager, PromptManager
from .database import ConversationDB
from .context_manager import ContextManager, ContextUtilization
from .tools_client import ToolsAPIClient

# Note: current_span ContextVar removed - no longer needed with endpoint-based tracing

logger = logging.getLogger(__name__)

# Header name for context utilization strategy
CONTEXT_UTILIZATION_HEADER = "X-Context-Utilization"


def parse_context_utilization_header(
    header_value: str | None,
) -> ContextUtilization:
    """
    Parse X-Context-Utilization header into ContextUtilization enum.

    Args:
        header_value: Value of header (case-insensitive)

    Returns:
        ContextUtilization enum value (defaults to FULL if not provided)

    Raises:
        ValueError: If header value is invalid

    Examples:
        >>> parse_context_utilization_header(None)
        ContextUtilization.FULL
        >>> parse_context_utilization_header("low")
        ContextUtilization.LOW
        >>> parse_context_utilization_header("MEDIUM")
        ContextUtilization.MEDIUM
    """
    if header_value is None:
        return ContextUtilization.FULL

    # Case-insensitive matching (like X-Routing-Mode)
    normalized = header_value.lower().strip()

    try:
        return ContextUtilization(normalized)
    except ValueError:
        valid = [e.value for e in ContextUtilization]
        raise ValueError(
            f"Invalid context utilization: {header_value}. "
            f"Must be one of: {', '.join(valid)}",
        )


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
        self.prompt_manager_initialized: bool = False
        self.conversation_db: ConversationDB | None = None
        self.tools_api_client: ToolsAPIClient | None = None

    async def initialize(self) -> None:
        """
        Initialize services during app startup.

        Creates production-ready HTTP client with proper timeouts and
        connection pooling, and initializes prompt, database, and tools-api services.
        """
        self.http_client = create_production_http_client()

        # Initialize prompt manager
        await prompt_manager.initialize()
        self.prompt_manager_initialized = True

        # Initialize conversation database
        try:
            self.conversation_db = ConversationDB(settings.reasoning_database_url)
            await self.conversation_db.connect()
            logger.info("Conversation database initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize conversation database (continuing without conversation storage): {e}")  # noqa: E501
            self.conversation_db = None

        # Initialize tools-api client
        try:
            self.tools_api_client = ToolsAPIClient(
                base_url=settings.tools_api_url,
                timeout=settings.http_read_timeout,
            )
            # Test connection
            await self.tools_api_client.health_check()
            logger.info(f"Tools API client initialized successfully at {settings.tools_api_url}")
        except Exception as e:
            logger.warning(f"Failed to initialize tools-api client (continuing without tools-api): {e}")
            self.tools_api_client = None

    async def cleanup(self) -> None:
        """Cleanup services during app shutdown."""
        # Properly close connections when app shuts down
        if self.http_client:
            await self.http_client.aclose()
        if self.prompt_manager_initialized:
            await prompt_manager.cleanup()
        if self.conversation_db:
            await self.conversation_db.disconnect()
        if self.tools_api_client:
            await self.tools_api_client.close()

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


async def get_prompt_manager() -> PromptManager:
    """Get prompt manager dependency."""
    if not service_container.prompt_manager_initialized:
        raise RuntimeError(
            "Service container not initialized. "
            "Prompt manager should be available after app startup. "
            "If testing, ensure service_container.initialize() is called.",
        )
    return prompt_manager


async def get_tools_api_client() -> ToolsAPIClient | None:
    """Get tools-api client dependency."""
    return service_container.tools_api_client


async def get_tools_from_tools_api() -> list[Tool]:
    """
    Get available tools from tools-api service.

    Queries tools-api for available tools and converts them to Tool objects
    with tools_api_client set for HTTP-based execution.

    Returns:
        List of Tool objects configured to use tools-api client
    """
    tools_api_client = service_container.tools_api_client

    if tools_api_client is None:
        logger.info("No tools-api client available, returning empty tools list")
        return []

    try:
        tool_definitions = await tools_api_client.list_tools()
        tools = [
            Tool(
                name=tool_def.name,
                description=tool_def.description,
                input_schema=tool_def.parameters,
                tags=tool_def.tags,
                tools_api_client=tools_api_client,
            )
            for tool_def in tool_definitions
        ]
        logger.info(f"Loaded {len(tools)} tools from tools-api")
        return tools
    except Exception as e:
        logger.error(f"Failed to load tools from tools-api: {e}")
        return []


async def get_tools() -> list[Tool]:
    """
    Get available tools from tools-api service.

    Returns:
        List of tools from tools-api
    """
    return await get_tools_from_tools_api()


async def get_conversation_db() -> ConversationDB | None:
    """
    Get conversation database dependency.

    Returns None if database is not available (e.g., in tests without database).
    The endpoint will handle None gracefully by rejecting stateful requests.
    """
    return service_container.conversation_db


async def get_context_manager(
    request: Request,
) -> ContextManager:
    """
    Get ContextManager instance based on request headers.

    Reads X-Context-Utilization header to determine strategy.
    Defaults to FULL if header not provided.

    Args:
        request: FastAPI Request object

    Returns:
        ContextManager configured with the requested utilization strategy

    Raises:
        ValueError: If header value is invalid
    """
    header_value = request.headers.get(CONTEXT_UTILIZATION_HEADER)
    utilization = parse_context_utilization_header(header_value)
    return ContextManager(context_utilization=utilization)


# Type aliases for cleaner endpoint signatures
ToolsAPIClientDependency = Annotated[ToolsAPIClient | None, Depends(get_tools_api_client)]
ToolsDependency = Annotated[list[Tool], Depends(get_tools)]
PromptManagerDependency = Annotated[object, Depends(get_prompt_manager)]
ConversationDBDependency = Annotated[ConversationDB | None, Depends(get_conversation_db)]
ContextManagerDependency = Annotated[ContextManager, Depends(get_context_manager)]
