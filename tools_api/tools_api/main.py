"""
Tools API - FastAPI application for structured tool and prompt execution.

This service provides REST endpoints for executing tools and rendering prompts
with structured JSON responses. Also exposes an MCP (Model Context Protocol)
endpoint at /mcp for MCP clients.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from tools_api.config import settings
from tools_api.mcp_server import server as mcp_server
from tools_api.services.registry import PromptRegistry, ToolRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown tasks like initializing services,
    loading tools/prompts, etc.
    """
    # Import here to avoid circular imports
    from tools_api.services.tools.filesystem import (
        ReadTextFileTool,
        WriteFileTool,
        EditFileTool,
        CreateDirectoryTool,
        ListDirectoryTool,
        ListDirectoryWithSizesTool,
        SearchFilesTool,
        GetFileInfoTool,
        ListAllowedDirectoriesTool,
        MoveFileTool,
        DeleteFileTool,
        DeleteDirectoryTool,
        GetDirectoryTreeTool,
    )
    from tools_api.services.tools.github_dev_tools import (
        GetGitHubPullRequestInfoTool,
        GetLocalGitChangesInfoTool,
    )
    from tools_api.services.tools.web_search import WebSearchTool
    from tools_api.services.tools.web_scraper import WebScraperTool
    from tools_api.services.prompts import register_prompts_from_directory

    # Startup
    logger.info("Tools API starting up...")
    logger.info(f"Read-write base: {settings.read_write_base}")
    logger.info(f"Read-only base: {settings.read_only_base}")

    # Register filesystem tools
    filesystem_tools = [
        ReadTextFileTool(),
        WriteFileTool(),
        EditFileTool(),
        CreateDirectoryTool(),
        ListDirectoryTool(),
        ListDirectoryWithSizesTool(),
        SearchFilesTool(),
        GetFileInfoTool(),
        ListAllowedDirectoriesTool(),
        MoveFileTool(),
        DeleteFileTool(),
        DeleteDirectoryTool(),
        GetDirectoryTreeTool(),
    ]
    for tool in filesystem_tools:
        try:
            ToolRegistry.register(tool)
            logger.info(f"Registered filesystem tool: {tool.name}")
        except ValueError:
            pass  # Already registered

    # Register GitHub and dev tools
    github_dev_tools = [
        GetGitHubPullRequestInfoTool(),
        GetLocalGitChangesInfoTool(),
    ]
    for tool in github_dev_tools:
        try:
            ToolRegistry.register(tool)
            logger.info(f"Registered GitHub/dev tool: {tool.name}")
        except ValueError:
            pass  # Already registered

    # Register web tools
    web_tools = [
        WebSearchTool(),
        WebScraperTool(),
    ]
    for tool in web_tools:
        try:
            ToolRegistry.register(tool)
            logger.info(f"Registered web tool: {tool.name}")
        except ValueError:
            pass  # Already registered

    # Register file-based prompts from directory
    if settings.prompts_directory:
        try:
            count = register_prompts_from_directory(settings.prompts_directory)
            logger.info(f"Registered {count} prompts from {settings.prompts_directory}")
        except FileNotFoundError as e:
            logger.error(f"Prompts directory not found: {e}")
            raise  # Fail startup if configured directory doesn't exist
        except ValueError as e:
            logger.error(f"Duplicate prompt name: {e}")
            raise  # Fail startup on duplicate prompt names
    else:
        logger.info("No prompts directory configured, skipping file-based prompts")

    # Create and start MCP session manager (if enabled)
    # MCP can be disabled via MCP_ENABLED=false for testing, as the anyio task group
    # used by StreamableHTTPSessionManager conflicts with pytest-asyncio's event loop
    if settings.mcp_enabled:
        mcp_session_manager = StreamableHTTPSessionManager(
            app=mcp_server,
            json_response=True,  # Use JSON responses for simpler debugging
            stateless=True,  # Stateless mode - each request is independent
        )

        # Store session manager in app state for access by routes
        app.state.mcp_session_manager = mcp_session_manager

        logger.info("MCP server initialized at /mcp")
        logger.info(f"MCP tools count: {len(ToolRegistry.list())}")
        logger.info(f"MCP prompts count: {len(PromptRegistry.list())}")

        async with mcp_session_manager.run():
            yield
    else:
        logger.info("MCP server disabled (MCP_ENABLED=false)")
        yield

    # Shutdown
    logger.info("Tools API shutting down...")


app = FastAPI(
    title="Tools API",
    description="Structured tool and prompt execution service",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware - expose Mcp-Session-Id for MCP browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Mcp-Session-Id"],
)


# Include routers
from tools_api.routers import tools, prompts

app.include_router(tools.router)
app.include_router(prompts.router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "tools-api",
        "version": "1.0.0",
    }


@app.get("/mcp/health")
async def mcp_health_check() -> dict[str, str | int]:
    """MCP endpoint health check."""
    return {
        "status": "healthy",
        "transport": "streamable-http",
        "tools_count": len(ToolRegistry.list()),
        "prompts_count": len(PromptRegistry.list()),
    }


# MCP endpoint - mount as ASGI sub-application
# NOTE: This must be after /mcp/health route to avoid catching it
from starlette.types import Receive, Scope, Send


async def mcp_asgi_handler(scope: Scope, receive: Receive, send: Send) -> None:
    """
    ASGI handler for MCP endpoint.

    Routes MCP protocol messages to the StreamableHTTPSessionManager.
    The session manager handles the JSON-RPC protocol over HTTP.
    """
    # Get app from scope to access state
    app = scope.get("app")
    if app is None or not hasattr(app.state, "mcp_session_manager"):
        # Return 503 if MCP not initialized yet
        from starlette.responses import JSONResponse
        response = JSONResponse(
            {"error": "MCP server not initialized"},
            status_code=503,
        )
        await response(scope, receive, send)
        return

    session_manager: StreamableHTTPSessionManager = app.state.mcp_session_manager
    await session_manager.handle_request(scope, receive, send)


# Mount MCP at /mcp path
app.mount("/mcp", mcp_asgi_handler)
