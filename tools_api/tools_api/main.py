"""
Tools API - FastAPI application for structured tool and prompt execution.

This service provides REST endpoints for executing tools and rendering prompts
with structured JSON responses, replacing the MCP (Model Context Protocol)
architecture.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tools_api.config import settings

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
    )
    from tools_api.services.tools.github_dev_tools import (
        GetGitHubPullRequestInfoTool,
        GetLocalGitChangesInfoTool,
        GetDirectoryTreeTool,
    )
    from tools_api.services.tools.web_search_tool import BraveSearchTool
    from tools_api.services.prompts.example_prompt import GreetingPrompt
    from tools_api.services.registry import ToolRegistry, PromptRegistry

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
        GetDirectoryTreeTool(),
    ]
    for tool in github_dev_tools:
        try:
            ToolRegistry.register(tool)
            logger.info(f"Registered GitHub/dev tool: {tool.name}")
        except ValueError:
            pass  # Already registered

    # Register web search tools
    try:
        ToolRegistry.register(BraveSearchTool())
        logger.info("Registered web search tool: brave_search")
    except ValueError:
        pass  # Already registered

    # Register prompts
    try:
        PromptRegistry.register(GreetingPrompt())
        logger.info("Registered prompt: example_prompt")
    except ValueError:
        pass  # Already registered

    yield

    # Shutdown
    logger.info("Tools API shutting down...")


app = FastAPI(
    title="Tools API",
    description="Structured tool and prompt execution service",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
