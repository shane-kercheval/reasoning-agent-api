"""Tools router - REST endpoints for tool execution."""

from typing import Any

from fastapi import APIRouter, HTTPException

from tools_api.models import ToolDefinition, ToolResult
from tools_api.services.registry import ToolRegistry

router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("/")
async def list_tools() -> list[ToolDefinition]:
    """List all available tools."""
    return [
        ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            tags=tool.tags,
        )
        for tool in ToolRegistry.list()
    ]


@router.post("/{tool_name}")
async def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
) -> ToolResult:
    """
    Execute a tool with the provided arguments.

    Args:
        tool_name: Name of the tool to execute
        arguments: Tool-specific arguments

    Returns:
        ToolResult with success status and result data

    Raises:
        HTTPException: 404 if tool not found
    """
    tool = ToolRegistry.get(tool_name)
    if tool is None:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    return await tool(**arguments)
