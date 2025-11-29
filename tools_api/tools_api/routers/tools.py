"""Tools router - REST endpoints for tool execution."""

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from tools_api.models import ToolDefinition, ToolResult
from tools_api.services.registry import ToolRegistry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("/")
async def list_tools() -> list[ToolDefinition]:
    """List all available tools."""
    return [
        ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            output_schema=tool.output_schema,
            category=tool.category,
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
        logger.error(f"Tool not found: {tool_name}")
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    logger.info(f"Executing tool: {tool_name} with arguments:\n\n```\n{json.dumps(arguments, indent=2)}\n```")  # noqa: E501
    result = await tool(**arguments)

    if not result.success:
        logger.error(f"Tool execution failed: {tool_name}, error: {result.error}")
    else:
        logger.info(f"Tool execution succeeded: {tool_name}, execution_time: {result.execution_time_ms:.2f}ms")  # noqa: E501

    return result
