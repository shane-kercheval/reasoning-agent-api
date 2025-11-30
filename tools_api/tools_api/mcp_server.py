"""
MCP server wrapper for tools_api registries.

This module creates an MCP server that wraps the existing ToolRegistry and PromptRegistry,
exposing tools and prompts via the Model Context Protocol.

Reference: https://github.com/modelcontextprotocol/python-sdk
"""

import json
import logging

from mcp import types
from mcp.server.lowlevel import Server
from pydantic import BaseModel
from tools_api.services.registry import PromptRegistry, ToolRegistry

logger = logging.getLogger(__name__)


# Create MCP server instance
server = Server("tools-api")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """List all tools from ToolRegistry as MCP Tool types."""
    return [
        types.Tool(
            name=tool.name,
            description=f"[{tool.category}] {tool.description}" if tool.category else tool.description,  # noqa: E501
            inputSchema=tool.parameters,
        )
        for tool in ToolRegistry.list()
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Execute tool from ToolRegistry and return result as TextContent."""
    tool = ToolRegistry.get(name)
    if not tool:
        logger.warning(f"MCP call_tool: Tool '{name}' not found")
        raise ValueError(f"Tool '{name}' not found")

    logger.info(f"MCP call_tool: Starting '{name}' with args: {list(arguments.keys())}")
    result = await tool(**arguments)

    if result.success:
        logger.info(f"MCP call_tool: Completed '{name}' in {result.execution_time_ms:.1f}ms")
        # Serialize result to text content
        if isinstance(result.result, BaseModel):
            text = json.dumps(result.result.model_dump(), indent=2, default=str)
        elif isinstance(result.result, (dict, list)):
            text = json.dumps(result.result, indent=2, default=str)
        else:
            text = str(result.result)
        return [types.TextContent(type="text", text=text)]

    logger.error(f"MCP call_tool: Failed '{name}': {result.error}")
    raise Exception(result.error or "Tool execution failed")


@server.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    """List all prompts from PromptRegistry as MCP Prompt types."""
    return [
        types.Prompt(
            name=prompt.name,
            description=prompt.description,
            arguments=[
                types.PromptArgument(
                    name=arg["name"],
                    description=arg.get("description", ""),
                    required=arg.get("required", False),
                )
                for arg in prompt.arguments
            ],
        )
        for prompt in PromptRegistry.list()
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None) -> types.GetPromptResult:
    """Render prompt from PromptRegistry and return as GetPromptResult."""
    prompt = PromptRegistry.get(name)
    if not prompt:
        logger.warning(f"MCP get_prompt: Prompt '{name}' not found")
        raise ValueError(f"Prompt '{name}' not found")

    logger.info(f"MCP get_prompt: Rendering '{name}'")
    result = await prompt(**(arguments or {}))

    if result.success:
        logger.info(f"MCP get_prompt: Completed '{name}'")
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=result.content),
                ),
            ],
        )

    logger.error(f"MCP get_prompt: Failed '{name}': {result.error}")
    raise Exception(result.error or "Prompt rendering failed")
