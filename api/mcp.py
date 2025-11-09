"""
MCP (Model Context Protocol) integration using FastMCP.

This module provides FastMCP client integration and MCP-to-Tool conversion
for the reasoning agent. Uses standard JSON configuration format.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from collections.abc import Callable

from fastmcp import Client

from api.tools import Tool

logger = logging.getLogger(__name__)


def _expand_env_vars(value: Any) -> Any:
    """
    Recursively expand environment variables in config values.

    Supports ${VAR} and ${VAR:-default} syntax.

    Args:
        value: Config value (string, dict, list, or other)

    Returns:
        Value with environment variables expanded
    """
    if isinstance(value, str):
        # Match ${VAR} or ${VAR:-default}
        pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'

        def replace_env(match: re.Match) -> str:
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            return os.getenv(var_name, default_value)

        return re.sub(pattern, replace_env, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def load_mcp_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load MCP configuration from standard JSON format.

    Supports environment variable expansion in config values using:
    - ${VAR} - Replaced with environment variable value (empty if not set)
    - ${VAR:-default} - Replaced with environment variable or default value

    Args:
        config_path: Path to JSON configuration file

    Returns:
        MCP configuration dictionary with environment variables expanded

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"MCP configuration file not found: {config_path}")

    try:
        with open(config_path) as f:
            config = json.load(f)

        if "mcpServers" not in config:
            raise ValueError("Invalid MCP config: missing 'mcpServers' key")

        # Expand environment variables in config
        return _expand_env_vars(config)


    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in MCP config: {e}")


def create_mcp_client(config_path: str | Path) -> Client:
    """
    Create configured FastMCP client from JSON config file.

    Args:
        config_path: Path to MCP JSON configuration file

    Returns:
        Configured FastMCP Client instance

    Example:
        client = create_mcp_client("config/mcp_servers.json")
        tools = await to_tools(client)
    """
    config = load_mcp_config(config_path)
    return Client(config)


async def to_tools(client: Client) -> list[Tool]:
    """
    Convert MCP client tools to generic Tool objects.

    Creates Tool wrappers around MCP tools that can be used by the reasoning agent.
    Tool names are automatically prefixed by FastMCP with server names if multiple servers.
    Each tool wrapper manages its own client context when called.

    Args:
        client: Configured FastMCP Client instance

    Returns:
        List of Tool objects from all connected servers

    Example:
        client = create_mcp_client("config/mcp_servers.json")
        tools = await to_tools(client)
        # Tools handle client context internally when called
    """
    try:
        # List tools using client context manager
        async with client:
            mcp_tools = await client.list_tools()

        tools = []
        for mcp_tool in mcp_tools:

            # Create wrapper function that calls MCP tool
            # Use default parameter to capture tool_name properly in closure
            def create_tool_wrapper(tool_name: str = mcp_tool.name) -> Callable:
                async def wrapper(**kwargs) -> object:  # noqa: ANN003
                    # Call tool using the client in a context manager
                    async with client:
                        result = await client.call_tool(tool_name, kwargs)
                        # Return the tool result data
                        return result.data if hasattr(result, 'data') else result
                return wrapper

            tool_function = create_tool_wrapper()

            tool = Tool(
                name=mcp_tool.name,  # Already prefixed by FastMCP if multiple servers
                description=mcp_tool.description or "No description available",
                input_schema=mcp_tool.inputSchema or {},
                function=tool_function,
            )
            tools.append(tool)

        logger.info(f"Converted {len(tools)} MCP tools to Tool objects")
        return tools

    except Exception as e:
        logger.error(f"Failed to convert MCP tools: {e}")
        raise
