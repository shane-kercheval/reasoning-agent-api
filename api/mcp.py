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
from api.prompts import Prompt, PromptResult
from api.config import settings

logger = logging.getLogger(__name__)


def is_tool_deprecated(description: str) -> bool:
    """
    Check if a tool is marked as deprecated based on its description.

    Looks for common deprecation markers:
    - "DEPRECATED" (case-insensitive)
    - "(DEPRECATED)"
    - "DEPRECATED:"

    Args:
        description: Tool description to check

    Returns:
        True if tool appears to be deprecated, False otherwise

    Example:
        >>> is_tool_deprecated("Read a file. DEPRECATED: Use read_text_file instead.")
        True
        >>> is_tool_deprecated("Read a file from disk")
        False
    """
    if not description:
        return False

    # Case-insensitive search for DEPRECATED
    return "deprecated" in description.lower()


def strip_name_prefixes(name: str, prefixes: list[str]) -> str:
    """
    Strip configured prefixes from a tool/prompt name.

    Tries each prefix in order and returns the name with the first matching prefix removed.
    Used to remove proxy server prefixes (e.g., 'local_bridge_') from tool/prompt names
    when the MCP server is just a proxy to other servers.

    Args:
        name: The tool/prompt name (potentially with prefix)
        prefixes: List of prefixes to try stripping (empty list means no stripping)

    Returns:
        Name with first matching prefix removed, otherwise original name

    Example:
        >>> strip_name_prefixes("local_bridge_github__create_pr", ["local_bridge_", "proxy_"])
        'github__create_pr'
        >>> strip_name_prefixes("proxy_tool_name", ["local_bridge_", "proxy_"])
        'tool_name'
        >>> strip_name_prefixes("some_tool", [])
        'some_tool'
    """
    if not prefixes:
        return name

    for prefix in prefixes:
        if name.startswith(prefix):
            return name.removeprefix(prefix)

    return name


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


async def to_tools(client: Client, filter_deprecated: bool | None = None) -> list[Tool]:
    """
    Convert MCP client tools to generic Tool objects.

    Creates Tool wrappers around MCP tools that can be used by the reasoning agent.
    Tool names are automatically prefixed by FastMCP with server names if multiple servers.
    Each tool wrapper manages its own client context when called.

    Args:
        client: Configured FastMCP Client instance
        filter_deprecated: Whether to filter out deprecated tools (defaults to settings value)

    Returns:
        List of Tool objects from all connected servers (excluding deprecated if filtered)

    Example:
        client = create_mcp_client("config/mcp_servers.json")
        tools = await to_tools(client)
        # Tools handle client context internally when called
    """
    # Use provided value or fall back to settings
    if filter_deprecated is None:
        filter_deprecated = settings.mcp_filter_deprecated

    try:
        # List tools using client context manager
        async with client:
            mcp_tools = await client.list_tools()

        tools = []
        deprecated_count = 0

        for mcp_tool in mcp_tools:
            description = mcp_tool.description or "No description available"

            # Skip deprecated tools if filtering is enabled
            if filter_deprecated and is_tool_deprecated(description):
                deprecated_count += 1
                logger.debug(f"Filtering out deprecated tool: {mcp_tool.name}")
                continue

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

            # Strip configured prefixes (e.g., 'local_bridge_' for proxy servers)
            tool_name = strip_name_prefixes(mcp_tool.name, settings.mcp_prefixes_to_strip)

            tool = Tool(
                name=tool_name,
                description=description,
                input_schema=mcp_tool.inputSchema or {},
                function=tool_function,
            )
            tools.append(tool)

        if deprecated_count > 0:
            logger.info(f"Filtered out {deprecated_count} deprecated tool(s)")

        logger.info(f"Converted {len(tools)} MCP tools to Tool objects")
        return tools

    except Exception as e:
        logger.error(f"Failed to convert MCP tools: {e}")
        raise


async def to_prompts(client: Client) -> list[Prompt]:
    """
    Convert MCP client prompts to generic Prompt objects.

    Creates Prompt wrappers around MCP prompts that can be used by the reasoning agent.
    Prompt names are automatically prefixed by FastMCP with server names if multiple servers.
    Each prompt wrapper manages its own client context when called.

    Args:
        client: Configured FastMCP Client instance

    Returns:
        List of Prompt objects from all connected servers

    Example:
        client = create_mcp_client("config/mcp_servers.json")
        prompts = await to_prompts(client)
        # Prompts handle client context internally when called
    """
    try:
        # List prompts using client context manager
        async with client:
            mcp_prompts = await client.list_prompts()

        prompts = []

        for mcp_prompt in mcp_prompts:
            description = mcp_prompt.description or "No description available"

            # Convert MCP argument schema to our format
            # MCP arguments are a list of {name, description, required} dicts
            arguments = []
            if hasattr(mcp_prompt, 'arguments') and mcp_prompt.arguments:
                for arg in mcp_prompt.arguments:
                    arguments.append({
                        "name": arg.name,
                        "required": arg.required if hasattr(arg, 'required') else False,
                        "description": arg.description if hasattr(arg, 'description') else "",
                    })

            # Create wrapper function that calls MCP prompt
            # Use default parameter to capture prompt_name properly in closure
            def create_prompt_wrapper(prompt_name: str = mcp_prompt.name) -> Callable:
                async def wrapper(**kwargs) -> PromptResult:  # noqa: ANN003
                    # Call prompt using the client in a context manager
                    async with client:
                        result = await client.get_prompt(prompt_name, kwargs)

                        # Convert MCP PromptMessage objects to dict format
                        messages = []
                        if hasattr(result, 'messages') and result.messages:
                            for msg in result.messages:
                                # MCP PromptMessage has role and content attributes
                                message_dict = {"role": msg.role}

                                # Handle different content types
                                if hasattr(msg.content, 'text'):
                                    # TextContent type
                                    message_dict["content"] = msg.content.text
                                elif isinstance(msg.content, str):
                                    # Direct string content
                                    message_dict["content"] = msg.content
                                else:
                                    # Other content types - use string representation
                                    message_dict["content"] = str(msg.content)

                                messages.append(message_dict)

                        return PromptResult(
                            prompt_name=prompt_name,
                            success=True,
                            messages=messages,
                            execution_time_ms=0,  # Will be set by Prompt.__call__
                        )

                return wrapper

            prompt_function = create_prompt_wrapper()

            # Strip configured prefixes (e.g., 'local_bridge_' for proxy servers)
            prompt_name = strip_name_prefixes(mcp_prompt.name, settings.mcp_prefixes_to_strip)

            prompt = Prompt(
                name=prompt_name,
                description=description,
                arguments=arguments,
                function=prompt_function,
            )
            prompts.append(prompt)

        logger.info(f"Converted {len(prompts)} MCP prompts to Prompt objects")
        return prompts

    except Exception as e:
        logger.error(f"Failed to convert MCP prompts: {e}")
        raise
