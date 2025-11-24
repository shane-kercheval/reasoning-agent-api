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

import yaml
from pydantic import BaseModel, Field

from fastmcp import Client

from reasoning_api.tools import Tool
from reasoning_api.prompts import Prompt, PromptResult
from reasoning_api.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# MCP Naming Convention Support
# ============================================================================

class ParsedMCPName(BaseModel):
    """Parsed components of an MCP tool/prompt name."""

    base_name: str = Field(description="Clean base name (e.g., 'get_pr_info')")
    server_name: str | None = Field(
        default=None,
        description="Source server name (e.g., 'github-custom')",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Semantic tags for categorization",
    )
    mcp_name: str = Field(description="Original full MCP name")
    disabled: bool = Field(
        default=False,
        description="Whether this tool/prompt should be excluded from results",
    )


class NamingConfig(BaseModel):
    """Configuration for MCP naming conventions."""

    server_tags: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Default tags for each server",
    )
    tool_overrides: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Tool-specific overrides (name and tags)",
    )
    prompt_overrides: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Prompt-specific overrides (name and tags)",
    )

    @classmethod
    def load_from_yaml(cls, config_path: str | Path) -> "NamingConfig":
        """
        Load naming configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            NamingConfig instance with loaded settings

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If YAML format is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Naming config file not found: {config_path}")

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}

            return cls(
                server_tags=data.get("server_tags") or {},
                tool_overrides=data.get("tools") or {},
                prompt_overrides=data.get("prompts") or {},
            )

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in naming config: {e}")


def validate_tag(tag: str) -> None:
    """
    Validate tag format.

    Rules:
    - Lowercase alphanumeric with hyphens
    - Must start and end with alphanumeric
    - Pattern: ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$

    Valid: "git", "pull-request", "code-review", "a"
    Invalid: "Git", "pull request", "-git", "git_pull"

    Args:
        tag: Tag string to validate

    Raises:
        ValueError: If tag format is invalid
    """
    pattern = r'^[a-z0-9]([a-z0-9-]*[a-z0-9])?$'
    if not re.match(pattern, tag):
        raise ValueError(
            f"Invalid tag format: '{tag}'. "
            f"Tags must be lowercase alphanumeric with hyphens, "
            f"starting and ending with alphanumeric characters.",
        )


def parse_mcp_name(  # noqa: PLR0912
    raw_name: str,
    config: NamingConfig,
    resource_type: str,
) -> ParsedMCPName:
    """
    Parse MCP tool/prompt name into components.

    Process:
    1. Check manual override (exact match only - highest priority)
    2. Parse server__name pattern (double underscore separator)
    3. Merge server-level tags with resource-specific tags
    4. Validate all tags against format rules
    5. Remove duplicate tags (preserve order)

    Args:
        raw_name: Original MCP tool/prompt name
        config: Naming configuration
        resource_type: "tool" or "prompt"

    Returns:
        ParsedMCPName with base_name, server_name, tags, mcp_name

    Raises:
        ValueError: If tag validation fails

    Example:
        >>> config = NamingConfig(strip_prefixes=["local_bridge_"])
        >>> parse_mcp_name("local_bridge_github_custom__get_pr", config, "tool")
        ParsedMCPName(
            base_name="get_pr",
            server_name="github-custom",
            tags=[],
            mcp_name="local_bridge_github_custom__get_pr"
        )
    """
    # Determine override dict based on resource type
    overrides = config.tool_overrides if resource_type == "tool" else config.prompt_overrides

    # Check for exact match override (priority #1)
    if raw_name in overrides:
        override = overrides[raw_name]
        override_tags = override.get("tags", [])
        disabled = override.get("disable", False)

        # Determine base_name: use explicit override or auto-parse
        if "name" in override:
            base_name = override["name"]
        else:
            # Auto-parse if no explicit name provided
            base_name = raw_name
            if "__" in raw_name:
                parts = raw_name.rsplit("__", 1)
                if len(parts) == 2:
                    base_name = parts[1]

        # Extract server name from raw_name if present (for server tag lookup)
        server_name = None
        if "__" in raw_name:
            server_part = raw_name.split("__")[0]
            server_name = server_part.replace("_", "-")

        # Merge server tags with override tags
        tags = []
        if server_name and server_name in config.server_tags:
            tags.extend(config.server_tags[server_name])
        tags.extend(override_tags)

        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in tags:
            if tag not in seen:
                validate_tag(tag)
                seen.add(tag)
                unique_tags.append(tag)

        return ParsedMCPName(
            base_name=base_name,
            server_name=server_name,
            tags=unique_tags,
            mcp_name=raw_name,
            disabled=disabled,
        )

    # Auto-parse (priority #2)
    # Parse server__name pattern (last occurrence of __)
    server_name = None
    base_name = raw_name

    if "__" in raw_name:
        # Split on last __ to handle cases like "a__b__c" -> server="a__b", name="c"
        parts = raw_name.rsplit("__", 1)
        if len(parts) == 2:
            server_part, base_name = parts
            server_name = server_part.replace("_", "-")

    # Get server tags if server name exists
    tags = []
    if server_name and server_name in config.server_tags:
        tags.extend(config.server_tags[server_name])

    # Validate tags
    for tag in tags:
        validate_tag(tag)

    # Remove duplicates while preserving order
    seen = set()
    unique_tags = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)

    return ParsedMCPName(
        base_name=base_name,
        server_name=server_name,
        tags=unique_tags,
        mcp_name=raw_name,
    )


# ============================================================================
# Existing MCP Functions
# ============================================================================


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
    Tool names are automatically cleaned and parsed using naming conventions.
    Each tool wrapper manages its own client context when called.

    Args:
        client: Configured FastMCP Client instance
        filter_deprecated: Whether to filter out deprecated tools (defaults to settings value)

    Returns:
        List of Tool objects from all connected servers (excluding deprecated if filtered)

    Raises:
        ValueError: If duplicate tool names are detected after parsing

    Example:
        client = create_mcp_client("config/mcp_servers.json")
        tools = await to_tools(client)
        # Tools have clean names and metadata
    """
    # Use provided value or fall back to settings
    if filter_deprecated is None:
        filter_deprecated = settings.mcp_filter_deprecated

    # Load naming configuration
    naming_config = NamingConfig()
    config_path = Path(settings.mcp_overrides_path)
    if config_path.exists():
        try:
            naming_config = NamingConfig.load_from_yaml(config_path)
            logger.debug(f"Loaded naming overrides from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load naming config from {config_path}: {e}")

    try:
        # List tools using client context manager
        async with client:
            mcp_tools = await client.list_tools()

        tools = []
        deprecated_count = 0
        seen_names: dict[str, str] = {}  # name -> mcp_name mapping for duplicate detection

        for mcp_tool in mcp_tools:
            description = mcp_tool.description or "No description available"

            # Skip deprecated tools if filtering is enabled
            if filter_deprecated and is_tool_deprecated(description):
                deprecated_count += 1
                logger.debug(f"Filtering out deprecated tool: {mcp_tool.name}")
                continue

            # Parse MCP name to extract clean name, server, and tags
            try:
                parsed = parse_mcp_name(mcp_tool.name, naming_config, "tool")
            except ValueError as e:
                logger.error(f"Failed to parse tool name '{mcp_tool.name}': {e}")
                raise

            # Skip disabled tools
            if parsed.disabled:
                logger.debug(f"Skipping disabled tool: {mcp_tool.name}")
                continue

            # Detect duplicate names
            if parsed.base_name in seen_names:
                raise ValueError(
                    f"Duplicate tool name '{parsed.base_name}' detected:\n"
                    f"  - {seen_names[parsed.base_name]}\n"
                    f"  - {parsed.mcp_name}\n\n"
                    f"Resolve by adding overrides in {settings.mcp_overrides_path}:\n\n"
                    f"tools:\n"
                    f'  "{seen_names[parsed.base_name]}":\n'
                    f"    name: {parsed.base_name}_1\n"
                    f'  "{parsed.mcp_name}":\n'
                    f"    name: {parsed.base_name}_2\n",
                )

            seen_names[parsed.base_name] = parsed.mcp_name

            # Create wrapper function that calls MCP tool
            # Use mcp_name (not cleaned name) for actual MCP call
            def create_tool_wrapper(mcp_name: str = parsed.mcp_name) -> Callable:
                async def wrapper(**kwargs) -> object:  # noqa: ANN003
                    # Call tool using the client in a context manager
                    async with client:
                        result = await client.call_tool(mcp_name, kwargs)
                        # Return the tool result data
                        return result.data if hasattr(result, 'data') else result
                return wrapper

            tool_function = create_tool_wrapper()

            tool = Tool(
                name=parsed.base_name,
                description=description,
                input_schema=mcp_tool.inputSchema or {},
                function=tool_function,
                server_name=parsed.server_name,
                tags=parsed.tags,
                mcp_name=parsed.mcp_name,
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
    Prompt names are automatically cleaned and parsed using naming conventions.
    Each prompt wrapper manages its own client context when called.

    Args:
        client: Configured FastMCP Client instance

    Returns:
        List of Prompt objects from all connected servers

    Raises:
        ValueError: If duplicate prompt names are detected after parsing

    Example:
        client = create_mcp_client("config/mcp_servers.json")
        prompts = await to_prompts(client)
        # Prompts have clean names and metadata
    """
    # Load naming configuration
    naming_config = NamingConfig()
    config_path = Path(settings.mcp_overrides_path)
    if config_path.exists():
        try:
            naming_config = NamingConfig.load_from_yaml(config_path)
            logger.debug(f"Loaded naming overrides from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load naming config from {config_path}: {e}")

    try:
        # List prompts using client context manager
        async with client:
            mcp_prompts = await client.list_prompts()

        prompts = []
        seen_names: dict[str, str] = {}  # name -> mcp_name mapping for duplicate detection

        for mcp_prompt in mcp_prompts:
            description = mcp_prompt.description or "No description available"

            # Parse MCP name to extract clean name, server, and tags
            try:
                parsed = parse_mcp_name(mcp_prompt.name, naming_config, "prompt")
            except ValueError as e:
                logger.error(f"Failed to parse prompt name '{mcp_prompt.name}': {e}")
                raise

            # Skip disabled prompts
            if parsed.disabled:
                logger.debug(f"Skipping disabled prompt: {mcp_prompt.name}")
                continue

            # Detect duplicate names
            if parsed.base_name in seen_names:
                raise ValueError(
                    f"Duplicate prompt name '{parsed.base_name}' detected:\n"
                    f"  - {seen_names[parsed.base_name]}\n"
                    f"  - {parsed.mcp_name}\n\n"
                    f"Resolve by adding overrides in {settings.mcp_overrides_path}:\n\n"
                    f"prompts:\n"
                    f'  "{seen_names[parsed.base_name]}":\n'
                    f"    name: {parsed.base_name}_1\n"
                    f'  "{parsed.mcp_name}":\n'
                    f"    name: {parsed.base_name}_2\n",
                )

            seen_names[parsed.base_name] = parsed.mcp_name

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
            # Use mcp_name (not cleaned name) for actual MCP call
            def create_prompt_wrapper(mcp_name: str = parsed.mcp_name) -> Callable:
                async def wrapper(**kwargs) -> PromptResult:  # noqa: ANN003
                    # Call prompt using the client in a context manager
                    async with client:
                        result = await client.get_prompt(mcp_name, kwargs)

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
                            prompt_name=mcp_name,
                            success=True,
                            messages=messages,
                        )

                return wrapper

            prompt_function = create_prompt_wrapper()

            prompt = Prompt(
                name=parsed.base_name,
                description=description,
                arguments=arguments,
                function=prompt_function,
                server_name=parsed.server_name,
                tags=parsed.tags,
                mcp_name=parsed.mcp_name,
            )
            prompts.append(prompt)

        logger.info(f"Converted {len(prompts)} MCP prompts to Prompt objects")
        return prompts

    except Exception as e:
        logger.error(f"Failed to convert MCP prompts: {e}")
        raise
