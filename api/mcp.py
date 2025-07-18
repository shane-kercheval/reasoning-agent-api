"""
MCP (Model Context Protocol) client and management implementation.

This module provides a clean separation between individual MCP server communication
(MCPClient) and orchestration of multiple servers (MCPManager). It is designed to be
independent of the reasoning agent and can be used by any component that needs MCP functionality.

Supports both HTTP/WebSocket servers and stdio subprocess servers.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.exceptions import ClientError, ToolError
from fastmcp.client.transports import PythonStdioTransport
from pydantic import BaseModel, Field, ConfigDict
import yaml

from api.tools import Tool

logger = logging.getLogger(__name__)


class MCPServerConfig(BaseModel):
    """Configuration for MCP servers supporting HTTP/WebSocket and stdio transports."""

    name: str = Field(description="Unique name for the server")
    # HTTP/WebSocket servers
    url: str | None = Field(default=None, description="HTTP/WebSocket URL for the MCP server")
    # Stdio servers
    command: str | None = Field(default=None, description="Command to start the stdio server")
    args: list[str] = Field(default_factory=list, description="Arguments for the stdio command")
    env: dict[str, str] = Field(
        default_factory=dict, description="Environment variables for stdio server",
    )
    # Common fields
    auth_env_var: str | None = Field(
        default=None,
        description="Environment variable containing auth token (for HTTP servers)",
    )
    enabled: bool = Field(default=True, description="Whether this server is enabled")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "web_search",
                    "url": "https://mcp-web-search.example.com",
                    "auth_env_var": "WEB_SEARCH_API_KEY",
                    "enabled": True,
                },
                {
                    "name": "local_tools",
                    "url": "http://localhost:8001/mcp/",
                    "enabled": True,
                },
                {
                    "name": "stdio_server",
                    "command": "uv",
                    "args": ["run", "python", "server.py"],
                    "env": {"API_KEY": "value"},
                    "enabled": True,
                },
            ],
        },
    )

    def model_post_init(self, __context: object) -> None:
        """Validate that either url or command is provided (unless using in-memory testing)."""
        # Skip validation if both url and command are None (in-memory testing)
        if self.url is None and self.command is None:
            return
        if self.url and self.command:
            raise ValueError(f"Server '{self.name}' cannot have both 'url' and 'command'")


class ToolInfo(BaseModel):
    """Information about an available MCP tool."""

    server_name: str = Field(description="Name of the server providing this tool")
    tool_name: str = Field(description="Name of the tool")
    description: str = Field(description="Description of what the tool does")
    input_schema: dict[str, Any] = Field(description="JSON schema for tool input parameters")


class ToolRequest(BaseModel):
    """Request to execute a tool on an MCP server."""

    server_name: str = Field(description="Name of the server to execute the tool on")
    tool_name: str = Field(description="Name of the tool to execute")
    arguments: dict[str, Any] = Field(description="Arguments to pass to the tool")


class ToolResult(BaseModel):
    """Result of tool execution."""

    server_name: str = Field(description="Name of the server that executed the tool")
    tool_name: str = Field(description="Name of the tool that was executed")
    success: bool = Field(description="Whether the tool execution was successful")
    result: Any = Field(default=None, description="Tool execution result data")
    error: str | None = Field(default=None, description="Error message if execution failed")
    execution_time_ms: float = Field(description="Execution time in milliseconds")


class ToolExecutionError(Exception):
    """Exception raised when tool execution fails."""

    def __init__(self, message: str, tool_name: str, server_name: str):
        super().__init__(message)
        self.tool_name = tool_name
        self.server_name = server_name
        self.message = message


class MCPServersConfig(BaseModel):
    """Configuration for multiple MCP servers."""

    servers: list[MCPServerConfig] = Field(
        default_factory=list, description="List of MCP server configurations",
    )


class MCPClient:
    """
    Client for communicating with a single MCP server using FastMCP.

    This class handles connection management and tool operations for one MCP server.
    Supports both HTTP/WebSocket and stdio transports with session-based connections.
    """

    def __init__(self, config: MCPServerConfig):
        """
        Initialize MCP client for a single server.

        Args:
            config: Configuration for the MCP server

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validated = False
        self._server_instance = None

    def set_server_instance(self, server_instance: object) -> None:
        """Set FastMCP server instance for in-memory testing."""
        self._server_instance = server_instance

    def _create_client(self) -> Client:
        """Create a FastMCP client based on server configuration."""
        if self.config.url:
            # HTTP/WebSocket transport - auto-inferred by FastMCP
            auth = None
            if self.config.auth_env_var:
                token = os.getenv(self.config.auth_env_var)
                if token:
                    auth = token  # FastMCP automatically adds "Bearer" prefix
            return Client(self.config.url, auth=auth)
        if self.config.command:
            # Stdio transport
            transport = PythonStdioTransport(
                command=self.config.command,
                args=self.config.args,
                env=self.config.env,
            )
            return Client(transport)
        # Check if this is an in-memory server setup
        if hasattr(self, '_server_instance'):
            return Client(self._server_instance)
        raise ValueError(f"Server '{self.config.name}' needs either 'url' or 'command'")

    async def validate_connection(self) -> None:
        """
        Test connection to the MCP server.

        Raises:
            Exception: If connection fails for any reason
        """
        try:
            client = self._create_client()
            async with client:
                # Test connection by trying to list tools
                await client.list_tools()

            self._validated = True
            logger.info(f"Successfully validated connection to MCP server: {self.config.name}")

        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{self.config.name}': {e}")
            raise

    async def list_tools(self) -> list[ToolInfo]:
        """
        List all available tools from this MCP server.

        Returns:
            List of tools available from this server

        Raises:
            Exception: If server communication fails
        """
        try:
            client = self._create_client()
            async with client:
                tools_response = await client.list_tools()
                tools = []

                for tool in tools_response:
                    tool_info = ToolInfo(
                        server_name=self.config.name,
                        tool_name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema or {},
                    )
                    tools.append(tool_info)

                logger.debug(f"Listed {len(tools)} tools from server '{self.config.name}'")
                return tools

        except Exception as e:
            logger.error(f"Failed to list tools from server '{self.config.name}': {e}")
            raise

    async def call_tool(self, tool_name: str, arguments: dict[str, object]) -> object:
        """
        Call a tool on this MCP server using exception-based error handling.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result data (structured Python objects)

        Raises:
            ToolExecutionError: If tool execution fails
        """
        try:
            client = self._create_client()
            async with client:
                # Use high-level call_tool() with exception handling
                result = await client.call_tool(tool_name, arguments)
                # Use .data attribute for structured results
                return result.data
        except (ClientError, ToolError) as e:
            # Tool execution failed
            raise ToolExecutionError(
                f"Tool '{tool_name}' failed: {e}", tool_name, self.config.name,
            )
        except Exception as e:
            logger.error(f"Failed to call tool '{tool_name}' on server '{self.config.name}': {e}")
            raise


class MCPManager:
    """
    Manager for multiple MCP servers.

    This class orchestrates multiple MCPClient instances and provides unified
    tool discovery and execution across all connected servers. It implements
    fail-fast behavior when servers are unavailable.
    """

    def __init__(self, server_configs: list[MCPServerConfig]):
        """
        Initialize MCP manager with multiple server configurations.

        Args:
            server_configs: List of server configurations to manage
        """
        self.server_configs = server_configs
        self._clients: dict[str, MCPClient] = {}
        self._tool_cache: dict[str, list[ToolInfo]] = {}
        self._cache_timeout = 300  # 5 minutes
        self._last_tool_discovery = 0.0

    async def initialize(self) -> None:
        """
        Initialize connections to all configured MCP servers.

        This method implements fail-fast behavior - if any enabled server
        fails to connect, the entire initialization fails.

        Raises:
            Exception: If any enabled server fails to connect
        """
        enabled_configs = [config for config in self.server_configs if config.enabled]

        if not enabled_configs:
            logger.info("No MCP servers configured")
            return

        # Create clients for all enabled servers
        clients_to_validate = {}
        for config in enabled_configs:
            client = MCPClient(config)
            clients_to_validate[config.name] = client

        # Validate all connections in parallel (fail-fast)
        validation_tasks = []
        for name, client in clients_to_validate.items():
            task = asyncio.create_task(client.validate_connection())
            validation_tasks.append((name, client, task))

        # Wait for all validations to complete
        failed_servers = []
        for name, client, task in validation_tasks:
            try:
                await task
                self._clients[name] = client
            except Exception as e:
                failed_servers.append((name, str(e)))

        # Fail fast if any server failed to connect
        if failed_servers:
            error_details = "; ".join([f"{name}: {error}" for name, error in failed_servers])
            raise RuntimeError(f"Failed to connect to MCP servers: {error_details}")

        logger.info(f"MCP initialization complete: {len(self._clients)} servers connected")

    async def get_available_tools(self, force_refresh: bool = False) -> list[ToolInfo]:
        """
        Get all available tools from connected MCP servers.

        Args:
            force_refresh: If True, bypass cache and refresh tool list

        Returns:
            List of available tools across all connected servers

        Raises:
            Exception: If tool discovery fails on any server
        """
        current_time = time.time()

        # Check if we need to refresh the cache
        if (force_refresh or
            current_time - self._last_tool_discovery > self._cache_timeout or
            not self._tool_cache):

            await self._discover_tools()
            self._last_tool_discovery = current_time

        # Flatten all tools from all servers
        all_tools = []
        for tools in self._tool_cache.values():
            all_tools.extend(tools)

        return all_tools

    async def _discover_tools(self) -> None:
        """
        Discover tools from all connected servers and update cache.

        Raises:
            Exception: If any server fails during tool discovery
        """
        discovery_tasks = []
        for name, client in self._clients.items():
            task = asyncio.create_task(client.list_tools())
            discovery_tasks.append((name, task))

        # Wait for all discovery tasks to complete
        for name, task in discovery_tasks:
            try:
                tools = await task
                self._tool_cache[name] = tools
            except Exception as e:
                logger.error(f"Tool discovery failed for server '{name}': {e}")
                raise

    async def execute_tool(self, request: ToolRequest) -> ToolResult:
        """
        Execute a single tool on the specified MCP server.

        Args:
            request: Tool execution request

        Returns:
            Result of tool execution
        """
        start_time = time.time()

        try:
            logger.info(
                f"Executing tool: {request.tool_name} on server: {request.server_name} "
                f"with args: {request.arguments}",
            )

            # Find the client for the requested server
            if request.server_name not in self._clients:
                error_msg = f"Server '{request.server_name}' not found or not connected"
                logger.error(error_msg)
                return ToolResult(
                    server_name=request.server_name,
                    tool_name=request.tool_name,
                    success=False,
                    error=error_msg,
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

            client = self._clients[request.server_name]

            # Execute the tool using new exception-based API
            result_data = await client.call_tool(request.tool_name, request.arguments)
            execution_time = (time.time() - start_time) * 1000

            logger.info(
                f"Tool execution successful: {request.tool_name} - "
                f"Execution time: {execution_time:.2f}ms",
            )
            logger.debug(f"Tool result data: {result_data}")

            return ToolResult(
                server_name=request.server_name,
                tool_name=request.tool_name,
                success=True,
                result=result_data,
                execution_time_ms=execution_time,
            )

        except ToolExecutionError as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Tool execution failed: {request.server_name}.{request.tool_name}: {e}")

            return ToolResult(
                server_name=request.server_name,
                tool_name=request.tool_name,
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Tool execution failed: {request.server_name}.{request.tool_name}: {e}")

            return ToolResult(
                server_name=request.server_name,
                tool_name=request.tool_name,
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
            )

    async def execute_tools_parallel(self, requests: list[ToolRequest]) -> list[ToolResult]:
        """
        Execute multiple tools in parallel across different servers.

        Args:
            requests: List of tool execution requests

        Returns:
            List of tool execution results in the same order as requests
        """
        if not requests:
            return []

        # Create tasks for parallel execution
        tasks = []
        for request in requests:
            task = asyncio.create_task(self.execute_tool(request))
            tasks.append(task)

        # Wait for all tools to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert any exceptions to ToolResult objects
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                request = requests[i]
                final_results.append(ToolResult(
                    server_name=request.server_name,
                    tool_name=request.tool_name,
                    success=False,
                    error=f"Parallel execution error: {result}",
                    execution_time_ms=0.0,
                ))
            else:
                final_results.append(result)

        return final_results

    def get_connected_servers(self) -> set[str]:
        """
        Get names of successfully connected servers.

        Returns:
            Set of server names that are connected
        """
        return set(self._clients.keys())

    def is_server_connected(self, server_name: str) -> bool:
        """
        Check if a specific server is connected.

        Args:
            server_name: Name of the server to check

        Returns:
            True if server is connected, False otherwise
        """
        return server_name in self._clients

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check on all connected servers.

        Returns:
            Dictionary with health status of each server
        """
        health_status = {
            "total_servers": len(self.server_configs),
            "connected_servers": len(self._clients),
            "servers": {},
        }

        for config in self.server_configs:
            server_name = config.name
            is_connected = self.is_server_connected(server_name)

            health_status["servers"][server_name] = {
                "enabled": config.enabled,
                "connected": is_connected,
                "url": config.url,
                "tools_cached": server_name in self._tool_cache,
            }

            # Add command field only if it exists (for backward compatibility)
            if hasattr(config, 'command') and config.command:
                health_status["servers"][server_name]["command"] = config.command

            if is_connected and server_name in self._tool_cache:
                health_status["servers"][server_name]["tool_count"] = len(
                    self._tool_cache[server_name],
                )

        return health_status


# Configuration loading utilities

def load_yaml_config(path: str | Path) -> MCPServersConfig:
    """Load MCP servers configuration from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    servers = []
    for server_data in data.get("servers", []):
        servers.append(MCPServerConfig(**server_data))

    return MCPServersConfig(servers=servers)


def load_mcp_json_config(path: str | Path) -> MCPServersConfig:
    """Load MCP servers configuration from standard MCP JSON format."""
    with open(path) as f:
        data = json.load(f)

    servers = []
    for name, config in data.get("mcpServers", {}).items():
        if "command" in config:
            # Stdio server
            servers.append(MCPServerConfig(
                name=name,
                command=config["command"],
                args=config.get("args", []),
                env=config.get("env", {}),
                enabled=config.get("enabled", True),
            ))
        elif "url" in config:
            # HTTP server
            servers.append(MCPServerConfig(
                name=name,
                url=config["url"],
                auth_env_var=config.get("auth_env_var"),
                enabled=config.get("enabled", True),
            ))

    return MCPServersConfig(servers=servers)


def export_to_mcp_json(config: MCPServersConfig) -> dict:
    """Export config to standard MCP JSON format."""
    mcp_servers = {}

    for server in config.servers:
        server_config = {}

        if server.url:
            server_config["url"] = server.url
            if server.auth_env_var:
                server_config["auth_env_var"] = server.auth_env_var
        elif server.command:
            server_config["command"] = server.command
            if server.args:
                server_config["args"] = server.args
            if server.env:
                server_config["env"] = server.env

        mcp_servers[server.name] = server_config

    return {"mcpServers": mcp_servers}


def validate_mcp_config(config_path: str | Path) -> list[str]:
    """Validate MCP configuration file and return list of errors."""
    errors = []

    try:
        if str(config_path).endswith('.json'):
            config = load_mcp_json_config(config_path)
        else:
            config = load_yaml_config(config_path)

        # Validate each server config
        for server in config.servers:
            try:
                # This will run model_post_init validation
                MCPServerConfig.model_validate(server.model_dump())
            except ValueError as e:
                errors.append(f"Server '{server.name}': {e}")

        # Check for duplicate names
        names = [server.name for server in config.servers]
        duplicates = {name for name in names if names.count(name) > 1}
        for name in duplicates:
            errors.append(f"Duplicate server name: '{name}'")

    except Exception as e:
        errors.append(f"Failed to load config: {e}")

    return errors


# Tool conversion utilities

async def to_tools(client: Client) -> list[Tool]:
    """
    Convert MCP servers/tools from a FastMCP client to generic Tool objects. Wraps the MCP tool
    functions to match the Tool interface so that users can call tools directly which call MCP
    tools.

    Args:
        client: Configured FastMCP Client instance

    Returns:
        List of Tool objects from all connected servers

    Example:
        config = {"mcpServers": {"server1": {"url": "..."}}}
        client = Client(config)
        tools = await to_tools(client)
    """
    async with client:
        mcp_tools = await client.list_tools()

        tools = []
        for mcp_tool in mcp_tools:
            # Create a wrapper function that calls the MCP tool
            # Use default parameter to capture the variable properly
            def create_tool_wrapper(tool_name: str = mcp_tool.name):  # noqa: ANN202
                async def wrapper(**kwargs):  # noqa: ANN003, ANN202
                    return await client.call_tool(tool_name, kwargs)
                return wrapper

            tool_function = create_tool_wrapper()

            tool = Tool(
                name=mcp_tool.name,
                description=mcp_tool.description or "No description available",
                input_schema=mcp_tool.inputSchema or {},
                function=tool_function,
            )
            tools.append(tool)

        return tools
