"""
MCP (Model Context Protocol) client and management implementation.

This module provides a clean separation between individual MCP server communication
(MCPClient) and orchestration of multiple servers (MCPManager). It is designed to be
independent of the reasoning agent and can be used by any component that needs MCP functionality.
"""

import asyncio
import logging
import time
from typing import Any

from fastmcp import Client
from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


class MCPServerConfig(BaseModel):
    """Configuration for an HTTP-based MCP server."""

    name: str = Field(description="Unique name for the server")
    url: str = Field(description="HTTP URL for the MCP server")
    auth_env_var: str | None = Field(
        default=None,
        description="Environment variable containing auth token",
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
            ],
        },
    )


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


class MCPClient:
    """
    Client for communicating with a single MCP server using FastMCP.

    This class handles connection management and tool operations for one MCP server.
    It uses async context managers for each operation and does not maintain persistent connections.
    """

    def __init__(self, config: MCPServerConfig):
        """
        Initialize MCP client for a single server.

        Args:
            config: Configuration for the MCP server

        Raises:
            ValueError: If configuration is invalid
        """
        if not config.url:
            raise ValueError(f"MCP server '{config.name}' requires a URL")

        self.config = config
        self._validated = False

    async def validate_connection(self) -> None:
        """
        Test connection to the MCP server.

        Raises:
            Exception: If connection fails for any reason
        """
        try:
            client = Client(self.config.url)
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
            client = Client(self.config.url)
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
        Call a tool on this MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result data

        Raises:
            Exception: If tool execution fails
        """
        try:
            client = Client(self.config.url)
            async with client:
                # Use call_tool_mcp to get raw result without exception on error
                response = await client.call_tool_mcp(tool_name, arguments)

                # Handle FastMCP 2.0 CallToolResult format
                if hasattr(response, 'isError') and response.isError:
                    # Tool execution failed but MCP communication succeeded
                    # Return error information instead of raising exception
                    error_content = "Unknown error"
                    if hasattr(response, 'content') and response.content:
                        if isinstance(response.content, list) and response.content:
                            first_content = response.content[0]
                            if hasattr(first_content, 'text'):
                                error_content = first_content.text
                            else:
                                error_content = str(first_content)
                        else:
                            error_content = str(response.content)

                    return {
                        "error": True,
                        "error_message": error_content,
                        "tool_name": tool_name,
                        "server_name": self.config.name,
                    }

                # Use .structuredContent attribute which contains structured response in FastMCP 2.0
                if hasattr(response, 'structuredContent') and response.structuredContent is not None:
                    return response.structuredContent

                # Fallback for other result types
                return {"result": str(response)}

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
            # Find the client for the requested server
            if request.server_name not in self._clients:
                return ToolResult(
                    server_name=request.server_name,
                    tool_name=request.tool_name,
                    success=False,
                    error=f"Server '{request.server_name}' not found or not connected",
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

            client = self._clients[request.server_name]

            # Execute the tool
            result_data = await client.call_tool(request.tool_name, request.arguments)
            execution_time = (time.time() - start_time) * 1000

            return ToolResult(
                server_name=request.server_name,
                tool_name=request.tool_name,
                success=True,
                result=result_data,
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

            if is_connected and server_name in self._tool_cache:
                health_status["servers"][server_name]["tool_count"] = len(self._tool_cache[server_name])  # noqa: E501

        return health_status
