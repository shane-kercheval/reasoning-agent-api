"""MCP Server Manager for connection management and tool execution."""
import asyncio
import logging
import time
import json
from typing import Any

from fastmcp import Client

from .reasoning_models import MCPServerConfig, ToolInfo, ToolRequest, ToolResult

logger = logging.getLogger(__name__)


class MCPServerManager:
    """
    Manages MCP server connections and tool execution.

    This class handles both connection management AND tool execution to avoid
    over-engineering. It supports multiple HTTP-based MCP servers and provides
    unified tool discovery and execution capabilities.
    """

    def __init__(self, server_configs: list[MCPServerConfig]):
        """
        Initialize the MCP server manager.

        Args:
            server_configs: List of server configurations to manage
        """
        self.server_configs = server_configs
        self._connection_cache: dict[str, bool] = {}
        self._tool_cache: dict[str, list[ToolInfo]] = {}
        self._cache_timeout = 300  # 5 minutes
        self._last_tool_discovery = 0.0

    async def initialize(self) -> None:
        """
        Initialize connections to all configured MCP servers.

        This method attempts to test connections to all enabled HTTP-based MCP servers
        but continues gracefully if some servers fail to connect. Failed connections are
        logged but do not stop the initialization process.
        """
        connection_tasks = []
        enabled_configs = [config for config in self.server_configs if config.enabled]

        for config in enabled_configs:
            task = asyncio.create_task(self._test_server_connection(config))
            connection_tasks.append(task)

        if connection_tasks:
            # Wait for all connection attempts to complete
            results = await asyncio.gather(*connection_tasks, return_exceptions=True)

            successful_connections = 0
            for i, result in enumerate(results):
                config = enabled_configs[i]
                if isinstance(result, Exception):
                    logger.warning(
                        f"Failed to connect to MCP server '{config.name}': {result}",
                    )
                    self._connection_cache[config.name] = False
                else:
                    successful_connections += 1
                    self._connection_cache[config.name] = True

            logger.info(
                f"MCP initialization complete: {successful_connections}/{len(enabled_configs)} servers connected",  # noqa: E501
            )
        else:
            logger.info("No MCP servers configured")

    async def _test_server_connection(self, config: MCPServerConfig) -> None:
        """
        Test connection to a single MCP server.

        Args:
            config: Configuration for the server to test

        Raises:
            Exception: If connection fails for any reason
        """
        if not config.url:
            raise ValueError(f"MCP server '{config.name}' requires a URL")

        try:
            client = Client(config.url)
            async with client:
                # Test connection by trying to list tools
                await client.list_tools()
            logger.info(f"Successfully connected to MCP server: {config.name}")

        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{config.name}': {e}")
            raise

    def _create_client(self, config: MCPServerConfig) -> Client:
        """
        Create a FastMCP client for the given configuration.

        Args:
            config: Server configuration

        Returns:
            Configured FastMCP client instance

        Raises:
            ValueError: If configuration is invalid
        """
        if not config.url:
            raise ValueError(f"MCP server '{config.name}' requires a URL")

        # TODO: Add authentication support for FastMCP client
        # Currently FastMCP Client doesn't expose headers in constructor
        return Client(config.url)

    async def get_available_tools(self, force_refresh: bool = False) -> list[ToolInfo]:
        """
        Get all available tools from connected MCP servers.

        Args:
            force_refresh: If True, bypass cache and refresh tool list

        Returns:
            List of available tools across all connected servers
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
        """Discover tools from all connected servers and update cache."""
        discovery_tasks = []
        enabled_configs = [config for config in self.server_configs if config.enabled]

        for config in enabled_configs:
            if self._connection_cache.get(config.name, False):
                task = asyncio.create_task(self._discover_server_tools(config))
                discovery_tasks.append(task)

        if discovery_tasks:
            results = await asyncio.gather(*discovery_tasks, return_exceptions=True)

            for i, result in enumerate(results):
                config = [
                    c for c in enabled_configs
                    if self._connection_cache.get(c.name, False)
                ][i]
                if isinstance(result, Exception):
                    logger.warning(f"Tool discovery failed for server '{config.name}': {result}")
                    self._tool_cache[config.name] = []
                else:
                    self._tool_cache[config.name] = result

    async def _discover_server_tools(self, config: MCPServerConfig) -> list[ToolInfo]:
        """
        Discover tools from a specific server.

        Args:
            config: Server configuration

        Returns:
            List of tools available from this server
        """
        try:
            client = self._create_client(config)
            async with client:
                tools_response = await client.list_tools()
                tools = []

                for tool in tools_response:
                    tool_info = ToolInfo(
                        server_name=config.name,
                        tool_name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema or {},
                    )
                    tools.append(tool_info)

                logger.debug(f"Discovered {len(tools)} tools from server '{config.name}'")
                return tools

        except Exception as e:
            logger.error(f"Failed to list tools from server '{config.name}': {e}")
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
            # Find the server configuration
            server_config = None
            for config in self.server_configs:
                if config.name == request.server_name and config.enabled:
                    server_config = config
                    break

            if not server_config:
                return ToolResult(
                    server_name=request.server_name,
                    tool_name=request.tool_name,
                    success=False,
                    error=f"Server '{request.server_name}' not found or disabled",
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

            if not self._connection_cache.get(request.server_name, False):
                return ToolResult(
                    server_name=request.server_name,
                    tool_name=request.tool_name,
                    success=False,
                    error=f"Server '{request.server_name}' not connected",
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

            # Execute the tool using FastMCP client
            client = self._create_client(server_config)
            async with client:
                response = await client.call_tool(request.tool_name, request.arguments)

                # Handle different response types from FastMCP
                result_data = self._parse_tool_response(response)

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

    def _parse_tool_response(self, response: Any) -> Any:  # noqa: ANN401, PLR0911, PLR0912
        """
        Parse tool response from FastMCP client.

        Args:
            response: Response from FastMCP client (mcp.types.CallToolResult)

        Returns:
            Parsed response data
        """
        # Handle mcp.types.CallToolResult format
        if hasattr(response, 'structuredContent') and response.structuredContent is not None:
            # Prefer structured content if available
            return response.structuredContent
        if hasattr(response, 'content') and response.content:
            # Parse content list - extract text from TextContent items
            if isinstance(response.content, list):
                text_contents = []
                for content_item in response.content:
                    if hasattr(content_item, 'text'):
                        text_contents.append(content_item.text)
                    elif hasattr(content_item, 'data'):
                        text_contents.append(str(content_item.data))
                    else:
                        text_contents.append(str(content_item))

                # If only one text content, try to parse as JSON
                if len(text_contents) == 1:
                    try:
                        return json.loads(text_contents[0])
                    except (json.JSONDecodeError, TypeError):
                        return {"result": text_contents[0]}
                else:
                    # Multiple content items
                    return {"content": text_contents}

            # Fallback for legacy content format (dict/string)
            elif isinstance(response.content, dict):
                return response.content
            elif isinstance(response.content, str):
                try:
                    return json.loads(response.content)
                except json.JSONDecodeError:
                    return {"result": response.content}
            else:
                return {"result": str(response.content)}
        elif hasattr(response, 'data'):
            # Some FastMCP results use .data (legacy)
            return response.data if isinstance(response.data, dict) else {"result": response.data}
        else:
            # Fallback for other result types
            return {"result": str(response)}

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
        return {
            name for name, connected in self._connection_cache.items()
            if connected
        }

    def is_server_connected(self, server_name: str) -> bool:
        """
        Check if a specific server is connected.

        Args:
            server_name: Name of the server to check

        Returns:
            True if server is connected, False otherwise
        """
        return self._connection_cache.get(server_name, False)

    async def cleanup(self) -> None:
        """
        Clean up all server connections.

        FastMCP clients handle their own cleanup automatically through
        async context managers, so we just need to clear our caches.
        """
        self._connection_cache.clear()
        self._tool_cache.clear()
        logger.info("MCP server manager cleanup complete")

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check on all connected servers.

        Returns:
            Dictionary with health status of each server
        """
        health_status = {
            "total_servers": len(self.server_configs),
            "connected_servers": len(self.get_connected_servers()),
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
