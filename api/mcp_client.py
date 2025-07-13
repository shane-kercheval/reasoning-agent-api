"""
MCP (Model Context Protocol) client implementation using FastMCP 2.0.

This module provides a client for communicating with FastMCP servers using
the built-in FastMCP Client class for proper MCP protocol communication.
"""

from dataclasses import dataclass

from fastmcp import Client


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""

    name: str
    url: str


class MCPClient:
    """Client for communicating with MCP servers using FastMCP 2.0."""

    def __init__(self, server_configs: list[MCPServerConfig]):
        self.servers = {config.name: config for config in server_configs}

    async def list_tools(self) -> dict[str, list[str]]:
        """List available tools from all servers."""
        all_tools = {}

        for server_name, config in self.servers.items():
            try:
                client = Client(config.url)
                async with client:
                    tools = await client.list_tools()
                    all_tools[server_name] = [tool.name for tool in tools]
            except Exception as e:
                # If server is not available, don't assume any tools exist
                print(f"Warning: Cannot connect to MCP server '{server_name}' at {config.url}: {e}")
                all_tools[server_name] = []

        return all_tools

    async def call_tool(self, tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
        """Call a tool on the appropriate server."""
        # Find which server has this tool
        server_config = await self._find_server_for_tool(tool_name)
        if not server_config:
            return {"error": f"Tool '{tool_name}' not found on any server"}

        try:
            return await self._call_remote_tool(server_config, tool_name, arguments)
        except Exception as e:
            return {
                "error": f"Failed to call tool '{tool_name}' on server '{server_config.name}': {e!s}",  # noqa: E501
                "server_name": server_config.name,
                "tool_name": tool_name,
                "server_url": server_config.url,
            }

    async def _find_server_for_tool(self, tool_name: str) -> MCPServerConfig | None:
        """Find which server provides a specific tool."""
        tools_by_server = await self.list_tools()
        for server_name, tools in tools_by_server.items():
            if tool_name in tools:
                return self.servers[server_name]
        return None

    async def _call_remote_tool(
            self,
            server_config: MCPServerConfig,
            tool_name: str,
            arguments: dict[str, object],
        ) -> dict[str, object]:
        """Call actual remote MCP server using FastMCP Client."""
        client = Client(server_config.url)

        async with client:
            result = await client.call_tool(tool_name, arguments)

            # Handle FastMCP 2.0 CallToolResult - just use .data since it's structured
            if hasattr(result, 'is_error') and result.is_error:
                return {"error": f"Tool execution failed: {result}"}

            if hasattr(result, 'data') and result.data is not None:
                return result.data

            # Fallback for other result types
            return {"result": str(result)}


    async def close(self) -> None:
        """Clean up - FastMCP Client handles its own cleanup."""
        # FastMCP Client handles cleanup automatically in async context manager
        pass


# Default configuration for MCP servers
DEFAULT_MCP_CONFIG = [
    MCPServerConfig(
        name="tools",
        url="http://localhost:8001/mcp/",  # Your deployed MCP server
    ),
]
