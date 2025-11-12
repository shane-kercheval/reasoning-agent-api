"""
Simple stdio MCP server for testing.

Provides basic echo functionality to test bridge connectivity.
"""

from fastmcp import FastMCP

mcp = FastMCP("test-echo")


@mcp.tool
async def echo(message: str) -> str:
    """Echo a message back."""
    return message


@mcp.tool
async def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


if __name__ == "__main__":
    mcp.run(transport="stdio")
