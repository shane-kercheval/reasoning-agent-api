"""
Math operations stdio MCP server for testing.

Provides basic math operations to test multiple server functionality.
"""

from fastmcp import FastMCP

mcp = FastMCP("test-math")


@mcp.tool
async def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@mcp.tool
async def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


if __name__ == "__main__":
    mcp.run(transport="stdio")
