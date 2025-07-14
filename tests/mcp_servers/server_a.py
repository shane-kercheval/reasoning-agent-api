"""
Demo MCP Server - provides weather and search tools for testing.

This server runs on port 8001 and provides example tools that can be used
with the reasoning agent for demonstrations and development.

Available tools:
- weather_api: Get fake weather data for any location
- web_search: Simulate web search with fake results
- failing_tool: Tool that can be made to fail for error testing

Usage:
    uv run python tests/mcp_servers/server_a.py

The server will start on http://localhost:8001/mcp/
"""

import os
import random
from datetime import datetime
from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("test-server-a")


@mcp.tool
async def weather_api(location: str = "San Francisco") -> dict:
    """Get current weather information for a location."""
    conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Clear"]
    temperatures = list(range(15, 28))

    return {
        "location": location,
        "current": {
            "temperature": f"{random.choice(temperatures)}°C",
            "condition": random.choice(conditions),
            "humidity": f"{random.randint(40, 80)}%",
            "wind": f"{random.randint(5, 20)} km/h {random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])}",  # noqa: E501
            "pressure": f"{random.randint(1010, 1025)} mb",
        },
        "forecast": [
            {
                "date": "Today",
                "high": f"{random.randint(20, 30)}°C",
                "low": f"{random.randint(10, 18)}°C",
                "condition": random.choice(conditions),
            },
            {
                "date": "Tomorrow",
                "high": f"{random.randint(20, 30)}°C",
                "low": f"{random.randint(10, 18)}°C",
                "condition": random.choice(conditions),
            },
        ],
        "last_updated": datetime.now().isoformat(),
        "source": "Test Weather API",
        "server": "test-server-a",
    }


@mcp.tool
async def web_search(query: str) -> dict:
    """Search the web for information."""
    fake_results = [
        {
            "title": f"Everything about {query}",
            "url": f"https://example.com/search/{query.replace(' ', '-')}",
            "snippet": f"Comprehensive information about {query}. This article covers recent developments...",  # noqa: E501
            "date": "2025-07-12",
            "source": "Example News",
        },
        {
            "title": f"Latest {query} updates",
            "url": f"https://news.example.com/{query.replace(' ', '-')}-updates",
            "snippet": f"Recent news about {query} shows significant developments in the field...",
            "date": "2025-07-11",
            "source": "Tech Today",
        },
    ]

    return {
        "query": query,
        "results": fake_results,
        "total_results": len(fake_results),
        "search_time": f"{random.uniform(0.1, 0.5):.2f}s",
        "server": "test-server-a",
    }


@mcp.tool
async def failing_tool(should_fail: bool = True) -> dict:
    """A tool that intentionally fails for testing error handling."""
    if should_fail:
        # This will cause FastMCP to return is_error=True
        raise ValueError("This tool intentionally failed for testing purposes")

    return {
        "success": True,
        "message": "Tool executed successfully",
        "server": "test-server-a",
    }


def get_server_instance():
    """Get the FastMCP server instance for in-memory testing."""
    return mcp


if __name__ == "__main__":
    # Run as HTTP server on configurable port (default 8001)
    port = int(os.getenv("PORT", "8001"))
    mcp.run(transport="http", host="0.0.0.0", port=port)
