"""
Test MCP Server A - provides weather and search tools.

This server runs on port 8001 and provides tools for testing MCPClient and MCPManager.
"""

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
            "wind": f"{random.randint(5, 20)} km/h {random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])}",
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
            "snippet": f"Comprehensive information about {query}. This article covers recent developments...",
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


if __name__ == "__main__":
    # Run as HTTP server on port 8001
    mcp.run(transport="http", host="0.0.0.0", port=8001)
