"""
Test MCP Server B - provides filesystem and news tools.

This server runs on port 8002 and provides tools for testing MCPClient and MCPManager.
"""

import os
import random
from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("test-server-b")


@mcp.tool
async def filesystem(query: str, path: str = "/documents") -> dict:
    """Search and access filesystem for documents."""
    file_types = [".txt", ".md", ".pdf", ".docx", ".json"]
    fake_files = []

    for i in range(random.randint(2, 5)):
        filename = f"{query.replace(' ', '_')}_doc_{i+1}{random.choice(file_types)}"
        fake_files.append({
            "name": filename,
            "path": f"{path}/{filename}",
            "size": f"{random.randint(1, 100)} KB",
            "modified": "2025-07-12",
            "preview": f"This document contains detailed information about {query}...",
        })

    return {
        "query": query,
        "search_path": path,
        "files_found": fake_files,
        "total_files": len(fake_files),
        "search_time": f"{random.uniform(0.05, 0.2):.2f}s",
        "server": "test-server-b",
    }


@mcp.tool
async def search_news(query: str) -> dict:
    """Search for recent news articles."""
    fake_news = [
        {
            "headline": f"Breaking: Major developments in {query}",
            "summary": f"Latest news about {query} reveals significant changes in the industry...",
            "url": f"https://news.example.com/breaking-{query.replace(' ', '-')}",
            "published": "2025-07-12T10:30:00Z",
            "source": "Test News Network",
        },
        {
            "headline": f"{query} trends continue to evolve",
            "summary": f"Analysis shows that {query} is experiencing rapid growth and innovation...",  # noqa: E501
            "url": f"https://business.example.com/{query.replace(' ', '-')}-trends",
            "published": "2025-07-12T08:15:00Z",
            "source": "Business Today",
        },
    ]

    return {
        "query": query,
        "articles": fake_news,
        "total_articles": len(fake_news),
        "server": "test-server-b",
    }


def get_server_instance():
    """Get the FastMCP server instance for in-memory testing."""
    return mcp


if __name__ == "__main__":
    # Run as HTTP server on configurable port (default 8002)
    port = int(os.getenv("PORT", "8002"))
    mcp.run(transport="http", host="0.0.0.0", port=port)
