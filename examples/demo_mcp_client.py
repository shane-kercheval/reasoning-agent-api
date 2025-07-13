#!/usr/bin/env python3
"""
Demo script for MCP Client integration.

This script demonstrates how to connect to and use MCP servers with the
reasoning agent's MCP client. It shows tool discovery, execution, and
error handling.
"""

import asyncio
import json
import time
import socket
import sys
from pathlib import Path

# Add parent directory to path for importing api modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.mcp_client import MCPClient, MCPServerConfig


async def wait_for_server_ready(host: str = "localhost", port: int = 8001, timeout: int = 10) -> bool:
    """Wait for MCP server to be ready to accept connections."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:  # Port is open
                return True
        except Exception:
            pass
        await asyncio.sleep(0.1)
    return False


async def demo_mcp_client_basic() -> None:
    """Demonstrate basic MCP client functionality."""
    print("🔧 MCP Client Basic Demo")
    print("=" * 50)

    # Configure MCP server
    config = MCPServerConfig(
        name="reasoning_agent_tools",
        url="http://localhost:8001/mcp/",
        tools=["web_search", "weather_api", "filesystem", "search_news"],
    )

    client = MCPClient([config])

    print("📋 Discovering available tools...")
    tools = await client.list_tools()
    print(f"Found tools: {tools}")
    print()

    # Test each tool
    print("🌤️  Testing weather_api tool...")
    weather_result = await client.call_tool("weather_api", {"location": "San Francisco"})
    print(f"Weather in San Francisco: {json.dumps(weather_result, indent=2)}")
    print()

    print("🔍 Testing web_search tool...")
    search_result = await client.call_tool("web_search", {"query": "FastMCP framework"})
    print(f"Search results: {json.dumps(search_result, indent=2)}")
    print()

    print("📁 Testing filesystem tool...")
    fs_result = await client.call_tool("filesystem", {"query": "python files", "path": "/projects"})
    print(f"Filesystem results: {json.dumps(fs_result, indent=2)}")
    print()

    print("📰 Testing search_news tool...")
    news_result = await client.call_tool("search_news", {"query": "artificial intelligence"})
    print(f"News results: {json.dumps(news_result, indent=2)}")
    print()


async def demo_mcp_client_error_handling() -> None:
    """Demonstrate MCP client error handling."""
    print("⚠️  MCP Client Error Handling Demo")
    print("=" * 50)

    config = MCPServerConfig(
        name="test_server",
        url="http://localhost:8001/mcp/",
        tools=["web_search", "weather_api", "nonexistent_tool"],
    )

    client = MCPClient([config])

    print("❌ Testing non-existent tool...")
    error_result = await client.call_tool("nonexistent_tool", {"param": "value"})
    print(f"Error result: {json.dumps(error_result, indent=2)}")
    print()

    print("🔌 Testing connection to non-existent server...")
    bad_config = MCPServerConfig(
        name="bad_server",
        url="http://localhost:9999/mcp/",
        tools=["fake_tool"],
    )
    bad_client = MCPClient([bad_config])

    server_error_result = await bad_client.call_tool("fake_tool", {"query": "test"})
    print(f"Server error result: {json.dumps(server_error_result, indent=2)}")
    print()

    print("🚫 Testing tool not configured on any server...")
    unknown_tool_result = await client.call_tool("completely_unknown_tool", {"param": "value"})
    print(f"Unknown tool result: {json.dumps(unknown_tool_result, indent=2)}")
    print()


async def demo_mcp_client_multiple_servers() -> None:
    """Demonstrate MCP client with multiple server configurations."""
    print("🔗 MCP Client Multiple Servers Demo")
    print("=" * 50)

    # Configure multiple servers (second one will fail)
    configs = [
        MCPServerConfig(
            name="primary_tools",
            url="http://localhost:8001/mcp/",
            tools=["web_search", "weather_api"],
        ),
        MCPServerConfig(
            name="secondary_tools",
            url="http://localhost:8002/mcp/",
            tools=["database_query", "code_analysis"],
        ),
    ]

    client = MCPClient(configs)

    print("📋 Discovering tools from all servers...")
    all_tools = await client.list_tools()
    print(f"Tools from all servers: {json.dumps(all_tools, indent=2)}")
    print()

    # Test tool from working server
    print("✅ Testing tool from primary server...")
    result1 = await client.call_tool("weather_api", {"location": "Tokyo"})
    print(f"Primary server result: {json.dumps(result1, indent=2)}")
    print()

    # Test tool from non-working server (will return error)
    print("🔄 Testing tool from secondary server (error expected)...")
    result2 = await client.call_tool("database_query", {"sql": "SELECT * FROM users"})
    print(f"Secondary server result (error): {json.dumps(result2, indent=2)}")
    print()


async def demo_mcp_performance() -> None:
    """Demonstrate MCP client performance characteristics."""
    print("⚡ MCP Client Performance Demo")
    print("=" * 50)

    config = MCPServerConfig(
        name="perf_test",
        url="http://localhost:8001/mcp/",
        tools=["weather_api", "web_search", "filesystem", "search_news"],
    )

    client = MCPClient([config])

    # Sequential execution
    print("⏱️  Testing sequential tool execution...")
    start_time = time.time()

    await client.call_tool("weather_api", {"location": "London"})
    await client.call_tool("web_search", {"query": "London weather"})
    await client.call_tool("search_news", {"query": "London events"})

    sequential_time = time.time() - start_time
    print(f"Sequential execution time: {sequential_time:.2f} seconds")
    print()

    # Parallel execution (simulated)
    print("🚀 Testing parallel tool execution...")
    start_time = time.time()

    tasks = [
        client.call_tool("weather_api", {"location": "Paris"}),
        client.call_tool("web_search", {"query": "Paris attractions"}),
        client.call_tool("search_news", {"query": "Paris events"}),
    ]

    results = await asyncio.gather(*tasks)
    parallel_time = time.time() - start_time

    print(f"Parallel execution time: {parallel_time:.2f} seconds")
    print(f"Performance improvement: {((sequential_time - parallel_time) / sequential_time * 100):.1f}%")
    print(f"All results completed successfully: {len(results)} tools")
    print()


async def main() -> None:
    """Run all MCP client demos."""
    print("🎯 MCP Client Integration Demo")
    print("=" * 60)
    print()

    # Check if MCP server is running
    print("🔍 Checking if MCP server is running...")
    if not await wait_for_server_ready():
        print("❌ MCP server not found on localhost:8001")
        print("💡 Please start the MCP server first:")
        print("   uv run python mcp_server/server.py")
        return

    print("✅ MCP server is running on localhost:8001")
    print()

    try:
        # Run all demos
        await demo_mcp_client_basic()
        await demo_mcp_client_error_handling()
        await demo_mcp_client_multiple_servers()
        await demo_mcp_performance()

        print("🎉 All MCP client demos completed successfully!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
