#!/usr/bin/env python3
"""
Simple integration test script for MCP client.

This script provides a quick way to test MCP client functionality
without running the full test suite. It demonstrates proper error
handling and successful tool execution.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path for importing api modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.mcp_client import MCPClient, MCPServerConfig


async def main() -> None:
    """Test MCP client integration with real and fake servers."""
    print("🔧 MCP Client Integration Test")
    print("=" * 40)

    # Test 1: Working MCP server
    print("✅ Test 1: Working MCP server on localhost:8001")
    working_config = MCPServerConfig(
        name="working_server",
        url="http://localhost:8001/mcp/",
        tools=["weather_api", "web_search"],
    )

    working_client = MCPClient([working_config])

    # Test tool discovery
    tools = await working_client.list_tools()
    print(f"Tools discovered: {json.dumps(tools, indent=2)}")

    # Test successful tool call
    weather_result = await working_client.call_tool("weather_api", {"location": "London"})
    if "error" in weather_result:
        print(f"❌ Weather tool failed: {weather_result['error']}")
    else:
        print(f"✅ Weather tool success: Location={weather_result.get('location')}, Temp={weather_result.get('current', {}).get('temperature')}")

    print()

    # Test 2: Non-existent server
    print("❌ Test 2: Non-existent MCP server on localhost:9999")
    broken_config = MCPServerConfig(
        name="broken_server",
        url="http://localhost:9999/mcp/",
        tools=["fake_tool"],
    )

    broken_client = MCPClient([broken_config])

    # Test error handling
    error_result = await broken_client.call_tool("fake_tool", {"param": "value"})
    if "error" in error_result:
        print(f"✅ Expected error: {error_result['error']}")
        print(f"   Server: {error_result.get('server_name')}")
        print(f"   Tool: {error_result.get('tool_name')}")
    else:
        print(f"❌ Unexpected success: {error_result}")

    print()

    # Test 3: Unknown tool on working server
    print("❓ Test 3: Unknown tool on working server")
    unknown_result = await working_client.call_tool("unknown_tool", {"param": "value"})
    if "error" in unknown_result:
        print(f"✅ Expected error: {unknown_result['error']}")
    else:
        print(f"❌ Unexpected success: {unknown_result}")

    print()

    # Test 4: Tool not found on any server
    print("🚫 Test 4: Tool not configured on any server")
    not_found_result = await working_client.call_tool("completely_missing_tool", {"param": "value"})
    if "error" in not_found_result and "not found on any server" in not_found_result["error"]:
        print(f"✅ Expected error: {not_found_result['error']}")
    else:
        print(f"❌ Unexpected result: {not_found_result}")

    print()
    print("🎯 Integration test complete!")


if __name__ == "__main__":
    asyncio.run(main())
