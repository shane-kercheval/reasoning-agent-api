"""
Integration tests for MCP Bridge HTTP endpoint.

These tests verify that the bridge properly exposes stdio MCP servers via HTTP.
The bridge is automatically started/stopped for each test session.
"""

import pytest
import subprocess
import time
from pathlib import Path
from fastmcp import Client
import socket


@pytest.fixture(scope="module")
def bridge_process() -> subprocess.Popen: # type: ignore
    """Start bridge server for testing, stop it after tests complete."""
    config_path = Path("mcp_bridge/config.test.json")
    port = 9999  # Use non-standard port to avoid conflicts with user's bridge

    # Pre-check: Skip if port is already in use (user might have bridge running)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        if result == 0:
            pytest.skip(
                f"Port {port} is already in use. "
                "Skipping HTTP integration tests to avoid conflicts. "
                "Stop the service using this port to run these tests.",
            )
    except Exception:
        pass

    # Start bridge in subprocess
    process = subprocess.Popen(
        ["uv", "run", "python", "mcp_bridge/server.py",
         "--config", str(config_path), "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for port to be listening (max 10 seconds)
    # Bridge needs time to initialize stdio servers before starting HTTP
    start_time = time.time()
    while time.time() - start_time < 10:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            if result == 0:  # Port is open
                time.sleep(0.5)  # Give it a bit more time to fully initialize
                break
        except Exception:
            pass
        time.sleep(0.2)
    else:
        stdout, stderr = process.communicate(timeout=1)
        process.kill()
        pytest.fail(
            f"Bridge failed to start within 10 seconds\n"
            f"STDOUT: {stdout[:500]}\n"
            f"STDERR: {stderr[:500]}",
        )

    yield process

    # Cleanup
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


@pytest.fixture
def bridge_url(bridge_process: subprocess.Popen) -> str:  # noqa: ARG001
    """Bridge HTTP URL for testing."""
    return "http://localhost:9999/mcp/"


@pytest.mark.integration
@pytest.mark.asyncio
class TestBridgeHTTP:
    """Integration tests for MCP bridge HTTP endpoint."""

    async def test_bridge_http_connection(self, bridge_url: str) -> None:
        """Test that we can connect to the bridge via HTTP."""
        config = {
            "mcpServers": {
                "bridge": {
                    "url": bridge_url,
                    "transport": "http",
                },
            },
        }

        client = Client(config)
        async with client:
            # If we get here without exception, connection succeeded
            assert True

    async def test_bridge_lists_tools_via_http(self, bridge_url: str) -> None:
        """Test that bridge exposes tools from stdio servers via HTTP."""
        config = {
            "mcpServers": {
                "bridge": {
                    "url": bridge_url,
                    "transport": "http",
                },
            },
        }

        async with Client(config) as client:
            tools = await client.list_tools()

            # Should have 4 tools total (2 from echo, 2 from math)
            assert len(tools) == 4

            tool_names = [tool.name for tool in tools]

            # Check for echo server tools (with prefix)
            assert "echo_echo" in tool_names
            assert "echo_greet" in tool_names

            # Check for math server tools (with prefix)
            assert "math_add" in tool_names
            assert "math_multiply" in tool_names

    async def test_bridge_executes_echo_tool_via_http(self, bridge_url: str) -> None:
        """Test executing echo tool through HTTP bridge."""
        config = {
            "mcpServers": {
                "bridge": {
                    "url": bridge_url,
                    "transport": "http",
                },
            },
        }

        async with Client(config) as client:
            # Call echo tool
            result = await client.call_tool("echo_echo", {"message": "test message"})

            # Verify result
            assert result is not None
            assert hasattr(result, 'content')
            result_text = result.content[0].text
            assert "test message" in result_text

    async def test_bridge_executes_greet_tool_via_http(self, bridge_url: str) -> None:
        """Test executing greet tool through HTTP bridge."""
        config = {
            "mcpServers": {
                "bridge": {
                    "url": bridge_url,
                    "transport": "http",
                },
            },
        }

        async with Client(config) as client:
            # Call greet tool
            result = await client.call_tool("echo_greet", {"name": "Alice"})

            # Verify result
            assert result is not None
            result_text = result.content[0].text
            assert "Hello, Alice!" in result_text

    async def test_bridge_executes_math_add_via_http(self, bridge_url: str) -> None:
        """Test executing math add tool through HTTP bridge."""
        config = {
            "mcpServers": {
                "bridge": {
                    "url": bridge_url,
                    "transport": "http",
                },
            },
        }

        async with Client(config) as client:
            # Call add tool
            result = await client.call_tool("math_add", {"a": 10.0, "b": 20.0})

            # Verify result
            assert result is not None
            result_text = result.content[0].text
            assert "30" in result_text or "30.0" in result_text

    async def test_bridge_executes_math_multiply_via_http(self, bridge_url: str) -> None:
        """Test executing math multiply tool through HTTP bridge."""
        config = {
            "mcpServers": {
                "bridge": {
                    "url": bridge_url,
                    "transport": "http",
                },
            },
        }

        async with Client(config) as client:
            # Call multiply tool
            result = await client.call_tool("math_multiply", {"a": 5.0, "b": 6.0})

            # Verify result
            assert result is not None
            result_text = result.content[0].text
            assert "30" in result_text or "30.0" in result_text

    async def test_bridge_handles_multiple_sequential_calls(self, bridge_url: str) -> None:
        """Test multiple sequential tool calls through HTTP bridge."""
        config = {
            "mcpServers": {
                "bridge": {
                    "url": bridge_url,
                    "transport": "http",
                },
            },
        }

        async with Client(config) as client:
            # Make multiple sequential calls
            result1 = await client.call_tool("echo_echo", {"message": "First"})
            result2 = await client.call_tool("echo_echo", {"message": "Second"})
            result3 = await client.call_tool("math_add", {"a": 1.0, "b": 2.0})

            # Verify all results
            assert "First" in result1.content[0].text
            assert "Second" in result2.content[0].text
            assert "3" in result3.content[0].text
