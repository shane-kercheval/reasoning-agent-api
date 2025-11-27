"""
Integration tests for MCP server via MCP Python SDK client.

These tests verify the full MCP protocol including:
- MCP initialization handshake
- Tool discovery and execution via MCP protocol
- Prompt discovery and rendering via MCP protocol
- Error handling via JSON-RPC

Uses ClientSession which handles MCP initialization automatically.

NOTE: These tests are skipped by default due to anyio/pytest-asyncio incompatibility
in the MCP SDK client. The tests pass but have teardown errors. Run manually with:
    pytest tools_api/tests/integration_tests/test_mcp_integration.py -v -p no:skip

Or test manually with MCP Inspector:
    npx @modelcontextprotocol/inspector http://localhost:8001/mcp
"""

import json
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Skip all tests in this module - MCP client SDK has anyio/pytest-asyncio conflicts
pytestmark = pytest.mark.skip(
    reason="MCP SDK client has anyio/pytest-asyncio teardown conflicts. "
    "Run manually with: pytest path/to/test_mcp_integration.py -v -p no:skip",
)


def is_port_open(host: str, port: int) -> bool:
    """Check if a port is open."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


# Test port - use a dedicated port to avoid conflicts
TEST_PORT = 8099


@pytest.fixture(scope="module")
def tools_api_server() -> str:
    """
    Start tools_api server for integration tests.

    Module-scoped to start once per test module.
    Uses a dedicated test port to avoid conflicts.
    """
    if is_port_open("localhost", TEST_PORT):
        # Server already running (e.g., started manually)
        yield f"http://localhost:{TEST_PORT}/mcp"
        return

    # Start server as subprocess - cwd must be tools_api directory for imports to work
    tools_api_dir = Path(__file__).parent.parent.parent
    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "uvicorn",
            "tools_api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(TEST_PORT),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tools_api_dir,
    )

    # Wait for server to be ready
    for _ in range(30):  # 30 second timeout
        if is_port_open("localhost", TEST_PORT):
            break
        time.sleep(1)
    else:
        proc.kill()
        stdout, stderr = proc.communicate()
        raise RuntimeError(
            f"Server failed to start.\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}",
        )

    yield f"http://localhost:{TEST_PORT}/mcp"

    # Cleanup
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest_asyncio.fixture(loop_scope="function")
async def mcp_session(tools_api_server: str) -> ClientSession:
    """
    Create MCP ClientSession connected to tools_api.

    ClientSession handles the MCP initialization handshake automatically:
    1. Sends initialize request
    2. Receives server capabilities
    3. Sends initialized notification
    4. Ready for tools/prompts calls
    """
    async with (
        streamablehttp_client(tools_api_server) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


class TestMCPToolDiscovery:
    """Test tool discovery via MCP protocol."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_registered_tools(
        self, mcp_session: ClientSession,
    ) -> None:
        """Verify tools/list returns registered tools."""
        result = await mcp_session.list_tools()

        assert len(result.tools) > 0

        # Verify expected tools are present
        tool_names = {t.name for t in result.tools}
        assert "read_text_file" in tool_names
        assert "write_file" in tool_names
        assert "list_directory" in tool_names
        assert "list_allowed_directories" in tool_names

        # Verify tool structure
        for tool in result.tools:
            assert tool.name
            assert tool.description
            assert tool.inputSchema


class TestMCPToolExecution:
    """Test tool execution via MCP protocol."""

    @pytest.mark.asyncio
    async def test_call_list_allowed_directories(
        self, mcp_session: ClientSession,
    ) -> None:
        """Verify tools/call executes tool and returns result."""
        result = await mcp_session.call_tool(
            "list_allowed_directories",
            arguments={},
        )

        assert not result.isError
        assert len(result.content) > 0
        assert result.content[0].type == "text"

        # Parse the JSON result
        data = json.loads(result.content[0].text)
        assert "directories" in data

    @pytest.mark.asyncio
    async def test_call_tool_with_invalid_tool_returns_error(
        self, mcp_session: ClientSession,
    ) -> None:
        """Verify tools/call returns error for non-existent tool."""
        result = await mcp_session.call_tool(
            "nonexistent_tool",
            arguments={},
        )

        assert result.isError
        assert len(result.content) > 0
        # Error message should indicate tool not found
        error_text = result.content[0].text.lower()
        assert "not found" in error_text or "unknown" in error_text


class TestMCPPromptDiscovery:
    """Test prompt discovery via MCP protocol."""

    @pytest.mark.asyncio
    async def test_list_prompts_returns_registered_prompts(
        self, mcp_session: ClientSession,
    ) -> None:
        """Verify prompts/list returns registered prompts."""
        result = await mcp_session.list_prompts()

        assert len(result.prompts) >= 1

        # Verify prompt structure
        for prompt in result.prompts:
            assert prompt.name
            assert prompt.description


class TestMCPPromptRendering:
    """Test prompt rendering via MCP protocol."""

    @pytest.mark.asyncio
    async def test_get_prompt_renders_with_arguments(
        self, mcp_session: ClientSession,
    ) -> None:
        """Verify prompts/get renders prompt with arguments."""
        # Note: MCP protocol requires all prompt arguments to be strings
        # We only pass 'name' here since 'formal' is optional and the example
        # prompt expects it as a bool which MCP can't represent
        result = await mcp_session.get_prompt(
            "greeting",
            arguments={"name": "MCPUser"},
        )

        assert len(result.messages) > 0
        assert result.messages[0].role == "user"

        # Content should contain the rendered prompt with user's name
        content = result.messages[0].content
        if hasattr(content, "text"):
            assert "MCPUser" in content.text
        else:
            # Handle case where content is a string
            assert "MCPUser" in str(content)

    @pytest.mark.asyncio
    async def test_get_prompt_with_invalid_prompt_raises_error(
        self, mcp_session: ClientSession,
    ) -> None:
        """Verify prompts/get raises error for non-existent prompt."""
        with pytest.raises(Exception, match=r"(?i)not found"):
            await mcp_session.get_prompt(
                "nonexistent_prompt",
                arguments={},
            )


class TestMCPHealthEndpoint:
    """Test MCP health endpoint (accessible without MCP client)."""

    @pytest.mark.asyncio
    async def test_mcp_health_endpoint(self, tools_api_server: str) -> None:
        """Verify /mcp/health endpoint returns status info."""
        # Get base URL without /mcp path
        base_url = tools_api_server.replace("/mcp", "")

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/mcp/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["transport"] == "streamable-http"
        assert "tools_count" in data
        assert "prompts_count" in data
        assert data["tools_count"] > 0
