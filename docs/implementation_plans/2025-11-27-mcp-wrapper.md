# MCP Wrapper Implementation Plan

## Overview

Add MCP (Model Context Protocol) support to `tools_api` by creating a thin wrapper layer that exposes existing tools and prompts via the MCP protocol. This enables MCP clients (Claude Desktop, MCP Inspector, other MCP-compatible applications) to connect to and use the tools_api service.

---

## Documentation References

### Official MCP Documentation

| Topic | URL | Use For |
|-------|-----|---------|
| **MCP Introduction** | https://modelcontextprotocol.io/docs/getting-started/intro | Understanding MCP concepts, use cases |
| **MCP Architecture** | https://modelcontextprotocol.io/docs/learn/architecture | Client-server model, transports, primitives (tools/resources/prompts) |
| **Building MCP Servers** | https://modelcontextprotocol.io/docs/develop/build-server | Server implementation patterns, transport setup |
| **Building MCP Clients** | https://modelcontextprotocol.io/docs/develop/build-client | How clients connect and use servers |
| **Remote Server Connections** | https://modelcontextprotocol.io/docs/develop/connect-remote-servers | HTTP transport, authentication |
| **Local Server Connections** | https://modelcontextprotocol.io/docs/develop/connect-local-servers | Stdio transport, Claude Desktop config |
| **MCP Inspector** | https://modelcontextprotocol.io/docs/tools/inspector | Testing and debugging MCP servers |

### Python SDK Documentation

| Topic | URL | Use For |
|-------|-----|---------|
| **Python SDK README** | https://github.com/modelcontextprotocol/python-sdk/blob/main/README.md | FastMCP usage, transport options, ASGI mounting, examples |
| **Python SDK Source** | https://github.com/modelcontextprotocol/python-sdk | Low-level API reference, type definitions |
| **PyPI Package** | https://pypi.org/project/mcp/ | Version info, installation |

### Key SDK Modules (for implementation reference)

```
mcp/
├── server/
│   ├── fastmcp.py          # High-level FastMCP class (decorator-based)
│   ├── lowlevel.py         # Low-level Server class (dynamic registration)
│   ├── sse.py              # SSE transport implementation
│   └── streamable_http.py  # Streamable HTTP transport (recommended)
├── types.py                # Tool, Prompt, Resource, TextContent types
└── client/                 # Client implementation (for testing)
```

---

## MCP Concepts Summary

### Architecture (from [MCP Architecture](https://modelcontextprotocol.io/docs/learn/architecture))

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  MCP Host   │────▶│  MCP Client │────▶│  MCP Server │
│ (Claude,    │     │ (connection │     │ (your app)  │
│  IDE, etc)  │     │  manager)   │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

- **Host**: AI application (Claude Desktop, IDE plugin)
- **Client**: Manages connections to servers
- **Server**: Exposes tools, resources, prompts

### Transport Options (from [Python SDK README](https://github.com/modelcontextprotocol/python-sdk))

| Transport | Use Case | Notes |
|-----------|----------|-------|
| **Streamable HTTP** | Remote servers, production | Recommended for HTTP. Stateless, scalable. |
| **SSE (Server-Sent Events)** | Remote servers | Legacy, being superseded by Streamable HTTP |
| **Stdio** | Local servers | Direct process communication |

### Primitives (from [MCP Architecture](https://modelcontextprotocol.io/docs/learn/architecture))

| Primitive | Description | Our Mapping |
|-----------|-------------|-------------|
| **Tools** | Executable functions LLMs can invoke | `BaseTool` → `mcp.types.Tool` |
| **Prompts** | Reusable interaction templates | `BasePrompt` → `mcp.types.Prompt` |
| **Resources** | Read-only data sources | Not implemented (future) |

---

## Current Architecture

The `tools_api` service has:

- **ToolRegistry**: Holds `BaseTool` instances with `name`, `description`, `parameters` (JSON Schema), and async `_execute()`
- **PromptRegistry**: Holds `BasePrompt` instances with `name`, `description`, `arguments`, and async `render()`
- **REST Endpoints**:
  - `GET /tools/` → list tools
  - `POST /tools/{tool_name}` → execute tool
  - `GET /prompts/` → list prompts
  - `POST /prompts/{prompt_name}` → render prompt

## Target Architecture

```
tools_api/
├── main.py              # Existing FastAPI app + MCP mount
├── mcp_server.py        # NEW: MCP server handlers
└── services/
    └── registry.py      # Existing registries (unchanged)
```

Endpoints after implementation:
- `/tools/*` - Existing REST API (unchanged)
- `/prompts/*` - Existing REST API (unchanged)
- `/mcp` - MCP Streamable HTTP endpoint for MCP clients

---

## Implementation Steps

### Step 1: Add MCP SDK Dependency

```bash
cd tools_api
uv add "mcp[cli]"
```

**Reference**: [Python SDK Installation](https://github.com/modelcontextprotocol/python-sdk#installation)

### Step 2: Create MCP Server Module

Create `tools_api/mcp_server.py`:

```python
"""MCP server wrapper for tools_api registries."""

import json
from mcp.server.lowlevel import Server
import mcp.types as types

from tools_api.services.registry import ToolRegistry, PromptRegistry

# Create MCP server instance
server = Server("tools-api")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """List all tools from ToolRegistry."""
    return [
        types.Tool(
            name=tool.name,
            description=tool.description,
            inputSchema=tool.parameters,
        )
        for tool in ToolRegistry.list()
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Execute tool from ToolRegistry."""
    tool = ToolRegistry.get(name)
    if not tool:
        raise ValueError(f"Tool '{name}' not found")

    result = await tool(**arguments)
    if result.success:
        # Serialize result to text content
        if isinstance(result.result, (dict, list)):
            text = json.dumps(result.result, indent=2, default=str)
        else:
            text = str(result.result)
        return [types.TextContent(type="text", text=text)]
    else:
        raise Exception(result.error or "Tool execution failed")


@server.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    """List all prompts from PromptRegistry."""
    return [
        types.Prompt(
            name=prompt.name,
            description=prompt.description,
            arguments=[
                types.PromptArgument(
                    name=arg["name"],
                    description=arg.get("description", ""),
                    required=arg.get("required", False),
                )
                for arg in prompt.arguments
            ],
        )
        for prompt in PromptRegistry.list()
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None) -> types.GetPromptResult:
    """Render prompt from PromptRegistry."""
    prompt = PromptRegistry.get(name)
    if not prompt:
        raise ValueError(f"Prompt '{name}' not found")

    result = await prompt(**(arguments or {}))
    if result.success:
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=result.content),
                )
            ]
        )
    else:
        raise Exception(result.error or "Prompt rendering failed")
```

**References**:
- Low-level Server API: [Python SDK - Low-Level Server](https://github.com/modelcontextprotocol/python-sdk#low-level-server)
- Type definitions: `mcp.types` module ([source](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/types.py))
- Tool schema format: [MCP Architecture - Tools](https://modelcontextprotocol.io/docs/learn/architecture#primitives)

### Step 3: Mount MCP Transport in FastAPI

Modify `tools_api/main.py` to mount the Streamable HTTP transport:

```python
# Add imports at top
from starlette.routing import Mount
from mcp.server.streamable_http import StreamableHTTPServerTransport

from tools_api.mcp_server import server as mcp_server

# After app creation and router includes, mount MCP transport:
mcp_transport = StreamableHTTPServerTransport(
    mcp_server,
    path="/",  # Relative to mount point
)
app.mount("/mcp", mcp_transport.app)
```

**Transport Choice**: We use **Streamable HTTP** (not SSE) because:
- SSE is deprecated and will be removed soon
- Claude Desktop/Code support Streamable HTTP via `--transport http`
- Simpler single-endpoint design (no separate `/sse` and `/messages` endpoints)

**References**:
- Streamable HTTP transport: [Python SDK - Running Your Server](https://github.com/modelcontextprotocol/python-sdk#running-your-server)
- Mounting in ASGI apps: [Python SDK - Mounting to an Existing ASGI Server](https://github.com/modelcontextprotocol/python-sdk#mounting-to-an-existing-asgi-server)
- Transport specification: [MCP Transports](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports)

### Step 4: Add Health Check for MCP

Add MCP-specific health indicator:

```python
@app.get("/mcp/health")
async def mcp_health() -> dict[str, str]:
    """MCP endpoint health check."""
    return {
        "status": "healthy",
        "transport": "streamable-http",
        "tools_count": str(len(ToolRegistry.list())),
        "prompts_count": str(len(PromptRegistry.list())),
    }
```

### Step 5: Write Tests

See [Testing Strategy](#testing-strategy) section below for comprehensive test plan.

### Step 6: Integration Test with MCP Inspector

After implementation, test using the MCP Inspector:

```bash
# Start tools_api
cd tools_api
uv run uvicorn tools_api.main:app --host 0.0.0.0 --port 8001

# In another terminal, run MCP Inspector
npx @modelcontextprotocol/inspector http://localhost:8001/mcp
```

Verify:
- [ ] Tools tab shows all registered tools
- [ ] Prompts tab shows all registered prompts
- [ ] Tool execution works with test inputs
- [ ] Prompt rendering works with test arguments

**References**:
- MCP Inspector usage: [MCP Inspector Documentation](https://modelcontextprotocol.io/docs/tools/inspector)
- Inspector GitHub: https://github.com/modelcontextprotocol/inspector

---

## Testing Strategy

### Overview

We use a **three-tier testing approach** matching our existing patterns in `tools_api/tests/`:

| Tier | Test Type | What It Tests | Containers? |
|------|-----------|---------------|-------------|
| 1 | Unit Tests | MCP handler functions in isolation | No |
| 2 | Integration Tests | Full MCP protocol via `ClientSession` | No |
| 3 | Manual/E2E | MCP Inspector, Claude Desktop | No |

**No testcontainers needed** - MCP is stateless HTTP and doesn't require databases. We follow the same pattern as `tools_api/tests/integration_tests/test_tools_api_http.py` which uses `ASGITransport` for in-process testing.

### Testing Documentation References

| Topic | URL | Use For |
|-------|-----|---------|
| **FastMCP Testing Guide** | https://gofastmcp.com/patterns/testing | In-memory client testing patterns |
| **MCP Unit Testing** | https://mcpcat.io/guides/writing-unit-tests-mcp-servers/ | Pytest fixtures, mocking |
| **MCP Integration Testing** | https://mcpcat.io/guides/integration-tests-mcp-flows/ | End-to-end MCP flows |
| **Python SDK Client** | https://github.com/modelcontextprotocol/python-sdk#client | `ClientSession`, `streamablehttp_client` |
| **MCP Inspector** | https://modelcontextprotocol.io/docs/tools/inspector | Manual testing tool |

### Tier 1: Unit Tests

Test MCP handler functions directly without transport layer.

Create `tools_api/tests/unit_tests/test_mcp_server.py`:

```python
"""Unit tests for MCP server handler functions."""

import json
import pytest
from typing import Any

import mcp.types as types

from tools_api.mcp_server import list_tools, call_tool, list_prompts, get_prompt
from tools_api.services.registry import ToolRegistry, PromptRegistry
from tools_api.services.base import BaseTool, BasePrompt
from tools_api.models import ToolResult, PromptResult


# =============================================================================
# Test Fixtures
# =============================================================================


class MockTool(BaseTool):
    """Mock tool for testing."""

    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "A mock tool for testing"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Test input"},
            },
            "required": ["input"],
        }

    async def _execute(self, input: str) -> dict[str, str]:
        return {"output": f"processed: {input}"}


class MockFailingTool(BaseTool):
    """Mock tool that always fails."""

    @property
    def name(self) -> str:
        return "failing_tool"

    @property
    def description(self) -> str:
        return "A tool that always fails"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def _execute(self) -> None:
        raise RuntimeError("Tool execution failed")


class MockPrompt(BasePrompt):
    """Mock prompt for testing."""

    @property
    def name(self) -> str:
        return "mock_prompt"

    @property
    def description(self) -> str:
        return "A mock prompt for testing"

    @property
    def arguments(self) -> list[dict[str, Any]]:
        return [
            {"name": "name", "required": True, "description": "User name"},
            {"name": "formal", "required": False, "description": "Use formal greeting"},
        ]

    async def render(self, name: str, formal: bool = False) -> str:
        if formal:
            return f"Good day, {name}."
        return f"Hello, {name}!"


@pytest.fixture(autouse=True)
def clear_registries():
    """Clear registries before and after each test."""
    ToolRegistry.clear()
    PromptRegistry.clear()
    yield
    ToolRegistry.clear()
    PromptRegistry.clear()


@pytest.fixture
def mock_tool() -> MockTool:
    return MockTool()


@pytest.fixture
def mock_failing_tool() -> MockFailingTool:
    return MockFailingTool()


@pytest.fixture
def mock_prompt() -> MockPrompt:
    return MockPrompt()


# =============================================================================
# list_tools Tests
# =============================================================================


class TestListTools:
    @pytest.mark.asyncio
    async def test_empty_registry_returns_empty_list(self) -> None:
        """list_tools returns empty list when no tools registered."""
        result = await list_tools()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_registered_tools(self, mock_tool: MockTool) -> None:
        """list_tools returns all registered tools as MCP Tool types."""
        ToolRegistry.register(mock_tool)

        result = await list_tools()

        assert len(result) == 1
        assert isinstance(result[0], types.Tool)
        assert result[0].name == "mock_tool"
        assert result[0].description == "A mock tool for testing"
        assert result[0].inputSchema == mock_tool.parameters

    @pytest.mark.asyncio
    async def test_returns_multiple_tools(
        self, mock_tool: MockTool, mock_failing_tool: MockFailingTool,
    ) -> None:
        """list_tools returns all registered tools."""
        ToolRegistry.register(mock_tool)
        ToolRegistry.register(mock_failing_tool)

        result = await list_tools()

        assert len(result) == 2
        tool_names = {t.name for t in result}
        assert tool_names == {"mock_tool", "failing_tool"}


# =============================================================================
# call_tool Tests
# =============================================================================


class TestCallTool:
    @pytest.mark.asyncio
    async def test_tool_not_found_raises_error(self) -> None:
        """call_tool raises ValueError for non-existent tool."""
        with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
            await call_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_successful_execution_returns_text_content(
        self, mock_tool: MockTool,
    ) -> None:
        """call_tool returns TextContent with JSON result."""
        ToolRegistry.register(mock_tool)

        result = await call_tool("mock_tool", {"input": "test"})

        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)
        assert result[0].type == "text"

        # Verify JSON content
        parsed = json.loads(result[0].text)
        assert parsed == {"output": "processed: test"}

    @pytest.mark.asyncio
    async def test_tool_failure_raises_exception(
        self, mock_failing_tool: MockFailingTool,
    ) -> None:
        """call_tool raises exception when tool execution fails."""
        ToolRegistry.register(mock_failing_tool)

        with pytest.raises(Exception, match="Tool execution failed"):
            await call_tool("failing_tool", {})


# =============================================================================
# list_prompts Tests
# =============================================================================


class TestListPrompts:
    @pytest.mark.asyncio
    async def test_empty_registry_returns_empty_list(self) -> None:
        """list_prompts returns empty list when no prompts registered."""
        result = await list_prompts()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_registered_prompts(self, mock_prompt: MockPrompt) -> None:
        """list_prompts returns all registered prompts as MCP Prompt types."""
        PromptRegistry.register(mock_prompt)

        result = await list_prompts()

        assert len(result) == 1
        assert isinstance(result[0], types.Prompt)
        assert result[0].name == "mock_prompt"
        assert result[0].description == "A mock prompt for testing"

        # Verify arguments mapping
        assert len(result[0].arguments) == 2
        arg_names = {a.name for a in result[0].arguments}
        assert arg_names == {"name", "formal"}


# =============================================================================
# get_prompt Tests
# =============================================================================


class TestGetPrompt:
    @pytest.mark.asyncio
    async def test_prompt_not_found_raises_error(self) -> None:
        """get_prompt raises ValueError for non-existent prompt."""
        with pytest.raises(ValueError, match="Prompt 'nonexistent' not found"):
            await get_prompt("nonexistent", {})

    @pytest.mark.asyncio
    async def test_successful_render_returns_prompt_result(
        self, mock_prompt: MockPrompt,
    ) -> None:
        """get_prompt returns GetPromptResult with rendered content."""
        PromptRegistry.register(mock_prompt)

        result = await get_prompt("mock_prompt", {"name": "World"})

        assert isinstance(result, types.GetPromptResult)
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert isinstance(result.messages[0].content, types.TextContent)
        assert result.messages[0].content.text == "Hello, World!"

    @pytest.mark.asyncio
    async def test_render_with_optional_args(self, mock_prompt: MockPrompt) -> None:
        """get_prompt handles optional arguments correctly."""
        PromptRegistry.register(mock_prompt)

        result = await get_prompt("mock_prompt", {"name": "World", "formal": True})

        assert result.messages[0].content.text == "Good day, World."
```

### Tier 2: Integration Tests

Test full MCP protocol using the SDK's `ClientSession` which handles the MCP initialization handshake automatically.

**MCP Initialization Flow**: The MCP protocol requires a handshake before tools/prompts can be used:
1. Client sends `initialize` request with capabilities
2. Server responds with its capabilities
3. Client sends `initialized` notification
4. Client can now call `tools/list`, `tools/call`, etc.

The `ClientSession` handles this automatically, so we use it for integration tests.

Create `tools_api/tests/integration_tests/test_mcp_integration.py`:

```python
"""
Integration tests for MCP server via MCP Python SDK client.

These tests verify the full MCP protocol including:
- MCP initialization handshake
- Tool discovery and execution via MCP protocol
- Prompt discovery and rendering via MCP protocol
- Error handling via JSON-RPC

Uses ClientSession which handles MCP initialization automatically.
Requires a running server (started via pytest fixture or manually).
"""

import json
import pytest
import pytest_asyncio
import subprocess
import time
import socket

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


def is_port_open(host: str, port: int) -> bool:
    """Check if a port is open."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


@pytest.fixture(scope="module")
def tools_api_server():
    """
    Start tools_api server for integration tests.

    Module-scoped to start once per test module.
    Uses a dedicated test port to avoid conflicts.
    """
    port = 8099  # Test port
    if is_port_open("localhost", port):
        # Server already running (e.g., started manually)
        yield f"http://localhost:{port}/mcp"
        return

    # Start server as subprocess
    proc = subprocess.Popen(
        [
            "uv", "run", "uvicorn", "tools_api.main:app",
            "--host", "127.0.0.1",
            "--port", str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    for _ in range(30):  # 30 second timeout
        if is_port_open("localhost", port):
            break
        time.sleep(1)
    else:
        proc.kill()
        raise RuntimeError("Server failed to start")

    yield f"http://localhost:{port}/mcp"

    # Cleanup
    proc.terminate()
    proc.wait(timeout=5)


@pytest_asyncio.fixture(loop_scope="function")
async def mcp_session(tools_api_server: str):
    """
    Create MCP ClientSession connected to tools_api.

    ClientSession handles the MCP initialization handshake automatically:
    1. Sends initialize request
    2. Receives server capabilities
    3. Sends initialized notification
    4. Ready for tools/prompts calls
    """
    async with streamablehttp_client(tools_api_server) as (read, write, _):
        async with ClientSession(read, write) as session:
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
        result = await mcp_session.get_prompt(
            "greeting",
            arguments={"name": "MCPUser", "formal": True},
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
    async def test_get_prompt_with_invalid_prompt_returns_error(
        self, mcp_session: ClientSession,
    ) -> None:
        """Verify prompts/get returns error for non-existent prompt."""
        with pytest.raises(Exception) as exc_info:
            await mcp_session.get_prompt(
                "nonexistent_prompt",
                arguments={},
            )

        assert "not found" in str(exc_info.value).lower()
```

### Tier 3: Manual Testing with MCP Inspector

The MCP Inspector provides interactive testing without writing code.

```bash
# Start tools_api
uv run uvicorn tools_api.main:app --host 0.0.0.0 --port 8001

# In another terminal
npx @modelcontextprotocol/inspector http://localhost:8001/mcp
```

**Inspector Test Checklist:**

- [ ] **Tools Tab**: All filesystem/github/web tools appear
- [ ] **Tools Tab**: Execute `list_allowed_directories` → returns paths
- [ ] **Tools Tab**: Execute `read_text_file` with valid path → returns content
- [ ] **Tools Tab**: Execute `read_text_file` with invalid path → returns error
- [ ] **Prompts Tab**: All prompts appear with arguments
- [ ] **Prompts Tab**: Render `greeting` prompt → returns formatted text
- [ ] **Notifications**: No errors in notification pane

### pytest Configuration

Add to `tools_api/pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
```

---

## Configuration Options

### Environment Variables

Add to `tools_api/config.py`:

```python
class Settings(BaseSettings):
    # ... existing settings ...

    # MCP settings
    mcp_enabled: bool = True
    mcp_path: str = "/mcp"
```

### Docker Compose

No changes needed - the MCP endpoint is exposed on the same port (8001) as the REST API.

#### Why Same Port Works

MCP over HTTP is just standard HTTP - the "Streamable HTTP" transport uses `POST` requests with JSON-RPC 2.0 payloads. When you mount the MCP transport at `/mcp`, FastAPI routes requests by **path**, not port:

```
Port 8001 (single FastAPI app)
├── /tools/*      → REST API router (existing)
├── /prompts/*    → REST API router (existing)
├── /health       → REST API endpoint (existing)
└── /mcp          → MCP transport (JSON-RPC over HTTP)
```

The MCP "server" isn't a separate process - it's a **Starlette sub-application** mounted at `/mcp`. When you call:

```python
app.mount("/mcp", mcp_transport.app)
```

FastAPI delegates any request to `/mcp/*` to the MCP transport handler. MCP clients send standard HTTP POST requests to `http://localhost:8001/mcp` with JSON-RPC payloads - no special protocol at the TCP level.

**Analogy**: This is similar to having both REST and GraphQL on the same server:
```
Port 3000
├── /api/users    → REST endpoint
└── /graphql      → GraphQL endpoint (different protocol, same port)
```

---

## Client Configuration

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "tools-api": {
      "url": "http://localhost:8001/mcp"
    }
  }
}
```

**Reference**: [Connecting Remote Servers](https://modelcontextprotocol.io/docs/develop/connect-remote-servers)

### Other MCP Clients

Connect to: `http://<host>:8001/mcp`

---

## Troubleshooting Guide

### Common Issues

| Issue | Cause | Solution | Reference |
|-------|-------|----------|-----------|
| MCP Inspector can't connect | Wrong transport or path | Verify URL matches mount path | [Inspector Docs](https://modelcontextprotocol.io/docs/tools/inspector) |
| Tools not showing | Registry not populated | Ensure lifespan runs before MCP mount | [Build Server](https://modelcontextprotocol.io/docs/develop/build-server) |
| JSON-RPC errors | Schema mismatch | Verify `inputSchema` matches MCP Tool schema | [MCP Architecture](https://modelcontextprotocol.io/docs/learn/architecture) |
| Initialization fails | Handshake not completed | Use `ClientSession` which handles init automatically | [Python SDK Client](https://github.com/modelcontextprotocol/python-sdk#client) |

### Debug Logging

Enable MCP debug logging:

```python
import logging
logging.getLogger("mcp").setLevel(logging.DEBUG)
```

### Claude Desktop Logs

Check Claude Desktop logs for connection issues:
- macOS: `~/Library/Logs/Claude/mcp.log`
- Windows: `%APPDATA%\Claude\logs\mcp.log`

**Reference**: [Build Server - Debugging](https://modelcontextprotocol.io/docs/develop/build-server#debugging)

---

## Rollout Plan

1. **Development**: Implement in feature branch, test with MCP Inspector
2. **Staging**: Deploy to staging, verify with Claude Desktop
3. **Production**: Deploy with `MCP_ENABLED=true` (default)

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| MCP SDK version incompatibility | Pin SDK version, test upgrades |
| Blocking REST endpoints | MCP is mounted separately, won't affect `/tools` |
| Memory leaks from connections | Use proper async context managers, add connection limits |

---

## Success Criteria

- [ ] MCP Inspector can list all tools and prompts
- [ ] Tool execution via MCP returns same results as REST API
- [ ] Prompt rendering via MCP returns same results as REST API
- [ ] Claude Desktop can connect and use tools
- [ ] No performance regression on existing REST endpoints
- [ ] Unit and integration tests pass

---

## Future Enhancements

1. **Resources**: Add MCP Resources for read-only data access (e.g., file contents, configs)
   - Reference: [Python SDK - Resources](https://github.com/modelcontextprotocol/python-sdk#resources)

2. **Authentication**: Implement OAuth 2.1 token verification for MCP connections
   - Reference: [Python SDK - Authentication](https://github.com/modelcontextprotocol/python-sdk#authentication)

3. **Streaming**: Add progress reporting for long-running tools via MCP context
   - Reference: [Python SDK - Context](https://github.com/modelcontextprotocol/python-sdk#context)

4. **Metrics**: Add observability for MCP connections and tool calls

---

## Appendix: Type Mappings

### Tool Mapping

| tools_api | MCP | Notes |
|-----------|-----|-------|
| `BaseTool.name` | `types.Tool.name` | Direct mapping |
| `BaseTool.description` | `types.Tool.description` | Direct mapping |
| `BaseTool.parameters` | `types.Tool.inputSchema` | JSON Schema format |
| `ToolResult.result` | `types.TextContent.text` | Serialize to JSON string |
| `ToolResult.error` | Exception | Raise instead of returning |

### Prompt Mapping

| tools_api | MCP | Notes |
|-----------|-----|-------|
| `BasePrompt.name` | `types.Prompt.name` | Direct mapping |
| `BasePrompt.description` | `types.Prompt.description` | Direct mapping |
| `BasePrompt.arguments` | `types.Prompt.arguments` | List of `PromptArgument` |
| `PromptResult.content` | `types.GetPromptResult.messages` | Wrap in `PromptMessage` |
