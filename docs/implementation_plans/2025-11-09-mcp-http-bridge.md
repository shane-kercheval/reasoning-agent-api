# MCP HTTP Bridge Implementation Plan

**Created**: 2025-11-09
**Status**: Planning

## Overview

Build an HTTP bridge service that runs locally (not in Docker) to manage stdio-based MCP servers and expose their tools via HTTP. This allows the reasoning API (running in Docker) to access local MCP servers (filesystem, mcp-this, etc.) without filesystem complexity.

## Architecture

```
API (Docker) --HTTP--> MCP Bridge (localhost:9000) --stdio--> MCP Servers (local processes)
                                                               ├─ filesystem
                                                               ├─ mcp-this (thinking)
                                                               ├─ github
                                                               └─ custom servers
```

**Key Benefits**:
- API fully containerized (same dev/prod)
- Stdio MCP servers run natively with full local file access
- Single HTTP connection from API to bridge
- Bridge manages multiple stdio servers internally
- No Docker volume mount complexity

## Milestones

### M1: Research & Validation ✅

**Goal**: Verify FastMCP can be used as both client and server simultaneously

**Research Questions**:
1. Can FastMCP Client load multiple stdio servers from config?
2. Can we expose those tools via FastMCP HTTP server?
3. Does FastMCP merge tools from multiple servers automatically?
4. How does tool naming work with multiple servers (prefixing)?

**Findings**:

✅ **All research questions confirmed viable**

**FastMCP Proxy Support** ([docs](https://gofastmcp.com/servers/proxy)):
- `FastMCP.as_proxy()` creates proxy servers that bridge transports
- Supports config-based multi-server proxies with automatic tool merging
- Example: Stdio backend → HTTP frontend bridge

**Multi-Server Client** ([docs](https://gofastmcp.com/clients/client)):
- Client supports MCP configuration dictionaries with multiple servers
- Each server can use different transports (stdio, HTTP, SSE)
- Tools are automatically prefixed: `weather_get_forecast`, `filesystem_read_file`

**Tool Composition**:
- Multiple servers are merged into single tool namespace
- Automatic prefixing prevents name conflicts
- All tools accessible via single HTTP endpoint

**Performance Considerations**:
- `list_tools()` may take hundreds of milliseconds with remote backends
- Not an issue for our use case (local stdio servers)

**Recommended Implementation**:
```python
# Option A: Config-based proxy (recommended)
config = {
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": ["@modelcontextprotocol/server-filesystem", "/Users/shanekercheval/repos"],
            "transport": "stdio"
        },
        "thinking": {
            "command": "uvx",
            "args": ["mcp-this", "--config-path", "..."],
            "transport": "stdio"
        }
    }
}

bridge = FastMCP.as_proxy(config, name="MCP Bridge")
bridge.run(transport="http", host="0.0.0.0", port=9000)
```

**Success Criteria**: ✅ Met
- Approach is viable using FastMCP.as_proxy()
- No custom bridge code needed - use FastMCP features
- Can handle unlimited stdio servers via config

---

### M2: Build MCP Bridge Service ✅

**Goal**: Create standalone HTTP bridge service using FastMCP

**Key Components**:
- `mcp_bridge/server.py` - FastMCP proxy server (~30 lines)
- `mcp_bridge/config.json` - Configuration for stdio MCP servers
- Startup script/command

**Implementation**:
- Use `FastMCP.as_proxy(config)` with stdio server config
- Run as HTTP server on configurable port
- Handle server lifecycle (startup, shutdown, errors)
- Logging for debugging stdio server issues

**Package Management**:
- **Add dependencies**: `uv add <package>`
- **Run commands**: `uv run python mcp_bridge/server.py`
- **Run tests**: `uv run pytest tests/test_mcp_bridge.py`

**Configuration Format** (example):
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem", "/Users/shanekercheval/repos"],
      "transport": "stdio"
    },
    "thinking": {
      "command": "uvx",
      "args": ["mcp-this", "--config-path", "/Users/shanekercheval/repos/playbooks/src/thinking/config.yaml"],
      "transport": "stdio"
    }
  }
}
```

**Testing Strategy**:

1. **Unit Tests** (fast, no subprocesses):
   - Config loading and validation
   - Port/host configuration parsing
   - Error handling (invalid config, missing commands)

2. **Integration Tests** (real bridge, real stdio servers):
   - Create simple stdio test servers using FastMCP (~10 lines each)
   - Test bridge → stdio → tool execution flow
   - Test multiple servers simultaneously
   - Verify tool name prefixing (e.g., `server1_echo`, `server2_add`)
   - Test error handling (server crash, invalid tool call)

3. **Test Fixtures**:
   - `tests/fixtures/simple_stdio_server.py` - Minimal echo server
   - `tests/fixtures/math_stdio_server.py` - Simple math operations
   - Do NOT reuse `mcp_servers/fake_server.py` (it's HTTP, not stdio)

**Example Test Server**:
```python
# tests/fixtures/simple_stdio_server.py
from fastmcp import FastMCP

mcp = FastMCP("test-server")

@mcp.tool
async def echo(message: str) -> str:
    """Echo a message back."""
    return message

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

**Success Criteria**:
- Bridge starts and listens on configurable port
- Spawns stdio servers from config
- Exposes tools via HTTP endpoints with correct prefixing
- Returns tool results correctly
- Handles stdio server failures gracefully
- All tests passing (unit + integration)

---

### M3: Update Reasoning API Configuration ✅

**Goal**: Configure reasoning API to connect to MCP bridge

**Implemented**:

1. **Environment Variable Support** (`api/mcp.py`):
   - Added `_expand_env_vars()` function for ${VAR:-default} syntax
   - Updated `load_mcp_config()` to expand variables in config
   - Supports recursive expansion in dicts, lists, and strings
   - 9 unit tests passing (test_mcp_env_vars.py)

2. **Configuration Files**:
   - Updated `config/mcp_servers.json` with local_bridge entry
   - Created `tests/fixtures/mcp_servers.json` for testing
   - Created `mcp_bridge/config.test.json` for bridge testing
   - Bridge URL configurable via `MCP_BRIDGE_URL` environment variable

3. **Docker Networking Support**:
   - Default: `http://localhost:9000/mcp/` (local development)
   - Docker: Set `MCP_BRIDGE_URL=http://host.docker.internal:9000/mcp/`
   - Linux Docker: Use `http://172.17.0.1:9000/mcp/`

**Configuration Example**:
```json
{
  "mcpServers": {
    "local_bridge": {
      "url": "${MCP_BRIDGE_URL:-http://localhost:9000/mcp/}",
      "transport": "http",
      "enabled": true,
      "description": "Local MCP bridge for stdio servers"
    }
  }
}
```

**Testing Approach**:
- Manual testing documented in `docs/mcp_bridge_testing.md`
- Unit tests for env var expansion (9 tests passing)
- Integration testing via manual steps:
  1. Start bridge: `uv run python mcp_bridge/server.py --config mcp_bridge/config.test.json`
  2. Start API: `MCP_CONFIG_PATH=tests/fixtures/mcp_servers.json make api`
  3. Verify tools: `curl http://localhost:8000/tools`
  4. Test execution: Send chat completion request

**Success Criteria**: ✅ Met
- Environment variable expansion working
- Configuration files created with proper defaults
- Docker networking documented
- Manual testing guide provided

---

### M4: Integration & Documentation

**Goal**: Full integration testing and documentation updates

**Status**: Ready for testing with real MCP servers

**Testing Plan**:
1. **Bridge + API Integration**:
   - Follow `docs/mcp_bridge_testing.md` for manual testing
   - Test with echo/math test servers (working)
   - Test with real MCP servers (filesystem, mcp-this, github)

2. **Reasoning Agent Integration**:
   - Send requests with `X-Routing-Mode: reasoning`
   - Verify tools appear in reasoning events
   - Test tool execution through reasoning agent
   - Verify tool results flow correctly

3. **Multiple Server Testing**:
   - Add multiple stdio servers to bridge config
   - Verify tool name prefixing works
   - Test concurrent tool execution

**Documentation**:
- ✅ `mcp_bridge/README.md` - Bridge usage and configuration
- ✅ `docs/mcp_bridge_testing.md` - Manual testing guide
- ⏳ README: Add MCP bridge setup section
- ⏳ CLAUDE.md: Update MCP architecture documentation

**Next Steps**:
1. Test with real filesystem MCP server: `npx @modelcontextprotocol/server-filesystem`
2. Test with mcp-this server: `uvx mcp-this`
3. Update README with bridge setup instructions
4. Update CLAUDE.md with architecture changes

---

## Follow-Up Work (Not in Initial Plan)

**Production Remote Access** (Future):
- Tailscale setup for cloud API → local bridge
- Authentication/security for bridge HTTP endpoint
- Monitoring and health checks
- Rate limiting

**Additional MCP Servers** (As Needed):
- GitHub integration
- Database access
- Custom domain-specific tools

## Dependencies

- FastMCP library (already in project)
- Node.js/npx (for existing MCP servers)
- Docker networking to localhost

## Risk Factors

1. **FastMCP limitations**: May not support this bridge pattern (M1 research will validate)
2. **Tool naming conflicts**: Multiple servers with same tool names (FastMCP should handle via prefixing)
3. **Process management**: Bridge must handle stdio server crashes gracefully
4. **Docker networking**: `host.docker.internal` may vary by platform (Linux uses `host.docker.internal` or `172.17.0.1`)

## Open Questions

1. Should bridge be a standalone repo or part of reasoning-agent-api?
2. How should bridge handle stdio server version updates?
3. Should bridge support HTTP MCP servers too (as pass-through)?
4. What's the startup order (bridge before API)?
