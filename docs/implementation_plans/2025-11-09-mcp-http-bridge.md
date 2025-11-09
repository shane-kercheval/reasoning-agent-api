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

### M3: Update Reasoning API Configuration

**Goal**: Configure reasoning API to connect to MCP bridge

**Key Changes**:
- Update `config/mcp_servers.json` to point to bridge
- Handle Docker → host networking (`host.docker.internal`)
- Environment variable for bridge URL

**Configuration**:
```json
{
  "mcpServers": {
    "local_bridge": {
      "url": "http://host.docker.internal:9000/mcp/",
      "transport": "http",
      "enabled": true,
      "description": "Local MCP bridge for stdio servers"
    }
  }
}
```

**Testing Strategy**:
- Start bridge locally
- Start API in Docker
- Verify API can discover tools from bridge
- Test tool execution end-to-end

**Success Criteria**:
- API connects to bridge successfully
- Tools from stdio servers appear in `/tools` endpoint
- Reasoning agent can call tools via bridge
- Tool results flow back correctly

---

### M4: Integration & Documentation

**Goal**: Full integration testing and documentation updates

**Integration Tests**:
- API + bridge + real MCP servers (filesystem, mcp-this)
- Test file operations through reasoning agent
- Test multiple tools in single reasoning session
- Verify error handling (bridge down, tool failure, etc.)

**Documentation Updates**:
- README: Add MCP bridge setup instructions
- CLAUDE.md: Document MCP architecture
- Development guide: How to add new MCP servers
- Troubleshooting: Common issues

**Success Criteria**:
- Reasoning agent successfully uses local filesystem tools
- Multiple MCP servers work simultaneously
- Documentation clear for new developers
- Setup process documented in README

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
