# MCP Bridge

HTTP proxy for stdio MCP servers. Allows the reasoning API (running in Docker) to access local MCP servers (filesystem, mcp-this, etc.) without Docker volume complexity.

## Architecture

```
API (Docker) --HTTP--> MCP Bridge (localhost:9000) --stdio--> MCP Servers (processes)
                                                              ├─ filesystem
                                                              ├─ mcp-this
                                                              ├─ github
                                                              └─ custom servers
```

## Quick Start

1. **Configure servers** in `config.json`
2. **Start the bridge**:
   ```bash
   uv run python mcp_bridge/server.py
   ```
3. **Access tools** at `http://localhost:9000/mcp/`

## Configuration

Edit `mcp_bridge/config.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem", "/Users/you/repos"],
      "transport": "stdio",
      "enabled": true
    },
    "your-custom-server": {
      "command": "uv",
      "args": ["run", "python", "path/to/server.py"],
      "transport": "stdio",
      "enabled": true
    }
  }
}
```

**Note**: Only servers with `"enabled": true` will be started.

## Usage

**Start with default config**:
```bash
uv run python mcp_bridge/server.py
```

**Start with custom config**:
```bash
uv run python mcp_bridge/server.py --config path/to/config.json
```

**Start on different port**:
```bash
uv run python mcp_bridge/server.py --port 8888
```

**Environment variables**:
```bash
export MCP_BRIDGE_PORT=9000
export MCP_BRIDGE_HOST=0.0.0.0
uv run python mcp_bridge/server.py
```

## Tool Access

Tools are automatically prefixed with server names:

- `filesystem_read_file` - from filesystem server
- `filesystem_write_file` - from filesystem server
- `thinking_analyze` - from thinking server
- `myserver_custom_tool` - from your custom server

## Testing

See `tests/test_mcp_bridge.py` for examples.

## Troubleshooting

**Bridge won't start:**
- Check config file exists and is valid JSON
- Verify all `command` executables are available (npx, uvx, etc.)

**Server not responding:**
- Check server logs for errors
- Verify server config (command, args, working directory)
- Test server manually: `npx @modelcontextprotocol/server-filesystem /path`

**Tools not appearing:**
- Verify `"enabled": true` in config
- Check bridge startup logs for server initialization
- List tools via HTTP: `curl http://localhost:9000/mcp/`
