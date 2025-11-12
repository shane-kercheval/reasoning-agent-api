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

1. **Configure servers**:
   ```bash
   cp config/mcp_bridge_config.example.json config/mcp_bridge_config.json
   # Edit config/mcp_bridge_config.json with your paths
   ```
2. **Start the bridge**:
   ```bash
   uv run python mcp_bridge/server.py
   ```
3. **Access tools** at `http://localhost:9000/mcp/`

## Configuration

Copy the example configuration and customize it:

```bash
cp config/mcp_bridge_config.example.json config/mcp_bridge_config.json
```

Edit `config/mcp_bridge_config.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/your/repos"],
      "transport": "stdio",
      "enabled": true
    },
    "github-custom": {
      "command": "uvx",
      "args": ["mcp-this", "--preset", "github"],
      "transport": "stdio",
      "enabled": true
    }
  }
}
```

**Notes**:
- Only servers with `"enabled": true` will be started
- The config file is gitignored (user-specific paths)
- Update paths in the config to match your system

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

## Testing with MCP Inspector

The MCP Inspector is the proper way to test and debug MCP servers:

1. **Start the bridge with test config** (includes echo & math tools):
   ```bash
   uv run python mcp_bridge/server.py --config mcp_bridge/config.test.json
   ```

   Or enable servers in `config/mcp_bridge_config.json` first (set `"enabled": true`), then:
   ```bash
   uv run python mcp_bridge/server.py
   ```

2. **Launch MCP Inspector**:
   ```bash
   npx @modelcontextprotocol/inspector
   ```

3. **Connect to bridge**:
   - Set `Transport Type` to `Streamable HTTP`
   - Enter `http://localhost:9000/mcp/` in the URL field
   - Click `Connect`

4. **Test tools**:
   - Go to `Tools` tab
   - Click `List Tools` to see all available tools from stdio servers
   - Select a tool and test it with different parameters

**Note**: Do not use `curl` to test the bridge - MCP uses JSON-RPC 2.0 protocol, not standard REST endpoints.

## Troubleshooting

**Bridge won't start:**
- Check config file exists and is valid JSON
- Verify all `command` executables are available (npx, uvx, etc.)
- Check port is not in use: `lsof -i :9000`

**Server not responding:**
- Check server logs for errors
- Verify server config (command, args, working directory)
- Test server manually: `npx @modelcontextprotocol/server-filesystem /path`

**Tools not appearing:**
- Verify `"enabled": true` in `config/mcp_bridge_config.json`
- Check bridge startup logs for server initialization
- Use MCP Inspector to list tools (see "Testing with MCP Inspector" above)

**Inspector won't connect:**
- Ensure bridge is running and shows "Uvicorn running on http://0.0.0.0:9000"
- Check you're using `Streamable HTTP` transport type
- Verify URL includes `/mcp/` at the end: `http://localhost:9000/mcp/`
- Try restarting both bridge and Inspector
