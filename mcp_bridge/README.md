# MCP Bridge

HTTP proxy for stdio MCP servers. Allows the reasoning API (running in Docker) to access local MCP servers (filesystem, mcp-this, etc.) without Docker volume complexity.

## Understanding the Configuration Files

The MCP system uses **two separate configuration files**:

1. **`config/mcp_servers.json`** - Used by the **API server** (Docker)
   - Tells the API where to find MCP servers via HTTP
   - To use the bridge, you must enable `local_bridge` here
   - Example: `{"mcpServers": {"local_bridge": {"url": "http://host.docker.internal:9000/mcp/", "enabled": true}}}`

2. **`config/mcp_bridge_config.json`** - Used by **this bridge** (host machine)
   - Configures which stdio MCP servers this bridge will run
   - You edit this file to add/remove servers like filesystem, mcp-this, etc.
   - Only needed if you're running the bridge

**In other words**: The API uses `mcp_servers.json` to find the bridge, and the bridge uses `mcp_bridge_config.json` to know which stdio servers to run.

## Architecture

```
API (Docker)              MCP Bridge (localhost:9000)       MCP Servers (stdio processes)
Uses:                     Uses:
mcp_servers.json ─HTTP──> mcp_bridge_config.json ─stdio──> ├─ filesystem
(enable local_bridge)     (configure servers)               ├─ mcp-this
                                                            ├─ github
                                                            └─ custom servers
```

## Quick Start

1. **Configure bridge servers** (which stdio servers to run):
   ```bash
   cp config/mcp_bridge_config.example.json config/mcp_bridge_config.json
   # Edit config/mcp_bridge_config.json - set "enabled": true for servers you want
   ```

2. **Configure API connection** (tell API about the bridge):
   ```bash
   cp config/mcp_servers.example.json config/mcp_servers.json
   # Edit config/mcp_servers.json - set "enabled": true for "local_bridge"
   ```

3. **Start the bridge**:
   ```bash
   make mcp_bridge
   # Keep this running
   ```

4. **Start the API** (in another terminal):
   ```bash
   docker compose up -d
   ```

5. **Verify tools loaded**:
   ```bash
   curl http://localhost:8000/tools | jq
   ```

**See [README_MCP_QUICKSTART.md](../README_MCP_QUICKSTART.md) for complete setup guide.**

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

**Start with default config** (recommended):
```bash
make mcp_bridge
```

**Start with custom config**:
```bash
uv run python mcp_bridge/server.py --config path/to/config.json
```

**Alternative - direct python**:
```bash
uv run python mcp_bridge/server.py
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
