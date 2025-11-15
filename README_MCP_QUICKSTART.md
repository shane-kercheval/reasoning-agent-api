# MCP Bridge Quick Start Guide

Complete guide to setting up the MCP Bridge to connect stdio MCP servers to your Dockerized reasoning API.

## Understanding MCP Configuration

The API uses **two separate configuration files**:

1. **`config/mcp_servers.json`** - Used by the **API server** (Docker)
   - Tells the API which HTTP MCP endpoints to connect to
   - You must create this file and enable `local_bridge` to use the bridge
   - Example: `{"mcpServers": {"local_bridge": {"url": "http://host.docker.internal:9000/mcp/", "enabled": true}}}`

2. **`config/mcp_bridge_config.json`** - Used by the **MCP Bridge** (your host machine)
   - Tells the bridge which stdio MCP servers to run
   - Example: filesystem, mcp-this, github servers
   - You configure which servers to enable and their paths here

## What is the MCP Bridge?

The MCP Bridge solves a key problem: many MCP servers (filesystem, mcp-this, github) only support stdio transport and need access to local files (e.g., `/Users/you/repos/`). Running them inside Docker requires complex volume mounts.

**Solution**: The bridge runs stdio servers **on your host machine** (with native file access) and exposes them via **HTTP** so the API (running in Docker) can access them easily.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Your Host Machine                     │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │ API Server (Docker)                              │   │
│  │ Uses: config/mcp_servers.json                    │   │
│  │ Looks for: local_bridge at host.docker.internal │   │
│  └───────────────────┬──────────────────────────────┘   │
│                      │ HTTP                              │
│                      ▼                                   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ MCP Bridge (localhost:9000)                      │   │
│  │ Uses: config/mcp_bridge_config.json              │   │
│  │ Start: make mcp_bridge                           │   │
│  │                                                   │   │
│  │ Runs stdio servers:                              │   │
│  │ ├─ filesystem (access local files)               │   │
│  │ ├─ mcp-this/github (git tools)                   │   │
│  │ └─ brave-search (web search)                     │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
                      ▲
                      │ HTTP
                      │
                 User Requests
```

## Step 1: Prerequisites

Make sure you have:
- `uv` installed (for `uvx`)
- `node`/`npm` installed (for `npx`)

That's it - the bridge will handle downloading MCP servers on first run.

## Step 2: Configure Bridge Servers (mcp_bridge_config.json)

This configures which stdio servers the **bridge** will run.

Copy the example configuration and customize it:

```bash
cp config/mcp_bridge_config.example.json config/mcp_bridge_config.json
```

Then edit `config/mcp_bridge_config.json` to specify which stdio MCP servers to run and their settings.

Example configuration:

```json
{
  "mcpServers": {
    "github-custom": {
      "command": "uvx",
      "args": ["mcp-this", "--preset", "github"],
      "transport": "stdio",
      "enabled": true,
      "_comment": "GitHub PR info, local git changes, directory tree tools"
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/you/repos",
        "/Users/you/Downloads"
      ],
      "transport": "stdio",
      "enabled": true,
      "_comment": "Update paths to directories you want to access"
    }
  }
}
```

**Important**: Update file paths to match your system. See the example file for more server options.

## Step 3: Start the Bridge

The bridge runs **on your host machine** (outside Docker):

```bash
# Start bridge on default port 9000 (recommended)
make mcp_bridge

# Or use python directly with custom port
uv run python mcp_bridge/server.py --port 8888
```

You should see:
```
2025-11-09 14:57:16 - INFO - Loading configuration from: mcp_bridge/config.json
2025-11-09 14:57:16 - INFO - Found 5 enabled servers
2025-11-09 14:57:16 - INFO - Connecting to server: github-custom
2025-11-09 14:57:16 - INFO -   Found 2 tools from github-custom
2025-11-09 14:57:17 - INFO - Connecting to server: filesystem
2025-11-09 14:57:17 - INFO -   Found 8 tools from filesystem
2025-11-09 14:57:17 - INFO - Bridge server created successfully
2025-11-09 14:57:17 - INFO - Starting bridge server on 0.0.0.0:9000
INFO:     Uvicorn running on http://0.0.0.0:9000 (Press CTRL+C to quit)
```

**Keep this terminal open** - the bridge needs to stay running.

## Step 4: Configure API to Connect to Bridge (mcp_servers.json)

This tells the **API** where to find MCP servers via HTTP.

Create the config file if it doesn't exist:

```bash
cp config/mcp_servers.example.json config/mcp_servers.json
```

Then edit `config/mcp_servers.json` and enable the local_bridge:

```json
{
  "mcpServers": {
    "local_bridge": {
      "url": "${MCP_BRIDGE_URL:-http://host.docker.internal:9000/mcp/}",
      "transport": "http",
      "enabled": true,
      "description": "Local MCP bridge for stdio servers"
    }
  }
}
```

**For Linux users**: Docker doesn't support `host.docker.internal` by default. Use the bridge network IP instead:

```bash
# Add to .env file
MCP_BRIDGE_URL=http://172.17.0.1:9000/mcp/
```

## Step 5: Start the API

```bash
# Start API with Docker Compose
docker compose up -d

# Or rebuild if needed
docker compose up --build -d
```

The API container will automatically connect to the bridge via `host.docker.internal:9000`.

## Step 6: Verify Everything Works

**Check tools are loaded:**
```bash
curl http://localhost:8000/tools | jq
```

You should see tools like:
- `github_custom_get_github_pull_request_info`
- `github_custom_get_local_git_changes_info`
- `github_custom_get_directory_tree`
- `meta_*` (your meta-prompts)
- `thinking_*` (your thinking prompts)
- `dev_*` (your development prompts)
- `filesystem_read_file`
- `filesystem_write_file`
- `filesystem_list_directory`

**Test with reasoning agent:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Routing-Mode: reasoning" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {
        "role": "user",
        "content": "List the files in /Users/shanekercheval/repos/reasoning-agent-api"
      }
    ]
  }'
```

## Troubleshooting

### Bridge won't start
- Check config file: `cat config/mcp_bridge_config.json | jq`
- Verify commands exist: `which uvx`, `which npx`
- Check port availability: `lsof -i :9000`

### API can't connect to bridge
- **macOS/Windows**: Ensure using `host.docker.internal:9000`
- **Linux**: Use `172.17.0.1:9000` or add `--add-host=host.docker.internal:host-gateway` to docker run
- Check bridge is running: `curl http://localhost:9000/mcp/` (from host)
- Check API logs: `docker compose logs reasoning-api | grep -i bridge`

### Tools not appearing
- Verify `"enabled": true` in `config/mcp_bridge_config.json`
- Check bridge startup logs for errors
- Ensure file paths in config are absolute and exist

### Testing bridge directly

Use MCP Inspector to test the bridge:

```bash
# Start inspector
npx @modelcontextprotocol/inspector

# Connect to: http://localhost:9000/mcp/
# Transport: Streamable HTTP
# Click "Connect", then "List Tools"
```

## Environment Variables

Optional environment variables for customization:

```bash
# Bridge URL (overrides default in config/mcp_servers.json)
MCP_BRIDGE_URL=http://host.docker.internal:9000/mcp/

# Bridge port (when starting bridge)
MCP_BRIDGE_PORT=9000
MCP_BRIDGE_HOST=0.0.0.0
```

## Production Deployment

For production, run the bridge as a systemd service or Docker container on the same host as your files.

**Example systemd service** (`/etc/systemd/system/mcp-bridge.service`):

```ini
[Unit]
Description=MCP Bridge Server
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/reasoning-agent-api
ExecStart=/usr/local/bin/uv run python mcp_bridge/server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable mcp-bridge
sudo systemctl start mcp-bridge
```

## Adding New Servers

To add a new stdio MCP server:

1. Add entry to `config/mcp_bridge_config.json`:
```json
{
  "mcpServers": {
    "your-server": {
      "command": "uvx",
      "args": ["your-mcp-server", "--arg"],
      "transport": "stdio",
      "enabled": true
    }
  }
}
```

2. Restart the bridge:
```bash
# Stop bridge (Ctrl+C in terminal where make mcp_bridge is running)
# Start again
make mcp_bridge
```

3. Verify tools appear:
```bash
curl http://localhost:8000/tools | jq
```

Tools will be automatically prefixed with server name (e.g., `your_server_tool_name`).

## See Also

- Bridge implementation: `mcp_bridge/server.py`
- Bridge README: `mcp_bridge/README.md`
- API MCP integration: `api/mcp.py`
- Main README: `README.md`
