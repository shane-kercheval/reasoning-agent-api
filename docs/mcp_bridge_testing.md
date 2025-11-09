# MCP Bridge Testing Guide

This guide documents how to test the MCP bridge integration with the reasoning API.

## Manual Testing Steps

### Step 1: Start the MCP Bridge

In terminal 1, start the bridge with test configuration:

```bash
uv run python mcp_bridge/server.py --config mcp_bridge/config.test.json --port 9000
```

You should see:
```
Starting Bridge Server on 0.0.0.0:9000
Found 2 configured servers
  - echo
  - math
Bridge server created successfully
Bridge accessible at: http://0.0.0.0:9000/mcp/
```

### Step 2: Verify Bridge is Running

In another terminal, test the bridge directly:

```bash
# Check if server is up
curl http://localhost:9000/

# This should work once FastMCP HTTP endpoints are properly configured
```

### Step 3: Start the Reasoning API (pointing to bridge)

Update `config/mcp_servers.json` to enable the local_bridge:

```json
{
  "mcpServers": {
    "local_bridge": {
      "url": "http://localhost:9000/mcp/",
      "transport": "http",
      "enabled": true
    }
  }
}
```

Or use the test config:

```bash
MCP_CONFIG_PATH=config/mcp_servers.test.json uv run python -m api.main
```

### Step 4: Verify Tools are Loaded

Check the `/tools` endpoint:

```bash
curl http://localhost:8000/tools
```

You should see tools from the bridge with prefixes:
```json
{
  "tools": [
    "local_bridge_echo_echo",
    "local_bridge_echo_greet",
    "local_bridge_math_add",
    "local_bridge_math_multiply"
  ]
}
```

### Step 5: Test Tool Execution via Reasoning Agent

Send a request that uses a tool:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Routing-Mode: reasoning" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{
      "role": "user",
      "content": "Use the echo tool to say hello"
    }],
    "stream": false
  }'
```

The reasoning agent should:
1. Discover available tools from the bridge
2. Plan to use the echo tool
3. Execute the tool via HTTP to the bridge
4. The bridge forwards to the stdio server
5. Return the result

## Docker Testing

### For API in Docker + Bridge on Host

Update docker config to use host.docker.internal:

```bash
# Set environment variable for Docker
export MCP_BRIDGE_URL=http://host.docker.internal:9000/mcp/

# Or update docker-compose.yml:
environment:
  - MCP_BRIDGE_URL=http://host.docker.internal:9000/mcp/
  - MCP_CONFIG_PATH=config/mcp_servers.json
```

Make sure local_bridge is enabled in config, then:

```bash
# Terminal 1: Start bridge on host
uv run python mcp_bridge/server.py --config mcp_bridge/config.test.json

# Terminal 2: Start API in Docker
docker compose up reasoning-api
```

Test from host machine:
```bash
curl http://localhost:8000/tools
```

## Troubleshooting

### Bridge won't start
- Check that test servers exist: `ls tests/fixtures/mcp_servers/`
- Check Python/uv is available
- Check port 9000 is not in use: `lsof -i :9000`

### API can't connect to bridge
- Verify bridge is running: `curl http://localhost:9000/`
- Check `MCP_BRIDGE_URL` environment variable
- Check `local_bridge` is `enabled: true` in config
- For Docker: use `host.docker.internal` not `localhost`

### Tools not appearing
- Check `/tools` endpoint returns tools
- Verify bridge logs show servers starting
- Check MCP config has correct server definitions

### Tool execution fails
- Check bridge logs for errors
- Verify stdio servers are working
- Test tool directly via bridge
