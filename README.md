[![Tests](https://github.com/shane-kercheval/reasoning-agent-api/actions/workflows/tests.yaml/badge.svg)](https://github.com/shane-kercheval/reasoning-agent-api/actions/workflows/tests.yaml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Reasoning Agent API

An OpenAI-compatible API that adds reasoning capabilities and tool usage through MCP (Model Context Protocol) servers. Includes a simple web interface for interactive conversations.

## Features

- **🔄 OpenAI Compatible**: Drop-in replacement for OpenAI's chat completion API
- **🧠 Intelligent Request Routing**: Three execution paths (passthrough, reasoning, orchestration) with auto-classification
- **🌉 LiteLLM Gateway**: Unified LLM proxy for centralized observability and connection pooling
- **🤖 Reasoning Agent**: Single-loop reasoning with visual thinking steps
- **🔧 MCP Tool Integration**: Extensible with Model Context Protocol tools
- **💾 Conversation Storage**: PostgreSQL-backed persistent conversation history (Milestone 1 complete, API integration coming in M2-M3)
- **🖥️ Desktop Client**: Native Electron app with React, TypeScript, and Tailwind CSS (Milestone 1 complete)
- **🎨 Web Interface**: MonsterUI-powered chat interface with reasoning visualization
- **📊 Real-time Streaming**: See reasoning and responses as they happen
- **⏹️ Request Cancellation**: Stop reasoning immediately when clients disconnect
- **🔒 Simple Authentication**: Token-based authentication with multiple token support
- **🐳 Docker Ready**: Full Docker Compose setup for easy deployment
- **📈 Phoenix Observability**: LLM tracing and monitoring with Phoenix Arize

## Quick Start

### Prerequisites

- **OpenAI API key** (required)
- **Docker & Docker Compose** (recommended) OR **Python 3.13+ & uv** (for local development)
- **Node.js 18+** (optional, for desktop client only)

### Option 1: Docker Compose (Recommended)

Get everything running in 5 minutes:

1. **Setup environment**
    - Run `cp .env.dev.example .env`
    - Edit `.env` and set:
      - `OPENAI_API_KEY=your-openai-key-here` (real OpenAI key)
      - `LITELLM_MASTER_KEY=` (generate with: `uv run python -c "import secrets; print('sk-' + secrets.token_urlsafe(32))"`)
      - `LITELLM_POSTGRES_PASSWORD=` (generate with: `uv run python -c "import secrets; print(secrets.token_urlsafe(16))"`)
      - `REASONING_POSTGRES_PASSWORD=` (generate with: `uv run python -c "import secrets; print(secrets.token_urlsafe(16))"`)

1b. **Setup MCP servers** (optional)
    - Run `cp config/mcp_servers.example.json config/mcp_servers.json`
    - Default config has all servers disabled
    - Enable `demo_tools` for fake weather/stocks tools (requires `docker compose --profile demo up`)
    - Enable `local_bridge` for filesystem access (see [README_MCP_QUICKSTART.md](README_MCP_QUICKSTART.md))

2. **Start all services**
    - Run `make docker_up` (or `docker compose up -d`)
    - Optional: Add demo MCP server with `docker compose --profile demo up -d`
    - Wait for services to be up

3. **Run database migrations** (for conversation storage)
    - Ensure `REASONING_DATABASE_URL` is set in your `.env` file (see step 1)
    - Run `uv run alembic upgrade head`
    - This creates the tables needed for persistent conversation history

4. **Setup LiteLLM virtual keys**
    - **What**: Virtual keys allow per-environment usage tracking in LiteLLM (dev/test/eval)
    - **Why**: The script creates these keys in LiteLLM's database via its API (saves manual UI setup)
    - Run `make litellm_setup`
    - Copy the generated keys to `.env`:
      - `LITELLM_API_KEY=sk-...` (development/production usage)
      - `LITELLM_TEST_KEY=sk-...` (integration tests - separate tracking)
      - `LITELLM_EVAL_KEY=sk-...` (LLM behavioral evaluations)
    - Run `docker compose restart reasoning-api` to apply new keys

5. **Access your services**
    - Web Interface: http://localhost:8080
    - Desktop Client: `cd client && npm install && npm run dev` (native Electron app)
    - API Documentation: http://localhost:8000/docs
    - LiteLLM Dashboard: http://localhost:4000
    - Phoenix UI: http://localhost:6006
    - MCP Tools: http://localhost:8001/mcp/

6. **Test MCP tools with Inspector** (optional)
    - Run `npx @modelcontextprotocol/inspector`
    - Set `Transport Type` to `Streamable HTTP`
    - Enter `http://localhost:8001/mcp/` and click `Connect`
    - Go to `Tools` and click `List Tools` to see available tools
    - For local stdio servers (filesystem, mcp-this), see [mcp_bridge/README.md](mcp_bridge/README.md)

### Option 2: Local Development

For development with individual service control, you'll need LiteLLM running (via Docker) and local services:

```bash
# 1. Setup environment (see .env.dev.example for details)
cp .env.dev.example .env
cp config/mcp_servers.example.json config/mcp_servers.json

# 2. Start required Docker services
docker compose up -d litellm postgres-litellm postgres-reasoning
make litellm_setup  # Setup virtual keys, copy generated keys to .env

# 3. Run database migrations (requires REASONING_DATABASE_URL in .env)
uv run alembic upgrade head  # Create conversation storage tables

# 4. Install dependencies
make dev

# 5. Start local services (separate terminals)
make demo_mcp_server  # Terminal 1: MCP tools
make api              # Terminal 2: API server (connects to LiteLLM in Docker)
make web_client       # Terminal 3: Web interface

# 6. Access at http://localhost:8080
```

## Request Routing

The API intelligently routes requests through three execution paths:

### **Route A: Passthrough** (Default)
- Direct OpenAI API call via LiteLLM proxy
- Lowest latency, no reasoning overhead
- **Use when**: Simple queries, structured outputs, or tool calls needed
- **Activate**: Default (no header), or `X-Routing-Mode: passthrough`

### **Route B: Reasoning Agent**
- Single-loop reasoning with visual thinking steps
- Shows AI's thought process before final answer
- **Use when**: Testing/comparing with orchestration, baseline measurements
- **Activate**: `X-Routing-Mode: reasoning`

### **Route C: Orchestration** (Coming Soon)
- Multi-agent coordination via A2A protocol
- Complex task decomposition and execution
- **Use when**: Research queries, multi-step tasks requiring planning
- **Activate**: `X-Routing-Mode: orchestration` (returns 501 until M3-M4)

### **Auto-Routing**
- LLM classifier chooses between passthrough and orchestration
- Uses GPT-4o-mini for classification (fast, deterministic)
- **Activate**: `X-Routing-Mode: auto`

**Note**: Requests with `response_format` or `tools` always use passthrough (Tier 1 rule).

## API Usage

### Request Routing Examples

```bash
# Default: Passthrough (fastest)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hello"}]}'

# Reasoning path (show thinking steps)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Routing-Mode: reasoning" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Explain quantum computing"}]}'

# Auto-routing (LLM decides)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Routing-Mode: auto" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Research climate change impacts"}]}'
```

### OpenAI SDK Integration

```python
from openai import AsyncOpenAI

# Point to your reasoning agent API
client = AsyncOpenAI(
    api_key="your-api-token",  # Token from API_TOKENS environment variable
    base_url="http://localhost:8000/v1",
)

# Use exactly like OpenAI's API
response = await client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather like?"}],
    stream=True,  # See reasoning steps in real-time
)
async for chunk in response:
    # reasoning_event is available in the stream
    if chunk.choices[0].delta.reasoning_event:
        print("Reasoning Event:")
        print(chunk.choices[0].delta.reasoning_event)
        print("---")
    # content is the final response
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

**Important Notes:**
- `api_key` should be a token from your API's `API_TOKENS` environment variable (not an OpenAI key)
- If `REQUIRE_AUTH=false` (development mode), any value works (e.g., `"dummy"`)
- The API handles LLM calls to OpenAI using its own LiteLLM virtual key
- Users consume your API as a service, not providing their own OpenAI credentials

### Direct API Usage

```bash
# Test with curl
# curl -X POST https://reasoning-agent-api.onrender.com/v1/chat/completions \
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer web-client-dev-token" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is the weather like?"}],
    "stream": true
  }'
```

### API Endpoints

- **Chat Completions**: `POST /v1/chat/completions` (OpenAI compatible)
- **Models**: `GET /v1/models` (List available models)
- **Tools**: `GET /tools` (List available MCP tools)
- **Health**: `GET /health` (Health check)
- **Documentation**: `GET /docs` (Interactive API docs)

## Desktop Client

### Overview

Native Electron desktop application built with React, TypeScript, and Tailwind CSS. Provides a modern, responsive interface for the Reasoning Agent API.

**Status**: ✅ Milestone 1 Complete (Project Scaffolding)

### Quick Start

```bash
# Navigate to client directory
cd client

# Install dependencies (first time only)
npm install

# Start development mode
npm run dev
```

The Electron app will open automatically. Ensure backend services are running (`make docker_up`).

### Features

- **Native Desktop App**: Electron-based, cross-platform (macOS, Windows, Linux)
- **Modern UI**: React 18 + TypeScript + Tailwind CSS + shadcn/ui (coming in M3)
- **Type-Safe**: Strict TypeScript mode with full API type definitions
- **Real-time Streaming**: SSE-based streaming for chat and reasoning steps (coming in M2)
- **Conversation Management**: Persistent conversations via backend API (coming in M9)
- **Security-First**: Electron security best practices (contextIsolation, sandbox)

### Client Architecture

```
Developer's Machine:
┌─────────────────────────────┐
│  Electron App (native)      │
│  cd client && npm run dev   │
└──────────┬──────────────────┘
           │ HTTP
           ▼
    http://localhost:8000
           │
┌──────────┴──────────────────┐
│  Docker Compose Services    │
│  make docker_up             │
│  - reasoning-api            │
│  - litellm                  │
│  - postgres                 │
│  - phoenix                  │
└─────────────────────────────┘
```

### Development Commands

```bash
cd client

npm run dev           # Start Electron with hot-reload
npm test              # Run Jest tests
npm run type-check    # TypeScript type checking
npm run build         # Build production app (DMG/EXE/AppImage)
```

See [client/README.md](client/README.md) for detailed documentation.

## MCP Architecture Overview

The reasoning API supports two ways to add tools via MCP (Model Context Protocol):

### Option 1: Custom HTTP MCP Servers (`mcp_servers/`)

Build your own MCP servers that run as HTTP services:

```python
# Example: mcp_servers/fake_server.py
from fastmcp import FastMCP
mcp = FastMCP("my-tools")

@mcp.tool
async def my_tool(input: str) -> str:
    return f"Processed: {input}"

mcp.run(transport="http", port=8001)
```

**Start**: `make demo_mcp_server` or `docker compose --profile demo up`
**Connect API**: Already configured in `config/mcp_servers.json` as `demo_tools`

### Option 2: MCP Bridge for Stdio Servers (`mcp_bridge/`)

**Why?** Many MCP servers (filesystem, mcp-this, github) only support stdio transport. These need local file access (e.g., `/Users/you/repos/`). Running them inside Docker requires complex volume mounts. The bridge solves this by running stdio servers locally on your host machine and providing HTTP access for the API (which runs in Docker).

Access existing stdio-based MCP servers via HTTP bridge:

```bash
# 1. Configure stdio servers
cp config/mcp_bridge_config.example.json config/mcp_bridge_config.json
# Edit config/mcp_bridge_config.json with your paths

# 2. Start bridge
uv run python mcp_bridge/server.py

# 3. Enable in config/mcp_servers.json (set "enabled": true for "local_bridge")
```

**See [README_MCP_QUICKSTART.md](README_MCP_QUICKSTART.md) for complete Docker setup guide.**
**See [mcp_bridge/README.md](mcp_bridge/README.md) for bridge documentation.**

### How API Connects to MCP Servers

The API uses `config/mcp_servers.json` to list all HTTP MCP endpoints:

```json
{
  "mcpServers": {
    "demo_tools": {
      "url": "http://localhost:8001/mcp/",
      "enabled": true
    },
    "local_bridge": {
      "url": "http://localhost:9000/mcp/",
      "enabled": false
    }
  }
}
```

**Key Point**: Both approaches expose tools via HTTP - the API doesn't care if they're custom servers (`mcp_servers/`) or the bridge (`mcp_bridge/`).

## Development

### Common Commands

Run `make help` to see all available commands. Most common:

```bash
# Setup
make dev                    # Install dependencies

# Docker
make docker_up              # Start all services
make docker_logs            # View logs
make docker_restart         # Restart services
make docker_down            # Stop services

# Local Development
make api                    # Start API server
make web_client             # Start web interface
make demo_mcp_server        # Start demo MCP server

# Testing
make tests                  # Linting + all tests
make non_integration_tests  # Fast tests only
make integration_tests      # Full integration tests
```

See `Makefile` for complete list of commands and advanced options.

### Adding New MCP Servers

**Local stdio servers** (filesystem, mcp-this):
- See [README_MCP_QUICKSTART.md](README_MCP_QUICKSTART.md) for complete setup guide
- Configure in `config/mcp_bridge_config.json` (see [mcp_bridge/README.md](mcp_bridge/README.md))

**HTTP MCP servers** (deployed services):
1. Create server (see `mcp_servers/fake_server.py` example)
2. Add to `config/mcp_servers.json`:
   ```json
   {
     "mcpServers": {
       "your_server": {
         "url": "http://localhost:8002/mcp/",
         "transport": "http",
         "enabled": true
       }
     }
   }
   ```
3. Add to Docker Compose if needed (see existing services for template)

Override config path: `MCP_CONFIG_PATH=path/to/config.json make api`

## Deployment

### Container Platforms (Recommended)

Deploy using Docker Compose on any container platform:

```bash
# 1. Push to GitHub with docker-compose.yml
# 2. Connect to platform (Railway, Fly.io, etc.)
# 3. Set environment variables:
#    - OPENAI_API_KEY=your-key
#    - API_TOKENS=web-client-prod-token,admin-prod-token
#    - REASONING_API_TOKEN=web-client-prod-token
#    - REQUIRE_AUTH=true
# 4. Deploy (platform auto-detects docker-compose.yml)
```
 s
### Individual Service Deployment

For platforms requiring separate services:

**API Service:**
- Build Command: `uv sync --group api --no-dev`
- Start Command: `uv run uvicorn api.main:app --host 0.0.0.0 --port $PORT`

**Web Client:**
- Build Command: `uv sync --group web --no-dev`
- Start Command: `uv run python web-client/main.py`
- Environment: `REASONING_API_URL=https://your-api-service.com`

**MCP Server:**
- Build Command: `uv sync --group mcp --no-dev`
- Start Command: `uv run python mcp_servers/fake_server.py`

## Configuration

### Dependencies

The project uses a single `pyproject.toml` with dependency groups for clean separation:

- **Base dependencies**: Shared across all services (httpx, uvicorn, etc.)
- **`api` group**: FastAPI, OpenAI client, MCP integration
- **`web` group**: FastHTML, MonsterUI for web interface  
- **`mcp` group**: FastMCP for building MCP servers
- **`dev` group**: Testing, linting, and development tools

```bash
# Install everything for local development
make dev                       # or: uv sync --all-groups

# Install specific service dependencies
uv sync --group api            # API service only
uv sync --group web            # Web client only  
uv sync --group mcp            # MCP server only
```

### Environment Variables

Configuration is managed through `.env` files:

- **Development**: Copy `.env.dev.example` to `.env` - see file for detailed documentation
- **Production**: Copy `.env.prod.example` to `.env` - includes secure defaults

**Key Variables**:
- `OPENAI_API_KEY` - Your OpenAI key (used only by LiteLLM proxy)
- `LITELLM_API_KEY` - Virtual key for app code (generated via `make litellm_setup`)
- `LITELLM_TEST_KEY` - Virtual key for integration tests
- `LITELLM_EVAL_KEY` - Virtual key for evaluations
- `API_TOKENS` - Comma-separated auth tokens for API access
- `REQUIRE_AUTH` - Enable/disable authentication (false for dev, true for prod)

See example files for complete configuration options and detailed comments.

### MCP Server Configuration

**Setup:**
```bash
# Copy example configuration
cp config/mcp_servers.example.json config/mcp_servers.json

# Edit config/mcp_servers.json to enable/disable servers
```

Example `config/mcp_servers.json`:
```json
{
  "mcpServers": {
    "local_bridge": {
      "url": "${MCP_BRIDGE_URL:-http://localhost:9000/mcp/}",
      "enabled": false,
      "description": "Local stdio servers via bridge"
    }
  }
}
```

**Notes:**
- Environment variables allow Docker to override URLs (e.g., `localhost` → `fake-mcp-server`)
- `config/mcp_servers.json` is gitignored (user-specific configuration)
- Override config path: `MCP_CONFIG_PATH=path/to/config.json make api`

For stdio servers (filesystem, mcp-this): See [README_MCP_QUICKSTART.md](README_MCP_QUICKSTART.md)

### Authentication

The API supports multiple authentication tokens:

```bash
# Development (permissive)
REQUIRE_AUTH=false

# Production (secure)
REQUIRE_AUTH=true
API_TOKENS=web-client-prod-token,admin-prod-token,mobile-prod-token
```

Generate secure tokens:
```bash
uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Testing

### Test Commands

```bash
# Full test suite
make tests                      # Linting + all tests

# Test variations
make non_integration_tests      # Fast tests (no OpenAI API)
make integration_tests          # Full integration (requires setup - see below)
make linting                    # Code formatting only

# Docker testing
make docker_test                # Run tests in container
```

### Integration Tests Setup

Integration tests require LiteLLM proxy and virtual keys:

```bash
# 1. Start LiteLLM stack
docker compose up -d litellm postgres-litellm

# 2. Generate virtual keys (if not already done)
make litellm_setup
# Copy LITELLM_TEST_KEY to .env

# 3. Run integration tests
make integration_tests
```

**Note**: Integration tests use `LITELLM_TEST_KEY` from `.env` automatically. See `.env.dev.example` for configuration details.

### Test Structure

- **Unit Tests**: `tests/test_*.py` (API, models, reasoning logic)
- **Integration Tests**: Marked with `@pytest.mark.integration`, use LiteLLM proxy
- **Evaluations**: LLM behavioral testing with `flex-evals` (opt-in with `make evaluations`)
- **CI/CD**: Uses `non_integration_tests` for speed, `integration_tests` for validation

## Demo Scripts

### Available Demos

```bash
# Complete demo with MCP tools
make demo                       # Requires: API + MCP server running

# Individual demo scripts
uv run python examples/demo_complete.py      # Full reasoning demo
uv run python examples/demo_basic.py         # Basic OpenAI SDK demo
uv run python examples/demo_raw_api.py       # Low-level HTTP demo
```

### Demo MCP Server

The project includes a full-featured demo MCP server:

```bash
# Start demo server
make demo_mcp_server

# Available tools:
# - get_weather: Weather information
# - search_web: Web search results
# - get_stock_price: Stock market data
# - analyze_text: Sentiment analysis
# - translate_text: Language translation
```

## Advanced Usage

### Custom System Prompts

The web interface includes a power user mode for custom prompts:

1. Open http://localhost:8080
2. Use the settings panel on the left
3. Enter custom system prompts
4. Adjust temperature, max tokens, etc.

### MCP Tool Development & Testing

Create custom MCP servers using FastMCP:

```python
from fastmcp import FastMCP

mcp = FastMCP("my-tools")

@mcp.tool
async def my_custom_tool(input: str) -> dict:
    """My custom tool description."""
    return {"result": f"Processed: {input}"}

# Deploy as HTTP server
mcp.run(transport="http", host="0.0.0.0", port=8002)
```

### MCP Inspector

Test MCP servers:

```bash
npx @modelcontextprotocol/inspector
```

1. Start server (demo: `make demo_mcp_server`, bridge: see [README_MCP_QUICKSTART.md](README_MCP_QUICKSTART.md))
2. Set `Transport Type` to `Streamable HTTP`
3. Enter URL: `http://localhost:8001/mcp/` (demo) or `http://localhost:9000/mcp/` (bridge)
4. Click `Connect` → `Tools` → `List Tools`

**Note**: Always include `/mcp/` at the end of URLs.

### Phoenix Playground Setup

To test your reasoning API using Phoenix's built-in playground:

1. **Start services**: `make docker_up`
2. **Open Phoenix UI**: http://localhost:6006
3. **Navigate to Playground**: Click on the "Playground" tab
4. **Configure API settings**:
   - Base URL: `http://reasoning-api:8000/v1`
5. **Test**: Send messages to see reasoning steps in action

**Important**: Use `http://reasoning-api:8000/v1` (not `localhost`) when running in Docker containers.


### Monitoring and Health Checks

All services include health endpoints:

```bash
# Check service health
curl http://localhost:8000/health  # API
curl http://localhost:8001/        # MCP Server
```

## Troubleshooting

### Common Issues

**"Connection refused"**:
- Ensure services are running: `make docker_up` or start individually
- Check ports are available: `lsof -i :8000 :8080 :8001`

**Authentication errors**:
- Verify `REASONING_API_TOKEN` matches one in `API_TOKENS`
- For development, set `REQUIRE_AUTH=false`

**Environment variables not found**:
- Ensure `.env` file exists in project root
- Copy from template: `cp .env.dev.example .env`

**Docker issues**:
- Clean restart: `make docker_down && make docker_up`
- Check logs: `make docker_logs`

### Getting Help

1. **Check logs**: `make docker_logs` or individual service logs
2. **Run health checks**: Verify all services are healthy
3. **Test API directly**: Use curl commands from this README
4. **Review configuration**: Ensure `.env` file is properly configured

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make tests`
5. Test with Docker: `make docker_up`
6. Submit a pull request

## License

Apache 2.0 License - see [LICENSE](LICENSE) file for details.

---

## Additional Resources

- **MCP Bridge Setup**: [README_MCP_QUICKSTART.md](README_MCP_QUICKSTART.md) - Complete guide to setting up stdio MCP servers with Docker
- **MCP Bridge Documentation**: [mcp_bridge/README.md](mcp_bridge/README.md) - MCP bridge technical documentation
- **Docker Setup**: [README_DOCKER.md](README_DOCKER.md) - Detailed Docker instructions
- **Phoenix Setup**: [README_PHOENIX.md](README_PHOENIX.md) - LLM observability and tracing
- **MCP Inspector**: Use `npx @modelcontextprotocol/inspector` to test MCP servers
- **OpenAI API Docs**: [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- **FastMCP**: [FastMCP Documentation](https://github.com/jlowin/fastmcp) for building MCP servers
- **Phoenix Arize**: [Phoenix Documentation](https://arize.com/docs/phoenix) for observability
