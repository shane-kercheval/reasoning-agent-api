[![Tests](https://github.com/shane-kercheval/reasoning-agent-api/actions/workflows/tests.yaml/badge.svg)](https://github.com/shane-kercheval/reasoning-agent-api/actions/workflows/tests.yaml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Reasoning Agent API

An OpenAI-compatible API that adds reasoning capabilities and structured tool execution.

## Features

- **OpenAI Compatible**: Drop-in replacement for OpenAI's chat completion API
- **Intelligent Request Routing**: Three execution paths (passthrough, reasoning, orchestration) with auto-classification
- **LiteLLM Gateway**: Unified LLM proxy for centralized observability and connection pooling
- **Reasoning Agent**: Single-loop reasoning with visual thinking steps
- **Tools API Integration**: REST-based tool execution with structured responses
- **Conversation Storage**: PostgreSQL-backed persistent conversation history
- **Desktop Client**: Native Electron app with React, TypeScript, and Tailwind CSS
- **Real-time Streaming**: See reasoning and responses as they happen
- **Request Cancellation**: Stop reasoning immediately when clients disconnect
- **Simple Authentication**: Token-based authentication with multiple token support
- **Docker Ready**: Full Docker Compose setup for easy deployment
- **Phoenix Observability**: LLM tracing and monitoring with Phoenix Arize

## Quick Start

### Prerequisites

- **Docker & Docker Compose** (required)
- **OpenAI API key** (required)
- **Node.js 18+** (optional, for desktop client only)

### Setup

1. **Setup environment**

   ```bash
   cp .env.dev.example .env
   ```

   Edit `.env` and set (see file for detailed documentation):
   - `OPENAI_API_KEY=your-openai-key-here`
   - `LITELLM_MASTER_KEY=` (generate with: `python -c "import secrets; print('sk-' + secrets.token_urlsafe(32))"`)
   - `LITELLM_POSTGRES_PASSWORD=` (generate with: `python -c "import secrets; print(secrets.token_urlsafe(16))"`)
   - `PHOENIX_POSTGRES_PASSWORD=` (generate with: `python -c "import secrets; print(secrets.token_urlsafe(16))"`)
   - `REASONING_POSTGRES_PASSWORD=` (generate with: `python -c "import secrets; print(secrets.token_urlsafe(16))"`)

2. **Start all services**

   ```bash
   make docker_up
   ```

   Wait for services to be healthy.

3. **Run database migrations** (for conversation storage)

   ```bash
   make reasoning_migrate
   ```

4. **Setup LiteLLM virtual keys**

   ```bash
   make litellm_setup
   ```

   Copy the generated keys to `.env`:
   - `LITELLM_API_KEY=sk-...` (development/production usage)
   - `LITELLM_TEST_KEY=sk-...` (integration tests)
   - `LITELLM_EVAL_KEY=sk-...` (LLM behavioral evaluations)

   Then restart the API to apply the new keys:

   ```bash
   make docker_restart
   ```

5. **Access your services**
   - API Documentation: http://localhost:8000/docs
   - LiteLLM Dashboard: http://localhost:4000
   - Phoenix UI: http://localhost:6006
   - Tools API: http://localhost:8001/tools/

6. **Setup tools-api volume mounts** (optional)

   To give the tools-api access to your local filesystem:

   ```bash
   cp docker-compose.override.yml.example docker-compose.override.yml
   ```

   Edit `docker-compose.override.yml` to add your local paths. The pattern mirrors the full host path inside the container:

   ```yaml
   services:
     tools-api:
       volumes:
         # Read-write directories (agent can edit)
         - /Users/yourname/repos:/mnt/read_write/Users/yourname/repos:rw
         - /Users/yourname/workspace:/mnt/read_write/Users/yourname/workspace:rw

         # Read-only directories (agent can only read)
         - /Users/yourname/Downloads:/mnt/read_only/Users/yourname/Downloads:ro
         - /Users/yourname/Documents:/mnt/read_only/Users/yourname/Documents:ro
   ```

   Then restart: `make docker_restart`

7. **Start desktop client** (optional)

   ```bash
   make client
   ```

   Or manually: `cd client && npm install && npm run dev`

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Your Machine                                    │
│                                                                              │
│  ┌──────────────────┐                                                        │
│  │  Desktop Client  │ (Electron - runs natively, not in Docker)              │
│  └────────┬─────────┘                                                        │
│           │ HTTP                                                             │
│           ▼                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                         Docker Compose Network                         │  │
│  │                                                                        │  │
│  │  ┌─────────────────────┐       ┌─────────────────────┐                 │  │
│  │  │   reasoning-api     │       │     tools-api       │                 │  │
│  │  │   localhost:8000    │──────▶│   localhost:8001    │                 │  │
│  │  │                     │ HTTP  │                     │                 │  │
│  │  │  - Chat completions │       │  - File operations  │                 │  │
│  │  │  - Request routing  │       │  - GitHub tools     │                 │  │
│  │  │  - Reasoning agent  │       │  - Web search       │                 │  │
│  │  └──────────┬──────────┘       └──────────┬──────────┘                 │  │
│  │             │                             │                            │  │
│  │             │ HTTP                        │ Volume Mounts              │  │
│  │             ▼                             ▼                            │  │
│  │  ┌─────────────────────┐       ┌─────────────────────┐                 │  │
│  │  │      litellm        │       │   /mnt/read_write   │◀── Your repos   │  │
│  │  │   localhost:4000    │       │   /mnt/read_only    │◀── Your docs    │  │
│  │  │                     │       └─────────────────────┘                 │  │
│  │  │  - LLM proxy        │                                               │  │
│  │  │  - Virtual keys     │                                               │  │
│  │  │  - Usage tracking   │                                               │  │
│  │  └──────────┬──────────┘                                               │  │
│  │             │                                                          │  │
│  │             │ HTTP (OpenAI API)                                        │  │
│  │             ▼                                                          │  │
│  │       ┌───────────┐                                                    │  │
│  │       │  OpenAI   │ (external)                                         │  │
│  │       └───────────┘                                                    │  │
│  │                                                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │  │                    phoenix (localhost:6006)                     │   │  │
│  │  │                                                                 │   │  │
│  │  │  Receives OpenTelemetry (OTLP) traces from:                     │   │  │
│  │  │  • reasoning-api - request routing, reasoning steps, tool calls │   │  │
│  │  │  • litellm - LLM API calls, token usage, costs                  │   │  │
│  │  │                                                                 │   │  │
│  │  │  View at: http://localhost:6006                                 │   │  │
│  │  └─────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. Desktop client sends chat requests to `reasoning-api`
2. `reasoning-api` routes requests (passthrough, reasoning, or orchestration)
3. LLM calls go through `litellm` proxy for unified API access and usage tracking
4. `litellm` forwards requests to OpenAI (or other configured providers)
5. `reasoning-api` can call `tools-api` for file operations, GitHub, web search
6. `tools-api` accesses your local filesystem via Docker volume mounts

**Observability:**
- All services send OpenTelemetry traces to Phoenix for distributed tracing
- `reasoning-api` traces: HTTP requests, routing decisions, reasoning iterations, tool execution
- `litellm` traces: LLM API calls with token counts, latency, and costs
- View full request traces at http://localhost:6006

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
- **Activate**: `X-Routing-Mode: orchestration` (returns 501 until implemented)

### **Auto-Routing**
- LLM classifier chooses between passthrough and orchestration
- Uses GPT-4o-mini for classification (fast, deterministic)
- **Activate**: `X-Routing-Mode: auto`

**Note**: Requests with `response_format` or `tools` always use passthrough.

## API Usage

### Request Examples

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
    stream=True,
)
async for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

**Notes:**
- `api_key` should be a token from your API's `API_TOKENS` environment variable
- If `REQUIRE_AUTH=false` (development mode), any value works
- The API handles LLM calls using its own LiteLLM virtual key

### API Endpoints

**Reasoning API:**
- `POST /v1/chat/completions` - Chat completions (OpenAI compatible)
- `GET /v1/models` - List available models
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

**Tools API:**
- `GET /tools/` - List all available tools
- `POST /tools/{tool_name}` - Execute a tool
- `GET /prompts/` - List all available prompts
- `POST /prompts/{prompt_name}` - Render a prompt template
- `GET /health` - Health check
- `POST /mcp/` - MCP protocol endpoint (JSON-RPC over HTTP)
- `GET /mcp/health` - MCP server health check

## Desktop Client

Native Electron desktop application built with React, TypeScript, and Tailwind CSS.

```bash
# Start development mode (requires backend services running)
make client

# Run tests
make client_tests

# Build for production
make client_build
```

See [client/README.md](client/README.md) for detailed documentation.

## Tools API

The tools-api service provides structured tool execution via REST endpoints.

### Volume Mounts

Tools API uses volume mounts to access your local filesystem. Configure in `docker-compose.override.yml`:

- **Read-write** (`/mnt/read_write/...`): Directories the agent can edit (repos, workspace)
- **Read-only** (`/mnt/read_only/...`): Directories the agent can only read (downloads, documents)

The path inside the container mirrors the host path for transparent path translation.

### Available Tools

**Filesystem Tools** (require volume mounts):
- `read_text_file`, `write_file`, `edit_file`
- `list_directory`, `search_files`, `get_file_info`
- `list_allowed_directories`

**GitHub Tools** (require `GITHUB_TOKEN` in `.env`):
- `get_github_pull_request_info`
- `get_local_git_changes_info`
- `get_directory_tree`

**Web Search** (requires `BRAVE_API_KEY` in `.env`):
- `web_search`

### Testing Tools

```bash
# List all available tools
curl http://localhost:8001/tools/ | jq

# Execute a tool
curl -X POST http://localhost:8001/tools/list_allowed_directories | jq

# Health check
curl http://localhost:8001/health
```

### MCP Protocol

Tools API also exposes an MCP (Model Context Protocol) endpoint at `/mcp/` for MCP-compatible clients. This provides the same tools and prompts via the standardized MCP JSON-RPC protocol.

```bash
# MCP health check
curl http://localhost:8001/mcp/health | jq

# Test with MCP Inspector
npx @modelcontextprotocol/inspector http://localhost:8001/mcp
```

See [tools_api/README.md](tools_api/README.md) for detailed documentation.

## Development

Run `make help` to see all available commands.

### Integration Tests Setup

Integration tests require LiteLLM proxy and virtual keys:

```bash
# 1. Start services
make docker_up

# 2. Generate virtual keys (if not already done)
make litellm_setup
# Copy LITELLM_TEST_KEY to .env

# 3. Run integration tests
make integration_tests
```

### Adding New Tools

See [tools_api/README.md](tools_api/README.md) for patterns and examples.

## Authentication

The API supports token-based authentication:

```bash
# Development (permissive)
REQUIRE_AUTH=false

# Production (secure)
REQUIRE_AUTH=true
API_TOKENS=token1,token2,token3
```

When `REQUIRE_AUTH=true`, requests must include a valid token from `API_TOKENS`:

```bash
curl -H "Authorization: Bearer token1" http://localhost:8000/v1/chat/completions ...
```

Generate secure tokens:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Troubleshooting

### Common Issues

**"Connection refused"**:
- Ensure services are running: `make docker_up`
- Check ports are available: `lsof -i :8000 :8001 :4000`

**Authentication errors**:
- Verify your token is in `API_TOKENS`
- For development, set `REQUIRE_AUTH=false`

**Tools API "Path not accessible"**:
- Ensure volume mounts mirror the full host path (see Quick Start step 6)
- Restart after changing mounts: `make docker_restart`
- Verify mounts: `docker compose exec tools-api ls -la /mnt/read_write/`

**Docker issues**:
- Clean restart: `make docker_down && make docker_up`
- Check logs: `make docker_logs`
- Full rebuild: `make docker_rebuild`

### Health Checks

```bash
curl http://localhost:8000/health  # Reasoning API
curl http://localhost:8001/health  # Tools API
curl http://localhost:4000/health  # LiteLLM Proxy
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make tests`
5. Submit a pull request

## License

Apache 2.0 License - see [LICENSE](LICENSE) file for details.

## Additional Resources

- [Tools API Documentation](tools_api/README.md)
- [Desktop Client Documentation](client/README.md)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Phoenix Documentation](https://arize.com/docs/phoenix)
