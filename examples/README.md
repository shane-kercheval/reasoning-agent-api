# Reasoning Agent Examples

This directory contains demo scripts showing different ways to use the reasoning agent API.

## Prerequisites

Before running any demo:

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Setup environment**:
   ```bash
   # Copy environment file
   cp .env.dev.example .env

   # Edit .env and set:
   # - OPENAI_API_KEY (real OpenAI key for LiteLLM)
   # - LITELLM_MASTER_KEY (generate with: python -c "import secrets; print('sk-' + secrets.token_urlsafe(32))")
   # - LITELLM_POSTGRES_PASSWORD (generate with: python -c "import secrets; print(secrets.token_urlsafe(16))")
   ```

3. **Start services and setup LiteLLM**:
   ```bash
   # Start all services (including LiteLLM)
   make docker_up

   # Generate LiteLLM virtual keys
   make litellm_setup

   # Copy generated keys to .env:
   # LITELLM_API_KEY=sk-...
   # LITELLM_EVAL_KEY=sk-...

   # Restart API to load keys
   docker compose restart reasoning-api
   ```

4. **For MCP tools demos**, the demo MCP server should already be running from step 3.
   - MCP server runs on port 8001
   - API runs on port 8000

## Available Demos

### 1. `demo_complete.py` - **Recommended Starting Point** 🌟

The most comprehensive demo showing all features with production-ready patterns.

**Features**: OpenAI SDK, MCP tools, streaming, error handling, colored output
**When to use**: Learning the API, building production applications

```bash
# Terminal 1: Start demo MCP server
make demo_mcp_server

# Terminal 2: Start API with demo config  
make demo_api

# Terminal 3: Run demo
uv run python examples/demo_complete.py
```

### 2. `demo_basic.py` - Simple OpenAI SDK Usage

Minimal example showing OpenAI SDK compatibility without MCP tools.

**Features**: Basic chat completions, model listing, authentication, request routing
**When to use**: Quick testing, verifying API is working, testing routing paths

```bash
# With Docker (recommended):
make docker_up  # API, LiteLLM, and all services running
uv run python examples/demo_basic.py

# Or local development:
# Requires LiteLLM running (via Docker or locally)
make api  # Start API locally (will connect to LiteLLM)
uv run python examples/demo_basic.py
```

### 3. `demo_raw_api.py` - Low-Level HTTP API

Educational demo showing raw HTTP requests without using the OpenAI SDK.

**Features**: Direct API calls, manual SSE parsing, reasoning event details
**When to use**: Understanding the API internals, building custom clients, debugging

```bash
# Terminal 1: Start demo MCP server
make demo_mcp_server

# Terminal 2: Start API with demo config
MCP_CONFIG_PATH=examples/configs/demo_raw_api.json make api

# Terminal 3: Run demo
uv run python examples/demo_raw_api.py
```

## Authentication

If your API requires authentication (when `REQUIRE_AUTH=true`):

1. Set authentication tokens in your `.env` file:
   ```
   API_TOKENS=token1,token2,token3
   ```

2. The demos will automatically use the first token for Bearer authentication.

## Configuration

### LiteLLM Integration

All demos connect to the API through the LiteLLM proxy:
- **Local development**: Set `LITELLM_BASE_URL=http://localhost:4000` in `.env`
- **Docker**: Automatically configured to `http://litellm:4000` via docker-compose.yml
- **Virtual keys**: Use `LITELLM_API_KEY` from virtual key setup (`make litellm_setup`)

All LLM requests (including demo scripts) flow through LiteLLM for:
- Centralized observability via Phoenix
- Virtual key tracking
- Usage metrics
- OTEL trace collection

### MCP Configuration

Each demo uses its own MCP configuration file from `examples/configs/`:

- `demo_complete.json` - Enables demo MCP server with all tools
- `demo_basic.json` - No MCP tools (empty configuration)
- `demo_raw_api.json` - Enables demo MCP server for raw API testing

The demos automatically use the appropriate configuration when started with `make demo_api` or by setting `MCP_CONFIG_PATH`.

## Next Steps

- Start with `demo_complete.py` to see all features
- Use `demo_basic.py` as a template for your own applications
- Refer to `demo_raw_api.py` only if you need low-level API access

For more details, see the main [README](../README.md).