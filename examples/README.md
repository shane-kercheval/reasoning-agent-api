# Reasoning Agent Examples

This directory contains demo scripts showing different ways to use the reasoning agent API.

## Prerequisites

Before running any demo:

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Set your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY='your-key-here'
   ```

3. **For MCP tools demos**, start the demo MCP server:
   ```bash
   make demo_mcp_server
   # This starts on port 8001 to avoid conflict with the API on port 8000
   ```

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

**Features**: Basic chat completions, model listing, authentication
**When to use**: Quick testing, verifying API is working

```bash
# Start API (no MCP tools needed for basic demo)
make api

# Run demo
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

Each demo uses its own MCP configuration file from `examples/configs/`:

- `demo_complete.yaml` - Enables demo MCP server with all tools
- `demo_basic.yaml` - No MCP tools (empty configuration)  
- `demo_raw_api.yaml` - Enables demo MCP server for raw API testing

The demos automatically use the appropriate configuration when started with `make demo_api` or by setting `MCP_CONFIG_PATH`.

## Next Steps

- Start with `demo_complete.py` to see all features
- Use `demo_basic.py` as a template for your own applications
- Refer to `demo_raw_api.py` only if you need low-level API access

For more details, see the main [README](../README.md).