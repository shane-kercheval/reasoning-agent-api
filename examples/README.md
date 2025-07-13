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

3. **Start the reasoning agent**:
   ```bash
   make api
   ```

4. **For MCP tools demos**, also start the demo MCP server:
   ```bash
   uv run python mcp_servers/fake_server.py
   ```

## Available Demos

### 1. `demo_complete.py` - **Recommended Starting Point** 🌟

The most comprehensive demo showing all features with production-ready patterns.

**Features**: OpenAI SDK, MCP tools, streaming, error handling, colored output
**When to use**: Learning the API, building production applications

```bash
uv run python examples/demo_complete.py
```

### 2. `demo_basic.py` - Simple OpenAI SDK Usage

Minimal example showing OpenAI SDK compatibility without MCP tools.

**Features**: Basic chat completions, model listing, authentication
**When to use**: Quick testing, verifying API is working

```bash
uv run python examples/demo_basic.py
```

### 3. `demo_raw_api.py` - Low-Level HTTP API

Educational demo showing raw HTTP requests without using the OpenAI SDK.

**Features**: Direct API calls, manual SSE parsing, reasoning event details
**When to use**: Understanding the API internals, building custom clients, debugging

```bash
uv run python examples/demo_raw_api.py
```

## Authentication

If your API requires authentication (when `REQUIRE_AUTH=true`):

1. Set authentication tokens in your `.env` file:
   ```
   API_TOKENS=token1,token2,token3
   ```

2. The demos will automatically use the first token for Bearer authentication.

## Next Steps

- Start with `demo_complete.py` to see all features
- Use `demo_basic.py` as a template for your own applications
- Refer to `demo_raw_api.py` only if you need low-level API access

For more details, see the main [README](../README.md).