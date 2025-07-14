[![Tests](https://github.com/shane-kercheval/reasoning-agent-api/actions/workflows/tests.yaml/badge.svg)](https://github.com/shane-kercheval/reasoning-agent-api/actions/workflows/tests.yaml)
[![Python 3.11+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Reasoning Agent API

An OpenAI-compatible API wrapper that adds reasoning capabilities through MCP (Model Context Protocol) tools.

## Features

- **OpenAI SDK Compatible**: Drop-in replacement for OpenAI's API
- **Reasoning Steps**: Streams thinking process before final responses
- **MCP Tool Integration**: Extensible with Model Context Protocol tools
- **Full API Compatibility**: Supports all OpenAI chat completion parameters

## Quick Start

### Prerequisites

- OpenAI API key
- uv package manager

### Running the Server

1. **Configure Environment**:

    ```bash
    # Copy the example configuration
    cp examples/.env.example .env
    ```

    Update `.env` with your OpenAI API key and other settings

    ```
    # For local development (recommended):
    OPENAI_API_KEY=your-openai-api-key-here
    # REQUIRE_AUTH=false
    REQUIRE_AUTH=true
    API_TOKENS=token1,token2,token3
    ```

2. **Start the Server**:

    ```bash
    # Start the API server
    make api
    # or: uv run uvicorn api.main:app --reload --port 8000
    ```

3. **Access the API**:
   - **API Base URL**: http://127.0.0.1:8000
   - **Swagger UI**: http://127.0.0.1:8000/docs
   - **ReDoc**: http://127.0.0.1:8000/redoc
   - **OpenAPI JSON**: http://127.0.0.1:8000/openapi.json
   - **Health Check**: http://127.0.0.1:8000/health

Minimal request body:

    ```
    {
        "model": "gpt-4o-mini",
        "messages": [
            {
            "role": "user",
            "content": "Hello! Can you help me test this API?"
            }
        ]
    }
    ```

4. **Test It with curl**

  If you have `REQUIRE_AUTH=true` and `API_TOKENS=token1,token2,token3`, you can test with:

    ```bash
    curl -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer token1" \
    -d '{
        "model": "gpt-4o-mini",
        "messages": [
        {
            "role": "user", 
            "content": "Hello! Can you help me test this API?"
        }
        ]
    }'
    ```

5. **Test It in Swagger UI**

    1. Go to http://127.0.0.1:8000/docs
    2. Click "Authorize" button
    3. Enter: token1 (without "Bearer " prefix)
    4. Click "POST /v1/chat/completions" to expand it
    5. Click "Try it out"
    6. Paste the JSON above into the request body
    7. Click "Execute"

### Usage with OpenAI SDK

```python
from openai import AsyncOpenAI

# Point to your local server
client = AsyncOpenAI(
    api_key="your-openai-api-key",
    base_url="http://localhost:8000/v1"
)

# Use exactly like OpenAI's API
response = await client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather like on Mars?"}],
    stream=True  # See reasoning steps in real-time
)

async for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

## Web Client

A beautiful MonsterUI-powered web interface is available for easy interaction with the reasoning agent.

### Features

- **🎨 Beautiful UI**: Modern, responsive design with MonsterUI
- **💬 Real-time Chat**: Streaming conversations with instant updates
- **🧠 Reasoning Steps**: Collapsible tree view of AI thinking process
- **⚙️ Power User Mode**: Advanced settings for system prompts and parameters
- **📱 Mobile Friendly**: Works on all devices

### Quick Start (Web Client)

1. **Start the API** (if not already running):
   ```bash
   make api
   ```

2. **Start the web client**:
   ```bash
   make web_client
   ```

3. **Open your browser**:
   - Web Interface: http://localhost:8080
   - Start chatting with the reasoning agent!

### Web Client Development

```bash
# Install web client dependencies
cd web-client
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start web client
uv run python main.py
```

### Deployment (Two Services)

For production, deploy as two separate services:

1. **API Service**: Deploy the main directory
   - Root Directory: `/` (main project)
   - Start Command: `uv run uvicorn api.main:app --host 0.0.0.0 --port $PORT`

2. **Web Client Service**: Deploy the web client
   - Root Directory: `web-client`
   - Start Command: `uv run python main.py`
   - Environment Variable: `REASONING_API_URL=https://your-api-service.onrender.com`

See `web-client/README.md` for detailed deployment instructions.

### Demo Scripts

#### Quick Demo Setup

```bash
# Install dependencies
uv sync

# Set OpenAI API key
export OPENAI_API_KEY='your-key-here'

# Start demo MCP server (terminal 1)
make demo_mcp_server

# Start reasoning agent with demo config (terminal 2)
make demo_api

# Run complete demo (terminal 3)
make demo
```

#### Available Demos

1. **Complete Demo** (Recommended) 🌟
   ```bash
   uv run python examples/demo_complete.py
   ```
   - Full reasoning with MCP tools
   - OpenAI SDK integration
   - Error handling and prerequisites
   - Production-ready patterns

2. **Basic OpenAI SDK Demo**
   ```bash
   uv run python examples/demo_basic.py
   ```
   - Simple chat completions
   - Quick compatibility test
   - No MCP tools required

3. **Raw API Demo** (Educational)
   ```bash
   uv run python examples/demo_raw_api.py
   ```
   - Low-level HTTP requests
   - Manual SSE parsing
   - For understanding internals

#### MCP Demo Server

The project includes a production-ready demo MCP server with realistic fake tools:

```bash
# Start locally for development
uv run python mcp_servers/fake_server.py

# Available tools:
# - get_weather: Weather information
# - search_web: Web search results  
# - get_stock_price: Stock market data
# - analyze_text: Sentiment analysis
# - translate_text: Language translation
```

## MCP Configuration

The reasoning agent uses configurable MCP server connections. Configuration files specify which MCP servers to connect to and their settings.

### Configuration Files

- **Default**: `config/mcp_servers.yaml` - Main configuration for production use
- **Demo Configs**: `examples/configs/` - Specific configurations for each demo

### Custom Configuration

Set a custom MCP configuration file using the `MCP_CONFIG_PATH` environment variable:

```bash
# Use a custom config file
MCP_CONFIG_PATH=my_config.yaml make api

# Use demo-specific config
MCP_CONFIG_PATH=examples/configs/demo_complete.yaml make api
```

### Configuration Format

Both YAML and JSON formats are supported:

```yaml
# mcp_servers.yaml
servers:
  - name: my_mcp_server
    url: http://localhost:8001/mcp/
    enabled: true
    auth_env_var: MY_API_KEY  # optional
```

## Testing

### Test Commands

```bash
# Show all available commands
make help

# Main development test suite (recommended)
make tests                   # Linting + all tests (unit + integration)

# Test variations
make unit_tests              # All tests (non-integration + integration)
make non_integration_tests   # Non-integration tests only (fast)
make integration_tests       # Integration tests only (needs OPENAI_API_KEY)

# Individual components
make linting                 # Code formatting and linting only
```

### Test Organization

#### **Non-Integration Tests** (Fast, no external dependencies)
- **`tests/test_api.py`** - API endpoints + OpenAI SDK unit tests
- **`tests/test_reasoning_agent.py`** - Core reasoning logic
- **`tests/test_models.py`** - Pydantic model validation
- **`tests/test_mcp_client.py`** - MCP client functionality

#### **Integration Tests** (Auto-start servers, require OPENAI_API_KEY)
- **`tests/test_api.py::TestOpenAISDKCompatibility`** - Full OpenAI SDK integration
- **`tests/test_openai_api_compatibility.py`** - Real OpenAI API compatibility

### Testing Strategy

1. **Development Workflow**: Use `make non_integration_tests` for rapid feedback
2. **Pre-commit**: Run `make tests` before committing changes
3. **CI/CD**: Use `make non_integration_tests` for fast CI, `make integration_tests` for full validation

### Integration Test Requirements

Integration tests automatically start their own servers and clean up afterwards. You only need:

```bash
export OPENAI_API_KEY="your-api-key"
make integration_tests
```

The tests will:
- Start a server on a random available port
- Run OpenAI SDK compatibility tests
- Test real OpenAI API integration
- Clean up automatically

## MCP Server Deployment

- Inspector can be started via `npx @modelcontextprotocol/inspector`
- Example: start demo MCP server locally for testing via `make demo_mcp_server`
    - Set `Transport Type` to `Streamable HTTP`
    - type `http://0.0.0.0:8001/mcp/` into the URL and click `Connect`
    - Go to `Tools` and click `List Tools` to see available MCP tools
    - Select a tool and test it out.

### Deploy `fake_server.py` MCP Server

The `fake_server.py` MCP server can be deployed to any hosting platform for remote usage/testing:

#### Example: Render.com

1. **One-click Deploy**:
    - Fork this repository
    - Connect to [Render.com](https://render.com)

2. **Manual Deploy**:
    ```bash
    # Create new web service on Render
    # Repository: your-fork-url
    # Build Command: `uv sync --frozen && uv cache prune --ci`  # default value
    # Start Command: `uv run python mcp_servers/fake_server.py`
    ```

3. **Test it in Inspector**:
    - Start Inspector via `npx @modelcontextprotocol/inspector`
    - Set `Transport Type` to `Streamable HTTP`
    - Set URL to your deployed server, e.g. `https://your-fake-server.onrender.com/mcp/`
        - don't forget to add `/mcp/` at the end
    - Click `Connect`
    - Use the Inspector to test tools like `get_weather`, `search_web`, etc.

4. **Update the API**:
    
    - Add environment variable (e.g. `.env`):
        - Name: `MCP_CONFIG_PATH`
        - Value: `config/fake_mcp_server.yaml`
    - Redeploy your API service

5. **Test the API**:
    - Use the OpenAI SDK or curl to test the API with the MCP server
    - Example request:

    ```bash
    curl -X POST https://reasoning-agent-api.onrender.com/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer token1" \
      -d '{
          "model": "gpt-4o-mini",
          "messages": [
          {
              "role": "user",
              "content": "Use the weather tool to get the weather in Paris."
          }
          ],
          "stream": true
      }'
    ```

## Production Deployment

### Configuration

The API uses environment variables for all configuration. Copy `examples/.env.example` to `.env` and configure your values:

```bash
cp examples/.env.example .env
# Edit .env with your configuration
```

#### Required Configuration

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Authentication (Required for Production)
API_TOKENS=token1,token2,token3  # Comma-separated bearer tokens
REQUIRE_AUTH=true
```

#### Optional Configuration

```bash
# HTTP Client Timeouts (seconds)
HTTP_CONNECT_TIMEOUT=5.0    # Fast failure for connection issues
HTTP_READ_TIMEOUT=30.0      # Reasonable for AI responses
HTTP_WRITE_TIMEOUT=10.0     # Request upload timeout

# HTTP Connection Pooling
HTTP_MAX_CONNECTIONS=20               # Total connection limit
HTTP_MAX_KEEPALIVE_CONNECTIONS=5      # Reused connections
HTTP_KEEPALIVE_EXPIRY=30.0           # Connection lifetime

# Server Settings
API_HOST=0.0.0.0
API_PORT=8000
```

### Authentication

The API supports bearer token authentication:

#### Generating Secure Tokens

```bash
# Generate a secure token
uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
```

#### Using Tokens

```bash
# Configure multiple tokens (comma-separated)
API_TOKENS=prod-token-abc123,backup-token-xyz789

# Enable authentication
REQUIRE_AUTH=true
```

#### Client Usage

```python
# With OpenAI SDK
client = AsyncOpenAI(
    api_key="your-openai-api-key",
    base_url="https://your-api.com/v1",
    default_headers={"Authorization": "Bearer prod-token-abc123"}
)

# With curl
curl -H "Authorization: Bearer prod-token-abc123" \
     -H "Content-Type: application/json" \
     -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Hello"}]}' \
     https://your-api.com/v1/chat/completions
```

### Monitoring and Health Checks

#### Health Check Endpoint

```bash
# Public endpoint for monitoring (no auth required)
curl https://your-api.com/health

# Response
{"status": "healthy", "timestamp": 1234567890}
```

#### Request/Response Example

```bash
# Test your deployed API
curl -H "Authorization: Bearer your-token" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "gpt-4o-mini",
       "messages": [{"role": "user", "content": "Hello!"}],
       "max_tokens": 50
     }' \
     https://your-api.com/v1/chat/completions
```

### Troubleshooting

#### Common Issues

**Authentication Errors (401)**:
- Check `API_TOKENS` environment variable is set
- Verify `Authorization: Bearer <token>` header format
- Ensure token is in the configured token list

**Server Configuration Errors (500)**:
- Check `OPENAI_API_KEY` is set correctly
- Verify all required environment variables are configured

**Connection Issues**:
- Check HTTP timeout settings (`HTTP_CONNECT_TIMEOUT`, `HTTP_READ_TIMEOUT`)
- Verify OpenAI API key has sufficient credits/permissions

## License

MIT License
