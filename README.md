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
```

### Demo Scripts

#### Quick Demo Setup

```bash
# Install dependencies
uv sync

# Set OpenAI API key
export OPENAI_API_KEY='your-key-here'

# Start reasoning agent API
# or `uv run python -m api.main`
make api

# Start demo MCP server (in another terminal)
# or `uv run python mcp_servers/fake_server.py`
make demo_mcp_server

# Run complete demo (in third terminal)
# or `uv run python examples/demo_complete.py`
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

### Deploy Demo MCP Server

The demo MCP server can be deployed to any hosting platform for remote testing:

#### Render.com (Recommended)

1. **One-click Deploy**:
   - Fork this repository
   - Connect to [Render.com](https://render.com)
   - Use the included `render.yaml` configuration

2. **Manual Deploy**:
   ```bash
   # Create new web service on Render
   # Repository: your-fork-url
   # Build Command: pip install fastmcp
   # Start Command: python mcp_servers/fake_server.py
   # Environment: Set PORT (auto-generated by Render)
   ```

3. **Configure Remote Server**:
   ```yaml
   # config/mcp_servers.yaml
   servers:
     - name: remote_demo
       url: https://your-app.onrender.com/mcp/
       enabled: true
   ```

#### Other Platforms

The demo server works on any platform with Python support:

- **Railway**: `railway deploy` with `python mcp_servers/fake_server.py`
- **Heroku**: Add `Procfile` with `web: python mcp_servers/fake_server.py`  
- **DigitalOcean**: Deploy as container or app platform
- **AWS/GCP**: Use App Runner, Cloud Run, or Lambda

#### Environment Variables

For deployment platforms:
```bash
PORT=8000          # Auto-set by most platforms
HOST=0.0.0.0       # Required for external access
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
