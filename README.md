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
    API_TOKENS=dev-token-123
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

#### OpenAI SDK Demo

A complete demo script showing basic API usage:

```bash
# Run the demo (requires server to be running)
make api
uv run python examples/demo_openai_sdk.py
```

This demonstrates:
- Non-streaming chat completions
- Streaming with reasoning steps  
- Models listing
- Error handling

#### Reasoning Agent Demo

Simple demonstration showing how to use the reasoning agent:

```bash
# 1. Start the reasoning agent server
make api

# 2. (Optional) Start MCP server for tool demos
uv run python mcp_server/server.py

# 3. Run the simple demo
uv run python examples/demo_reasoning_agent.py
```

The demo shows:
- **Non-streaming requests** with reasoning applied
- **Streaming responses** with reasoning events
- **Tool integration** checking available MCP tools
- **Copy-paste ready code** for your own implementation

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
