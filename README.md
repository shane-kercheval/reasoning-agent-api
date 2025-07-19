[![Tests](https://github.com/shane-kercheval/reasoning-agent-api/actions/workflows/tests.yaml/badge.svg)](https://github.com/shane-kercheval/reasoning-agent-api/actions/workflows/tests.yaml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Reasoning Agent API

An OpenAI-compatible API that adds reasoning capabilities and tool usage through MCP (Model Context Protocol) servers. Includes a beautiful web interface for interactive conversations.

## Features

- **🔄 OpenAI Compatible**: Drop-in replacement for OpenAI's chat completion API
- **🧠 Reasoning Steps**: Streams AI thinking process before final responses
- **🔧 MCP Tool Integration**: Extensible with Model Context Protocol tools
- **🎨 Web Interface**: MonsterUI-powered chat interface with reasoning visualization
- **📊 Real-time Streaming**: See reasoning and responses as they happen
- **🔒 Simple Authentication**: Token-based authentication with multiple token support
- **🐳 Docker Ready**: Full Docker Compose setup for easy deployment

## Quick Start

### Prerequisites

- **OpenAI API key** (required)
- **Docker & Docker Compose** (recommended) OR **Python 3.13+ & uv** (for local development)

### Option 1: Docker Compose (Recommended)

Get everything running in 60 seconds:

1. Setup environment
    - run `cp .env.dev.example .env`
    - Edit .env and modify `OPENAI_API_KEY=your-key-here`
2. Start all services
    - run `make docker_up`
3. Access your services
    - Web Interface: http://localhost:8080
    - API Documentation: http://localhost:8000/docs
    - MCP Tools: http://localhost:8001/mcp/
4. Test MCP tools with Inspector
    - Run `npx @modelcontextprotocol/inspector`
    - Set `Transport Type` to `Streamable HTTP`
    - Enter `http://localhost:8001/mcp/` in the URL field and click `Connect`
    - Go to `Tools` and click `List Tools` to see available
    - Select a tool and test it out.

### Option 2: Local Development

For development with individual service control:

```bash
# 1. Setup environment
cp .env.dev.example .env
# Edit .env and add your OpenAI API key

# 2. Install dependencies
make dev          # Installs all dependencies for development

# 3. Start services (3 terminals)
make demo_mcp_server  # Terminal 1: MCP tools
make api              # Terminal 2: API server
make web_client       # Terminal 3: Web interface

# 4. Access at http://localhost:8080
```

## API Usage

### OpenAI SDK Integration

```python
from openai import AsyncOpenAI

# Point to your reasoning agent
client = AsyncOpenAI(
    api_key="your-openai-api-key",
    base_url="http://localhost:8000/v1"
)

# Use exactly like OpenAI's API
response = await client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather like?"}],
    stream=True  # See reasoning steps in real-time
)

async for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

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

## Web Interface

### Features

- **💬 Real-time Chat**: Streaming conversations with instant updates
- **🧠 Reasoning Visualization**: Collapsible tree view of AI thinking process
- **🔧 Tool Usage Display**: Shows MCP tool calls and results
- **⚙️ Power User Controls**: Advanced settings for prompts and parameters
- **📱 Responsive Design**: Works on desktop, tablet, and mobile

### Using the Web Interface

1. **Start the services** (Docker or local as shown above)
2. **Open http://localhost:8080** in your browser
3. **Try example prompts**:
   - "What's the weather like in Paris?" (uses fake weather tool)
   - "Search for information about Python" (uses fake search tool)
   - "Analyze the sentiment of this text: I love this project!"

### Web Interface Architecture

```
Browser ←→ MonsterUI Web Client ←→ Reasoning Agent API ←→ MCP Servers
        (Port 8080)              (Port 8000)         (Port 8001+)
```

## Development

### Development Workflow

#### Docker Development
```bash
# Start everything
make docker_up

# View logs
make docker_logs

# Rebuild after changes
make docker_restart

# Stop everything
make docker_down
```

#### Local Development
```bash
# Install dependencies (one time setup)
make dev                # Install all dependencies for development

# Start individual services
make api                    # API server
make web_client             # Web interface  
make demo_mcp_server        # MCP tools

# Run tests
make tests                  # All tests
make non_integration_tests  # Fast tests only
```

### Adding New MCP Servers

1. **Create MCP server** (see `mcp_servers/fake_server.py` example)
2. **Add to Docker Compose**:
   ```yaml
   new-mcp-server:
     build:
       context: .
       dockerfile: Dockerfile.new-mcp
     ports:
       - "8002:8002"
     networks:
       - reasoning-network
   ```
3. **Update MCP configuration** in `config/mcp_servers.json`

    The reasoning agent uses configurable MCP server connections (e.g `config/mcp_servers.json`). Configuration files specify which MCP servers to connect to and their settings.

    Set a custom MCP configuration file using the MCP_CONFIG_PATH environment variable (e.g. `MCP_CONFIG_PATH=my_config.json`)

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

The project uses a unified `.env` file for all services:

```bash
# Required
OPENAI_API_KEY=your-openai-api-key-here
API_TOKENS=web-client-dev-token,admin-dev-token,mobile-dev-token
REASONING_API_TOKEN=web-client-dev-token
REQUIRE_AUTH=false  # true for production

# Service Configuration
REASONING_API_URL=http://localhost:8000  # Web client → API
WEB_CLIENT_PORT=8080
API_PORT=8000
MCP_SERVER_PORT=8001

# MCP Configuration
MCP_CONFIG_PATH=config/mcp_servers.json  # Path to MCP configuration file

# Optional HTTP Settings
HTTP_CONNECT_TIMEOUT=5.0
HTTP_READ_TIMEOUT=30.0
HTTP_WRITE_TIMEOUT=10.0
```

### MCP Server Configuration

Configure MCP servers in `config/mcp_servers.json` using the standard JSON format:

```json
{
  "mcpServers": {
    "demo_tools": {
      "url": "http://localhost:8001/mcp/",
      "transport": "http"
    },
    "production_tools": {
      "url": "https://your-mcp-server.com/mcp/", 
      "auth_env_var": "MCP_API_KEY",
      "transport": "http"
    }
  }
}
```

**Environment Variables:**
- `MCP_CONFIG_PATH`: Path to MCP configuration file (default: `config/mcp_servers.json`)

Example usage:
```bash
# Use custom MCP configuration
MCP_CONFIG_PATH=config/production_mcp.json make api

# Use demo configuration  
MCP_CONFIG_PATH=examples/configs/demo_complete.json make api
```

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
uv python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Testing

### Test Commands

```bash
# Full test suite
make tests                      # Linting + all tests

# Test variations  
make non_integration_tests      # Fast tests (no OpenAI API)
make integration_tests          # Full integration (requires OPENAI_API_KEY)
make linting                    # Code formatting only

# Docker testing
make docker_test                # Run tests in container
```

### Test Structure

- **Unit Tests**: `tests/test_*.py` (API, models, reasoning logic)
- **Integration Tests**: Marked with `@pytest.mark.integration`
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

### Inspector

- `Inspector` can be started via `npx @modelcontextprotocol/inspector`
- Example: start demo MCP server locally (either via Docker or directly with `make demo_mcp_server`)
- Set `Transport Type` to `Streamable HTTP`
- type `http://0.0.0.0:8001/mcp/` into the URL and click `Connect`
    - Do not forget to add `/mcp/` at the end e.g. `https://your-fake-server.onrender.com/mcp/`
- Go to `Tools` and click `List Tools` to see available MCP tools
- Select a tool and test it out.


### Monitoring and Health Checks

All services include health endpoints:

```bash
# Check service health
curl http://localhost:8000/health  # API
curl http://localhost:8080/health  # Web Client
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

- **Docker Setup**: [README_DOCKER.md](README_DOCKER.md) - Detailed Docker instructions
- **MCP Inspector**: Use `npx @modelcontextprotocol/inspector` to test MCP servers
- **OpenAI API Docs**: [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- **FastMCP**: [FastMCP Documentation](https://github.com/jlowin/fastmcp) for building MCP servers
