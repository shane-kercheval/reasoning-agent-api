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

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd reasoning-agent-api

# Set your OpenAI API key; or use a .env file
export OPENAI_API_KEY="your-openai-api-key-here"
```

### Running the Server

```bash
# Start the API server
make api
# or: uv run uvicorn api.main:app --reload --port 8000
```

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

### Demo Script

A complete demo script is available:

```bash
# Run the demo (requires server to be running)
python demo_openai_sdk.py
```

This demonstrates:
- Non-streaming chat completions
- Streaming with reasoning steps  
- Models listing
- Error handling

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

## License

MIT License
