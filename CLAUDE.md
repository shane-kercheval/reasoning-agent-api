# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Core Development Commands
```bash
# Main test command - runs linting + all tests (recommended for CI/commits)
make tests

# Development workflow commands
make linting                  # Run ruff linting/formatting
make non_integration_tests    # Fast unit tests (no external dependencies)
make integration_tests        # Full integration tests (requires OPENAI_API_KEY)
make unit_tests              # All tests (non-integration + integration)

# Start API server
make api                     # Start on localhost:8000
# Alternative: uv run python -m api.main
```

### Environment Setup
- Uses `uv` package manager with Python 3.13+
- Configuration via `.env` file (copy from `examples/.env.example`)
- Required: `OPENAI_API_KEY` for integration tests
- Authentication: `API_TOKENS` (comma-separated) + `REQUIRE_AUTH=true`

### Test Organization
- **Non-integration tests**: Fast, no external dependencies (`tests/test_*.py`)
- **Integration tests**: Marked with `@pytest.mark.integration`, auto-start servers
- Use `make non_integration_tests` for rapid development feedback
- Integration tests require `OPENAI_API_KEY` environment variable

## Architecture Overview

### Core Components

**FastAPI Application** (`api/main.py`):
- OpenAI-compatible `/v1/chat/completions` endpoint
- Dependency injection architecture for testability
- Bearer token authentication system
- Health check and tools listing endpoints

**ReasoningAgent** (`api/reasoning_agent.py`):
- Proxies requests to OpenAI API while injecting reasoning steps
- Supports both streaming and non-streaming responses
- Injects visual reasoning steps (🔍, 🤔, ✅) before actual LLM response
- Uses dependency-injected HTTP client and optional MCP client

**Service Container** (`api/dependencies.py`):
- Manages HTTP client lifecycle with connection pooling
- Configures timeouts and connection limits via environment
- Provides dependency injection for ReasoningAgent and MCPClient

**Models** (`api/models.py`):
- Pydantic models for OpenAI API compatibility
- Request/response validation for chat completions
- Streaming response chunk definitions

### Key Design Patterns

1. **Dependency Injection**: All components use FastAPI's dependency system for better testability
2. **Service Container**: Centralized management of HTTP clients and MCP connections
3. **OpenAI Compatibility**: Drop-in replacement for OpenAI SDK with added reasoning
4. **Streaming Enhancement**: Injects reasoning steps before actual LLM response in streaming mode

### Authentication Flow
- Optional bearer token authentication via `API_TOKENS` environment variable
- Tokens verified in `auth.py` dependency
- Can be disabled with `REQUIRE_AUTH=false` for development

### HTTP Client Configuration
Environment variables control HTTP behavior:
- `HTTP_CONNECT_TIMEOUT`, `HTTP_READ_TIMEOUT`, `HTTP_WRITE_TIMEOUT`
- `HTTP_MAX_CONNECTIONS`, `HTTP_MAX_KEEPALIVE_CONNECTIONS`
- `HTTP_KEEPALIVE_EXPIRY`

### MCP Integration
- Optional Model Context Protocol client for tool integration
- Placeholder architecture for future tool enhancement
- Tools listing available at `/tools` endpoint