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

# Evaluations (LLM behavioral testing with flex-evals)
make evaluations                                       # Run all LLM behavioral evaluations 
python tests/evaluations/run_evaluations.py -k weather # Run specific evaluation
python tests/evaluations/example_manual_run.py        # Manual evaluation example
```

### Environment Setup
- Uses `uv` package manager with Python 3.13+
- Configuration via `.env` file (copy from `examples/.env.example`)
- Required: `OPENAI_API_KEY` for integration tests
- Authentication: `API_TOKENS` (comma-separated) + `REQUIRE_AUTH=true`

### Test Organization
- **Non-integration tests**: Fast, no external dependencies (`tests/test_*.py`)
- **Integration tests**: Marked with `@pytest.mark.integration`, auto-start servers
- **Evaluations**: LLM behavioral testing with real API calls (`tests/evaluations/`)
- Use `make non_integration_tests` for rapid development feedback
- Integration tests and evaluations require `OPENAI_API_KEY` environment variable

## Architecture Overview

### Infrastructure

**PostgreSQL Databases**:
- **postgres-phoenix** (port 5432) - Phoenix Arize observability data
- **postgres-litellm** (port 5433) - LiteLLM virtual keys and usage tracking
- **postgres-reasoning** (port 5434) - Conversation storage (Milestone 1 complete)
  - Tables: `conversations`, `messages`
  - Managed via Alembic migrations: `uv run alembic upgrade head`
  - Connection: `postgresql+asyncpg://reasoning_user:password@localhost:5434/reasoning`
  - Status: Schema ready, API integration pending (M2-M3)

**Database Migrations**:
- Tool: Alembic with async asyncpg support
- Location: `alembic/versions/`
- Configuration: `alembic.ini` (default connection) and `alembic/env.py` (async runtime)
- Environment variable override: `REASONING_DATABASE_URL`
- Run migrations: `uv run alembic upgrade head`

### Reasoning Model Support

The API provides first-class support for native reasoning models (OpenAI o1/o3/GPT-5, Anthropic Claude 3.7+, DeepSeek) through OpenAI-compatible interfaces:

**Request Parameter:**
- `reasoning_effort`: Controls model's internal reasoning depth
- Values: `"minimal"`, `"low"`, `"medium"`, `"high"`
- Optional parameter compatible with all models
- Passed through to LiteLLM for provider-specific handling

**Model Discovery:**
- `/v1/models` endpoint includes `supports_reasoning` field for each model
- Uses `litellm.supports_reasoning()` for dynamic detection
- Helps clients determine which models support the `reasoning_effort` parameter

**Reasoning Content Conversion:**

Models with native reasoning capabilities (Anthropic Claude 3.7+, DeepSeek) return reasoning content separately from their final response. The API automatically converts this to a unified event format:

1. **Provider Behavior:** Models stream `reasoning_content` chunks (hidden thinking)
2. **API Buffering:** All reasoning chunks are buffered internally
3. **Event Emission:** When reasoning completes, API emits single `EXTERNAL_REASONING` event
4. **Content Streaming:** Regular content streaming continues after reasoning event

**Response Format:**

For models with native reasoning, you receive a `reasoning_event` in the delta before regular content:

```json
{
  "choices": [{
    "delta": {
      "reasoning_event": {
        "type": "external_reasoning",
        "step_iteration": 1,
        "metadata": {
          "thought": "Full reasoning content...",
          "provider": "anthropic"
        }
      }
    }
  }]
}
```

Then regular content follows in subsequent chunks.

**OpenAI Models:**
- No separate reasoning content (reasoning may appear inline in response)
- No reasoning events emitted
- `reasoning_effort` still controls internal reasoning depth

---

### Core Components

**FastAPI Application** (`api/main.py`):
- OpenAI-compatible `/v1/chat/completions` endpoint
- **Intelligent request routing** to three execution paths:
  - **A) Passthrough**: Direct OpenAI API (default, lowest latency)
  - **B) Reasoning**: Single-loop reasoning agent (baseline, manual selection)
  - **C) Orchestration**: Multi-agent coordination (A2A protocol, future)
- `/v1/models` endpoint - Dynamic model discovery via LiteLLM proxy
  - Fetches available models from LiteLLM's `/v1/models` endpoint
  - Returns proper HTTP errors (503/500) if LiteLLM is unavailable
  - Enables clients to discover available models dynamically
- Dependency injection architecture for testability
- Bearer token authentication system
- Health check endpoint

**Request Routing** (`api/request_router.py`):
- Three-tier routing strategy for intelligent query classification
- **Tier 1**: Explicit passthrough rules (`response_format`, `tools` → force passthrough)
- **Tier 2**: Header-based routing via `X-Routing-Mode` header
  - Values: `passthrough`, `reasoning`, `orchestration`, `auto` (case-insensitive)
  - Manual selection for testing and explicit control
- **Tier 3**: Default behavior → passthrough (matches OpenAI experience)
- **Auto-routing**: `X-Routing-Mode: auto` → LLM classifier (GPT-4o-mini) chooses passthrough or orchestration
- **Note**: LLM classifier never chooses reasoning mode (manual-only for baseline testing)

**Passthrough Path** (`api/passthrough.py`):
- Direct LLM API call via litellm for straightforward queries (Route A)
- Default execution path when no header provided (matches OpenAI experience)
- Fast, low-latency responses without reasoning or orchestration overhead
- Full streaming and non-streaming support
- OpenTelemetry tracing and error forwarding
- Uses `litellm.acompletion()` for unified multi-provider support

**Reasoning Path** (`api/reasoning_agent.py`):
- Single-loop reasoning agent accessible via `X-Routing-Mode: reasoning` (Route B)
- Baseline for comparison with multi-agent orchestration
- Manual selection only (LLM classifier never chooses this path)
- Proxies requests via litellm while injecting reasoning steps
- Supports both streaming and non-streaming responses
- Injects visual reasoning steps (🔍, 🤔, ✅) before actual LLM response
- Uses `litellm.acompletion()` with structured JSON outputs
- Will be migrated to A2A service architecture in M6

**Service Container** (`api/dependencies.py`):
- Manages HTTP client lifecycle with connection pooling
- Configures timeouts and connection limits via environment
- Provides dependency injection for ReasoningAgent and PromptManager
- LiteLLM handles connection pooling internally via `acompletion()`

**LiteLLM Integration** (`api/config.py`, all execution paths):
- **Unified LLM Gateway**: All LLM API calls use `litellm.acompletion()` for multi-provider support
- **Built-in Connection Pooling**: LiteLLM manages HTTP connections internally
- **Virtual Keys**: Environment-specific API keys (`dev`, `eval`, `prod`) with unlimited budgets
- **OTEL Trace Propagation**: W3C TraceContext headers (`traceparent`) propagated via `extra_headers`
  - Used in: `passthrough.py`, `reasoning_agent.py`, `request_router.py` (classifier)
  - Pattern: `carrier = {}; propagate.inject(carrier); litellm.acompletion(..., extra_headers=carrier)`
- **Environment Variables**:
  - `LITELLM_API_KEY` - Virtual key for current environment (dev/eval/prod)
  - `LITELLM_BASE_URL` - LiteLLM proxy endpoint (default: `http://localhost:4000`)
  - `ROUTING_CLASSIFIER_MODEL` - Model for auto-routing (default: `gpt-4o-mini`)
- **Configuration**: `litellm-config.yaml` defines virtual keys, models, and rate limits
- **Direct Function Calls**: Uses `await litellm.acompletion()` instead of AsyncOpenAI client
- **No Direct OpenAI Calls**: All LLM interactions go through LiteLLM proxy

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

### Request Routing Configuration
Environment variables control routing behavior:
- `ROUTING_CLASSIFIER_MODEL` (default: `gpt-4o-mini`) - Model for auto-routing classification
- `ROUTING_CLASSIFIER_TEMPERATURE` (default: `0.0`) - Temperature for classifier (deterministic)

**Routing Header (`X-Routing-Mode`):**
- `passthrough` - Force Route A (direct OpenAI API, lowest latency)
- `reasoning` - Force Route B (single-loop reasoning agent, baseline)
- `orchestration` - Force Route C (multi-agent coordination, 501 stub until M3-M4)
- `auto` - Use LLM classifier to choose between passthrough or orchestration
- Case-insensitive (e.g., "PASSTHROUGH", "Passthrough", "passthrough" all work)

**Other Headers:**
- `X-Session-ID` - Session identifier for tracing correlation

**Routing Behavior (evaluated in priority order):**
1. **Passthrough Rules (highest priority)**: Requests with `response_format` or `tools` → always passthrough
2. **Header Override**: `X-Routing-Mode` header → explicit path selection
3. **Auto-routing**: `X-Routing-Mode: auto` → LLM classifier chooses passthrough or orchestration
4. **Default**: No header provided → passthrough (matches OpenAI experience)

**Important Notes:**
- LLM classifier only chooses between passthrough and orchestration (never reasoning)
- Reasoning path is manual-only for baseline testing and comparison
- Orchestration path returns 501 Not Implemented until M3-M4 implementation

### Tools API Integration

The reasoning-api integrates with the tools-api service for structured tool execution:

**Architecture:**
```
reasoning-api (Docker) → tools-api (Docker) → direct implementations
                            ↓
                      volume mounts (/repos, /downloads)
```

**Tools API Service** (`tools-api/`):
- FastAPI REST service providing tools and prompts via HTTP
- Direct tool implementations using pathlib, httpx, subprocess
- Returns structured JSON responses with metadata
- Path security with blocked patterns (`.git/`, `node_modules/`, `.env`, etc.)
- Volume mounts for filesystem access (`/mnt/repos`, `/mnt/downloads`, `/mnt/workspace`)

**Integration Points:**
- Tools discovery: `GET http://tools-api:8001/tools/`
- Tool execution: `POST http://tools-api:8001/tools/{tool_name}`
- Prompts: `GET http://tools-api:8001/prompts/`
- Health check: `GET http://tools-api:8001/health`

**Benefits:**
- Structured responses (JSON with metadata, execution time, file stats)
- Simple Docker Compose deployment (no separate bridge process)
- Better testability (mock HTTP endpoints)
- Direct implementations (no protocol translation layers)
- Improved observability (standard HTTP logs)

**Configuration:**
- `TOOLS_API_URL` - Tools API endpoint (default: `http://tools-api:8001`)
- Volume mounts configured in `docker-compose.yml`
- API keys (GitHub, Brave Search) configured via environment variables

**Available Tools:**
- **Filesystem**: Read/write files, list directories, search, file info
- **GitHub**: PR info, local git changes, directory tree
- **Web Search**: Brave Search API integration
- See tools-api documentation for complete list

---

## Document Style Guidelines

**When updating documentation:**
- Remove outdated information cleanly - don't add "No longer doing X" or "Changed from Y to Z"
- Simply present the current approach as if writing it fresh
- The reader doesn't have context for historical decisions - don't create confusion
- Update sections to reflect current reality, not revision history
- If something changed, just describe the new approach definitively
- No "we decided to change..." or "instead of..." phrasing
- Keep it clean: what ARE we doing, not what we're NOT doing anymore

**Goal:** A reader seeing documentation for the first time should get a clear picture of the current approach without noise about past iterations.
