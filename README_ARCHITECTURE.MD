# Architecture Documentation

This document provides a comprehensive overview of the Reasoning Agent API architecture, including the request routing system, LiteLLM proxy integration, and core components.

## Table of Contents

- [System Overview](#system-overview)
- [Request Routing Architecture](#request-routing-architecture)
- [LiteLLM Proxy Integration](#litellm-proxy-integration)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Service Architecture](#service-architecture)

## System Overview

The Reasoning Agent API is an OpenAI-compatible service that provides intelligent request routing between three execution paths:

1. **Passthrough**: Direct OpenAI API forwarding (default, lowest latency)
2. **Reasoning**: Single-loop reasoning agent with visual thinking steps
3. **Orchestration**: Multi-agent coordination using A2A protocol (future)

All LLM requests flow through a unified LiteLLM gateway for centralized observability, connection pooling, and usage tracking.

### High-Level Architecture

```
+-------------+     +----------------------+     +-------------+     +-------------+
|   Browser   | --> | Reasoning Agent API  | --> |   LiteLLM   | --> |   OpenAI    |
|             |     |   (Port 8000)        |     |   Proxy     |     |     API     |
+-------------+     +-----------+----------+     | (Port 4000) |     +-------------+
                                |                 +-------------+
                                |
                                v
                    +----------------------+
                    |    MCP Servers       |
                    |   (Ports 8001+)      |
                    +-----------+----------+
                                |
                                v
                    +----------------------+
                    |      Phoenix         |
                    |   (OTEL Tracing)     |
                    |   (Port 6006)        |
                    +----------------------+
```

## Request Routing Architecture

The API uses a **three-tier routing strategy** to determine which execution path to use for each request.

### Routing Decision Flow

```
+-------------------------------------------------------------+
|                    Incoming Request                         |
+-------------------------+-----------------------------------+
                          |
                          v
        +-----------------------------+
        |   Tier 1: Passthrough Rules |
        |                             |
        |  - Has response_format?     | ----> Yes ---> Passthrough
        |  - Has tools parameter?     | ----> Yes ---> Passthrough
        +-------------+---------------+
                      | No
                      v
        +-----------------------------+
        |   Tier 2: Header Override   |
        |                             |
        |  Check X-Routing-Mode:      |
        |  - passthrough              | ----> Passthrough
        |  - reasoning                | ----> Reasoning
        |  - orchestration            | ----> Orchestration
        |  - auto                     | ----> Auto-routing (Tier 3)
        +-------------+---------------+
                      | No header
                      v
        +-----------------------------+
        |   Tier 3: Default Behavior  |
        |                             |
        |  No explicit routing        | ----> Passthrough (default)
        +-----------------------------+
```

### Route A: Passthrough

**Purpose**: Direct OpenAI API forwarding with minimal overhead

**Implementation**: `api/passthrough.py`

**Key Features**:
- Full OpenAI API compatibility
- Streaming and non-streaming support
- OTEL trace propagation to Phoenix
- Connection pooling via shared AsyncOpenAI client
- Lowest latency path (~40-50ms proxy overhead)

**Use Cases**:
- Simple chat completions
- Structured outputs (`response_format`)
- Function/tool calling
- Production workloads requiring low latency

**Request Flow**:
```
Client Request -> Reasoning API -> LiteLLM Proxy -> OpenAI API
                                        |
                                        v
                                   Phoenix OTEL
```

### Route B: Reasoning Agent

**Purpose**: Single-loop reasoning with visual thinking steps

**Implementation**: `api/reasoning_agent.py`

**Key Features**:
- Visual reasoning steps (🔍 analyzing, 🤔 thinking, ✅ completed)
- Tool usage via MCP protocol
- Streaming synthesis with reasoning events
- Manual selection only (baseline for orchestration comparison)

**Use Cases**:
- Demonstrating AI thinking process
- Debugging reasoning logic
- Baseline measurements for orchestration
- Educational/transparency purposes

**Request Flow**:
```
Client Request -> Reasoning Agent -> Step Generation -> Tool Calls (MCP)
                                          |                 |
                                          v                 v
                                    LiteLLM Proxy    MCP Servers
                                          |
                                          v
                                   Synthesis -> Stream to Client
```

### Route C: Orchestration (Future - M3-M4)

**Purpose**: Multi-agent coordination via A2A protocol

**Status**: Currently returns 501 Not Implemented

**Planned Features**:
- Task decomposition
- Multi-agent collaboration
- Complex research queries
- Autonomous planning and execution

**Auto-Routing Classifier**:
- Model: GPT-4o-mini (configurable via `ROUTING_CLASSIFIER_MODEL`)
- Temperature: 0.0 (deterministic)
- Classification: Decides between **passthrough** or **orchestration**
- Note: Never chooses reasoning (manual-only for baseline testing)

## LiteLLM Proxy Integration

### Architecture

The LiteLLM proxy serves as a unified gateway for all LLM API calls, providing centralized observability and virtual key management.

```
+-------------------------------------------------------------+
|                      LLM Request Flow                       |
+-------------------------------------------------------------+

Application Code -> LITELLM_API_KEY (virtual) -> LiteLLM Proxy -> OPENAI_API_KEY (real) -> OpenAI API
                                                        |
                                                        v
                                                  Phoenix OTEL
                                                        |
                                                        v
                                                PostgreSQL (litellm)
```

### Key Concepts

**Virtual Keys**:
- Environment-specific API keys (dev, eval, prod)
- Tracked independently in LiteLLM dashboard
- Unlimited budgets by default (no rate limiting)
- Generated via `make litellm_setup`

**Real Key**:
- Single OpenAI API key used by LiteLLM proxy
- Never used directly by application code
- Set via `OPENAI_API_KEY` environment variable

**Connection Pooling**:
- Single `AsyncOpenAI` client instance created at app startup
- Managed by `ServiceContainer` in `api/dependencies.py`
- Shared across all requests via dependency injection
- Efficient connection reuse to LiteLLM proxy

**Component Metadata**:
- OTEL metadata tags LLM calls by component:
  - `reasoning-agent`: Step generation and synthesis
  - `passthrough`: Direct API forwarding
  - `routing-classifier`: Auto-routing classification
- Enables filtering in Phoenix dashboard

**OTEL Trace Propagation**:
- W3C TraceContext headers (`traceparent`) propagated to LiteLLM
- Uses `opentelemetry.propagate.inject()` to inject trace context
- Unified parent-child trace visualization in Phoenix

### Configuration

**LiteLLM Service** (`config/litellm_config.yaml`):
```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      num_retries: 3
      timeout: 600

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
  store_model_in_db: true

litellm_settings:
  callbacks: ["otel"]
  success_callback: ["otel"]
  failure_callback: ["otel"]
```

**Application Configuration** (`api/config.py`):
```python
class Settings(BaseSettings):
    llm_api_key: str  # Virtual key from LiteLLM
    llm_base_url: str = "http://litellm:4000"  # LiteLLM proxy endpoint
```

### Database Architecture

**PostgreSQL for LiteLLM** (`postgres-litellm`):
- Port: 5433 (external), 5432 (internal)
- Database: `litellm`
- User: `litellm_user`
- Purpose: Virtual key storage, usage tracking, model definitions

**PostgreSQL for Phoenix** (`postgres-phoenix`):
- Port: 5432 (external/internal)
- Database: `phoenix`
- User: `phoenix_user`
- Purpose: OTEL trace data storage

## Core Components

### FastAPI Application (`api/main.py`)

**Key Responsibilities**:
- OpenAI-compatible `/v1/chat/completions` endpoint
- Request routing via `determine_routing()`
- Dependency injection for services
- Bearer token authentication
- Health checks and tools listing

**Main Endpoint**:
```python
@app.post("/v1/chat/completions")
async def chat_completions(
    request: OpenAIChatRequest,
    openai_client: Annotated[AsyncOpenAI, Depends(get_openai_client)],
    reasoning_agent: Annotated[ReasoningAgent, Depends(get_reasoning_agent)],
    # ...
):
    # Determine routing path
    routing_decision = await determine_routing(request, headers, openai_client)

    # Execute based on routing
    if routing_decision.path == "passthrough":
        return execute_passthrough_stream(request, openai_client, ...)
    elif routing_decision.path == "reasoning":
        return reasoning_agent.process_request(request, ...)
    elif routing_decision.path == "orchestration":
        raise HTTPException(status_code=501, detail="Not Implemented")
```

### Service Container (`api/dependencies.py`)

**Purpose**: Centralized lifecycle management for shared services

**Managed Services**:
- `AsyncOpenAI` client (LiteLLM proxy connection)
- `httpx.AsyncClient` (HTTP connection pooling for MCP)
- `MCPClient` (MCP server connections)
- `PromptManager` (reasoning prompts)

**Lifecycle**:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize services
    await service_container.initialize()
    yield
    # Shutdown: Cleanup services
    await service_container.cleanup()
```

### Request Router (`api/request_router.py`)

**Purpose**: Intelligent routing decision logic

**Three-Tier Strategy**:
1. **Tier 1**: Explicit passthrough rules (structured outputs, tools)
2. **Tier 2**: Header-based routing (`X-Routing-Mode`)
3. **Tier 3**: Default behavior (passthrough)

**Auto-Routing Classifier**:
- Uses structured outputs with Pydantic schema
- Classification prompt optimized for passthrough vs orchestration
- Temperature 0.0 for deterministic results
- Model configurable via `ROUTING_CLASSIFIER_MODEL`

### Passthrough (`api/passthrough.py`)

**Purpose**: Direct OpenAI API forwarding

**Key Functions**:
- `execute_passthrough_stream()`: Streaming responses
- OTEL trace propagation via `extra_headers`
- Component metadata via `extra_body`
- Client disconnect handling

### Reasoning Agent (`api/reasoning_agent.py`)

**Purpose**: Single-loop reasoning with tool usage

**Key Methods**:
- `process_request()`: Main entry point for reasoning path
- `_generate_reasoning_step()`: Generate individual reasoning steps
- `_stream_final_synthesis()`: Synthesize and stream final answer
- `_call_tool()`: Execute MCP tool calls

**Reasoning Loop**:
1. Analyze query
2. Identify required tools
3. Execute tool calls via MCP
4. Synthesize final answer
5. Stream with reasoning events

## Data Flow

### Passthrough Path

```
1. Client Request
   |
   v
2. FastAPI Endpoint (/v1/chat/completions)
   |
   v
3. Authentication (Bearer Token)
   |
   v
4. Request Routing (determine_routing)
   |  [Tier 1: No tools/response_format]
   v
5. execute_passthrough_stream()
   |  [Inject trace headers]
   v
6. AsyncOpenAI Client (shared instance)
   |  [HTTP connection pool]
   v
7. LiteLLM Proxy (virtual key auth)
   |  [OTEL callbacks]
   v
8. OpenAI API (real key)
   |
   v
9. Stream Response -> Client
   |
   v
10. Phoenix Trace (OTEL collector)
```

### Reasoning Path

```
1. Client Request
   |
   v
2. FastAPI Endpoint
   |
   v
3. Authentication
   |
   v
4. Request Routing
   |  [X-Routing-Mode: reasoning]
   v
5. ReasoningAgent.process_request()
   |
   v
6. Generate Reasoning Steps
   +-- _generate_reasoning_step() -> LiteLLM -> OpenAI
   +-- Identify Tool Calls
   +-- _call_tool() -> MCP Server
   +-- Repeat until complete
   |
   v
7. _stream_final_synthesis()
   +-- Stream reasoning events (🔍, 🤔, ✅)
   +-- Stream LLM response -> Client
   |
   v
8. Phoenix Trace (component: reasoning-agent)
```

### Auto-Routing Path

```
1. Client Request
   |
   v
2. Request Routing
   |  [X-Routing-Mode: auto]
   v
3. _classify_with_llm()
   +-- Classification Prompt
   +-- AsyncOpenAI Client -> LiteLLM -> OpenAI
   +-- Structured Output (Pydantic schema)
   |
   v
4. Classification Result
   +-- passthrough -> Route A
   +-- orchestration -> Route C (501 until M3-M4)
   |
   v
5. Execute Selected Path
```

## Service Architecture

### Docker Compose Services

**Application Services**:
- `reasoning-api`: Main API service (Port 8000)
- `web-client`: MonsterUI web interface (Port 8080)
- `fake-mcp-server`: Demo MCP tools (Port 8001)

**Infrastructure Services**:
- `litellm`: LiteLLM proxy gateway (Port 4000)
- `postgres-litellm`: LiteLLM database (Port 5433)
- `phoenix`: OTEL collector and UI (Port 6006)
- `postgres-phoenix`: Phoenix database (Port 5432)

### Service Dependencies

```
reasoning-api depends_on:
  - fake-mcp-server (condition: service_started)
  - litellm (condition: service_healthy)

web-client depends_on:
  - reasoning-api

litellm depends_on:
  - postgres-litellm (condition: service_healthy)
  - phoenix (condition: service_healthy)

phoenix depends_on:
  - postgres-phoenix (condition: service_healthy)
```

### Network Architecture

**Docker Network**: `reasoning-network` (bridge)

**Internal Communication**:
- `reasoning-api` -> `litellm` (http://litellm:4000)
- `reasoning-api` -> `fake-mcp-server` (http://fake-mcp-server:8001)
- `reasoning-api` -> `phoenix` (http://phoenix:4317 for OTEL)
- `web-client` -> `reasoning-api` (http://reasoning-api:8000)
- `litellm` -> `postgres-litellm` (postgresql://postgres-litellm:5432)
- `litellm` -> `phoenix` (http://phoenix:4317 for OTEL)
- `phoenix` -> `postgres-phoenix` (postgresql://postgres-phoenix:5432)

### Health Checks

All services include health check endpoints:
- `reasoning-api`: `GET /health`
- `web-client`: `GET /health`
- `fake-mcp-server`: `GET /`
- `litellm`: `GET /health/readiness`
- `phoenix`: Built-in health check
- `postgres-*`: `pg_isready` command

## Performance Characteristics

### Latency Overhead

- **Passthrough**: OpenAI latency + ~40-50ms (LiteLLM proxy overhead)
- **Reasoning**: Multiple LLM calls + tool execution + synthesis
- **Auto-routing**: Classification call (~200-500ms) + selected path overhead

### Connection Pooling

**AsyncOpenAI Client**:
- Single instance per application lifecycle
- Shared via dependency injection
- HTTP connection reuse to LiteLLM proxy

**HTTP Client (MCP)**:
- Max connections: 20 (configurable)
- Max keepalive connections: 5
- Keepalive expiry: 30 seconds

### Timeout Configuration

- Connection timeout: 5.0 seconds
- Read timeout: 60.0 seconds (streaming)
- Write timeout: 10.0 seconds

## Observability

### Phoenix Integration

**Trace Collection**:
- OTEL gRPC endpoint: `http://phoenix:4317`
- Automatic span creation for all LLM calls
- Component metadata for filtering
- Unified trace trees via context propagation

**Dashboard Features**:
- Request traces with full context
- Latency analysis (P50/P95/P99)
- Token usage and costs
- Error tracking
- Tool usage patterns

### LiteLLM Dashboard

**Features**:
- Virtual key management
- Usage metrics per key
- Model configuration
- Request logs
- Error tracking

**Access**: http://localhost:4000

## Security

### Authentication

**Bearer Token System**:
- Multiple tokens supported (`API_TOKENS`)
- Token validation in `api/auth.py`
- Configurable via `REQUIRE_AUTH` flag
- Development: Auth disabled (`REQUIRE_AUTH=false`)
- Production: Auth required (`REQUIRE_AUTH=true`)

### API Key Separation

**Virtual Keys** (Application):
- Used by application code
- Environment-specific tracking
- Revocable without affecting other environments

**Real Key** (LiteLLM Only):
- Never exposed to application code
- Single point of access control
- Rotated independently

### Network Isolation

**Docker Network**:
- Internal services not exposed externally
- Only necessary ports published
- Service-to-service communication isolated

## Future Architecture

### Orchestration Path (M3-M4)

**Planned Components**:
- A2A (Agent-to-Agent) protocol integration
- Task decomposition engine
- Multi-agent coordinator
- State management for complex workflows
- Advanced reasoning strategies

**Architecture Changes**:
- New `api/orchestration.py` module
- A2A service integration
- Enhanced routing classifier prompts
- Orchestration-specific observability

### Scaling Considerations

**Horizontal Scaling**:
- Stateless API design (ready for multiple instances)
- Shared LiteLLM proxy (connection pooling)
- PostgreSQL databases (already externalized)

**Performance Optimizations**:
- Redis caching for classification results
- Response streaming for all paths
- Adaptive timeout configuration
- Circuit breakers for MCP services

---

## Additional Resources

- **Main README**: [README.md](README.md)
- **Docker Setup**: [README_DOCKER.md](README_DOCKER.md)
- **Phoenix Setup**: [README_PHOENIX.md](README_PHOENIX.md)
- **Implementation Plans**: [docs/implementation_plans/](docs/implementation_plans/)
