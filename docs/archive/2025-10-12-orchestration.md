# Implementation Plan: Multi-Agent Orchestration with A2A Protocol

## Project Overview

### Current State
- Single FastAPI service with embedded ReasoningAgent
- OpenAI-compatible `/v1/chat/completions` endpoint
- MCP protocol for tool integration (weather, search, etc.)
- OpenTelemetry tracing to Phoenix
- All state in-memory (lost on restart)
- No multi-agent coordination
- No persistent sessions for pause/resume workflows
- No human-in-the-loop approval workflows
- Running via docker-compose (infrastructure services) + local process (API)

### Target State
- **Three execution paths**: Passthrough (default), Reasoning (baseline), Orchestration (multi-agent)
- **OpenAI-compatible external API** maintaining industry standard interface
- **A2A protocol** for agent-to-agent communication (orchestration path only)
- **Orchestrator agent** that plans and coordinates multi-agent workflows
- **Multiple specialized agents** (planning, RAG, search) communicating via A2A
- **Reasoning agent** remains accessible for baseline comparison (manual selection)
- **Persistent session management** with Redis for long-running workflows
- **Human-in-the-loop workflow support** via A2A `auth-required` state
- **Dynamic agent discovery** via A2A Agent Cards
- **MCP protocol maintained** for tool integration (agents use MCP for external tools)
- **Vector database** (PostgreSQL + pgvector) for RAG semantic search
- **Docker-compose for infrastructure** (Redis, PostgreSQL, Phoenix)
- **Services run as separate FastAPI apps** (main API, orchestrator, planning agent, RAG agent)

### Key Architectural Decisions

**Separate Services Architecture:**
- Each agent is an independent FastAPI service running on its own port
- Services communicate via HTTP using A2A protocol
- Each service can be developed, deployed, scaled, and restarted independently
- Clear separation of concerns: main API (user-facing), orchestrator (coordination), specialized agents (capabilities)
- No shared state between services except via Redis (sessions) and PostgreSQL (vectors)
- Services discover each other via Agent Cards (HTTP GET to /.well-known/agent-card.json)
- This is NOT a monolith with instantiated classes - these are truly separate processes

**Why A2A Protocol for Agent Communication:**
- Industry standard for agent-to-agent communication (not a custom protocol)
- Agent Cards provide automatic capability discovery (`/.well-known/agent-card.json`)
- Built-in human-in-the-loop via `auth-required` task state
- Native Server-Sent Events (SSE) support for real-time streaming
- Task lifecycle management (created → running → auth-required → completed/failed)
- Designed for long-running workflows with pause/resume semantics
- Enables services to communicate regardless of implementation (language, framework)
- Standardized error handling: clear task states
- Complements MCP: A2A for agent-to-agent, MCP for agent-to-tools

**State Management with Redis:**
- Simple key-value operations for session state
- Direct client control and debugging with `redis` library
- Proven, mature technology
- Easy to understand and troubleshoot
- Excellent performance for caching and session storage

**Vector Search with PostgreSQL + pgvector:**
- Native vector search capability via pgvector extension
- Proven relational database with vector support
- Standard `asyncpg` library for async operations
- Good performance for MVP scale
- Easy backup, restore, and operational tooling
- Unified database infrastructure

**Infrastructure with Docker-Compose:**
- Local development environment
- Manages infrastructure services (Redis, PostgreSQL, Phoenix)
- Simple orchestration for development
- Easy to add new services
- Production-ready containerization

**Built-in Quality Practices (Every Milestone):**
- **Observability**: Structured logging with correlation IDs (session_id, task_id), OpenTelemetry tracing across all services, metrics collection in Phoenix
- **Error Handling**: Retries with exponential backoff, timeout handling, circuit breakers for agent calls, graceful degradation, clear error messages
- **Testing**: Unit tests, integration tests, end-to-end tests for each component
- These are not separate "Phase 4" tasks - they're part of development from day 1

**Request Routing Strategy:**
- **Three Execution Paths**:
  - **Passthrough** (majority, default): Main API calls OpenAI/LLM directly, streams back immediately
    - No agents, no A2A protocol, no orchestration overhead
    - Fast path for straightforward queries like "What's 2+2?" or "Summarize this text"
  - **Reasoning** (manual, baseline): Single-loop ReasoningAgent with visual reasoning steps
    - Accessible via `X-Routing-Mode: reasoning` header
    - Serves as baseline for orchestration comparison
  - **Orchestration** (multi-agent): Main API delegates to Orchestrator via A2A protocol
    - Orchestrator uses Planning Agent to generate DAG
    - Orchestrator coordinates multiple agents (RAG, search, planning, etc.)
    - Aggregates results and streams back
- **Dual Protocol**:
  - External: OpenAI API (industry standard)
  - Internal: A2A protocol for agent communication (orchestration path only)
  - Translation layer converts between protocols for orchestration path

---

## A2A Protocol Architecture

### Overview

The system routes requests to one of three execution paths:
- **Passthrough** → direct OpenAI/LLM call (default, fast path)
- **Reasoning** → single-loop ReasoningAgent (manual, baseline)
- **Orchestration** → A2A protocol orchestration (multi-agent coordination)

```
┌─────────────────────────────────────────────────────────────┐
│  External Clients (OpenAI SDK, curl, etc.)                  │
└─────────────┬───────────────────────────────────────────────┘
              │ OpenAI Protocol
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Main API (FastAPI)                                         │
│  - /v1/chat/completions (OpenAI compatible)                 │
│  - Request routing: passthrough / reasoning / orchestration │
│  - X-Routing-Mode header (default: passthrough)             │
│  - Session management (Redis)                               │
└─────────────┬───────────────────────────────────────────────┘
              │
         ┌────┴─────┐
         │ ROUTING  │
         └────┬─────┘
              │
    ┌─────────┼────────────┐
    │         │            │
    │ A)      │ B)         │ C)
    │ Passthrough  Reasoning    Orchestration
    │         │            │
    ▼         ▼            ▼
┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐
│  OpenAI     │  │ Reasoning    │  │  Orchestrator (A2A)    │
│  API        │  │ Agent        │  │  - Agent Card          │
│  Direct     │  │ (Embedded)   │  │  - Planning DAG        │
│  (Default)  │  │ Single-loop  │  │  - Coordinates agents  │
│             │  │ Baseline     │  │  - Aggregates results  │
│             │  │              │  │  - Redis state         │
└─────────────┘  └──────────────┘  └─────────┬──────────────┘
                                              │ A2A Protocol
                                              │
                                   ┌──────────┼──────────┐
                                   ▼ A2A      ▼ A2A      ▼ A2A
                               ┌─────────┐ ┌──────┐ ┌─────────┐
                               │ RAG     │ │Search│ │ Planning│
                               │ Agent   │ │Agent │ │ Agent   │
                               │         │ │      │ │         │
                               │ Card    │ │ Card │ │ Card    │
                               │ Tasks   │ │Tasks │ │ Tasks   │
                               │ SSE     │ │ SSE  │ │ SSE     │
                               └────┬────┘ └──┬───┘ └────┬────┘
                                    │         │          │
                                    │ MCP     │          │ MCP
                                    ▼         ▼          ▼
                               ┌──────────┐ ┌──────────┐ ┌──────────┐
                               │ Vector   │ │ External │ │ MCP      │
                               │ DB       │ │ Search   │ │ Tools    │
                               │(pgvector)│ │ APIs     │ │(Weather) │
                               └──────────┘ └──────────┘ └──────────┘
```

### Key A2A Concepts

**Agent Card (`/.well-known/agent-card.json`):**
- Self-describing JSON document published by each agent
- Declares agent capabilities, endpoint, version
- Enables dynamic discovery (no manual registry needed)
- Planning agent fetches cards to understand available agents

**A2A Task Lifecycle:**
```
created → running → auth-required (pause for approval) → running → completed
                 → failed
                 → canceled
                 → rejected
```

**A2A Task Structure:**
- `task_id`: Unique identifier
- `context_id`: Session/conversation identifier
- `status`: Current lifecycle state
- `messages`: Chat-like message history
- `artifacts`: Output artifacts (text, images, etc.)
- `auth_context`: Approval request details (when auth-required)

**Human-in-the-Loop with A2A:**
- Agent sets task status to `auth-required` with approval context
- Orchestrator pauses workflow, notifies UI
- User reviews and approves via PUT request
- Orchestrator resumes workflow with approval token

**Multi-Agent Streaming:**
- Orchestrator creates multiple A2A tasks in parallel
- Subscribes to SSE streams from each agent
- Aggregates artifacts as they arrive
- Translates to OpenAI streaming format for client

---

## Implementation Milestones

The milestones are ordered by dependency and value delivery. **Key principle: Build DAG orchestration correctly from the start, test with mocks, then add real agents.**

- **Phase 1 (Foundation & Three Paths)**: Get all three execution paths working - immediate value
  - M1: Request Routing + Three Execution Paths (passthrough/reasoning/orchestration)
  - M2: A2A Protocol Foundation (models, client, translation)

- **Phase 2 (Orchestration Path Implementation)**: Build full DAG logic, test with mocks
  - M3: Planning Agent with Full DAG Generation
  - M4: Orchestrator with Full DAG Execution + Session Management
  - M5: Mock Agents + End-to-End Testing (prove orchestration works)

- **Phase 3 (Real Agents)**: Replace mocks with production agents
  - M6: Reasoning Agent as A2A Service (migrate from embedded)
  - M7: Vector DB + RAG Agent
  - M8: Agent Discovery (replace hardcoded endpoints)
  - M9: Human-in-the-Loop

- **Phase 4 (Production Optimization)**: Performance tuning and optimization
  - M10: Performance Optimization

**Note:** Observability (structured logging, tracing, metrics) and error handling (retries, timeouts, circuit breakers) are built into every milestone from the start, not separate phases.

### Phase 1: Foundation & Three Paths

#### Milestone 1: Request Routing + Three Execution Paths

**What:**
Implement intelligent request routing to support three distinct execution paths: **passthrough** (direct OpenAI), **reasoning** (single-loop reasoning agent), and **orchestration** (multi-agent coordination). The routing system uses an LLM classifier (when `auto` mode is requested) or explicit header values to determine the appropriate path.

**Three Execution Paths:**
- **A) Passthrough**: Direct OpenAI API call - no reasoning, no orchestration, lowest latency
- **B) Reasoning**: Single-loop reasoning agent - baseline for comparison, manual selection only
- **C) Orchestration**: Multi-agent coordination with DAG execution - most powerful, future implementation

**Why:**
- Most queries (estimated 80%+) benefit from direct passthrough (fast, low-latency)
- Reasoning agent provides baseline for testing and comparison (manual selection)
- Orchestration path prepared for M3-M4 (not yet implemented)
- Default to passthrough unless explicitly requested or auto-routed
- Maintains full OpenAI compatibility
- Clear separation of concerns for testing and development

**How (High-Level):**

**Routing Module (`api/request_router.py`):**
- Create `RoutingMode` enum: `passthrough`, `reasoning`, `orchestration`, `auto`
- Create `RoutingDecision` Pydantic model with `routing_mode: RoutingMode` and `reason: str`
- Implement `determine_routing()` function with classification logic:
  - **Explicit passthrough rules**: If request has `response_format` or `tools` → always passthrough
  - **Header-based routing**: Check `X-Routing-Mode` header with values: `passthrough`, `reasoning`, `orchestration`, `auto`
  - **Default behavior**: If header not provided → passthrough (matches OpenAI experience)
  - **Auto-routing (LLM classifier)**: If header is `auto` → use GPT-4o-mini with structured outputs to classify
    - LLM chooses between `passthrough` or `orchestration` ONLY (never chooses `reasoning`)
    - Reasoning path is manual-only for baseline testing
- Classifier prompt analyzes if query needs multi-agent orchestration vs. direct answer

**Passthrough Implementation:**
- Update `/v1/chat/completions` endpoint to call routing logic
- For passthrough requests: create AsyncOpenAI client and call directly
- Stream response in OpenAI format (maintains compatibility)
- No ReasoningAgent, no reasoning steps (pure passthrough)
- Preserve tracing, error handling, authentication

**Reasoning Path:**
- Keep existing ReasoningAgent implementation active
- Accessible via `X-Routing-Mode: reasoning` header
- Single-loop reasoning with visual reasoning steps
- Serves as baseline for orchestration comparison

**Orchestration Path Stub:**
- Return HTTP 501 Not Implemented with clear message
- Prepares for M3-M4 when orchestrator is ready

**Implementation Decisions & Clarifications:**

1. **Routing Header Design**:
   - Single header: `X-Routing-Mode` with enum values (case-insensitive)
   - Values: `passthrough`, `reasoning`, `orchestration`, `auto`
   - **Default behavior**: If header not provided → `passthrough` (matches OpenAI experience)
   - **Auto-routing**: If header is `auto` → LLM classifier chooses between `passthrough` or `orchestration`
   - **Manual selection**: User can explicitly request any of the three paths via header
   - **Reasoning path**: Manual-only (LLM classifier never chooses reasoning, only user can)
   - No backward compatibility concerns - this is new, unreleased project

2. **Classifier Model Configuration**:
   - Classifier model configurable via `ROUTING_CLASSIFIER_MODEL` environment variable
   - Defaults to `gpt-4o-mini` for cost/performance balance
   - Temperature configurable via `ROUTING_CLASSIFIER_TEMPERATURE` (default 0.0 for deterministic)
   - Accept ~100-300ms latency and ~$0.0001 per request as necessary overhead
   - Only invoked when `X-Routing-Mode: auto` is requested
   - Consider optimization in M10 if needed (local model, caching, etc.)

3. **Passthrough Rules Interpretation**:
   - `response_format`: User wants specific output format (JSON mode, structured outputs) → force passthrough
   - `tools`: User is doing their own function calling → force passthrough
   - These take precedence over all routing logic (even explicit headers)
   - Ensures OpenAI API compatibility for advanced features

4. **Three-Route Architecture**:
   - **Passthrough**: Direct OpenAI API, no reasoning layer, implemented in M1
   - **Reasoning**: Single-loop ReasoningAgent stays active, accessible via header, baseline for comparison
   - **Orchestration**: Multi-agent coordination via A2A protocol, returns 501 stub until M3-M4
   - ReasoningAgent does NOT become temporarily unused - it remains available as route B
   - Orchestration will NOT call reasoning agent - orchestration IS the evolved reasoning (multi-agent)

5. **Passthrough Path Requirements**:
   - ✅ Bearer token authentication (existing `verify_token`)
   - ✅ OpenTelemetry tracing (span management like current endpoint)
   - ✅ Both streaming and non-streaming support
   - ✅ OpenAI error forwarding (httpx.HTTPStatusError handling)
   - ✅ Client disconnection handling
   - ✅ Session ID header support (for future orchestration path compatibility)

6. **Testing Strategy**:
   - **Unit tests**: Mock LLM classifier responses, test passthrough rules, test all 4 header values
   - **Integration tests**: Real GPT-4o-mini API calls for end-to-end path validation (require OPENAI_API_KEY)
   - **Test all three paths**: passthrough, reasoning (with header), orchestration stub (501)
   - **Evaluations**: Use flex-evals framework for non-deterministic LLM behavior testing
   - Fix existing ReasoningAgent tests by adding `X-Routing-Mode: reasoning` header
   - Focus on deterministic software logic in unit/integration tests

**Deliverables:**
- `api/request_router.py` with `RoutingMode` enum and routing logic
- Pydantic `RoutingDecision` model with `routing_mode: RoutingMode`
- Passthrough rules for `response_format` and `tools` (force passthrough)
- Header-based routing (`X-Routing-Mode: passthrough|reasoning|orchestration|auto`)
- Routing configuration in `api/config.py` (classifier model, temperature)
- Updated `/v1/chat/completions` with three-path routing logic
- Passthrough path works end-to-end (API → OpenAI → client)
- Reasoning path remains accessible via header
- Orchestration path returns 501 stub with clear message
- Unit tests (passthrough rules, all 4 header modes, mocked LLM classifier)
- Integration tests (end-to-end for all three paths)
- Fixed existing ReasoningAgent tests (add routing header)
- Documentation of routing logic, headers, and configuration

---

#### Milestone 2: A2A Protocol Foundation

**What:**
Build the A2A protocol foundation: Pydantic models, HTTP client library, and translation utilities. This prepares for complex path without needing working agents yet.

**Why:**
- Foundation for all agent communication
- Can be developed and tested independently
- Reusable across all agents (orchestrator, reasoning, RAG, etc.)
- Get protocol details right before building agents
- Translation layer tested in isolation

**How (High-Level):**
- **A2A Protocol Models (Pydantic)**:
  - Task models: TaskCreate, TaskStatus, TaskUpdate
  - Message models: A2AMessage, Artifact
  - Agent Card model: AgentCard with capabilities
  - Task lifecycle states: created, running, auth-required, completed, failed, canceled

- **A2A Client Library**:
  - HTTP client for A2A endpoints (POST /tasks, GET /tasks/{id}, GET /tasks/{id}/stream)
  - SSE stream subscription and parsing
  - Task lifecycle management
  - Agent Card fetching
  - Error handling and retries

- **Translation Utilities**:
  - `openai_to_a2a_task()`: Convert OpenAI messages to A2A task
  - `a2a_artifacts_to_openai_stream()`: Convert A2A SSE to OpenAI chunks
  - Map task status → OpenAI finish_reason
  - Preserve metadata (tokens, timing)

**Deliverables:**
- A2A protocol models (Pydantic schemas)
- A2A client library with full protocol support
- Translation functions with unit tests
- Mock A2A server for testing (FastAPI with stub responses)
- Documentation of A2A protocol usage

---

### Phase 2: Orchestration Path Implementation (Build DAG Logic, Test with Mocks)

#### Milestone 3: Planning Agent with Full DAG Generation

**What:**
Build planning agent with complete DAG generation capabilities: multi-agent workflows, dependencies, parallelization. Build it right from the start.

**Why:**
- Get the orchestration logic correct before dealing with real agents
- DAG generation is complex - build it properly once
- Can test thoroughly with mock agents
- No "build simple, enhance later" technical debt

**How (High-Level):**
- FastAPI service in `services/planning_agent/`
- Implement A2A protocol (Agent Card, task endpoints, SSE)
- Use LLM with structured output to analyze user intent
- **Generate complex DAGs**:
  - Multiple agents with dependencies
  - Parallel execution opportunities (independent steps)
  - Conditional logic (if/then branching)
  - Data flow between agents (output of A feeds input of B)
- **DAG Schema (Pydantic)**:
  - Nodes: agent_name, inputs, outputs
  - Edges: dependencies between nodes
  - Metadata: parallelizable groups, conditional branches
- Validate DAG: no cycles, valid structure
- Returns structured DAG to orchestrator
- Publishes Agent Card

**Deliverables:**
- Planning agent service with A2A protocol
- Full DAG generation using LLM
- DAG schema defined (Pydantic models)
- DAG validation logic (cycle detection, etc.)
- Can generate plans with 3-5+ agents
- Unit tests with sample queries → expected DAGs
- Agent Card published

---

### Session Management Design

**Note:** Session management is implemented as part of M4 (Orchestrator) since that's when persistence is actually needed. This section documents the design decisions made upfront.

**What Sessions Are:**
- Sessions are workflow-scoped, not conversation-scoped
- Track DAG execution state, agent results, and approval requests
- Do NOT store conversation history (client sends full `messages` array each request, maintaining OpenAI compatibility)
- Short-lived (hours to days), tied to a specific workflow execution

**Session Lifecycle:**
1. Server creates session on first complex request, returns `session_id` in response header
2. Client includes `session_id` in subsequent requests to resume workflow
3. Session persists while status is `active` or `waiting_approval`
4. Session deleted immediately when workflow completes/fails/cancelled
5. Abandoned sessions auto-expire after TTL (7 days)

**Session State Schema:**
```python
class SessionState:
    session_id: str  # Server-generated UUID
    status: SessionStatus  # active, waiting_approval, completed, failed, cancelled
    created_at: datetime
    updated_at: datetime

    # Workflow state (M4)
    dag: dict | None  # DAG structure from planning agent
    completed_nodes: list[dict] | None  # Results from completed agents
    current_nodes: list[str] | None  # Currently executing nodes

    # Human-in-the-loop (M9)
    approval_context: dict | None  # What needs approval, why, risks

    # Error tracking
    error_message: str | None
    error_type: str | None

    # Metadata
    user_id: str | None
```

**Key Design Decisions:**

1. **TTL Strategy:**
   - Fixed 7-day TTL (accommodates slow approval workflows)
   - Explicit deletion when workflow completes/fails/cancelled
   - Redis auto-deletes abandoned sessions after TTL
   - Only `active` and `waiting_approval` sessions persist in Redis

2. **Session ID Handling:**
   - Server always generates session IDs (UUIDs)
   - Client never generates IDs, only echoes back what server provides
   - Initial request: no session ID → server creates and returns in header
   - Resume request: client passes session ID → server loads state

3. **Concurrency Control:**
   - **Required from day 1:** Redis distributed locks for session access
   - Use Redis `SET NX EX` for lock acquisition with timeout
   - Prevent concurrent requests from corrupting same session state
   - Lock pattern: acquire lock → load → modify → save → release lock
   - This is NOT optional - concurrent access is a real scenario

4. **Error Handling:**
   - Redis failures: retry with exponential backoff (tenacity), then fail with 503
   - Lock timeout: fail request with 409 Conflict (another request in progress)
   - Missing session: return 404 (expired or never existed)

5. **Storage Format:**
   - JSON serialization for SessionState
   - Redis key pattern: `session:{session_id}`
   - Connection pooling for performance

6. **Testing Strategy:**
   - Unit tests: `fakeredis` for SessionManager logic
   - Integration tests: real Redis for persistence/TTL validation
   - Concurrency tests: simulate concurrent requests to same session

---

#### Milestone 4: Orchestrator with Full DAG Execution + Session Management

**What:**
Build orchestrator with complete DAG execution engine AND implement session management for workflow state persistence. This combines orchestration logic with the state management needed to support pause/resume workflows.

**Why:**
- DAG execution is complex - don't build it twice
- Sessions are only needed when DAG execution requires pause/resume
- Building together avoids building unused infrastructure
- Can test full orchestration + persistence as a unit

**How (High-Level):**

**Infrastructure:**
- Add Redis to docker-compose (`redis:7-alpine`, health check, persistent volume)
- `uv add "redis[asyncio]"` and `uv add "tenacity"` to api group
- Configure Redis connection settings in `config.py`

**Session Management (`api/session_models.py`, `api/session_manager.py`):**
- `SessionState` Pydantic model with DAG fields
- `SessionManager` class with Redis connection pool
- Methods: create(), load(), save(), delete(), cancel(), list_sessions()
- **Distributed locking:** `acquire_lock()` and `release_lock()` using Redis `SET NX EX`
- Lock wrapper: `with_lock()` context manager for safe session access
- Retry logic with tenacity for Redis failures
- Integrate into `ServiceContainer` for dependency injection
- Unit tests with `fakeredis`, integration tests with real Redis

**Orchestrator Service:**
- FastAPI service in `services/orchestrator/`
- Implement A2A protocol (Agent Card, task endpoints, SSE)
- Receives complex requests from main API (via A2A)
- **Session Integration:**
  - Create session on workflow start
  - Store DAG in session state
  - Update session with completed nodes
  - Lock session during state updates
  - Delete session on completion
- **DAG Execution Engine**:
  - Topological sort for dependency resolution
  - `asyncio.gather()` for parallel execution
  - Track node states (pending, running, completed, failed)
  - Save progress to session after each node completes
  - Handle agent failures (retries, fallbacks)
  - Collect artifacts from each agent
- **Dynamic DAG Adjustment**:
  - Evaluate intermediate results
  - Modify remaining DAG based on outcomes
  - Update session with adjusted DAG
  - Skip/add/modify nodes based on runtime conditions
- **Pause/Resume Support**:
  - Save DAG execution state to session
  - Resume from session state on retry/approval
- Stream aggregated progress to main API
- Publishes Agent Card

**API Integration:**
- Add session endpoints to main API:
  - `GET /v1/sessions/{id}` - retrieve session state
  - `DELETE /v1/sessions/{id}` - cancel workflow
  - `GET /v1/sessions` - list sessions (admin/debug)
- Complex path returns session ID in response header

**Open Questions:**

**Reasoning Event Streaming:**
- How do agent reasoning steps flow back to the end user during orchestration?
- Current system: ReasoningAgent streams events like `🔍 Analyzing query...` with custom `reasoning_event` field
- Multi-agent scenario: Each agent (reasoning, RAG, search) generates its own reasoning steps
- Questions to resolve:
  - Does A2A protocol support streaming reasoning events/artifacts during task execution?
  - How should orchestrator aggregate events from multiple concurrent agents?
  - Event ordering: How to maintain coherent sequence when agents run in parallel?
  - Event attribution: How to indicate which agent produced which reasoning step?
  - Orchestrator events: Should orchestrator generate its own coordination events ("Calling RAG agent...", "Waiting for dependencies...")?
- Research A2A specification for artifact streaming patterns
- Design event aggregation and forwarding strategy for M5 testing

**Deliverables:**
- Redis running in docker-compose
- Session management implementation (models, manager, locking)
- Orchestrator service with A2A protocol
- Full DAG execution engine (parallel, dependencies)
- Dynamic DAG adjustment logic
- Session persistence with distributed locking
- Pause/resume capability (foundation for M9)
- Handles errors and retries
- Ready to work with mock agents (M5)
- Agent Card published
- Comprehensive tests (unit, integration, concurrency)

---

#### Milestone 5: Mock Agents + End-to-End Testing

**What:**
Create 3-4 mock agents (simple HTTP endpoints with canned responses) and prove the entire complex flow works end-to-end with real DAG execution and session persistence.

**Why:**
- Test orchestration logic without real agent complexity
- Validate DAG generation, execution, and session persistence
- Prove parallel execution, dependencies, dynamic adjustment
- Test session locking under concurrent load
- Fast iteration without LLM costs
- Confidence before building real agents

**How (High-Level):**
- **Mock Agents** (simple FastAPI services):
  - Mock Reasoning Agent: returns canned reasoning response
  - Mock RAG Agent: returns canned "found documents" response
  - Mock Search Agent: returns canned search results
  - Mock Analysis Agent: returns canned analysis
  - Each implements A2A protocol (tasks, SSE streaming)
  - Configurable delays to simulate real execution
  - Publishes Agent Card with capabilities

- **End-to-End Testing**:
  - Complex request → Main API → Orchestrator → Planning Agent → DAG
  - Orchestrator executes DAG with mock agents
  - Test parallel execution (2+ agents running concurrently)
  - Test dependencies (agent B waits for agent A)
  - Test dynamic adjustment (skip/add nodes based on results)
  - Test error handling (mock agent failure, retry)
  - Validate streaming (artifacts flow through SSE)
  - Validate translation (OpenAI format maintained)
  - Test session persistence (survive service restart)
  - Test concurrent session access (locking validation)

**Deliverables:**
- 3-4 mock agent services with A2A protocol
- Complex requests work end-to-end
- Parallel execution validated
- Dependencies working correctly
- Dynamic DAG adjustment proven
- Session persistence validated
- Concurrency/locking tested
- Integration tests covering full stack
- Orchestration logic proven before real agents

---

### Phase 3: Real Agents (Replace Mocks with Production Implementations)

#### Milestone 6: Reasoning Agent as A2A Service

**What:**
Replace mock reasoning agent with real reasoning agent. Migrates existing reasoning logic to A2A architecture.

**Why:**
- First real agent with actual LLM logic
- Proves agent implementation pattern
- Establishes template for all future agents (RAG, search, etc.)
- DAG orchestration already tested with mocks

**How (High-Level):**
- Create FastAPI service in `services/reasoning_agent/`
- Implement full A2A protocol (tasks, streaming, Agent Card)
- Migrate reasoning logic from current embedded ReasoningAgent
- Maintains own MCP client for tools
- Store task state in Redis
- Stream artifacts via SSE
- Replace mock reasoning agent with real one in orchestrator config

**Deliverables:**
- Real reasoning agent service with A2A protocol
- Reasoning logic migrated from current codebase
- MCP tools integration working
- Can be called by orchestrator
- Mock reasoning agent replaced in workflows

---

#### Milestone 7: Vector Database + RAG Agent

**What:**
Set up PostgreSQL + pgvector for semantic search, create RAG agent for retrieval-augmented generation.

**Why:**
- Enables knowledge base queries
- Demonstrates value of multi-agent (RAG + reasoning)
- Provides another agent for DAG testing
- Reduces hallucinations with grounding

**How (High-Level):**

**Vector Database:**
- Add PostgreSQL + pgvector to docker-compose
- Database schema for documents and embeddings
- Choose embedding model (OpenAI API or local)
- Vector operations: insert, search, filter by metadata
- Data ingestion tool for loading documents

**RAG Agent:**
- FastAPI service in `services/rag_agent/`
- Implement A2A protocol
- Receives query → generates embedding
- Searches vector DB for relevant documents
- Constructs LLM context from results
- Generates response with citations
- Streams via SSE with source attribution
- Publishes Agent Card with RAG capabilities

**Deliverables:**
- PostgreSQL + pgvector running
- Vector operations working
- RAG agent service with A2A protocol
- Replace mock RAG agent in workflows
- Multi-agent workflow tested: reasoning + RAG

---

#### Milestone 8: Agent Discovery via A2A Agent Cards

**What:**
Implement dynamic agent discovery system using A2A Agent Cards. Replace hardcoded agent endpoints with discovery.

**Why:**
- Agents declare capabilities automatically
- Planning agent discovers available agents dynamically
- Agents can be added/removed without code changes
- More flexible than hardcoded endpoints
- Industry standard (A2A protocol)

**How (High-Level):**
- Create agent discovery module
- Fetch Agent Cards from configured endpoints
- Parse capabilities, endpoint, version from cards
- Cache cards with TTL (refresh periodically)
- Update planning agent to use discovery instead of hardcoded list
- Handle agent unavailability gracefully

**Deliverables:**
- Agent discovery module (shared library)
- Cached agent registry (Redis)
- Planning agent uses discovery for plan generation
- Hardcoded endpoints removed
- Discovery API for querying available agents

---

#### Milestone 9: Human-in-the-Loop with A2A auth-required

**What:**
Implement human approval workflows using A2A `auth-required` state. Agents or orchestrator can pause, request approval, resume after user approves.

**Why:**
- Critical for production AI systems (safety, oversight)
- Required for sensitive operations (data deletion, expensive actions)
- Enables trust and control
- A2A protocol natively supports this
- Session management (M4) provides pause/resume foundation

**How (High-Level):**
- Agent/orchestrator sets task status to `auth-required`
- Store approval context in session state: action, why, risks, impact
- Update session status to `waiting_approval`
- Expose approval endpoints: GET/POST `/approvals/{session_id}`
- User reviews and submits decision (approve/reject/modify)
- Resume execution from session state
- Handle timeout (fail or use default)

**Deliverables:**
- Approval workflow in orchestrator
- Approval endpoints (get pending, submit approval)
- Session state tracks approval context (leverages M4 infrastructure)
- Workflow resumes after approval
- Timeout handling
- Web UI or API for approvals

---

### Phase 4: Production Optimization

#### Milestone 10: Performance Optimization

**What:**
Optimize performance: caching, connection pooling, batch operations, lazy loading.

**Why:**
- Reduce latency
- Lower costs
- Better user experience
- Handle higher load

**How (High-Level):**
- Cache agent cards (don't fetch every time)
- Connection pooling for Redis and PostgreSQL
- Batch Redis operations where possible
- Lazy load session state (don't load full history unnecessarily)
- Cache LLM responses (optional, for repeated queries)
- Optimize vector search (index tuning)
- Profile and identify bottlenecks

**Deliverables:**
- Caching strategy implemented
- Connection pooling configured
- Batch operations where applicable
- Performance benchmarks
- Latency improvements measured

---

## Development Workflow

**Infrastructure (docker-compose):**
- Redis (session state)
- PostgreSQL (pgvector for RAG)
- Phoenix (observability)

**Application Services (separate FastAPI apps):**
- Main API (OpenAI-compatible, A2A translation)
- Reasoning Agent (A2A service)
- Planning Agent (A2A service)
- Orchestrator (A2A service)
- RAG Agent (A2A service)

**Running the System:**
1. Start infrastructure: `docker-compose up -d`
2. Start each service in separate terminal or via process manager
3. Services communicate via HTTP (A2A protocol)
4. Main API at localhost:8000 (OpenAI-compatible)

**Service Communication:**
- All agents publish Agent Cards
- Services discover each other via configured endpoints
- Communication via A2A protocol (HTTP + SSE)
- State stored in Redis
- Vector data in PostgreSQL

---

## Next Steps

This document provides the high-level "what", "why", and "how" for each milestone. The next step is to expand each milestone with:
- Detailed implementation tasks
- Code examples and patterns
- Testing strategies
- Success criteria
- Documentation requirements

We'll work through each milestone, expanding the details as we go, starting with Milestone 1.
