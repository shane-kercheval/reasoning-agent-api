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
- **Request routing**: Simple requests → direct OpenAI call; Complex requests → orchestrator
- **OpenAI-compatible external API** maintaining industry standard interface
- **A2A protocol** for agent-to-agent communication (complex path only)
- **Orchestrator agent** that plans and coordinates multi-agent workflows
- **Multiple specialized agents** (reasoning, planning, RAG) communicating via A2A
- **Persistent session management** with Redis for long-running workflows
- **Human-in-the-loop workflow support** via A2A `auth-required` state
- **Dynamic agent discovery** via A2A Agent Cards
- **MCP protocol maintained** for tool integration (agents use MCP for external tools)
- **Vector database** (PostgreSQL + pgvector) for RAG semantic search
- **Docker-compose for infrastructure** (Redis, PostgreSQL, Phoenix)
- **Services run as separate FastAPI apps** (main API, orchestrator, planning agent, reasoning agent, RAG agent)

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

**Request Routing Strategy:**
- **Simple requests** (majority): Main API calls OpenAI/LLM directly, streams back immediately
  - No agents, no A2A protocol, no orchestration overhead
  - Fast path for straightforward queries like "What's 2+2?" or "Summarize this text"
- **Complex requests** (multi-step): Main API delegates to Orchestrator via A2A protocol
  - Orchestrator uses Planning Agent to generate DAG
  - Orchestrator coordinates multiple agents (reasoning, RAG, search, etc.)
  - Aggregates results and streams back
- **Dual Protocol**:
  - External: OpenAI API (industry standard)
  - Internal: A2A protocol for agent communication
  - Translation layer converts between protocols for complex path only

---

## A2A Protocol Architecture

### Overview

The system routes requests based on complexity:
- **Simple requests** → direct OpenAI/LLM call (fast path)
- **Complex requests** → A2A protocol orchestration (multi-agent coordination)

```
┌─────────────────────────────────────────────────────────────┐
│  External Clients (OpenAI SDK, curl, etc.)                  │
└─────────────┬───────────────────────────────────────────────┘
              │ OpenAI Protocol
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Main API (FastAPI)                                         │
│  - /v1/chat/completions (OpenAI compatible)                 │
│  - Request routing: simple vs complex                       │
│  - Session management (Redis)                               │
└─────────────┬───────────────────────────────────────────────┘
              │
         ┌────┴─────┐
         │ ROUTING  │
         └────┬─────┘
              │
    ┌─────────┴──────────┐
    │                    │
    │ Simple             │ Complex
    │                    │
    ▼                    ▼
┌─────────────┐    ┌─────────────────────────────────────────┐
│  OpenAI     │    │  Orchestrator Agent (A2A)               │
│  API        │    │  - Agent Card                           │
│  Direct     │    │  - Uses Planning Agent for DAG          │
│             │    │  - Delegates to multiple agents         │
│             │    │  - Aggregates artifacts                 │
│             │    │  - Manages workflow state (Redis)       │
└─────────────┘    └─────────────┬───────────────────────────┘
                                 │ A2A Protocol
                                 │
                       ┌─────────┴─────────┬──────────────┐
                       ▼ A2A               ▼ A2A          ▼ A2A
                   ┌──────────────┐  ┌──────────────┐  ┌─────────┐
                   │  Reasoning   │  │  RAG Agent   │  │ Planning│
                   │  Agent       │  │              │  │ Agent   │
                   │              │  │              │  │         │
                   │ Agent Card   │  │ Agent Card   │  │ Agent   │
                   │ A2A Tasks    │  │ A2A Tasks    │  │ Card    │
                   │ SSE Stream   │  │ SSE Stream   │  │ A2A     │
                   └──────┬───────┘  └──────┬───────┘  └────┬────┘
                          │                 │               │
                          │ MCP             │               │ MCP
                          ▼                 ▼               ▼
                   ┌──────────────┐  ┌──────────────┐  ┌─────────────┐
                   │ MCP Tools    │  │ Vector DB    │  │ External    │
                   │ (Weather,    │  │ (pgvector)   │  │ Search APIs │
                   │  Search, etc)│  │              │  │             │
                   └──────────────┘  └──────────────┘  └─────────────┘
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

The milestones are ordered by dependency and value delivery:
- **Phase 1**: Get simple requests working (immediate value)
- **Phase 2**: Add complex orchestration (multi-agent coordination)
- **Phase 3**: Enhance with advanced agents (RAG, discovery, human-in-loop)
- **Phase 4**: Production hardening

### Phase 1: Foundation & Simple Path

#### Milestone 1: Session Management with Redis

**What:**
Implement persistent session management using Redis directly. Sessions track reasoning state, workflow progress, and enable pause/resume capability.

**Why:**
- Foundation for all stateful workflows
- Required for human-in-the-loop (can't pause/resume without state)
- Sessions survive service restarts
- Enables cross-request coordination (mobile cancelling web, Tab A cancelling Tab B)

**How (High-Level):**
- Add Redis to docker-compose infrastructure
- Create session manager module with Redis client
- Define session state schema (Pydantic models)
- Sessions store: session_id, status, reasoning context, timestamps, metadata
- Session status aligns with A2A task states (active, waiting_approval, completed, failed, cancelled)
- Update API endpoints to extract/return session IDs via headers
- Store sessions with TTL for automatic cleanup

**Deliverables:**
- Redis running in docker-compose
- Session manager module with create/save/load/delete operations
- Session state persists across service restarts
- API endpoints support X-Session-ID header
- Sessions can be queried and cancelled

---

#### Milestone 2: Request Routing + Simple Path

**What:**
Implement request routing logic to classify simple vs complex queries, and handle simple queries by calling OpenAI/LLM directly. This provides immediate value and establishes the fast path for most requests.

**Why:**
- Most queries (estimated 80%+) are simple and don't need orchestration
- Get value immediately - API works end-to-end for simple requests
- Fast path: no agent overhead, no A2A protocol, direct LLM streaming
- Establishes routing infrastructure for later complex path
- Maintains OpenAI compatibility

**How (High-Level):**
- **Routing Decision**: Classify incoming requests as simple or complex
  - Options to explore: LLM classifier, explicit flag in request, rule-based patterns, hybrid
  - Start simple (e.g., explicit flag or rule-based), can enhance later
  - Document routing criteria

- **Simple Path Implementation**:
  - For simple requests: call OpenAI API directly from main API
  - Stream response back in OpenAI format (already supported)
  - Use existing OpenAI client in main API
  - Store in session if session_id provided

- **Complex Path Stub**:
  - For complex requests: return "not implemented yet" or route to simple path temporarily
  - Prepares for Milestone 4 when orchestrator is ready

**Deliverables:**
- Routing logic in `/v1/chat/completions` endpoint
- Simple requests work end-to-end (API → OpenAI → client)
- Routing decision documented (criteria for simple vs complex)
- Tests for routing logic
- Complex path has placeholder/stub
- System provides value for simple queries immediately

---

#### Milestone 3: A2A Protocol Foundation

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

### Phase 2: Complex Path - Basic Orchestration

#### Milestone 4: Orchestrator + Planning Agent

**What:**
Build orchestrator and planning agent together. Planning agent generates DAGs, orchestrator executes them. Start with simple single-agent delegation to prove the flow, then enhance.

**Why:**
- These components are tightly coupled (orchestrator needs planning agent)
- Building together proves end-to-end complex path architecture
- Can start simple (single agent) and enhance (multi-agent DAG)
- Establishes pattern for all future agent coordination

**How (High-Level):**

**Planning Agent** (`services/planning_agent/`):
- FastAPI service implementing A2A protocol
- Receives user request, analyzes intent with LLM
- For now: simple plans ("call reasoning agent")
- Later: complex DAGs with multiple agents, dependencies, parallelization
- Returns structured plan (Pydantic model)
- Publishes Agent Card

**Orchestrator** (`services/orchestrator/`):
- FastAPI service implementing A2A protocol
- Receives complex requests from main API (via A2A)
- Calls planning agent to get workflow plan (DAG)
- Executes plan: for now just delegates to one agent
- Streams results back to main API
- Manages state in Redis
- Publishes Agent Card

**Start Simple, Enhance Later:**
- Phase 2: Single-agent delegation (proves flow)
- Phase 3: Multi-agent DAG execution with discovery

**Deliverables:**
- Planning agent service with A2A protocol
- Orchestrator service with A2A protocol
- Can receive complex request from main API
- Can generate simple plan ("call reasoning agent")
- Can execute simple plan (delegate to one agent)
- State stored in Redis
- Both services publish Agent Cards

---

#### Milestone 5: Reasoning Agent as A2A Service

**What:**
Create reasoning agent as independent FastAPI service implementing A2A protocol. This is the first "worker agent" that orchestrator will delegate to.

**Why:**
- Orchestrator needs at least one agent to call
- Proves agent implementation pattern
- Establishes template for all future agents (RAG, search, etc.)
- Migrates existing reasoning logic to A2A architecture

**How (High-Level):**
- Create FastAPI service in `services/reasoning_agent/`
- Implement full A2A protocol:
  - POST /tasks - Create reasoning task
  - GET /tasks/{id} - Get task status
  - GET /tasks/{id}/stream - SSE stream
  - PUT /tasks/{id} - Update (for approvals)
  - DELETE /tasks/{id} - Cancel
  - GET /.well-known/agent-card.json - Capabilities
- Migrate reasoning logic from current embedded ReasoningAgent
- Store task state in Redis
- Stream artifacts via SSE
- Maintains own MCP client for tools
- Declares capabilities in Agent Card

**Deliverables:**
- Reasoning agent service with A2A protocol
- All task endpoints functional
- SSE streaming working
- Agent Card published
- Can be called by orchestrator

---

#### Milestone 6: End-to-End Complex Flow

**What:**
Wire together the complete complex path: Main API → Orchestrator → Planning Agent → Reasoning Agent → back to client. Prove the full stack works end-to-end.

**Why:**
- Validates entire complex request architecture
- Proves all components integrate correctly
- Tests A2A protocol across service boundaries
- Tests translation layer (OpenAI ↔ A2A)
- Provides working complex path (even if just single agent for now)

**How (High-Level):**
- Update main API routing: complex requests → Orchestrator (via A2A client)
- Orchestrator receives A2A task from main API
- Orchestrator → Planning Agent: "analyze this request"
- Planning Agent returns simple plan: "call reasoning agent"
- Orchestrator → Reasoning Agent: delegates via A2A
- Reasoning Agent streams artifacts back via SSE
- Orchestrator aggregates and streams to main API
- Main API translates A2A SSE → OpenAI streaming format
- Client receives OpenAI-formatted response

**Integration Points to Test:**
- Main API uses A2A client to call orchestrator
- Orchestrator uses A2A client to call planning agent
- Orchestrator uses A2A client to call reasoning agent
- SSE streaming works across all boundaries
- Translation layer works (OpenAI → A2A → OpenAI)
- Sessions track complex workflow state

**Deliverables:**
- Complex requests work end-to-end
- Main API successfully routes to orchestrator
- Full chain: API → Orch → Planning → Reasoning → back
- SSE streaming across all services
- Translation layer validated in production flow
- Integration tests covering complete flow

---

### Phase 3: Multi-Agent Enhancement

#### Milestone 7: Agent Discovery via A2A Agent Cards

**What:**
Implement dynamic agent discovery system using A2A Agent Cards. Agents self-describe capabilities, orchestrator/planning agent can discover agents dynamically.

**Why:**
- Agents declare capabilities automatically (no manual config)
- Planning agent can discover available agents and their capabilities
- Agents can be added/removed without code changes
- Foundation for multi-agent DAG execution
- Industry standard (A2A protocol)

**How (High-Level):**
- Create agent discovery module
- Fetch Agent Cards from configured endpoints (`/.well-known/agent-card.json`)
- Parse capabilities, endpoint, version from cards
- Cache cards with TTL (refresh periodically)
- Planning agent uses discovery to find available agents
- Match capabilities to user intent during planning
- Handle agent unavailability gracefully

**Deliverables:**
- Agent discovery module (shared library or service)
- Cached agent registry (Redis)
- Planning agent uses discovery for plan generation
- Can discover: orchestrator, planning, reasoning agents
- Discovery API for querying available agents

---

#### Milestone 8: Multi-Agent DAG Execution

**What:**
Enhance orchestrator and planning agent to handle complex multi-agent DAGs: multiple agents, dependencies, parallel execution, dynamic DAG adjustment based on intermediate results.

**Why:**
- Enables true multi-agent coordination
- Handles complex queries requiring multiple specialized agents
- Optimizes workflow with parallelization
- Adapts plan based on runtime results
- Core value proposition of multi-agent system

**How (High-Level):**

**Planning Agent Enhancement:**
- Use agent discovery to find available agents
- Generate complex DAGs with multiple agents
- Define dependencies between steps
- Identify parallel execution opportunities
- Output structured DAG (nodes, edges, dependencies)

**Orchestrator Enhancement:**
- Parse and validate DAG from planning agent
- Execute steps respecting dependencies (topological sort)
- Use `asyncio.gather()` for parallel execution
- Subscribe to multiple agent SSE streams concurrently
- **Dynamic DAG Adjustment:**
  - Evaluate intermediate results from agents
  - Adjust remaining DAG based on outcomes
  - Examples:
    - "RAG agent found nothing → skip summarization step"
    - "Search unclear → add clarification step"
    - "Tool failed → retry with different agent"
  - Requires: result evaluation logic, DAG mutation, re-planning triggers
- Aggregate artifacts from all agents
- Store current DAG state in Redis
- Stream aggregated progress to client

**Deliverables:**
- Planning agent generates multi-agent DAGs
- Orchestrator executes complex DAGs
- Parallel execution working (`asyncio.gather`)
- Multi-agent SSE aggregation
- Dynamic DAG adjustment implemented
- Tests with 2-3 agent workflows

---

#### Milestone 9: Vector Database + RAG Agent

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
- Planning agent can discover and use RAG agent
- Multi-agent workflow: reasoning + RAG

---

#### Milestone 10: Human-in-the-Loop with A2A auth-required

**What:**
Implement human approval workflows using A2A `auth-required` state. Agents or orchestrator can pause, request approval, resume after user approves.

**Why:**
- Critical for production AI systems (safety, oversight)
- Required for sensitive operations (data deletion, expensive actions)
- Enables trust and control
- A2A protocol natively supports this

**How (High-Level):**
- Agent/orchestrator sets task status to `auth-required`
- Store approval context: action, why, risks, impact
- Update session state to reflect pause
- Expose approval endpoints: GET/POST `/approvals/{session_id}`
- User reviews and submits decision (approve/reject/modify)
- Resume execution from paused state
- Handle timeout (fail or use default)

**Deliverables:**
- Approval workflow in orchestrator
- Approval endpoints (get pending, submit approval)
- Session state tracks approval context
- Workflow resumes after approval
- Timeout handling
- Web UI or API for approvals

---

### Phase 4: Production Readiness

#### Milestone 11: Observability and Monitoring

**What:**
Enhanced observability: structured logging, distributed tracing, metrics, cost tracking.

**Why:**
- Debug multi-agent workflows
- Monitor performance and costs
- Track errors across services
- Understand user behavior

**How (High-Level):**
- Structured logging with correlation IDs (session_id, task_id)
- OpenTelemetry distributed tracing across all agents
- Trace A2A task flows end-to-end
- Metrics: request rates, latencies, error rates, agent usage
- Cost tracking: token usage per agent, per session
- Dashboards in Phoenix or Grafana

**Deliverables:**
- Structured logging across all services
- Distributed tracing with A2A context
- Metrics collection and dashboards
- Cost tracking and attribution
- Alert rules for critical errors

---

#### Milestone 12: Error Handling and Resilience

**What:**
Production-grade error handling, retries, circuit breakers, graceful degradation.

**Why:**
- Handle agent failures gracefully
- Prevent cascading failures
- Maintain system availability
- Good user experience during issues

**How (High-Level):**
- Retry logic with exponential backoff for agent calls
- Circuit breakers for unhealthy agents
- Timeout handling with sensible defaults
- Graceful degradation (fallback to single agent if orchestrator fails)
- Clear error messages for users
- Dead letter queues for failed tasks (optional)

**Deliverables:**
- Retry logic in orchestrator
- Circuit breakers for agent calls
- Timeout configurations
- Graceful degradation paths
- Error message standards

---

#### Milestone 13: Performance Optimization

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
