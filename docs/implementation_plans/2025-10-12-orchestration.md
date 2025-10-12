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
- **A2A protocol** for agent-to-agent communication
- **OpenAI-compatible external API** with A2A translation layer
- **Orchestrator agent** that plans and coordinates multi-agent workflows
- **Multiple specialized agents** (reasoning, planning, RAG, search) communicating via A2A
- **Persistent session management** with Redis for long-running workflows
- **Human-in-the-loop workflow support** via A2A `auth-required` state
- **Dynamic agent discovery** via A2A Agent Cards
- **MCP protocol maintained** for tool integration (agents internally use MCP for external tools)
- **Vector database** (PostgreSQL + pgvector) for RAG semantic search
- **Docker-compose for infrastructure** (Redis, PostgreSQL, Phoenix)
- **Services run as separate FastAPI apps** (main API, reasoning agent, planning agent, orchestrator, RAG agent)

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

**Dual Protocol Strategy:**
- **OpenAI API**: Primary external interface - industry standard, existing SDKs, drop-in replacement for OpenAI
- **A2A Protocol**: Internal agent communication - standardized, self-describing, built-in lifecycle management
- **Translation Layer**: Seamless conversion between protocols
- **Best of both worlds**: User-facing compatibility + standardized agent coordination

---

## A2A Protocol Architecture

### Overview

The system uses **A2A (Agent2Agent) protocol** for all inter-agent communication while maintaining **OpenAI compatibility** for external clients.

```
┌─────────────────────────────────────────────────────────────┐
│  External Clients (OpenAI SDK, curl, etc.)                  │
└─────────────┬───────────────────────────────────────────────┘
              │ OpenAI Protocol
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Main API (FastAPI)                                         │
│  - /v1/chat/completions (OpenAI compatible)                 │
│  - Translation Layer: OpenAI ↔ A2A                          │
│  - Session management (Redis)                               │
└─────────────┬───────────────────────────────────────────────┘
              │ A2A Protocol
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Orchestrator Agent (A2A)                                   │
│  - Agent Card: /.well-known/agent-card.json                 │
│  - Discovers other agents via Agent Cards                   │
│  - Creates A2A tasks and delegates to agents                │
│  - Aggregates A2A artifacts from agents                     │
│  - Manages workflow lifecycle and state (Redis)             │
└─────────────┬───────────────────────────────────────────────┘
              │ A2A Protocol (Task Delegation)
              │
    ┌─────────┴─────────┬──────────────┬───────────────┐
    ▼ A2A               ▼ A2A          ▼ A2A           ▼ A2A
┌──────────────┐  ┌──────────────┐  ┌─────────┐  ┌──────────────┐
│  Reasoning   │  │  RAG Agent   │  │ Planning│  │   Search     │
│  Agent       │  │              │  │ Agent   │  │   Agent      │
│              │  │              │  │         │  │              │
│ Agent Card   │  │ Agent Card   │  │ Agent   │  │ Agent Card   │
│              │  │              │  │ Card    │  │              │
│ A2A Tasks    │  │ A2A Tasks    │  │ A2A     │  │ A2A Tasks    │
│ SSE Stream   │  │ SSE Stream   │  │ Tasks   │  │ SSE Stream   │
└──────┬───────┘  └──────┬───────┘  └────┬────┘  └──────────────┘
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

### Phase 1: Foundation

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

#### Milestone 2: Request Routing and A2A Integration

**What:**
Implement request routing logic to determine simple vs complex queries, plus A2A client for agent communication, plus format translation between OpenAI and A2A protocols.

**Why:**
- OpenAI API is the primary interface (industry standard, existing SDKs)
- Not all requests need orchestration - simple queries can go directly to agents
- Need standardized way to communicate with agents (A2A protocol)
- Must seamlessly convert between OpenAI format (what users send) and A2A format (what agents speak)
- Maintain familiar OpenAI API experience for users

**How (High-Level):**
- **Routing Decision**: Determine if request is simple or complex
  - Options: LLM classifier, explicit flag in request, rule-based patterns, or hybrid approach
  - Simple requests → route to reasoning agent directly
  - Complex requests → route to orchestrator for multi-agent coordination

- **A2A Client Module**: Communicate with agents
  - HTTP client for A2A endpoints (POST /tasks, GET /tasks/{id}/stream)
  - Subscribe to SSE streams from agents
  - Handle A2A task lifecycle
  - Fetch Agent Cards for discovery

- **Format Translation**: Convert between protocols
  - OpenAI messages → A2A task messages (different structure)
  - A2A artifacts (SSE events) → OpenAI streaming chunks
  - Map task status → OpenAI finish_reason
  - Preserve metadata (token usage, timing)

**Deliverables:**
- Routing logic in main API endpoint
- A2A client module for making agent requests
- A2A protocol models (Pydantic)
- Translation functions: openai_to_a2a_task, a2a_artifacts_to_openai_stream
- SSE streaming from agents working
- OpenAI API maintains standard interface

---

#### Milestone 3: Reasoning Agent as Separate A2A Service

**What:**
Create the reasoning agent as an independent FastAPI service that implements A2A protocol. This service runs separately from the main API and communicates via HTTP/SSE. Migrates reasoning logic from the embedded ReasoningAgent class to this new service.

**Why:**
- Proves A2A protocol works end-to-end across services
- Establishes pattern for all other agents (planning, RAG, etc.)
- Enables independent scaling and deployment of reasoning agent
- Agent can be updated/restarted without affecting main API
- Clear architectural separation (API layer vs agent layer)
- Foundation for orchestrator to coordinate multiple independent agents

**How (High-Level):**
- Create new independent FastAPI service in `services/reasoning_agent/`
- Service runs on its own port (e.g., 8001) separate from main API (8000)
- Implement full A2A protocol:
  - POST /tasks - Create reasoning task
  - GET /tasks/{id} - Get task status
  - GET /tasks/{id}/stream - SSE stream of reasoning progress
  - PUT /tasks/{id} - Update task (for approvals)
  - DELETE /tasks/{id} - Cancel task
  - GET /.well-known/agent-card.json - Publish capabilities
- Agent Card declares capabilities: reasoning, tool_use, analysis, mcp_tools
- Migrate reasoning logic from main API's embedded class to this service
- Store A2A task state (task_id, status, messages, artifacts) in Redis
- Stream artifacts via SSE as reasoning progresses
- Service maintains its own MCP client for tool integration
- Update main API to communicate with reasoning agent via A2A HTTP calls

**Deliverables:**
- Reasoning agent service running independently on separate port
- Agent Card accessible via HTTP
- All A2A task endpoints functional
- SSE streaming artifacts in real-time
- Main API successfully makes HTTP calls to reasoning service
- A2A translation layer working end-to-end
- OpenAI API maintains standard interface (no client-facing changes)

---

### Phase 2: Multi-Agent Orchestration

#### Milestone 4: Agent Discovery via A2A Agent Cards

**What:**
Implement dynamic agent discovery system using A2A Agent Cards. Agents self-describe their capabilities, no manual registry needed.

**Why:**
- Agents declare their capabilities automatically
- No manual configuration to keep in sync
- Planning agent can discover what agents exist and what they can do
- Agents can be added/removed without code changes
- Industry standard approach (A2A protocol)

**How (High-Level):**
- Create agent discovery module
- Fetch Agent Cards from configured endpoints (`/.well-known/agent-card.json`)
- Parse capabilities, endpoint, version from cards
- Cache cards with TTL (refresh periodically)
- Provide discovery API for planning agent to query available agents
- Handle agent unavailability gracefully (discovery failures)

**Deliverables:**
- Agent discovery module
- Cached agent registry (in-memory or Redis)
- Can discover reasoning agent's capabilities
- API to query available agents

---

#### Milestone 5: Planning Agent

**What:**
Create planning agent that analyzes user requests and generates workflow plans. Determines which agents to invoke, in what order, and identifies parallel execution opportunities.

**Why:**
- Separates planning logic from execution
- Enables intelligent multi-agent coordination
- Handles complex queries requiring multiple agents
- Optimizes workflows (parallelization, minimal steps)
- Critical for orchestrator to know what to execute

**How (High-Level):**
- Create new FastAPI service in `services/planning_agent/`
- Implement A2A protocol (Agent Card, task endpoints, SSE)
- Use LLM (with structured output) to analyze user intent
- Fetch available agents from discovery system
- Match user intent to agent capabilities
- Generate workflow plan: list of steps, dependencies, parallel flags
- Validate plan: no circular dependencies, valid agent references
- Plan stored as structured data (Pydantic model)

**Deliverables:**
- Planning agent service
- Workflow plan schema (steps, dependencies, parallel execution)
- Plan generation using LLM
- Plan validation logic
- Planning agent publishes Agent Card

---

#### Milestone 6: Orchestrator Implementation

**What:**
Implement orchestrator service that executes workflow plans by coordinating multiple agents via A2A protocol. Handles parallel execution, error handling, and aggregates results.

**Why:**
- Core of multi-agent system
- Executes plans from planning agent
- Coordinates multiple agents concurrently
- Aggregates results from different agents
- Handles workflow lifecycle and state

**How (High-Level):**
- Create new FastAPI service in `services/orchestrator/`
- Implement A2A protocol (Agent Card, task endpoints, SSE)
- Receive workflow plan from planning agent
- Execute steps in dependency order
- Use asyncio.gather for parallel steps
- Subscribe to multiple agent SSE streams concurrently
- Aggregate artifacts from all agents
- Store workflow state in Redis (current step, completed steps, artifacts)
- Stream aggregated progress to client
- Handle agent failures and retries

**Deliverables:**
- Orchestrator service
- Workflow execution engine (sequential and parallel)
- Multi-agent streaming aggregation
- State persistence for workflows
- Error handling and retry logic
- Update main API to delegate to orchestrator instead of reasoning agent directly

---

#### Milestone 7: Human-in-the-Loop with A2A auth-required

**What:**
Implement human approval workflows using A2A `auth-required` state. Agents or orchestrator can pause execution, request approval, and resume after user approves.

**Why:**
- Critical for production AI systems (safety, oversight)
- Required for sensitive operations (data deletion, expensive actions)
- Enables trust and control for users
- A2A protocol natively supports this via task states

**How (High-Level):**
- Agent or orchestrator sets task status to `auth-required`
- Store approval context: what action, why, risks, estimated impact
- Update session state to reflect pause
- Expose approval endpoint: GET /approvals/{session_id}
- User reviews via UI or API
- User submits approval: POST /approvals/{session_id} with decision (approve/reject/modify)
- Store approval decision in session
- Resume execution: agent/orchestrator continues from paused state
- Handle timeout: if no approval within timeout, fail or use default

**Deliverables:**
- Approval workflow in orchestrator
- Approval endpoints (get pending, submit approval)
- Session state tracks approval context
- Web UI shows pending approvals
- Workflow resumes after approval
- Timeout handling

---

### Phase 3: Advanced Agents

#### Milestone 8: Vector Database Setup (PostgreSQL + pgvector)

**What:**
Set up vector database infrastructure using PostgreSQL with pgvector extension for semantic search and retrieval.

**Why:**
- Required for RAG agent
- Enables semantic search (not just keyword matching)
- Grounds responses in retrieved knowledge
- Reduces hallucinations

**How (High-Level):**
- Add PostgreSQL with pgvector to docker-compose
- Create database schema for documents and embeddings
- Choose embedding model (OpenAI embeddings API or local model)
- Implement vector operations: insert, search by similarity, filter by metadata
- Create data ingestion tool for loading documents
- Chunk documents appropriately (semantic or fixed-size)
- Generate and store embeddings with metadata

**Deliverables:**
- PostgreSQL with pgvector running
- Database schema for vectors
- Vector operations module (CRUD)
- Embedding generation module
- Data ingestion tool
- Initial documents loaded

---

#### Milestone 9: RAG Agent

**What:**
Create RAG (Retrieval Augmented Generation) agent that performs semantic search and generates context-aware responses using retrieved information.

**Why:**
- Enables knowledge base queries
- Grounds responses in retrieved documents
- Provides source citations
- Reduces hallucinations

**How (High-Level):**
- Create new FastAPI service in `services/rag_agent/`
- Implement A2A protocol (Agent Card with RAG capabilities)
- Receive query from orchestrator via A2A task
- Generate query embedding
- Search vector database for relevant documents
- Optionally rerank results
- Construct LLM context from top results
- Generate response using LLM with retrieved context
- Stream response via SSE with citations
- Return artifacts with source attribution

**Deliverables:**
- RAG agent service
- Semantic search integration with vector DB
- Context construction logic
- Response generation with citations
- RAG agent publishes Agent Card
- Planning agent can discover and use RAG agent

---

### Phase 4: Production Readiness

#### Milestone 10: Observability and Monitoring

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

#### Milestone 11: Error Handling and Resilience

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

#### Milestone 12: Performance Optimization

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
