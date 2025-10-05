# Implementation Plan: Dapr Orchestration & RAG System

## Project Overview

### Current State
- Single FastAPI service with embedded ReasoningAgent
- All state in-memory (lost on restart)
- No service-to-service communication capability
- No persistent state management for workflows (e.g. human-in-the-loop)
- Tools abstracted via MCP protocol
- OpenTelemetry tracing to Phoenix

### Target State
- Dapr-based distributed architecture
- Orchestrator agent that plans and coordinates multi-agent workflows
- RAG agent for semantic search and retrieval
- Persistent state management for long-running workflows
- Human-in-the-loop workflow support
- Multiple specialized agents communicating via Dapr

### Key Architectural Decisions

**Why Docker Compose:**
- Provides local development environment matching production
- Manages multiple services (API, agents, Redis, PostgreSQL, Phoenix)
- Easy to add new services (new agents, databases)
- Consistent with your existing setup
- Good foundation for eventual Kubernetes migration

**Why Dapr:**
- Provides state persistence needed for human-in-the-loop workflows
- Handles service discovery for multi-agent communication
- Built-in retries, circuit breakers, and resilience patterns
- Event-driven pub/sub for workflow coordination
- Actor model perfect for per-request orchestrator instances
- Language-agnostic (can add agents in any language)

**Why Actor Model for Orchestrator:**
- One actor instance per orchestration session
- Automatic state persistence and recovery
- Natural fit for workflow with multiple steps
- Built-in support for pause/resume (human-in-the-loop)
- Distributed across multiple instances automatically

**No Backwards Compatibility:**
- Breaking changes are encouraged for better architecture
- Remove legacy patterns that conflict with Dapr
- Prioritize clean design over migration paths
- Focus on production-ready patterns from the start

---

## Phase 1: Dapr Infrastructure Transformation

### Milestone 1: Dapr Foundation and Local Development Setup

#### Objective
Establish Dapr infrastructure locally with proper configuration, without modifying existing application code. This creates the foundation for all subsequent work.

#### Why This First
- Tests Dapr installation and configuration independently
- Establishes patterns for components and configuration
- Validates local development workflow
- Provides foundation for all subsequent milestones
- Low risk - no code changes yet

#### Prerequisites and Learning
**Required Reading Before Implementation:**
- [Dapr Overview](https://docs.dapr.io/concepts/overview/) - Core concepts
- [Dapr Components](https://docs.dapr.io/concepts/components-concept/) - How components work
- [State Management API](https://docs.dapr.io/developing-applications/building-blocks/state-management/state-management-overview/) - State building block
- [Pub/Sub API](https://docs.dapr.io/developing-applications/building-blocks/pubsub/pubsub-overview/) - Messaging building block
- [Service Invocation API](https://docs.dapr.io/developing-applications/building-blocks/service-invocation/service-invocation-overview/) - Service communication

**Install Dapr CLI:**
```bash
# macOS/Linux
curl -fsSL https://raw.githubusercontent.com/dapr/cli/master/install/install.sh | /bin/bash

# Initialize Dapr locally (installs Redis, Zipkin, Placement service)
dapr init

# Verify installation
dapr --version
```

#### Implementation Tasks

##### 1. Create Dapr Directory Structure
Create `dapr/` directory with subdirectories for `components/` and `config/`. This separates Dapr configuration from application code and makes environment management easier.

##### 2. Configure State Store Component
Create a state store component YAML that uses Redis. This will provide persistent state for sessions, workflows, and agent results.

**Why Redis for development:** Simple setup, good performance, matches Dapr defaults. Can be swapped for PostgreSQL in production with just configuration changes.

**Key decisions to make:**
- Should state have TTL (time-to-live)?
- Do we want Redis persistence enabled?
- Connection pooling settings?

##### 3. Configure Pub/Sub Component
Create a pub/sub component using Redis Streams. This enables event-driven communication between agents (e.g., "reasoning complete" → trigger next agent).

**Why Redis Streams:** Built-in, works immediately, good for development. Can switch to Kafka for production scale.

##### 4. Configure Dapr Runtime Settings
Create a Dapr configuration file that:
- Points tracing to your existing Phoenix instance (`http://phoenix:4317`)
- Enables actor runtime (needed for orchestrator in Milestone 5)
- Sets appropriate timeouts and limits

**Important:** Make sure OTEL tracing endpoint matches your Phoenix service name in docker-compose.

##### 5. Update Docker Compose
Add Redis service and Dapr sidecar for the main API.

**Pattern to follow:**
- Each application service gets a companion Dapr sidecar
- Sidecar runs as separate container with specific ports (3500 for HTTP, 50001 for gRPC)
- Main app communicates with sidecar via localhost
- Sidecar handles all external communication

**Port allocation strategy:** 
- reasoning-api: 8000 (app), 3500 (Dapr HTTP), 50001 (Dapr gRPC)
- Future services will increment (3501, 50002, etc.)

##### 6. Update Dependencies
Add Dapr SDK to `pyproject.toml` in the `api` dependency group. Use version 1.14.0 or higher for latest actor improvements.

#### Testing Strategy

**Validation criteria:**
- Dapr CLI commands work without errors
- Can start all services without port conflicts
- State store component loads successfully (check logs)
- Can save and retrieve test data via Dapr API directly
- Existing API functionality unchanged
- Phoenix receives traces from Dapr-instrumented services

**What to test:**
1. Dapr installation - `dapr list` should show running services
2. Component loading - Docker logs should show successful component initialization
3. State operations - Use curl to save/retrieve data via Dapr HTTP API
4. Service health - All containers start and pass health checks
5. End-to-end - Existing chat completions endpoint still works
6. Tracing - Phoenix UI shows traces from Dapr services

**Success criteria:**
- Zero errors in Dapr sidecar logs
- Can perform basic state operations
- No regression in existing API behavior
- Tracing spans appear in Phoenix with Dapr metadata

#### Documentation Updates
Create or update:
- `README.md` with Dapr setup instructions
- `docs/dapr-architecture.md` explaining component choices
- Port allocation scheme documentation
- Troubleshooting guide for common Dapr issues

#### Questions to Ask Before Proceeding
1. Should we use Redis or PostgreSQL for the state store from the start?
2. Do we want persistent volumes for Redis data between container restarts?
3. Should we enable mTLS between services now or wait until production?

---

### Milestone 2: State Management Integration

#### Objective
Integrate Dapr state management into existing ReasoningAgent to enable session persistence and recovery. This is foundational for human-in-the-loop workflows.

#### Why This Next
- Builds on Dapr foundation from Milestone 1
- Provides immediate value (resumable sessions)
- Tests Dapr integration with existing code
- Foundation for orchestrator state management
- Enables the critical pause/resume capability you need

#### Prerequisites
- Milestone 1 complete and validated
- **Read:** [State Management How-To](https://docs.dapr.io/developing-applications/building-blocks/state-management/howto-get-save-state/)
- **Read:** [State Management Best Practices](https://docs.dapr.io/developing-applications/building-blocks/state-management/state-management-overview/#best-practices)

#### Key Concepts

**State Management Strategy:**
You need to decide on a session state schema that captures:
- User request and session ID
- Current reasoning context (steps taken, tool results)
- Workflow status (processing, awaiting approval, completed)
- Timestamps for tracking and TTL

**Why sessions are critical:** Without persistent sessions, you cannot implement human-in-the-loop workflows. If the API restarts mid-request or while waiting for human approval, all progress is lost.

#### Implementation Tasks

##### 1. Create Session Management Module
Build a `api/session_manager.py` module that abstracts all Dapr state operations.

**Purpose:** 
- Single place for all session state logic
- Clean interface for reasoning agent to use
- Easy to test and mock
- Can swap storage backends later

**Key interface to implement:**
```python
class SessionManager:
    async def create_session(self, request: OpenAIChatRequest) -> str:
        """Create new session, return session_id"""
        
    async def save_session(self, session_id: str, state: SessionState):
        """Persist complete session state"""
        
    async def load_session(self, session_id: str) -> Optional[SessionState]:
        """Load session state, returns None if not found"""
```

**Design decisions:**
- Should you save complete state every time or use incremental updates?
- How to handle concurrent updates to same session?
- What should the state key format be? (e.g., `session-{id}` vs `session/{id}`)

##### 2. Define Session State Schema
Create Pydantic models for session state that include all necessary fields for resuming work.

**What to include:**
- Original user request
- Current status enum (processing, awaiting_approval, completed, error)
- Reasoning context (steps, tool results, current iteration)
- Timestamps and metadata
- Any results produced so far

**Consider:** Should the schema be versioned for future migrations?

##### 3. Update ReasoningAgent for State Persistence
Modify the `ReasoningAgent` class to:
- Accept a `session_id` parameter
- Load existing session state if resuming
- Save progress after each reasoning iteration
- Handle session not found gracefully

**Pattern to implement:**
After each iteration in `_core_reasoning_process`, save the current state. This ensures minimal data loss if interrupted. Consider the performance impact - you may want to batch state updates.

**Key decision:** When exactly should state be saved? After every tool execution? After each reasoning step? Balance between durability and performance.

##### 4. Update FastAPI Endpoints
Modify `api/main.py` to support session tracking:

**Changes needed:**
- Extract `session_id` from `X-Session-ID` header or generate new one
- Return `session_id` in response headers for client tracking
- Add endpoint to query session status
- Add endpoint to cancel/delete sessions

**Example pattern:**
```python
@app.post("/v1/chat/completions")
async def chat_completions(request: OpenAIChatRequest, http_request: Request):
    # Get or create session_id
    session_id = http_request.headers.get("X-Session-ID") or str(uuid.uuid4())
    
    # Pass to reasoning agent...
    # Return session_id in response headers
```

##### 5. Update Dependencies
Add `SessionManager` to the dependency injection system so it's available throughout the application.

#### Testing Strategy

**What to test:**

1. **Session lifecycle:** Create, save, load, delete operations work correctly
2. **State persistence:** Data survives service restarts (stop/start Docker containers)
3. **Resume capability:** Can continue reasoning from saved state without repeating steps
4. **Concurrent access:** Multiple requests can read same session safely
5. **Error handling:** Corrupted state, missing sessions, Dapr unavailable scenarios
6. **Performance:** State save/load operations complete within acceptable time (< 50ms)

**Success criteria:**
- Session state correctly persisted to Redis via Dapr
- Can resume interrupted reasoning from last saved step
- All reasoning steps preserved (no duplication or loss)
- Session endpoints return correct status
- No data races or corruption with concurrent access
- Performance impact acceptable

**Edge cases to test:**
- Very large session state (many tool results)
- Session doesn't exist (returns 404)
- Invalid session_id format
- Dapr state store temporarily unavailable
- Concurrent updates to same session

#### Error Handling Considerations

**Scenarios to handle:**
1. Session not found - return 404 with helpful message
2. Corrupted session state - log error, create new session
3. State store unavailable - retry with backoff, eventually fail gracefully
4. Concurrent modifications - use Dapr's etag feature for optimistic locking

#### Documentation Updates
- Document session state schema
- Add examples of resuming sessions
- Update API docs with new session endpoints
- Create troubleshooting guide for session issues

#### Questions to Ask Before Proceeding
1. What should be the default TTL for sessions?
2. Should we implement session state compression for large contexts?
3. Do we want a session listing/search endpoint?
4. How should we handle session cleanup (automatic vs manual)?

---

### Milestone 3: Service Invocation and Multi-Service Architecture

#### Objective
Extract ReasoningAgent into a separate Dapr service and update main API to use Dapr service invocation. This establishes the foundation for multi-agent architecture.

#### Why This Next
- Proves service-to-service communication works
- Enables independent scaling of reasoning agent
- Establishes pattern for adding more agent types
- Tests distributed tracing across services
- Critical prerequisite for orchestrator

#### Prerequisites
- Milestone 2 complete (state management working)
- **Read:** [Service Invocation Overview](https://docs.dapr.io/developing-applications/building-blocks/service-invocation/service-invocation-overview/)
- **Read:** [Service-to-Service Communication](https://docs.dapr.io/developing-applications/building-blocks/service-invocation/howto-invoke-discover-services/)

#### Architecture Transformation

**Before:** Single service with embedded reasoning agent
**After:** Two services communicating via Dapr

**Why separate the reasoning agent:**
- Can scale reasoning independently from API layer
- Better fault isolation (agent crash doesn't kill API)
- Easier to add more agent types later
- Clear separation of concerns
- Can deploy/update agents independently

#### Implementation Tasks

##### 1. Create Reasoning Agent Service
Create a new service directory structure: `services/reasoning_agent/`

**What this service needs:**
- FastAPI app with single `execute` endpoint
- Move reasoning agent code from `api/` directory
- Health check endpoint for service discovery
- Proper error handling and timeouts
- Uses same session manager for state

**Key design question:** Should the service be completely stateless (always load from state store) or maintain some in-memory cache?

##### 2. Define Service Interface
Design a clean request/response contract between main API and reasoning service.

**Request should include:**
- User's OpenAI request
- Session ID
- Any context from orchestrator (for future use)

**Response should include:**
- Reasoning result
- Session status
- Any metadata (token usage, duration, etc.)

**Pattern example:**
```python
# Reasoning service endpoint
@app.post("/execute")
async def execute_reasoning(request: ReasoningExecutionRequest):
    """
    Execute reasoning for given request and session.
    
    Loads session state, performs reasoning, saves progress.
    Returns result or status if paused.
    """
    # Implementation here
```

##### 3. Update Main API for Service Invocation
Modify the chat completions endpoint to invoke reasoning service via Dapr instead of calling it directly.

**Why Dapr invocation over direct HTTP:**
- No hardcoded URLs or ports
- Automatic retries with exponential backoff
- Circuit breaker pattern built-in
- mTLS encryption between services
- Distributed tracing across service boundary
- Load balancing if multiple instances

**Pattern:**
```python
from dapr.clients import DaprClient

async with DaprClient() as dapr:
    result = await dapr.invoke_method(
        app_id="reasoning-agent",  # Service name, not URL
        method_name="execute",
        data=request_data,
        http_verb="POST"
    )
```

##### 4. Implement Resilience Wrapper
Create a service client wrapper that adds error handling, retries, and circuit breaking on top of Dapr invocation.

**Why:** While Dapr provides basic retries, you may want application-level retry logic and custom error handling.

**Responsibilities:**
- Retry transient failures (503, network errors)
- Circuit breaker state management
- Timeout handling
- Error mapping and logging

##### 5. Add Docker Compose Configuration
Add the reasoning agent service and its Dapr sidecar to docker-compose.yml.

**Important:** Each service needs its own Dapr sidecar with unique app-id and ports.

##### 6. Create Dockerfile
Build a Dockerfile for the reasoning agent service. Should be similar to your existing API Dockerfile but point to the new service directory.

#### Testing Strategy

**What to test:**

1. **Service discovery:** Main API can find reasoning service via app-id (no URLs)
2. **Cross-service execution:** Complete request flow through both services
3. **Service failures:** Graceful handling when reasoning service is down
4. **Distributed tracing:** Traces span both services with parent-child relationship
5. **State sharing:** Both services can access session state
6. **Performance:** Service invocation overhead is acceptable (< 10ms)
7. **Concurrent requests:** Multiple requests handled correctly

**Success criteria:**
- Reasoning agent runs as independent service
- Main API successfully invokes it via Dapr
- Distributed traces show complete request path
- Error scenarios handled gracefully (service down, timeout, etc.)
- No regression in functionality
- Performance impact minimal

**Failure scenarios to test:**
- Reasoning service completely down
- Reasoning service slow to respond (timeout)
- Network partition between services
- Service returning errors

#### Error Handling Strategy

**Service invocation errors to handle:**
1. Service unavailable (503) - retry with backoff
2. Timeout (504) - return error, allow retry
3. Invalid response - log and return 500
4. Circuit breaker open - return cached result or error

**Error response format:** Design consistent error responses that clients can understand and handle appropriately.

#### Observability Requirements

**What to log and trace:**
- Service invocation attempts and results
- Latency between services
- Circuit breaker state changes
- Error rates per service

**Metrics to track:**
- Service-to-service latency
- Error rate
- Request volume
- Circuit breaker state

#### Documentation Updates
- Add architecture diagram showing service separation
- Document service invocation patterns
- Add troubleshooting guide for cross-service issues
- Update deployment guide for multi-service setup

#### Questions to Ask Before Proceeding
1. Should we add authentication between services beyond mTLS?
2. What should be the default timeout for reasoning service calls?
3. Do we want versioning in the service interface?
4. Should we add load balancing configuration for multiple reasoning instances?

---

## Phase 2: Orchestration Agent

### Milestone 4: Workflow Planning Agent

#### Objective
Create a planning agent that analyzes user requests and generates executable workflow plans (DAGs). This is the "brain" that decides which agents to invoke and in what order.

#### Why This Milestone
- Separates planning logic from execution
- Enables intelligent agent selection
- Foundation for dynamic multi-agent coordination
- Critical for complex query handling

#### Prerequisites
- Milestone 3 complete (service invocation working)
- **Read:** [Prefect ControlFlow Documentation](https://controlflow.ai/guides/quickstart) - For workflow patterns
- **Read:** [LangGraph Planning Patterns](https://langchain-ai.github.io/langgraph/concepts/planning/) - For planning concepts

#### Planning Agent Responsibilities

**What it does:**
- Analyzes user query to understand intent
- Identifies required capabilities (search, reasoning, RAG, etc.)
- Selects appropriate agents based on capabilities
- Determines optimal execution order
- Identifies parallelization opportunities
- Estimates cost and duration

**What it produces:**
A workflow plan (DAG) specifying which agents to run, dependencies between steps, and execution strategy (sequential vs parallel).

#### Workflow Plan Schema

**Key components:**
- List of steps, each with: agent name, parameters, dependencies
- Execution metadata: estimated duration, cost, requires approval
- DAG structure: which steps can run in parallel

**Example structure concept:**
```python
# Step with no dependencies runs first
# Steps with same dependencies can run in parallel
# Final step depends on all previous steps
```

#### Implementation Tasks

##### 1. Create Planning Agent Service
Build a new service in `services/planning_agent/` with FastAPI.

**Endpoints needed:**
- `POST /plan` - Generate workflow plan for a request
- `POST /validate` - Validate a workflow plan's correctness
- `POST /optimize` - Optimize existing plan (find more parallelization)

**Design decision:** Should the planner be stateless or cache plans for similar queries?

##### 2. Define Workflow Models
Create Pydantic models for workflow plans.

**Models needed:**
- `WorkflowStep` - Single step in workflow
- `WorkflowPlan` - Complete execution plan
- `AgentCapability` - Agent capabilities registry

**Key fields for WorkflowStep:**
- Unique step_id
- Agent to invoke
- Dependencies (list of step_ids that must complete first)
- Can run in parallel (boolean)
- Parameters to pass to agent

##### 3. Implement Planning Logic
Build the core planning algorithm that takes user request and produces workflow plan.

**Strategy to implement:**
1. Use LLM (with structured output) to analyze user intent
2. Match intent to available agent capabilities
3. Determine optimal agent selection
4. Build dependency graph
5. Identify parallelization opportunities
6. Validate plan correctness

**Key consideration:** Should you use prompt engineering with structured outputs or a fine-tuned model? Start with prompting.

**Planning prompt strategy:**
- Provide list of available agents and their capabilities
- Give examples of good workflow plans
- Request JSON output matching your schema
- Include optimization hints (prefer parallel where possible)

##### 4. Implement Plan Validation
Create validation logic that checks:
- No circular dependencies
- All referenced step_ids exist
- Parallel steps don't depend on each other
- At least one entry point (step with no dependencies)
- Selected agents have required capabilities

**Why validation matters:** Invalid plans will cause orchestrator to fail. Better to catch issues at planning time.

##### 5. Add Agent Capability Registry
Build a registry that tracks available agents and their capabilities.

**Information to store:**
- Agent ID and description
- Capabilities list (search, reasoning, rag, etc.)
- Cost/latency characteristics
- Input/output schemas

**Design decision:** Should this be hardcoded, configuration file, or dynamic discovery via service queries?

##### 6. Add to Docker Compose
Add planning service and its Dapr sidecar.

#### Testing Strategy

**What to test:**

1. **Simple plans:** Single-agent workflows for simple queries
2. **Complex plans:** Multi-agent workflows with proper dependencies
3. **Parallelization:** Identifies steps that can run concurrently
4. **Plan validation:** Catches circular dependencies and invalid references
5. **Plan optimization:** Finds additional parallelization opportunities
6. **Edge cases:** Ambiguous queries, no suitable agents, etc.

**Success criteria:**
- Simple queries generate single-step plans
- Complex queries generate appropriate multi-step plans
- Parallel opportunities correctly identified
- All generated plans pass validation
- Plans are executable by orchestrator

**Test plan types:**
- Simple query → single reasoning agent
- Search query → reasoning + search agents
- Comparison query → reasoning + parallel searches + synthesis
- Knowledge query → reasoning + RAG agent

#### Plan Quality Metrics

**How to evaluate plan quality:**
- Correctness: Does it solve the user's query?
- Efficiency: Minimal steps, maximum parallelization
- Cost: Uses appropriate (not unnecessarily expensive) agents
- Time: Estimated duration reasonable

#### Error Handling

**Planning failures to handle:**
1. Cannot determine user intent → ask for clarification
2. No agents with required capabilities → explain limitation
3. Too complex to plan → break down or request simplification
4. Ambiguous request → generate multiple plan options

#### Documentation Updates
- Document planning algorithm approach
- Add workflow plan schema documentation
- Include example plans for common query types
- Document agent capability registration process

#### Questions to Ask Before Proceeding
1. Should planning agent cache plans for similar queries?
2. How should we handle ambiguous queries (multiple possible plans)?
3. Should users be able to provide custom workflow plans?
4. Do we want plan versioning for iterative refinement?

---

### Milestone 5: Orchestrator Actor Implementation

#### Objective
Implement Dapr actor-based orchestrator that executes workflow plans, coordinates agents, and manages long-running workflows with pause/resume capability.

#### Why Actors for Orchestration
- One actor instance per workflow session
- Automatic state persistence
- Built-in pause/resume for human-in-the-loop
- Survives restarts
- Distributed automatically across instances

**Key benefit:** Actors give you stateful, per-session orchestration with minimal code. The actor runtime handles persistence, recovery, and lifecycle management.

#### Prerequisites
- Milestone 4 complete (planning agent working)
- **Read:** [Dapr Actors Overview](https://docs.dapr.io/developing-applications/building-blocks/actors/actors-overview/)
- **Read:** [Actors How-To](https://docs.dapr.io/developing-applications/building-blocks/actors/howto-actors/)
- **Read:** [Actor Best Practices](https://docs.dapr.io/developing-applications/building-blocks/actors/actors-features-concepts/)

#### Actor Responsibilities

**What the orchestrator actor does:**
1. Receives workflow plan from planning agent
2. Executes steps in dependency order
3. Manages parallel vs sequential execution
4. Collects results from each agent
5. Handles errors and retries
6. Pauses for human approval when needed
7. Resumes after approval
8. Aggregates final results

**Actor state to maintain:**
- Current workflow plan
- Completed vs pending steps
- Results from each step
- Current status (running, paused, completed, error)
- Error information if failed

#### Implementation Tasks

##### 1. Create Orchestrator Service
Build new service in `services/orchestrator/` with actor hosting.

**Critical components:**
- FastAPI app that registers and hosts actors
- Actor class implementing orchestration logic
- Actor interface defining methods
- State management for workflow execution

**Dapr actor requirements:**
- Must inherit from `Actor` base class
- Must implement `ActorInterface`
- Methods must use `@ActorMethod` decorator
- State saved via `_state_manager`

##### 2. Define Actor Interface
Create interface specifying orchestrator methods.

**Methods needed:**
```python
class IOrchestratorActor(ActorInterface):
    @ActorMethod(name="StartWorkflow")
    async def start_workflow(self, request: dict) -> dict:
        """Initialize and begin workflow execution"""
        
    @ActorMethod(name="GetStatus")
    async def get_status(self) -> dict:
        """Query current workflow status"""
        
    @ActorMethod(name="ApproveAndContinue")
    async def approve_and_continue(self, approval: dict) -> dict:
        """Handle approval and resume execution"""
        
    @ActorMethod(name="Cancel")
    async def cancel(self) -> dict:
        """Cancel in-progress workflow"""
```

##### 3. Implement Orchestrator Actor Class
Build the actor class with core workflow execution logic.

**Key methods to implement:**

**_on_activate:** Load workflow state when actor activates
**_on_deactivate:** Save workflow state when actor deactivates
**start_workflow:** Initialize state and begin execution
**_execute_workflow:** Core loop that processes workflow steps
**_execute_step:** Execute single step via service invocation
**_find_ready_steps:** Identify steps whose dependencies are satisfied

**Critical algorithm:** Workflow execution loop
```python
# Pseudocode for main execution loop:
while pending_steps:
    # Find steps ready to run (dependencies satisfied)
    ready_steps = find_steps_where_all_dependencies_completed()
    
    # Separate parallel vs sequential
    parallel_steps = [s for s in ready_steps if s.parallel]
    sequential_steps = [s for s in ready_steps if not s.parallel]
    
    # Execute parallel steps concurrently
    if parallel_steps:
        results = await gather_parallel_execution(parallel_steps)
        
    # Execute sequential steps one by one
    for step in sequential_steps:
        result = await execute_via_dapr_invocation(step)
        
        # Check if needs human approval
        if result.requires_approval:
            save_state_and_pause()
            return
            
    # Update completed/pending lists
```

**State management pattern:**
Save state after each step completion, not just at end. This ensures minimal data loss if orchestrator crashes mid-execution.

##### 4. Implement Actor Hosting in FastAPI
Set up FastAPI app to host Dapr actors.

**Setup requirements:**
- Import `DaprActor` from `dapr.ext.fastapi`
- Register actor types on startup
- Create endpoints for interacting with actors

**Pattern:**
```python
from dapr.ext.fastapi import DaprActor

app = FastAPI()
actor_runtime = DaprActor(app)

@app.on_event("startup")
async def register_actors():
    await actor_runtime.register_actor(OrchestratorActor)
```

##### 5. Create Orchestration Endpoints
Build REST endpoints that interact with orchestrator actors.

**Endpoints needed:**
- `POST /orchestrate` - Start new orchestration (creates actor)
- `GET /orchestration/{session_id}/status` - Query status (calls actor)
- `POST /orchestration/{session_id}/approve` - Approve action (calls actor)
- `DELETE /orchestration/{session_id}` - Cancel (calls actor)

**Actor proxy pattern:**
```python
from dapr.actor import ActorProxy, ActorId

# Create proxy to talk to specific actor instance
proxy = ActorProxy.create(
    "OrchestratorActor",
    ActorId(session_id),
    IOrchestratorActor
)

# Invoke methods
result = await proxy.StartWorkflow(data)
```

##### 6. Update Main API
Modify chat completions endpoint to use orchestrator instead of calling reasoning agent directly.

**New flow:**
1. Receive chat completion request
2. Create orchestrator actor for session
3. Actor gets plan from planning agent
4. Actor executes workflow
5. Return results to client

##### 7. Implement Event Streaming
Add real-time orchestration status updates via SSE.

**Approach options:**
1. Poll orchestrator status periodically
2. Subscribe to Dapr pub/sub events from orchestrator
3. Hybrid: polling with event notifications

**Consider:** Orchestrator can publish events when step completes, allowing clients to stream progress without polling.

##### 8. Add to Docker Compose
Add orchestrator service with Dapr sidecar configured for actor runtime.

**Important:** Actor configuration requires placement service (already in Dapr init).

#### Testing Strategy

**What to test:**

1. **Basic execution:** Simple workflow completes successfully
2. **Parallel execution:** Parallel steps run concurrently (check timing)
3. **Dependency handling:** Steps wait for dependencies before running
4. **State persistence:** Actor state survives service restart
5. **Pause/resume:** Can pause for approval and resume later
6. **Error handling:** Gracefully handles agent failures
7. **Cancellation:** Can cancel in-progress workflows
8. **Concurrent orchestrations:** Multiple sessions run independently

**Success criteria:**
- Workflows execute according to plan
- Parallel steps complete faster than sequential
- State persists across restarts
- Can pause and resume at any point
- Failed steps handled gracefully
- Multiple sessions don't interfere

**Complex scenarios to test:**
- Workflow with 5+ steps, mix of parallel and sequential
- Resume after service restart mid-workflow
- Human approval pauses workflow, resumes hours later
- One agent fails, workflow handles gracefully
- Cancel workflow mid-execution

#### Error Handling Strategy

**Failure scenarios to handle:**

1. **Agent failure:** Retry with backoff, eventually fail step
2. **Timeout:** Mark step as failed, option to retry
3. **Invalid response:** Log error, fail step
4. **Planning failure:** Return error before starting execution
5. **State corruption:** Attempt recovery or fail cleanly

**Retry strategy:** Exponential backoff with max attempts (e.g., 3 retries with 1s, 2s, 4s delays).

**Failure propagation:** Decide whether agent failure should fail entire workflow or continue with partial results.

#### Performance Considerations

**Optimization opportunities:**
- Batch state saves (don't save after every operation)
- Cache planning results for similar queries
- Parallel step execution (use asyncio.gather)
- Lazy result loading (don't load full history each time)

**Monitor:** Actor activation time, step execution time, state save time.

#### Documentation Updates
- Add orchestrator architecture diagram
- Document actor lifecycle and state management
- Create workflow execution examples
- Add troubleshooting guide for orchestration issues

#### Questions to Ask Before Proceeding
1. How should we handle partial workflow failures (fail all vs continue)?
2. What's the appropriate timeout for each step?
3. Should we support workflow checkpointing (save intermediate results)?
4. Do we want workflow retry at orchestrator level or just step level?

---

## Phase 3: RAG Agent and Vector Database

### Milestone 6: Vector Database Setup

#### Objective
Set up vector database infrastructure for semantic search and retrieval. This enables the RAG agent to find relevant context from stored knowledge.

#### Why Vector Database
- Semantic search (find similar concepts, not just keyword matches)
- Required for RAG (Retrieval Augmented Generation)
- Enables agent to access relevant context
- Improves answer quality with grounded information

#### Prerequisites
- Previous milestones complete (orchestrator working)
- **Read:** [Chroma Documentation](https://docs.trychroma.com/) - For local development
- **Read:** [pgvector Documentation](https://github.com/pgvector/pgvector) - Alternative using PostgreSQL
- **Read:** [Vector Database Comparison](https://github.com/superlinked/VectorHub) - Understanding options

#### Vector Database Selection

**Options to consider:**

1. **Chroma** - Python-native, simple for development
2. **pgvector** - PostgreSQL extension, leverages existing DB
3. **Weaviate** - Production-grade, more features
4. **Qdrant** - High performance, good for large scale

**Recommendation:** Start with **pgvector** since you already have PostgreSQL for Phoenix.

**Why pgvector:**
- Already have PostgreSQL running
- No additional service needed
- Good performance
- Easy backup/restore
- Can upgrade to dedicated vector DB later

#### Implementation Tasks

##### 1. Add pgvector Extension
Enable pgvector extension in your existing PostgreSQL database.

**Steps:**
- Add pgvector to PostgreSQL image or install at runtime
- Create dedicated database/schema for vectors
- Enable extension with `CREATE EXTENSION vector`

##### 2. Create Vector Storage Schema
Design database schema for storing embeddings and metadata.

**Tables needed:**
- Documents table (id, content, metadata, embedding vector)
- Collections/namespaces for organizing documents
- Metadata indexes for filtering

**Key design decisions:**
- Vector dimensions (match your embedding model, e.g., 1536 for OpenAI)
- Metadata schema (JSON column for flexibility)
- Indexing strategy (HNSW vs IVFFlat)

##### 3. Add Embedding Service
Create service or module that generates embeddings from text.

**Options:**
- Use OpenAI embeddings API
- Run local model (sentence-transformers)
- Dedicated embedding service

**Considerations:**
- Cost (OpenAI charges per token)
- Latency (API vs local)
- Quality (OpenAI generally better)

##### 4. Implement Vector Operations
Build basic CRUD operations for vectors:
- Insert documents with embeddings
- Search by similarity
- Filter by metadata
- Update/delete documents

**Core functionality:**
```python
# Insert document
await vector_store.add_document(
    content="document text",
    metadata={"source": "...", "type": "..."},
    embedding=get_embedding(text)
)

# Similarity search
results = await vector_store.search(
    query_embedding=get_embedding(query),
    k=5,  # top 5 results
    filter={"type": "documentation"}
)
```

##### 5. Add Dapr State Component for Vector Metadata
Optionally use Dapr state management for vector metadata (embeddings stored in PostgreSQL, metadata in Dapr state).

**Why split:** Fast metadata access via Dapr, efficient vector search in PostgreSQL.

##### 6. Update Docker Compose
Modify PostgreSQL service to include pgvector and create dedicated volume for vector data.

**Consider:** Separate database for vectors vs Phoenix data, or same database with different schemas?

##### 7. Create Data Ingestion Tool
Build utility for loading initial data into vector database.

**Features:**
- Load documents from files/directories
- Chunk large documents appropriately
- Generate embeddings
- Store with metadata

**Design decision:** Should this be CLI tool, API endpoint, or both?

#### Testing Strategy

**What to test:**

1. **Vector operations:** Insert, search, update, delete work correctly
2. **Similarity search:** Returns relevant results for queries
3. **Metadata filtering:** Can filter results by metadata fields
4. **Performance:** Search completes in acceptable time (< 100ms for small datasets)
5. **Scalability:** Performance with increasing data size
6. **Concurrent access:** Multiple searches simultaneously

**Success criteria:**
- Can store and retrieve documents
- Similarity search returns semantically related results
- Search latency acceptable
- Database handles expected data volume
- All CRUD operations work correctly

**Test scenarios:**
- Insert 100 documents, verify searchable
- Search with various queries, check relevance
- Filter by metadata, verify results match
- Measure search time with different dataset sizes

#### Data Considerations

**Initial data sources:**
- Project documentation (your README files)
- Code documentation
- API documentation
- Any relevant knowledge base

**Chunking strategy:** Decide how to split large documents:
- Fixed size chunks (e.g., 1000 characters)
- Semantic chunks (paragraphs, sections)
- Overlapping chunks for context

**Metadata to store:**
- Document source
- Creation/update timestamps
- Document type
- Any domain-specific tags

#### Performance Tuning

**Optimization strategies:**
- Choose appropriate index type (HNSW for accuracy, IVFFlat for speed)
- Tune index parameters
- Batch operations when possible
- Cache embeddings to avoid regeneration

**Monitor:** Query latency, index build time, storage size.

#### Documentation Updates
- Document vector database setup
- Add data ingestion guide
- Document search API
- Include performance tuning notes

#### Questions to Ask Before Proceeding
1. Should we use OpenAI embeddings or local model?
2. What chunking strategy for long documents?
3. Do we need real-time indexing or batch updates?
4. What metadata fields should we support?

---

### Milestone 7: RAG Agent Implementation

#### Objective
Create RAG (Retrieval Augmented Generation) agent that performs semantic search and generates context-aware responses using retrieved information.

#### Why RAG Agent
- Grounds responses in retrieved knowledge
- Reduces hallucinations
- Enables knowledge base queries
- Provides source attribution

#### Prerequisites
- Milestone 6 complete (vector database working)
- **Read:** [RAG Best Practices](https://www.anthropic.com/index/contextual-retrieval)
- **Read:** [Advanced RAG Techniques](https://blog.llamaindex.ai/a-cheat-sheet-and-some-recipes-for-building-advanced-rag-803a9d94c41b)

#### RAG Agent Architecture

**RAG flow:**
1. Receive query from orchestrator
2. Generate embedding for query
3. Search vector database for relevant documents
4. Rerank results (optional but recommended)
5. Construct context from top results
6. Generate response using LLM with retrieved context
7. Return response with source citations

#### Implementation Tasks

##### 1. Create RAG Agent Service
Build new service in `services/rag_agent/` following pattern from reasoning agent.

**Endpoints:**
- `POST /execute` - Main RAG execution endpoint
- `GET /health` - Health check

**Design decision:** Should RAG agent be stateless or cache recent retrievals?

##### 2. Implement Retrieval Logic
Build semantic search functionality.

**Steps:**
1. Convert query to embedding
2. Search vector database
3. Filter by relevance threshold
4. Optionally rerank using semantic similarity
5. Return top K documents

**Reranking:** Consider using cross-encoder for more accurate relevance scoring of retrieved documents.

##### 3. Implement Context Construction
Create logic to build LLM context from retrieved documents.

**Challenges:**
- Limited context window
- Relevance vs completeness tradeoff
- Citation formatting

**Strategy:**
- Prioritize most relevant chunks
- Truncate intelligently if exceeds token limit
- Include source metadata for citations

##### 4. Implement Response Generation
Generate final response using LLM with retrieved context.

**Prompt structure:**
```
System: You are a helpful assistant. Answer based on provided context.

Context: [retrieved documents]

User query: [original question]

Instructions: Cite sources using [source_id] notation.
```

**Important:** Instruct LLM to cite sources and stay grounded in provided context.

##### 5. Add Source Attribution
Include source information in responses.

**Format:**
- Inline citations in text
- List of sources at end
- Source metadata (title, URL, relevance score)

##### 6. Implement Hybrid Search (Optional Enhancement)
Combine semantic search with keyword search for better results.

**Strategy:**
- Perform both vector search and keyword search
- Combine results using weighted scoring
- Rerank combined results

##### 7. Add to Docker Compose
Add RAG agent service and Dapr sidecar.

##### 8. Register with Planning Agent
Update planning agent to know about RAG agent capabilities.

**RAG agent capabilities:**
- Knowledge base queries
- Semantic search
- Document retrieval
- Grounded Q&A

#### Testing Strategy

**What to test:**

1. **Retrieval accuracy:** Retrieves relevant documents for queries
2. **Response quality:** Generated responses use retrieved context
3. **Citation correctness:** Sources properly attributed
4. **Irrelevant queries:** Handles queries with no relevant documents
5. **Context limits:** Gracefully handles too many/too few results
6. **Performance:** End-to-end latency acceptable

**Success criteria:**
- Retrieves relevant documents (manually verify for test queries)
- Responses grounded in retrieved content
- Citations accurate and traceable
- Handles edge cases gracefully
- Latency under 3 seconds for typical queries

**Test scenarios:**
- Query about documented topics → retrieves correct docs
- Query about undocumented topic → acknowledges no relevant info
- Very broad query → uses most relevant subset of results
- Query requiring multiple documents → synthesizes from multiple sources

#### Retrieval Quality Metrics

**How to evaluate:**
- Precision@K: Relevant docs in top K results
- Recall: Percentage of relevant docs retrieved
- Mean Reciprocal Rank: Position of first relevant result
- User feedback: Thumbs up/down on results

**Consider:** Implement feedback loop to improve retrieval over time.

#### Error Handling

**Scenarios to handle:**
1. No relevant documents found
2. Vector database unavailable
3. Embedding generation fails
4. LLM generation fails
5. Context exceeds token limit

**Graceful degradation:** If vector search fails, fall back to reasoning agent without RAG.

#### Documentation Updates
- Document RAG agent capabilities
- Add examples of queries it handles well
- Document citation format
- Include retrieval tuning guide

#### Questions to Ask Before Proceeding
1. Should we implement reranking from the start?
2. What's the right tradeoff between precision and recall?
3. Do we want user feedback on retrieval quality?
4. Should we cache embeddings for common queries?

---

## Phase 4: Integration and Refinement

### Milestone 8: End-to-End Multi-Agent Workflows

#### Objective
Integrate all agents (planning, orchestrator, reasoning, RAG) into cohesive multi-agent workflows. Test complex scenarios end-to-end.

#### Why This Milestone
- Validates full system integration
- Tests realistic usage patterns
- Identifies integration issues
- Proves architecture works for intended use cases

#### Implementation Tasks

##### 1. Update Planning Agent with All Capabilities
Ensure planning agent knows about all available agents and their capabilities.

**Agent registry should include:**
- reasoning-agent: analysis, logical reasoning, tool use
- rag-agent: knowledge base queries, grounded Q&A
- search-agent: web search (if implemented)
- synthesis-agent: combining multiple sources (if implemented)

##### 2. Create Integration Tests
Build comprehensive end-to-end tests for complex workflows.

**Test scenarios to implement:**

**Simple query:** "What is 2+2?"
- Should use only reasoning agent
- Single-step workflow

**Knowledge query:** "What does our documentation say about X?"
- Should use RAG agent
- Single-step but with retrieval

**Complex query:** "Compare A and B based on latest information and our documentation"
- Should use reasoning → parallel (RAG + search) → synthesis
- Multi-step with parallelization

**Approval workflow:** "Make a significant decision about X"
- Should pause for human approval
- Tests pause/resume

##### 3. Implement Workflow Optimization
Add logic to optimize workflows based on observed performance.

**Optimizations:**
- Cache planning decisions for similar queries
- Learn which agent combinations work best
- Adjust parallelization based on observed latency

##### 4. Add Workflow Visualization
Create endpoint or tool that visualizes workflow execution.

**Shows:**
- DAG structure
- Current progress
- Step durations
- Success/failure status

##### 5. Implement Result Aggregation
Create synthesis logic that combines results from multiple agents.

**Challenges:**
- Results in different formats
- Conflicting information
- Citation merging

##### 6. Add Monitoring and Metrics
Implement comprehensive monitoring for multi-agent workflows.

**Metrics to track:**
- End-to-end latency
- Per-agent latency
- Success/failure rates
- Token usage and costs
- Parallelization efficiency

#### Testing Strategy

**What to test:**

1. **All workflow types:** Simple, complex, parallel, sequential
2. **Agent combinations:** Every pairing of agents
3. **Error propagation:** Failures handled at all levels
4. **Performance:** Latency within targets
5. **Cost tracking:** Token usage correctly attributed
6. **Observability:** Can trace requests through all agents

**Success criteria:**
- All workflow types complete successfully
- Parallel execution faster than sequential
- Errors handled gracefully without cascading failures
- Distributed tracing shows complete request path
- Token usage and costs tracked accurately

**Complex workflow scenarios:**
- 5+ step workflow with mix of parallel and sequential
- Workflow with multiple human approval points
- Workflow with agent failure and recovery
- Very long-running workflow (hours/days with pauses)

#### Performance Benchmarks

**Target metrics:**
- Simple query: < 2 seconds
- RAG query: < 3 seconds
- Complex multi-agent: < 10 seconds
- Human-in-the-loop overhead: < 100ms for pause/resume

#### Documentation Updates
- Add end-to-end workflow examples
- Document common workflow patterns
- Create troubleshooting guide for complex scenarios
- Add performance tuning guide

#### Questions to Ask Before Proceeding
1. Should we add workflow templates for common patterns?
2. Do we want automatic workflow optimization learning?
3. How should we handle conflicting results from different agents?

---

### Milestone 9: Human-in-the-Loop UI and Testing

#### Objective
Implement complete human-in-the-loop functionality with UI for approvals and testing framework for interactive workflows.

#### Why This Milestone
- Critical feature for your use case
- Validates pause/resume architecture
- Enables real-world usage
- Tests long-running workflow scenarios

#### Implementation Tasks

##### 1. Create Approval API Endpoints
Build REST API for human approval workflows.

**Endpoints:**
- `GET /approvals/pending` - List workflows awaiting approval
- `GET /approvals/{session_id}` - Get approval details
- `POST /approvals/{session_id}/approve` - Approve action
- `POST /approvals/{session_id}/reject` - Reject action
- `POST /approvals/{session_id}/modify` - Modify and approve

##### 2. Design Approval Context
Define what information to show for approval requests.

**Context should include:**
- What action is being requested
- Why it's needed (reasoning from agent)
- Potential impact or risks
- Relevant data/context
- Suggested decision

##### 3. Update Web Client
Modify existing FastHTML web client to support approval workflows.

**Features needed:**
- Show pending approvals
- Display approval context
- Approve/reject buttons
- Modification interface (if applicable)
- Real-time updates when new approvals needed

##### 4. Implement Notification System
Add notifications for approval requests.

**Options:**
- In-app notifications
- Email notifications
- Webhook to external systems
- Pub/sub events for external listeners

##### 5. Add Approval Timeout
Implement timeout for approval requests.

**Behavior:**
- After timeout, auto-reject or escalate
- Configurable per workflow
- Notification before timeout

##### 6. Testing Approval Workflows
Build test framework for human-in-the-loop scenarios.

**Test utilities needed:**
- Mock approval service for automated testing
- Approval simulation helpers
- Timeout testing utilities
- Concurrent approval handlers

**Key testing challenges:**
- Async nature of approvals (can happen hours later)
- Testing timeout behavior
- Testing rejection vs approval paths

##### 7. Implement Approval History
Store approval decisions for audit trail.

**Track:**
- Who approved/rejected
- When decision made
- Original context shown
- Any modifications made
- Reasoning for decision

**Storage:** Use Dapr state management for approval history with appropriate TTL.

#### Testing Strategy

**What to test:**

1. **Approval flow:** Complete workflow pauses, resumes after approval
2. **Rejection flow:** Workflow handles rejection appropriately
3. **Timeout:** Approval requests time out correctly
4. **Concurrent approvals:** Multiple workflows needing approval simultaneously
5. **State persistence:** Pending approvals survive service restarts
6. **UI interaction:** Web client correctly displays and handles approvals

**Success criteria:**
- Workflows pause at approval points
- Can approve/reject and workflow resumes
- Timeouts work as configured
- Approval context provides sufficient information for decision
- Audit trail captures all decisions
- UI responsive and intuitive

**Scenarios to test:**
- Workflow with single approval point
- Workflow with multiple sequential approvals
- Workflow with parallel branches needing separate approvals
- Approve after delay (minutes/hours)
- Reject and verify workflow terminates correctly
- Timeout triggers default behavior

#### User Experience Considerations

**Approval UX design:**
- Clear explanation of what's being approved
- Show relevant context without overwhelming
- Provide recommendations but allow override
- Confirmation before submitting
- Clear indication of what happens next

#### Documentation Updates
- Document approval workflow patterns
- Add guide for implementing approval points in agents
- Create user guide for approval interface
- Document approval timeout configuration

#### Questions to Ask Before Proceeding
1. Should approvals be revocable after submission?
2. What's appropriate default timeout for approvals?
3. Do we need multi-level approvals (different approval authorities)?
4. Should we support delegated approvals?

---

### Milestone 10: Observability and Production Readiness

#### Objective
Enhance observability, add comprehensive monitoring, implement production-ready error handling and logging.

#### Why This Milestone
- Critical for operating in production
- Debugging complex multi-agent workflows
- Performance optimization
- Incident response

#### Prerequisites
- All previous milestones complete
- **Read:** [OpenTelemetry Best Practices](https://opentelemetry.io/docs/concepts/signals/)
- **Read:** [Distributed Tracing Patterns](https://www.aspecto.io/blog/distributed-tracing-best-practices/)

#### Implementation Tasks

##### 1. Enhanced Distributed Tracing
Improve tracing to provide complete visibility.

**Trace improvements:**
- Add custom spans for important operations
- Include rich span attributes (agent parameters, results, etc.)
- Proper error recording with stack traces
- Context propagation across all service boundaries
- Baggage for session-level tracking

**Key spans to instrument:**
- Planning agent decisions
- Each step in workflow execution
- Agent invocations with parameters
- State operations (save/load)
- Approval wait times

##### 2. Structured Logging
Implement consistent structured logging across all services.

**Logging strategy:**
- Use JSON format for machine parsing
- Include correlation IDs in every log
- Standardize log levels across services
- Log important state transitions
- Contextual information in logs

**Key log points:**
- Service startup/shutdown
- Request start/end with duration
- Error conditions with full context
- State changes
- Configuration loading

##### 3. Metrics Collection
Add application metrics for monitoring.

**Metrics to collect:**
- Request rates per service
- Latency distributions (p50, p95, p99)
- Error rates
- Token usage and costs
- Queue depths (if applicable)
- Workflow step durations
- Agent success/failure rates
- Database query performance

**Instrumentation approach:** Use OpenTelemetry metrics or Prometheus client library.

##### 4. Health Checks
Implement comprehensive health checks.

**Health check types:**
- Liveness: Is service running?
- Readiness: Can service handle requests?
- Deep health: Are dependencies available?

**Dependencies to check:**
- Database connectivity
- Dapr sidecar
- Other required services
- External APIs (OpenAI)

##### 5. Error Handling and Retry Policies
Standardize error handling across all services.

**Error categories:**
- Transient errors (retry with backoff)
- Permanent errors (fail immediately)
- Timeout errors (configurable retry)
- Resource exhaustion (backpressure)

**Retry configuration:**
- Max retry attempts per error type
- Backoff strategy (exponential, linear)
- Circuit breaker thresholds
- Timeout values per operation type

##### 6. Alerting Rules
Define alerting rules for critical conditions.

**Alerts needed:**
- Service down
- High error rate
- High latency (p95 > threshold)
- Resource exhaustion
- State store unavailable
- Approval timeout rate high

**Alert routing:** Different severity levels with appropriate notification channels.

##### 7. Dashboards
Create monitoring dashboards.

**Dashboard views:**
- System overview (all services health)
- Per-service metrics
- Workflow execution view
- Cost tracking dashboard
- Error tracking dashboard

**Tools:** Use Phoenix UI for tracing, Grafana for metrics (if needed).

#### Testing Strategy

**What to test:**

1. **Tracing completeness:** Every request creates complete trace
2. **Log correlation:** Logs correlate with traces via IDs
3. **Metrics accuracy:** Metrics reflect actual behavior
4. **Health checks:** Correctly report service state
5. **Alerts:** Trigger at appropriate thresholds
6. **Error handling:** All error types handled correctly

**Success criteria:**
- Can trace any request through all services
- Logs provide sufficient debugging information
- Metrics available for all key operations
- Health checks accurate
- Alerts fire when they should (and not when they shouldn't)
- Error handling consistent and appropriate

**Observability validation:**
- Trigger various failure modes, verify observable
- Trace complex multi-agent workflow, verify all steps visible
- Check metrics during load, verify accuracy
- Test health check failure scenarios

#### Production Checklist

**Before deploying to production:**
- [ ] All services have health checks
- [ ] Distributed tracing covers all paths
- [ ] Structured logging implemented
- [ ] Metrics collection working
- [ ] Alerts configured and tested
- [ ] Dashboards created
- [ ] Error handling standardized
- [ ] Retry policies configured
- [ ] Circuit breakers in place
- [ ] Rate limiting implemented (if needed)

#### Documentation Updates
- Create operations runbook
- Document alert response procedures
- Add troubleshooting guide using traces/logs
- Document metrics and their meanings
- Create incident response guide

#### Questions to Ask Before Proceeding
1. What metrics retention policy should we use?
2. Which metrics are most critical for alerts?
3. Should we implement rate limiting at this stage?
4. Do we need log aggregation beyond Phoenix?

---

## Phase 5: Optimization and Advanced Features

### Milestone 11: Performance Optimization

#### Objective
Optimize system performance through caching, parallelization improvements, and resource management.

#### Why This Milestone
- Improve user experience (faster responses)
- Reduce costs (fewer API calls)
- Better resource utilization
- Handle higher load

#### Implementation Tasks

##### 1. Response Caching
Implement intelligent caching of agent responses.

**What to cache:**
- Planning decisions for similar queries
- RAG retrieval results for common queries
- Embedding vectors for repeated content
- Tool results with TTL

**Cache strategy:**
- Cache key design (query normalization)
- TTL policies per content type
- Cache invalidation rules
- Size limits

**Implementation options:**
- Dapr state management with TTL
- Redis with separate namespace
- In-memory with LRU eviction

##### 2. Batch Operations
Implement batching where beneficial.

**Batch opportunities:**
- Multiple embedding generations
- Parallel agent invocations
- State saves (accumulate then flush)
- Database queries

**Trade-off:** Batching adds latency but improves throughput.

##### 3. Connection Pooling
Optimize connection management.

**Connections to pool:**
- Database connections
- HTTP clients
- Dapr client connections

**Configuration:** Set appropriate pool sizes based on expected load.

##### 4. Query Optimization
Optimize expensive operations.

**Targets:**
- Vector search queries (index tuning)
- State lookups (index on session_id)
- Planning logic (cache frequent patterns)

##### 5. Resource Limits
Implement resource management and limits.

**Limits needed:**
- Concurrent workflow limit per service
- Token budget per request
- Maximum reasoning iterations
- Vector search result limits
- Memory usage caps

**Why limits:** Prevent resource exhaustion and runaway costs.

##### 6. Load Testing
Conduct load testing to identify bottlenecks.

**Test scenarios:**
- Sustained load (many requests over time)
- Spike load (sudden burst)
- Complex workflows under load
- Concurrent multi-agent workflows

**Measure:**
- Latency under load
- Error rate increase
- Resource utilization
- Bottleneck identification

#### Testing Strategy

**Performance tests:**

1. **Baseline performance:** Record current metrics
2. **Cache effectiveness:** Hit rate, latency improvement
3. **Batch performance:** Throughput improvement
4. **Load testing:** Behavior under various load patterns
5. **Resource limits:** Verify limits prevent exhaustion

**Success criteria:**
- Latency improved from baseline
- Cache hit rate > 20% for common queries
- System stable under 10x expected load
- Resource limits prevent exhaustion
- No performance regressions

**Benchmarks to establish:**
- Simple query: target < 1 second
- RAG query: target < 2 seconds
- Complex workflow: target < 5 seconds
- Throughput: X requests/second

#### Documentation Updates
- Document caching strategies
- Add performance tuning guide
- Include load testing results
- Document resource limits and configuration

#### Questions to Ask Before Proceeding
1. What's acceptable latency for different query types?
2. Should we implement adaptive rate limiting?
3. Do we need request prioritization?
4. What cache hit rate indicates success?

---

### Milestone 12: Advanced RAG Features

#### Objective
Enhance RAG agent with advanced features like reranking, hybrid search, and multi-hop reasoning.

#### Why This Milestone
- Improve retrieval quality
- Handle complex queries better
- Reduce irrelevant results
- Better citation accuracy

#### Prerequisites
- Milestone 7 complete (basic RAG working)
- **Read:** [Contextual Retrieval](https://www.anthropic.com/index/contextual-retrieval) - Anthropic's approach
- **Read:** [ColBERT Reranking](https://github.com/stanford-futuredata/ColBERT) - Advanced reranking

#### Implementation Tasks

##### 1. Implement Reranking
Add semantic reranking of retrieved documents.

**Why reranking:** Initial vector search may miss nuances; reranking with cross-encoder improves precision.

**Approach:**
1. Retrieve top 20-50 candidates with vector search
2. Rerank using cross-encoder model
3. Return top 5-10 after reranking

**Model options:**
- Use API-based reranker (Cohere, Jina)
- Self-host cross-encoder model
- Use LLM for relevance scoring

##### 2. Hybrid Search
Combine semantic and keyword search.

**Strategy:**
- Perform vector search and keyword search (full-text) in parallel
- Combine results using Reciprocal Rank Fusion (RRF) or weighted scoring
- Rerank combined results

**Benefits:** Catches both semantic matches and exact keyword matches.

##### 3. Query Expansion
Expand user queries for better retrieval.

**Techniques:**
- Use LLM to generate similar queries
- Synonym expansion
- Acronym expansion
- Multi-language if applicable

**Implementation:** Generate 2-3 query variations, search with each, merge results.

##### 4. Contextual Retrieval
Add context to document chunks before embedding.

**Anthropic's approach:**
- For each chunk, generate context explaining what it's about
- Prepend context to chunk before embedding
- Store both chunk and context

**Benefits:** Chunks self-contained, better retrieval accuracy.

##### 5. Multi-Hop Reasoning
Implement iterative retrieval for complex queries.

**Pattern:**
1. Initial retrieval for query
2. Analyze results, identify missing information
3. Generate follow-up retrieval queries
4. Combine information from multiple retrievals

**Use case:** Questions requiring information from multiple documents.

##### 6. Citation Improvements
Enhance citation quality and accuracy.

**Features:**
- Exact quote extraction
- Relevance scoring per citation
- Citation verification (ensure LLM used cited content)
- Rich citation metadata (page numbers, sections, etc.)

#### Testing Strategy

**What to test:**

1. **Reranking effectiveness:** Improved precision over baseline
2. **Hybrid search:** Better results than semantic alone
3. **Query expansion:** Retrieves documents missed by original query
4. **Contextual retrieval:** Improved accuracy vs standard chunking
5. **Multi-hop:** Correctly answers complex multi-step questions
6. **Citation accuracy:** Citations match actual content used

**Success criteria:**
- Reranking improves Precision@5 by X%
- Hybrid search finds more relevant results
- Complex queries answered correctly
- Citations verifiable and accurate

**Evaluation approach:**
- Create test set of queries with known relevant documents
- Measure precision, recall, MRR before and after improvements
- Manual evaluation of response quality

#### Documentation Updates
- Document advanced RAG features
- Add configuration guide for reranking
- Include query expansion examples
- Document when to use which features

#### Questions to Ask Before Proceeding
1. Is reranking worth the latency cost?
2. Should hybrid search be default or opt-in?
3. How many query variations for expansion?
4. Should contextual retrieval be applied to all documents?

---

### Milestone 13: Cost Management and Optimization

#### Objective
Implement cost tracking, budgeting, and optimization strategies for API usage and compute resources.

#### Why This Milestone
- Control costs as usage scales
- Visibility into spending patterns
- Budget enforcement
- Optimization opportunities

#### Implementation Tasks

##### 1. Token Usage Tracking
Track token usage across all LLM calls.

**Track:**
- Prompt tokens vs completion tokens
- Per-agent usage
- Per-session usage
- Per-user usage (if applicable)

**Storage:** Use Dapr state or database for usage metrics.

##### 2. Cost Attribution
Attribute costs to different components.

**Breakdown by:**
- Agent type (reasoning vs RAG vs planning)
- Query type (simple vs complex)
- User or session
- Time period

**Implementation:** Multiply token counts by pricing model, aggregate as needed.

##### 3. Budget Enforcement
Implement budget limits and enforcement.

**Budget types:**
- Per-session limit
- Per-user daily/monthly limit
- Overall service budget
- Per-agent type budget

**Enforcement actions:**
- Reject requests over budget
- Throttle expensive operations
- Switch to cheaper models
- Alert when approaching limit

##### 4. Model Selection Strategy
Intelligently select models based on query complexity and budget.

**Strategy:**
- Use cheaper models (GPT-4o mini) for simple queries
- Use expensive models (GPT-4o) for complex queries
- Fall back to cheaper model if over budget
- Allow override for critical queries

**Implementation:** Planning agent includes model selection in workflow plan.

##### 5. Caching Strategy for Cost
Aggressive caching to reduce API calls.

**Cache:**
- Identical queries (exact match)
- Similar queries (semantic match with high threshold)
- Planning decisions
- Embeddings

**Invalidation:** TTL-based with different durations per content type.

##### 6. Cost Dashboard
Create dashboard showing cost metrics.

**Metrics to display:**
- Current month spending
- Spending by agent type
- Most expensive queries
- Budget utilization
- Cost per request
- Projected monthly cost

#### Testing Strategy

**What to test:**

1. **Tracking accuracy:** Token counts match API usage
2. **Cost calculation:** Correct pricing applied
3. **Budget enforcement:** Limits actually enforced
4. **Model selection:** Appropriate model chosen for query type
5. **Cache effectiveness:** Reduced API calls vs baseline

**Success criteria:**
- Token tracking accurate within 1%
- Budgets enforced correctly
- Model selection appropriate
- Cost reduced through caching
- Dashboard shows real-time accurate data

**Cost scenarios to test:**
- Exceed session budget
- Approach monthly budget
- Mix of simple and complex queries
- Cache hit reduces cost

#### Documentation Updates
- Document cost tracking system
- Add budget configuration guide
- Include cost optimization strategies
- Create cost analysis documentation

#### Questions to Ask Before Proceeding
1. What should be default budget limits?
2. Should we support cost alerts?
3. How should we handle over-budget requests?
4. Do we need per-user cost isolation?

---

### Milestone 14: System Documentation and Deployment

**FINAL MILESTONE - STOP FOR REVIEW**

#### Objective
Complete comprehensive documentation and create production deployment guide.

#### Why This Final Milestone
- Knowledge transfer
- Onboarding new developers
- Production deployment readiness
- Maintenance and operations

#### Implementation Tasks

##### 1. Architecture Documentation
Create comprehensive architecture documentation.

**Documents needed:**
- System architecture diagram (all services and connections)
- Data flow diagrams
- State management design
- Security architecture
- Scalability design

**Format:** Markdown with diagrams (Mermaid or draw.io).

##### 2. API Documentation
Document all APIs comprehensively.

**For each service:**
- Endpoint descriptions
- Request/response schemas
- Authentication requirements
- Rate limits
- Example requests

**Tool:** Use OpenAPI/Swagger spec generation from FastAPI.

##### 3. Deployment Guide
Create step-by-step deployment guide.

**Includes:**
- Prerequisites
- Environment setup
- Configuration guide
- Initial data loading
- Verification steps
- Rollback procedures

**Environments to document:**
- Local development
- Staging
- Production
- Docker Compose
- Kubernetes (future)

##### 4. Operations Runbook
Create operations runbook for common tasks.

**Covers:**
- Service restart procedures
- Database backup/restore
- Log access and analysis
- Common troubleshooting scenarios
- Incident response procedures
- Scaling procedures

##### 5. Developer Guide
Create guide for developers working on the system.

**Includes:**
- Development environment setup
- Code organization
- Adding new agents
- Testing guidelines
- Contribution workflow
- Code review checklist

##### 6. Configuration Reference
Document all configuration options.

**For each service:**
- Environment variables
- Configuration file formats
- Default values
- Valid ranges
- Impact of changes

##### 7. Migration Guide
Document how to migrate data and configurations.

**Covers:**
- Database migrations
- Configuration updates
- Breaking changes
- Version upgrade procedures

#### Testing Strategy

**Documentation validation:**

1. **Completeness:** All features documented
2. **Accuracy:** Instructions work as written
3. **Clarity:** New developer can follow successfully
4. **Maintenance:** Documentation stays updated

**Validation approach:**
- Have someone unfamiliar follow deployment guide
- Test all documented procedures
- Verify all configuration examples work
- Check all links valid

**Success criteria:**
- New developer can set up environment using docs
- All deployment scenarios covered
- Operations team can maintain system
- API documentation complete and accurate

#### Final Production Readiness Checklist

- [ ] All services deployed and healthy
- [ ] Monitoring and alerting configured
- [ ] Backup procedures established
- [ ] Security review completed
- [ ] Performance benchmarks met
- [ ] Documentation complete
- [ ] Team trained on operations
- [ ] Incident response plan ready
- [ ] Scaling plan defined
- [ ] Cost monitoring active

#### Documentation Structure

```
docs/
├── architecture/
│   ├── overview.md
│   ├── services.md
│   ├── data-flow.md
│   └── security.md
├── api/
│   ├── openapi.yaml
│   └── examples.md
├── deployment/
│   ├── local.md
│   ├── docker-compose.md
│   └── production.md
├── operations/
│   ├── runbook.md
│   ├── monitoring.md
│   └── troubleshooting.md
├── development/
│   ├── setup.md
│   ├── contributing.md
│   └── testing.md
└── configuration/
    ├── reference.md
    └── examples/
```

#### Questions to Consider
1. What documentation format is most maintainable?
2. Should we auto-generate API docs from code?
3. Do we need video tutorials?
4. How do we keep documentation updated as code changes?

---

## Implementation Guidelines Summary

### General Principles

1. **Read documentation first:** Always review linked documentation before implementing
2. **Test comprehensively:** Include unit, integration, and end-to-end tests
3. **Ask questions:** Clarify ambiguous requirements before implementing
4. **No backwards compatibility:** Breaking changes encouraged for better design
5. **Meaningful tests:** Focus on behavior and edge cases, not coverage numbers
6. **Stop for review:** Complete each milestone fully before proceeding

### Testing Philosophy

**What to test:**
- Core functionality works correctly
- Edge cases handled appropriately
- Error conditions managed gracefully
- Performance within acceptable ranges
- Integration points work correctly

**What not to test:**
- Trivial getters/setters
- Framework functionality
- External APIs (mock them)

**Test quality over quantity:** Better to have 10 comprehensive tests than 100 shallow ones.

### Code Organization

**Service structure pattern:**
```
services/
└── {agent_name}/
    ├── main.py           # FastAPI app
    ├── {agent}.py        # Core logic
    ├── models.py         # Pydantic models
    └── tests/            # Service-specific tests
```

**Shared code location:**
```
api/
├── session_manager.py    # Shared state management
├── service_client.py     # Service invocation helpers
└── models.py            # Shared models
```

### Configuration Management

**Environment variables:** Use for secrets and environment-specific values
**Configuration files:** Use for complex structured configuration
**Defaults:** Provide sensible defaults for all optional settings

### Error Handling Pattern

**Principle:** Fail fast, provide context, enable recovery.

```python
try:
    result = await operation()
except SpecificError as e:
    logger.error("Operation failed", extra={"context": details})
    # Decide: retry, fail, or degrade
    raise AppropriateException(f"Failed because: {e}")
```

### Success Criteria for Completion

**Project is complete when:**
- All 14 milestones implemented and tested
- Documentation comprehensive and accurate
- System passes production readiness checklist
- Team can operate system independently
- Performance targets met
- Cost within budget
- All critical workflows tested end-to-end

---

## Appendix: Useful Resources

### Dapr Resources
- [Dapr Documentation](https://docs.dapr.io/)
- [Dapr Best Practices](https://docs.dapr.io/operations/best-practices/)
- [Dapr Python SDK](https://github.com/dapr/python-sdk)

### RAG Resources
- [Anthropic Contextual Retrieval](https://www.anthropic.com/index/contextual-retrieval)
- [LlamaIndex RAG Guide](https://docs.llamaindex.ai/)
- [Vector Database Comparison](https://github.com/superlinked/VectorHub)

### Observability
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
- [Phoenix Tracing](https://docs.arize.com/phoenix)

### Workflow Patterns
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Prefect Documentation](https://docs.prefect.io/)

### Testing
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
