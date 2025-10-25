# DAG-Based Orchestration Design

**Date**: 2025-10-25
**Purpose**: Design document for DAG-based multi-agent orchestration

---

## Overview

The orchestration system enables complex multi-step tasks by coordinating multiple specialized agents through a Directed Acyclic Graph (DAG) execution model. The system:

- Generates execution plans dynamically using LLM structured outputs
- Executes agent nodes asynchronously with automatic parallelization
- Streams results back to users in OpenAI-compatible format
- Supports multiple concurrent requests with isolated execution state

**Key Principles:**
- Agents define behavior (prompts, tools), not models
- Model selection happens at request time, not in agent registry
- Fully event-driven execution with zero CPU waste
- Concurrent request safety through isolated state

---

## Architecture

```
User Request (OpenAI protocol)
    ↓
Main API (api/main.py)
    ↓
Request Router (routing_mode=orchestration)
    ↓
Orchestrator Service
    │
    ├─ DAG Generator (LLM with structured output)
    │   └─ Generates execution plan as DAG
    │
    └─ DAG Executor (event-driven, async)
        ├─ Dispatches nodes reactively when dependencies complete
        ├─ Runs independent nodes in parallel
        └─ Streams results back (OpenAI SSE format)
```

---

## Core Components

### 1. Agent Registry (`api/agent_registry.py`)

Agent Cards define pre-configured agents with specific behaviors. The DAG generator selects agents from this registry based on the user's request.

```python
class AgentCard(BaseModel):
    """
    Pre-defined agent with capabilities.

    Defines WHAT the agent does and HOW it works.
    Model selection is NOT part of the agent definition.
    """
    # Public interface (visible to DAG generator)
    name: str                          # "web_researcher"
    description: str                   # What this agent does
    objective_template: str            # Expected input format

    # Internal implementation
    type: Literal["llm", "class", "service"]
    prompt: str | None = None          # System prompt (for llm type)
    tools: list[str] | None = None     # Tool names (for llm type)
    class_name: str | None = None      # Python class (for class type)
    service_url: str | None = None     # HTTP endpoint (for service type)


class ModelConfig(BaseModel):
    """
    Model and parameters for LLM execution.

    Separated from AgentCard to allow:
    - Same agent with different models (gpt-4o vs gpt-4o-mini)
    - Different parameters per request (temperature, etc.)
    - Runtime model selection without modifying registry
    """
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int | None = None
    top_p: float | None = None


# Agent Registry Example
AGENT_REGISTRY = {
    "web_researcher": AgentCard(
        name="web_researcher",
        description="Searches the web and summarizes findings on any topic",
        objective_template="Research {topic} and provide summary",
        type="llm",
        prompt="You are a web research specialist. Use search tools to find information and provide concise, factual summaries with citations.",
        tools=["web_search", "summarize"]
    ),

    "travel_planner": AgentCard(
        name="travel_planner",
        description="Plans travel itineraries including flights, hotels, and activities",
        objective_template="Plan {duration} trip to {destination} for {num_people} people",
        type="llm",
        prompt="You are a travel planning expert. Create detailed itineraries with flights, hotels, and activities based on user preferences.",
        tools=["flight_search", "hotel_search", "activity_search"]
    ),

    "deep_reasoner": AgentCard(
        name="deep_reasoner",
        description="Performs multi-step reasoning for complex questions",
        objective_template="Analyze and reason about: {question}",
        type="class",
        class_name="ReasoningAgent"
    ),

    "code_analyzer": AgentCard(
        name="code_analyzer",
        description="Reads and analyzes code files, identifying patterns and issues",
        objective_template="Analyze code in {file_path} for {purpose}",
        type="llm",
        prompt="You are a code analysis expert. Read code files, understand their structure, and provide insights on quality, patterns, and potential improvements.",
        tools=["read_file", "list_directory", "search_files"]
    )
}
```

**Design Rationale:**
- **Immutable registry**: Agents are read-only, preventing race conditions
- **No model in registry**: Models chosen per-request, enabling flexibility
- **Simple for LLM**: DAG generator only sees name, description, template

---

### 2. DAG Models (`api/dag_models.py`)

The DAG structure defines the execution plan with nodes and dependencies.

```python
class DAGNode(BaseModel):
    """
    Node in the execution graph.

    References a pre-defined agent from registry and specifies:
    - What this node should accomplish (objective)
    - Which nodes must complete first (dependencies)
    """
    id: str                       # Unique node identifier
    agent: str                    # Reference to AgentCard name in registry
    objective: str                # What this node should accomplish
    depends_on: list[str] = []    # List of node IDs this depends on


class DAG(BaseModel):
    """Directed Acyclic Graph of agent execution"""
    nodes: list[DAGNode]
    metadata: dict[str, Any] | None = None

    def validate_dag(self) -> None:
        """
        Validate DAG structure:
        - No cycles (topological sort)
        - All dependencies exist
        - All agent references are valid
        """
        # Implementation: cycle detection, dependency validation
```

**Example DAG:**
```python
DAG(nodes=[
    DAGNode(
        id="research_flights",
        agent="web_researcher",
        objective="Research round-trip flights to Paris from San Francisco in June",
        depends_on=[]
    ),
    DAGNode(
        id="research_hotels",
        agent="web_researcher",
        objective="Find hotels in Paris for 3-night stay in June under $200/night",
        depends_on=[]
    ),
    DAGNode(
        id="create_itinerary",
        agent="travel_planner",
        objective="Create comprehensive 3-day Paris itinerary with flights and hotels from previous research",
        depends_on=["research_flights", "research_hotels"]
    )
])
```

---

### 3. DAG Generator (`api/dag_generator.py`)

Uses LLM with structured outputs to generate execution plans.

```python
from litellm import acompletion
import json
import os


async def generate_dag(
    user_request: str,
    agent_registry: dict[str, AgentCard],
    litellm_api_key: str | None = None,
    litellm_base_url: str | None = None
) -> DAG:
    """
    Generate DAG using LLM with structured output.

    Args:
        user_request: User's natural language request
        agent_registry: Available agents for DAG nodes
        litellm_api_key: LiteLLM API key (defaults to env)
        litellm_base_url: LiteLLM base URL (defaults to env)

    Returns:
        DAG object with validated structure
    """
    # Build system prompt with available agents
    agent_descriptions = "\n".join([
        f"- {name}: {card.description}"
        for name, card in agent_registry.items()
    ])

    system_prompt = f"""
You are an execution planner. Generate a DAG (Directed Acyclic Graph) to accomplish the user's goal.

Available agents:
{agent_descriptions}

For each node:
1. Choose an appropriate agent from the list above
2. Write a clear objective describing what this node should accomplish
3. Specify dependencies (which nodes must complete before this one)

Independent nodes (no dependencies) will run in parallel for efficiency.
"""

    # Generate DAG using structured output
    response = await acompletion(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_request}
        ],
        response_format=DAG,  # Pydantic model for structured output
        api_key=litellm_api_key or os.getenv("LITELLM_PROXY_API_KEY"),
        api_base=litellm_base_url or os.getenv("LITELLM_PROXY_BASE_URL"),
    )

    # Parse and validate DAG
    dag = DAG(**json.loads(response.choices[0].message.content))
    dag.validate_dag()

    return dag
```

---

### 4. DAG Executor (`api/dag_executor.py`)

Executes DAG with fully event-driven architecture (no polling/spinning).

**Architecture:** Single event loop that reacts to worker completion events and dispatches new work reactively when dependencies are satisfied.

**Key Features:**
- Zero CPU waste: `await queue.get()` blocks on semaphore until events arrive
- Reactive dispatch: New nodes start immediately when dependencies complete
- Concurrent execution: Independent nodes run in parallel
- Isolated state: Each request has separate queue and state

```python
from asyncio import Queue, Event
from typing import AsyncIterator


class ExecutionEvent(BaseModel):
    """Events emitted during DAG execution"""
    type: Literal["node_started", "node_completed", "node_failed", "human_feedback_requested", "cancelled"]
    node_id: str
    data: dict[str, Any] | None = None


class DAGExecutor:
    def __init__(
        self,
        agent_registry: dict[str, AgentCard],
        tool_registry: dict[str, Callable],
        litellm_client: AsyncOpenAI
    ):
        self.agents = agent_registry
        self.tools = tool_registry
        self.litellm = litellm_client

    async def execute(
        self,
        dag: DAG,
        session_id: str,
        model_config: ModelConfig,
        cancellation_token: Event | None = None
    ) -> AsyncIterator[NodeResult]:
        """
        Execute DAG with fully event-driven architecture.

        Args:
            dag: DAG to execute
            session_id: Session identifier
            model_config: Model and parameters for LLM agents
            cancellation_token: Event to signal cancellation

        Yields:
            NodeResult as nodes complete
        """
        # Event queue for worker → executor communication
        event_queue: Queue[ExecutionEvent] = Queue()

        # Track execution state
        state = ExecutionState(
            dag=dag,
            model_config=model_config,
            completed={},
            running=set(),
            pending=set(node.id for node in dag.nodes),
            failed=set(),
            cancelled=False
        )

        # Start cancellation monitor if token provided
        if cancellation_token:
            asyncio.create_task(
                self._cancellation_monitor(cancellation_token, event_queue)
            )

        # Dispatch initial nodes (no dependencies)
        self._dispatch_ready_nodes(state, event_queue)

        # Main event loop - blocks on queue.get() (no spinning!)
        try:
            while not state.is_done():
                event = await event_queue.get()  # Suspends until event arrives

                if event.type == "node_completed":
                    result = event.data["result"]
                    state.mark_completed(event.node_id, result)

                    # Reactively dispatch newly ready nodes
                    self._dispatch_ready_nodes(state, event_queue)

                    yield result

                elif event.type == "node_failed":
                    state.mark_failed(event.node_id, event.data["error"])

                    # Other branches may still be executable
                    self._dispatch_ready_nodes(state, event_queue)

                elif event.type == "human_feedback_requested":
                    # Future: Pause execution, wait for human feedback
                    await self._handle_human_feedback(event, state)

                elif event.type == "cancelled":
                    state.cancelled = True
                    break

        finally:
            # Cleanup running workers
            await self._cleanup_running_workers(state)

    def _dispatch_ready_nodes(
        self,
        state: ExecutionState,
        event_queue: Queue
    ) -> None:
        """
        Dispatch all nodes whose dependencies are satisfied.

        Called reactively when:
        - Execution starts (initial nodes)
        - Node completes (dependent nodes may now be ready)
        - Node fails (independent branches may still be ready)
        """
        ready_nodes = [
            node for node in state.dag.nodes
            if self._is_ready(node, state)
        ]

        for node in ready_nodes:
            state.mark_running(node.id)
            # Fire-and-forget worker task
            asyncio.create_task(
                self._worker(node, state, event_queue)
            )

    async def _cancellation_monitor(
        self,
        cancellation_token: Event,
        event_queue: Queue
    ) -> None:
        """
        Monitor cancellation token and inject cancellation event.

        Uses Event.wait() which blocks on internal condition variable
        (no polling).
        """
        await cancellation_token.wait()  # Blocks until set
        await event_queue.put(ExecutionEvent(type="cancelled", node_id=""))

    def _is_ready(self, node: DAGNode, state: ExecutionState) -> bool:
        """Check if node is ready to execute."""
        if node.id in state.running or node.id in state.completed:
            return False

        # All dependencies must be completed
        for dep_id in node.depends_on:
            if dep_id not in state.completed:
                return False

        return True

    async def _worker(
        self,
        node: DAGNode,
        state: ExecutionState,
        event_queue: Queue
    ):
        """
        Worker executes a single node and signals completion.

        Runs independently, communicates with executor via event queue.
        """
        try:
            # Signal start
            await event_queue.put(
                ExecutionEvent(type="node_started", node_id=node.id)
            )

            # Execute node
            result = await self._execute_node(node, state)

            # Check if node requests human feedback (future feature)
            if self._requests_human_feedback(result):
                await event_queue.put(
                    ExecutionEvent(
                        type="human_feedback_requested",
                        node_id=node.id,
                        data={"result": result}
                    )
                )
                return

            # Signal completion
            await event_queue.put(
                ExecutionEvent(
                    type="node_completed",
                    node_id=node.id,
                    data={"result": result}
                )
            )

        except Exception as e:
            # Signal failure
            await event_queue.put(
                ExecutionEvent(
                    type="node_failed",
                    node_id=node.id,
                    data={"error": str(e)}
                )
            )

    async def _execute_node(
        self,
        node: DAGNode,
        state: ExecutionState
    ) -> Any:
        """Execute single node by looking up agent and calling appropriate handler."""
        agent_card = self.agents[node.agent]

        if agent_card.type == "llm":
            return await self._execute_llm_agent(agent_card, node, state)
        elif agent_card.type == "class":
            return await self._execute_class_agent(agent_card, node, state)
        elif agent_card.type == "service":
            return await self._execute_service_agent(agent_card, node, state)

    async def _execute_llm_agent(
        self,
        agent_card: AgentCard,
        node: DAGNode,
        state: ExecutionState
    ) -> str:
        """
        Execute LLM agent with pre-configured prompt and tools.

        Model and parameters come from state.model_config (passed at execution time),
        NOT from agent_card (which only defines behavior).
        """
        # Build messages with context from dependencies
        messages = [
            {"role": "system", "content": agent_card.prompt}
        ]

        # Add context from completed dependencies
        if node.depends_on:
            context_parts = []
            for dep_id in node.depends_on:
                if dep_id in state.completed:
                    context_parts.append(f"[{dep_id}]: {state.completed[dep_id]}")
            if context_parts:
                messages.append({
                    "role": "user",
                    "content": f"Context from previous steps:\n" + "\n".join(context_parts)
                })

        # Add node objective
        messages.append({
            "role": "user",
            "content": node.objective
        })

        # Execute with agent's tools and model from state.model_config
        response = await self.litellm.chat.completions.create(
            model=state.model_config.model,
            messages=messages,
            tools=[self.tools[t] for t in agent_card.tools] if agent_card.tools else None,
            temperature=state.model_config.temperature,
            max_tokens=state.model_config.max_tokens,
            top_p=state.model_config.top_p,
        )

        return response.choices[0].message.content


class ExecutionState:
    """Tracks DAG execution state"""
    def __init__(self, dag: DAG, model_config: ModelConfig, **kwargs):
        self.dag = dag
        self.model_config = model_config
        self.completed: dict[str, NodeResult] = kwargs.get("completed", {})
        self.running: set[str] = kwargs.get("running", set())
        self.pending: set[str] = kwargs.get("pending", set())
        self.failed: set[str] = kwargs.get("failed", set())
        self.cancelled: bool = kwargs.get("cancelled", False)

    def is_done(self) -> bool:
        """Check if execution is complete"""
        return (
            self.cancelled
            or len(self.completed) + len(self.failed) == len(self.dag.nodes)
        )

    def mark_running(self, node_id: str):
        self.pending.discard(node_id)
        self.running.add(node_id)

    def mark_completed(self, node_id: str, result: NodeResult):
        self.running.discard(node_id)
        self.completed[node_id] = result

    def mark_failed(self, node_id: str, error: str):
        self.running.discard(node_id)
        self.failed.add(node_id)
```

**Synchronization:**
- Uses `asyncio.Queue` with internal semaphores for event signaling
- `await queue.get()` suspends coroutine (no busy-waiting)
- `queue.put()` wakes up suspended coroutines

**Capabilities:**
- User cancellation via `Event.wait()` (no polling)
- Human-in-the-loop feedback (Phase 4-5, stubbed for now)
- Dynamic replanning (future enhancement)
- Progress streaming to user in real-time

---

### 5. Tool Registry (`api/tool_registry.py`)

Maps tool names to callable functions (MCP tools or custom implementations).

```python
# Tool registry maps tool name → function
TOOL_REGISTRY = {
    "read_file": mcp_client.call_tool("read_file", ...),
    "web_search": web_search_function,
    "summarize": summarize_function,
    # ...
}
```

**Tool Integration:**
- **MCP Tools**: Filesystem (read_file, write_file, list_directory, search_files)
- **Custom Tools**: Application-specific functions, API integrations, data transformations

---

## Concurrency: Multiple Simultaneous Requests

The design naturally supports multiple concurrent API requests executing DAGs simultaneously.

**Example:**
```python
# Request 1: "Plan Paris trip" with gpt-4o
async def handle_request_1():
    dag1 = await generate_dag("Plan Paris trip")
    model_config1 = ModelConfig(model="gpt-4o", temperature=0.7)
    async for result in executor.execute(dag1, session_id="abc", model_config=model_config1):
        yield result

# Request 2: "Analyze code quality" with gpt-4o-mini (concurrent)
async def handle_request_2():
    dag2 = await generate_dag("Analyze code quality")
    model_config2 = ModelConfig(model="gpt-4o-mini", temperature=0.0)
    async for result in executor.execute(dag2, session_id="def", model_config=model_config2):
        yield result

# Both run concurrently without interference
# Same agents can use different models per request
```

**Isolation Guarantees:**
- Each `execute()` call creates isolated resources:
  - Separate `event_queue` (no cross-talk)
  - Separate `state` (no shared mutable state)
  - Separate `model_config` (different models/parameters per request)
  - Separate worker tasks (scoped to execution)

- Shared resources are safe:
  - `agent_registry` (immutable, read-only, NO model info)
  - `tool_registry` (immutable, read-only)
  - `litellm_client` (designed for concurrent use)

**Scalability:**
- 10-100 concurrent requests: ✅ No problem
- 100-1000 requests: ⚠️ May hit LiteLLM rate limits or connection pool limits
- 1000+ requests: ❌ Need horizontal scaling (multiple API servers)

**Limiting Factors:**
1. LiteLLM/OpenAI API rate limits (external, most restrictive)
2. HTTP connection pool (configurable via `HTTP_MAX_CONNECTIONS`)
3. Memory (minimal: ~1-10 KB per request)
4. CPU (event-driven = efficient)

---

## Implementation Phases

### Phase 0: LiteLLM Migration (M1 - Prerequisite)

**Objective:** Replace all AsyncOpenAI usage with LiteLLM's `acompletion` for unified multi-provider support.

**Why:** LiteLLM provides standardized interface across model providers (OpenAI, Anthropic, etc.) with built-in features like cost tracking, retries, and fallbacks. AsyncOpenAI assumes OpenAI-compatible server only.

**Current State:**
- ✅ `api/request_router.py` - Routing classifier already uses `acompletion`
- ❌ `api/passthrough.py` - Uses `AsyncOpenAI.chat.completions.create()`
- ❌ `api/reasoning_agent.py` - Uses `AsyncOpenAI.chat.completions.create()`
- ❌ `api/dependencies.py` - ServiceContainer manages singleton `AsyncOpenAI` client

**Migration Tasks:**

#### 1. API Compatibility Verification
- [ ] Compare `acompletion()` parameter signature with `AsyncOpenAI.chat.completions.create()`
- [ ] Document parameter mapping (especially: model, messages, temperature, max_tokens, top_p)
- [ ] Verify response object structure matches (choices, usage, etc.)
- [ ] Test non-streaming response compatibility

#### 2. Streaming Response Verification
- [ ] Verify `acompletion(stream=True)` returns async iterator
- [ ] Confirm streaming chunk structure matches OpenAI chunks:
  - `delta` field with content/role
  - `finish_reason` field
  - `usage` field in final chunk (if present in OpenAI streaming)
- [ ] Test streaming with our `OpenAIStreamResponse` and SSE conversion
- [ ] Verify `[DONE]` marker behavior

#### 3. Trace Context Propagation
- [ ] Verify `acompletion()` accepts `extra_headers` parameter
- [ ] Test W3C TraceContext propagation pattern:
  ```python
  carrier: dict[str, str] = {}
  propagate.inject(carrier)
  await acompletion(..., extra_headers=carrier)
  ```
- [ ] Confirm traceparent headers reach LiteLLM proxy
- [ ] Validate OpenTelemetry span correlation works end-to-end

#### 4. Connection Pooling & Performance
- [ ] Research how `acompletion` manages HTTP connections internally
- [ ] Determine if connection pooling is automatic or needs configuration
- [ ] Test concurrent request performance (connection reuse)
- [ ] Verify timeout configuration (connect, read, write)
- [ ] Document any needed httpx client configuration for acompletion

#### 5. Configuration Management
- [ ] Decide: Environment variables vs. per-call parameters
  - Current env vars: `LITELLM_PROXY_API_KEY`, `LITELLM_PROXY_BASE_URL`
  - LiteLLM convention: `LITELLM_API_KEY`, `LITELLM_BASE_URL`?
- [ ] Update ServiceContainer to remove AsyncOpenAI client
- [ ] Create helper function or wrapper if needed for common acompletion config
- [ ] Ensure configuration works with LiteLLM proxy setup

#### 6. Error Handling Updates
- [ ] Document exception types raised by `acompletion`
- [ ] Compare with current `httpx.HTTPStatusError` handling
- [ ] Update error handling in passthrough.py and reasoning_agent.py
- [ ] Test error scenarios: rate limits, invalid auth, timeout, network errors
- [ ] Ensure OpenAI-compatible error responses maintained

#### 7. Structured Output Support (for DAG Generator)
- [ ] Verify `acompletion(response_format=PydanticModel)` works
- [ ] Test with both streaming and non-streaming modes
- [ ] Confirm JSON schema generation from Pydantic models
- [ ] Test DAG model as response_format (from dag_models.py)

#### 8. Code Migration
- [ ] Update `api/passthrough.py`:
  - Replace `openai_client.chat.completions.create()` with `acompletion()`
  - Update imports
  - Verify streaming response handling
  - Update trace propagation
- [ ] Update `api/reasoning_agent.py`:
  - Replace client calls in `_generate_reasoning_step()`
  - Replace client calls in `_synthesize_final_response()`
  - Update imports
  - Maintain streaming chunk format
- [ ] Update `api/dependencies.py`:
  - Remove `AsyncOpenAI` client from ServiceContainer
  - Remove httpx client if not needed elsewhere
  - Simplify startup/shutdown lifecycle
- [ ] Update type hints and function signatures

#### 9. Test Updates
- [ ] Update test mocks from `AsyncOpenAI` to `acompletion`
- [ ] Update fixtures for streaming responses
- [ ] Verify integration tests pass (require `OPENAI_API_KEY`)
- [ ] Add specific tests for acompletion features (cost tracking, etc.)
- [ ] Ensure evaluations still work

#### 10. Documentation & Cleanup
- [ ] Update CLAUDE.md with new LiteLLM patterns
- [ ] Document acompletion best practices
- [ ] Remove AsyncOpenAI references from docs
- [ ] Update architecture diagrams if needed

**Migration Strategy:** Big bang replacement in single PR
- All AsyncOpenAI → acompletion in one change
- Comprehensive testing before merge
- Rollback plan: revert single commit

**Success Criteria:**
- ✅ All tests passing (unit, integration, evaluations)
- ✅ Streaming responses work identically to before
- ✅ Trace propagation maintains distributed tracing
- ✅ No performance regression
- ✅ Ready for DAG generator implementation (structured outputs work)

---

### Phase 1: Core DAG System (M2)
- [ ] DAG models with validation (`dag_models.py`)
- [ ] DAG executor with event-driven architecture (`dag_executor.py`)
- [ ] Basic tool registry (`tool_registry.py`)
- [ ] Unit tests for executor and models

### Phase 2: DAG Generation (M3)
- [ ] LLM-based DAG generator (`dag_generator.py`)
- [ ] Structured output schemas with LiteLLM
- [ ] Integration with agent registry
- [ ] Tests with various request types

### Phase 3: Orchestrator Service (M4)
- [ ] FastAPI orchestration endpoint
- [ ] Integrate DAG generator + executor
- [ ] OpenAI-compatible streaming
- [ ] Main API routing integration

### Phase 4: Enhanced Features (M5)
- [ ] Human-in-the-loop (HITL) feedback support
  - Extended SSE format for feedback requests
  - Feedback types: APPROVAL, SELECTION, GUIDANCE, CLARIFICATION, INFORMATION
  - Redis-backed session state for pause/resume
  - Feedback endpoint: `POST /v1/orchestration/{session_id}/feedback`
- [ ] Session management (Redis)
- [ ] MCP tool integration
- [ ] Error handling and retries

**Note:** HITL feedback was discussed and designed but deferred to Phase 4. Phase 1 will stub out `_requests_human_feedback()` to return `False` and `_handle_human_feedback()` as placeholder.

### Phase 5: Advanced Agents (M6)
- [ ] Migrate ReasoningAgent to class agent
- [ ] Add predefined specialist agents
- [ ] Prompt optimization
- [ ] Performance tuning

---

## Open Design Questions

Questions ordered by priority for Phase 1 implementation (M2-M3).

### Critical for Phase 1 (Must Decide Before Implementation)

#### Q1: Streaming Format & OpenAI Compatibility

**Question:** How should we stream DAG execution progress while maintaining OpenAI compatibility?

**Context:**
- Current reasoning agent uses `OpenAIStreamResponse` with custom `reasoning_event` delta field
- Reasoning events: ITERATION_START, PLANNING, TOOL_EXECUTION_START, TOOL_RESULT, etc.
- All responses use SSE format (`create_sse()` from `openai_protocol.py`)
- OpenAI SDK clients expect standard chunk format

**Options:**

A. **Reasoning Event Pattern** (extends current approach):
```python
# Node execution as reasoning events
OpenAIStreamResponse(
    delta={
        "reasoning_event": {
            "type": "NODE_STARTED",
            "node_id": "research_flights",
            "agent": "web_researcher",
            "metadata": {...}
        }
    }
)
# Then actual content chunks
OpenAIStreamResponse(delta={"content": "Found 3 flights..."})
```
- ✅ Consistent with current reasoning agent pattern
- ✅ Progress visibility without breaking OpenAI format
- ✅ Clients can filter/display events as needed
- ❌ Requires clients to understand custom reasoning_event field

B. **Content-Only with Hidden Orchestration**:
```python
# Only stream final content from nodes
OpenAIStreamResponse(delta={"content": "Found 3 flights..."})
OpenAIStreamResponse(delta={"content": "\n\nFound 5 hotels..."})
```
- ✅ Perfect OpenAI compatibility
- ✅ Clean user experience
- ❌ No visibility into DAG execution progress
- ❌ Can't distinguish between node boundaries

C. **Content Markers** (simple but pollutes output):
```python
OpenAIStreamResponse(delta={"content": "🔍 [research_flights] Starting..."})
OpenAIStreamResponse(delta={"content": "Found 3 flights..."})
OpenAIStreamResponse(delta={"content": "✅ [research_flights] Complete"})
```
- ✅ OpenAI compatible
- ✅ Some progress visibility
- ❌ Markers pollute actual content
- ❌ Hard to parse programmatically

**Recommendation:** Option A (Reasoning Event Pattern)
- Natural extension of current architecture
- Best balance of compatibility and observability
- Clients already handle custom reasoning events in current system
- Easy to add new event types (NODE_STARTED, NODE_COMPLETED, NODE_FAILED)

**Decision needed:** Approve Option A or discuss alternatives?

---

#### Q2: Node Result Format & Context Passing

**Question:** How should node results be formatted and passed to dependent nodes?

**Context:**
- Current reasoning agent yields `OpenAIStreamResponse` objects
- DAG executor needs to collect results and pass to dependent nodes
- Different node types (llm, class, service) may return different formats
- Design doc shows string concatenation: `[node_id]: {result_content}`

**Issues to resolve:**

1. **Result Collection:**
   - Streaming chunks need to be assembled into complete result
   - Need `NodeResult` model to wrap output
   - Should include metadata (tokens, duration, tool calls)?

2. **Context Injection:**
   - Current design: Append all dependency results as user message
   - Alternative: Structured context with node boundaries
   - How much context per dependency? (summary vs full output)

3. **Memory Tracking:**
   - Should we track tool calls, files accessed, sources cited?
   - Data lineage for explainability?
   - Phase 1: Simple or defer to later phase?

**Proposed approach for Phase 1:**
```python
class NodeResult(BaseModel):
    node_id: str
    content: str  # Assembled from streaming chunks
    metadata: dict[str, Any] = {}  # Optional: tokens, duration, etc.

# Context injection (simple string concatenation)
context_message = {
    "role": "user",
    "content": "Context from previous steps:\n" +
               "\n\n".join([f"[{dep_id}]: {state.completed[dep_id].content}"
                           for dep_id in node.depends_on])
}
```

**Decision needed:** Approve simple approach or design richer context format?

---

#### Q3: Tool Registry Architecture

**Question:** How should tools be shared between agents in the DAG?

**Context:**
- Current: `tool_registry` dict maps tool name → callable function
- Agents reference tools by name in `AgentCard.tools: list[str]`
- MCP tools + custom tools both in registry

**Options:**

A. **Global Tool Registry** (design doc approach):
```python
# Single registry, all agents share same tools
TOOL_REGISTRY = {
    "read_file": mcp_read_file,
    "web_search": web_search_func,
    ...
}
agent_card.tools = ["read_file", "web_search"]  # Reference by name
```
- ✅ Simple, matches current architecture
- ✅ Tools initialized once at startup
- ❌ All agents have access to all tools (security?)
- ❌ Hard to have agent-specific tool configurations

B. **Agent-Specific Tool Sets**:
```python
# Each agent defines its own tools
class AgentCard(BaseModel):
    tools: dict[str, Tool]  # Tool instances, not just names
```
- ✅ Agent isolation and security
- ✅ Different tool configurations per agent
- ❌ Tool duplication
- ❌ More complex initialization

C. **Hybrid Approach**:
```python
# Global registry + agent subsets
TOOL_REGISTRY = {...}  # All available tools
agent_card.allowed_tools = ["read_file", "web_search"]  # Subset
# Executor filters tools based on allowed_tools
```
- ✅ Tool reuse with access control
- ✅ Security and isolation
- ❌ More complex executor logic

**Recommendation:** Option A (Global Registry) for Phase 1
- Simpler implementation
- Matches current architecture
- Can add access control in later phase if needed

**Decision needed:** Approve global registry or implement access control now?

---

### Important for Phase 2-3 (Can Defer Initially)

#### Q4: Agent Registry Storage & Extensibility

**Question:** How should agent definitions be stored and updated?

**Current design:** Python dict in code (`AGENT_REGISTRY`)

**Trade-offs:**

**Static Registry (Code-based):**
- ✅ Simple, no database dependency
- ✅ Version controlled with code
- ✅ Immutable (safe for concurrent requests)
- ❌ Requires deployment to add agents
- ❌ Can't customize per-user/tenant

**Database Storage:**
- ✅ Runtime updates without deployment
- ✅ Per-tenant customization possible
- ❌ Complexity (migrations, cache invalidation)
- ❌ Race conditions if mutable
- ❌ Not needed for MVP

**Hybrid Approach:**
- Core agents in code (system-level)
- Optional custom agents in database (user-level)
- Registry merge at runtime

**Recommendation for Phase 1:** Static registry in code
- Sufficient for initial launch
- Can migrate to hybrid in Phase 5 if needed
- Keep registry immutable for safety

**Decision needed:** Confirm static registry for Phase 1?

---

#### Q5: Error Handling & Partial Failures

**Question:** What happens when a node fails in the DAG?

**Current design:**
- `ExecutionEvent` has `node_failed` type
- State tracks failed nodes
- Comment: "Other branches may still be executable"

**Need to decide:**

1. **Failure Propagation:**
   - Should dependent nodes be cancelled when dependency fails?
   - Or mark as failed but continue independent branches?

2. **Retry Strategy:**
   - Automatic retries for transient failures?
   - Exponential backoff?
   - Retry budget per node?

3. **User Communication:**
   - Stream error events to user?
   - Partial results when some nodes fail?
   - Final response format with errors?

4. **Span Recording:**
   - How to represent failed nodes in OpenTelemetry spans?
   - Error attributes and status codes?

**Proposed for Phase 1:**
```python
# Simple approach: fail dependent nodes, continue independent
if event.type == "node_failed":
    state.mark_failed(event.node_id, event.data["error"])
    # Mark all dependents as failed
    for node in state.dag.nodes:
        if event.node_id in node.depends_on:
            state.mark_failed(node.id, f"Dependency {event.node_id} failed")
    # Continue with independent branches
    self._dispatch_ready_nodes(state, event_queue)
```

**Decision needed:** Approve fail-fast for dependents, or explore retry logic?

---

#### Q6: Model Configuration & Request Routing

**Question:** How does `ModelConfig` flow through the system?

**Current design:**
- `ModelConfig` passed to executor at request time
- All LLM agents use same model for single request
- Separates agent behavior from model selection

**Need to clarify:**

1. **Where does ModelConfig come from?**
   - Extracted from `OpenAIRequest.model` field?
   - Default values for temperature, max_tokens?
   - Override via request headers?

2. **Per-Node Model Selection?**
   - Some nodes might need different models (fast vs. powerful)
   - Should `DAGNode` have optional `model_config` override?
   - Or always use request-level config?

3. **Integration with Request Router:**
   - How does orchestration path create `ModelConfig`?
   - Match pattern from passthrough/reasoning paths?

**Proposed for Phase 1:**
```python
# In orchestration handler (api/main.py)
async def handle_orchestration_request(request: OpenAIRequest):
    # Extract model config from request
    model_config = ModelConfig(
        model=request.model,
        temperature=request.temperature or 0.0,
        max_tokens=request.max_tokens,
        # ... other params
    )

    # Generate DAG
    dag = await generate_dag(request.messages[-1].content, agent_registry)

    # Execute with request-level model config (all nodes use same model)
    async for result in executor.execute(dag, session_id, model_config):
        yield result
```

**Decision needed:** Confirm request-level model config for Phase 1?

---

### Advanced Features (Phase 4-5)

#### Q7: Dynamic Agent Support

**Question:** Should DAG generator create agents on-the-fly vs. registry-only?

**Registry-Only (Phase 1):**
- DAG generator selects from `AGENT_REGISTRY`
- Pre-defined, tested agents only
- Quality and safety guaranteed

**Dynamic Agents (Future):**
- DAG generator creates custom prompts for specific sub-tasks
- More flexible but less predictable
- Quality/safety concerns

**Recommendation:** Registry-only for Phase 1-3, revisit in Phase 4

**Decision needed:** Confirm registry-only approach?

---

#### Q8: DAG Re-planning & Adaptation

**Question:** Should the system regenerate the DAG if execution fails or produces unexpected results?

**Static Execution (Phase 1):**
- Execute DAG as initially generated
- Simpler, more predictable
- No infinite loops

**Dynamic Re-planning (Future):**
- Detect failures or poor results
- Generate new DAG to recover
- Risk of infinite re-planning

**Proposed trigger conditions:**
- All paths blocked by failures
- User provides feedback/correction
- Explicit re-plan request

**Recommendation:** Static execution for Phase 1-3, add re-planning in Phase 4

**Decision needed:** Confirm static execution for Phase 1?

---

#### Q9: Human-in-the-Loop (HITL) Feedback

**Question:** How should the system pause for human input?

**Current design:**
- `ExecutionEvent` has `human_feedback_requested` type
- Stubbed for Phase 1 (`_requests_human_feedback()` returns `False`)
- Planned for Phase 4

**Need to design (later):**
1. Feedback types (APPROVAL, SELECTION, GUIDANCE, etc.)
2. Session persistence (Redis for pause/resume)
3. Feedback endpoint: `POST /v1/orchestration/{session_id}/feedback`
4. SSE format for feedback requests
5. Timeout handling if user doesn't respond

**Recommendation:** Stub for Phase 1, design in Phase 4

**Decision needed:** Confirm HITL is Phase 4 scope?

---

#### Q10: Session Management & State Persistence

**Question:** Should DAG execution state be persisted?

**In-Memory (Phase 1):**
- `ExecutionState` exists only during request
- No persistence
- Client disconnection → lost state

**Persistent State (Phase 4):**
- Redis-backed state storage
- Pause/resume capability
- HITL feedback support
- Session recovery

**Recommendation:** In-memory for Phase 1, Redis in Phase 4 for HITL

**Decision needed:** Confirm in-memory state for Phase 1?

---

## Summary

**Core Architecture:**
- Event-driven DAG execution with zero CPU waste
- Agent behavior separate from model selection
- Concurrent request safety through isolated state
- LiteLLM-based DAG generation with structured outputs

**Key Benefits:**
- Maximum parallelism with reactive dispatch
- Flexible model selection per request
- Clean separation: agent behavior vs. execution configuration
- Production-ready concurrency support

**Next Steps:**
1. Implement Phase 1 (Core DAG System)
2. Validate with unit tests
3. Proceed to Phase 2 (DAG Generation)
