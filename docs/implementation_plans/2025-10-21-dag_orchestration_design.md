# DAG-Based Orchestration Design

**Status**: Draft for Review
**Date**: 2025-10-20
**Purpose**: Design document for DAG-based orchestration without A2A protocol

---

## Overview

Replace the A2A-based orchestration approach with a simpler, more flexible DAG-based system where:
- Most "agents" are just LLM calls with different prompts and tools
- Nodes execute asynchronously/concurrently when dependencies allow
- DAG is generated dynamically via LLM structured outputs
- No external protocol needed initially (A2A can be added later for external agents)

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
    └─ DAG Executor (async/concurrent)
        ├─ Executes nodes in topological order
        ├─ Runs independent nodes in parallel
        └─ Streams results back (OpenAI SSE format)
```

---

## Core Components

### 1. Agent Registry (`api/agent_registry.py`)

**Agent Cards** - Pre-defined agents with capabilities (inspired by A2A Agent Card concept)

```python
class AgentCard(BaseModel):
    """
    Pre-defined agent with capabilities.

    Agent Cards define WHAT an agent does (exposed to DAG generator)
    and HOW it works (internal implementation details).

    IMPORTANT: Model and model parameters are NOT part of the agent definition.
    They are passed at execution time via ModelConfig.
    """
    # Public interface (visible to DAG generator)
    name: str                          # "web_researcher"
    description: str                   # What this agent does
    objective_template: str            # Expected input format

    # Internal implementation (NOT exposed to DAG generator LLM)
    type: Literal["llm", "class", "service"]
    prompt: str | None = None          # System prompt (for llm type)
    tools: list[str] | None = None     # Tool names (for llm type)
    class_name: str | None = None      # Python class (for class type)
    service_url: str | None = None     # HTTP endpoint (for service type)


class ModelConfig(BaseModel):
    """
    Model and parameters for LLM execution.

    Separated from AgentCard because:
    - Same agent can be invoked with different models (gpt-4o vs gpt-4o-mini)
    - Different executions may need different parameters (temperature, etc.)
    - Allows runtime model selection without modifying agent registry
    """
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int | None = None
    top_p: float | None = None
    # ... other model parameters


# Example Agent Registry (NO MODEL SPECIFIED)
AGENT_REGISTRY = {
    "web_researcher": AgentCard(
        name="web_researcher",
        description="Searches the web and summarizes findings on any topic",
        objective_template="Research {topic} and provide summary",
        type="llm",
        prompt="You are a web research specialist. Use search tools to find information and provide concise, factual summaries with citations.",
        tools=["web_search", "summarize"]
        # NOTE: No model specified - passed at execution time
    ),

    "travel_planner": AgentCard(
        name="travel_planner",
        description="Plans travel itineraries including flights, hotels, and activities",
        objective_template="Plan {duration} trip to {destination} for {num_people} people",
        type="llm",
        prompt="You are a travel planning expert. Create detailed itineraries with flights, hotels, and activities based on user preferences.",
        tools=["flight_search", "hotel_search", "activity_search"]
        # NOTE: No model specified - passed at execution time
    ),

    "deep_reasoner": AgentCard(
        name="deep_reasoner",
        description="Performs multi-step reasoning for complex questions requiring deep analysis",
        objective_template="Analyze and reason about: {question}",
        type="class",
        class_name="ReasoningAgent"
    ),

    "code_analyzer": AgentCard(
        name="code_analyzer",
        description="Reads and analyzes code files, identifying patterns, issues, and improvements",
        objective_template="Analyze code in {file_path} for {purpose}",
        type="llm",
        prompt="You are a code analysis expert. Read code files, understand their structure, and provide insights on quality, patterns, and potential improvements.",
        tools=["read_file", "list_directory", "search_files"]
        # NOTE: No model specified - passed at execution time
    )
}
```

**Key Design Decision:**
- **LLM generating DAG only sees**: `name`, `description`, `objective_template`
- **LLM does NOT choose**: prompts, tools, models (these are pre-configured in registry)
- **Models and parameters**: Specified at execution time via `ModelConfig`, not in registry
- **Benefits**:
  - Same agent can use different models (gpt-4o vs gpt-4o-mini)
  - No shared mutable state (model config passed per request)
  - Agent behavior is decoupled from model selection
- **Result**: Simpler DAG generation = higher quality outputs

---

### 2. DAG Models (`api/dag_models.py`)

**Simplified Node Structure** - References agents from registry

```python
class DAGNode(BaseModel):
    """
    Simple node that references a pre-defined agent from registry.

    LLM generating DAG only needs to:
    1. Choose which agent to use (from registry)
    2. Specify the objective (what this node should accomplish)
    3. Declare dependencies (which nodes must complete first)
    """
    id: str                       # Unique node identifier
    agent: str                    # Reference to AgentCard name in registry
    objective: str                # What this node should accomplish (natural language)
    depends_on: list[str] = []    # List of node IDs this depends on

class DAG(BaseModel):
    """Directed Acyclic Graph of agent execution"""
    nodes: list[DAGNode]
    metadata: dict[str, Any] | None = None

    def validate_dag(self) -> None:
        """
        Validate DAG structure:
        - Check for cycles (must be acyclic)
        - Verify all dependencies exist
        - Ensure all agent references are valid
        """
        # Cycle detection via topological sort
        # Dependency validation
        # Agent registry validation
```

**Example DAG (generated by LLM):**
```python
{
    "nodes": [
        {
            "id": "research_paris",
            "agent": "web_researcher",
            "objective": "Research Paris hotels for 3-night stay in June with budget under $200/night",
            "depends_on": []
        },
        {
            "id": "research_flights",
            "agent": "web_researcher",
            "objective": "Research round-trip flights to Paris from San Francisco in June",
            "depends_on": []
        },
        {
            "id": "create_itinerary",
            "agent": "travel_planner",
            "objective": "Create comprehensive 3-day Paris itinerary incorporating hotels and flights from previous research",
            "depends_on": ["research_paris", "research_flights"]
        }
    ]
}
```

**Benefits:**
- ✅ **Simpler for LLM**: Choose agent + write objective (not prompts/tools/models)
- ✅ **Higher quality**: LLM makes fewer complex decisions
- ✅ **Consistent**: Agents use tested, optimized prompts
- ✅ **Maintainable**: Update agent prompts without regenerating DAGs
- ✅ **Testable**: Each agent can be tested independently

### 3. DAG Generator (`api/dag_generator.py`)

Uses LLM (via LiteLLM) with structured outputs to create execution plan.

**Input:**
- User request
- Agent registry (list of available agents with descriptions)
- Context from previous interactions

**Output:**
- DAG with nodes referencing agents from registry

**Implementation Example:**
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
        litellm_api_key: LiteLLM API key (defaults to env LITELLM_PROXY_API_KEY)
        litellm_base_url: LiteLLM base URL (defaults to env LITELLM_PROXY_BASE_URL)

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
    dag.validate_dag()  # Check for cycles, validate dependencies

    return dag
```

**Usage Example:**
```python
# Generate DAG from user request
dag = await generate_dag(
    user_request="Research a trip to Paris, including flights, hotels, and itinerary",
    agent_registry=AGENT_REGISTRY
)

# Example generated DAG:
# DAG(nodes=[
#     DAGNode(id="research_flights", agent="web_researcher",
#             objective="Research round-trip flights to Paris", depends_on=[]),
#     DAGNode(id="research_hotels", agent="web_researcher",
#             objective="Find hotels in Paris for 3 nights", depends_on=[]),
#     DAGNode(id="create_itinerary", agent="travel_planner",
#             objective="Create 3-day Paris itinerary with flights and hotels",
#             depends_on=["research_flights", "research_hotels"])
# ])
```

---

### 4. DAG Executor (`api/dag_executor.py`)

Executes DAG with async/concurrent node execution using event-driven architecture.

#### **Execution Architecture: Fully Event-Driven (No Polling/Spinning)**

**Why Event-Driven vs. Sequential Waves:**

**Sequential Wave Approach (Simpler, Less Flexible):**
```python
# Execute nodes in waves, wait for each wave to complete
for wave in waves:
    await asyncio.gather(*[execute(node) for node in wave])
    # Blocked until ALL nodes in wave complete
    # Cannot handle dynamic events during execution
```

**Event-Driven Approach (More Flexible, Recommended):**
```python
# Single event loop reacts to worker events
# Dispatches new work REACTIVELY when dependencies complete
# No polling or spinning - uses asyncio.Queue semaphore for signaling
# Workers signal completion → executor updates state → dispatches ready nodes
```

**Benefits of Event-Driven:**
1. **Zero CPU Waste**: No polling loops - `await queue.get()` blocks on semaphore until events arrive
2. **Reactive Dispatch**: New nodes dispatched immediately when dependencies complete (not on timer)
3. **Dynamic Control**: Can cancel, pause, or replan while nodes are running
4. **Better HITL**: Nodes can signal "need approval", executor pauses that branch
5. **Streaming**: Workers emit progress events, executor streams to user in real-time

**Synchronization Mechanism:**
- Uses `asyncio.Queue` which internally uses semaphores for event signaling
- `await queue.get()` suspends the coroutine (no busy-waiting)
- `queue.put()` wakes up suspended coroutines via semaphore signaling

---

#### **Proposed Architecture:**

```python
from asyncio import Queue, Event
from typing import AsyncIterator

class ExecutionEvent(BaseModel):
    """Events emitted during DAG execution"""
    type: Literal["node_started", "node_completed", "node_failed", "approval_required", "cancelled"]
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
        Execute DAG with fully event-driven architecture (no polling/spinning).

        Args:
            dag: DAG to execute
            session_id: Session identifier for state management
            model_config: Model and parameters for LLM agent execution
            cancellation_token: Event to signal cancellation

        Yields:
            NodeResult as nodes complete (for streaming to user)
        """
        # Event queue for worker → executor communication
        event_queue: Queue[ExecutionEvent] = Queue()

        # Track execution state
        state = ExecutionState(
            dag=dag,
            model_config=model_config,  # Pass model config to state
            completed={},  # node_id → NodeResult
            running=set(),  # node_ids currently executing
            pending=set(node.id for node in dag.nodes),  # not yet started
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
                event = await event_queue.get()  # ← Suspends until event arrives

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

                elif event.type == "approval_required":
                    # Pause execution, wait for user approval
                    await self._handle_approval(event, state)

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

        Avoids polling by using Event.wait() which blocks on internal condition variable.
        """
        await cancellation_token.wait()  # ← Blocks until set (no polling!)
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

        Runs independently, communicates with boss via event queue.
        """
        try:
            # Signal start
            await event_queue.put(
                ExecutionEvent(type="node_started", node_id=node.id)
            )

            # Execute node (pass entire state for model config access)
            result = await self._execute_node(node, state)

            # Check if node requires approval
            if self._requires_approval(result):
                await event_queue.put(
                    ExecutionEvent(
                        type="approval_required",
                        node_id=node.id,
                        data={"result": result}
                    )
                )
                # Worker pauses here, boss handles approval
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

class ExecutionState:
    """Tracks DAG execution state"""
    def __init__(self, dag: DAG, model_config: ModelConfig, **kwargs):
        self.dag = dag
        self.model_config = model_config  # Store model config for agent execution
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

    async def _execute_node(
        self,
        node: DAGNode,
        state: ExecutionState
    ) -> Any:
        """Execute single node by looking up agent and calling appropriate handler."""
        # Look up agent card from registry
        agent_card = self.agents[node.agent]

        # Execute based on agent type
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
        NOT from agent_card (which only defines behavior, not model).
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

        # Execute with agent's tools and MODEL FROM state.model_config
        response = await self.litellm.chat.completions.create(
            model=state.model_config.model,  # ← From execution-time config, not agent card
            messages=messages,
            tools=[self.tools[t] for t in agent_card.tools] if agent_card.tools else None,
            temperature=state.model_config.temperature,
            max_tokens=state.model_config.max_tokens,
            top_p=state.model_config.top_p,
        )

        return response.choices[0].message.content
```

---

#### **Key Capabilities Enabled by Event-Driven Architecture:**

**1. User Cancellation:**
```python
# User clicks "stop" in UI
cancellation_token.set()

# Boss detects cancellation in next loop iteration
# → Signals all running workers to stop
# → Cleans up resources
# → Returns partial results
```

**2. Human-in-the-Loop (HITL):**
```python
# Node signals: "Need approval to book $1500 hotel"
# → Worker emits "approval_required" event
# → Boss pauses execution (doesn't dispatch dependent nodes)
# → Streams approval request to user
# → User approves/rejects via API call
# → Boss resumes or cancels based on decision
```

**3. Dynamic Replanning (Future):**
```python
# Node fails: "All hotels booked for those dates"
# → Worker emits "node_failed" event with context
# → Boss detects failure pattern
# → Boss triggers DAG regeneration with new context
# → Boss replaces remaining pending nodes with new DAG
# → Execution continues with new plan
```

**4. Progress Streaming:**
```python
# Workers emit events as they work:
# → "node_started" → Boss streams to user: "[web_researcher] Starting..."
# → LLM streaming chunks → Boss forwards to user in real-time
# → "node_completed" → Boss streams to user: "[web_researcher] Complete"
```

**5. Parallel Execution with Dynamic Dispatch:**
```python
# Unlike wave-based (wait for all nodes in wave):
for wave in waves:
    await gather(*wave)  # Blocked until slowest node finishes

# Event-driven (dispatch as soon as dependencies met):
# → Node A, B, C have no dependencies → dispatch immediately at startup
# → Node D depends on A → dispatch reactively when A completes (don't wait for B, C)
# → Node E depends on B, C → dispatch reactively as soon as both complete
# → Maximum parallelism, minimum wait time, zero CPU waste
```

**6. Graceful Failure Handling:**
```python
# Node fails → doesn't crash entire DAG
# → Boss marks node as failed
# → Dependent nodes are skipped (dependencies not met)
# → Independent branches continue executing
# → User gets partial results + error information
```

---

#### **Comparison: Sequential vs. Event-Driven**

| Feature | Sequential Waves | Event-Driven (Recommended) |
|---------|------------------|----------------------------|
| **Parallelism** | Within wave only | Maximum (dispatch ASAP) |
| **CPU Efficiency** | Good | Excellent (zero spinning/polling) |
| **Dispatch Latency** | Delayed until wave completes | Immediate on dependency completion |
| **Cancellation** | Hard to implement | Built-in via cancellation_token |
| **HITL** | Would block entire wave | Pauses only affected branch |
| **Dynamic Replanning** | Not possible | Can inject new nodes mid-execution |
| **Progress Visibility** | Only between waves | Real-time per node |
| **Synchronization** | Simple (asyncio.gather) | asyncio.Queue + Event (semaphore-based) |
| **Complexity** | Lower | Moderate |
| **Flexibility** | Limited | High |

**Recommendation:** Use event-driven architecture as it provides better parallelism, zero CPU waste, and enables critical features (cancellation, HITL, dynamic control) that would be very difficult to add later. The complexity is well-managed through clear event handling patterns.

---

#### **Key Architecture Features:**
- ✅ **Fully event-driven**: No polling/spinning - uses semaphore-based signaling via `asyncio.Queue`
- ✅ **Zero CPU waste**: `await queue.get()` suspends coroutine until events arrive
- ✅ **Reactive dispatch**: New nodes dispatched immediately when dependencies complete
- ✅ **Concurrent execution**: Independent nodes run in parallel
- ✅ **Cancellation support**: User can stop mid-execution (via `Event.wait()`, no polling)
- ✅ **HITL-ready**: Workers can pause for approval
- ✅ **Maximum parallelism**: No wave blocking - dispatch as soon as dependencies met
- ✅ **Context passing**: Results from dependencies available to dependent nodes
- ✅ **Agent abstraction**: Executor doesn't know implementation details (uses AgentCard)
- ✅ **Multi-request support**: Handles multiple concurrent API requests safely (see below)

---

#### **Concurrency: Multiple Simultaneous Requests**

**The design naturally supports multiple concurrent API requests** executing DAGs simultaneously:

**How It Works:**
```python
# Request 1: "Plan Paris trip" with gpt-4o
async def handle_request_1():
    dag1 = await generate_dag("Plan Paris trip")
    model_config1 = ModelConfig(model="gpt-4o", temperature=0.7)
    async for result in executor.execute(dag1, session_id="abc", model_config=model_config1):
        yield result  # Stream to client 1

# Request 2: "Analyze code quality" with gpt-4o-mini (running at same time)
async def handle_request_2():
    dag2 = await generate_dag("Analyze code quality")
    model_config2 = ModelConfig(model="gpt-4o-mini", temperature=0.0)
    async for result in executor.execute(dag2, session_id="def", model_config=model_config2):
        yield result  # Stream to client 2

# Both run concurrently without interference
# Same agents (e.g., "web_researcher") but different models per request
```

**Isolation Guarantees:**
- Each `execute()` call creates isolated resources:
  - Separate `event_queue` (no cross-talk between requests)
  - Separate `state` (no shared mutable state)
  - Separate `model_config` (different models/parameters per request)
  - Separate worker tasks (each belongs to specific execution)
- Shared resources are safe:
  - `agent_registry` (immutable dict - read-only, NO model info stored here)
  - `tool_registry` (immutable dict - read-only)
  - `litellm_client` (AsyncOpenAI designed for concurrent use)

**Key Benefit - Model Flexibility:**
```python
# Same agent ("web_researcher"), different models per request
# Request A: Uses gpt-4o for high-quality research
model_config_a = ModelConfig(model="gpt-4o", temperature=0.7)

# Request B: Uses gpt-4o-mini for cost-effective research
model_config_b = ModelConfig(model="gpt-4o-mini", temperature=0.0)

# Both use same agent card (same prompt, tools, behavior)
# But different model configs (different LLM, parameters)
# No conflicts because model config passed per-request, not stored in registry
```

**Scalability:**
```
Concurrent Requests    Bottleneck
──────────────────────────────────────────────────────
10-100 requests        ✅ None (smooth operation)
100-1000 requests      ⚠️ LiteLLM rate limits + connection pool
1000+ requests         ❌ Need horizontal scaling (multiple API servers)
```

**Limiting Factors:**
1. **LiteLLM API rate limits** (external bottleneck, most restrictive)
2. **HTTP connection pool** (configurable via `HTTP_MAX_CONNECTIONS`)
3. **Memory** (minimal: ~1-10 KB per request, not a concern until 100k+ requests)
4. **CPU** (event-driven = efficient, not a concern until 10k+ requests)

**Concurrency Safety Checklist:**
- ✅ No shared mutable state between executions
- ✅ Each execution has isolated event queue and state
- ✅ Model config passed per-request (not stored in agent registry)
- ✅ Worker tasks scoped to specific execution
- ✅ Agent/tool registries are immutable (read-only, NO model info)
- ✅ LiteLLM client handles concurrent requests internally
- ✅ FastAPI automatically spawns separate coroutines per request
- ✅ Same agent can use different models across concurrent requests

---

### 5. Tool Registry (`api/tool_registry.py`)

**MCP Tools:**
- Filesystem: `read_file`, `write_file`, `list_directory`, `search_files`
- Web search: (need to research available MCP servers)
- Database: (need to research available MCP servers)

**Custom Tools:**
- Application-specific functions
- API integrations
- Data transformations

```python
# Tool registry maps tool name → function
TOOL_REGISTRY = {
    "read_file": mcp_client.call_tool("read_file", ...),
    "web_search": web_search_function,
    "summarize": summarize_function,
    # ...
}
```

---

## Open Design Questions

### Q1: Context Passing & Memory Tracking

**Challenge: How do dependent nodes access results AND track data lineage?**

**Problem:**
```python
# Node n2 depends on n1
{
    "id": "n2",
    "agent": "travel_planner",
    "objective": "Create itinerary based on previous research",
    "depends_on": ["n1"]  # How does n2 access n1's results?
}

# After execution, we need to know:
# 1. What data did n2 use from n1?
# 2. What files/sources were accessed?
# 3. What memory/context influenced the result?
```

**Dual Problem:**
1. **Context Passing**: How does n2 get n1's data?
2. **Memory Tracking**: How do we track what data/sources were used?

---

#### **Part 1: Context Passing**

**Option A: Automatic String Concatenation**
```python
# Executor adds context as user message
messages = [
    {"role": "system", "content": agent_card.prompt},
    {"role": "user", "content": f"Previous results:\n[n1]: {completed['n1']['result']}"},
    {"role": "user", "content": node.objective}
]
```

**Option B: Structured Context with Metadata**
```python
# Pass results AND metadata
context = {
    "dependencies": {
        "n1": {
            "result": "Found 5 hotels...",
            "agent": "web_researcher",
            "sources": ["booking.com", "hotels.com"],  # Data sources
            "files_accessed": [],
            "tokens_used": 1200
        }
    }
}
messages = [
    {"role": "system", "content": agent_card.prompt},
    {"role": "user", "content": f"Context: {json.dumps(context)}"},
    {"role": "user", "content": node.objective}
]
```

**Option C: LLM-Selected Context**
```python
# Another LLM call determines relevant context
relevant_context = await select_relevant_context(
    objective=node.objective,
    available_results=completed
)
```

---

#### **Part 2: Memory/Data Tracking**

**Why Track Data Lineage?**
1. **Debugging**: "Why did the itinerary include this hotel?" → Trace back to search results
2. **Caching**: "We already searched Paris hotels this session" → Reuse results
3. **Cost tracking**: "How many API calls/tokens did this orchestration use?"
4. **Compliance**: "What data sources influenced this decision?" (GDPR, auditing)
5. **Provenance**: "Which files were read to generate this code analysis?"

**Proposed: Enhanced Node Result Model**

```python
class NodeResult(BaseModel):
    """Result from executing a DAG node"""
    node_id: str
    agent: str
    result: str | dict  # Actual output

    # Memory/Data tracking
    metadata: NodeMetadata

class NodeMetadata(BaseModel):
    """Track what data/memory was used"""

    # Data sources accessed
    sources: list[DataSource] = []

    # Files accessed (via MCP filesystem)
    files_accessed: list[FileAccess] = []

    # Dependencies used
    dependencies_used: list[str] = []  # Which dependency results were actually used

    # API/tool usage
    tools_called: list[ToolCall] = []

    # Token usage
    tokens: TokenUsage | None = None

    # Timing
    duration_ms: int
    started_at: datetime
    completed_at: datetime

class DataSource(BaseModel):
    """External data source accessed"""
    type: Literal["web", "database", "api", "file"]
    identifier: str  # URL, table name, file path, etc.
    query: str | None = None  # Search query, SQL, etc.

class FileAccess(BaseModel):
    """File accessed via MCP"""
    path: str
    operation: Literal["read", "write", "search"]
    content_preview: str | None = None  # First 100 chars

class ToolCall(BaseModel):
    """Tool invocation record"""
    tool_name: str
    arguments: dict[str, Any]
    result_summary: str
```

**Example Node Result with Tracking:**
```python
NodeResult(
    node_id="research_hotels",
    agent="web_researcher",
    result="Found 5 hotels in Paris: Hotel A ($200), Hotel B ($150)...",
    metadata=NodeMetadata(
        sources=[
            DataSource(type="web", identifier="booking.com", query="Paris hotels June"),
            DataSource(type="web", identifier="hotels.com", query="Paris hotels June")
        ],
        files_accessed=[],
        dependencies_used=[],  # No dependencies
        tools_called=[
            ToolCall(
                tool_name="web_search",
                arguments={"query": "Paris hotels June", "max_results": 10},
                result_summary="10 results found"
            )
        ],
        tokens=TokenUsage(prompt=150, completion=800, total=950),
        duration_ms=3500,
        started_at=datetime(...),
        completed_at=datetime(...)
    )
)
```

**How Executor Collects This:**

```python
async def _execute_llm_agent(
    self,
    agent_card: AgentCard,
    node: DAGNode,
    completed: dict[str, NodeResult]
) -> NodeResult:
    """Execute LLM agent and track data usage."""

    start_time = datetime.now()
    metadata = NodeMetadata(started_at=start_time)

    # Track which dependencies were used
    if node.depends_on:
        metadata.dependencies_used = node.depends_on
        # Could use LLM to determine ACTUAL usage (not just all deps)

    # Build messages
    messages = [...]

    # Execute with tools
    response = await self.litellm.chat.completions.create(
        model=agent_card.model,
        messages=messages,
        tools=[self.tools[t] for t in agent_card.tools] if agent_card.tools else None
    )

    # Track tool calls from response
    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            metadata.tools_called.append(
                ToolCall(
                    tool_name=tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                    result_summary="..."  # From tool execution
                )
            )

            # If tool is web_search, track source
            if tool_call.function.name == "web_search":
                args = json.loads(tool_call.function.arguments)
                metadata.sources.append(
                    DataSource(type="web", identifier="search", query=args.get("query"))
                )

            # If tool is read_file, track file access
            if tool_call.function.name == "read_file":
                args = json.loads(tool_call.function.arguments)
                metadata.files_accessed.append(
                    FileAccess(path=args["path"], operation="read")
                )

    # Track tokens
    if response.usage:
        metadata.tokens = TokenUsage(
            prompt=response.usage.prompt_tokens,
            completion=response.usage.completion_tokens,
            total=response.usage.total_tokens
        )

    # Track timing
    end_time = datetime.now()
    metadata.completed_at = end_time
    metadata.duration_ms = int((end_time - start_time).total_seconds() * 1000)

    return NodeResult(
        node_id=node.id,
        agent=node.agent,
        result=response.choices[0].message.content,
        metadata=metadata
    )
```

**Benefits of Memory/Data Tracking:**

1. **Session-Level Caching**:
   ```python
   # Check if we already have this data in session
   if session.has_result(query="Paris hotels June"):
       return session.get_cached_result(...)
   ```

2. **Cost Tracking**:
   ```python
   # Aggregate across all nodes
   total_tokens = sum(node.metadata.tokens.total for node in results)
   total_api_calls = sum(len(node.metadata.tools_called) for node in results)
   ```

3. **Debugging/Provenance**:
   ```python
   # User asks: "Why did you recommend Hotel A?"
   # Trace back through DAG:
   # - Node "create_itinerary" used dependency "research_hotels"
   # - Node "research_hotels" accessed booking.com, hotels.com
   # - Node "research_hotels" called web_search with query "Paris hotels June"
   ```

4. **Context Window Management**:
   ```python
   # If context getting large, identify what to summarize/drop
   for node in completed.values():
       if len(node.result) > 5000:  # Large result
           # Check if downstream nodes actually used it
           if node.node_id not in downstream_deps_used:
               # Can drop or summarize
   ```

**Questions:**
- Should we track this in Phase 1 or defer?
- How detailed should tracking be? (Every tool call vs. just sources)
- Should we use this for caching immediately or just logging?
- Store session memory in Redis or in-memory?

---

### Q2: Dynamic Agent Support

**Should we support dynamic agents (not in registry)?**

**Use Case:**
User request requires specialized agent not in our predefined registry.

**Option A: Registry Only**
- Only allow agents from `AGENT_REGISTRY`
- DAG generator must map tasks to available agents
- Predictable, testable, high quality

**Option B: Dynamic Agent Card**
```python
# Special agent in registry
"dynamic_agent": AgentCard(
    name="dynamic_agent",
    description="Create custom agent for any specialized task",
    objective_template="Define task intent and agent will be created dynamically",
    type="dynamic"
)

# LLM can use it
{
    "id": "special_task",
    "agent": "dynamic_agent",
    "objective": "Analyze sentiment of customer reviews and categorize by product feature",
    "depends_on": []
}

# Executor generates prompt on-the-fly
if agent_card.type == "dynamic":
    # Use another LLM call to generate prompt from objective
    prompt = await generate_prompt_from_objective(node.objective)
    tools = await infer_needed_tools(node.objective)
```

**Questions:**
- Does dynamic agent reduce quality (LLM writes its own prompts)?
- Should this be Phase 1 or add later if needed?
- How do we prevent bad dynamic prompts?

---

### Q3: Agent Registry Storage

**Where should Agent Cards be stored?**

**Option A: Python Code (dict)**
```python
# api/agent_registry.py
AGENT_REGISTRY = {
    "web_researcher": AgentCard(...),
    "travel_planner": AgentCard(...),
}
```
- ✅ Version controlled
- ✅ Easy to start
- ✅ Type-safe
- ❌ Requires deployment to update

**Option B: Database**
```python
# Agents stored in PostgreSQL/MongoDB
agents = await db.agent_cards.find_all()
```
- ✅ Update without deployment
- ✅ Enable/disable agents dynamically
- ❌ More complex
- ❌ Migration needed

**Option C: Hybrid**
```python
# Core agents in code, custom agents in DB
registry = {**CORE_AGENTS, **await load_custom_agents_from_db()}
```

**Questions:**
- Start with code and migrate to DB later?
- Do we need runtime agent updates?

---

### Q4: Human-in-the-Loop (HITL)

**How do we handle user feedback and approval?**

**Scenarios:**
1. **Approval Required**: "Should I book this $500/night hotel?"
2. **User Interrupt**: User clicks "stop" mid-execution
3. **User Guidance**: "Actually, I prefer budget hotels"

**Option A: Extended SSE Format**
```python
# Executor streams approval request as custom event
yield "event: approval_required\n"
yield f"data: {json.dumps({
    'node_id': 'book_hotel',
    'action': 'Book hotel for $1500',
    'options': ['approve', 'reject', 'modify']
})}\n\n"

# User responds via new endpoint
POST /v1/orchestration/{session_id}/respond
{"node_id": "book_hotel", "decision": "approve"}
```

**Option B: OpenAI tool_calls Format**
```python
# Use tool_calls for "approval" function
yield {
    "choices": [{
        "delta": {
            "tool_calls": [{
                "function": {
                    "name": "request_approval",
                    "arguments": '{"action": "book_hotel", "cost": 1500}'
                }
            }]
        }
    }]
}
```

**Questions:**
- Can we extend SSE without breaking OpenAI compatibility?
- How do we maintain session state during pause (Redis)?
- Should approval be optional (flag in agent card)?

---

### Q5: Streaming and User Control

**How do we stream DAG execution progress?**

**Option A: Content Only (Hide DAG Details)**
```python
# Stream LLM responses from nodes, hide node boundaries
# User sees: natural language response, not execution details
yield {"choices": [{"delta": {"content": "I found 5 hotels in Paris..."}}]}
yield {"choices": [{"delta": {"content": "Based on your budget..."}}]}
```
- ✅ Clean UX (like ChatGPT)
- ❌ User can't see progress
- ❌ No visibility into which agent is running

**Option B: Node Metadata + Content**
```python
# Add custom "orchestration" field (OpenAI SDK ignores unknown fields)
# Node start
yield {
    "choices": [{"delta": {}, "finish_reason": None}],
    "orchestration": {"node_id": "hotel_search", "agent": "web_researcher", "status": "running"}
}

# Content from node
yield {"choices": [{"delta": {"content": "Found 5 hotels..."}}]}

# Node complete
yield {
    "choices": [{"delta": {}, "finish_reason": None}],
    "orchestration": {"node_id": "hotel_search", "status": "completed"}
}
```
- ✅ Progress visibility
- ✅ Compatible with OpenAI SDK (extra fields ignored)
- ❌ More complex streaming

**Option C: Content with Node Markers**
```python
# Embed node info in content (like reasoning agent uses emojis)
yield {"choices": [{"delta": {"content": "[web_researcher] Searching for Paris hotels...\n"}}]}
yield {"choices": [{"delta": {"content": "Found 5 hotels in central Paris.\n"}}]}
yield {"choices": [{"delta": {"content": "[travel_planner] Creating itinerary...\n"}}]}
```
- ✅ Simple (just content streaming)
- ❌ Pollutes LLM output with metadata

**User Control:**
- **Stop**: Standard OpenAI behavior (client cancels request)
- **Pause/Resume**: Need session management for HITL

**Questions:**
- Should DAG execution be transparent or hidden?
- How do we handle stop mid-DAG (cancel all running nodes)?
- Do we need resume capability (Phase 1 or later)?

---

### Q6: Dynamic DAG Re-evaluation

**Should DAG be re-evaluated after nodes complete with unexpected results?**

**Scenario:**
```
Initial DAG:
  Node 1: Research Paris hotels
  Node 2: Create itinerary (depends on Node 1)

Node 1 Result: "All hotels fully booked for selected dates"

Options:
  A) Continue with original DAG (create itinerary with empty hotel list)
  B) Re-generate DAG (pivot to different dates or alternative cities)
```

**Option A: Static DAG (Proposed for Phase 1)**
- Execute plan as initially generated
- Let final synthesis node handle unexpected results
- Simpler, more predictable, avoids infinite loops

**Option B: Dynamic Re-planning (Future Enhancement)**
```python
async def execute_with_replanning(dag: DAG):
    for node in topological_order(dag):
        result = await execute_node(node)

        # Check if result indicates replanning needed
        if should_replan(result):
            new_dag = await dag_generator.regenerate(
                original_request, completed_nodes, new_context=result
            )
            return await execute_with_replanning(new_dag)
```

**Questions:**
- Start with static DAG, add replanning later if needed?
- How do we detect replanning signals?
- Risk of infinite replanning loops?

---

## Available Tools & MCP Integration

### Existing MCP Servers

**Filesystem MCP:**
- read_file, write_file, list_directory, search_files
- **Use case**: Agent needs to read/write local files

**Other Available MCP Servers** (need to research):
- Web search
- Database access
- API integrations

**Questions:**
- Which MCP servers should we integrate first?
- Should all tools be MCP or mix of MCP + custom?
- How do we document available tools for DAG generator?

### Custom Tools

What custom tools do we need?
- Database queries specific to our app?
- Internal API calls?
- Data transformations?

---

## Initial Agent Registry (Phase 1)

**Core Agents to Build:**

1. **web_researcher** (type: `llm`)
   - Description: "Searches the web and summarizes findings on any topic"
   - Tools: `web_search`, `summarize`
   - Use cases: Research, fact-checking, current events

2. **code_analyzer** (type: `llm`)
   - Description: "Reads and analyzes code files, identifying patterns and issues"
   - Tools: `read_file`, `list_directory`, `search_files` (MCP filesystem)
   - Use cases: Code review, refactoring suggestions, bug analysis

3. **deep_reasoner** (type: `class`)
   - Description: "Performs multi-step reasoning for complex questions"
   - Implementation: Uses existing `ReasoningAgent` class
   - Use cases: Complex analysis, multi-step problem solving

**Future Agents (Phase 2+):**
- `travel_planner` - Flight/hotel booking
- `data_analyst` - SQL queries + insights
- `debug_assistant` - Error analysis + fixes

**Advanced Agent Example: Code Execution Agent (Future)**

**Use Case:** Execute Python code in isolated environment for CPU-intensive tasks

**Implementation Approach:**
- **Type:** `service` (separate service with boss/worker pattern)
- **Architecture:**
  ```
  Orchestrator
      ↓ (HTTP call via AgentCard.service_url)
  Code Execution Service (Boss)
      ├─ FastAPI endpoint
      ├─ Job queue (Redis/RabbitMQ)
      └─ Worker pool (multiple processes)
          ├─ Worker 1 (isolated Python 3.14+ environment)
          ├─ Worker 2 (isolated Python 3.14+ environment)
          └─ Worker N (isolated Python 3.14+ environment)
  ```

**Why Boss/Worker Pattern:**
- CPU-intensive code execution needs process isolation
- Python 3.14+ has better GIL-free threading for CPU tasks
- Multiple workers handle concurrent requests
- Boss manages job distribution and aggregates results

**Why Separate Service:**
- Resource isolation (prevent code execution from blocking orchestrator)
- Horizontal scaling (add more workers as needed)
- Security isolation (untrusted code runs in separate containers)
- Different runtime requirements (Python 3.14+ for workers)

**AgentCard Example:**
```python
"code_executor": AgentCard(
    name="code_executor",
    description="Generates and executes Python code in isolated environment for computational tasks",
    objective_template="Write and execute Python code to: {task_description}",
    type="service",
    service_url="http://code-execution-service:8003"  # Separate service
)
```

**Service Implementation:**
```python
# services/code_executor/main.py
from fastapi import FastAPI
from celery import Celery

app = FastAPI()
celery_app = Celery('code_executor', broker='redis://localhost:6379')

@celery_app.task
def execute_code_isolated(code: str, timeout: int = 30):
    """Execute code in isolated Python 3.14+ environment"""
    # Use subprocess with resource limits
    # Return stdout, stderr, execution time
    pass

@app.post("/execute")
async def execute(request: CodeExecutionRequest):
    # Submit to worker queue
    task = execute_code_isolated.delay(request.code)
    # Poll for result or stream progress
    return await task.get()
```

**DAG Usage Example:**
```python
{
    "nodes": [
        {
            "id": "analyze_data",
            "agent": "code_executor",
            "objective": "Write Python code to analyze CSV data and compute statistics",
            "depends_on": []
        },
        {
            "id": "visualize",
            "agent": "code_executor",
            "objective": "Generate matplotlib visualization from previous analysis",
            "depends_on": ["analyze_data"]
        }
    ]
}
```

**Benefits:**
- Isolated execution prevents orchestrator blocking
- Python 3.14+ GIL improvements enable true parallel CPU work
- Workers can be scaled independently
- Failures in code execution don't crash orchestrator
- Can add resource limits (CPU, memory, timeout) per worker

**Similar Pattern for Other Compute-Heavy Agents:**
- Video processing agent (GPU workers)
- Large file analysis agent (I/O intensive)
- ML inference agent (GPU/TPU workers)

---

## Proposed Decisions (For Discussion)

### Decision 1: Agent Registry-Based Approach ✅

**Adopted from discussion:**
- Use simplified `DAGNode` with agent references (not prompts/tools/models)
- Pre-define agents in `AGENT_REGISTRY` with `AgentCard` structure
- LLM generating DAG only chooses agents + writes objectives
- Executor looks up implementation details from registry

**Benefits:**
- Higher quality DAG generation (simpler task for LLM)
- Consistent, tested agent prompts
- Easy to version and update agents

---

### Decision 2: Context Passing - Automatic String Concatenation (Proposed)

**Proposal:**
- Use Option A: Automatic string concatenation
- Executor adds all dependency results as context

**Rationale:**
- Simple to implement
- LLMs are good at extracting relevant info from context
- Can optimize later if context becomes too large

---

### Decision 3: Registry Storage - Start with Code (Proposed)

**Proposal:**
- Phase 1: Store agents in Python dict (`AGENT_REGISTRY`)
- Future: Migrate to hybrid (core in code, custom in DB)

**Rationale:**
- Fast to build
- Version controlled
- Can add DB layer later without changing interfaces

---

### Decision 4: No Dynamic Agents in Phase 1 (Proposed)

**Proposal:**
- Only allow agents from registry initially
- Can add `dynamic_agent` later if needed

**Rationale:**
- Simpler, more predictable
- Ensures quality (all prompts tested)
- Can assess need after Phase 1

---

### Decision 5: Static DAG Execution (Proposed)

**Proposal:**
- Execute DAG as initially generated
- No mid-execution re-planning
- Let final synthesis handle unexpected results

**Rationale:**
- Simpler to implement and debug
- Avoids infinite loops
- Can add dynamic re-planning in future if needed

---

### Decision 6: Streaming Format - Defer to Implementation (Needs Decision)

**Options:**
- A: Content only (clean but no progress)
- B: Metadata + Content (progress visible, more complex)
- C: Content with markers (simple but pollutes output)

**Action:** Test options during implementation, choose based on UX

---

## Implementation Phases

### Phase 1: Core DAG System (M2)
- [ ] DAG models with validation
- [ ] DAG executor (async, concurrent)
- [ ] Basic tool registry
- [ ] Unit tests

### Phase 2: DAG Generation (M3)
- [ ] LLM-based DAG generator
- [ ] Structured output schemas
- [ ] Integration with tool/agent registry
- [ ] Tests with various request types

### Phase 3: Orchestrator Service (M4)
- [ ] FastAPI service
- [ ] Integrate DAG generator + executor
- [ ] OpenAI-compatible streaming
- [ ] Main API routing integration

### Phase 4: Enhanced Features (M5)
- [ ] Human-in-the-loop support
- [ ] Session management (Redis)
- [ ] MCP tool integration
- [ ] Error handling and retries

### Phase 5: Advanced Agents (M6)
- [ ] Migrate ReasoningAgent to class agent
- [ ] Add predefined specialist agents
- [ ] Prompt optimization
- [ ] Performance tuning

---

## Outstanding Questions for Review

### Critical (Need Decision Before Implementation):

1. **Q1 - Context Passing**: Which option for passing dependency results to downstream nodes?
   - **Recommendation**: Option A (automatic string concatenation)
   - **Need your input**: Acceptable or prefer structured approach?

2. **Q3 - Agent Registry Storage**: Code vs. Database vs. Hybrid?
   - **Recommendation**: Start with Python dict, migrate to DB later
   - **Need your input**: Do you need runtime agent updates immediately?

3. **Q5 - Streaming Format**: Content-only vs. Node metadata vs. Markers?
   - **Recommendation**: Test during implementation
   - **Need your input**: Your preference for UX?

### Important (Can Defer but Should Discuss):

4. **Q2 - Dynamic Agents**: Support dynamic agent creation or registry-only?
   - **Recommendation**: Registry-only for Phase 1
   - **Need your input**: Agree?

5. **Q6 - DAG Re-planning**: Static vs. dynamic re-evaluation?
   - **Recommendation**: Static for Phase 1
   - **Need your input**: Agree?

6. **Q4 - Human-in-the-Loop**: How to handle approval workflows?
   - **Can defer**: Not needed for Phase 1
   - **Need your input**: Should we design for this now or add later?

### Research Needed:

7. **MCP Servers**: Which MCP servers exist and should we integrate?
   - **Known**: Filesystem MCP (read_file, write_file, etc.)
   - **Need research**: Web search, database, APIs
   - **Action**: Research available MCP servers

8. **Tool Registry**: What custom tools do we need beyond MCP?
   - **Need your input**: Application-specific requirements?

---

## Next Steps

1. ✅ **Review design doc** - Iterate on agent registry approach (DONE - updated with simplified DAGNode)
2. 🔄 **Your review** - Answer outstanding questions above
3. **Research MCP servers** - Document available servers and tools
4. **Finalize Phase 1 agents** - Define 3 core agents (web_researcher, code_analyzer, deep_reasoner)
5. **Begin M2 implementation**:
   - `api/agent_registry.py` - AgentCard model + registry
   - `api/dag_models.py` - Simplified DAGNode + DAG
   - `api/dag_executor.py` - Async executor with topological sort
   - `api/tool_registry.py` - Tool mapping
   - Unit tests for all components

---

## Summary of Key Changes

**What Changed from Original Design:**
1. ❌ **Removed**: Complex DAGNode with prompts/tools/models fields
2. ✅ **Added**: Agent Registry with AgentCard (inspired by A2A concept)
3. ✅ **Simplified**: DAGNode now just references agents + specifies objective
4. ✅ **Benefit**: LLM generating DAG has simpler task = higher quality outputs

**Architecture Benefits:**
- Separation of concerns (DAG structure vs. agent implementation)
- Easy to test and version agents independently
- Can migrate to A2A protocol later if needed (AgentCard → A2A Agent Card)
- LLM makes fewer complex decisions during DAG generation

