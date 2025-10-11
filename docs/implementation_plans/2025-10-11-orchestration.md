# Implementation Plan: Multi-Agent Orchestration System

## Project Overview

### Current State
- Single FastAPI service with embedded ReasoningAgent
- OpenAI-compatible `/v1/chat/completions` endpoint
- OpenTelemetry tracing to Phoenix
- MCP protocol for tool integration
- All state in-memory (lost on restart)
- No multi-agent coordination
- No human-in-the-loop workflows
- No persistent sessions

### Target State
- Custom orchestrator coordinating multiple specialized agents (no LangGraph - simple async patterns)
- MCP protocol for agent-to-agent communication
- Persistent session management with Redis
- Human-in-the-loop workflow support with pause/resume
- Agent registry for dynamic discovery
- Streaming multi-agent responses via SSE
- Vector database (Chroma/Pinecone) for semantic memory
- Prefect for dynamic DAG workflow execution
- OpenAI-compatible external API maintained

### Key Architectural Decisions

**Why Custom Orchestration (No LangGraph):**
- Agents are just: LLM + specific prompt + specific tools (MCP)
- Can instantiate any number of agents dynamically
- No heavy abstractions - simple FastAPI orchestration logic
- Full control over coordination patterns
- Easy to understand and debug

**Why MCP Protocol for Agent Communication:**
- Industry standard for tool/agent communication
- Already integrated in existing ReasoningAgent
- Simple HTTP-based protocol
- Agent discovery via standardized endpoints
- No vendor lock-in

**Why Direct Redis (Not Dapr):**
- Simple key-value operations sufficient for session state
- Lower complexity than abstraction layers
- Direct client control and debugging
- Standard Python ecosystem tooling

**Why Prefect for Dynamic Workflows:**
- Excellent for AI-specific DAG execution
- Dynamic workflow generation based on user requests
- Good observability and debugging
- Can integrate with LangGraph for complex orchestration

**Why SSE for Streaming:**
- Native FastAPI support
- Works with existing OpenAI streaming pattern
- Simple browser compatibility
- No additional infrastructure needed

**No Backwards Compatibility:**
- Breaking changes encouraged for better architecture
- Clean design prioritized over migration paths
- Remove legacy patterns that conflict with new design
- Focus on production-ready patterns from the start

---

## Implementation Phases

This plan is organized into focused milestones that build incrementally toward the full orchestration system.

**Phase 1: Foundation (Milestones 1-3)**
- Session management infrastructure
- Cancellation and human-in-the-loop core
- Redis integration for persistent state

**Phase 2: Multi-Agent Core (Milestones 4-6)**
- MCP-based agent registry
- Custom orchestrator service (FastAPI)
- Multi-agent coordination patterns

**Phase 3: Advanced Features (Milestones 7-9)**
- Prefect DAG workflows
- Vector database semantic memory
- Production hardening and optimization

---

## Milestone 1: Connection-Based Cancellation with AsyncIO

### Goal
Implement OpenAI-compatible cancellation using asyncio task management. Clients can cancel reasoning by closing HTTP connections (AbortController), which immediately interrupts ongoing LLM calls via `asyncio.CancelledError` propagation.

**Scope:** Single-request cancellation only. Each client controls their own reasoning process via connection closing. Cross-client/cross-tab cancellation requires session management (Milestone 2).

### Why This First
- Foundation for all async patterns in subsequent milestones
- Proves asyncio cancellation works with OpenAI client
- No new dependencies - builds on existing FastAPI patterns
- Low risk - isolated to request lifecycle
- Essential for responsive user experience

### Success Criteria
- [ ] Clients can cancel by closing connections (`AbortController.abort()`)
- [ ] Cancellation interrupts ongoing LLM calls immediately
- [ ] Cancelled streams end gracefully with proper OpenAI finish_reason
- [ ] Multiple concurrent clients operate independently
- [ ] No resource leaks or orphaned tasks
- [ ] All existing tests pass (no regression)
- [ ] Tracing properly reflects cancellation (status: OK, message: "cancelled by client")

### Prerequisites and Learning

**Required Reading:**
- [FastAPI Async Patterns](https://fastapi.tiangolo.com/async/) - Understanding async request handling
- [Python AsyncIO Cancellation](https://docs.python.org/3/library/asyncio-task.html#cancellation) - How CancelledError works
- [FastAPI Request Disconnection](https://fastapi.tiangolo.com/advanced/custom-request-and-route/#accessing-the-request-body-in-an-exception-handler) - Detecting client disconnect
- [OpenAI Streaming Spec](https://platform.openai.com/docs/api-reference/streaming) - Proper stream termination

**Key Concepts:**
- `asyncio.create_task()` for cancellation control
- `task.cancel()` for explicit cancellation
- `CancelledError` propagation through async call chains
- `request.is_disconnected()` for connection monitoring
- OpenTelemetry span handling during cancellation

### Key Changes

#### 1. Update `api/main.py` Chat Completions Endpoint

**What:** Add connection monitoring and task-based cancellation to streaming endpoint.

**Why:** Need explicit task handle to call `.cancel()` when client disconnects. Current generator pattern doesn't provide cancellation control.

**Pattern:**
```python
@app.post("/v1/chat/completions")
async def chat_completions(
    request: OpenAIChatRequest,
    http_request: Request,  # Make required (remove = None default)
    reasoning_agent: ReasoningAgent = Depends(get_reasoning_agent),
):
    if request.stream:
        # Create task for cancellation control
        stream_generator = reasoning_agent.execute_stream(request, parent_span=span)

        async def cancellation_aware_stream():
            try:
                async for chunk in stream_generator:
                    # Check disconnection before each yield
                    if await http_request.is_disconnected():
                        # Client disconnected - cancellation will propagate
                        break
                    yield format_sse_chunk(chunk)

            except asyncio.CancelledError:
                # Update tracing span
                span.set_status(trace.Status(trace.StatusCode.OK, "Request cancelled by client"))
                # Re-raise to properly close stream
                raise
            finally:
                span.end()

        return StreamingResponse(
            cancellation_aware_stream(),
            media_type="text/event-stream"
        )
```

**Key Implementation Details:**
- Make `http_request: Request` parameter required (remove `= None`)
- Check `await http_request.is_disconnected()` before each chunk yield
- Handle `asyncio.CancelledError` gracefully in tracing
- Set span status to OK (not ERROR) with cancellation message
- Ensure span ends properly in finally block

#### 2. Update `api/reasoning_agent.py` for Natural Cancellation Propagation

**What:** Remove any existing error handling that catches and suppresses `CancelledError`. Let it propagate naturally.

**Why:** AsyncIO's cancellation mechanism works via exception propagation. Catching `CancelledError` breaks the cancellation chain.

**What NOT to do:**
```python
# DON'T DO THIS - breaks cancellation
try:
    response = await self.openai_client.chat.completions.create(...)
except asyncio.CancelledError:
    logger.info("Cancelled")  # This suppresses the error!
    return None  # Cancellation won't propagate
```

**Correct pattern:**
```python
# DO THIS - let it propagate naturally
async def _generate_reasoning_step(self, ...):
    # Just call the async function - CancelledError will bubble up automatically
    response = await self.openai_client.chat.completions.create(...)
    return response
    # No try/except for CancelledError needed
```

**Verification Points:**
- Review all `try/except` blocks in reasoning_agent.py
- Ensure no broad `except Exception` that would catch CancelledError
- If catching specific exceptions, let CancelledError pass through
- Test that cancellation during OpenAI calls actually interrupts them

#### 3. Update OpenTelemetry Tracing for Cancellation

**What:** Add proper span status handling for cancelled requests.

**Why:** Cancelled requests are not errors - they're normal user actions. Tracing should reflect this.

**Pattern:**
```python
# In cancellation handler
span.set_attribute("cancellation.source", "client_disconnect")
span.set_attribute("http.status_code", 200)  # Not an error
span.set_status(trace.Status(trace.StatusCode.OK, "Request cancelled by client"))
```

### Testing Strategy

#### Unit Tests (tests/test_reasoning_agent.py)

**Mock-based cancellation tests:**
```python
@pytest.mark.asyncio
async def test_cancellation_during_reasoning_step():
    """Test CancelledError propagates through reasoning pipeline."""
    # Mock OpenAI client with delay
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=asyncio.CancelledError()
    )

    agent = ReasoningAgent(http_client=mock_http_client, openai_client=mock_client)

    # Create task and cancel it
    task = asyncio.create_task(agent.execute_stream(request))
    task.cancel()

    # Verify CancelledError propagates
    with pytest.raises(asyncio.CancelledError):
        async for _ in task:
            pass
```

**Multi-client isolation test:**
```python
@pytest.mark.asyncio
async def test_multiple_clients_independent_cancellation():
    """Cancelling one client doesn't affect others."""
    agent = ReasoningAgent(...)

    # Start two concurrent reasoning tasks
    task_a = asyncio.create_task(agent.execute_stream(request_a))
    task_b = asyncio.create_task(agent.execute_stream(request_b))

    # Let them start
    await asyncio.sleep(0.1)

    # Cancel only task A
    task_a.cancel()

    # Verify task B continues
    chunks_b = []
    async for chunk in task_b:
        chunks_b.append(chunk)

    assert len(chunks_b) > 0  # Task B completed
    assert task_a.cancelled()  # Task A was cancelled
```

#### Integration Tests (tests/integration/test_cancellation.py)

**Mark with `@pytest.mark.integration` - requires OPENAI_API_KEY:**

```python
@pytest.mark.integration
async def test_real_openai_cancellation():
    """Test cancellation interrupts actual OpenAI API calls."""
    async with httpx.AsyncClient() as client:
        # Start streaming request
        request = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Write a very long story"}],
            "stream": True
        }

        async with client.stream("POST", "http://localhost:8000/v1/chat/completions", json=request) as response:
            # Read first chunk
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    break

            # Close connection (cancellation)
            await response.aclose()

        # Verify no resource leaks (check server logs, metrics)
```

**Client disconnection simulation:**
```python
@pytest.mark.integration
async def test_client_abort_controller():
    """Test AbortController pattern (browser client simulation)."""
    # Use httpx timeout to simulate abort
    async with httpx.AsyncClient(timeout=1.0) as client:
        try:
            async with client.stream("POST", "http://localhost:8000/v1/chat/completions", json=request) as response:
                # Timeout will trigger during stream
                async for line in response.aiter_lines():
                    await asyncio.sleep(0.5)  # Slow consumption
        except httpx.ReadTimeout:
            pass  # Expected - simulates AbortController.abort()

    # Server should handle gracefully
```

#### Edge Cases to Test

- **Rapid cancellation**: Cancel immediately after starting
- **Cancellation during tool execution**: If MCP tools are running
- **Multiple rapid cancellations**: Stress test with many quick cancel/start cycles
- **Concurrent cancellations**: Multiple clients cancelling simultaneously
- **Natural completion race**: Cancel just as stream naturally completes

### Implementation Checklist

- [ ] Read all prerequisite documentation
- [ ] Review existing `api/main.py` streaming implementation
- [ ] Review existing `api/reasoning_agent.py` error handling
- [ ] Make `http_request: Request` parameter required in endpoint
- [ ] Add `request.is_disconnected()` check in streaming loop
- [ ] Add `asyncio.CancelledError` handler with proper tracing
- [ ] Review all try/except blocks in reasoning_agent.py
- [ ] Remove any CancelledError suppression
- [ ] Write unit tests for cancellation propagation
- [ ] Write integration tests with real OpenAI calls
- [ ] Test multi-client isolation
- [ ] Test edge cases (rapid cancel, concurrent, etc.)
- [ ] Update tracing to reflect cancellation as OK status
- [ ] Run all existing tests - ensure no regression
- [ ] Update documentation with cancellation behavior

### Risk Factors

**AsyncIO Complexity:**
- Risk: Improper task lifecycle management
- Mitigation: Thorough testing of task.cancel() behavior, review asyncio best practices

**Resource Cleanup:**
- Risk: OpenAI client connections not closing properly
- Mitigation: Test connection counts before/after cancellation, monitor for leaks

**Race Conditions:**
- Risk: Cancellation timing vs. natural completion
- Mitigation: Test edge cases, use asyncio synchronization primitives if needed

**Error Propagation:**
- Risk: CancelledError caught and suppressed somewhere
- Mitigation: Code review of all exception handlers, explicit tests

### Questions to Resolve Before Implementation

1. Should non-streaming requests also support cancellation? (Current scope: streaming only)
2. What's the expected behavior if cancellation happens during MCP tool execution?
3. Should we log cancellation events separately from errors in structured logs?
4. Any specific OpenTelemetry attributes needed for cancellation tracking?

### Documentation Updates

- Update `README.md` with cancellation behavior section
- Add client examples showing `AbortController` usage
- Document limitations (single-request scope)
- Add troubleshooting section for cancellation issues
- Update API documentation with cancellation semantics

---

**STOP HERE - Complete Milestone 1 fully (implementation + tests + docs) before proceeding to Milestone 2.**

---

## Milestone 2: Session Management Infrastructure with Redis

### Goal
Create centralized session management system using Redis for persistent state. Enables cross-request coordination needed for human-in-the-loop workflows and cross-client cancellation (mobile app cancelling web browser reasoning, Tab A cancelling Tab B).

**New Capability:** Session IDs enable coordination across different HTTP requests/clients and survive server restarts.

### Why This Next
- Builds on Milestone 1's cancellation foundation
- Required for human-in-the-loop (Milestone 3)
- Enables persistent state across server restarts
- Foundation for multi-agent orchestration (later milestones)
- Relatively isolated - doesn't affect existing single-request flows

### Success Criteria
- [ ] Redis client integrated with connection pooling
- [ ] Session manager handles lifecycle (create, get, update, delete)
- [ ] Sessions persist across server restarts
- [ ] Automatic cleanup of expired/abandoned sessions
- [ ] Session IDs passed via `X-Session-ID` header
- [ ] Thread-safe session operations
- [ ] Graceful handling of Redis unavailability
- [ ] Integration with dependency injection system
- [ ] Optional explicit cancellation endpoint (`POST /v1/reasoning/cancel`)

### Prerequisites and Learning

**Required Reading:**
- [Redis Python Client](https://redis.readthedocs.io/en/stable/) - Official redis-py documentation
- [Redis Best Practices](https://redis.io/docs/manual/patterns/) - Key patterns and anti-patterns
- [FastAPI Dependency Injection](https://fastapi.tiangolo.com/tutorial/dependencies/) - Advanced DI patterns
- [FastAPI Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/) - For cleanup tasks

**Key Concepts:**
- Redis connection pooling for performance
- TTL (Time To Live) for automatic expiration
- JSON serialization for complex state
- Session lifecycle state machine
- Background cleanup tasks

### Key Changes

#### 1. Add Redis Dependency to `pyproject.toml`

**What:** Add `redis` package with async support.

**Why:** Need persistent key-value store for session state.

```toml
[project]
dependencies = [
    # ... existing dependencies
    "redis>=5.0.0",  # Async Redis client
]
```

#### 2. Create `api/session_manager.py`

**What:** Core session management module with Redis backend.

**Why:** Centralized session logic, clean interface for other components.

**Key interfaces:**
```python
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import redis.asyncio as redis
import json
import asyncio
from datetime import datetime

class SessionStatus(str, Enum):
    """Session lifecycle states."""
    ACTIVE = "active"
    WAITING_INPUT = "waiting_input"  # Human-in-the-loop
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ReasoningSession:
    """Session state model."""
    session_id: str
    status: SessionStatus
    created_at: float
    updated_at: float
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Human-in-the-loop fields (Milestone 3)
    human_input_request: Optional[Dict[str, Any]] = None
    human_input_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        """Serialize for Redis storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'ReasoningSession':
        """Deserialize from Redis."""
        return cls(**data)

class ReasoningSessionManager:
    """Manages session lifecycle with Redis backend."""

    def __init__(
        self,
        redis_client: redis.Redis,
        default_ttl: int = 3600,  # 1 hour default
        cleanup_interval: int = 300,  # 5 minutes
    ):
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self._cleanup_task: Optional[asyncio.Task] = None

    async def create_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReasoningSession:
        """Create new session and persist to Redis."""
        pass  # Implementation by agent

    async def get_session(self, session_id: str) -> Optional[ReasoningSession]:
        """Load session from Redis."""
        pass

    async def update_session(self, session: ReasoningSession) -> bool:
        """Update session state in Redis."""
        pass

    async def delete_session(self, session_id: str) -> bool:
        """Remove session from Redis."""
        pass

    async def set_status(
        self,
        session_id: str,
        status: SessionStatus,
        metadata_update: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update session status atomically."""
        pass

    async def start_cleanup_task(self):
        """Start background task to cleanup expired sessions."""
        pass

    async def stop_cleanup_task(self):
        """Stop background cleanup task."""
        pass

    async def _cleanup_expired_sessions(self):
        """Background task - remove expired sessions."""
        pass
```

**Implementation Details:**
- Use `redis.asyncio` for async operations
- Store sessions as JSON with keys like `session:{session_id}`
- Set TTL on all keys for automatic expiration
- Handle Redis connection errors gracefully (log, don't crash)
- Use Redis pipelining for atomic updates

#### 3. Update `api/dependencies.py`

**What:** Add Redis client and session manager to ServiceContainer.

**Why:** Centralized lifecycle management, dependency injection.

**Pattern:**
```python
from api.session_manager import ReasoningSessionManager
import redis.asyncio as redis

class ServiceContainer:
    def __init__(self):
        # ... existing fields
        self.redis_client: Optional[redis.Redis] = None
        self.session_manager: Optional[ReasoningSessionManager] = None

    async def initialize(self):
        """Initialize all services."""
        # ... existing initialization

        # Initialize Redis
        self.redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            password=config.REDIS_PASSWORD,
            decode_responses=True,  # Get strings, not bytes
            max_connections=config.REDIS_MAX_CONNECTIONS,
        )

        # Test connection
        await self.redis_client.ping()

        # Initialize session manager
        self.session_manager = ReasoningSessionManager(
            redis_client=self.redis_client,
            default_ttl=config.SESSION_TTL,
        )

        # Start cleanup background task
        await self.session_manager.start_cleanup_task()

    async def cleanup(self):
        """Cleanup all resources."""
        # Stop session cleanup
        if self.session_manager:
            await self.session_manager.stop_cleanup_task()

        # Close Redis
        if self.redis_client:
            await self.redis_client.close()

        # ... existing cleanup

# Dependency function
async def get_session_manager(
    container: ServiceContainer = Depends(get_service_container)
) -> ReasoningSessionManager:
    return container.session_manager
```

#### 4. Update `api/config.py`

**What:** Add Redis configuration settings.

**Why:** Externalize configuration for different environments.

```python
class Settings(BaseSettings):
    # ... existing settings

    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_MAX_CONNECTIONS: int = 50
    SESSION_TTL: int = 3600  # 1 hour
    SESSION_CLEANUP_INTERVAL: int = 300  # 5 minutes
```

#### 5. Update `api/main.py`

**What:** Extract/create session IDs, return in headers, integrate session tracking.

**Why:** Enable clients to track and coordinate sessions.

**Pattern:**
```python
from api.session_manager import ReasoningSessionManager, SessionStatus

@app.post("/v1/chat/completions")
async def chat_completions(
    request: OpenAIChatRequest,
    http_request: Request,
    reasoning_agent: ReasoningAgent = Depends(get_reasoning_agent),
    session_manager: ReasoningSessionManager = Depends(get_session_manager),
):
    # Extract or create session ID
    session_id = http_request.headers.get("X-Session-ID")
    if not session_id:
        session_id = str(uuid.uuid4())
        # Create new session
        await session_manager.create_session(
            session_id=session_id,
            metadata={"model": request.model, "started_at": time.time()}
        )
    else:
        # Verify session exists
        session = await session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

    # Set session to active
    await session_manager.set_status(session_id, SessionStatus.ACTIVE)

    try:
        if request.stream:
            # ... streaming logic from Milestone 1

            async def session_aware_stream():
                try:
                    async for chunk in cancellation_aware_stream():
                        yield chunk
                except asyncio.CancelledError:
                    # Update session status
                    await session_manager.set_status(
                        session_id,
                        SessionStatus.CANCELLED
                    )
                    raise
                except Exception as e:
                    # Update session status
                    await session_manager.set_status(
                        session_id,
                        SessionStatus.FAILED,
                        metadata_update={"error": str(e)}
                    )
                    raise
                else:
                    # Success
                    await session_manager.set_status(
                        session_id,
                        SessionStatus.COMPLETED
                    )

            response = StreamingResponse(
                session_aware_stream(),
                media_type="text/event-stream"
            )
            # Return session ID in header
            response.headers["X-Session-ID"] = session_id
            return response

    except Exception as e:
        await session_manager.set_status(session_id, SessionStatus.FAILED)
        raise
```

#### 6. Add Explicit Cancellation Endpoint

**What:** Optional endpoint for cross-client cancellation.

**Why:** Enables mobile app to cancel web browser reasoning, or Tab A to cancel Tab B.

```python
from pydantic import BaseModel

class CancelRequest(BaseModel):
    session_id: str
    reason: Optional[str] = None

@app.post("/v1/reasoning/cancel")
async def cancel_reasoning(
    cancel_request: CancelRequest,
    session_manager: ReasoningSessionManager = Depends(get_session_manager),
):
    """Cancel active reasoning session."""
    session = await session_manager.get_session(cancel_request.session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status not in [SessionStatus.ACTIVE, SessionStatus.WAITING_INPUT]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel session in status: {session.status}"
        )

    # Update session status
    success = await session_manager.set_status(
        cancel_request.session_id,
        SessionStatus.CANCELLED,
        metadata_update={"cancel_reason": cancel_request.reason}
    )

    # Note: Actual task cancellation happens via connection monitoring (Milestone 1)
    # This endpoint just marks the session as cancelled for cross-client coordination

    return {
        "session_id": cancel_request.session_id,
        "status": "cancelled",
        "success": success
    }

@app.get("/v1/reasoning/session/{session_id}")
async def get_session_status(
    session_id: str,
    session_manager: ReasoningSessionManager = Depends(get_session_manager),
):
    """Get session status and metadata."""
    session = await session_manager.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session.session_id,
        "status": session.status,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "metadata": session.metadata,
    }
```

### Testing Strategy

#### Unit Tests (tests/test_session_manager.py)

**Redis mock-based tests:**
```python
import pytest
from unittest.mock import AsyncMock
from api.session_manager import ReasoningSessionManager, SessionStatus

@pytest.mark.asyncio
async def test_create_session():
    """Test session creation and persistence."""
    mock_redis = AsyncMock()
    manager = ReasoningSessionManager(mock_redis)

    session = await manager.create_session("test-123", user_id="user-1")

    assert session.session_id == "test-123"
    assert session.status == SessionStatus.ACTIVE
    assert session.user_id == "user-1"
    mock_redis.setex.assert_called_once()  # Verify Redis call

@pytest.mark.asyncio
async def test_update_session_status():
    """Test atomic status updates."""
    mock_redis = AsyncMock()
    manager = ReasoningSessionManager(mock_redis)

    success = await manager.set_status("test-123", SessionStatus.COMPLETED)

    assert success
    mock_redis.get.assert_called()
    mock_redis.setex.assert_called()

@pytest.mark.asyncio
async def test_cleanup_expired_sessions():
    """Test background cleanup task."""
    mock_redis = AsyncMock()
    mock_redis.scan_iter.return_value = [
        "session:old-1",
        "session:old-2",
    ]

    manager = ReasoningSessionManager(mock_redis, cleanup_interval=0.1)
    await manager.start_cleanup_task()
    await asyncio.sleep(0.2)
    await manager.stop_cleanup_task()

    # Verify cleanup ran
    assert mock_redis.scan_iter.called
```

#### Integration Tests (tests/integration/test_session_redis.py)

**Real Redis integration tests:**
```python
@pytest.mark.integration
async def test_session_persistence_real_redis():
    """Test sessions persist to real Redis instance."""
    redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    manager = ReasoningSessionManager(redis_client, default_ttl=60)

    # Create session
    session = await manager.create_session("integration-test")
    assert session.session_id == "integration-test"

    # Retrieve session
    retrieved = await manager.get_session("integration-test")
    assert retrieved is not None
    assert retrieved.session_id == session.session_id
    assert retrieved.status == SessionStatus.ACTIVE

    # Cleanup
    await manager.delete_session("integration-test")
    await redis_client.close()

@pytest.mark.integration
async def test_cross_client_cancellation_api():
    """Test explicit cancellation endpoint."""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        # Start reasoning with session ID
        session_id = str(uuid.uuid4())

        # Make request with session header
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [...]},
            headers={"X-Session-ID": session_id}
        )

        # Cancel from different client
        cancel_response = await client.post(
            "/v1/reasoning/cancel",
            json={"session_id": session_id, "reason": "User cancelled"}
        )

        assert cancel_response.status_code == 200
        assert cancel_response.json()["status"] == "cancelled"

        # Verify session status
        status_response = await client.get(f"/v1/reasoning/session/{session_id}")
        assert status_response.json()["status"] == "cancelled"
```

#### Edge Cases to Test

- **Redis unavailable**: Graceful degradation when Redis is down
- **Concurrent session updates**: Race conditions with multiple writers
- **Session expiration**: TTL behavior and cleanup
- **Large metadata**: Session with large metadata objects
- **Invalid session IDs**: Malformed or non-existent sessions

### Implementation Checklist

- [ ] Read all prerequisite documentation (Redis, FastAPI DI)
- [ ] Add redis dependency to pyproject.toml
- [ ] Create api/session_manager.py with core classes
- [ ] Implement SessionManager methods (create, get, update, delete)
- [ ] Implement background cleanup task
- [ ] Add Redis configuration to api/config.py
- [ ] Update api/dependencies.py with Redis client and SessionManager
- [ ] Update api/main.py to extract/create session IDs
- [ ] Add session status tracking to streaming endpoint
- [ ] Implement POST /v1/reasoning/cancel endpoint
- [ ] Implement GET /v1/reasoning/session/{id} endpoint
- [ ] Write unit tests with Redis mocks
- [ ] Write integration tests with real Redis
- [ ] Test concurrent session operations
- [ ] Test Redis failure scenarios
- [ ] Update docker-compose.yml with Redis service
- [ ] Update .env.example with Redis config
- [ ] Update documentation

### Docker Compose Updates

Add Redis service to `docker-compose.yml`:

```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

volumes:
  redis_data:
```

### Risk Factors

**Redis Availability:**
- Risk: Service unavailable, sessions lost
- Mitigation: Graceful degradation, connection retry logic, health checks

**Memory Management:**
- Risk: Redis memory exhaustion with many sessions
- Mitigation: TTL on all keys, monitoring, cleanup tasks

**Concurrency:**
- Risk: Race conditions with concurrent session updates
- Mitigation: Redis atomic operations, proper locking patterns

**Data Consistency:**
- Risk: Session state inconsistent with actual request state
- Mitigation: Clear state transitions, validation, testing

### Questions to Resolve Before Implementation

1. What should happen if Redis is unavailable? (Degrade to in-memory? Fail requests?)
2. Should sessions have different TTLs based on status? (e.g., completed sessions expire faster)
3. What metadata should be automatically tracked? (request timing, model used, etc.)
4. Should we compress large session metadata? (Above what size?)
5. How to handle session ID collisions? (UUID should prevent, but what if?)

### Documentation Updates

- Update README.md with Redis setup instructions
- Document session lifecycle and status transitions
- Add examples of using X-Session-ID header
- Document cancellation endpoint usage
- Add troubleshooting section for Redis connection issues
- Update environment variable documentation

---

**STOP HERE - Complete Milestone 2 fully (implementation + tests + docs) before proceeding to Milestone 3.**

---

## Milestone 3: Human-in-the-Loop Core Functionality

### Goal
Enable reasoning processes to pause and request human input, then resume execution with the provided response. Supports timeout handling, input validation, and graceful continuation of reasoning workflows.

**New Capability:** Reasoning agent can pause mid-execution, wait for human approval/input, then continue with the response incorporated into its context.

### Why This Next
- Critical feature for production AI systems (safety, oversight)
- Builds directly on session management from Milestone 2
- Tests async coordination patterns needed for orchestration
- Relatively self-contained before multi-agent complexity
- High value for user control and trust

### Success Criteria
- [ ] Reasoning can request human input and pause execution
- [ ] Human responses provided via separate HTTP endpoint
- [ ] Timeout handling for unresponded input requests
- [ ] Multiple input types supported (text, confirmation, multiple choice)
- [ ] Input/response matching via unique request IDs
- [ ] Proper SSE events for input requests and responses
- [ ] Resumed reasoning incorporates human input into context
- [ ] Session state reflects waiting/resumed status correctly

### Prerequisites and Learning

**Required Reading:**
- [AsyncIO Events and Conditions](https://docs.python.org/3/library/asyncio-sync.html) - For wait/notify patterns
- [AsyncIO wait_for with timeout](https://docs.python.org/3/library/asyncio-task.html#asyncio.wait_for) - Timeout handling
- [Server-Sent Events Spec](https://html.spec.whatwg.org/multipage/server-sent-events.html) - SSE event types
- [Pydantic Discriminated Unions](https://docs.pydantic.dev/latest/concepts/unions/) - For different input types

**Key Concepts:**
- AsyncIO Event for wait/notify coordination
- `asyncio.wait_for()` for timeout handling
- SSE custom event types for human input requests
- State transitions: ACTIVE → WAITING_INPUT → ACTIVE → COMPLETED

### Key Changes

#### 1. Extend `api/reasoning_models.py`

**What:** Add human-in-the-loop action and input/response models.

**Why:** Need structured types for different human input scenarios.

```python
from enum import Enum
from typing import Literal, Union
from pydantic import BaseModel, Field

class ReasoningAction(str, Enum):
    """Extended with human input action."""
    CONTINUE = "continue"
    USE_TOOL = "use_tool"
    PROVIDE_ANSWER = "provide_answer"
    REQUEST_HUMAN_INPUT = "request_human_input"  # NEW

# Input request types using discriminated unions
class TextInputRequest(BaseModel):
    """Request free-form text input from human."""
    type: Literal["text"] = "text"
    prompt: str
    placeholder: Optional[str] = None
    timeout_seconds: int = 300  # 5 minutes default

class ConfirmationRequest(BaseModel):
    """Request yes/no confirmation from human."""
    type: Literal["confirmation"] = "confirmation"
    prompt: str
    action_description: str  # What will happen if confirmed
    default: Optional[bool] = None
    timeout_seconds: int = 60

class MultipleChoiceRequest(BaseModel):
    """Request selection from multiple options."""
    type: Literal["multiple_choice"] = "multiple_choice"
    prompt: str
    options: List[str]
    timeout_seconds: int = 120

HumanInputRequest = Union[
    TextInputRequest,
    ConfirmationRequest,
    MultipleChoiceRequest
]

class HumanInputResponse(BaseModel):
    """Human's response to input request."""
    request_id: str
    response_type: Literal["text", "confirmation", "multiple_choice"]
    value: Union[str, bool, int]  # int for choice index
    timestamp: float

class ReasoningStep(BaseModel):
    """Extended with human input request field."""
    thought: str
    next_action: ReasoningAction
    tool_name: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    answer: Optional[str] = None
    human_input_request: Optional[HumanInputRequest] = None  # NEW
```

**Implementation Details:**
- Use Pydantic discriminated unions for type-safe input requests
- Each request type has appropriate timeout defaults
- Request IDs generated as UUIDs for tracking

#### 2. Update `api/session_manager.py`

**What:** Add human input coordination methods.

**Why:** Sessions need to track pending input requests and responses.

```python
import asyncio

class ReasoningSessionManager:
    """Extended with human input coordination."""

    def __init__(self, redis_client, default_ttl, cleanup_interval):
        # ... existing init
        # Track in-memory events for active input waits
        self._input_events: Dict[str, asyncio.Event] = {}

    async def request_human_input(
        self,
        session_id: str,
        request: HumanInputRequest
    ) -> str:
        """
        Register human input request for session.
        Returns request_id for tracking.
        """
        request_id = str(uuid.uuid4())

        # Load session
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Update session state
        session.status = SessionStatus.WAITING_INPUT
        session.human_input_request = {
            "request_id": request_id,
            "request": request.model_dump(),
            "created_at": time.time(),
        }
        session.human_input_response = None

        await self.update_session(session)

        # Create event for this request
        self._input_events[request_id] = asyncio.Event()

        return request_id

    async def wait_for_human_input(
        self,
        request_id: str,
        timeout: int
    ) -> Optional[HumanInputResponse]:
        """
        Wait for human to provide input.
        Returns None if timeout expires.
        """
        event = self._input_events.get(request_id)
        if not event:
            raise ValueError(f"No input request for {request_id}")

        try:
            # Wait with timeout
            await asyncio.wait_for(event.wait(), timeout=timeout)

            # Load response from session
            # Find session by request_id (scan Redis or use index)
            session = await self._find_session_by_request_id(request_id)
            if session and session.human_input_response:
                return HumanInputResponse(**session.human_input_response)

            return None

        except asyncio.TimeoutError:
            # Timeout - cleanup event
            del self._input_events[request_id]
            return None

    async def provide_human_input(
        self,
        request_id: str,
        response: HumanInputResponse
    ) -> bool:
        """
        Provide human response to pending request.
        Notifies waiting task to resume.
        """
        # Find session by request_id
        session = await self._find_session_by_request_id(request_id)
        if not session:
            return False

        # Validate request exists and is waiting
        if session.status != SessionStatus.WAITING_INPUT:
            return False

        # Store response
        session.human_input_response = response.model_dump()
        session.status = SessionStatus.ACTIVE
        await self.update_session(session)

        # Notify waiting task
        event = self._input_events.get(request_id)
        if event:
            event.set()
            del self._input_events[request_id]

        return True

    async def _find_session_by_request_id(self, request_id: str) -> Optional[ReasoningSession]:
        """Find session containing this request_id."""
        # Scan Redis keys or maintain index
        # Implementation by agent
        pass
```

**Implementation Details:**
- Use `asyncio.Event` for in-memory wait/notify
- Store request/response in session for persistence
- Handle timeout gracefully (cleanup, return None)
- Use Redis scan or maintain request_id → session_id index

#### 3. Update `api/reasoning_agent.py`

**What:** Handle REQUEST_HUMAN_INPUT action in reasoning loop.

**Why:** Reasoning needs to pause, request input, wait, then continue.

**Pattern:**
```python
async def _core_reasoning_process(
    self,
    request: OpenAIChatRequest,
    session_id: str,
    parent_span: Optional[trace.Span] = None,
) -> AsyncGenerator[ReasoningEvent, None]:
    """Core reasoning loop with human input support."""

    # ... existing setup

    while iteration < max_iterations:
        # Generate reasoning step
        step = await self._generate_reasoning_step(context)

        # Handle different actions
        if step.next_action == ReasoningAction.REQUEST_HUMAN_INPUT:
            # Pause for human input
            yield ReasoningEvent(
                type=ReasoningEventType.HUMAN_INPUT_REQUIRED,
                data=step.human_input_request.model_dump(),
            )

            # Request input via session manager
            request_id = await self.session_manager.request_human_input(
                session_id=session_id,
                request=step.human_input_request
            )

            # Wait for response
            response = await self.session_manager.wait_for_human_input(
                request_id=request_id,
                timeout=step.human_input_request.timeout_seconds
            )

            if response is None:
                # Timeout - handle gracefully
                yield ReasoningEvent(
                    type=ReasoningEventType.HUMAN_INPUT_TIMEOUT,
                    data={"request_id": request_id}
                )
                # Either fail or continue without input
                break

            # Got response - incorporate into context
            yield ReasoningEvent(
                type=ReasoningEventType.HUMAN_INPUT_RECEIVED,
                data=response.model_dump(),
            )

            # Add to context
            context.append({
                "role": "user",
                "content": f"Human provided input: {response.value}"
            })

            # Continue reasoning
            continue

        # ... handle other actions (CONTINUE, USE_TOOL, etc.)
```

**Implementation Details:**
- Yield SSE events for input required/received/timeout
- Pass session_id to reasoning agent (new parameter)
- Inject session_manager into ReasoningAgent via dependencies
- Handle timeout gracefully (don't crash, log, maybe retry)

#### 4. Add Human Input Endpoint to `api/main.py`

**What:** Endpoint for clients to provide human input responses.

**Why:** Separate endpoint allows different clients/tabs to provide input.

```python
from api.reasoning_models import HumanInputResponse

@app.post("/v1/reasoning/human-input")
async def provide_human_input(
    response: HumanInputResponse,
    session_manager: ReasoningSessionManager = Depends(get_session_manager),
):
    """Provide human response to pending input request."""

    success = await session_manager.provide_human_input(
        request_id=response.request_id,
        response=response
    )

    if not success:
        raise HTTPException(
            status_code=400,
            detail="Invalid request_id or no pending request"
        )

    return {
        "request_id": response.request_id,
        "status": "received",
        "success": True
    }

@app.get("/v1/reasoning/pending-input/{session_id}")
async def get_pending_input_request(
    session_id: str,
    session_manager: ReasoningSessionManager = Depends(get_session_manager),
):
    """Get pending input request for session (if any)."""
    session = await session_manager.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != SessionStatus.WAITING_INPUT:
        return {"pending": False}

    return {
        "pending": True,
        "request": session.human_input_request,
    }
```

#### 5. Update ReasoningAgent Dependency Injection

**What:** Inject session_manager into ReasoningAgent.

**Why:** Agent needs session manager to coordinate human input.

```python
# In api/dependencies.py
async def get_reasoning_agent(
    container: ServiceContainer = Depends(get_service_container)
) -> ReasoningAgent:
    """Create reasoning agent with session manager."""
    return ReasoningAgent(
        http_client=container.http_client,
        openai_client=container.openai_client,
        mcp_client=container.mcp_client,
        session_manager=container.session_manager,  # NEW
        config=container.config,
    )
```

### Testing Strategy

#### Unit Tests (tests/test_human_in_loop.py)

**Mock-based tests for wait/notify coordination:**
```python
@pytest.mark.asyncio
async def test_request_and_provide_human_input():
    """Test basic human input request/response flow."""
    mock_redis = AsyncMock()
    manager = ReasoningSessionManager(mock_redis)

    # Create session
    session = await manager.create_session("test-session")

    # Request input
    request = TextInputRequest(prompt="Enter your name", timeout_seconds=10)
    request_id = await manager.request_human_input("test-session", request)

    # Simulate human providing input (in parallel)
    async def provide_input():
        await asyncio.sleep(0.1)
        response = HumanInputResponse(
            request_id=request_id,
            response_type="text",
            value="Alice",
            timestamp=time.time()
        )
        await manager.provide_human_input(request_id, response)

    # Wait for input
    input_task = asyncio.create_task(provide_input())
    response = await manager.wait_for_human_input(request_id, timeout=5)

    assert response is not None
    assert response.value == "Alice"
    await input_task

@pytest.mark.asyncio
async def test_human_input_timeout():
    """Test timeout when no input provided."""
    mock_redis = AsyncMock()
    manager = ReasoningSessionManager(mock_redis)

    session = await manager.create_session("test-session")
    request = TextInputRequest(prompt="Enter something", timeout_seconds=1)
    request_id = await manager.request_human_input("test-session", request)

    # Wait without providing input
    response = await manager.wait_for_human_input(request_id, timeout=0.5)

    assert response is None  # Timeout

@pytest.mark.asyncio
async def test_reasoning_with_human_confirmation():
    """Test reasoning pauses for confirmation and resumes."""
    # Mock reasoning agent that requests confirmation
    agent = ReasoningAgent(...)

    # Mock LLM to return REQUEST_HUMAN_INPUT action
    mock_step = ReasoningStep(
        thought="This action will delete data. I should confirm with user.",
        next_action=ReasoningAction.REQUEST_HUMAN_INPUT,
        human_input_request=ConfirmationRequest(
            prompt="Delete all data?",
            action_description="Permanently delete database",
            timeout_seconds=10
        )
    )

    # ... test that agent pauses, waits, resumes after confirmation
```

#### Integration Tests (tests/integration/test_human_in_loop_api.py)

**End-to-end human input workflow:**
```python
@pytest.mark.integration
async def test_human_input_workflow_end_to_end():
    """Test complete human input workflow via API."""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        session_id = str(uuid.uuid4())

        # Start reasoning that will request human input
        # (requires prompt engineering LLM to request input)
        request = {
            "model": "gpt-4o-mini",
            "messages": [{
                "role": "user",
                "content": "You need to ask me for confirmation before proceeding."
            }],
            "stream": True
        }

        # Start streaming
        response = client.stream(
            "POST",
            "/v1/chat/completions",
            json=request,
            headers={"X-Session-ID": session_id}
        )

        # Wait for HUMAN_INPUT_REQUIRED event
        request_id = None
        async with response as stream:
            async for line in stream.aiter_lines():
                if "HUMAN_INPUT_REQUIRED" in line:
                    # Extract request_id from event
                    request_id = ...  # parse from SSE data
                    break

        assert request_id is not None

        # Provide input via separate endpoint
        input_response = await client.post(
            "/v1/reasoning/human-input",
            json={
                "request_id": request_id,
                "response_type": "confirmation",
                "value": True,
                "timestamp": time.time()
            }
        )

        assert input_response.status_code == 200

        # Continue consuming stream - should resume reasoning
        # ... verify completion
```

#### Edge Cases to Test

- **Multiple pending requests**: Only one input request per session at a time
- **Expired request**: Providing input to expired/completed request
- **Invalid request_id**: Non-existent or malformed IDs
- **Concurrent input**: Multiple clients providing input simultaneously
- **Session cleanup during wait**: Session deleted while waiting for input
- **Timeout recovery**: What happens after timeout - retry or fail?

### Implementation Checklist

- [ ] Read all prerequisite documentation
- [ ] Extend api/reasoning_models.py with human input types
- [ ] Add human input methods to api/session_manager.py
- [ ] Implement asyncio.Event coordination
- [ ] Update api/reasoning_agent.py to handle REQUEST_HUMAN_INPUT
- [ ] Add human input endpoint to api/main.py
- [ ] Add pending input query endpoint
- [ ] Update ReasoningAgent dependency injection
- [ ] Write unit tests for wait/notify coordination
- [ ] Write unit tests for timeout scenarios
- [ ] Write integration tests for end-to-end workflow
- [ ] Test edge cases (concurrent, expired, invalid)
- [ ] Update SSE event types documentation
- [ ] Add client examples for human input
- [ ] Update API documentation

### Risk Factors

**AsyncIO Coordination Complexity:**
- Risk: Deadlocks or race conditions in wait/notify
- Mitigation: Thorough testing, use asyncio primitives correctly

**Timeout Handling:**
- Risk: Resources not cleaned up after timeout
- Mitigation: Ensure events deleted, sessions updated properly

**State Consistency:**
- Risk: Session state vs in-memory event state mismatch
- Mitigation: Atomic updates, clear state transitions

**Memory Leaks:**
- Risk: Events not cleaned up for abandoned requests
- Mitigation: Cleanup in timeout path, periodic scan for orphaned events

### Questions to Resolve Before Implementation

1. What should happen after timeout - retry request or fail reasoning entirely?
2. Should there be a max number of human input requests per session?
3. How to handle malicious clients requesting input repeatedly?
4. Should input requests persist across server restarts? (events are in-memory)
5. What's the UX for multiple simultaneous input requests in different tabs?

### Documentation Updates

- Update README.md with human-in-the-loop examples
- Document SSE event types for human input
- Add client code examples (JavaScript, Python)
- Document timeout behavior and recovery
- Add troubleshooting section for input coordination issues
- Update API reference with new endpoints

---

**STOP HERE - Complete Milestone 3 fully (implementation + tests + docs) before proceeding to Milestone 4.**

---

## Milestone 4: MCP-Based Agent Registry and Discovery

### Goal
Build dynamic agent registry system using MCP (Model Context Protocol) for discovering available agents, their capabilities, and endpoints. This enables the orchestrator to know what agents exist and how to communicate with them.

**New Capability:** Agents can be discovered dynamically via MCP protocol. Registry tracks agent capabilities, health status, and provides routing information.

### Why This Next
- Foundation for multi-agent orchestration (Milestone 5)
- Establishes standard agent communication protocol
- Enables dynamic agent addition/removal without code changes
- Proves MCP integration works at system level
- Low risk - doesn't affect existing single-agent flows

### Success Criteria
- [ ] Agent registry stores agent metadata and capabilities
- [ ] MCP protocol used for agent discovery and communication
- [ ] Agents self-register on startup
- [ ] Health checks verify agent availability
- [ ] Registry provides agent lookup by capability
- [ ] Registry stored in Redis for persistence
- [ ] API endpoints to query available agents
- [ ] Proper handling of agent registration/deregistration

### Prerequisites and Learning

**Required Reading:**
- [Model Context Protocol Specification](https://modelcontextprotocol.io/introduction) - MCP overview
- [MCP Server Implementation](https://modelcontextprotocol.io/docs/concepts/servers) - Building MCP servers
- [MCP Tool Discovery](https://modelcontextprotocol.io/docs/concepts/tools) - Tool registration and discovery
- Existing `api/mcp.py` - Current MCP client implementation

**Key Concepts:**
- MCP servers expose tools and capabilities
- MCP clients discover and invoke tools
- Agents are MCP servers with specific tool sets
- Registry maps capabilities → agent endpoints
- Health checks ensure agent availability

### Key Changes

#### 1. Create `api/agent_registry.py`

**What:** Core agent registry module managing agent metadata and capabilities.

**Why:** Centralized registry for all agents in the system.

**Key interfaces:**
```python
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Optional
import redis.asyncio as redis
from enum import Enum

class AgentType(str, Enum):
    """Types of agents in the system."""
    REASONING = "reasoning"
    SEARCH = "search"
    PLANNING = "planning"
    CODE = "code"
    DATA_ANALYSIS = "data_analysis"
    CUSTOM = "custom"

@dataclass
class AgentCapability:
    """A capability an agent can perform."""
    name: str  # e.g., "web_search", "code_generation", "reasoning"
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentMetadata:
    """Agent registration metadata."""
    agent_id: str
    agent_type: AgentType
    name: str
    description: str
    capabilities: List[AgentCapability]
    mcp_endpoint: str  # URL for MCP communication
    system_prompt: str  # Agent's specific prompt
    model: str = "gpt-4o-mini"  # Default model
    tools: List[str] = field(default_factory=list)  # MCP tool IDs
    health_check_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: float = 0.0
    last_health_check: Optional[float] = None
    is_healthy: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'AgentMetadata':
        # Convert capabilities back to AgentCapability objects
        if 'capabilities' in data:
            data['capabilities'] = [
                AgentCapability(**cap) if isinstance(cap, dict) else cap
                for cap in data['capabilities']
            ]
        return cls(**data)

class AgentRegistry:
    """Registry for discovering and managing agents."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self._registry_key_prefix = "agent_registry:"
        self._capability_index_prefix = "capability_index:"

    async def register_agent(self, agent: AgentMetadata) -> bool:
        """Register new agent or update existing."""
        import time
        agent.registered_at = time.time()
        agent.last_health_check = time.time()

        # Store agent metadata
        key = f"{self._registry_key_prefix}{agent.agent_id}"
        await self.redis.setex(
            key,
            86400,  # 24 hour TTL
            json.dumps(agent.to_dict())
        )

        # Index by capabilities
        for capability in agent.capabilities:
            cap_key = f"{self._capability_index_prefix}{capability.name}"
            await self.redis.sadd(cap_key, agent.agent_id)

        return True

    async def deregister_agent(self, agent_id: str) -> bool:
        """Remove agent from registry."""
        # Get agent to cleanup capability indexes
        agent = await self.get_agent(agent_id)
        if not agent:
            return False

        # Remove from capability indexes
        for capability in agent.capabilities:
            cap_key = f"{self._capability_index_prefix}{capability.name}"
            await self.redis.srem(cap_key, agent_id)

        # Remove agent metadata
        key = f"{self._registry_key_prefix}{agent_id}"
        await self.redis.delete(key)

        return True

    async def get_agent(self, agent_id: str) -> Optional[AgentMetadata]:
        """Get agent metadata by ID."""
        key = f"{self._registry_key_prefix}{agent_id}"
        data = await self.redis.get(key)
        if not data:
            return None

        return AgentMetadata.from_dict(json.loads(data))

    async def list_agents(
        self,
        agent_type: Optional[AgentType] = None,
        healthy_only: bool = True
    ) -> List[AgentMetadata]:
        """List all registered agents."""
        pattern = f"{self._registry_key_prefix}*"
        agents = []

        async for key in self.redis.scan_iter(pattern):
            data = await self.redis.get(key)
            if data:
                agent = AgentMetadata.from_dict(json.loads(data))

                # Filter by type
                if agent_type and agent.agent_type != agent_type:
                    continue

                # Filter by health
                if healthy_only and not agent.is_healthy:
                    continue

                agents.append(agent)

        return agents

    async def find_agents_by_capability(
        self,
        capability: str,
        healthy_only: bool = True
    ) -> List[AgentMetadata]:
        """Find agents that have specific capability."""
        cap_key = f"{self._capability_index_prefix}{capability}"
        agent_ids = await self.redis.smembers(cap_key)

        agents = []
        for agent_id in agent_ids:
            agent = await self.get_agent(agent_id)
            if agent:
                if healthy_only and not agent.is_healthy:
                    continue
                agents.append(agent)

        return agents

    async def update_health_status(
        self,
        agent_id: str,
        is_healthy: bool
    ) -> bool:
        """Update agent health status."""
        agent = await self.get_agent(agent_id)
        if not agent:
            return False

        import time
        agent.is_healthy = is_healthy
        agent.last_health_check = time.time()

        await self.register_agent(agent)  # Re-register with updated health
        return True
```

**Implementation Details:**
- Store agents in Redis with TTL for automatic cleanup
- Maintain capability index (Redis sets) for fast lookups
- Agents must re-register periodically (heartbeat pattern)
- Health checks update `is_healthy` and `last_health_check`

#### 2. Create Agent MCP Server Pattern

**What:** Template for creating MCP-compatible agent services.

**Why:** Standardize how agents expose their capabilities via MCP.

**Pattern (example for planning agent):**
```python
# services/planning_agent/main.py
from fastapi import FastAPI
from api.mcp import MCPServer, Tool, ToolParameter
from api.agent_registry import AgentMetadata, AgentCapability, AgentType
import httpx

app = FastAPI()
mcp_server = MCPServer()

# Define agent's system prompt
PLANNING_AGENT_PROMPT = """
You are a planning agent that analyzes user requests and creates execution plans.
Your job is to:
1. Understand the user's goal
2. Identify required capabilities
3. Create a step-by-step execution plan
4. Determine which agents should handle each step
"""

# Define MCP tools this agent provides
@mcp_server.tool(
    name="create_execution_plan",
    description="Analyze request and create execution plan for multi-agent workflow"
)
async def create_execution_plan(
    user_request: str,
    available_agents: List[dict]
) -> dict:
    """Create execution plan using LLM."""
    # Call LLM with planning prompt + user request + available agents
    # Return structured plan
    pass

@mcp_server.tool(
    name="validate_plan",
    description="Validate execution plan for correctness and feasibility"
)
async def validate_plan(plan: dict) -> dict:
    """Validate plan structure and agent availability."""
    pass

# MCP endpoint
@app.get("/mcp/tools")
async def list_tools():
    """MCP tool discovery endpoint."""
    return mcp_server.list_tools()

@app.post("/mcp/execute/{tool_name}")
async def execute_tool(tool_name: str, parameters: dict):
    """MCP tool execution endpoint."""
    return await mcp_server.execute_tool(tool_name, parameters)

# Health check
@app.get("/health")
async def health():
    return {"status": "healthy"}

# Self-registration on startup
@app.on_event("startup")
async def register_with_registry():
    """Register this agent with central registry."""
    agent_metadata = AgentMetadata(
        agent_id="planning-agent-001",
        agent_type=AgentType.PLANNING,
        name="Planning Agent",
        description="Analyzes requests and creates multi-agent execution plans",
        capabilities=[
            AgentCapability(
                name="planning",
                description="Create execution plans for complex tasks"
            ),
            AgentCapability(
                name="plan_validation",
                description="Validate execution plans"
            )
        ],
        mcp_endpoint="http://planning-agent:8002/mcp",
        system_prompt=PLANNING_AGENT_PROMPT,
        tools=["create_execution_plan", "validate_plan"],
        health_check_url="http://planning-agent:8002/health"
    )

    # Register with central registry
    async with httpx.AsyncClient() as client:
        await client.post(
            "http://api:8000/v1/agents/register",
            json=agent_metadata.to_dict()
        )
```

#### 3. Update `api/dependencies.py`

**What:** Add agent registry to ServiceContainer.

**Why:** Make registry available via dependency injection.

```python
from api.agent_registry import AgentRegistry

class ServiceContainer:
    def __init__(self):
        # ... existing fields
        self.agent_registry: Optional[AgentRegistry] = None

    async def initialize(self):
        """Initialize all services."""
        # ... existing initialization

        # Initialize agent registry (requires Redis)
        self.agent_registry = AgentRegistry(self.redis_client)

async def get_agent_registry(
    container: ServiceContainer = Depends(get_service_container)
) -> AgentRegistry:
    return container.agent_registry
```

#### 4. Add Agent Registry Endpoints to `api/main.py`

**What:** API endpoints for agent registration and discovery.

**Why:** Allow agents to self-register and clients to discover agents.

```python
from api.agent_registry import AgentRegistry, AgentMetadata, AgentType

@app.post("/v1/agents/register")
async def register_agent(
    agent: AgentMetadata,
    registry: AgentRegistry = Depends(get_agent_registry),
):
    """Register or update agent in registry."""
    success = await registry.register_agent(agent)

    return {
        "agent_id": agent.agent_id,
        "status": "registered" if success else "failed",
        "success": success
    }

@app.delete("/v1/agents/{agent_id}")
async def deregister_agent(
    agent_id: str,
    registry: AgentRegistry = Depends(get_agent_registry),
):
    """Deregister agent from registry."""
    success = await registry.deregister_agent(agent_id)

    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")

    return {"agent_id": agent_id, "status": "deregistered"}

@app.get("/v1/agents")
async def list_agents(
    agent_type: Optional[AgentType] = None,
    healthy_only: bool = True,
    registry: AgentRegistry = Depends(get_agent_registry),
):
    """List all registered agents."""
    agents = await registry.list_agents(
        agent_type=agent_type,
        healthy_only=healthy_only
    )

    return {
        "agents": [agent.to_dict() for agent in agents],
        "count": len(agents)
    }

@app.get("/v1/agents/{agent_id}")
async def get_agent(
    agent_id: str,
    registry: AgentRegistry = Depends(get_agent_registry),
):
    """Get specific agent metadata."""
    agent = await registry.get_agent(agent_id)

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    return agent.to_dict()

@app.get("/v1/agents/capability/{capability}")
async def find_agents_by_capability(
    capability: str,
    healthy_only: bool = True,
    registry: AgentRegistry = Depends(get_agent_registry),
):
    """Find agents by capability."""
    agents = await registry.find_agents_by_capability(
        capability=capability,
        healthy_only=healthy_only
    )

    return {
        "capability": capability,
        "agents": [agent.to_dict() for agent in agents],
        "count": len(agents)
    }

@app.post("/v1/agents/{agent_id}/health")
async def update_agent_health(
    agent_id: str,
    health_status: dict,  # {"is_healthy": true}
    registry: AgentRegistry = Depends(get_agent_registry),
):
    """Update agent health status."""
    success = await registry.update_health_status(
        agent_id=agent_id,
        is_healthy=health_status.get("is_healthy", True)
    )

    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")

    return {"agent_id": agent_id, "health_updated": True}
```

#### 5. Extend `api/mcp.py` for Agent-to-Agent Communication

**What:** Add MCP client methods for calling other agents as tools.

**Why:** Orchestrator needs to invoke agents via MCP protocol.

**Pattern:**
```python
class MCPAgentClient:
    """MCP client for communicating with other agents."""

    def __init__(self, http_client: httpx.AsyncClient):
        self.http_client = http_client

    async def list_agent_tools(self, agent_endpoint: str) -> List[dict]:
        """Discover tools provided by agent."""
        response = await self.http_client.get(f"{agent_endpoint}/mcp/tools")
        response.raise_for_status()
        return response.json()

    async def execute_agent_tool(
        self,
        agent_endpoint: str,
        tool_name: str,
        parameters: dict
    ) -> dict:
        """Execute tool on remote agent."""
        response = await self.http_client.post(
            f"{agent_endpoint}/mcp/execute/{tool_name}",
            json=parameters
        )
        response.raise_for_status()
        return response.json()

    async def check_agent_health(self, health_url: str) -> bool:
        """Check if agent is healthy."""
        try:
            response = await self.http_client.get(health_url, timeout=5.0)
            return response.status_code == 200
        except:
            return False
```

### Testing Strategy

#### Unit Tests (tests/test_agent_registry.py)

**Registry operations:**
```python
@pytest.mark.asyncio
async def test_register_and_get_agent():
    """Test agent registration and retrieval."""
    mock_redis = AsyncMock()
    registry = AgentRegistry(mock_redis)

    agent = AgentMetadata(
        agent_id="test-agent",
        agent_type=AgentType.REASONING,
        name="Test Agent",
        description="Test",
        capabilities=[AgentCapability(name="reasoning", description="Test")],
        mcp_endpoint="http://test:8000/mcp",
        system_prompt="Test prompt"
    )

    success = await registry.register_agent(agent)
    assert success

@pytest.mark.asyncio
async def test_find_agents_by_capability():
    """Test capability-based agent discovery."""
    mock_redis = AsyncMock()
    # Mock Redis to return agent IDs for capability
    mock_redis.smembers.return_value = {"agent-1", "agent-2"}

    registry = AgentRegistry(mock_redis)
    agents = await registry.find_agents_by_capability("web_search")

    assert len(agents) == 2
```

#### Integration Tests (tests/integration/test_mcp_agents.py)

**End-to-end agent discovery and invocation:**
```python
@pytest.mark.integration
async def test_agent_self_registration():
    """Test agent registers itself on startup."""
    # Start a test agent service
    # Verify it appears in registry
    # Verify its MCP tools are discoverable

    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        # Wait for agent to register
        await asyncio.sleep(1)

        # List agents
        response = await client.get("/v1/agents")
        agents = response.json()["agents"]

        assert len(agents) > 0
        assert any(a["agent_id"] == "test-agent" for a in agents)

@pytest.mark.integration
async def test_mcp_tool_invocation():
    """Test invoking agent tool via MCP."""
    async with httpx.AsyncClient() as client:
        # Discover tools
        tools_response = await client.get("http://planning-agent:8002/mcp/tools")
        tools = tools_response.json()

        assert "create_execution_plan" in [t["name"] for t in tools]

        # Execute tool
        exec_response = await client.post(
            "http://planning-agent:8002/mcp/execute/create_execution_plan",
            json={"user_request": "Search the web and summarize findings"}
        )

        plan = exec_response.json()
        assert "steps" in plan
```

#### Edge Cases to Test

- **Duplicate agent IDs**: Same agent registers twice
- **Stale agents**: Agents that don't heartbeat/re-register
- **Unhealthy agents**: Health check failures
- **Capability overlap**: Multiple agents with same capability
- **Invalid MCP endpoints**: Agent with bad/unreachable endpoint

### Implementation Checklist

- [ ] Read all prerequisite documentation (MCP spec, existing mcp.py)
- [ ] Create api/agent_registry.py with core classes
- [ ] Implement AgentRegistry methods (register, get, list, find)
- [ ] Add agent registry to api/dependencies.py
- [ ] Add registry endpoints to api/main.py
- [ ] Create example agent service (planning agent)
- [ ] Implement agent self-registration pattern
- [ ] Extend api/mcp.py with MCPAgentClient
- [ ] Write unit tests for registry operations
- [ ] Write integration tests for agent discovery
- [ ] Test MCP tool invocation
- [ ] Test health checks
- [ ] Update docker-compose with example agent service
- [ ] Document agent creation pattern
- [ ] Update API documentation

### Docker Compose Updates

Add example agent services:

```yaml
services:
  # Planning agent example
  planning-agent:
    build:
      context: .
      dockerfile: Dockerfile.agent
    environment:
      - AGENT_ID=planning-agent-001
      - AGENT_TYPE=planning
      - REGISTRY_URL=http://api:8000
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    ports:
      - "8002:8002"
    depends_on:
      - redis
      - api
```

### Risk Factors

**Agent Discovery Reliability:**
- Risk: Agents not discoverable due to registration failures
- Mitigation: Health checks, retry logic, monitoring

**MCP Protocol Compatibility:**
- Risk: Agents implement MCP incorrectly
- Mitigation: Standard templates, validation, integration tests

**Registry Consistency:**
- Risk: Registry state out of sync with actual agents
- Mitigation: TTL-based expiration, health checks, periodic cleanup

**Network Failures:**
- Risk: Agent-to-agent communication failures
- Mitigation: Retries, circuit breakers, fallbacks

### Questions to Resolve Before Implementation

1. Should agents have version numbers for capability evolution?
2. How to handle agent capacity/load balancing when multiple agents have same capability?
3. Should registry track agent performance metrics (latency, success rate)?
4. What's the agent re-registration interval (heartbeat)?
5. Should there be agent authentication/authorization for registration?

### Documentation Updates

- Create "Agent Development Guide" documenting how to create MCP agents
- Document agent registration flow and lifecycle
- Add MCP protocol usage examples
- Document capability naming conventions
- Update architecture diagrams with agent registry
- Add troubleshooting guide for agent discovery issues

---

**STOP HERE - Complete Milestone 4 fully (implementation + tests + docs) before proceeding to Milestone 5.**

---

## Milestone 5: Custom Orchestrator Service (No LangGraph)

### Goal
Build custom orchestrator service that coordinates multiple agents using simple Python async patterns. The orchestrator analyzes requests, selects appropriate agents from the registry, executes them (sequential or parallel), and aggregates results. No heavy framework abstractions - just FastAPI, asyncio, and direct MCP calls.

**New Capability:** Multi-agent coordination with custom logic. Orchestrator can invoke multiple agents, handle dependencies, and stream aggregated results.

###Why This Next
- Core multi-agent capability
- Builds on agent registry (Milestone 4)
- Demonstrates value of modular agent system
- Simple, understandable orchestration logic
- Foundation for complex workflows

### Success Criteria
- [ ] Orchestrator analyzes requests and selects agents
- [ ] Supports sequential agent execution
- [ ] Supports parallel agent execution
- [ ] Aggregates results from multiple agents
- [ ] Streams progress via SSE
- [ ] Handles agent failures gracefully
- [ ] Integrates with session management
- [ ] Uses agent registry for dynamic discovery

### Prerequisites and Learning

**Required Reading:**
- [AsyncIO gather for parallel execution](https://docs.python.org/3/library/asyncio-task.html#asyncio.gather) - Running tasks concurrently
- [AsyncIO TaskGroup](https://docs.python.org/3/library/asyncio-task.html#task-groups) - Managing task groups
- Your `notes/orchestrator-notes.md` - Architecture patterns

**Key Concepts:**
- Simple coordination patterns (no complex frameworks)
- Agent = LLM + prompt + tools (MCP)
- Orchestrator decides: which agents, what order, sequential vs parallel
- Results aggregated and returned to user

### Key Changes

#### 1. Create `api/orchestrator.py`

**What:** Core orchestration logic using simple async patterns.

**Why:** Coordinate multiple agents without heavy abstractions.

**Key interfaces:**
```python
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
import asyncio
from api.agent_registry import AgentRegistry, AgentMetadata
from api.mcp import MCPAgentClient

@dataclass
class OrchestrationStep:
    """Single step in orchestration."""
    step_id: str
    agent_id: str
    tool_name: str
    parameters: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)  # Step IDs
    parallel: bool = True  # Can run in parallel with other steps

@dataclass
class OrchestrationPlan:
    """Plan for executing multi-agent workflow."""
    plan_id: str
    steps: List[OrchestrationStep]
    estimated_duration: Optional[float] = None

@dataclass
class StepResult:
    """Result from executing a step."""
    step_id: str
    agent_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    duration: float = 0.0

class SimpleOrchestrator:
    """
    Simple orchestrator - no LangGraph, no complex abstractions.
    Just: analyze request → select agents → execute → aggregate.
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        mcp_client: MCPAgentClient,
        http_client: httpx.AsyncClient,
    ):
        self.registry = agent_registry
        self.mcp_client = mcp_client
        self.http_client = http_client

    async def analyze_request(
        self,
        user_request: str
    ) -> OrchestrationPlan:
        """
        Analyze user request and create execution plan.
        Simple heuristic-based planning (can be enhanced with LLM later).
        """
        # Get available agents
        agents = await self.registry.list_agents(healthy_only=True)

        # Simple analysis: if request contains certain keywords, use certain agents
        plan = OrchestrationPlan(
            plan_id=str(uuid.uuid4()),
            steps=[]
        )

        # Example: If "search" in request, use search agent
        if "search" in user_request.lower():
            search_agents = await self.registry.find_agents_by_capability("web_search")
            if search_agents:
                plan.steps.append(
                    OrchestrationStep(
                        step_id="step-1",
                        agent_id=search_agents[0].agent_id,
                        tool_name="web_search",
                        parameters={"query": user_request},
                        parallel=True
                    )
                )

        # Always use reasoning agent for synthesis
        reasoning_agents = await self.registry.find_agents_by_capability("reasoning")
        if reasoning_agents:
            plan.steps.append(
                OrchestrationStep(
                    step_id="step-2",
                    agent_id=reasoning_agents[0].agent_id,
                    tool_name="reason",
                    parameters={"context": user_request},
                    depends_on=["step-1"] if len(plan.steps) > 0 else [],
                    parallel=False
                )
            )

        return plan

    async def execute_plan(
        self,
        plan: OrchestrationPlan,
        session_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute orchestration plan, streaming progress.
        """
        completed_steps: Dict[str, StepResult] = {}

        # Yield plan start event
        yield {
            "type": "plan_started",
            "plan_id": plan.plan_id,
            "total_steps": len(plan.steps)
        }

        # Execute steps respecting dependencies
        remaining_steps = plan.steps.copy()

        while remaining_steps:
            # Find steps ready to run (dependencies met)
            ready_steps = [
                step for step in remaining_steps
                if all(dep_id in completed_steps for dep_id in step.depends_on)
            ]

            if not ready_steps:
                # No steps ready - circular dependency or error
                yield {
                    "type": "error",
                    "message": "No steps ready to execute - circular dependency?"
                }
                break

            # Group by parallel vs sequential
            parallel_steps = [s for s in ready_steps if s.parallel]
            sequential_steps = [s for s in ready_steps if not s.parallel]

            # Execute parallel steps concurrently
            if parallel_steps:
                yield {
                    "type": "parallel_execution_started",
                    "steps": [s.step_id for s in parallel_steps]
                }

                # Execute all parallel steps with gather
                tasks = [
                    self._execute_step(step, completed_steps)
                    for step in parallel_steps
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for step, result in zip(parallel_steps, results):
                    if isinstance(result, Exception):
                        step_result = StepResult(
                            step_id=step.step_id,
                            agent_id=step.agent_id,
                            success=False,
                            result=None,
                            error=str(result)
                        )
                    else:
                        step_result = result

                    completed_steps[step.step_id] = step_result
                    remaining_steps.remove(step)

                    # Yield step completion
                    yield {
                        "type": "step_completed",
                        "step_id": step.step_id,
                        "success": step_result.success,
                        "result": step_result.result
                    }

            # Execute sequential steps one by one
            for step in sequential_steps:
                yield {
                    "type": "step_started",
                    "step_id": step.step_id,
                    "agent_id": step.agent_id
                }

                step_result = await self._execute_step(step, completed_steps)
                completed_steps[step.step_id] = step_result
                remaining_steps.remove(step)

                yield {
                    "type": "step_completed",
                    "step_id": step.step_id,
                    "success": step_result.success,
                    "result": step_result.result
                }

        # Aggregate final results
        final_result = self._aggregate_results(completed_steps)

        yield {
            "type": "plan_completed",
            "plan_id": plan.plan_id,
            "result": final_result
        }

    async def _execute_step(
        self,
        step: OrchestrationStep,
        completed_steps: Dict[str, StepResult]
    ) -> StepResult:
        """Execute single orchestration step."""
        import time
        start_time = time.time()

        try:
            # Get agent metadata
            agent = await self.registry.get_agent(step.agent_id)
            if not agent:
                raise ValueError(f"Agent {step.agent_id} not found")

            # Inject dependencies into parameters
            parameters = step.parameters.copy()
            for dep_id in step.depends_on:
                if dep_id in completed_steps:
                    parameters[f"dep_{dep_id}"] = completed_steps[dep_id].result

            # Execute via MCP
            result = await self.mcp_client.execute_agent_tool(
                agent_endpoint=agent.mcp_endpoint,
                tool_name=step.tool_name,
                parameters=parameters
            )

            return StepResult(
                step_id=step.step_id,
                agent_id=step.agent_id,
                success=True,
                result=result,
                duration=time.time() - start_time
            )

        except Exception as e:
            return StepResult(
                step_id=step.step_id,
                agent_id=step.agent_id,
                success=False,
                result=None,
                error=str(e),
                duration=time.time() - start_time
            )

    def _aggregate_results(
        self,
        completed_steps: Dict[str, StepResult]
    ) -> Dict[str, Any]:
        """Aggregate results from all steps."""
        # Simple aggregation - collect all successful results
        successful_results = {
            step_id: result.result
            for step_id, result in completed_steps.items()
            if result.success
        }

        failed_steps = [
            {"step_id": step_id, "error": result.error}
            for step_id, result in completed_steps.items()
            if not result.success
        ]

        return {
            "successful_steps": len(successful_results),
            "failed_steps": len(failed_steps),
            "results": successful_results,
            "errors": failed_steps if failed_steps else None
        }
```

#### 2. Add Orchestration Endpoint to `api/main.py`

**What:** Endpoint that triggers orchestrated multi-agent execution.

**Why:** Entry point for users to leverage multi-agent capabilities.

```python
from api.orchestrator import SimpleOrchestrator

@app.post("/v1/orchestrate")
async def orchestrate_request(
    request: dict,  # {"user_request": "...", "stream": true}
    http_request: Request,
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator),
    session_manager: ReasoningSessionManager = Depends(get_session_manager),
):
    """
    Orchestrate multi-agent workflow.
    Analyzes request, selects agents, executes, aggregates.
    """
    user_request = request.get("user_request")
    stream = request.get("stream", True)

    # Create or get session
    session_id = http_request.headers.get("X-Session-ID") or str(uuid.uuid4())
    await session_manager.create_session(session_id, metadata={"type": "orchestration"})

    # Analyze request and create plan
    plan = await orchestrator.analyze_request(user_request)

    if stream:
        # Stream execution progress
        async def stream_orchestration():
            try:
                async for event in orchestrator.execute_plan(plan, session_id):
                    yield f"data: {json.dumps(event)}\n\n"

                await session_manager.set_status(session_id, SessionStatus.COMPLETED)

            except Exception as e:
                await session_manager.set_status(session_id, SessionStatus.FAILED)
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        response = StreamingResponse(
            stream_orchestration(),
            media_type="text/event-stream"
        )
        response.headers["X-Session-ID"] = session_id
        return response
    else:
        # Non-streaming: execute and return final result
        results = []
        async for event in orchestrator.execute_plan(plan, session_id):
            results.append(event)

        final_event = next((e for e in reversed(results) if e["type"] == "plan_completed"), None)

        await session_manager.set_status(session_id, SessionStatus.COMPLETED)

        return {
            "session_id": session_id,
            "plan_id": plan.plan_id,
            "result": final_event["result"] if final_event else None,
            "events": results
        }

# Dependency
async def get_orchestrator(
    container: ServiceContainer = Depends(get_service_container)
) -> SimpleOrchestrator:
    return SimpleOrchestrator(
        agent_registry=container.agent_registry,
        mcp_client=container.mcp_agent_client,
        http_client=container.http_client
    )
```

#### 3. Update `api/dependencies.py`

**What:** Add orchestrator and MCP agent client to ServiceContainer.

```python
from api.orchestrator import SimpleOrchestrator
from api.mcp import MCPAgentClient

class ServiceContainer:
    def __init__(self):
        # ... existing fields
        self.mcp_agent_client: Optional[MCPAgentClient] = None
        self.orchestrator: Optional[SimpleOrchestrator] = None

    async def initialize(self):
        """Initialize all services."""
        # ... existing initialization

        # Initialize MCP agent client
        self.mcp_agent_client = MCPAgentClient(self.http_client)

        # Initialize orchestrator
        self.orchestrator = SimpleOrchestrator(
            agent_registry=self.agent_registry,
            mcp_client=self.mcp_agent_client,
            http_client=self.http_client
        )
```

#### 4. Enhance Analysis with LLM (Optional Enhancement)

**What:** Use LLM to analyze request and generate smarter plans.

**Why:** Better than keyword matching for complex requests.

```python
async def analyze_request_with_llm(
    self,
    user_request: str
) -> OrchestrationPlan:
    """
    Use LLM to analyze request and create execution plan.
    LLM decides which agents to use based on request and available agents.
    """
    # Get available agents
    agents = await self.registry.list_agents(healthy_only=True)

    # Create prompt for LLM
    agents_description = "\n".join([
        f"- {agent.name}: {agent.description} (capabilities: {[c.name for c in agent.capabilities]})"
        for agent in agents
    ])

    planning_prompt = f"""
    You are an orchestration planner. Analyze the user's request and create an execution plan.

    Available agents:
    {agents_description}

    User request: {user_request}

    Create a plan with steps. Each step specifies:
    - Which agent to use
    - What tool to call
    - What parameters to pass
    - Dependencies (which steps must complete first)
    - Whether it can run in parallel

    Respond with JSON following this schema:
    {{
        "steps": [
            {{
                "step_id": "step-1",
                "agent_id": "agent-name",
                "tool_name": "tool_name",
                "parameters": {{}},
                "depends_on": [],
                "parallel": true
            }}
        ]
    }}
    """

    # Call LLM to generate plan
    response = await self.http_client.post(
        "https://api.openai.com/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": planning_prompt}],
            "response_format": {"type": "json_object"}
        }
    )

    plan_json = response.json()["choices"][0]["message"]["content"]
    plan_data = json.loads(plan_json)

    # Convert to OrchestrationPlan
    steps = [OrchestrationStep(**step_data) for step_data in plan_data["steps"]]

    return OrchestrationPlan(
        plan_id=str(uuid.uuid4()),
        steps=steps
    )
```

### Testing Strategy

#### Unit Tests (tests/test_orchestrator.py)

**Orchestration logic tests:**
```python
@pytest.mark.asyncio
async def test_parallel_step_execution():
    """Test parallel steps execute concurrently."""
    mock_registry = AsyncMock()
    mock_mcp = AsyncMock()

    orchestrator = SimpleOrchestrator(mock_registry, mock_mcp, AsyncMock())

    plan = OrchestrationPlan(
        plan_id="test-plan",
        steps=[
            OrchestrationStep(
                step_id="step-1",
                agent_id="agent-1",
                tool_name="tool-1",
                parameters={},
                parallel=True
            ),
            OrchestrationStep(
                step_id="step-2",
                agent_id="agent-2",
                tool_name="tool-2",
                parameters={},
                parallel=True
            )
        ]
    )

    # Execute and collect events
    events = []
    async for event in orchestrator.execute_plan(plan, "session-123"):
        events.append(event)

    # Verify parallel execution started
    assert any(e["type"] == "parallel_execution_started" for e in events)

@pytest.mark.asyncio
async def test_sequential_dependency_handling():
    """Test steps with dependencies execute in order."""
    orchestrator = ...

    plan = OrchestrationPlan(
        plan_id="test-plan",
        steps=[
            OrchestrationStep(step_id="step-1", ...),
            OrchestrationStep(
                step_id="step-2",
                ...,
                depends_on=["step-1"],  # Must wait for step-1
                parallel=False
            )
        ]
    )

    # Verify step-2 executes after step-1
```

#### Integration Tests (tests/integration/test_orchestration.py)

**End-to-end orchestration:**
```python
@pytest.mark.integration
async def test_multi_agent_orchestration():
    """Test full orchestration with real agents."""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        # Send orchestration request
        response = await client.post(
            "/v1/orchestrate",
            json={
                "user_request": "Search for Python tutorials and summarize",
                "stream": False
            }
        )

        result = response.json()

        assert "plan_id" in result
        assert "result" in result
        assert result["result"]["successful_steps"] > 0
```

### Implementation Checklist

- [ ] Read prerequisite documentation
- [ ] Create api/orchestrator.py with SimpleOrchestrator
- [ ] Implement analyze_request (heuristic-based)
- [ ] Implement execute_plan with streaming
- [ ] Implement parallel execution with asyncio.gather
- [ ] Implement sequential execution
- [ ] Implement dependency resolution
- [ ] Add orchestration endpoint to api/main.py
- [ ] Update api/dependencies.py with orchestrator
- [ ] Write unit tests for orchestration logic
- [ ] Write integration tests with real agents
- [ ] Test parallel vs sequential execution
- [ ] Test error handling
- [ ] Document orchestration API
- [ ] Add examples of orchestrated workflows

### Risk Factors

**Complexity Management:**
- Risk: Orchestration logic becomes too complex
- Mitigation: Keep it simple, avoid over-engineering

**Error Handling:**
- Risk: One agent failure crashes entire workflow
- Mitigation: Graceful degradation, partial results

**Performance:**
- Risk: Sequential execution too slow
- Mitigation: Maximize parallelization where possible

### Questions to Resolve

1. Should failed steps halt entire workflow or continue with partial results?
2. How to handle timeouts for long-running agents?
3. Should there be a max parallelism limit (e.g., max 5 concurrent agents)?
4. How to handle agent rate limits?

### Documentation Updates

- Document orchestration concepts and patterns
- Add examples of simple and complex workflows
- Document plan structure and step dependencies
- Add troubleshooting guide

---

**STOP HERE - Complete Milestone 5 fully before proceeding to Milestone 6.**

---

## Milestone 6: Vector Database for Semantic Memory (Optional Enhancement)

### Goal
Integrate vector database (Chroma or Pinecone) for semantic memory across agents and sessions. Enables agents to remember past interactions, learn from history, and provide context-aware responses.

**New Capability:** Agents can store and retrieve semantic memories. Long-term context persistence beyond individual sessions.

### Why This Milestone
- Enables learning from past interactions
- Improves context-aware responses
- Supports RAG (Retrieval Augmented Generation) patterns
- Valuable for complex multi-turn conversations
- **Optional** - can be added later if needed

### Success Criteria
- [ ] Vector database integrated (Chroma for local, Pinecone for production)
- [ ] Semantic memory storage and retrieval working
- [ ] Memories associated with sessions/users
- [ ] Similarity search functioning correctly
- [ ] Memory cleanup and management
- [ ] Integrated with at least one agent

### Prerequisites and Learning

**Required Reading:**
- [Chroma Documentation](https://docs.trychroma.com/) - Local vector DB
- [Pinecone Documentation](https://docs.pinecone.io/docs/overview) - Production vector DB
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings) - Generating embeddings

**Key Concepts:**
- Vector embeddings for semantic similarity
- Similarity search (cosine similarity, euclidean distance)
- Chunking strategies for long documents
- Metadata filtering

### Key Changes

#### 1. Choose Vector Database

**For Development:** Chroma (local, simple)
**For Production:** Pinecone (scalable, managed)

**Add dependency:**
```toml
# For Chroma
dependencies = ["chromadb>=0.4.0"]

# OR for Pinecone
dependencies = ["pinecone-client>=2.0.0"]
```

#### 2. Create `api/vector_memory.py`

**What:** Semantic memory interface for agents.

```python
from typing import List, Dict, Any, Optional
import chromadb  # or pinecone
from openai import AsyncOpenAI

class SemanticMemory:
    """Semantic memory using vector database."""

    def __init__(self, chroma_client: chromadb.Client, openai_client: AsyncOpenAI):
        self.chroma = chroma_client
        self.openai = openai_client
        self.collection = self.chroma.get_or_create_collection("agent_memory")

    async def store_memory(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Store memory with semantic embedding."""
        # Generate embedding
        embedding_response = await self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=content
        )
        embedding = embedding_response.data[0].embedding

        # Store in vector DB
        memory_id = str(uuid.uuid4())
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata]
        )

        return memory_id

    async def retrieve_similar(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve k most similar memories."""
        # Generate query embedding
        embedding_response = await self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = embedding_response.data[0].embedding

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter_metadata
        )

        # Format results
        memories = []
        for i in range(len(results['ids'][0])):
            memories.append({
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })

        return memories
```

**Implementation Note:** This milestone is optional and can be skipped if semantic memory is not immediately needed. Focus on core orchestration first.

---

**STOP HERE - Milestone 6 is optional. Proceed to Milestone 7 for advanced workflows, or skip to Milestone 8 for production hardening.**

---

## Milestone 7: Prefect DAG Workflows (Optional Advanced)

### Goal
Integrate Prefect for dynamic DAG workflow management. Enables complex workflow patterns with proper dependency management, retries, and observability.

**New Capability:** Complex workflows modeled as DAGs with Prefect's workflow engine.

### Why This Milestone
- **Advanced feature** for complex orchestration
- Better than custom code for complex DAG patterns
- Built-in retry logic, caching, observability
- **Optional** - SimpleOrchestrator (Milestone 5) sufficient for most cases

### Success Criteria
- [ ] Prefect installed and configured
- [ ] Workflows defined as Prefect flows
- [ ] Integration with existing orchestrator
- [ ] Prefect UI accessible for monitoring
- [ ] Dynamic workflow generation working

### Prerequisites and Learning

**Required Reading:**
- [Prefect Documentation](https://docs.prefect.io/)
- [Prefect ControlFlow](https://controlflow.ai/) - AI-specific workflows

### Key Changes

**Add Prefect dependency:**
```toml
dependencies = ["prefect>=2.14.0"]
```

**Example Prefect workflow:**
```python
from prefect import flow, task
import controlflow as cf

@task
async def invoke_agent(agent_id: str, parameters: dict) -> dict:
    """Invoke agent as Prefect task."""
    # Use MCP client to call agent
    pass

@flow
async def multi_agent_workflow(user_request: str):
    """Prefect flow for multi-agent coordination."""
    # Dynamically create tasks based on planning
    plan = await analyze_request(user_request)

    results = {}
    for step in plan.steps:
        # Create task dynamically
        if step.parallel:
            # Submit to run in parallel
            future = invoke_agent.submit(step.agent_id, step.parameters)
            results[step.step_id] = future
        else:
            # Wait for dependencies
            for dep_id in step.depends_on:
                await results[dep_id]

            result = await invoke_agent(step.agent_id, step.parameters)
            results[step.step_id] = result

    return results
```

**Implementation Note:** Prefect adds complexity. Only implement if you need advanced workflow features like conditional branching, dynamic task generation, workflow versioning, etc.

---

**STOP HERE - Milestone 7 is optional advanced feature. Proceed to Milestone 8 for production hardening.**

---

## Milestone 8: Production Hardening and Observability

### Goal
Harden the system for production use with comprehensive error handling, monitoring, rate limiting, and performance optimization.

### Success Criteria
- [ ] Comprehensive error handling and recovery
- [ ] Rate limiting on API endpoints
- [ ] Performance monitoring and metrics
- [ ] Health checks for all services
- [ ] Graceful degradation patterns
- [ ] Load testing completed
- [ ] Security hardening (authentication, input validation)
- [ ] Documentation complete

### Key Changes

#### 1. Add Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/v1/orchestrate")
@limiter.limit("10/minute")  # 10 requests per minute
async def orchestrate_request(...):
    pass
```

#### 2. Enhanced Health Checks

```python
@app.get("/health/live")
async def liveness():
    """Kubernetes liveness probe."""
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness(
    container: ServiceContainer = Depends(get_service_container)
):
    """Kubernetes readiness probe - checks dependencies."""
    checks = {
        "redis": await check_redis_health(container.redis_client),
        "agents": await check_agents_health(container.agent_registry)
    }

    if not all(checks.values()):
        raise HTTPException(status_code=503, detail="Not ready")

    return {"status": "ready", "checks": checks}
```

#### 3. Metrics and Monitoring

```python
from prometheus_client import Counter, Histogram

request_count = Counter('api_requests_total', 'Total API requests')
request_duration = Histogram('api_request_duration_seconds', 'Request duration')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    request_count.inc()
    with request_duration.time():
        response = await call_next(request)
    return response
```

#### 4. Graceful Shutdown

```python
@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown - cleanup resources."""
    container = get_service_container()
    await container.cleanup()
```

#### 5. Security Hardening

**Input Validation:**
- Validate all user inputs
- Sanitize prompts for injection attacks
- Rate limit by user/API key

**Authentication:**
- Keep existing bearer token auth
- Add API key rotation
- Audit logging

### Testing Strategy

**Load Testing:**
```bash
# Use locust or k6 for load testing
k6 run --vus 100 --duration 30s load_test.js
```

**Chaos Engineering:**
- Test Redis failures
- Test agent unavailability
- Test network partitions

### Documentation Updates

- Complete API reference
- Deployment guide
- Troubleshooting playbook
- Architecture decision records
- Performance tuning guide

---

## Summary and Next Steps

This implementation plan provides a complete roadmap from basic cancellation to full multi-agent orchestration. The plan is structured to:

1. **Build incrementally** - Each milestone adds value independently
2. **Stop for review** - Human review after each milestone prevents scope creep
3. **No heavy frameworks** - Simple, understandable code (no LangGraph)
4. **Production-ready** - Proper error handling, testing, documentation throughout

**Core Milestones (Required):**
1. Connection-Based Cancellation ✓
2. Session Management with Redis ✓
3. Human-in-the-Loop ✓
4. Agent Registry and Discovery ✓
5. Custom Orchestrator ✓
8. Production Hardening ✓

**Optional Milestones (As Needed):**
6. Vector Database (if semantic memory needed)
7. Prefect Workflows (if complex DAG patterns needed)

**Key Principles:**
- Agents = LLM + prompt + tools (MCP)
- No LangGraph - custom async orchestration
- MCP for agent communication
- Redis for persistence
- Simple, maintainable code

---

