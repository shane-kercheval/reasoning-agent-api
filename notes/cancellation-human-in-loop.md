# Reasoning Agent Enhancement Implementation Plan

## Project Overview

Add interruption/cancellation and human-in-the-loop capabilities to the existing reasoning agent API while maintaining OpenAI compatibility. The current architecture is well-suited for these enhancements due to its async, streaming, and event-driven design.

## Implementation Approach

1. **Start with simple asyncio task cancellation** (OpenAI-compatible)
2. **Add session manager only when needed** for human-in-the-loop
3. **Prioritize clean architecture**. Implementation DOES NOT need to be backwards compatible with the existing implementation. This project is not in production and we need to prioritize clean architecture and future extensibility over maintaining existing interfaces.
4. **Test thoroughly** at each milestone

---

## Milestone 1: Basic Connection-Based Cancellation (Single Client Scope)

### Goal
Implement OpenAI-compatible cancellation that works when clients close HTTP connections using `AbortController`. This matches real OpenAI behavior and provides immediate cancellation even during long LLM calls.

**Scope Limitation**: Each client can only cancel their own reasoning process via connection closing. Cross-tab or cross-client cancellation is NOT supported in this milestone (that requires session management in Milestone 2).

### Success Criteria
- [ ] Clients can cancel reasoning by closing connections (AbortController.abort())
- [ ] Cancellation interrupts ongoing LLM calls immediately via asyncio.CancelledError
- [ ] Cancelled streams end gracefully with proper OpenAI-compatible finish_reason
- [ ] Multiple concurrent clients each manage their own reasoning processes independently
- [ ] No resource leaks or orphaned tasks
- [ ] All existing functionality continues to work unchanged

**Limitation**: Cross-tab cancellation (Tab A cancelling Tab B's reasoning) is not supported. Each HTTP request can only cancel itself via connection closing.

### Multiple Client Behavior in Milestone 1

**What Works:**
- Client A starts reasoning → gets isolated reasoning task
- Client B starts reasoning → gets separate isolated reasoning task  
- Client A disconnects → only Client A's reasoning stops
- Client B continues unaffected

**What Doesn't Work (requires Milestone 2):**
- Tab 1 starts reasoning, Tab 2 wants to cancel it
- Mobile app starts reasoning, web browser wants to cancel it
- Any cross-request coordination

**Why No Session IDs Yet:** Connection-based cancellation creates a direct 1:1 relationship between HTTP request and reasoning task. The asyncio task is automatically scoped to the specific HTTP request, so no additional session coordination is needed for basic cancellation.

**1. Modify `api/main.py` chat_completions endpoint:**
- Make `http_request: Request` parameter required (remove `= None` default)
- Wrap reasoning execution in `asyncio.create_task()` in the endpoint
- Monitor for client disconnection using `await request.is_disconnected()` before each chunk yield
- Cancel reasoning task when disconnection detected
- Handle `asyncio.CancelledError` gracefully with proper span status (OK with "Request cancelled by client")

**2. Update `api/reasoning_agent.py`:**
- Let `asyncio.CancelledError` propagate naturally through all methods
- No special cleanup needed - rely on asyncio's automatic resource management
- Verify OpenAI client calls respect cancellation (test empirically)

**3. ~~Update `api/reasoning_models.py`:~~ (NOT NEEDED for Milestone 1)
- ~~Add `ReasoningEventType.CANCELLED` event type~~
- Connection closure IS the cancellation signal - no special events needed

### Key Implementation Patterns

```python
# In main.py - Task-based cancellation with connection monitoring
if request.stream:
    # Create task for cancellation control
    stream_generator = reasoning_agent.execute_stream(request, parent_span=span)
    reasoning_task = asyncio.create_task(
        anext(aiter(stream_generator))  # Convert generator to task
    )
    
    async def cancellation_aware_stream():
        try:
            # First yield from the task
            first_chunk = await reasoning_task
            yield first_chunk
            
            # Then continue with the rest of the stream
            async for chunk in stream_generator:
                # Check disconnection before each yield
                if await http_request.is_disconnected():
                    # Cancel the underlying task
                    reasoning_task.cancel()
                    break
                yield chunk
        except asyncio.CancelledError:
            # Update span status
            span.set_attribute("http.status_code", 200)
            span.set_status(trace.Status(trace.StatusCode.OK, "Request cancelled by client"))
            span.end()
            context.detach(token)
            # No special cancellation event - connection closes
```

```python
# In reasoning_agent.py - Natural propagation (NO special handling needed)
async def _generate_reasoning_step(self, ...):
    # Just let CancelledError propagate naturally
    response = await self.openai_client.chat.completions.create(...)
    return response
    # No try/except for CancelledError - let it bubble up
```

### Testing Strategy

**Unit Tests:**
- Test `asyncio.create_task()` and `task.cancel()` behavior
- Verify `CancelledError` propagation through reasoning pipeline
- Mock `request.is_disconnected()` to simulate client disconnection
- Use `asyncio.sleep()` delays in test reasoning steps for predictable cancellation windows
- Mock OpenAI client with controlled timing to test cancellation behavior

**Integration Tests:**
- **Multi-client isolation**: Start 2+ concurrent reasoning processes, cancel one, verify others continue unaffected
- **Real OpenAI cancellation**: Test cancellation during actual OpenAI API calls (requires `OPENAI_API_KEY`)
- **HTTP client disconnection**: Use `httpx.AsyncClient` with timeouts to simulate real disconnections
- **Resource cleanup verification**: Monitor task counts and memory usage during cancellation
- **Streaming response cancellation**: Test `AbortController.abort()` integration end-to-end

**Multi-Client Test Scenarios:**
```python
async def test_multiple_clients_independent_cancellation():
    """Test that cancelling one client doesn't affect others."""
    # Start reasoning for Client A
    task_a = asyncio.create_task(reasoning_agent.execute_stream(request_a))
    
    # Start reasoning for Client B  
    task_b = asyncio.create_task(reasoning_agent.execute_stream(request_b))
    
    # Cancel Client A
    task_a.cancel()
    
    # Verify Client B continues normally
    chunks_b = []
    async for chunk in task_b:
        chunks_b.append(chunk)
    
    assert len(chunks_b) > 0  # Client B completed
    assert task_a.cancelled()  # Client A was cancelled
```

**Real OpenAI Integration Tests** (marked with `@pytest.mark.integration`):**
```python
async def test_cancellation_during_real_openai_call():
    """Test cancellation interrupts actual OpenAI API calls."""
    request = OpenAIRequestBuilder().model("gpt-4o-mini").message("user", "Write a very long story").build()
    
    reasoning_task = asyncio.create_task(
        reasoning_agent.execute_stream(request)
    )
    
    # Let reasoning start
    await asyncio.sleep(0.5)
    
    # Cancel during OpenAI call
    reasoning_task.cancel()
    
    # Verify task was cancelled quickly (not after full OpenAI response)
    start_time = time.time()
    try:
        async for chunk in reasoning_task:
            pass
    except asyncio.CancelledError:
        pass
    
    # Should cancel quickly, not wait for full LLM response
    assert time.time() - start_time < 5.0
```

**Edge Cases:**
- Cancellation during OpenAI API calls
- Cancellation during tool execution  
- Multiple rapid cancellation attempts
- Cancellation of non-streaming requests
- **Concurrent multi-client cancellation**: Multiple clients cancelling simultaneously
- **Race conditions**: Cancellation vs. natural completion timing

### Dependencies
- None (builds on existing architecture)

### Risk Factors
- **AsyncIO complexity**: Proper task lifecycle management
- **Resource cleanup**: Ensuring OpenAI client connections close properly
- **Race conditions**: Cancellation timing vs. natural completion
- **Error propagation**: `CancelledError` must bubble up correctly

### Documentation Updates
- Update README with cancellation behavior explanation
- Add client examples showing `AbortController` usage
- Document limitations and expected behavior

---

## Milestone 2: Session Manager Infrastructure (Cross-Client Coordination)

### Goal
Create centralized session management system to support cross-request coordination needed for human-in-the-loop scenarios AND cross-client cancellation (e.g., mobile app cancelling web browser reasoning, or Tab A cancelling Tab B).

**New Capability**: Session IDs enable coordination across different HTTP requests/clients.

### Success Criteria
- [ ] Session manager handles session lifecycle (create, track, cleanup)
- [ ] Automatic cleanup of expired/abandoned sessions
- [ ] Thread-safe session operations
- [ ] Session IDs can be passed via headers (`X-Session-ID`)
- [ ] Graceful handling of missing/invalid sessions
- [ ] Integration with existing dependency injection system

### Key Changes

**1. Create `api/session_manager.py`:**
- `ReasoningSessionManager` class with async lifecycle management
- Session creation, retrieval, and cleanup operations
- Automatic background cleanup of expired sessions
- Session status tracking (active, completed, failed, etc.)

**2. Update `api/dependencies.py`:**
- Add session manager to `ServiceContainer`
- Create dependency injection for session manager
- Initialize/cleanup session manager in app lifecycle

**3. Modify `api/main.py`:**
- Extract `X-Session-ID` headers
- Create sessions when not provided
- Return session IDs in response headers
- Integrate session completion tracking

**4. Add optional explicit cancellation endpoint:**
- `POST /v1/reasoning/cancel` with session_id parameter
- Enables cross-client cancellation (mobile cancels web, Tab A cancels Tab B)
- Supplements connection-based cancellation from Milestone 1
- Extract `X-Session-ID` headers
- Create sessions when not provided
- Return session IDs in response headers
- Integrate session completion tracking

### Key Implementation Patterns

```python
# Session model
@dataclass
class ReasoningSession:
    session_id: str
    status: SessionStatus
    created_at: float
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Session manager interface
class ReasoningSessionManager:
    async def create_session(self, user_id: Optional[str] = None) -> str
    async def get_session(self, session_id: str) -> Optional[ReasoningSession]
    async def complete_session(self, session_id: str, success: bool = True)
    async def cleanup_session(self, session_id: str)
```

### Testing Strategy

**Unit Tests:**
- Session creation and retrieval
- Session status transitions
- Cleanup logic and timing
- Concurrent session operations
- Invalid session handling

**Integration Tests:**
- Session manager lifecycle with FastAPI app
- Header parsing and response setting
- Session persistence across requests
- Cleanup during app shutdown

**Load Tests:**
- Many concurrent sessions
- Memory usage over time
- Cleanup performance with large session counts

### Dependencies
- Milestone 1 (cancellation) must be complete
- Session manager integrates with existing cancellation

### Risk Factors
- **Memory management**: Sessions must not leak
- **Concurrency**: Multiple requests accessing same session
- **Cleanup timing**: Balance between immediate cleanup and utility
- **Integration complexity**: Must not break existing flows

---

## Milestone 3: Human-in-the-Loop Core Functionality

### Goal
Enable reasoning processes to pause and request human input, then resume with the provided response. This includes timeout handling, input validation, and graceful continuation.

### Success Criteria
- [ ] Reasoning can request human input and pause execution
- [ ] Human responses can be provided via separate HTTP endpoint
- [ ] Timeout handling for unresponded input requests
- [ ] Multiple input types supported (text, multiple choice)
- [ ] Input/response matching via unique IDs
- [ ] Proper streaming events for input requests and responses

### Key Changes

**1. Extend `api/reasoning_models.py`:**
- Add `ReasoningAction.REQUEST_HUMAN_INPUT`
- Create `HumanInputRequest` and `HumanInputResponse` models
- Add human input event types (HUMAN_INPUT_REQUIRED, HUMAN_INPUT_RECEIVED)

**2. Update `api/session_manager.py`:**
- Add human input state to sessions
- Implement input request/response coordination
- Add timeout handling with `asyncio.wait_for`

**3. Extend `api/reasoning_agent.py`:**
- Handle `REQUEST_HUMAN_INPUT` action in reasoning steps
- Generate human input requests from reasoning context
- Integrate human responses into reasoning context
- Stream human input events to clients

**4. Add human input endpoint to `api/main.py`:**
- `POST /v1/reasoning/human-input` for providing responses
- Input validation and session matching
- Proper error handling for invalid/expired requests

### Key Implementation Patterns

```python
# Human input in reasoning step
class ReasoningStep(BaseModel):
    thought: str
    next_action: ReasoningAction  # Can be REQUEST_HUMAN_INPUT
    human_input_request: Optional[HumanInputRequest] = None

# Waiting for human input
async def _wait_for_human_input(
    self, session_id: str, request: HumanInputRequest
) -> Optional[HumanInputResponse]:
    success = self.session_manager.request_human_input(session_id, request)
    if not success:
        return None
    
    return await self.session_manager.wait_for_human_input(
        session_id, timeout=request.timeout_seconds
    )
```

### Testing Strategy

**Unit Tests:**
- Human input request generation
- Input/response matching logic
- Timeout behavior
- Invalid input handling
- Session state transitions

**Integration Tests:**
- End-to-end human input flow
- Multiple concurrent input requests
- Timeout scenarios
- Client disconnect during input waiting

**Manual Testing:**
- Web interface integration
- User experience flows
- Error message clarity
- Cross-browser compatibility

### Dependencies
- Milestone 2 (session manager) must be complete
- Builds on session coordination infrastructure

### Risk Factors
- **Timeout handling**: Complex async coordination
- **State consistency**: Session state during input waiting
- **Client experience**: Clear UI/UX for input requests
- **Resource usage**: Long-running input waits

---

## Milestone 4: Enhanced Web Interface Integration

### Goal
Update the web interface to support cancellation and human input features, providing a seamless user experience for the new capabilities.

### Success Criteria
- [ ] Cancel button stops reasoning immediately
- [ ] Human input requests show modal/dialog interfaces
- [ ] Clear visual indicators for reasoning states (active, waiting, cancelled)
- [ ] Proper error handling and user feedback
- [ ] Session persistence across page refreshes
- [ ] Mobile-responsive input interfaces

### Key Changes

**1. Update `web-client/main.py`:**
- Add cancellation UI components
- Implement human input modal/dialog system
- Handle session ID management in browser
- Add visual state indicators

**2. JavaScript enhancements:**
- `AbortController` integration for cancellation
- Event listeners for human input events
- Local storage for session persistence
- Error handling and user feedback

### Testing Strategy

**Manual Testing:**
- User workflow testing
- Cross-browser compatibility
- Mobile responsiveness
- Error scenario handling

**Automated Testing:**
- Selenium tests for key user flows
- JavaScript unit tests for core logic
- API integration tests from web client

### Dependencies
- Milestone 3 (human input) must be complete
- Requires understanding of existing web client architecture

### Risk Factors
- **UI/UX complexity**: Making features discoverable and intuitive
- **Browser compatibility**: AbortController and modern JS features
- **State management**: Keeping client and server state synchronized

---

## Milestone 5: Performance Optimization and Production Readiness

### Goal
Optimize performance, add monitoring, and prepare the enhanced system for production-like usage with proper error handling and observability.

### Success Criteria
- [ ] Performance benchmarking shows no regression
- [ ] Comprehensive logging and monitoring
- [ ] Error handling covers all edge cases
- [ ] Documentation is complete and accurate
- [ ] Load testing validates concurrent usage
- [ ] Memory usage is bounded and predictable

### Key Changes

**1. Performance optimizations:**
- Session cleanup efficiency
- Memory usage optimization
- Connection pooling verification

**2. Monitoring and observability:**
- Enhanced tracing for cancellation flows
- Metrics for human input response times
- Error rate monitoring

**3. Documentation updates:**
- Complete API documentation
- Architecture decision records
- Deployment guide updates

### Testing Strategy

**Performance Testing:**
- Load testing with many concurrent sessions
- Memory leak detection
- Cancellation performance under load

**Reliability Testing:**
- Extended runtime testing
- Error injection testing
- Recovery scenario testing

### Dependencies
- All previous milestones must be complete

### Risk Factors
- **Performance regressions**: New features impacting existing performance
- **Production deployment**: Real-world usage patterns
- **Monitoring overhead**: Observability without performance impact

---

## General Implementation Guidelines

### Before Starting Each Milestone

1. **Read the existing codebase** to understand current patterns
2. **Review FastAPI documentation** for async patterns: https://fastapi.tiangolo.com/async/
3. **Study asyncio cancellation** patterns: https://docs.python.org/3/library/asyncio-task.html#cancellation
4. **Understand OpenAI API specification**: https://platform.openai.com/docs/api-reference/streaming

### Code Quality Standards

- **Type hints everywhere** - leverage existing Pydantic patterns
- **Comprehensive error handling** - fail gracefully with clear messages
- **Async best practices** - proper resource cleanup and cancellation handling
- **Testing first** - write tests before implementation when possible
- **Clean interfaces** - minimal coupling between components

### Testing Philosophy

- **Focus on behavior, not implementation** - test what the system does, not how
- **Test edge cases thoroughly** - cancellation timing, timeouts, errors
- **Integration tests for user flows** - end-to-end scenario testing
- **Performance tests for scalability** - ensure new features don't degrade performance

### Documentation Requirements

- **Update README** for each milestone
- **API documentation** for new endpoints
- **Architecture decisions** in markdown files
- **Client examples** showing usage patterns

### Questions to Ask During Implementation

1. How does this integrate with existing OpenAI compatibility?
2. What happens if the client disconnects unexpectedly?
3. How do we prevent resource leaks?
4. What error conditions need special handling?
5. How will this behave under load?
6. What would a malicious client try to do?

---

## Validation Checkpoints

After each milestone, validate:

- [ ] All existing tests still pass
- [ ] New functionality works as designed
- [ ] Performance hasn't regressed
- [ ] Documentation reflects current state
- [ ] Error handling is comprehensive
- [ ] Resource cleanup is working
- [ ] User experience is intuitive

Stop after each milestone for human review before proceeding to the next.