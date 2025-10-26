# Response to Agent Questions - Electron Desktop Client Migration

## Answers to Your Questions

### 1. API Authentication
**Decision**: Support optional authentication matching backend behavior.

**Implementation**:
- Desktop client reads `REASONING_API_TOKEN` from `.env` file (Electron main process)
- If token is set, pass as `Authorization: Bearer {token}` header
- If empty/unset, don't send Authorization header (works when `REQUIRE_AUTH=false` on API)
- No complexity needed - this is a dev tool connecting to local services

**Rationale**: Keeps desktop client simple while matching backend's optional auth model.

---

### 2. Routing Mode Selection
**Decision**: Expose routing mode in chat interface (NOT buried in settings panel).

**Implementation**:
- Add routing mode selector in chat UI (near message input or as toolbar button)
- Options: Dropdown or segmented control with 3 choices:
  - `Passthrough` - Fast, no reasoning (default)
  - `Reasoning` - Show thinking steps
  - `Auto` - Let classifier decide
- Send selected mode via `X-Routing-Mode` header on each request
- Optionally display which mode was used in response (could show badge on assistant messages)

**Rationale**: This is a power-user/developer tool. Quick experimentation with routing modes is valuable. Users should be able to toggle mid-conversation to compare behaviors.

---

### 3. Session Management & Conversation History
**Decision**: Backend conversation storage with smart stateful/stateless hybrid approach.

**Smart Hybrid Approach**:

**Stateful Mode** (conversation storage):
- Client sends **only user messages** → Backend stores in postgres
- `conversation_id` provided → Backend loads history, appends new message
- `conversation_id` omitted → Backend creates new conversation, returns `conversation_id`
- Backend manages full conversation history
- Desktop client sends only new user messages on subsequent requests

**Stateless Mode** (no storage):
- Client sends **system message** → Backend does NOT store conversation
- Client sends full message history with each request (current behavior)
- Useful for: custom system prompts, ephemeral chats, testing, programmatic use

**Rationale**:
- System message signals "I'm managing my own context"
- No system message signals "please manage my conversation for me"
- Backward compatible - existing stateless behavior preserved
- Desktop client opts into stateful mode (default: no system message)
- See full implementation plan: `docs/implementation_plans/2025-10-25-conversation-storage.md`

**Implementation Timeline**:
- Backend conversation storage implemented FIRST
- Desktop client consumes conversation API afterward
- Desktop client Milestone 6 (State Management) depends on backend being ready

---

### 4. Reasoning Event Display
**Confirmed**: Reasoning events unchanged.

**Current Structure** (from `api/reasoning_models.py`):
```python
class ReasoningEvent(BaseModel):
    type: ReasoningEventType  # e.g., "iteration_start", "planning", "tool_result"
    step_iteration: int
    metadata: dict[str, Any]  # Contains tools, thoughts, results, etc.
    error: str | None
```

**SSE Stream Format** (from `api/openai_protocol.py`):
```python
delta: {
    content: str | None,           # Regular response content
    reasoning_event: ReasoningEvent | None  # Reasoning metadata
}
```

**Action**: Continue using this structure as designed. Reference `web-client/main.py` lines 188-389 for event type rendering patterns.

---

### 5. LiteLLM Model Discovery
**Decision**: API must proxy LiteLLM's `/v1/models` endpoint. Desktop client fetches models dynamically.

**Backend Changes Required** (separate implementation plan - see `2025-10-25-litellm-models-proxy.md`):
- Modify `GET /v1/models` endpoint in `api/main.py`
- Proxy to `http://litellm:4000/v1/models` instead of hardcoded list
- Forward LiteLLM's response to client

**Desktop Client Implementation**:
- On app startup, fetch `GET /v1/models` from API
- Populate model selector dropdown dynamically
- Store available models in Zustand state
- Handle fetch errors gracefully (show hardcoded defaults if API unavailable)

**Rationale**:
- Current hardcoded list (`gpt-4o`, `gpt-4o-mini`) is brittle
- LiteLLM config (`litellm_config.yaml`) is source of truth for available models
- Dynamic discovery means users see exactly what's available
- Supports future model additions without code changes

---

### 6. Testing Strategy
**Decision**: Set up test infrastructure early, write tests incrementally, comprehensive coverage in polish phase.

**Milestone 1 (Scaffolding)**:
- Install Jest + React Testing Library + @testing-library/react
- Configure test scripts in package.json
- Add basic smoke test (App renders without crashing)

**Milestones 2-7 (Incremental Testing)**:
- Write unit tests for critical logic (SSE parser, API client, Zustand store)
- Write component tests for key components (ChatMessage, ReasoningStep)
- Write integration tests for API interaction and streaming
- Focus on critical paths, not 100% coverage yet

**Milestone 9 (Comprehensive Testing)**:
- Fill coverage gaps
- Add edge case tests
- Add accessibility tests
- Performance tests (100+ messages)

**Rationale**: Test infrastructure from the start enables TDD without blocking progress. Incremental testing maintains quality without slowing development.

---

## Additional Guidance

### Observations Responses

**Conversation history persistence**: Addressed in Question 3 - client-side for now, backend storage deferred to orchestration work.

**Copy/export functionality**: Good catch! Add this to Milestone 9 (Polish):
- Copy individual messages to clipboard
- Export conversation as markdown/JSON
- Share conversation (future enhancement)

**Orchestration 501 handling**: Desktop client should handle gracefully:
- Show user-friendly error when orchestration returns 501
- Suggest using `passthrough` or `reasoning` mode instead
- Don't crash or show raw error JSON

### Dependencies

This migration depends on two backend implementation plans:
1. **LiteLLM Models Proxy** (`2025-10-25-litellm-models-proxy.md`)
   - Required for Milestone 5 (settings panel needs model list)
   - Small, quick implementation

2. **Conversation Storage** (`2025-10-25-conversation-storage.md`)
   - Required for Milestone 6+ (state management needs conversation API)
   - Medium-sized implementation (1-2 weeks)

**Implementation Strategy**:
- Start backend conversation storage immediately
- Start desktop client Milestones 1-5 in parallel
- Desktop client Milestone 6 blocks on backend conversation storage completion
- This allows parallel development without blocking

---

## Updated Implementation Order

1. **Backend: LiteLLM Models Proxy** (small, quick, unblocks desktop client Milestone 5)
2. **Desktop Client: Milestones 1-10** (main Electron migration work)
3. **Backend: Conversation Storage** (larger, deferred to orchestration design phase)

This allows the desktop client migration to proceed without blocking on conversation storage architecture decisions.
