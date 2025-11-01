# Refactor: Main.py and Conversation DB Cleanup

**Date**: 2025-11-01
**Status**: Planning
**Goal**: Simplify conversation storage, reduce duplication, improve testability

---

## Problems Identified

### 1. conversation_db.py Issues
- `routing_mode` in Conversation doesn't make sense (routing can change per-request)
- Message storage split between `create_conversation` and `append_messages` causes complexity
- `_insert_messages` loops instead of using batch insert (performance issue)

### 2. main.py Issues
- `extract_system_message` should be in utility module with unit tests
- Conversation header parsing scattered throughout main.py
- `messages_for_llm` building logic not reusable or testable
- 70+ lines of duplicate streaming buffering/storage code in passthrough and reasoning branches
- `is_new_conversation` tracking is confusing workaround

### 3. Executor Inconsistencies
- `execute_passthrough_stream` is a function, `ReasoningAgent.execute_stream` is a method
- No common interface between executors
- `ReasoningAgent.execute_stream` missing `check_disconnected` parameter (should support cancellation)
- Duplicate code in both executors (span attributes, content buffering)

---

## Design Decisions

### Conversation Storage Flow (New)

**Before:** Messages stored in create_conversation OR append_messages (conditional logic)

**After:** Clear separation of concerns
1. `create_conversation(system_message)` → creates conversation record with system message, returns UUID
2. `append_messages(conversation_id, messages)` → ALWAYS used for user/assistant message storage

**Result:** `is_new_conversation` tracking becomes obsolete

### Conversation Modes

Use enum instead of string literals:
```python
class ConversationMode(str, Enum):
    STATELESS = "stateless"  # No X-Conversation-ID header
    NEW = "new"               # Header is "" or "null"
    CONTINUING = "continuing" # Header is valid UUID
```

### Executor Interface

Create base class with common interface:
- `execute_stream()` - unified streaming method with disconnection support
- `get_buffered_content()` - retrieve buffered assistant content after streaming
- `_extract_content_from_chunk()` - shared SSE parsing logic
- Shared span attribute setting logic

---

## Implementation Plan

### Phase 1: Database Layer Refactor

**File**: `api/database/conversation_db.py`

**Changes:**

1. **Remove `routing_mode` from database**
   - Drop `routing_mode` column from conversations table (create Alembic migration)
   - Remove `routing_mode` parameter from `create_conversation()`
   - Remove `routing_mode` field from `Conversation` dataclass
   - Update docstrings

2. **Separate conversation creation from message storage**
   - Change `create_conversation()` signature:
     ```python
     async def create_conversation(
         self,
         system_message: str = "You are a helpful assistant.",
         title: str | None = None,
     ) -> UUID:
         """Create conversation record with system message (no user/assistant messages)."""
     ```
   - Remove `messages` parameter
   - Remove call to `_insert_messages` from `create_conversation`
   - All user/assistant message storage happens via `append_messages()` only

3. **Optimize batch insert**
   - Change `_insert_messages()` to use `executemany()` instead of loop
   - Keep transaction safety (already in transaction context)

4. **Database migration**
   - Create Alembic migration to drop `routing_mode` column from `conversations` table
   - Test migration runs successfully (up and down)
   - Apply migration: `uv run alembic upgrade head`

**Testing:**
- Update integration tests in `tests/test_conversation_db.py`:
  - Test `create_conversation()` creates conversation with system message (no messages in messages table)
  - Test `append_messages()` works for first message set after creation
  - Test batch insert (multiple messages in single call)
  - Test concurrent append with locking (existing test)

**✅ Run all tests before proceeding to Phase 2:**
```bash
make non_integration_tests  # Quick unit test feedback
make integration_tests      # Full database integration tests
```

---

### Phase 2: Utility Modules

**Files to Update/Create:**

#### 2.1 Update `api/openai_protocol.py`

Add message parsing utility:

```python
def extract_system_message(messages: list[dict[str, object]]) -> str | None:
    """
    Extract first system message from messages list.

    Move from main.py - this is OpenAI message parsing logic.
    """
    # Implementation from main.py
```

**Testing:**
- Unit tests in `tests/test_openai_protocol.py`:
  - `test_extract_system_message_present`
  - `test_extract_system_message_absent`
  - `test_extract_system_message_multiple` (returns first)

#### 2.2 New file: `api/conversation_utils.py`

```python
class ConversationMode(str, Enum):
    """Conversation mode based on X-Conversation-ID header."""
    STATELESS = "stateless"
    NEW = "new"
    CONTINUING = "continuing"

@dataclass
class ConversationContext:
    """Parsed conversation context from headers."""
    mode: ConversationMode
    conversation_id: UUID | None

def parse_conversation_header(header_value: str | None) -> ConversationContext:
    """
    Parse X-Conversation-ID header into conversation context.

    Raises ValueError if header has invalid UUID format.
    """

async def build_llm_messages(
    request_messages: list[dict],
    conversation_ctx: ConversationContext,
    conversation_db: ConversationDB | None,
) -> tuple[list[dict], str | None]:
    """
    Build complete message list for LLM based on conversation context.

    Returns: (messages_for_llm, system_message)

    Raises:
        ValueError: If system message in continuation request
        ValueError: If conversation not found
    """

async def store_conversation_messages(
    conversation_db: ConversationDB,
    conversation_id: UUID,
    request_messages: list[dict],
    assistant_content: str,
) -> None:
    """
    Store user messages + assistant response after streaming completes.

    Simple wrapper around append_messages that filters system messages
    and constructs the messages list.
    """
```

**Testing:**
- Unit tests in `tests/test_conversation_utils.py`:
  - `test_parse_conversation_header_stateless` (None)
  - `test_parse_conversation_header_new` ("", "null")
  - `test_parse_conversation_header_continuing` (valid UUID)
  - `test_parse_conversation_header_invalid_uuid` (raises ValueError)
  - `test_build_llm_messages_stateless`
  - `test_build_llm_messages_new`
  - `test_build_llm_messages_continuing` (mock conversation_db)
  - `test_build_llm_messages_system_in_continuation` (raises ValueError)
  - `test_store_conversation_messages` (mock conversation_db.append_messages)

**✅ Run all tests before proceeding to Phase 3:**
```bash
make non_integration_tests  # Quick unit test feedback
```

---

### Phase 3: Executor Base Class

**New Directory**: `api/executors/`
**New File**: `api/executors/base.py`

```python
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Callable
import json
from opentelemetry import trace

def extract_content_from_sse_chunk(chunk: str) -> str | None:
    """
    Extract content delta from SSE chunk.

    Shared by all executors for content buffering.
    Returns None if chunk is malformed or contains no content.

    Args:
        chunk: SSE formatted chunk string (e.g., "data: {...}\n\n")

    Returns:
        Content string if found, None otherwise
    """
    if not (isinstance(chunk, str) and "data: " in chunk):
        return None

    try:
        data_line = chunk.strip().replace("data: ", "")
        if data_line and data_line != "[DONE]":
            chunk_data = json.loads(data_line)
            if choices := chunk_data.get("choices"):
                delta = choices[0].get("delta", {})
                return delta.get("content")
    except (json.JSONDecodeError, KeyError, IndexError):
        pass
    return None


class BaseExecutor(ABC):
    """
    Base class for all execution paths (passthrough, reasoning, orchestration).

    Provides:
    - Common interface for streaming with disconnection support
    - Content buffering during streaming
    - Shared SSE content extraction
    """

    def __init__(self):
        self._content_buffer: list[str] = []

    @abstractmethod
    async def execute_stream(
        self,
        request: OpenAIChatRequest,
        parent_span: trace.Span | None = None,
        check_disconnected: Callable[[], bool] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Execute streaming request with content buffering.

        Args:
            request: OpenAI chat request
            parent_span: Optional parent span for tracing
            check_disconnected: Optional callback to check client disconnection

        Yields:
            SSE formatted chunks
        """
        pass

    def get_buffered_content(self) -> str:
        """Get complete buffered assistant content after streaming."""
        return "".join(self._content_buffer)

    def _reset_buffer(self) -> None:
        """Reset content buffer for new request."""
        self._content_buffer = []

    def _buffer_chunk(self, chunk: str) -> None:
        """Buffer content from SSE chunk."""
        if content := extract_content_from_sse_chunk(chunk):
            self._content_buffer.append(content)

    async def _check_disconnection(
        self,
        check_disconnected: Callable[[], bool] | None,
    ) -> None:
        """
        Check if client disconnected and raise CancelledError if so.

        Should be called before each yield in execute_stream.
        """
        if check_disconnected and await check_disconnected():
            raise asyncio.CancelledError("Client disconnected")
```

**Testing:**
- Unit tests in `tests/test_executors/test_base.py`:
  - `test_extract_content_from_sse_chunk_valid`
  - `test_extract_content_from_sse_chunk_done`
  - `test_extract_content_from_sse_chunk_malformed`
  - `test_buffer_chunk_valid_content`
  - `test_buffer_chunk_no_content`
  - `test_get_buffered_content`
  - `test_check_disconnection_triggered`
  - `test_check_disconnection_not_triggered`

**✅ Run all tests before proceeding to Phase 4:**
```bash
make non_integration_tests  # Quick unit test feedback
```

---

### Phase 4: Refactor Executors

**File Moves:**
- `api/passthrough.py` → `api/executors/passthrough.py`
- `api/reasoning_agent.py` → `api/executors/reasoning_agent.py`

#### 4.1 `api/executors/passthrough.py`

**Changes:**
1. Move file from `api/passthrough.py` to `api/executors/passthrough.py`
2. Convert `execute_passthrough_stream()` function to `PassthroughExecutor` class
3. Extend `BaseExecutor`
4. Implement content buffering using base class methods
5. Keep existing span attribute logic (or extract to base if 95% same)

```python
class PassthroughExecutor(BaseExecutor):
    """Executes passthrough streaming with content buffering."""

    async def execute_stream(
        self,
        request: OpenAIChatRequest,
        parent_span: trace.Span | None = None,
        check_disconnected: Callable[[], bool] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream passthrough response while buffering content."""
        self._reset_buffer()

        # Existing passthrough logic...
        async for chunk in litellm_stream:
            self._buffer_chunk(chunk)  # Buffer content
            await self._check_disconnection(check_disconnected)  # Check cancellation
            yield chunk
```

**Testing:**
- Update integration tests in `tests/test_passthrough.py`:
  - Change from function calls to class instantiation
  - Test content buffering works
  - Test `get_buffered_content()` returns correct content
  - Test disconnection handling

#### 4.2 `api/executors/reasoning_agent.py`

**Changes:**
1. Move file from `api/reasoning_agent.py` to `api/executors/reasoning_agent.py`
2. Extend `BaseExecutor`
3. Add `check_disconnected` parameter to `execute_stream()`
4. Implement content buffering using base class methods
5. Call `self._check_disconnection()` before each yield

```python
class ReasoningAgent(BaseExecutor):
    """Reasoning agent with content buffering."""

    def __init__(self, tools, prompt_manager):
        super().__init__()  # Initialize base class (buffer)
        self.tools = tools
        self.prompt_manager = prompt_manager

    async def execute_stream(
        self,
        request: OpenAIChatRequest,
        parent_span: trace.Span | None = None,
        check_disconnected: Callable[[], bool] | None = None,  # NEW
    ) -> AsyncGenerator[str, None]:
        """Execute reasoning with content buffering and cancellation support."""
        self._reset_buffer()

        # Existing reasoning logic...
        async for chunk in reasoning_stream:
            self._buffer_chunk(chunk)  # Buffer content
            await self._check_disconnection(check_disconnected)  # NEW
            yield chunk
```

**Testing:**
- Update integration tests in `tests/test_reasoning_agent.py`:
  - Test content buffering works
  - Test `get_buffered_content()` returns correct content
  - Test disconnection handling (NEW)

**Dependency Cleanup:**
- Remove `get_reasoning_agent()` from `api/dependencies.py`
- Remove `ReasoningAgentDependency` type alias from `api/dependencies.py`
- Keep `get_tools()` and `get_prompt_manager()` (still needed as dependencies)

**Import Updates:**
- Update all imports across codebase:
  - `from api.passthrough import execute_passthrough_stream` → `from api.executors.passthrough import PassthroughExecutor`
  - `from api.reasoning_agent import ReasoningAgent` → `from api.executors.reasoning_agent import ReasoningAgent`
- Update `api/dependencies.py` imports
- Update `api/main.py` imports

**Unit Test Updates:**

1. **Delete obsolete tests** in `tests/unit_tests/test_dependencies.py`:
   - `test__get_reasoning_agent__creates_agent_with_dependencies` (lines 209-245)
   - `test__get_reasoning_agent__works_with_litellm` (lines 247-266)
   - `TestReasoningAgentInstanceIsolation` class (lines 421-557)

   These test the DI function that no longer exists.

2. **Update** `tests/unit_tests/test_api.py`:
   - Change tests from mocking `get_reasoning_agent` to mocking dependencies
   - Example pattern:
     ```python
     # OLD:
     mock_agent = AsyncMock()
     app.dependency_overrides[get_reasoning_agent] = lambda: mock_agent

     # NEW:
     mock_tools = {}
     mock_prompt_manager = Mock()
     app.dependency_overrides[get_tools] = lambda: mock_tools
     app.dependency_overrides[get_prompt_manager] = lambda: mock_prompt_manager
     # Real ReasoningAgent instantiated with mocked dependencies
     ```

**Integration Test Updates:**

1. **Update** `tests/integration_tests/test_reasoning_integration.py`:
   - Remove `app.dependency_overrides[get_reasoning_agent]` lines
   - Keep dependency mocks for tools and prompt_manager
   - main.py will instantiate ReasoningAgent directly with mocked dependencies

2. **No changes needed** for:
   - `tests/integration_tests/test_conversation_storage_api.py` (doesn't use ReasoningAgent DI)
   - End-to-end tests that run real servers

**✅ Run all tests before proceeding to Phase 5:**
```bash
make non_integration_tests  # Quick unit test feedback
make integration_tests      # Full executor integration tests
```

---

### Phase 5: Simplify main.py

**File**: `api/main.py`

**Changes:**

1. **Update endpoint signature** - direct instantiation instead of DI:
   ```python
   # OLD:
   async def chat_completions(
       reasoning_agent: ReasoningAgentDependency,  # Injected
       conversation_db: ConversationDBDependency,
       ...
   ):

   # NEW:
   async def chat_completions(
       tools: ToolsDependency,  # Still injected
       prompt_manager: PromptManagerDependency,  # Still injected
       conversation_db: ConversationDBDependency,  # Still injected
       ...
   ):
       # Executors instantiated directly in routing branches
   ```

2. **Remove helper functions** - move to utilities:
   - Delete `extract_system_message()` (use `openai_protocol.extract_system_message`)
   - Delete conversation parsing logic (use `conversation_utils.parse_conversation_header`)

2. **Simplify conversation handling**:
   ```python
   # Parse conversation context
   conversation_ctx = parse_conversation_header(
       http_request.headers.get("X-Conversation-ID")
   )

   # Build messages for LLM
   messages_for_llm, system_msg = await build_llm_messages(
       request.messages,
       conversation_ctx,
       conversation_db,
   )

   # Create new conversation if needed
   conversation_id = None
   if conversation_ctx.mode == ConversationMode.NEW:
       conversation_id = await conversation_db.create_conversation(
           system_message=system_msg or "You are a helpful assistant.",
       )
   elif conversation_ctx.mode == ConversationMode.CONTINUING:
       conversation_id = conversation_ctx.conversation_id
   ```

3. **Simplify streaming branches with direct instantiation**:

   **Passthrough branch:**
   ```python
   # ROUTE A: PASSTHROUGH PATH
   if routing_decision.routing_mode == RoutingMode.PASSTHROUGH:
       llm_request = request.model_copy(update={"messages": messages_for_llm})

       async def span_aware_stream():
           executor = PassthroughExecutor()  # DIRECT INSTANTIATION

           try:
               async for chunk in executor.execute_stream(
                   llm_request,
                   parent_span=span,
                   check_disconnected=http_request.is_disconnected,
               ):
                   yield chunk

               # Store messages after streaming completes
               if conversation_id:
                   await store_conversation_messages(
                       conversation_db=conversation_db,
                       conversation_id=conversation_id,
                       request_messages=request.messages,
                       assistant_content=executor.get_buffered_content(),
                   )
           except asyncio.CancelledError:
               span.set_attribute("http.cancelled", True)
               span_cleanup(span, token)
               return
           finally:
               span_cleanup(span, token)

       return StreamingResponse(span_aware_stream(), ...)
   ```

   **Reasoning branch:**
   ```python
   # ROUTE B: REASONING PATH
   if routing_decision.routing_mode == RoutingMode.REASONING:
       llm_request = request.model_copy(update={"messages": messages_for_llm})

       async def span_aware_reasoning_stream():
           # DIRECT INSTANTIATION with injected dependencies
           reasoning_agent = ReasoningAgent(tools, prompt_manager)

           try:
               async for chunk in reasoning_agent.execute_stream(
                   llm_request,
                   parent_span=span,
                   check_disconnected=http_request.is_disconnected,  # NOW SUPPORTED
               ):
                   yield chunk

               # Store messages after streaming completes (SAME AS PASSTHROUGH)
               if conversation_id:
                   await store_conversation_messages(
                       conversation_db=conversation_db,
                       conversation_id=conversation_id,
                       request_messages=request.messages,
                       assistant_content=reasoning_agent.get_buffered_content(),
                   )
           except asyncio.CancelledError:
               span.set_attribute("http.cancelled", True)
               span_cleanup(span, token)
               return
           finally:
               span_cleanup(span, token)

       return StreamingResponse(span_aware_reasoning_stream(), ...)
   ```

4. **Remove obsolete variables**:
   - Delete `is_new_conversation` tracking (no longer needed)

**Why Direct Instantiation:**
- Executors are stateless (new instance per request anyway)
- No lifecycle management needed (DI was providing no benefit)
- Dependencies (`tools`, `prompt_manager`) still injected - just passed to constructors
- Simpler, more explicit, easier to understand
- Testing remains clean by mocking dependencies instead of executors

**Expected Impact:**
- Reduce main.py from ~800 lines to ~500 lines
- Eliminate 70+ lines of duplicate buffering/storage code
- Improve readability and maintainability

**Testing:**
- Update integration tests in `tests/test_main.py`:
  - Test new conversation flow (create empty -> append user -> stream -> append assistant)
  - Test continuing conversation flow (append user -> stream -> append assistant)
  - Test stateless mode (no storage)
  - Verify no regression in existing functionality

**✅ Run all tests before completion:**
```bash
make non_integration_tests  # Quick unit test feedback
make integration_tests      # Full end-to-end integration tests
make tests                  # Full test suite (linting + all tests)
```

---

## Testing Strategy

### Unit Tests (New)
- `tests/test_openai_protocol.py` - message parsing utilities (add tests for `extract_system_message`)
- `tests/test_conversation_utils.py` - conversation context and storage helpers
- `tests/test_executors/test_base.py` - base executor functionality and SSE content extraction

### Integration Tests (Updated)
- `tests/test_conversation_db.py` - verify new create/append flow, batch insert
- `tests/test_passthrough.py` - verify PassthroughExecutor with buffering
- `tests/test_reasoning_agent.py` - verify ReasoningAgent with buffering and disconnection
- `tests/test_main.py` - verify end-to-end conversation flows

### Manual Testing
- Test new conversation creation with stateful mode
- Test continuing conversation with stateful mode
- Test client disconnection handling in both passthrough and reasoning modes
- Verify no performance regression with batch insert

### Testing Approach for Direct Instantiation

**Key Principle**: Mock dependencies, not executors

**Before (with DI):**
```python
# Mock the executor itself
mock_agent = AsyncMock()
app.dependency_overrides[get_reasoning_agent] = lambda: mock_agent
```

**After (direct instantiation):**
```python
# Mock the dependencies - executor instantiated with mocks
mock_tools = {}
mock_prompt_manager = Mock()
app.dependency_overrides[get_tools] = lambda: mock_tools
app.dependency_overrides[get_prompt_manager] = lambda: mock_prompt_manager

# main.py creates: ReasoningAgent(mock_tools, mock_prompt_manager)
```

**Benefits:**
- ✅ Still uses FastAPI's clean DI override system (no monkeypatching)
- ✅ Tests real constructor and initialization logic
- ✅ More realistic testing (actual instantiation path)
- ✅ Less brittle (no string-based imports)

**Integration Tests:**
- Most integration tests don't need changes (they test end-to-end with real servers)
- Tests using dependency overrides just switch from mocking executors to mocking dependencies

---

## Migration Path

**Recommended Order:**
1. Phase 1 (Database + Migration) - breaking change, do first
2. Phase 2 (Utilities) - new code, no breaking changes
3. Phase 3 (Base Class) - new code, no breaking changes
4. Phase 4 (Executors) - refactor existing, test heavily
5. Phase 5 (Main.py) - simplify using new utilities

**Rollback Plan:**
- Each phase is independently testable
- If phase fails, can revert single phase without affecting others
- Database migration has explicit rollback

---

## Success Metrics

- [ ] `routing_mode` removed from database and codebase
- [ ] `create_conversation()` no longer accepts messages
- [ ] `_insert_messages()` uses batch insert
- [ ] All utility functions have unit tests with >90% coverage
- [ ] `BaseExecutor` base class used by all executors
- [ ] `ReasoningAgent.execute_stream()` supports disconnection checking
- [ ] `is_new_conversation` variable removed from main.py
- [ ] Duplicate buffering/storage code eliminated (DRY)
- [ ] main.py reduced by ~300 lines
- [ ] All integration tests pass
- [ ] No performance regression
