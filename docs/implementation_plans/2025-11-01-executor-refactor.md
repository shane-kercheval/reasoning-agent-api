# Executor Architecture Refactor

**Date:** 2025-11-01
**Status:** In Progress
**Goal:** Eliminate inefficiencies in executor architecture, enforce single-use pattern, and centralize common logic in base class

## Problems Identified

### 1. No Single-Use Enforcement
**Current state:**
- Both executors manually call `self._reset_buffer()` at start of `execute_stream()`
- Suggests executors could be reused, but this is unintended
- No runtime protection against accidental reuse

**Impact:**
- Potential bugs from executor reuse
- Unclear lifecycle contract

### 2. Duplicate Disconnection Checks (Critical Inefficiency)
**Current state:** Three levels of checking in ReasoningAgent:
1. In `execute_stream()` (reasoning_agent.py:224)
2. In `_core_reasoning_process()` (reasoning_agent.py:291)
3. In `_stream_final_synthesis()` (reasoning_agent.py:572)

**Impact:**
- Every event gets checked 2-3 times
- Wasteful async function calls and conditionals

### 3. SSE Roundtrip Inefficiency (Critical)
**Current state in PassthroughExecutor:**
```python
chunk_dict = chunk.model_dump()        # dict
sse_chunk = create_sse(chunk_dict)     # dict → "data: {...}\n\n"
self._buffer_chunk(sse_chunk)          # "data: {...}\n\n" → parse JSON → extract
yield sse_chunk
```

**Data flow:**
```
dict → SSE string → parse SSE → parse JSON → extract content
```

**Impact:**
- Unnecessary serialization/deserialization cycle
- JSON parsing overhead for every chunk

### 4. Duplicated Logic Across Executors
Both executors implement:
- SSE conversion (`create_sse()`)
- Content buffering
- Parent span attribute setting (input, output, metadata)
- Disconnection checking

**Impact:**
- Code duplication
- Inconsistent behavior risk
- Harder to maintain

### 5. Unclear API Contract
**Current:**
```python
executor = PassthroughExecutor()  # Empty constructor
async for chunk in executor.execute_stream(
    request,
    parent_span=span,  # Request-specific params in execute_stream
    check_disconnected=http_request.is_disconnected,
):
```

**Issue:**
- Request-specific params passed to method, not constructor
- Doesn't align with per-request instantiation pattern

## Proposed Solution

### New Architecture

**Key principle:** Subclasses yield structured data (`OpenAIStreamResponse` or `dict`), base class handles SSE conversion, buffering, disconnection checking, and span management.

```python
class BaseExecutor(ABC):
    def __init__(
        self,
        parent_span: trace.Span | None = None,
        check_disconnected: Callable[[], bool] | None = None,
    ):
        """Initialize executor with request-specific params."""
        self._content_buffer: list[str] = []
        self._parent_span = parent_span
        self._check_disconnected_callback = check_disconnected
        self._executed = False

    async def execute_stream(
        self,
        request: OpenAIChatRequest,
    ) -> AsyncGenerator[str]:
        """
        Public API - yields SSE strings.

        Handles:
        - Single-use enforcement
        - Span attribute setting (input, output, metadata)
        - Disconnection checking (single location!)
        - Content buffering (no SSE roundtrip!)
        - SSE conversion
        """
        if self._executed:
            raise RuntimeError("Executor can only be used once")
        self._executed = True

        # Set input attributes on parent span if provided
        if self._parent_span:
            self._set_span_attributes(request, self._parent_span)

        async for response in self._execute_stream(request):
            # SINGLE disconnection check (eliminates duplicates!)
            if self._check_disconnected_callback:
                if await self._check_disconnected_callback():
                    raise asyncio.CancelledError("Client disconnected")

            # Buffer content directly from structured response (no SSE roundtrip!)
            if response.choices and response.choices[0].delta.content:
                # Only buffer if NOT a reasoning event
                if not getattr(response.choices[0].delta, 'reasoning_event', None):
                    self._content_buffer.append(response.choices[0].delta.content)

            # Convert to SSE and yield
            yield create_sse(response)

        # Yield [DONE] marker
        yield SSE_DONE

        # Set output attribute after streaming completes
        if self._parent_span:
            complete_output = self.get_buffered_content()
            if complete_output:
                self._parent_span.set_attribute(SpanAttributes.OUTPUT_VALUE, complete_output)

    @abstractmethod
    async def _execute_stream(
        self,
        request: OpenAIChatRequest,
    ) -> AsyncGenerator[OpenAIStreamResponse | dict]:
        """
        Subclasses implement - yields structured responses (NOT SSE strings).

        Can yield either:
        - OpenAIStreamResponse objects
        - dict objects (converted to OpenAIStreamResponse by base)
        """
        pass

    def get_buffered_content(self) -> str:
        """Get complete buffered assistant content after streaming."""
        return "".join(self._content_buffer)

    @abstractmethod
    def _set_span_attributes(
        self,
        request: OpenAIChatRequest,
        span: trace.Span,
    ) -> None:
        """Set input and metadata attributes on span (executor-specific)."""
        pass
```

### Subclass Simplification

**PassthroughExecutor:**
```python
class PassthroughExecutor(BaseExecutor):
    """Executes passthrough streaming."""

    async def _execute_stream(
        self,
        request: OpenAIChatRequest,
    ) -> AsyncGenerator[dict]:
        """Stream LiteLLM responses - base handles rest."""
        with tracer.start_as_current_span(...) as span:
            # Inject trace context for LiteLLM
            carrier: dict[str, str] = {}
            propagate.inject(carrier)

            # Stream from LiteLLM
            stream = await litellm.acompletion(...)

            async for chunk in stream:
                # Track usage in span
                if chunk_usage := getattr(chunk, 'usage', None):
                    span.set_attribute("llm.token_count.prompt", chunk_usage.prompt_tokens)
                    # ...

                # Yield dict - base converts to SSE, buffers, checks disconnection
                yield chunk.model_dump()
```

**ReasoningAgent:**
```python
class ReasoningAgent(BaseExecutor):
    def __init__(
        self,
        tools: list[Tool],
        prompt_manager: PromptManager,
        parent_span: trace.Span | None = None,
        check_disconnected: Callable[[], bool] | None = None,
    ):
        super().__init__(parent_span, check_disconnected)
        self.tools = {tool.name: tool for tool in tools}
        self.prompt_manager = prompt_manager
        # ... rest of reasoning-specific init

    async def _execute_stream(
        self,
        request: OpenAIChatRequest,
    ) -> AsyncGenerator[OpenAIStreamResponse]:
        """Yield OpenAIStreamResponse objects - base handles rest."""
        # _core_reasoning_process already yields OpenAIStreamResponse!
        async for response in self._core_reasoning_process(request):
            yield response
```

### Main.py Usage

```python
# Passthrough
executor = PassthroughExecutor(
    parent_span=span,
    check_disconnected=http_request.is_disconnected,
)
async for chunk in executor.execute_stream(llm_request):
    yield chunk

# Reasoning
reasoning_agent = ReasoningAgent(
    tools=tools,
    prompt_manager=prompt_manager,
    parent_span=span,
    check_disconnected=http_request.is_disconnected,
)
async for chunk in reasoning_agent.execute_stream(llm_request):
    yield chunk
```

## Benefits

| Feature | Before | After |
|---------|--------|-------|
| **Single-use enforcement** | Manual `_reset_buffer()` | ✅ Automatic via `_executed` flag |
| **Disconnection checks** | 2-3 checks per event | ✅ 1 check in base class |
| **Buffering logic** | Duplicated + SSE roundtrip | ✅ Centralized, no roundtrip |
| **SSE conversion** | Each subclass | ✅ Base handles it |
| **Parent span management** | Duplicated logic | ✅ Base handles it |
| **API clarity** | 3 params to execute_stream | ✅ 1 param (request only) |
| **Subclass complexity** | High (SSE, buffering, etc.) | ✅ Low (just yield data) |

## Implementation Plan

### Phase 1: Refactor BaseExecutor ✅
- [x] Add `__init__` with `parent_span` and `check_disconnected` params
- [x] Add `_executed` flag and runtime check
- [x] Move disconnection checking to `execute_stream` wrapper
- [x] Move content buffering to `execute_stream` wrapper
- [x] Move SSE conversion to `execute_stream` wrapper
- [x] Add `SSE_DONE` yield at end
- [x] Move parent span output attribute setting
- [x] Change `execute_stream` to call abstract `_execute_stream`
- [x] Make `_set_span_attributes` abstract (executor-specific)
- [x] Remove `extract_content_from_sse_chunk` (no longer needed)
- [x] Remove `_buffer_chunk` (logic moved to wrapper)
- [x] Remove `_reset_buffer` (replaced by `_executed` flag)
- [x] Remove `_check_disconnection` (moved to wrapper)

### Phase 2: Update PassthroughExecutor ✅
- [x] Rename `execute_stream` to `_execute_stream`
- [x] Change signature to return `AsyncGenerator[dict]`
- [x] Remove manual `_reset_buffer()` call
- [x] Remove `parent_span` and `check_disconnected` params
- [x] Remove manual `_check_disconnection()` calls
- [x] Remove manual `_buffer_chunk()` calls
- [x] Remove manual `create_sse()` calls
- [x] Remove manual parent span output attribute setting
- [x] Implement `_set_span_attributes` (passthrough-specific)
- [x] Just yield `chunk.model_dump()` - base handles rest

### Phase 3: Update ReasoningAgent ✅
- [x] Update `__init__` to accept `parent_span` and `check_disconnected`
- [x] Pass to `super().__init__(...)`
- [x] Rename `execute_stream` to `_execute_stream`
- [x] Remove manual `_reset_buffer()` call
- [x] Remove `parent_span` and `check_disconnected` params from `_execute_stream`
- [x] Remove duplicate `_check_disconnection()` call from wrapper
- [x] Remove manual buffering logic (content vs reasoning_event check)
- [x] Remove manual `create_sse()` calls
- [x] Remove manual `SSE_DONE` yield
- [x] Remove manual parent span output attribute setting
- [x] Move `_set_span_attributes` from private method to override
- [x] Update `_core_reasoning_process` to remove `parent_span` param
- [x] Update `_core_reasoning_process` to remove `check_disconnected` param
- [x] Remove all internal `_check_disconnection()` calls
- [x] Just yield `OpenAIStreamResponse` objects - base handles rest

### Phase 4: Update main.py ✅
- [x] Update passthrough executor instantiation to pass params to constructor
- [x] Update reasoning agent instantiation to pass params to constructor
- [x] Remove params from `execute_stream()` calls
- [x] Remove manual assistant content buffering (use executor's buffer instead)
- [x] Update conversation storage to use `executor.get_buffered_content()`

### Phase 5: Testing ✅
- [x] Run unit tests
- [x] Run integration tests
- [x] Verify no regressions
- [x] Verify single-use enforcement works
- [x] Verify disconnection checking still works

## Testing Strategy

### Unit Tests
- **BaseExecutor:**
  - Single-use enforcement (calling execute_stream twice raises RuntimeError)
  - Disconnection checking integration
  - Content buffering (excluding reasoning events)
  - SSE conversion

- **PassthroughExecutor:**
  - Verify `_execute_stream` yields dicts
  - Verify no manual SSE/buffering/disconnection logic

- **ReasoningAgent:**
  - Verify `_execute_stream` yields OpenAIStreamResponse
  - Verify reasoning events not buffered
  - Verify final synthesis content is buffered

### Integration Tests
- Existing tests should pass with minimal changes
- Verify end-to-end streaming still works
- Verify conversation storage still works
- Verify tracing still works

## Migration Notes

### Breaking Changes
- `execute_stream` signature changed:
  - **Before:** `execute_stream(request, parent_span=None, check_disconnected=None)`
  - **After:** `execute_stream(request)`
- Executors now require params in constructor:
  - **Before:** `PassthroughExecutor()`
  - **After:** `PassthroughExecutor(parent_span=span, check_disconnected=callback)`

### Non-Breaking Changes
- Subclasses now implement `_execute_stream` instead of `execute_stream`
- `_set_span_attributes` now abstract method (executor-specific implementation)

## Rollback Plan

If issues arise:
1. Revert commits in reverse order (main.py → ReasoningAgent → PassthroughExecutor → BaseExecutor)
2. All changes are in executor files + main.py, minimal surface area
3. Tests will catch any regressions immediately

## Success Criteria

- ✅ All tests pass
- ✅ No performance regressions
- ✅ Single-use enforcement verified
- ✅ Disconnection checking works correctly
- ✅ Content buffering works correctly (including reasoning events)
- ✅ SSE format unchanged
- ✅ Tracing unchanged
- ✅ Code duplication eliminated
