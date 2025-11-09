# Implementation Plan: Native Reasoning Model Support

**Date:** 2025-11-08
**Status:** Planning
**Agent:** Should complete milestones sequentially with human review between each

---

## Overview

Add first-class support for reasoning models (OpenAI o1/o3/GPT-5, Anthropic Claude 3.7, DeepSeek) by:
1. Exposing OpenAI-compatible `reasoning_effort` parameter in requests
2. Dynamically detecting model reasoning support via `litellm.supports_reasoning()`
3. Converting provider-specific `reasoning_content` streams to our unified `reasoning_event` format
4. Maintaining OpenAI API compatibility while providing enhanced reasoning metadata

This enables clients to:
- Control reasoning depth via standard `reasoning_effort` parameter
- Discover which models support reasoning via `/v1/models` endpoint
- Get consistent reasoning events regardless of provider (Anthropic, DeepSeek, etc.)

---

## Background: LiteLLM Reasoning Response Structure

### What We Discovered (Testing Results)

Some models return a `reasoning_content` field in their streaming deltas:

```json
{
  "choices": [{
    "delta": {
      "reasoning_content": "To calculate 7 * 8...",  // ← Streamed incrementally
      "content": "",                                  // ← Empty during reasoning
      "role": "assistant"
    }
  }]
}
```

**Streaming Behavior when `reasoning_content` exists:**
- Early chunks: `reasoning_content` populated, `content` empty
- Later chunks: `content` populated, `reasoning_content` stops
- **Pattern:** Reasoning completes BEFORE content starts (clear separation)

**Other models don't have this field:**
- Some models only have `content` field
- Their reasoning (if any) appears inline in the content
- No separate `reasoning_content` field

### Key Insight

**LiteLLM provides:**
- ✅ Unified request parameter: `reasoning_effort` works for all reasoning models
- ❌ Response structure varies: Some models populate `reasoning_content`, others don't

**Our provider-agnostic approach:**
- **If `reasoning_content` exists in chunk:** Buffer it
- **When `reasoning_content` stops and `content` starts:** Emit single `reasoning_event`
- **If no `reasoning_content`:** Pass through content as normal
- **Works regardless of model/provider**

### OpenAI's Official API

**Official `reasoning_effort` parameter:**
```python
# From OpenAI SDK (openai/types/chat/completion_create_params.py)
reasoning_effort: Optional[ReasoningEffort] | Omit = omit
# Type: Literal["minimal", "low", "medium", "high"]
```

**Documentation excerpt:**
> Constrains effort on reasoning for reasoning models. Currently supported values are
> `minimal`, `low`, `medium`, and `high`. Reducing reasoning effort can result in faster
> responses and fewer tokens used on reasoning in a response.

---

## Our Approach: Convert to Unified `reasoning_event`

### Why Not Expose `reasoning_content` Directly?

1. **Not OpenAI-compatible** (not in official OpenAI SDK)
2. **Inconsistent** (some models have it, others don't)
3. **We already have a better system:** `reasoning_event`

### Our Existing Architecture

**Current `reasoning_event` (in `OpenAIDelta`):**
```python
# api/openai_protocol.py
class OpenAIDelta(BaseModel):
    reasoning_event: ReasoningEvent | None = None  # ← Already implemented
```

```typescript
// client/src/types/openai.ts
export interface Delta {
  reasoning_event?: ReasoningEvent;  // ← Client already handles this
}
```

**Event Types:**
- `ITERATION_START` - ReasoningAgent iterations
- `PLANNING` - ReasoningAgent plans
- `TOOL_EXECUTION_START` - Tool calls
- `TOOL_RESULT` - Tool results
- `ITERATION_COMPLETE` - Iteration done
- `REASONING_COMPLETE` - Final synthesis
- `ERROR` - Errors

### New Approach: Add `EXTERNAL_REASONING`

**Goal:** Convert provider-specific `reasoning_content` → our unified `reasoning_event` format

**New event type:**
```python
class ReasoningEventType(str, Enum):
    # ... existing types
    EXTERNAL_REASONING = "external_reasoning"  # ← Model's native reasoning
```

**Conversion logic:**
1. **Buffer** `reasoning_content` chunks as they arrive
2. **Detect transition** when reasoning stops and content starts
3. **Emit single event** with full reasoning text in metadata
4. **Continue** with regular content streaming

**Benefits:**
- ✅ Consistent client experience (all reasoning via `reasoning_event`)
- ✅ Works with existing UI (already handles reasoning_event)
- ✅ No client changes needed
- ✅ OpenAI-compatible (reasoning_event uses `extra='allow'`)
- ✅ Clear semantics: "external model did reasoning"

---

## Milestones

### Milestone 1: Add OpenAI-Compatible `reasoning_effort` Parameter

**Goal:** Add `reasoning_effort` to request models, matching OpenAI's API spec exactly.

**Success Criteria:**
- `OpenAIChatRequest` has `reasoning_effort` field with correct type
- Parameter is passed through to `litellm.acompletion()` unchanged
- OpenAI SDK clients can send `reasoning_effort` transparently
- Non-reasoning requests work unchanged (backward compatible)

**Key Changes:**

1. **Update `api/openai_protocol.py`:**
```python
class OpenAIChatRequest(BaseModel):
    # ... existing fields
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
```

2. **Update client types `client/src/types/openai.ts`:**
```typescript
export interface ChatCompletionRequest {
  // ... existing fields
  reasoning_effort?: 'minimal' | 'low' | 'medium' | 'high';
}
```

3. **Verify passthrough in executors:**
   - PassthroughExecutor already passes all request fields to litellm
   - ReasoningAgent needs to pass it through (or ignore for its own reasoning)

**Testing Strategy:**
- Unit tests: Validate `reasoning_effort` accepts only valid values
- Integration tests: Mock litellm response, verify parameter passed through
- Manual test: Call API with `reasoning_effort`, check logs show litellm received it

**Dependencies:** None (pure request model change)

**Risk Factors:**
- None (additive change, optional parameter)

---

### Milestone 2: Add `supports_reasoning` to `/v1/models` Endpoint

**Goal:** Dynamically detect and expose which models support reasoning via `/v1/models`.

**Success Criteria:**
- `/v1/models` response includes `supports_reasoning: bool` for each model
- Uses `litellm.supports_reasoning(model_id)` for dynamic detection
- Non-reasoning models show `supports_reasoning: false`
- Reasoning models (o3-mini, claude-3-7, etc.) show `supports_reasoning: true`

**Key Changes:**

1. **Update `api/openai_protocol.py` - ModelInfo:**
```python
class ModelInfo(BaseModel):
    model_config = ConfigDict(extra='allow')

    id: str
    object: str = "model"
    created: int
    owned_by: str
    supports_reasoning: bool | None = None  # ← New field
```

2. **Update `api/main.py` - list_models endpoint:**
```python
import litellm

@app.get("/v1/models")
async def list_models(...) -> ModelsResponse:
    # ... existing code to fetch models from LiteLLM

    models_data = [
        ModelInfo(
            id=model["id"],
            created=model.get("created", int(time.time())),
            owned_by=model.get("owned_by", "litellm"),
            supports_reasoning=litellm.supports_reasoning(model["id"]),  # ← Dynamic check
        )
        for model in data.get("data", [])
    ]

    return ModelsResponse(data=models_data)
```

3. **Update client types `client/src/types/openai.ts`:**
```typescript
export interface ModelInfo {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  supports_reasoning?: boolean;  // ← New field
}
```

**Testing Strategy:**
- Unit tests: Mock LiteLLM response, verify supports_reasoning calculated correctly
- Integration tests:
  - Test with known reasoning model (mock returns true)
  - Test with non-reasoning model (mock returns false)
  - Verify client can parse new field
- Manual verification:
  - Call `/v1/models`, check o3-mini/claude-3-7 show `supports_reasoning: true`
  - Check gpt-4o shows `supports_reasoning: false`

**Dependencies:** Milestone 1 (not strictly required, but logical order)

**Risk Factors:**
- `litellm.supports_reasoning()` might not recognize all reasoning models
- Solution: Test with multiple models, document any mismatches

---

### Milestone 3: Add `EXTERNAL_REASONING` Event Type

**Goal:** Define new reasoning event type for external model reasoning.

**Success Criteria:**
- `ReasoningEventType` enum has `EXTERNAL_REASONING` value
- Type is available in Python and TypeScript
- Documentation explains when it's used vs other event types

**Key Changes:**

1. **Update `api/reasoning_models.py`:**
```python
class ReasoningEventType(str, Enum):
    """
    Types of reasoning events for streaming.

    Each event type corresponds to reasoning process events:
    - ITERATION_START: Beginning of a reasoning step (ReasoningAgent)
    - PLANNING: Generated reasoning plan (ReasoningAgent)
    - TOOL_EXECUTION_START: Starting tool execution (ReasoningAgent)
    - TOOL_RESULT: Tool execution completed (ReasoningAgent)
    - ITERATION_COMPLETE: Reasoning step finished (ReasoningAgent)
    - REASONING_COMPLETE: Final synthesis completed (ReasoningAgent)
    - EXTERNAL_REASONING: Model's native reasoning (Anthropic/DeepSeek/etc.)
    - ERROR: Error occurred during reasoning
    """

    ITERATION_START = "iteration_start"
    PLANNING = "planning"
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_RESULT = "tool_result"
    ITERATION_COMPLETE = "iteration_complete"
    REASONING_COMPLETE = "reasoning_complete"
    EXTERNAL_REASONING = "external_reasoning"  # ← NEW
    ERROR = "error"
```

2. **Update `client/src/types/openai.ts`:**
```typescript
export enum ReasoningEventType {
  IterationStart = 'iteration_start',
  Planning = 'planning',
  ToolExecutionStart = 'tool_execution_start',
  ToolResult = 'tool_result',
  IterationComplete = 'iteration_complete',
  ReasoningComplete = 'reasoning_complete',
  ExternalReasoning = 'external_reasoning',  // ← NEW
  Error = 'error',
}
```

3. **Add documentation comment:**
```python
# When EXTERNAL_REASONING is used:
# - Emitted once when model completes native reasoning (Anthropic/DeepSeek)
# - metadata["thought"] contains full reasoning content
# - metadata["provider"] indicates which provider (e.g., "anthropic")
# - step_iteration is always 1 (single reasoning phase)
```

**Testing Strategy:**
- Unit tests: Verify enum value exists and serializes correctly
- Type checking: Verify TypeScript compilation passes
- No runtime tests needed (just adding enum value)

**Dependencies:** None (pure enum addition)

**Risk Factors:** None (additive change)

---

### Milestone 4: Implement Reasoning Content Buffering in PassthroughExecutor

**Goal:** Buffer `reasoning_content` chunks from LiteLLM and convert to single `EXTERNAL_REASONING` event.

**Success Criteria:**
- PassthroughExecutor detects `reasoning_content` in LiteLLM chunks
- All reasoning chunks are buffered until reasoning completes
- Single `EXTERNAL_REASONING` event emitted when transition to content happens
- Event contains full reasoning text in metadata
- Regular content streaming continues after reasoning event
- Works seamlessly with models that don't have reasoning_content (OpenAI)

**Key Changes:**

1. **Add buffering state to `api/executors/passthrough.py`:**
```python
class PassthroughExecutor(BaseExecutor):
    def __init__(self, parent_span: trace.Span | None, check_disconnected: Callable[[], bool]):
        super().__init__(parent_span, check_disconnected)
        self._reasoning_buffer: list[str] = []  # ← Buffer for reasoning_content
        self._reasoning_active = False
        self._reasoning_event_sent = False
```

2. **Detect and buffer reasoning in `execute_stream()`:**
```python
async def execute_stream(self, request: OpenAIChatRequest) -> AsyncGenerator[str]:
    """Stream response from LiteLLM, converting reasoning_content to events."""

    async for litellm_chunk in litellm.acompletion(...):
        # Convert to our format
        chunk = convert_litellm_to_stream_response(litellm_chunk)

        # Check if chunk has reasoning_content
        if chunk.choices:
            delta = chunk.choices[0].delta

            # Detect reasoning content
            has_reasoning = (
                hasattr(delta, 'reasoning_content')
                and delta.reasoning_content is not None
                and delta.reasoning_content != ""
            )

            has_content = (
                hasattr(delta, 'content')
                and delta.content is not None
                and delta.content != ""
            )

            # Buffer reasoning
            if has_reasoning:
                self._reasoning_buffer.append(delta.reasoning_content)
                self._reasoning_active = True
                continue  # Don't emit this chunk

            # Transition: reasoning → content
            if self._reasoning_active and has_content and not self._reasoning_event_sent:
                # Emit reasoning event
                reasoning_event = ReasoningEvent(
                    type=ReasoningEventType.EXTERNAL_REASONING,
                    step_iteration=1,
                    metadata={
                        "thought": "".join(self._reasoning_buffer),
                        "provider": self._detect_provider(request.model),
                    }
                )

                # Create chunk with reasoning event
                event_chunk = self._create_event_chunk(
                    chunk_id=chunk.id,
                    model=chunk.model,
                    created=chunk.created,
                    reasoning_event=reasoning_event,
                )

                yield create_sse(event_chunk)
                self._reasoning_event_sent = True
                self._reasoning_active = False

            # Emit regular content chunk
            if has_content:
                yield create_sse(chunk)
```

3. **Add helper methods:**
```python
def _detect_provider(self, model: str) -> str:
    """Detect provider from model name."""
    if "anthropic" in model.lower() or "claude" in model.lower():
        return "anthropic"
    elif "deepseek" in model.lower():
        return "deepseek"
    else:
        return "unknown"

def _create_event_chunk(
    self,
    chunk_id: str,
    model: str,
    created: int,
    reasoning_event: ReasoningEvent,
) -> OpenAIStreamResponse:
    """Create a chunk with only a reasoning_event in delta."""
    return OpenAIStreamResponse(
        id=chunk_id,
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[
            OpenAIStreamChoice(
                index=0,
                delta=OpenAIDelta(reasoning_event=reasoning_event),
                finish_reason=None,
            )
        ],
    )
```

**Testing Strategy:**

**Unit Tests:**
```python
# tests/unit_tests/test_executors/test_passthrough_reasoning.py

async def test_passthrough_buffers_reasoning_content():
    """Verify reasoning_content chunks are buffered and converted."""
    # Mock litellm to return reasoning chunks then content chunks
    mock_chunks = [
        # Reasoning chunks
        create_mock_chunk(reasoning_content="First thought"),
        create_mock_chunk(reasoning_content=" about the problem"),
        create_mock_chunk(reasoning_content=". Let me solve it."),
        # Content chunks (reasoning done)
        create_mock_chunk(content="The answer is 42."),
    ]

    executor = PassthroughExecutor(...)
    chunks = [chunk async for chunk in executor.execute_stream(request)]

    # Should have: 1 reasoning_event chunk + 1 content chunk
    assert len(chunks) == 2

    # First chunk should have reasoning_event
    event_chunk = parse_sse(chunks[0])
    assert event_chunk["choices"][0]["delta"]["reasoning_event"]["type"] == "external_reasoning"
    assert "First thought about the problem. Let me solve it." in \
        event_chunk["choices"][0]["delta"]["reasoning_event"]["metadata"]["thought"]

    # Second chunk should have regular content
    content_chunk = parse_sse(chunks[1])
    assert content_chunk["choices"][0]["delta"]["content"] == "The answer is 42."


async def test_passthrough_handles_no_reasoning_content():
    """Verify models without reasoning_content work unchanged (OpenAI)."""
    mock_chunks = [
        create_mock_chunk(content="Regular"),
        create_mock_chunk(content=" response"),
    ]

    executor = PassthroughExecutor(...)
    chunks = [chunk async for chunk in executor.execute_stream(request)]

    # Should just pass through content chunks
    assert len(chunks) == 2
    assert all("reasoning_event" not in parse_sse(c)["choices"][0]["delta"] for c in chunks)


async def test_passthrough_handles_mixed_reasoning_and_content():
    """Edge case: reasoning_content and content in same chunk (shouldn't happen but be defensive)."""
    # Test defensive handling
```

**Integration Tests:**
```python
# tests/integration_tests/test_reasoning_integration.py

async def test_anthropic_reasoning_converted_to_event(mock_litellm_anthropic_reasoning):
    """End-to-end test with mocked Anthropic reasoning response."""
    # Mock LiteLLM to return Anthropic-style reasoning chunks
    # Verify API returns proper reasoning_event

async def test_openai_no_reasoning_passes_through(mock_litellm_openai_response):
    """End-to-end test with OpenAI o3-mini (no reasoning_content)."""
    # Verify API works normally with OpenAI models
```

**Manual Test with Real API:**
```bash
# Test with Anthropic Claude 3.7
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-3-7-sonnet-20250219",
    "messages": [{"role": "user", "content": "What is 7 * 8? Show reasoning."}],
    "reasoning_effort": "low",
    "stream": true
  }'

# Expected: See reasoning_event chunk before content chunks
```

**Dependencies:** Milestones 1-3 (needs enum, parameter support)

**Risk Factors:**
- **Edge cases in buffering logic:**
  - What if reasoning_content appears after content starts? (Defensive: treat as new reasoning phase?)
  - What if chunks have both reasoning_content AND content? (Defensive: buffer reasoning, emit content separately)
  - What if stream ends during reasoning? (Emit buffered reasoning at end)
- **Solution:** Extensive edge case testing, defensive programming

---

### Milestone 5: Update Mock Fixtures for Testing

**Goal:** Create mock LiteLLM responses that match real Anthropic reasoning structure for testing.

**Success Criteria:**
- Mock fixtures mirror real LiteLLM chunk structure (from notebook tests)
- Separate fixtures for Anthropic (with reasoning_content) and OpenAI (without)
- Easy to use in unit and integration tests
- Cover edge cases (empty buffers, transitions, errors)

**Key Changes:**

1. **Create `tests/fixtures/litellm_reasoning_responses.py`:**
```python
"""Mock LiteLLM responses for reasoning models based on real API responses."""

from litellm.types.utils import ModelResponseStream
from api.openai_protocol import OpenAIStreamResponse


def create_anthropic_reasoning_chunks() -> list[dict]:
    """
    Create mock Anthropic reasoning response chunks.

    Based on real response structure from claude-3-7-sonnet-20250219.
    Mimics the pattern: reasoning_content chunks → content chunks
    """
    return [
        # Reasoning phase (chunks 0-3)
        {
            "id": "chatcmpl-test123",
            "created": 1234567890,
            "model": "claude-3-7-sonnet-20250219",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {
                    "reasoning_content": "To solve",
                    "thinking_blocks": [{"type": "thinking", "thinking": "To solve"}],
                    "content": "",
                    "role": "assistant",
                },
                "finish_reason": None,
            }],
        },
        {
            "id": "chatcmpl-test123",
            "created": 1234567890,
            "model": "claude-3-7-sonnet-20250219",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {
                    "reasoning_content": " this problem",
                    "thinking_blocks": [{"type": "thinking", "thinking": " this problem"}],
                    "content": "",
                    "role": None,
                },
                "finish_reason": None,
            }],
        },
        # ... more reasoning chunks

        # Content phase (chunks 4+)
        {
            "id": "chatcmpl-test123",
            "created": 1234567890,
            "model": "claude-3-7-sonnet-20250219",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {
                    "reasoning_content": None,  # No more reasoning
                    "thinking_blocks": None,
                    "content": "The answer",
                    "role": None,
                },
                "finish_reason": None,
            }],
        },
        # ... more content chunks
    ]


def create_openai_reasoning_chunks() -> list[dict]:
    """
    Create mock OpenAI o3-mini response chunks.

    Based on real response: NO reasoning_content field, only content.
    """
    return [
        {
            "id": "chatcmpl-openai123",
            "created": 1234567890,
            "model": "o3-mini",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": "To solve",
                    "role": "assistant",
                },
                "finish_reason": None,
            }],
        },
        # ... more content chunks (no reasoning_content)
    ]
```

2. **Update existing test patterns:**
```python
# tests/integration_tests/litellm_mocks.py

from tests.fixtures.litellm_reasoning_responses import (
    create_anthropic_reasoning_chunks,
    create_openai_reasoning_chunks,
)

def mock_anthropic_reasoning():
    """Mock for Anthropic reasoning model with reasoning_content."""
    return create_anthropic_reasoning_chunks()

def mock_openai_reasoning():
    """Mock for OpenAI reasoning model (no reasoning_content)."""
    return create_openai_reasoning_chunks()
```

**Testing Strategy:**
- Validate mock structure matches real API responses (from notebook)
- Use mocks in all Milestone 4 tests
- Document any deviations from real responses

**Dependencies:** Milestone 4 (tests need the implementation)

**Risk Factors:**
- Mock structure might drift from real API
- Solution: Keep notebook test results as reference, update mocks if API changes

---

### Milestone 6: Documentation and Manual Verification

**Goal:** Document the new reasoning support and verify it works end-to-end with real models.

**Success Criteria:**
- API documentation updated to describe `reasoning_effort` parameter
- Architecture docs explain reasoning_content → reasoning_event conversion
- Manual testing confirms it works with Anthropic Claude 3.7
- Manual testing confirms OpenAI models still work (no regression)

**Key Changes:**

1. **Update `CLAUDE.md` - Architecture section:**
```markdown
## Reasoning Model Support

The API supports native reasoning models via the `reasoning_effort` parameter:

### Request Parameter
- `reasoning_effort`: Controls model's internal reasoning depth
- Values: `"minimal"`, `"low"`, `"medium"`, `"high"`
- Compatible with: OpenAI (o1/o3/GPT-5), Anthropic (Claude 3.7+), DeepSeek

### Response Format
Models with native reasoning (Anthropic, DeepSeek) return reasoning content
separately from their final response. We convert this to a unified format:

**Anthropic/DeepSeek Behavior:**
1. Model streams `reasoning_content` (hidden thinking)
2. We buffer all reasoning chunks
3. When reasoning completes, we emit single `EXTERNAL_REASONING` event
4. Regular content streaming continues

**OpenAI Behavior:**
- No separate reasoning content (reasoning shown inline in response)
- No reasoning events emitted
- `reasoning_effort` still controls internal reasoning depth

### Discovery
Check `/v1/models` for `supports_reasoning: true/false` to know which
models support the reasoning_effort parameter.
```

2. **Update API endpoint documentation:**
```markdown
### POST /v1/chat/completions

**New Parameters:**
- `reasoning_effort` (optional): `"minimal" | "low" | "medium" | "high"`
  - Controls reasoning depth for reasoning models
  - Ignored for non-reasoning models

**Response - Reasoning Events:**
For models with native reasoning (Anthropic, DeepSeek), you may receive
a `reasoning_event` in the delta before regular content:

```json
{
  "choices": [{
    "delta": {
      "reasoning_event": {
        "type": "external_reasoning",
        "step_iteration": 1,
        "metadata": {
          "thought": "Full reasoning content...",
          "provider": "anthropic"
        }
      }
    }
  }]
}
```

Then regular content follows in subsequent chunks.
```

3. **Optional manual verification (if needed):**
```bash
# Example: Test model with reasoning_content field
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-7-sonnet-20250219",
    "messages": [{"role": "user", "content": "Solve 15 * 24 step by step"}],
    "reasoning_effort": "low",
    "stream": true
  }' | grep reasoning_event

# Example: Check model support
curl http://localhost:8000/v1/models | jq '.data[] | {id, supports_reasoning}'
```

**Note:** All actual tests use mocked LiteLLM responses. Manual testing is optional for verification only.

**Testing Strategy:**
- All unit and integration tests use mocked LiteLLM responses
- Mocks match real API response structures (from notebook testing)
- Documentation updated based on implementation
- Manual testing optional for final verification

**Dependencies:** All previous milestones

**Risk Factors:**
- Real API behavior might differ from testing
- Solution: Document any discrepancies, adjust implementation if needed

---

## Implementation Notes

### Key Design Decisions

1. **Why buffer reasoning_content instead of streaming it incrementally?**
   - Consistent with our existing `reasoning_event` architecture
   - Matches ReasoningAgent's event-based approach
   - Client gets complete reasoning in one event (simpler to display)
   - Still maintains streaming for regular content

2. **Why detect provider from model name?**
   - LiteLLM doesn't provide provider in response metadata
   - Model name is reliable indicator (e.g., "anthropic/", "deepseek/")
   - Useful for debugging and client display
   - Optional metadata, not critical to functionality

### Testing Philosophy

- **Mock structure fidelity:** Mocks must match real API responses exactly (use notebook results as reference)
- **Edge case coverage:** Test all buffering edge cases (empty, partial, errors, transitions)
- **Integration realism:** Use realistic model names and chunk sequences in mocks
- **No real API calls in tests:** All tests use mocked LiteLLM responses

### Breaking Changes

**None** - All changes are additive:
- New optional `reasoning_effort` parameter (clients can ignore)
- New optional `supports_reasoning` field in models (clients can ignore)
- New optional `reasoning_event` in delta (clients already handle this)
- Existing behavior unchanged for models without reasoning

---

## Success Metrics

After completion, verify:
- [ ] Can send `reasoning_effort` parameter to API
- [ ] `/v1/models` shows `supports_reasoning` for each model
- [ ] Models with `reasoning_content` return `EXTERNAL_REASONING` event before content
- [ ] Event contains full reasoning text in metadata
- [ ] Models without `reasoning_content` work unchanged (no reasoning events)
- [ ] All tests pass (unit + integration with mocked LiteLLM responses)
- [ ] Documentation is complete and accurate

---

## References

**LiteLLM Documentation:**
- https://docs.litellm.ai/docs/reasoning_content
- https://docs.litellm.ai/docs/providers/anthropic
- https://docs.litellm.ai/docs/providers/openai

**OpenAI Documentation:**
- https://platform.openai.com/docs/guides/reasoning
- https://platform.openai.com/docs/api-reference/chat/create (reasoning_effort parameter)

**Testing Results:**
- `notebooks/test_reasoning_responses.ipynb` - Real API response structures for reference
- Some models (Anthropic, DeepSeek): `reasoning_content` streamed separately, then `content`
- Other models: Only `content` field (reasoning inline if present)

**Related Files:**
- `api/openai_protocol.py` - Request/response models
- `api/reasoning_models.py` - ReasoningEvent types
- `api/executors/passthrough.py` - LiteLLM streaming logic
- `client/src/types/openai.ts` - TypeScript types
