# Executor Chunk Standardization

**Date:** 2025-11-02
**Status:** Planning
**Goal:** Standardize LiteLLM chunk conversion across executors using real data validation

## Background

Currently both PassthroughExecutor and ReasoningAgent receive `ModelResponseStream` from LiteLLM but handle conversion differently:
- **PassthroughExecutor**: Yields `chunk.model_dump()` (dict)
- **ReasoningAgent**: Manually constructs `OpenAIStreamResponse` with field mapping

BaseExecutor must handle both types with `isinstance()` checks and duplicate buffering logic.

## Real Data Validation

Captured actual LiteLLM chunks via `scripts/capture_litellm_chunks.py`:

**Chunk Type:** `litellm.types.utils.ModelResponseStream` (Pydantic model)

**Key Observations:**
```python
# Chunk 1 (first content chunk)
{
  "id": "chatcmpl-...",
  "created": 1762102237,
  "model": "gpt-4o-mini",
  "object": "chat.completion.chunk",
  "system_fingerprint": "fp_...",
  "choices": [{
    "index": 0,
    "delta": {
      "content": "1",           #  Has content
      "role": "assistant",      #  Role only in first chunk
      "tool_calls": null,
      "audio": null
    },
    "finish_reason": null
  }],
  "provider_specific_fields": null,
  "citations": null,            # Extra LiteLLM fields
  "service_tier": "default",
  "obfuscation": "..."
}

# Chunk 9 (finish chunk)
{
  "choices": [{
    "delta": {
      "content": null,          #   Content is None
      "role": null
    },
    "finish_reason": "stop"     #  Has finish_reason
  }]
}

# Chunk 10 (usage chunk)
{
  "choices": [{
    "delta": {
      "content": null,          #   Content is None
      "role": null
    },
    "finish_reason": null
  }],
  "usage": {                    #  Usage in model_extra
    "completion_tokens": 8,
    "prompt_tokens": 18,
    "total_tokens": 26
  }
}
```

**Critical Findings:**
1.  `ModelResponseStream` structure matches `OpenAIStreamResponse` fields
2.  `OpenAIStreamResponse` has `model_config = ConfigDict(extra='allow')` - accepts extra fields
3.   `delta.content` can be `None` (finish chunk, usage chunk)
4.   `delta.role` only in first chunk, `None` thereafter
5.  `usage` appears in final chunk's `model_extra`

## Proposed Solution

### 1. Create Centralized Conversion Function

Add to `api/openai_protocol.py`:

```python
def convert_litellm_to_stream_response(
    chunk,  # ModelResponseStream from LiteLLM (no type hint to avoid dependency)
    completion_id: str | None = None,
    created: int | None = None,
) -> OpenAIStreamResponse:
    """
    Convert LiteLLM ModelResponseStream to OpenAIStreamResponse.

    Uses chunk.model_dump() + override pattern for clean conversion.
    Extra LiteLLM fields (citations, obfuscation) are preserved via extra='allow'.

    Args:
        chunk: LiteLLM ModelResponseStream object
        completion_id: Override chunk.id (for ReasoningAgent's consistent ID)
        created: Override chunk.created (for ReasoningAgent's consistent timestamp)

    Returns:
        OpenAIStreamResponse with optional field overrides
    """
    data = chunk.model_dump()

    # Override fields if provided (ReasoningAgent needs consistent IDs across chunks)
    if completion_id is not None:
        data['id'] = completion_id
    if created is not None:
        data['created'] = created

    return OpenAIStreamResponse(**data)
```

**Why this works:**
- LiteLLM's `model_dump()` returns dict with all fields
- `OpenAIStreamResponse(**data)` unpacks dict into constructor
- Extra fields preserved due to `extra='allow'`
- Optional overrides support ReasoningAgent's ID consistency

### 2. Update PassthroughExecutor

**Before:**
```python
async for chunk in stream:
    yield chunk.model_dump()  # Yields dict
```

**After:**
```python
async for chunk in stream:
    yield convert_litellm_to_stream_response(chunk)  # Yields OpenAIStreamResponse
```

### 3. Update ReasoningAgent

**Before (manual field mapping):**
```python
yield OpenAIStreamResponse(
    id=completion_id,
    created=created,
    model=chunk.model,
    choices=[
        OpenAIStreamChoice(
            index=0,
            delta=OpenAIDelta(
                role=choice.delta.role,
                content=choice.delta.content,
                tool_calls=choice.delta.tool_calls,
            ),
            finish_reason=choice.finish_reason,
        ),
    ],
    usage=usage,
)
```

**After (centralized conversion):**
```python
yield convert_litellm_to_stream_response(
    chunk,
    completion_id=completion_id,  # Override for consistency
    created=created,
)
```

### 4. Simplify BaseExecutor

**Remove dict handling:**
```python
# Before
if isinstance(response, OpenAIStreamResponse):
    if response.choices and response.choices[0].delta.content:
        self._content_buffer.append(response.choices[0].delta.content)
elif isinstance(response, dict):
    if choices := response.get("choices"):
        if delta := choices[0].get("delta", {}):
            if content := delta.get("content"):
                self._content_buffer.append(content)

# After (defensive attribute checks for None content)
if response.choices and len(response.choices) > 0:
    delta = response.choices[0].delta
    if delta and delta.content:  # Handles None from finish/usage chunks
        self._content_buffer.append(delta.content)
```

### 5. Update Type Hints

Update `BaseExecutor._execute_stream` return type:

```python
@abstractmethod
async def _execute_stream(
    self,
    request: OpenAIChatRequest,
) -> AsyncGenerator[OpenAIStreamResponse]:  # Remove "| dict"
    """
    Execute streaming request and yield OpenAIStreamResponse objects.

    Subclasses must convert LiteLLM chunks using convert_litellm_to_stream_response().
    """
    pass
```

### 6. Update Mock Objects

**Create real-data-validated mocks in `tests/fixtures/openai_responses.py`:**

```python
def create_mock_litellm_chunk(
    content: str | None = "Hello",
    role: str | None = None,
    finish_reason: str | None = None,
    usage: dict | None = None,
) -> dict:
    """
    Create mock matching real LiteLLM ModelResponseStream structure.

    Validated against scripts/litellm_chunks_captured.json.
    """
    return {
        "id": "chatcmpl-test123",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "object": "chat.completion.chunk",
        "system_fingerprint": "fp_test",
        "choices": [{
            "index": 0,
            "delta": {
                "provider_specific_fields": None,
                "refusal": None,
                "content": content,
                "role": role,
                "function_call": None,
                "tool_calls": None,
                "audio": None,
            },
            "finish_reason": finish_reason,
            "logprobs": None,
        }],
        "provider_specific_fields": None,
        "citations": None,
        "service_tier": "default",
        "obfuscation": "test123",
        "usage": usage,
    }

# Usage examples
MOCK_CONTENT_CHUNK = create_mock_litellm_chunk(content="Hello", role="assistant")
MOCK_FINISH_CHUNK = create_mock_litellm_chunk(content=None, finish_reason="stop")
MOCK_USAGE_CHUNK = create_mock_litellm_chunk(
    content=None,
    usage={
        "completion_tokens": 8,
        "prompt_tokens": 18,
        "total_tokens": 26,
    }
)
```

### 7. Fix Test Imports

**Files to update:**
- `tests/unit_tests/test_reasoning_agent.py`
- `tests/integration_tests/test_reasoning_integration.py`
- `tests/evaluations/test_eval_reasoning_agent.py`
- `tests/fixtures/agents.py`
- `tests/conftest.py`

**Change:**
```python
# Before
from api.reasoning_agent import ReasoningAgent, ReasoningError

# After
from api.executors.reasoning_agent import ReasoningAgent, ReasoningError
```

**Note:** Keep compatibility shim at `api/reasoning_agent.py` for backward compatibility:
```python
"""Backward compatibility shim."""
from api.executors.reasoning_agent import ReasoningAgent, ReasoningError
__all__ = ["ReasoningAgent", "ReasoningError"]
```

## Implementation Plan

### Phase 1: Add Conversion Function 
- [ ] Add `convert_litellm_to_stream_response()` to `api/openai_protocol.py`
- [ ] Add unit tests for conversion edge cases (None content, usage chunks, overrides)

### Phase 2: Update Executors 
- [ ] Update PassthroughExecutor to use conversion function
- [ ] Update ReasoningAgent to use conversion function (both synthesis and reasoning steps)
- [ ] Remove manual field mapping code from ReasoningAgent

### Phase 3: Simplify BaseExecutor 
- [ ] Remove `isinstance(response, dict)` handling
- [ ] Add defensive attribute checks for None values
- [ ] Update type hints to remove `| dict`

### Phase 4: Update Mock Objects 
- [ ] Create `create_mock_litellm_chunk()` helper in test fixtures
- [ ] Add edge case mocks (None content, finish chunk, usage chunk)
- [ ] Reference `scripts/litellm_chunks_captured.json` as source of truth
- [ ] Update existing tests to use new mocks

### Phase 5: Fix Test Imports 
- [ ] Update all test files to import from `api.executors.reasoning_agent`
- [ ] Verify compatibility shim works for any missed imports
- [ ] Run all tests to ensure no import errors

### Phase 6: Testing 
- [ ] Run unit tests (`make non_integration_tests`)
- [ ] Run integration tests (`make integration_tests`)
- [ ] Verify PassthroughExecutor handles all chunk types
- [ ] Verify ReasoningAgent maintains consistent IDs
- [ ] Test edge cases: None content, finish chunks, usage chunks

## Edge Cases to Test

1. **Content is None**
   - Finish chunk: `delta.content = None, finish_reason = "stop"`
   - Usage chunk: `delta.content = None, usage = {...}`

2. **Role handling**
   - First chunk: `delta.role = "assistant", delta.content = "..."`
   - Subsequent chunks: `delta.role = None, delta.content = "..."`

3. **ID/timestamp override**
   - ReasoningAgent: All chunks get same `completion_id` and `created`
   - PassthroughExecutor: Uses LiteLLM's IDs as-is

4. **Extra fields preservation**
   - LiteLLM fields (citations, obfuscation, service_tier) preserved
   - No data loss in conversion

## Success Criteria

-  Single conversion function used by both executors
-  No `isinstance()` checks in BaseExecutor
-  All tests pass with real-data-validated mocks
-  Type hints accurate (`AsyncGenerator[OpenAIStreamResponse]`)
-  Edge cases (None content, finish/usage chunks) handled
-  No test import errors

## References

- Real chunk data: `scripts/litellm_chunks_captured.json`
- Capture script: `scripts/capture_litellm_chunks.py`
- LiteLLM chunk type: `litellm.types.utils.ModelResponseStream`
- Our type: `api/openai_protocol.py::OpenAIStreamResponse`
