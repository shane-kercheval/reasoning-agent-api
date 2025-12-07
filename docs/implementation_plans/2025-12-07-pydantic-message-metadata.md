# Implementation Plan: Pydantic Message Metadata

## Overview

Convert metadata handling from `dict[str, Any]` to Pydantic models for type safety, validation, and better maintainability as the metadata structure grows in complexity.

### Current State
- `BaseExecutor._metadata` is `dict[str, Any]`
- `merge_dicts()` recursively merges dicts, summing numeric values
- `build_metadata_from_response()` extracts usage/cost/model from litellm responses
- Metadata stored as JSONB in `messages` table

### Target State
- Pydantic models define metadata structure
- `merge_models()` utility for Pydantic-native merging (in `utils.py`)
- `merge_dicts()` moved to `utils.py` (kept for general utility)
- Type-safe access throughout codebase
- Runtime validation at model boundaries

### Key Design Decision
Use **Pydantic-native merge** - iterate `model_fields` directly without dict conversion:

```python
from typing import TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

def merge_models(existing: T | None, new: T | None) -> T | None:
    """Merge Pydantic models, summing numeric fields."""
    if existing is None:
        return new
    if new is None:
        return existing

    merged = {}
    for field_name in existing.model_fields:
        existing_val = getattr(existing, field_name)
        new_val = getattr(new, field_name)

        # Recursively merge nested models
        if isinstance(existing_val, BaseModel) and isinstance(new_val, BaseModel):
            merged[field_name] = merge_models(existing_val, new_val)
        # Sum numeric types
        elif isinstance(existing_val, (int, float)) and isinstance(new_val, (int, float)):
            merged[field_name] = existing_val + new_val
        # Take newer non-None value, fallback to existing
        elif new_val is not None:
            merged[field_name] = new_val
        else:
            merged[field_name] = existing_val

    return type(existing).model_validate(merged)
```

This approach:
- No dict conversion round-trip (`model_dump`/`model_validate`)
- Handles nested `BaseModel` instances with type-aware recursion
- Self-contained, doesn't depend on `merge_dicts`

---

## Milestone 1: Define Pydantic Metadata Models

### Goal
Create Pydantic models that capture the current metadata structure with room for future growth.

### Success Criteria
- Models defined in new `reasoning_api/metadata_models.py`
- Models match current metadata structure from `build_metadata_from_response()`
- All fields properly typed with appropriate defaults
- Unit tests validate model creation and serialization

### Key Changes

**New file: `reasoning_api/reasoning_api/metadata_models.py`**

Define these models based on current usage (see `conversation_utils.py:159-206` and exploration results):

```python
from pydantic import BaseModel

class UsageMetadata(BaseModel):
    """Token usage from LLM response."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Provider-specific details (OpenAI, Anthropic vary)
    completion_tokens_details: dict[str, int] | None = None
    prompt_tokens_details: dict[str, int] | None = None

class CostMetadata(BaseModel):
    """Cost breakdown from litellm.completion_cost()."""
    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    total_cost: float = 0.0

class ContextBreakdown(BaseModel):
    """Message type breakdown in context window."""
    system_messages: int = 0
    user_messages: int = 0
    assistant_messages: int = 0

class ContextUtilization(BaseModel):
    """Context window utilization info from ContextManager."""
    model_name: str | None = None
    strategy: str | None = None  # "low", "medium", "full"
    model_max_tokens: int | None = None
    max_input_tokens: int | None = None
    input_tokens_used: int | None = None
    messages_included: int | None = None
    messages_excluded: int | None = None
    breakdown: ContextBreakdown | None = None

class ResponseMetadata(BaseModel):
    """Complete metadata accumulated during request execution."""
    usage: UsageMetadata | None = None
    cost: CostMetadata | None = None
    model: str | None = None
    routing_path: str | None = None  # "passthrough", "reasoning", "orchestration"
    context_utilization: ContextUtilization | None = None
```

**Important considerations:**
- Use `= None` defaults for optional fields (metadata may be partial)
- Use `= 0` defaults for numeric fields that get summed
- Keep `completion_tokens_details` as `dict[str, int]` since provider formats vary
- Model should serialize to JSON cleanly for JSONB storage

### Testing Strategy

Create `tests/unit_tests/test_metadata_models.py`:

1. **Model instantiation** - Create models with various field combinations
2. **Default values** - Verify defaults work correctly
3. **Serialization** - `model_dump()` produces expected dict structure
4. **Deserialization** - `model_validate()` reconstructs from dict
5. **Nested models** - ContextUtilization with ContextBreakdown works correctly
6. **JSONB compatibility** - Serialized output matches current dict structure

### Dependencies
None - this is the foundation.

### Risk Factors
- Provider-specific fields in `completion_tokens_details` may vary more than expected
- Need to check actual litellm response structures in integration tests

---

## Milestone 2: Move merge_dicts and Add merge_models to utils.py

### Goal
1. Move `merge_dicts()` from `conversation_utils.py` to `utils.py` (general utility)
2. Add Pydantic-native `merge_models()` to `utils.py`
3. Move corresponding tests to `test_utils.py`

### Success Criteria
- `merge_dicts()` moved to `utils.py`, removed from `conversation_utils.py`
- `merge_models()` added to `utils.py` using Pydantic-native approach
- Tests for both functions in `tests/unit_tests/test_utils.py`
- Any imports of `merge_dicts` from `conversation_utils` updated
- All existing tests pass

### Key Changes

**Update: `reasoning_api/reasoning_api/utils.py`**

Add both functions:

```python
from typing import Any, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def merge_dicts(
    existing: dict[str, Any] | None,
    new: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Recursively merge two dictionaries, summing numeric values.

    [Keep existing docstring and implementation from conversation_utils.py]
    """
    # ... existing implementation ...


def merge_models(existing: T | None, new: T | None) -> T | None:
    """
    Merge two Pydantic models, summing numeric fields.

    Iterates model_fields directly without dict conversion.
    Recursively handles nested BaseModel instances.

    Args:
        existing: Existing model or None
        new: New model to merge or None

    Returns:
        Merged model, or None if both inputs are None
    """
    if existing is None:
        return new
    if new is None:
        return existing

    merged = {}
    for field_name in existing.model_fields:
        existing_val = getattr(existing, field_name)
        new_val = getattr(new, field_name)

        # Recursively merge nested models
        if isinstance(existing_val, BaseModel) and isinstance(new_val, BaseModel):
            merged[field_name] = merge_models(existing_val, new_val)
        # Sum numeric types
        elif isinstance(existing_val, (int, float)) and isinstance(new_val, (int, float)):
            merged[field_name] = existing_val + new_val
        # Take newer non-None value, fallback to existing
        elif new_val is not None:
            merged[field_name] = new_val
        else:
            merged[field_name] = existing_val

    return type(existing).model_validate(merged)
```

**Update: `reasoning_api/reasoning_api/conversation_utils.py`**

- Remove `merge_dicts()` function
- Update import in `accumulate_metadata` (if used) to import from `utils`

**Update: `reasoning_api/reasoning_api/executors/base.py`**

- Update import: `from reasoning_api.utils import merge_models`

**Move tests to: `tests/unit_tests/test_utils.py`**

- Move all `merge_dicts` tests from `test_conversation_utils.py`
- Add new `merge_models` tests

### Testing Strategy

Add/move tests to `tests/unit_tests/test_utils.py`:

**For `merge_dicts` (moved from test_conversation_utils.py):**
- Keep all existing test cases

**For `merge_models` (new):**
1. **Both None** - Returns None
2. **One None** - Returns the non-None model
3. **Numeric summing** - `usage.prompt_tokens` sums correctly
4. **Nested summing** - `cost.total_cost` sums correctly
5. **String replacement** - `model` field takes newer value
6. **Nested model merge** - `context_utilization.breakdown` merges correctly
7. **Partial models** - Models with different fields present merge correctly
8. **Type preservation** - Return type matches input type
9. **Mixed None nested** - One model has nested model, other has None

### Dependencies
- Milestone 1 (metadata models exist for `merge_models` tests)

### Risk Factors
- Finding all imports of `merge_dicts` from `conversation_utils`
- Edge cases with deeply nested structures in `merge_models`

---

## Milestone 3: Update build_metadata_from_response

### Goal
Update `build_metadata_from_response()` to return `ResponseMetadata` instead of `dict[str, Any]`.

### Success Criteria
- Function returns `ResponseMetadata` model
- Existing behavior preserved
- Callers updated to use model
- Tests updated

### Key Changes

**Update: `reasoning_api/reasoning_api/conversation_utils.py`**

```python
def build_metadata_from_response(response: Any) -> ResponseMetadata:
    """
    Build metadata from litellm response.

    Returns:
        ResponseMetadata model (use .model_dump() for dict)
    """
    usage = None
    cost = None
    model = None

    # Extract usage
    usage_data = extract_usage(response)
    if usage_data:
        usage = UsageMetadata.model_validate(usage_data)

    # Calculate cost
    cost_data = calculate_cost(response)
    if cost_data:
        cost = CostMetadata.model_validate(cost_data)

    # Extract model
    if hasattr(response, 'model'):
        model = response.model

    return ResponseMetadata(usage=usage, cost=cost, model=model)
```

**Update callers in:**
- `reasoning_api/executors/passthrough.py` - Lines 151-153
- `reasoning_api/executors/reasoning_agent.py` - Lines 525-527, 604

The callers currently do:
```python
metadata = build_metadata_from_response(chunk)
metadata["context_utilization"] = context_metadata  # dict assignment
self.accumulate_metadata(metadata)
```

This needs to change to work with Pydantic. Options:
1. Add `context_utilization` as parameter to `build_metadata_from_response()`
2. Use `model_copy(update=...)` to add fields
3. Create metadata model directly in caller

Recommend option 2 for minimal change:
```python
metadata = build_metadata_from_response(chunk)
metadata = metadata.model_copy(update={"context_utilization": context_utilization})
self.accumulate_metadata(metadata)
```

**Update: `reasoning_api/reasoning_api/context_manager.py`**

`context_utilization` is currently a dict from `ContextManager`. Update it to return `ContextUtilization` model for consistency. This keeps the entire metadata pipeline typed.

1. Find where context utilization dict is built/returned
2. Import `ContextUtilization` and `ContextBreakdown` from `metadata_models`
3. Return model instance instead of dict

### Testing Strategy

Update existing tests in `test_conversation_utils.py`:
- Verify return type is `ResponseMetadata`
- Verify `.model_dump()` produces expected dict for backwards compatibility
- Test with various litellm response shapes

Update/add tests for `ContextManager`:
- Verify it returns `ContextUtilization` model
- Verify nested `ContextBreakdown` is populated correctly

### Dependencies
- Milestone 1 (ResponseMetadata exists)
- Milestone 2 (merge_models exists, needed for accumulation)

### Risk Factors
- Litellm response variations across providers

---

## Milestone 4: Update BaseExecutor Metadata Handling

### Goal
Update `BaseExecutor` to use `ResponseMetadata` model instead of `dict[str, Any]`.

### Success Criteria
- `_metadata` typed as `ResponseMetadata | None`
- `accumulate_metadata()` uses `merge_models()`
- `get_metadata()` returns `ResponseMetadata`
- All executors work correctly
- Integration tests pass

### Key Changes

**Update: `reasoning_api/reasoning_api/executors/base.py`**

```python
from reasoning_api.metadata_models import ResponseMetadata
from reasoning_api.conversation_utils import merge_models

class BaseExecutor(ABC):
    def __init__(self, ...):
        ...
        self._metadata: ResponseMetadata | None = None

    def accumulate_metadata(self, new_metadata: ResponseMetadata | None) -> None:
        """Accumulate metadata using Pydantic model merge."""
        self._metadata = merge_models(self._metadata, new_metadata)

    def get_metadata(self) -> ResponseMetadata:
        """Get accumulated metadata."""
        return self._metadata or ResponseMetadata()
```

**Update subclass initializers:**

`passthrough.py`:
```python
def __init__(self, ...):
    super().__init__(...)
    self.accumulate_metadata(ResponseMetadata(routing_path="passthrough"))
```

`reasoning_agent.py`:
```python
def __init__(self, ...):
    super().__init__(...)
    self.accumulate_metadata(ResponseMetadata(routing_path="reasoning"))
```

**Update cost augmentation in `base.py` execute_stream:**

Current code accesses `self._metadata.get("cost")` - update to use model attribute:
```python
if response.usage and self._metadata and self._metadata.cost:
    cost_data = self._metadata.cost
    response.usage.prompt_cost = cost_data.prompt_cost
    response.usage.completion_cost = cost_data.completion_cost
    response.usage.total_cost = cost_data.total_cost
```

### Testing Strategy

1. **Unit tests** - Update/add tests for accumulate_metadata with models
2. **Integration tests** - Run full request flow, verify metadata collected
3. **Verify JSONB storage** - `get_metadata().model_dump()` produces valid JSON

### Dependencies
- Milestone 3 (build_metadata_from_response returns model)

### Risk Factors
- Places accessing metadata as dict that were missed
- Integration with conversation storage (expects dict for JSONB)

---

## Milestone 5: Update Conversation Storage

### Goal
Ensure metadata flows correctly to database storage.

### Success Criteria
- `store_conversation_messages()` handles `ResponseMetadata` model
- JSONB storage works correctly
- `total_cost` extraction works
- Existing conversation queries still work

### Key Changes

**Update: `reasoning_api/reasoning_api/conversation_utils.py`**

`store_conversation_messages()` currently takes `response_metadata: dict | None`. Update to:

```python
async def store_conversation_messages(
    conversation_db: ConversationDB,
    conversation_id: UUID,
    request_messages: list[dict],
    response_content: str,
    response_metadata: ResponseMetadata | None = None,
    reasoning_events: list[dict] | None = None,
) -> None:
    # Extract total_cost from model
    total_cost = None
    if response_metadata and response_metadata.cost:
        total_cost = response_metadata.cost.total_cost

    response_message = {
        "role": "assistant",
        "content": response_content,
        "metadata": response_metadata.model_dump() if response_metadata else {},
        "total_cost": total_cost,
        "reasoning_events": reasoning_events,
    }
    ...
```

**Update caller in `main.py`:**

The call at line 233 should work if `executor.get_metadata()` returns `ResponseMetadata` - just ensure the signature accepts it.

### Testing Strategy

1. **Unit test** - Verify `model_dump()` produces valid dict for storage
2. **Integration test** - Full flow: request -> metadata accumulation -> storage -> retrieval
3. **Verify stored JSON** - Query database directly to check structure

### Dependencies
- Milestone 4 (executor returns ResponseMetadata)

### Risk Factors
- JSON serialization edge cases
- Backwards compatibility with existing stored metadata (should be fine - structure unchanged)

---

## Milestone 6: Cleanup and Documentation

### Goal
Remove any remaining `dict[str, Any]` metadata handling, update docstrings, ensure consistency.

### Success Criteria
- No `dict[str, Any]` for metadata in executor code
- Docstrings updated to reference Pydantic models
- Type hints consistent throughout
- All tests pass
- `make tests` passes

### Key Changes

1. **Review all files** for remaining dict metadata patterns
2. **Update docstrings** - Remove dict examples, add model examples
3. **Update CLAUDE.md** if metadata section exists
4. **Remove unused code** - Any dict-specific helpers no longer needed

### Testing Strategy
- Run full test suite: `make tests`
- Manual review of type hints

### Dependencies
- All previous milestones

### Risk Factors
- Missed usages of dict pattern

---

## Summary

| Milestone | Component | Key Files |
|-----------|-----------|-----------|
| 1 | Define models | `metadata_models.py` (new) |
| 2 | Merge utilities | `utils.py`, `conversation_utils.py` (remove merge_dicts), `test_utils.py` |
| 3 | Build metadata | `conversation_utils.py`, `context_manager.py`, executors |
| 4 | Executor updates | `base.py`, `passthrough.py`, `reasoning_agent.py` |
| 5 | Storage updates | `conversation_utils.py`, `main.py` |
| 6 | Cleanup | All files |

**Estimated scope**: ~7 files modified, 1 new file, focused changes with good test coverage.

**No backwards compatibility required** - this is an internal refactor. Breaking changes to internal APIs are acceptable.
