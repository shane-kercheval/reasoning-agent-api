# Context Manager Integration Plan

## Overview

Integrate the ContextManager into both PassthroughExecutor and ReasoningAgent to manage LLM context windows intelligently, ensuring messages fit within token limits while preserving the most recent conversation history.

**Context**: The API currently passes all messages to the LLM without checking token limits. For long conversations or models with smaller context windows, this can cause failures or truncation. The ContextManager implements a sliding window approach that:
- Always includes system messages
- Works backward from most recent messages until token limit reached
- Returns metadata about what was included/excluded
- Supports configurable utilization strategies (LOW/MEDIUM/FULL)

**Key Decisions**:
- Store context metadata in existing `messages.metadata` JSONB field (no migration needed)
- Default to FULL utilization (use entire context window)
- Make strategy configurable via request header `X-Context-Utilization`
- Return metadata to clients in final usage chunk
- Apply uniformly to both reasoning steps and final synthesis in ReasoningAgent

---

**NOTE**: us `uv` to run python comamnds e.g. `uv run python ...` or `uv run pytest ...`.

---

## Milestone 1: Fix and Test ContextManager Core

**Goal**: Fix critical bugs in context_manager.py and add comprehensive test coverage

**Why This First**: ContextManager has several bugs that will cause runtime failures. Must be fixed and tested before integration.

### Critical Bugs to Fix

1. **Syntax Error (Line 15)**:
```python
# WRONG:
conversation_history: list[dict][str, str]

# CORRECT:
conversation_history: list[dict[str, str]]
```

2. **Enum Inconsistency (Line 86)**:
```python
# ContextUtilization is NOT an Enum class, so .value doesn't exist
# Either make it a proper Enum or remove .value
from enum import Enum

class ContextUtilization(str, Enum):
    """Enum for context window utilization strategies."""
    LOW = "low"
    MEDIUM = "medium"
    FULL = "full"
```

3. **Message Ordering Bug (Lines 69-82)**:
```python
# Current code reverses messages and appends, resulting in REVERSED final order
# WRONG:
for msg in reversed(messages):
    final_messages.append(msg)  # Results in reversed order!

# CORRECT - Option A (insert at beginning):
for msg in reversed(messages):
    if msg not in system_messages:
        final_messages.insert(len(system_messages), msg)

# CORRECT - Option B (reverse at end):
for msg in reversed(messages):
    if msg not in system_messages:
        final_messages.append(msg)
final_messages[len(system_messages):] = reversed(final_messages[len(system_messages):])
```

4. **Use pop_system_messages for Consistency (Line 55)**:
```python
# CURRENT: Manual filtering
system_messages = [msg for msg in messages if msg['role'] == 'system']

# BETTER: Use existing utility
from api.openai_protocol import pop_system_messages
system_message_contents, non_system_messages = pop_system_messages(messages)
# Then reconstruct system_messages for token counting if needed
```

5. **Remove Unused Fields**:
```python
# Remove from Context model (lines 16-17):
# retrieved_documents: list[str]
# memory: list[dict]

# Keep commented lines 93-94 for future reference
```

### Success Criteria
- [ ] All syntax errors fixed
- [ ] ContextUtilization is proper Enum with str mixin
- [ ] Messages returned in correct order (most recent last)
- [ ] Uses pop_system_messages utility
- [ ] Context model only has conversation_history field
- [ ] All tests pass

### Testing Strategy

**Create `tests/unit_tests/test_context_manager.py`**:

```python
"""Unit tests for ContextManager."""

class TestContextUtilization:
    """Test ContextUtilization enum."""

    def test_enum_values(self):
        """Ensure enum values are correct strings."""
        assert ContextUtilization.LOW.value == "low"
        assert ContextUtilization.MEDIUM.value == "medium"
        assert ContextUtilization.FULL.value == "full"

class TestContextManager:
    """Test ContextManager core functionality."""

    def test_preserves_message_order(self):
        """Most recent message should be last in output."""
        # Test that messages come out in chronological order

    def test_includes_all_system_messages(self):
        """All system messages should be included regardless of token limit."""

    def test_works_backward_from_recent(self):
        """Should include most recent messages first when hitting limit."""

    def test_respects_utilization_strategy(self):
        """LOW should use 33%, MEDIUM 66%, FULL 100% of context."""

    def test_handles_empty_conversation(self):
        """Should handle empty message list gracefully."""

    def test_handles_only_system_messages(self):
        """Should handle conversation with only system messages."""

    def test_metadata_includes_all_fields(self):
        """Metadata should include model, strategy, tokens, breakdown."""

    def test_raises_when_system_messages_exceed_limit(self):
        """Should raise ValueError if system messages alone exceed limit."""

    def test_token_counting_matches_litellm(self):
        """Token counts should match litellm.token_counter results."""
```

**Key Test Scenarios**:
- Message order preservation (critical!)
- System message handling (always included)
- Token limit enforcement (by utilization level)
- Metadata structure and completeness
- Edge cases (empty, system-only, exceeds limit)

### Risk Factors
- Message ordering is complex - needs careful validation
- Token counting relies on litellm.token_counter - verify accuracy
- Different models have different tokenizers - test with multiple models

---

## Milestone 2: Add Request Header Support

**Goal**: Enable clients to control context utilization strategy via `X-Context-Utilization` header

**Why**: Allows per-request control while maintaining sensible default (FULL)

### Key Changes

**1. Add Header Constant and Parsing**

Look at how other headers are handled (e.g., `X-Routing-Mode`, `X-Session-ID`) in the codebase:
- Check `api/main.py` for header extraction patterns
- Check `api/request_router.py` for header validation examples

**Add to relevant location** (likely near other header constants):
```python
# Header name
CONTEXT_UTILIZATION_HEADER = "X-Context-Utilization"

def parse_context_utilization_header(
    header_value: str | None
) -> ContextUtilization:
    """
    Parse X-Context-Utilization header into ContextUtilization enum.

    Args:
        header_value: Value of header (case-insensitive)

    Returns:
        ContextUtilization enum value (defaults to FULL if not provided)

    Raises:
        ValueError: If header value is invalid
    """
    if header_value is None:
        return ContextUtilization.FULL

    # Case-insensitive matching (like X-Routing-Mode)
    normalized = header_value.lower().strip()

    try:
        return ContextUtilization(normalized)
    except ValueError:
        valid = [e.value for e in ContextUtilization]
        raise ValueError(
            f"Invalid context utilization: {header_value}. "
            f"Must be one of: {', '.join(valid)}"
        )
```

**2. Wire Through Dependency Injection**

Update `api/dependencies.py`:
```python
def get_context_manager(
    request: Request,
) -> ContextManager:
    """
    Get ContextManager instance based on request headers.

    Reads X-Context-Utilization header to determine strategy.
    Defaults to FULL if header not provided.
    """
    header_value = request.headers.get(CONTEXT_UTILIZATION_HEADER)
    utilization = parse_context_utilization_header(header_value)
    return ContextManager(context_utilization=utilization)
```

**3. Update Executor Constructors**

Both `PassthroughExecutor` and `ReasoningAgent` should accept `ContextManager` as dependency:
```python
class PassthroughExecutor(BaseExecutor):
    def __init__(
        self,
        context_manager: ContextManager,  # Add this
        parent_span: trace.Span | None = None,
        check_disconnected: Callable[[], bool] | None = None,
    ):
        super().__init__(parent_span, check_disconnected)
        self.context_manager = context_manager
        self.accumulate_metadata({"routing_path": "passthrough"})
```

### Success Criteria
- [ ] Header parsing works case-insensitively
- [ ] Defaults to FULL when header not provided
- [ ] Raises clear error for invalid values
- [ ] ContextManager properly injected into executors
- [ ] All tests pass

### Testing Strategy

**Add to `tests/unit_tests/test_dependencies.py` (or create if doesn't exist)**:

```python
class TestContextUtilizationHeader:
    """Test X-Context-Utilization header parsing."""

    def test_defaults_to_full_when_missing(self):
        """Should default to FULL utilization when header not provided."""

    def test_parses_lowercase(self):
        """Should parse 'low', 'medium', 'full'."""

    def test_parses_uppercase(self):
        """Should parse 'LOW', 'MEDIUM', 'FULL' (case-insensitive)."""

    def test_parses_mixed_case(self):
        """Should parse 'Low', 'Medium', 'Full'."""

    def test_strips_whitespace(self):
        """Should handle ' low ', ' medium ', ' full '."""

    def test_raises_on_invalid_value(self):
        """Should raise ValueError with helpful message for invalid values."""
```

### Dependencies
- Milestone 1 must be complete (working ContextManager)

---

## Milestone 3: Integrate with PassthroughExecutor

**Goal**: Apply context management in passthrough path and save metadata

**Why**: Passthrough is the simplest execution path - good place to validate integration pattern before applying to ReasoningAgent

### Key Changes

**Update `api/executors/passthrough.py`**:

```python
class PassthroughExecutor(BaseExecutor):
    def __init__(
        self,
        context_manager: ContextManager,
        parent_span: trace.Span | None = None,
        check_disconnected: Callable[[], bool] | None = None,
    ):
        super().__init__(parent_span, check_disconnected)
        self.context_manager = context_manager
        # ... rest of init

    async def _execute_stream(
        self,
        request: OpenAIChatRequest,
    ) -> AsyncGenerator[OpenAIStreamResponse]:
        """Execute streaming with context management."""

        # 1. Apply context management BEFORE litellm call
        from api.context_manager import Context

        context = Context(conversation_history=request.messages)
        filtered_messages, context_metadata = self.context_manager(
            model_name=request.model,
            context=context,
        )

        # 2. Use filtered messages in API call
        stream = await litellm.acompletion(
            model=request.model,
            messages=filtered_messages,  # � Context-managed
            max_tokens=request.max_tokens,
            # ... other params
        )

        # 3. Accumulate context metadata with usage metadata
        async for chunk in stream:
            chunk_usage = getattr(chunk, 'usage', None)
            if chunk_usage:
                # Build metadata including context info
                metadata = build_metadata_from_response(chunk)
                metadata["context_utilization"] = context_metadata
                self.accumulate_metadata(metadata)

            # ... rest of streaming logic
```

**Update `api/conversation_utils.py`**:

The `build_metadata_from_response()` function is already flexible (returns dict), so no changes needed there. The executor will just add the `context_utilization` key to the metadata dict before calling `accumulate_metadata()`.

### Success Criteria
- [ ] Context manager applied before litellm call
- [ ] Filtered messages used in API request
- [ ] Context metadata saved alongside cost/usage metadata
- [ ] All existing passthrough tests still pass
- [ ] New integration tests pass

### Testing Strategy

**Add to `tests/integration_tests/test_passthrough.py`**:

```python
@pytest.mark.integration
class TestPassthroughContextManagement:
    """Test context management in passthrough executor."""

    async def test_uses_full_context_by_default(self):
        """Should use all messages when they fit in context window."""
        # Test with conversation that fits in context
        # Verify all messages sent to LLM

    async def test_truncates_when_exceeds_medium(self):
        """Should exclude old messages when using MEDIUM strategy."""
        # Test with long conversation + MEDIUM header
        # Verify oldest messages excluded, recent included

    async def test_saves_context_metadata(self):
        """Should save context utilization metadata with assistant message."""
        # Make request, collect response
        # Check accumulated metadata has context_utilization key
        # Verify structure matches expected format

    async def test_context_metadata_in_database(self):
        """Should persist context metadata to database."""
        # Make request with conversation persistence
        # Retrieve message from DB
        # Verify metadata.context_utilization exists
```

**Unit Tests** (`tests/unit_tests/test_passthrough.py`):

```python
class TestPassthroughContextManager:
    """Test PassthroughExecutor context management."""

    async def test_filters_messages_before_api_call(self, mock_litellm):
        """Should apply context filtering before calling litellm."""
        # Mock litellm.acompletion
        # Verify it receives filtered messages, not original

    async def test_includes_context_metadata_in_accumulation(self):
        """Should accumulate context metadata alongside usage/cost."""
        # Test that metadata dict has all expected keys
```

### Dependencies
- Milestone 1 (working ContextManager)
- Milestone 2 (header support)

### Risk Factors
- Context filtering happens before litellm call - ensure no side effects
- Metadata accumulation timing - verify context metadata saved correctly
- Long conversations might hit token limits even with FULL - need graceful handling

---

## Milestone 4: Integrate with ReasoningAgent

**Goal**: Apply context management to both reasoning steps (ensure fit) and final synthesis (save metadata)

**Why**: ReasoningAgent builds complex message lists internally - context management ensures they fit while providing visibility into what was included

### Key Changes

**Update `api/executors/reasoning_agent.py`**:

```python
class ReasoningAgent(BaseExecutor):
    def __init__(
        self,
        tools: list[Tool],
        prompt_manager: PromptManager,
        context_manager: ContextManager,  # Add this
        max_reasoning_iterations: int = 20,
        parent_span: trace.Span | None = None,
        check_disconnected: Callable[[], bool] | None = None,
    ):
        super().__init__(parent_span, check_disconnected)
        self.tools = {tool.name: tool for tool in tools}
        self.prompt_manager = prompt_manager
        self.context_manager = context_manager  # Store it
        # ... rest of init

    async def _generate_reasoning_step(
        self,
        request: OpenAIChatRequest,
        context: dict[str, Any],
        system_prompt: str,
    ) -> tuple[ReasoningStep, OpenAIUsage | None]:
        """Generate reasoning step with context management."""

        # Build message list as before
        messages = deepcopy(request.messages)
        _, messages = pop_system_messages(messages)
        messages.insert(0, {"role": "system", "content": system_prompt})

        # Add previous steps, tool results, etc.
        if context["steps"]:
            # ... existing logic

        # Apply context management to ensure messages fit
        # NOTE: We don't save metadata here - internal reasoning steps only
        ctx = Context(conversation_history=messages)
        filtered_messages, _ = self.context_manager(
            model_name=request.model,
            context=ctx,
        )

        # Use filtered messages in API call
        response = await litellm.acompletion(
            model=request.model,
            messages=filtered_messages,  # � Context-managed
            response_format={"type": "json_object"},
            # ... rest
        )
        # ... rest of function

    async def _stream_final_synthesis(
        self,
        request: OpenAIChatRequest,
        completion_id: str,
        created: int,
        reasoning_context: dict[str, Any],
    ) -> AsyncGenerator[OpenAIStreamResponse]:
        """Stream final synthesis with context management and metadata saving."""

        # Build synthesis messages as before
        messages = deepcopy(request.messages)
        system_prompts, messages = pop_system_messages(messages)

        synthesis_prompt = await self.prompt_manager.get_prompt("final_answer")
        if system_prompts:
            synthesis_prompt = (
                synthesis_prompt
                + "\n\n---\n\n**Custom User Prompt/Instructions:**\n\n"
                + "\n".join(system_prompts)
            )

        messages.insert(0, {"role": "system", "content": synthesis_prompt})

        # Add reasoning summary
        reasoning_summary = self._build_reasoning_summary(reasoning_context)
        messages.append({
            "role": "assistant",
            "content": f"My reasoning process:\n{reasoning_summary}",
        })

        # Apply context management and SAVE metadata (user-facing response)
        ctx = Context(conversation_history=messages)
        filtered_messages, context_metadata = self.context_manager(
            model_name=request.model,
            context=ctx,
        )

        # Stream synthesis
        stream = await litellm.acompletion(
            model=request.model,
            messages=filtered_messages,  # � Context-managed
            stream=True,
            # ... rest
        )

        async for chunk in stream:
            chunk_usage = getattr(chunk, 'usage', None)
            if chunk_usage:
                # Save context metadata alongside usage/cost
                metadata = build_metadata_from_response(chunk)
                metadata["context_utilization"] = context_metadata
                self.accumulate_metadata(metadata)

            yield convert_litellm_to_stream_response(...)
```

**Key Pattern**:
- **Reasoning steps**: Apply context management but DON'T save metadata (internal)
- **Final synthesis**: Apply context management AND save metadata (user-facing)

### Success Criteria
- [ ] Context management applied to reasoning steps (ensures fit)
- [ ] Context management applied to final synthesis (ensures fit + saves metadata)
- [ ] Context metadata only saved for final synthesis, not reasoning steps
- [ ] All existing reasoning agent tests pass
- [ ] New integration tests pass

### Testing Strategy

**Add to `tests/integration_tests/test_reasoning_agent.py`**:

```python
@pytest.mark.integration
class TestReasoningAgentContextManagement:
    """Test context management in reasoning agent."""

    async def test_reasoning_steps_fit_in_context(self):
        """Should filter reasoning step messages to fit context window."""
        # Create long conversation history
        # Verify reasoning steps complete successfully
        # Check that messages were filtered

    async def test_final_synthesis_saves_context_metadata(self):
        """Should save context metadata for final synthesis only."""
        # Run reasoning agent request
        # Check accumulated metadata has context_utilization
        # Verify it's only for final synthesis, not intermediate steps

    async def test_handles_large_reasoning_context(self):
        """Should handle case where reasoning summary is very large."""
        # Create scenario with many reasoning steps + tool results
        # Verify context management prevents token limit errors

    async def test_system_prompt_always_included(self):
        """Should always include system prompt even with context limits."""
        # Test with custom system prompt + long conversation
        # Verify system prompt present in filtered messages
```

**Unit Tests** (`tests/unit_tests/test_reasoning_agent.py`):

```python
class TestReasoningAgentContextManager:
    """Test ReasoningAgent context management."""

    async def test_filters_reasoning_step_messages(self, mock_litellm):
        """Should filter messages before generating reasoning steps."""
        # Mock context manager
        # Verify filtered messages passed to litellm

    async def test_filters_synthesis_messages(self, mock_litellm):
        """Should filter messages before final synthesis."""
        # Mock context manager
        # Verify filtered messages passed to litellm

    async def test_only_accumulates_synthesis_metadata(self):
        """Should only save context metadata for synthesis, not steps."""
        # Run full reasoning flow
        # Check metadata only accumulated once (synthesis)
```

### Dependencies
- Milestone 1 (working ContextManager)
- Milestone 2 (header support)
- Milestone 3 (passthrough integration validates pattern)

### Risk Factors
- Reasoning agent builds complex message structures - carefully validate filtering doesn't break reasoning
- Large reasoning summaries could exceed context limits - need testing with many steps
- Tool results in context might be large - verify token counting accuracy

---

## Milestone 5: Return Metadata to Clients

**Goal**: Include context utilization metadata in API responses for client visibility

**Why**: Clients need to know if their messages were truncated and understand token usage

### Key Changes

**Option A: Include in Usage Chunk (Recommended)**

Update `api/openai_protocol.py`:

```python
class OpenAIUsage(BaseModel):
    """OpenAI usage statistics - validated against real OpenAI API."""

    model_config = ConfigDict(extra='allow')

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: dict[str, Any] | None = None
    completion_tokens_details: dict[str, Any] | None = None

    # Cost extensions (optional, not in OpenAI spec)
    prompt_cost: float | None = None
    completion_cost: float | None = None
    total_cost: float | None = None

    # Context extensions (optional, not in OpenAI spec)
    context_utilization: dict[str, Any] | None = None
```

**Update executors to include context metadata in usage chunk**:

```python
# In PassthroughExecutor and ReasoningAgent
async for chunk in stream:
    chunk_usage = getattr(chunk, 'usage', None)
    if chunk_usage:
        metadata = build_metadata_from_response(chunk)
        metadata["context_utilization"] = context_metadata
        self.accumulate_metadata(metadata)

        # Also add to the usage object in the response chunk
        if hasattr(chunk, 'usage') and chunk.usage:
            # Convert to OpenAIUsage if needed
            usage = OpenAIUsage(
                prompt_tokens=chunk.usage.prompt_tokens,
                completion_tokens=chunk.usage.completion_tokens,
                total_tokens=chunk.usage.total_tokens,
                context_utilization=context_metadata,  # Add it here
            )
            # Update chunk's usage
            response_chunk = convert_litellm_to_stream_response(chunk)
            response_chunk.usage = usage
            yield response_chunk
```

### Success Criteria
- [ ] Context metadata included in final usage chunk
- [ ] Metadata structure documented in API docs
- [ ] Client can parse and display context information
- [ ] All tests pass

### Testing Strategy

**Add to integration tests**:

```python
@pytest.mark.integration
async def test_context_metadata_in_response():
    """Should include context metadata in usage chunk."""
    # Make streaming request
    # Collect all chunks
    # Find usage chunk (last non-DONE chunk)
    # Verify usage.context_utilization exists
    # Verify structure matches expected format

@pytest.mark.integration
async def test_context_metadata_structure():
    """Should return properly structured context metadata."""
    # Make request
    # Get usage chunk
    # Verify all expected fields present:
    #   - strategy
    #   - max_input_tokens
    #   - input_tokens_used
    #   - breakdown.system_messages
    #   - breakdown.user_messages
    #   - breakdown.assistant_messages
```

**Update API Documentation**:

Document in `README.md` or API docs:
```markdown
## Context Management

The API automatically manages context windows to ensure messages fit within model limits.

### Request Header

Control context utilization strategy with `X-Context-Utilization` header:
- `low` - Use 33% of model's context window
- `medium` - Use 66% of model's context window
- `full` - Use 100% of model's context window (default)

### Response Metadata

The final usage chunk includes context utilization information:

```json
{
  "usage": {
    "prompt_tokens": 45000,
    "completion_tokens": 500,
    "total_tokens": 45500,
    "context_utilization": {
      "strategy": "medium",
      "max_input_tokens": 66000,
      "input_tokens_used": 45000,
      "messages_included": 15,
      "messages_excluded": 3,
      "breakdown": {
        "system_messages": 500,
        "user_messages": 20000,
        "assistant_messages": 24500
      }
    }
  }
}
```

**Fields**:
- `strategy`: Context utilization level used (low/medium/full)
- `max_input_tokens`: Maximum input tokens allowed (after applying strategy)
- `input_tokens_used`: Actual tokens used from included messages
- `messages_included`: Count of messages sent to model
- `messages_excluded`: Count of messages excluded due to token limit
- `breakdown`: Token counts by message role
```

### Dependencies
- All previous milestones complete

---

## Testing Guidelines

### Test Priorities (High to Low)

1. **Critical Path**: Context filtering works correctly (messages in right order, system messages preserved)
2. **Integration**: PassthroughExecutor and ReasoningAgent apply context management correctly
3. **Persistence**: Context metadata saved to database correctly
4. **Edge Cases**: Empty conversations, token limit exceeded, system-only messages
5. **Headers**: Request header parsing and validation
6. **Metadata**: Response metadata structure and client visibility

### Test Data Strategies

**Token Limit Testing**:
```python
# Use models with known context limits for predictable testing
TEST_MODELS = {
    "gpt-4o-mini": 128_000,  # Large context
    "gpt-3.5-turbo": 16_385,  # Medium context
}

# Create conversations that exceed limits predictably
def create_long_conversation(target_tokens: int) -> list[dict]:
    """Create conversation that approximates target token count."""
    # Use repetitive text with known token counts
    # Verify actual tokens with litellm.token_counter
```

**Metadata Validation**:
```python
def assert_valid_context_metadata(metadata: dict):
    """Assert context metadata has correct structure."""
    assert "context_utilization" in metadata
    ctx = metadata["context_utilization"]

    # Required fields
    assert "strategy" in ctx
    assert ctx["strategy"] in ["low", "medium", "full"]
    assert "max_input_tokens" in ctx
    assert "input_tokens_used" in ctx
    assert "breakdown" in ctx

    # Breakdown structure
    breakdown = ctx["breakdown"]
    assert "system_messages" in breakdown
    assert "user_messages" in breakdown
    assert "assistant_messages" in breakdown
```

---

## Rollout Considerations

**Feature Flag (Optional)**:
If you want gradual rollout, consider adding environment variable:
```python
# In api/config.py
ENABLE_CONTEXT_MANAGEMENT: bool = Field(
    default=True,
    description="Enable context window management"
)

# In executors
if settings.ENABLE_CONTEXT_MANAGEMENT:
    filtered_messages, metadata = self.context_manager(...)
else:
    filtered_messages = messages
    metadata = None
```

**Monitoring**:
After deployment, monitor:
- Frequency of message truncation (messages_excluded > 0)
- Token utilization distribution (how close to limits)
- Error rates with/without context management
- Performance impact (token counting adds latency)

---

## Future Enhancements (Out of Scope)

These are explicitly NOT part of this plan but documented for future consideration:

1. **Semantic Message Selection**: Instead of just "most recent N messages", use embeddings to select most relevant messages
2. **Retrieved Documents**: Add support for RAG context (commented out in current implementation)
3. **Memory/Personas**: Add support for persistent memory context
4. **Smart Truncation**: Truncate individual long messages instead of excluding entirely
5. **Database Column Promotion**: If context queries become common, promote fields to dedicated columns with indexes (like total_cost)
6. **Logging/Warnings**: Log warnings when messages are excluded due to token limits

---

## Success Metrics

**Before Starting**: All tests pass
**After Each Milestone**: All tests still pass + new tests pass
**Final Validation**:
- [ ] ContextManager filters messages correctly
- [ ] PassthroughExecutor applies context management
- [ ] ReasoningAgent applies context management
- [ ] Context metadata saved to database
- [ ] Context metadata returned to clients
- [ ] Header-based strategy selection works
- [ ] No regressions in existing functionality
- [ ] Documentation updated
- [ ] All tests passing (unit + integration)

---

## Questions to Resolve

Before starting implementation, consider:

1. **Token Counting Performance**: Does calling `litellm.token_counter` on every message add significant latency? Should we cache results?

2. **System Message Token Limit**: Current code raises ValueError if system messages exceed token limit. Is this the right behavior? Alternative: truncate system message with warning?

3. **Message Exclusion Logging**: Should we log warnings when messages are excluded? At what level (debug/info/warning)?

4. **Backward Compatibility**: Are there existing clients that depend on all messages being sent? Need coordination with client teams?

5. **Default Strategy**: Confirmed FULL is right default? Could MEDIUM be safer default to leave headroom?

**Agent: Ask these questions before implementing if anything is unclear!**
