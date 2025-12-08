# LLM-Driven Context Management

## Overview

This plan introduces intelligent, LLM-driven context management to replace the current naive truncation approach. Instead of blindly truncating large content or using recency-based message selection, we use an LLM to analyze all available information and make intelligent decisions about what to include and how to process it.

**Core Insight**: Give an LLM all artifacts (previews + metadata), the user's question, and the token budget. Let it return a prioritized list with processing strategies for each artifact.

**Problem Statement**:

1. **Greedy fit bug**: Current algorithm skips large messages, potentially creating incoherent context
2. **No relevance filtering**: Uses recency, not semantic relevance to the user's question
3. **Naive truncation**: `preview()` just truncates; doesn't intelligently summarize based on what's needed
4. **No caching**: Re-processes same content repeatedly
5. **No document support**: Can't attach documents to conversations
6. **Single model**: One model used for everything; no way to use cheaper models for internal tasks
7. **Historical artifacts lost**: Tool results from previous turns are stored in `reasoning_events` but never used in follow-up messages

**Target State**:

- Historical tool results extracted from `reasoning_events` and made available for follow-up questions
- Intelligent decision about when LLM planning is worth the overhead
- Single LLM call plans entire context assembly (using cheap/fast model)
- Each artifact gets appropriate processing: include full, summarize, RAG extract, or exclude
- Summaries are cached to avoid re-computation
- Documents can be attached to requests and flow through the same pipeline as tool results
- Configurable models for different internal tasks (planning, summarization)

---

## Milestone 1: Artifact Abstraction

### Goal
Create a unified `Artifact` model that represents any piece of information that may go into LLM context (tool results, documents). This abstraction enables uniform handling of all information sources.

### Success Criteria
- `Artifact` and `ArtifactType` models defined in new `artifacts.py`
- `ProcessingStrategy`, `ArtifactDecision`, `ArtifactPlan` models for LLM output
- `extract_artifacts_from_steps()` converts `ReasoningStepRecord` list to artifacts
- Existing tests pass

### Key Interfaces

```python
class ArtifactType(str, Enum):
    TOOL_RESULT = "tool_result"
    DOCUMENT = "document"

class ProcessingStrategy(str, Enum):
    INCLUDE_FULL = "include_full"     # Small, relevant → include as-is
    SUMMARIZE = "summarize"           # Large, relevant → LLM summarize
    RAG_EXTRACT = "rag_extract"       # Large, specific lookup → semantic search
    EXCLUDE = "exclude"               # Not relevant → omit

class Artifact(BaseModel):
    artifact_id: str
    artifact_type: ArtifactType
    content: Any
    size_chars: int
    source: str                       # e.g., "step_1:web_scraper", "history:msg_3:web_search"
    timestamp: datetime
    metadata: dict[str, Any] = {}     # Includes "historical": True for past turns

class ArtifactDecision(BaseModel):
    """LLM's decision for a single artifact."""
    artifact_id: str
    strategy: ProcessingStrategy
    priority: int                     # 1 = highest
    relevance_rationale: str
    summarization_focus: str | None   # For SUMMARIZE: what to extract
    extraction_query: str | None      # For RAG_EXTRACT: what to search

class ArtifactPlan(BaseModel):
    """LLM's complete plan for context assembly."""
    artifacts: list[ArtifactDecision]  # Ordered by priority
    reasoning: str
```

### Testing Strategy
- Test artifact instantiation with various content types
- Test `extract_artifacts_from_steps()` with empty, single, multiple steps
- Verify artifact_id uniqueness

---

## Milestone 2: ConversationMessage Model & Historical Context

### Goal
Create a proper `ConversationMessage` Pydantic model that preserves the association between messages and their reasoning context (tool usage). Currently `build_llm_messages()` returns `list[dict]` which loses this association.

**The Bug**: In `build_llm_messages()` (conversation_utils.py:371-372), when loading a continuing conversation, only `role` and `content` are extracted. The `reasoning_events` field containing all tool results is ignored.

### Success Criteria
- `ConversationMessage` model with role, content, reasoning_steps, sequence_number
- `build_llm_messages()` returns `list[ConversationMessage]` instead of `list[dict]`
- Each message preserves its associated `ReasoningStepRecord`s
- Integration test: follow-up question can reference data from prior tool execution

### Key Interfaces

```python
class ConversationMessage(BaseModel):
    """A message with its reasoning context preserved."""
    role: str                                        # "system", "user", "assistant"
    content: str
    reasoning_steps: list[ReasoningStepRecord] = []  # Tool usage for this message
    sequence_number: int | None = None               # For ordering/identification

async def build_llm_messages(
    request_messages: list[dict],
    conversation_ctx: ConversationContext,
    conversation_db: ConversationDB | None,
) -> list[ConversationMessage]:
    ...
    # For continuing conversations, include reasoning_steps from reasoning_events
    for msg in conversation.messages:
        reasoning_steps = []
        if msg.role == "assistant" and msg.reasoning_events:
            reasoning_steps = parse_reasoning_events_to_steps(msg.reasoning_events)

        messages.append(ConversationMessage(
            role=msg.role,
            content=msg.content,
            reasoning_steps=reasoning_steps,
            sequence_number=msg.sequence_number,
        ))
    ...
```

### Key Changes
- Add `ConversationMessage` model (in `conversation_utils.py` or new file)
- Add `parse_reasoning_events_to_steps()` to convert stored events back to `ReasoningStepRecord`
- Update `build_llm_messages()` to return `list[ConversationMessage]`
- Update all callers of `build_llm_messages()`

### Testing Strategy
- Test `ConversationMessage` with and without reasoning_steps
- Test `parse_reasoning_events_to_steps()` conversion
- Test `build_llm_messages()` returns proper model with reasoning context
- Integration test: make request with tool usage, then follow-up, verify reasoning_steps populated

### Risk Factors
- **Breaking change**: `build_llm_messages()` return type changes from `list[dict]` to `list[ConversationMessage]`
- Need to update all callers

---

## Milestone 3: Request Artifact Attachment Support

### Goal
Enable artifacts (documents, etc.) to be attached to chat completion requests. These are request-scoped, not message-scoped.

### Success Criteria
- `RequestAttachment` model in `openai_protocol.py`
- `OpenAIChatRequest.attachments` field added
- `attachments_to_artifacts()` conversion function
- Validation: requires either `attachment_id` or `content`

### Key Interface

```python
class RequestAttachment(BaseModel):
    """Artifact attached to a request (document, data, etc.)."""
    attachment_id: str | None = None  # Reference to stored artifact (future)
    content: str | None = None        # Inline content
    title: str | None = None
    source: str | None = None         # URL, filepath, etc.
    attachment_type: str = "document" # "document", "data", etc.
    # Validation: requires attachment_id OR content

class OpenAIChatRequest(BaseModel):
    # ... existing fields ...
    attachments: list[RequestAttachment] | None = None  # NEW
```

### Testing Strategy
- Test validation (requires attachment_id OR content)
- Test conversion to Artifact model
- Integration test: request with attachments reaches executor

---

## Milestone 4: Context Model Update

### Goal
Update `Context` model to work with `ConversationMessage` and support request-attached artifacts.

### Success Criteria
- `Context.conversation_history` type changed to `list[ConversationMessage]`
- `Context.artifacts` field for request-attached artifacts (documents, etc.)
- `Context.all_artifacts()` method combines current steps + historical + attached
- `extract_artifacts_from_messages()` extracts from `ConversationMessage.reasoning_steps`
- `_build_reasoning_context` includes all artifact types in output

### Key Changes

```python
class Context(BaseModel):
    conversation_history: list[ConversationMessage]   # Messages with reasoning context
    step_records: list[ReasoningStepRecord] = []      # Current session reasoning steps
    goal: ContextGoal | None = None
    system_prompt_override: str | None = None
    artifacts: list[Artifact] = []                    # Request-attached artifacts (documents, etc.)

    def all_artifacts(self) -> list[Artifact]:
        """Combine artifacts from all sources: current steps, historical messages, attached."""
        current = extract_artifacts_from_steps(self.step_records)
        historical = extract_artifacts_from_messages(self.conversation_history)
        return current + historical + self.artifacts
```

Note: `extract_artifacts_from_messages()` iterates through `ConversationMessage.reasoning_steps` to extract tool results as artifacts.

### Testing Strategy
- Test `Context.all_artifacts()` combines all sources correctly
- Test artifact extraction from `ConversationMessage.reasoning_steps`
- Test `_build_reasoning_context` output includes all artifact types
- Test with empty artifact lists (no change to output)

---

## Milestone 5: Extend Metadata Models for Context Planning

### Goal
Extend existing metadata models to capture cost/usage from context planning LLM calls and track which artifacts were used and how. This enables clients to understand how the response was generated and provides cost transparency.

### Success Criteria
- `ContextPlanningUsage` and `ContextPlanningCost` models for internal LLM calls
- `ArtifactUsageRecord` to track artifact processing (no content, just reference + strategy)
- `ContextUtilizationMetadata` extended with artifact processing info
- Total cost calculation includes context planning costs

### Key Interfaces

```python
class ArtifactUsageRecord(BaseModel):
    """Record of how an artifact was used in context (no content stored)."""
    artifact_id: str                 # Reference to artifact
    source: str                      # e.g., "step_1:web_search", "history:msg_3:web_scraper"
    artifact_type: str               # "tool_result" or "document"
    strategy: str                    # "include_full", "summarize", "exclude"
    original_size_chars: int
    processed_size_chars: int | None = None  # If summarized
    cache_hit: bool = False

class ContextPlanningCost(BaseModel):
    """Cost from context planning LLM calls (separate from primary model)."""
    planner_cost: float = 0.0        # Cost of artifact planning call
    summarizer_cost: float = 0.0     # Cost of summarization calls
    total_cost: float = 0.0

class ContextUtilizationMetadata(BaseModel):
    # ... existing fields (model_name, strategy, tokens, messages, breakdown, goal) ...

    # NEW: Artifact processing info
    artifacts_processed: list[ArtifactUsageRecord] = []
    llm_planning_used: bool = False
    planning_cost: ContextPlanningCost | None = None
    cache_stats: dict[str, int] | None = None  # {"hits": 3, "misses": 1}
```

### Key Changes
- Add `ArtifactUsageRecord`, `ContextPlanningCost` models
- Extend `ContextUtilizationMetadata` (in `context_manager.py`) with artifact tracking fields
- Update `ArtifactProcessor` to return usage/cost from LLM calls
- Ensure total cost aggregation includes context planning costs

**Existing models to be aware of** (in `conversation_utils.py`):
- `UsageMetadata`: prompt_tokens, completion_tokens, total_tokens
- `CostMetadata`: prompt_cost, completion_cost, total_cost
- `ResponseMetadata`: aggregates usage + cost + model + routing_path + context_utilization

The context planning costs are tracked separately in `ContextUtilizationMetadata.planning_cost` and should be added to the total when reporting to clients.

### Testing Strategy
- Test `ArtifactUsageRecord` serialization/deserialization
- Test `ContextUtilizationMetadata` with artifact processing info
- Test cost aggregation includes context planning costs
- Verify artifact records don't contain content (only references)

---

## Milestone 6: Model Configuration

### Goal
Create `ModelConfig` abstraction separating primary model (from request) from internal task models (context planning, summarization).

### Success Criteria
- `ModelConfig` model with `primary`, `context_planner`, `summarizer` fields
- `ModelConfig.from_request()` factory using environment defaults
- `is_expensive_primary()` method for cost-based decisions
- Environment settings: `CONTEXT_PLANNER_MODEL`, `SUMMARIZER_MODEL` (default: `gpt-4o-mini`)

### Key Interface

```python
class ModelConfig(BaseModel):
    primary: str              # From request - reasoning + synthesis
    context_planner: str      # For planning (default: gpt-4o-mini)
    summarizer: str           # For summarizing (default: gpt-4o-mini)

    @classmethod
    def from_request(cls, request_model: str) -> "ModelConfig": ...

    def is_expensive_primary(self, threshold: float = 0.00001) -> bool:
        """Check if primary model cost exceeds threshold (catches o1, opus tier)."""
```

### Testing Strategy
- Test factory with defaults and overrides
- Test `is_expensive_primary()` with mocked model info
- Test graceful fallback when model info unavailable

---

## Milestone 7: Summary Cache

### Goal
Implement caching for artifact summaries to avoid re-summarization. Cache key includes artifact_id + summarization_focus (same artifact may need different summaries for different questions).

### Success Criteria
- `SummaryCache` with get/set operations
- `InMemoryCache` backend (Redis is future work)
- Cache key = `artifact_id + focus_hash`
- Hit/miss tracking

### Key Changes
- New `summary_cache.py` with `CacheBackend` ABC and `InMemoryCache`
- `SummaryCache` wrapper with key generation and stats

### Testing Strategy
- Test cache key generation (different focus = different key)
- Test hit/miss tracking
- Test stats calculation

---

## Milestone 8: LLM-Driven Context Planner

### Goal
Implement LLM-driven context planning. Single LLM call receives artifact previews, user query, goal, and token budget. Returns prioritized plan with processing strategy for each artifact.

### Success Criteria
- `should_use_llm_planning()` decides when LLM planning is worth the overhead
- `ArtifactProcessor` with `plan_context()` and `execute_plan()` methods
- Uses `ModelConfig.context_planner` for planning (cheap model)
- Integrates with `SummaryCache`

### Key Decision Logic for `should_use_llm_planning()`

| Condition | Use LLM Planning? |
|-----------|-------------------|
| No artifacts | No |
| Historical artifacts present | Yes |
| Total size < 5K chars | No |
| Fits with 30% buffer + cheap model | No |
| Near/over token limit | Yes |
| Expensive primary model | Yes |
| Any artifact > 10K chars | Yes |

### Processing Strategies
- `include_full`: Small, relevant → include as-is
- `summarize`: Large, relevant → LLM summarize with focus
- `rag_extract`: Large, specific lookup → (falls back to summarize for now)
- `exclude`: Not relevant → omit

### Key Interface

```python
class ArtifactProcessor:
    def __init__(self, summary_cache: SummaryCache, model_config: ModelConfig): ...

    async def plan_context(
        self,
        artifacts: list[Artifact],
        user_query: str,
        goal: ContextGoal,
        available_tokens: int,
    ) -> ArtifactPlan:
        """Single LLM call to plan how to handle each artifact."""

    async def execute_plan(
        self,
        plan: ArtifactPlan,
        artifacts: dict[str, Artifact],
        token_budget: int,
    ) -> list[tuple[Artifact, str]]:
        """Apply strategies, return processed content in priority order."""
```

### Testing Strategy
- Test `should_use_llm_planning()` decision boundaries
- Test `plan_context()` returns valid `ArtifactPlan`
- Test `execute_plan()` respects token budget
- Test cache integration (hit vs miss)
- Mock LLM responses for deterministic testing

---

## Milestone 9: Integrate ArtifactProcessor with ContextManager

### Goal
Wire `ArtifactProcessor` into `ContextManager` for intelligent context building when appropriate.

### Success Criteria
- `ContextManager` accepts `ModelConfig` and `ArtifactProcessor` dependencies
- Uses `should_use_llm_planning()` to decide approach
- Falls back to simple inclusion when LLM planning not needed
- Metadata includes `llm_planning_used`, cache stats

### Key Changes
- `ContextManager.__call__` becomes **async** (breaking change)
- Add `_build_with_llm_planning()` method
- Rename existing logic to `_build_reasoning_context_simple()`

### Testing Strategy
- Test with/without artifact_processor
- Test fallback when planning not needed
- Test metadata includes processing stats
- Test error handling (fall back to simple on failure)

---

## Milestone 10: Fix Greedy Fit Bug

### Goal
Fix the "greedy fit" bug that skips large messages, creating incoherent context.

**Current Bug** (context_manager.py:181-189): Algorithm skips messages that don't fit and continues with older ones. Example: `[msg1, msg2, huge_msg3, msg4]` → includes `[msg1, msg2, msg4]`, breaking coherence.

### Success Criteria
- Never skip a message and continue with older ones
- Stop at first message that doesn't fit
- Metadata correctly reports excluded count

### Key Changes

```python
# BEFORE (buggy): skips large message, continues with older ones
for msg in reversed(non_system_messages):
    if total_tokens_used + msg_tokens > max_input_tokens:
        messages_excluded += 1
        continue  # BUG: creates incoherent context
    # ... include message ...

# AFTER (fixed): stops at first exclusion, maintains coherence
for msg in reversed(non_system_messages):
    if total_tokens_used + msg_tokens > max_input_tokens:
        messages_excluded = len(non_system_messages) - messages_included
        break  # STOP - don't skip and continue
    # ... include message ...
```

### Testing Strategy
- Test with conversation where middle message is huge
- Verify included messages are contiguous from end
- Test metadata reports correct exclusion count

---

## Milestone 11: Wire Everything Together

### Goal
Update all callers to use enhanced ContextManager with artifact processing, model configuration, and attachment support.

### Success Criteria
- All `build_llm_messages()` callers updated for new return type
- Dependency injection provides `ModelConfig`, `ArtifactProcessor`, `ContextManager`
- End-to-end tests pass for attachments and historical tool results
- Response metadata includes context stats

### Key Changes
- Update `main.py` and executors for new `build_llm_messages()` return type
- Add dependencies in `dependencies.py`
- Convert attachments to artifacts and pass to `Context`

### Testing Strategy
- Integration: request with attachments → response references content
- Integration: follow-up question accesses historical tool results
- Integration: verify LLM planning used/skipped appropriately
- Test cache stats in response metadata

---

## Breaking Changes (Acceptable)

1. `build_llm_messages()` returns `list[ConversationMessage]` instead of `list[dict]`
2. `Context.conversation_history` type changes to `list[ConversationMessage]`
3. `ContextManager.__call__` becomes async
4. `Context` model structure changes (uses `artifacts` instead of separate fields)

## Files Changed Summary

| File | Changes |
|------|---------|
| `artifacts.py` | NEW - Artifact, ArtifactDecision, ArtifactPlan + extraction functions |
| `model_config.py` | NEW - ModelConfig |
| `summary_cache.py` | NEW - SummaryCache + InMemoryCache |
| `artifact_processor.py` | NEW - ArtifactProcessor + should_use_llm_planning |
| `openai_protocol.py` | Add RequestAttachment model, OpenAIChatRequest.attachments field |
| `conversation_utils.py` | Add ConversationMessage model, update build_llm_messages return type, add ArtifactUsageRecord |
| `context_manager.py` | Update Context (conversation_history type, artifacts field), ContextUtilizationMetadata (add artifact tracking, planning costs), integrate processor, fix greedy fit |
| `dependencies.py` | Add ModelConfig, ArtifactProcessor dependencies |
| `main.py` + executors | Handle new build_llm_messages return type, convert attachments |

Note: Existing metadata models (`UsageMetadata`, `CostMetadata`, `ResponseMetadata`) are extended, not replaced. `ContextUtilizationMetadata` gains new fields for artifact tracking and planning costs.

## Future Work (Out of Scope)

- Redis cache backend
- Full RAG implementation (vector search)
- Attachment storage with `attachment_id` lookup
- Per-request model overrides

## Questions to Resolve

1. Cost threshold for "expensive" models (`0.00001` per input token)?
2. Cache TTL for summaries?
3. Token estimation: rough (4 chars/token) vs precise (`token_counter`)?
4. Artifact size thresholds (5K total, 10K per artifact)?
5. Limit historical artifacts to last N messages?
