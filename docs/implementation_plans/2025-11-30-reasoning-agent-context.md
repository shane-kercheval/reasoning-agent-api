# Reasoning Agent Context Management Refactor

## Overview

This plan refactors how the ReasoningAgent builds and manages context for LLM calls. Currently, the ReasoningAgent manually constructs context strings and loses temporal association between steps and their tool results. This refactor:

1. Introduces `ReasoningStepRecord` to track steps with their associated tool predictions and results
2. Shifts context-building responsibility from ReasoningAgent to ContextManager
3. Enables goal-aware context filtering (planning vs synthesis need different context)
4. Adds support for intelligent context summarization via LLM

## Problem Statement

**Current Issues:**

1. **Lost temporal association**: `tool_results` is a flat list-we don't know which iteration produced which result
2. **Wrong responsibility**: ReasoningAgent manually builds context strings like:
   ```python
   f"Previous reasoning:\n\n```\n{context_summary}\n```"
   f"Tool execution results:\n\n```\n{tool_summary}\n```"
   ```
   This is context management logic in the wrong place.
3. **No goal-awareness**: Planning needs different context than synthesis (e.g., planner needs all search results to pick URLs; synthesizer needs full scraped content)
4. **Context bloat**: By iteration 3, the planner receives ALL previous tool results even when they're no longer relevant for deciding next steps

**Target State:**

- ReasoningAgent builds structured `ReasoningStepRecord` objects
- ContextManager receives structured data and builds appropriate context based on goal
- Tool results stay associated with their originating step
- Context can be intelligently filtered/summarized based on the current goal

---

## Milestone 1: ReasoningStepRecord Model

### Goal
Create a structured model that captures a complete reasoning step with its outcomes, preserving the temporal association between steps and their tool results.

### Success Criteria
- `ReasoningStepRecord` model defined with step_index, thought, tool_predictions, tool_results, timestamp
- Unit tests verify model creation and serialization
- Existing tests continue to pass

### Key Changes

**New model in `reasoning_api/reasoning_models.py`:**

```python
from datetime import datetime

class ReasoningStepRecord(BaseModel):
    """
    Complete record of a reasoning step and its outcomes.

    Used internally to track the full history of reasoning iterations,
    preserving the association between what was planned and what happened.
    """
    step_index: int = Field(description="0-based index of this step in the reasoning process")
    thought: str = Field(description="The reasoning/analysis from this step")
    next_action: ReasoningAction = Field(description="What action was decided")
    tool_predictions: list[ToolPrediction] = Field(
        default_factory=list,
        description="Tools that were planned to execute"
    )
    tool_results: list[ToolResult] = Field(
        default_factory=list,
        description="Results from executed tools (matches tool_predictions order)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this step was created"
    )
```

**Note:** This is NOT a replacement for `ReasoningStep`-that model is used for LLM JSON output. `ReasoningStepRecord` wraps a step with its execution outcomes.

### Testing Strategy
- Test model instantiation with all fields
- Test default values (empty lists, auto-timestamp)
- Test serialization/deserialization
- Test that ToolResult import works correctly (it's in tools.py, may need re-export)

### Dependencies
None-this is a new model with no changes to existing code.

### Risk Factors
- Need to ensure `ToolResult` can be imported cleanly into reasoning_models.py (may need to handle circular imports)

---

## Milestone 2: Update ReasoningAgent to Use ReasoningStepRecord

### Goal
Refactor ReasoningAgent to build `ReasoningStepRecord` objects instead of separate `steps` and `tool_results` lists.

### Success Criteria
- ReasoningAgent uses `step_records: list[ReasoningStepRecord]` instead of separate lists
- Each step record captures its associated tool results
- All existing reasoning agent tests pass
- Streaming events still work correctly

### Key Changes

**Update `reasoning_context` structure:**

```python
# Before (in __init__):
self.reasoning_context = {
    "steps": [],           # list[ReasoningStep]
    "tool_results": [],    # list[ToolResult] - FLAT, loses step association
    "final_thoughts": "",
    "user_request": None,
}

# After:
self.reasoning_context = {
    "step_records": [],    # list[ReasoningStepRecord] - structured
    "user_request": None,
}
```

**Update `_execute_stream` to build records:**

After generating a ReasoningStep and executing tools, create a ReasoningStepRecord:

```python
# After tool execution completes for this iteration:
step_record = ReasoningStepRecord(
    step_index=iteration,
    thought=reasoning_step.thought,
    next_action=reasoning_step.next_action,
    tool_predictions=reasoning_step.tools_to_use,
    tool_results=tool_results,  # Results from this iteration only
)
self.reasoning_context["step_records"].append(step_record)
```

**Temporarily maintain backward compatibility:**

For this milestone, keep `_generate_reasoning_step` and `_build_reasoning_summary` working by extracting data from the new structure. This will be cleaned up in Milestone 3.

### Testing Strategy
- Verify step_records are built correctly with proper associations
- Test that step N's tool_results are in step N's record (not a flat list)
- Verify timestamp is set on each record
- Run full integration tests to ensure streaming still works
- Test multi-step scenarios where different steps have different tools

### Dependencies
- Milestone 1 (ReasoningStepRecord model)

### Risk Factors
- Need to carefully update all places that read from reasoning_context
- Streaming events reference tool_results-ensure they still work

---

## Milestone 3: Extend ContextManager Interface

### Goal
Add a new method to ContextManager that accepts structured reasoning data and builds context appropriate for a specific goal.

### Success Criteria
- New `build_reasoning_context()` method on ContextManager
- Method accepts step_records and goal (planning/synthesis)
- Returns formatted messages ready for LLM
- Existing `__call__` method unchanged (backward compatible)

### Key Changes

**Add goal enum:**

```python
class ContextGoal(str, Enum):
    """What the context will be used for."""
    PLANNING = "planning"      # Deciding next reasoning step
    SYNTHESIS = "synthesis"    # Generating final response
```

**New method signature:**

```python
def build_reasoning_context(
    self,
    model_name: str,
    conversation_history: list[dict[str, str]],
    step_records: list[ReasoningStepRecord],
    goal: ContextGoal,
    system_prompt: str | None = None,
) -> tuple[list[dict[str, str]], dict]:
    """
    Build context for reasoning agent LLM calls.

    Args:
        model_name: Target model for token counting
        conversation_history: Original user conversation
        step_records: Structured reasoning history
        goal: What this context is for (affects filtering)
        system_prompt: Optional system prompt to prepend

    Returns:
        Tuple of (messages, metadata) ready for LLM call
    """
```

**Initial implementation (no LLM filtering yet):**

For this milestone, implement straightforward formatting:
- For PLANNING: Include all step thoughts and tool results (same as current behavior)
- For SYNTHESIS: Include all step thoughts and tool results (same as current behavior)

The intelligent filtering will come in Milestone 4.

### Testing Strategy
- Test with empty step_records
- Test with multiple steps, each having tool results
- Test both PLANNING and SYNTHESIS goals produce valid messages
- Test token counting and truncation still works
- Test system_prompt is properly prepended

### Dependencies
- Milestone 2 (ReasoningStepRecord in use)

### Risk Factors
- ContextManager currently doesn't import reasoning models-need to handle this cleanly

---

## Milestone 4: Migrate ReasoningAgent to Use New ContextManager Method

### Goal
Replace the manual context-building code in ReasoningAgent with calls to `ContextManager.build_reasoning_context()`.

### Success Criteria
- `_generate_reasoning_step` uses `build_reasoning_context(goal=PLANNING)`
- `_stream_final_synthesis` uses `build_reasoning_context(goal=SYNTHESIS)`
- Manual string formatting removed from ReasoningAgent
- All tests pass

### Key Changes

**Update `_generate_reasoning_step`:**

```python
# Before: Manual message construction
messages = deepcopy(request.messages)
_, messages = pop_system_messages(messages)
messages.insert(0, {"role": "system", "content": system_prompt})

if context["steps"]:
    context_summary = "\n".join([...])
    messages.append({"role": "assistant", "content": f"Previous reasoning:\n..."})

if context["tool_results"]:
    tool_summary_parts = [...]
    messages.append({"role": "assistant", "content": f"Tool execution results:\n..."})

# After: Delegate to ContextManager
messages, metadata = self.context_manager.build_reasoning_context(
    model_name=request.model,
    conversation_history=request.messages,
    step_records=self.reasoning_context["step_records"],
    goal=ContextGoal.PLANNING,
    system_prompt=reasoning_system_prompt,
)
```

**Update `_stream_final_synthesis`:**

```python
# Replace _build_reasoning_summary with ContextManager call
messages, metadata = self.context_manager.build_reasoning_context(
    model_name=request.model,
    conversation_history=request.messages,
    step_records=self.reasoning_context["step_records"],
    goal=ContextGoal.SYNTHESIS,
    system_prompt=synthesis_prompt,
)
```

**Remove deprecated methods:**
- Remove `_build_reasoning_summary` (logic moves to ContextManager)

### Testing Strategy
- Verify identical behavior before/after migration (same prompts generated)
- Test that PLANNING context includes necessary step/tool info
- Test that SYNTHESIS context includes full tool results
- Integration tests for complete reasoning flows

### Dependencies
- Milestone 3 (new ContextManager method)

### Risk Factors
- Need to ensure metadata is still properly accumulated
- Verify context_utilization settings still respected

---

## Milestone 5: Goal-Aware Context Filtering (Preview Integration)

### Goal
Implement intelligent context filtering where PLANNING goal uses previews of large tool results, while SYNTHESIS goal uses full content.

### Success Criteria
- PLANNING context uses `preview()` for tool results above a size threshold
- SYNTHESIS context always uses full tool results
- Large tool results don't bloat planning context
- Tests verify different context sizes for planning vs synthesis

### Key Changes

**Import and use preview utility:**

```python
from reasoning_api.utils import preview

# In build_reasoning_context for PLANNING goal:
def _format_tool_result_for_planning(self, result: ToolResult) -> str:
    """Format a tool result for planning context."""
    if not result.success:
        # Always include full error messages
        return f"Tool {result.tool_name}: FAILED - {result.error}"

    # Preview large successful results
    if isinstance(result.result, (dict, list)):
        result_data = preview(result.result)  # Uses defaults: 300 chars, 3 items
        result_str = json.dumps(result_data, indent=2)
        return f"Tool {result.tool_name}: SUCCESS (preview)\n{result_str}"
    else:
        result_str = str(result.result)
        if len(result_str) > 500:
            result_str = result_str[:500] + f"... [truncated, {len(result_str)} chars total]"
        return f"Tool {result.tool_name}: SUCCESS\n{result_str}"
```

**For SYNTHESIS goal, use full content:**

```python
def _format_tool_result_for_synthesis(self, result: ToolResult) -> str:
    """Format a tool result for synthesis context (full content)."""
    if not result.success:
        return f"Tool {result.tool_name}: FAILED - {result.error}"

    if isinstance(result.result, (dict, list)):
        result_str = json.dumps(result.result, indent=2, ensure_ascii=False)
    else:
        result_str = str(result.result)
    return f"Tool {result.tool_name}: SUCCESS\n{result_str}"
```

### Testing Strategy
- Test that planning context for large tool results is smaller than synthesis context
- Test that error messages are always full (not previewed)
- Test that small tool results are not modified
- Test preview indicators appear in planning context ("[truncated...]", "[... N more items]")
- Verify synthesis context has full content for final answer generation

### Dependencies
- Milestone 4 (using new ContextManager method)
- `preview()` utility from `reasoning_api/utils.py`

### Risk Factors
- Preview might lose critical info for some planning decisions (mitigated by keeping errors full)
- May need to tune preview parameters based on real usage

---

## Milestone 6: LLM-Based Context Summarization (Optional/Future)

### Goal
Add optional LLM-based context summarization for complex multi-step reasoning where even previews result in context bloat.

**Note:** This milestone is marked as optional/future. Implement only if Milestone 5's static preview approach proves insufficient in practice.

### Success Criteria
- Optional LLM call can summarize tool results based on user's original question
- Summarization is goal-aware (planning vs synthesis)
- Can be enabled/disabled via configuration
- Adds acceptable latency (measure and report)

### Key Changes

**Add summarization option to ContextManager:**

```python
class ContextManager:
    def __init__(
        self,
        context_utilization: ContextUtilization = ContextUtilization.FULL,
        use_llm_summarization: bool = False,
        summarization_model: str = "gpt-4o-mini",
    ):
        ...
```

**Summarization prompt:**

```python
SUMMARIZATION_PROMPT = """Given the user's question and this tool result, extract only the information relevant for {goal}.

User Question: {user_question}

Tool: {tool_name}
Result: {tool_result}

For planning: Focus on what's needed to decide next steps (URLs to follow, error details, success/failure, key metadata)
For synthesis: Focus on content needed to answer the user's question

Relevant information:"""
```

### Testing Strategy
- Test with summarization disabled (default behavior)
- Test summarization produces shorter context
- Test summarization preserves critical information
- Measure latency impact
- Test error handling if summarization LLM fails

### Dependencies
- Milestone 5 (goal-aware filtering working)

### Risk Factors
- Added latency from LLM call per tool result
- Added cost
- Risk of losing critical information in summarization
- Complexity increase

---

## Implementation Notes

### Breaking Changes
This refactor intentionally changes internal structures:
- `reasoning_context["steps"]` � `reasoning_context["step_records"]`
- `reasoning_context["tool_results"]` removed (now in step_records)
- `_build_reasoning_summary` removed (logic in ContextManager)

These are internal to ReasoningAgent-no public API changes.

### File Changes Summary

| File | Changes |
|------|---------|
| `reasoning_api/reasoning_models.py` | Add `ReasoningStepRecord` model |
| `reasoning_api/context_manager.py` | Add `ContextGoal` enum, `build_reasoning_context()` method |
| `reasoning_api/executors/reasoning_agent.py` | Use step_records, delegate context building |
| `reasoning_api/utils.py` | Already exists with `preview()` |
| Tests | New tests for each milestone |

### Questions to Clarify Before Implementation

1. Should `ReasoningStepRecord` live in `reasoning_models.py` or a new file? (Recommendation: reasoning_models.py to keep related models together)

2. For the preview threshold in Milestone 5, what size (in characters or tokens) should trigger preview? (Recommendation: Start with 1000 chars, tune based on usage)

3. Should we add a `concurrent_execution` field to `ReasoningStepRecord` to track whether tools ran in parallel? (Recommendation: Yes, for debugging/observability)

4. For Milestone 6 (LLM summarization), should this be a separate class or integrated into ContextManager? (Recommendation: Defer until we know we need it)
