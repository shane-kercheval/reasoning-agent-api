"""Context manager for LLMs."""
from __future__ import annotations

import json
from copy import deepcopy
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from litellm import get_model_info, token_counter

from reasoning_api.openai_protocol import pop_system_messages
from reasoning_api.utils import preview
from reasoning_api.reasoning_models import ReasoningStepRecord

if TYPE_CHECKING:
    from reasoning_api.tools import ToolResult

# Threshold for previewing tool results (in characters)
PREVIEW_THRESHOLD = 1000


class ContextUtilization(str, Enum):
    """Enum for context window utilization strategies."""

    LOW = "low"
    MEDIUM = "medium"
    FULL = "full"


class ContextGoal(str, Enum):
    """What the context will be used for."""

    PLANNING = "planning"      # Deciding next reasoning step
    SYNTHESIS = "synthesis"    # Generating final response


class Context(BaseModel):
    """
    Represents the full/ideal context for LLMs.

    This is the unified interface for all context passed to ContextManager.
    Basic usage only requires conversation_history. Reasoning agents can
    additionally provide step_records and goal for reasoning-aware context building.
    """

    conversation_history: list[dict[str, str]]

    # Reasoning-specific fields (optional)
    step_records: list[ReasoningStepRecord] = Field(default_factory=list)
    goal: ContextGoal | None = None

    # System prompt override: if set, replaces existing system messages.
    # If None, existing system messages in conversation_history are preserved.
    system_prompt_override: str | None = None

    # Future fields for additional context sources:
    # retrieved_documents: list[str] = Field(default_factory=list)
    # memory: list[dict] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class ContextManager:
    """
    Manages the context for LLMs.

    Provides a unified interface for context management. All callers use __call__
    with a Context object. The manager handles:
    - Token limit filtering (keeps recent messages within limit)
    - Reasoning context building (when step_records provided)
    - Goal-aware filtering (preview for PLANNING, full for SYNTHESIS)
    - System prompt override
    """

    def __init__(
            self,
            context_utilization: ContextUtilization = ContextUtilization.FULL,
        ):
        """Initialize the context manager."""
        self.context_utilization = context_utilization

    def __call__(
            self,
            model_name: str,
            context: Context,
        ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """
        Manage the context for the given model and context.

        This is the unified entry point for all context management. Based on
        the Context fields, it will:
        - If step_records provided: build reasoning context with tool results
        - Apply system_prompt_override if set (otherwise preserve existing)
        - Apply token limit filtering

        Args:
            model_name: The name of the LLM model.
            context: The full/ideal context with all relevant fields.

        Returns:
            Tuple of (filtered_messages, metadata) where:
            - filtered_messages: Messages that fit within token limit, in chronological order
            - metadata: Dict with utilization stats and token breakdown
        """
        # If reasoning context is needed, build it first
        if context.step_records or context.goal:
            return self._build_reasoning_context(model_name, context)

        # Otherwise, apply standard token limit filtering
        return self._apply_token_limits(model_name, context)

    def _apply_token_limits(  # noqa: PLR0912
            self,
            model_name: str,
            context: Context,
        ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """
        Apply token limit filtering to messages.

        Algorithm:
        1. Handle system_prompt_override (replace or preserve existing)
        2. Always include all system messages
        3. Work backwards from most recent messages until token limit reached
        4. Return messages in chronological order (most recent last)
        """
        # Calculate max tokens based on utilization strategy
        model_max_tokens = get_model_info(model=model_name)['max_input_tokens']
        max_input_tokens = model_max_tokens  # Start with full model capacity
        if self.context_utilization == ContextUtilization.LOW:
            max_input_tokens = int(model_max_tokens * 0.33)
        elif self.context_utilization == ContextUtilization.MEDIUM:
            max_input_tokens = int(model_max_tokens * 0.66)
        elif self.context_utilization != ContextUtilization.FULL:
            raise ValueError(f"Unknown context utilization: {self.context_utilization}")

        messages = context.conversation_history.copy()

        # Extract system messages using existing utility
        existing_system_contents, non_system_messages = pop_system_messages(messages)

        # Handle system_prompt_override
        if context.system_prompt_override is not None:
            # Replace existing system messages with override
            system_messages = [
                {"role": "system", "content": context.system_prompt_override},
            ]
        else:
            # Preserve existing system messages
            system_messages = [
                {"role": "system", "content": content}
                for content in existing_system_contents
            ]

        # Count system message tokens and validate
        if system_messages:
            tokens_system_messages = token_counter(model=model_name, messages=system_messages)
            if tokens_system_messages >= max_input_tokens:
                raise ValueError("System messages exceed max input tokens.")
        else:
            tokens_system_messages = 0

        # Start with system messages in final list
        final_messages = system_messages.copy()

        total_tokens_used = tokens_system_messages
        tokens_user_messages = 0
        tokens_assistant_messages = 0
        messages_included = len(system_messages)
        messages_excluded = 0

        # Work backwards through non-system messages, inserting after system messages
        for msg in reversed(non_system_messages):
            msg_tokens = token_counter(model=model_name, messages=[msg])

            # Check if adding this message would exceed limit
            if total_tokens_used + msg_tokens > max_input_tokens:
                messages_excluded += 1
                continue

            # Insert after system messages to maintain chronological order
            final_messages.insert(len(system_messages), msg)

            # Track tokens by role
            if msg['role'] == 'user':
                tokens_user_messages += msg_tokens
            elif msg['role'] == 'assistant':
                tokens_assistant_messages += msg_tokens
            else:
                raise ValueError(f"Unknown message role: {msg['role']}")

            total_tokens_used += msg_tokens
            messages_included += 1

        # Build metadata
        metadata: dict[str, Any] = {
            "model_name": model_name,
            "strategy": self.context_utilization.value,
            "model_max_tokens": model_max_tokens,  # Original model context size
            "max_input_tokens": max_input_tokens,  # Strategy-adjusted limit
            "input_tokens_used": total_tokens_used,
            "messages_included": messages_included,
            "messages_excluded": messages_excluded,
            "breakdown": {
                "system_messages": tokens_system_messages,
                "user_messages": tokens_user_messages,
                "assistant_messages": tokens_assistant_messages,
            },
        }

        return final_messages, metadata

    def _build_reasoning_context(
        self,
        model_name: str,
        context: Context,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """
        Build context for reasoning agent LLM calls.

        Constructs messages with reasoning history for use in either
        planning (deciding next steps) or synthesis (generating final response).
        """
        messages = deepcopy(context.conversation_history)

        # Extract existing system messages
        existing_system_contents, non_system_messages = pop_system_messages(messages)

        # Handle system_prompt_override
        if context.system_prompt_override is not None:
            # Use override
            messages = [{"role": "system", "content": context.system_prompt_override}]
        elif existing_system_contents:
            # Preserve existing system messages
            messages = [
                {"role": "system", "content": content}
                for content in existing_system_contents
            ]
        else:
            messages = []

        # Add non-system messages
        messages.extend(non_system_messages)

        # Add context from previous steps
        if context.step_records:
            context_summary = "\n".join([
                f"Step {record.step_index + 1}: {record.thought}"
                for record in context.step_records
            ])
            messages.append({
                "role": "assistant",
                "content": f"Previous reasoning:\n\n```\n{context_summary}\n```",
            })

        # Add tool results from all step records
        # Goal-aware filtering: PLANNING uses preview, SYNTHESIS uses full content
        goal = context.goal or ContextGoal.SYNTHESIS  # Default to full content
        all_tool_results = [
            result
            for record in context.step_records
            for result in record.tool_results
        ]
        if all_tool_results:
            tool_summary_parts = []
            for result in all_tool_results:
                formatted = self._format_tool_result(result, goal)
                tool_summary_parts.append(formatted)

            tool_summary = "\n\n".join(tool_summary_parts)
            messages.append({
                "role": "assistant",
                "content": f"Tool execution results:\n\n```\n{tool_summary}\n```",
            })

        # Apply token limit management
        token_context = Context(conversation_history=messages)
        filtered_messages, metadata = self._apply_token_limits(model_name, token_context)

        # Add goal to metadata for tracking
        metadata["goal"] = goal.value

        return filtered_messages, metadata

    def _format_tool_result(self, result: ToolResult, goal: ContextGoal) -> str:
        """
        Format a tool result based on the context goal.

        For PLANNING: Use preview() for large results to reduce context size
        For SYNTHESIS: Use full content for accurate final response generation

        Errors are always shown in full regardless of goal.
        """
        # Always show full error messages
        if not result.success:
            return f"Tool {result.tool_name}: FAILED - {result.error}"

        # For PLANNING, use preview for large results
        if goal == ContextGoal.PLANNING:
            if isinstance(result.result, (dict, list)):
                # Check size before deciding to preview
                full_str = json.dumps(result.result, indent=2, ensure_ascii=False)
                if len(full_str) > PREVIEW_THRESHOLD:
                    # Use preview utility for large structured data
                    previewed = preview(result.result)
                    result_str = json.dumps(previewed, indent=2, ensure_ascii=False)
                    return f"Tool {result.tool_name}: SUCCESS (preview)\n{result_str}"
                result_str = full_str
            else:
                result_str = str(result.result)
                if len(result_str) > PREVIEW_THRESHOLD:
                    result_str = (
                        result_str[:PREVIEW_THRESHOLD]
                        + f"... [truncated, {len(str(result.result))} chars total]"
                    )
                    return f"Tool {result.tool_name}: SUCCESS (truncated)\n{result_str}"
            return f"Tool {result.tool_name}: SUCCESS\n{result_str}"

        # For SYNTHESIS, use full content
        if isinstance(result.result, (dict, list)):
            result_str = json.dumps(result.result, indent=2, ensure_ascii=False)
        else:
            result_str = str(result.result)
        return f"Tool {result.tool_name}: SUCCESS\n{result_str}"
