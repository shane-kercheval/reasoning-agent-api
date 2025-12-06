"""Context manager for LLMs."""
from __future__ import annotations

import json
from copy import deepcopy
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from litellm import get_model_info, token_counter

from reasoning_api.openai_protocol import pop_system_messages
from reasoning_api.utils import preview

if TYPE_CHECKING:
    from reasoning_api.reasoning_models import ReasoningStepRecord
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
    """Represents the full/ideal context for LLMs."""

    conversation_history: list[dict[str, str]]
    # Future fields for additional context sources:
    # retrieved_documents: list[str]
    # memory: list[dict]


class ContextManager:
    """Manages the context for LLMs."""

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
        ) -> tuple[list[dict[str, str]], dict]:
        """
        Manage the context for the given model and context.

        Args:
            model_name: The name of the LLM model.
            context: The full/ideal context.

        Returns:
            Tuple of (filtered_messages, metadata) where:
            - filtered_messages: Messages that fit within token limit, in chronological order
            - metadata: Dict with utilization stats and token breakdown
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

        # Algorithm:
        # 1. Always include all system messages
        # 2. Work backwards from most recent messages until token limit reached
        # 3. Return messages in chronological order (most recent last)

        messages = context.conversation_history.copy()

        # Extract system messages using existing utility
        system_message_contents, non_system_messages = pop_system_messages(messages)

        # Reconstruct system messages for token counting
        system_messages = [
            {"role": "system", "content": content}
            for content in system_message_contents
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
        metadata = {
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
        # Future fields for reference:
        # "retrieved_documents": token_counter(model=model_name, text="\n".join(context.retrieved_documents)),  # noqa: E501
        # "memory": token_counter(model=model_name, messages=context.memory),

        return final_messages, metadata

    def build_reasoning_context(
        self,
        model_name: str,
        conversation_history: list[dict[str, str]],
        step_records: list[ReasoningStepRecord],
        goal: ContextGoal,
        system_prompt: str | None = None,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """
        Build context for reasoning agent LLM calls.

        This method constructs messages with reasoning history for use in either
        planning (deciding next steps) or synthesis (generating final response).

        Args:
            model_name: Target model for token counting
            conversation_history: Original user conversation
            step_records: Structured reasoning history with tool results
            goal: What this context is for (affects filtering in future milestones)
            system_prompt: Optional system prompt to prepend

        Returns:
            Tuple of (messages, metadata) ready for LLM call
        """
        messages = deepcopy(conversation_history)

        # Remove any existing system messages and track them
        _, messages = pop_system_messages(messages)

        # Prepend system prompt if provided
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        # Add context from previous steps
        if step_records:
            context_summary = "\n".join([
                f"Step {record.step_index + 1}: {record.thought}"
                for record in step_records
            ])
            messages.append({
                "role": "assistant",
                "content": f"Previous reasoning:\n\n```\n{context_summary}\n```",
            })

        # Add tool results from all step records
        # Goal-aware filtering: PLANNING uses preview, SYNTHESIS uses full content
        all_tool_results = [
            result
            for record in step_records
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

        # Apply token limit management via existing __call__ method
        ctx = Context(conversation_history=messages)
        filtered_messages, metadata = self(model_name, ctx)

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
