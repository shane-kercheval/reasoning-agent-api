"""Context manager for LLMs."""
from enum import Enum

from pydantic import BaseModel
from litellm import get_model_info, token_counter

from api.openai_protocol import pop_system_messages


class ContextUtilization(str, Enum):
    """Enum for context window utilization strategies."""

    LOW = "low"
    MEDIUM = "medium"
    FULL = "full"


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
