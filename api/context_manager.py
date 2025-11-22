"""Context manager for LLMs."""
from pydantic import BaseModel
from litellm import get_model_info, token_counter

class ContextUtilization:
    """Enum for context window utilization strategies."""

    LOW = "low"
    MEDIUM = "medium"
    FULL = "full"

class Context(BaseModel):
    """Represents the full/ideal context for LLMs."""

    conversation_history: list[dict][str, str]
    retrieved_documents: list[str]
    memory: list[dict]


class ContextManager:
    """Manages the context for LLMs."""

    def __init__(
            self,
            context_utilization: ContextUtilization = ContextUtilization.MEDIUM,
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
        """
        max_input_tokens = get_model_info(model=model_name)['max_input_tokens']
        if self.context_utilization == ContextUtilization.LOW:
            max_input_tokens = int(max_input_tokens * 0.33)
        elif self.context_utilization == ContextUtilization.MEDIUM:
            max_input_tokens = int(max_input_tokens * 0.66)
        elif self.context_utilization != ContextUtilization.FULL:
            raise ValueError(f"Unknown context utilization: {self.context_utilization}")

        # simple algorithm is to
        # - always include system messages
        # - work backwards from the most recent user/assistant messages until token limit is
        # reached (token limit is determined by context utilization setting & max_input_tokens)
        messages = context.conversation_history.copy()
        system_messages = [msg for msg in messages if msg['role'] == 'system']
        final_messages = []
        if system_messages:
            tokens_system_messages = token_counter(model=model_name, messages=system_messages)
            if tokens_system_messages >= max_input_tokens:
                raise ValueError("System messages exceed max input tokens.")
            final_messages.extend(system_messages)
        else:
            tokens_system_messages = 0

        total_tokens_used = tokens_system_messages
        tokens_user_messages = 0
        tokens_assistant_messages = 0

        for msg in reversed(messages):
            if msg in system_messages:
                continue  # already included
            msg_tokens = token_counter(model=model_name, messages=[msg])
            if total_tokens_used + msg_tokens > max_input_tokens:
                break
            final_messages.append(msg)
            if msg['role'] == 'user':
                tokens_user_messages += msg_tokens
            elif msg['role'] == 'assistant':
                tokens_assistant_messages += msg_tokens
            else:
                raise ValueError(f"Unknown message role: {msg['role']}")
            total_tokens_used += msg_tokens

        metadata = {
            "model_name": model_name,
            "context_utilization": self.context_utilization,
            "max_input_tokens": max_input_tokens,
            "input_tokens_used": total_tokens_used,
            "context_breakdown": {
                "system_messages": tokens_system_messages,
                "user_messages": tokens_user_messages,
                "assistant_messages": tokens_assistant_messages,
                # "retrieved_documents": token_counter(model=model_name, text="\n".join(context.retrieved_documents)),  # noqa: E501
                # "memory": token_counter(model=model_name, messages=context.memory),
            },
        }
        return final_messages, metadata
