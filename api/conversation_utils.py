"""
Conversation utilities for handling conversation context and message storage.

Provides utilities for parsing conversation headers, building message lists for LLMs,
and storing conversation messages.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any
from uuid import UUID

import litellm

from api.database.conversation_db import ConversationDB
from api.openai_protocol import extract_system_message


def merge_dicts(
    existing: dict[str, Any] | None,
    new: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Recursively merge two dictionaries, summing numeric values.

    This function traverses nested dict structures and sums numeric fields
    (int, float) while preserving non-numeric values from the newer dict.
    Useful for accumulating usage/cost data across multiple API calls.

    Args:
        existing: Existing dict or None
        new: New dict to merge or None

    Returns:
        Merged dict with numeric values summed

    Examples:
        >>> merge_dicts(None, {"count": 5})
        {"count": 5}

        >>> existing = {"tokens": 10, "cost": 0.01}
        >>> new = {"tokens": 5, "cost": 0.005}
        >>> merge_dicts(existing, new)
        {"tokens": 15, "cost": 0.015}

        >>> existing = {"usage": {"prompt": 10, "completion": 5}, "model": "gpt-4"}
        >>> new = {"usage": {"prompt": 20, "completion": 8}, "model": "gpt-4"}
        >>> merge_dicts(existing, new)
        {"usage": {"prompt": 30, "completion": 13}, "model": "gpt-4"}

        >>> existing = {"count": 10, "details": {"a": 1, "b": "text"}}
        >>> new = {"count": 5, "details": {"a": 2, "c": 3}}
        >>> merge_dicts(existing, new)
        {"count": 15, "details": {"a": 3, "b": "text", "c": 3}}
    """
    # Handle None cases
    if existing is None and new is None:
        return {}
    if existing is None:
        return new.copy() if new else {}
    if new is None:
        return existing.copy()

    # Start with a copy of existing
    merged = existing.copy()

    # Merge each key from new dict
    for key, new_value in new.items():
        if key not in merged:
            # New key - just add it
            merged[key] = new_value
        else:
            existing_value = merged[key]

            # If both are dicts, recursively merge
            if isinstance(existing_value, dict) and isinstance(new_value, dict):
                merged[key] = merge_dicts(existing_value, new_value)
            # If both are numeric (int or float), sum them
            elif isinstance(existing_value, (int, float)) and isinstance(new_value, (int, float)):
                merged[key] = existing_value + new_value
            # Otherwise, take the new value
            else:
                merged[key] = new_value

    return merged


def extract_usage(response: Any) -> dict[str, Any] | None:
    """
    Extract full usage from litellm response.

    Works with both streaming chunks and non-streaming responses.
    Returns the complete usage dict including any provider-specific fields.

    Args:
        response: LiteLLM response object (chunk or full response)

    Returns:
        Usage dict or None if no usage data available
    """
    usage_obj = getattr(response, 'usage', None)
    if usage_obj is None:
        return None

    # Convert to dict - try model_dump() first (Pydantic), fallback to __dict__
    if hasattr(usage_obj, 'model_dump'):
        return usage_obj.model_dump()
    return usage_obj.__dict__


def calculate_cost(response: Any) -> dict[str, float] | None:
    """
    Calculate cost from litellm response.

    Uses litellm.completion_cost() to get total cost, then calculates
    per-token costs based on usage data.

    Args:
        response: LiteLLM response object

    Returns:
        Cost dict with prompt_cost, completion_cost, total_cost or None
    """
    try:
        total_cost = litellm.completion_cost(completion_response=response)

        # Get usage for per-token cost calculation
        usage = getattr(response, 'usage', None)
        if usage is None or total_cost == 0:
            return {
                "prompt_cost": 0.0,
                "completion_cost": 0.0,
                "total_cost": total_cost,
            }

        # Calculate per-token costs
        total_tokens = getattr(usage, 'total_tokens', 0)
        prompt_tokens = getattr(usage, 'prompt_tokens', 0)
        completion_tokens = getattr(usage, 'completion_tokens', 0)

        if total_tokens > 0:
            prompt_cost = (total_cost / total_tokens) * prompt_tokens
            completion_cost = (total_cost / total_tokens) * completion_tokens
        else:
            prompt_cost = 0.0
            completion_cost = 0.0

        return {
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": total_cost,
        }

    except Exception:
        # If cost calculation fails, return None
        return None


def build_metadata_from_response(response: Any) -> dict[str, Any]:
    """
    Build metadata dict from litellm response.

    Extracts usage, cost, and model information from a litellm response
    (streaming chunk or non-streaming response).

    Args:
        response: LiteLLM response object (chunk or full response)

    Returns:
        Metadata dict with usage, cost, and model fields (only includes fields that are present)

    Example:
        >>> metadata = build_metadata_from_response(chunk)
        >>> metadata
        {
            "usage": {
                "prompt_tokens": 17,
                "completion_tokens": 8,
                "total_tokens": 25,
                "completion_tokens_details": {...}
            },
            "cost": {
                "prompt_cost": 0.000015,
                "completion_cost": 0.000075,
                "total_cost": 0.000090
            },
            "model": "gpt-4o-mini"
        }
    """
    metadata = {}

    # Extract usage
    usage = extract_usage(response)
    if usage:
        metadata["usage"] = usage

    # Calculate cost
    cost = calculate_cost(response)
    if cost:
        metadata["cost"] = cost

    # Extract model
    if hasattr(response, 'model'):
        metadata["model"] = response.model

    return metadata


class ConversationError(Exception):
    """Base exception for conversation operations."""

    pass


class SystemMessageInContinuationError(ConversationError):
    """Raised when system message is provided in a continuation request."""

    pass


class ConversationNotFoundError(ConversationError):
    """Raised when conversation doesn't exist in database."""

    pass


class InvalidConversationIDError(ConversationError):
    """Raised when conversation ID header has invalid UUID format."""

    pass


class ConversationMode(str, Enum):
    """Conversation mode based on X-Conversation-ID header."""

    STATELESS = "stateless"  # No X-Conversation-ID header
    NEW = "new"  # Header is "" or "null"
    CONTINUING = "continuing"  # Header is valid UUID


@dataclass
class ConversationContext:
    """Parsed conversation context from headers."""

    mode: ConversationMode
    conversation_id: UUID | None


def parse_conversation_header(header_value: str | None) -> ConversationContext:
    """
    Parse X-Conversation-ID header into conversation context.

    Args:
        header_value: Value of X-Conversation-ID header

    Returns:
        ConversationContext with mode and optional conversation_id

    Raises:
        ValueError: If header has invalid UUID format

    Examples:
        >>> parse_conversation_header(None)
        ConversationContext(mode=<ConversationMode.STATELESS: 'stateless'>, conversation_id=None)

        >>> parse_conversation_header("")
        ConversationContext(mode=<ConversationMode.NEW: 'new'>, conversation_id=None)

        >>> parse_conversation_header("null")
        ConversationContext(mode=<ConversationMode.NEW: 'new'>, conversation_id=None)

        >>> parse_conversation_header("550e8400-e29b-41d4-a716-446655440000")
        ConversationContext(mode=<ConversationMode.CONTINUING: 'continuing'>, ...)
    """
    # No header = stateless mode
    if header_value is None:
        return ConversationContext(mode=ConversationMode.STATELESS, conversation_id=None)

    # Empty string or "null" = new conversation
    if header_value in ("", "null"):
        return ConversationContext(mode=ConversationMode.NEW, conversation_id=None)

    # Valid UUID = continuing conversation
    try:
        conversation_id = UUID(header_value)
        return ConversationContext(
            mode=ConversationMode.CONTINUING,
            conversation_id=conversation_id,
        )
    except ValueError as e:
        raise InvalidConversationIDError(
            f"Invalid conversation ID format: {header_value}",
        ) from e


async def build_llm_messages(
    request_messages: list[dict],
    conversation_ctx: ConversationContext,
    conversation_db: ConversationDB | None,
) -> list[dict]:
    """
    Build complete message list for LLM based on conversation context.

    This function always returns a complete message list ready to send to the LLM,
    with system message included in the list (not returned separately).

    Args:
        request_messages: Messages from incoming request
        conversation_ctx: Parsed conversation context
        conversation_db: Database connection (can be None for stateless mode)

    Returns:
        Complete list of messages to send to LLM (includes system message)

    Raises:
        ValueError: If system message in continuation request
        ValueError: If conversation not found

    Examples:
        Stateless mode:
            >>> ctx = ConversationContext(ConversationMode.STATELESS, None)
            >>> messages = [{"role": "user", "content": "Hello"}]
            >>> result = await build_llm_messages(messages, ctx, None)
            >>> result == messages
            True

        New conversation:
            >>> ctx = ConversationContext(ConversationMode.NEW, None)
            >>> messages = [
            ...     {"role": "system", "content": "You are helpful."},
            ...     {"role": "user", "content": "Hi"}
            ... ]
            >>> result = await build_llm_messages(messages, ctx, db)
            >>> result == messages  # System message is IN the list
            True
    """
    system_message = extract_system_message(request_messages)

    # Stateless or new conversation: use request messages as-is
    if conversation_ctx.mode in (ConversationMode.STATELESS, ConversationMode.NEW):
        return request_messages

    # Continuing conversation
    if conversation_ctx.mode == ConversationMode.CONTINUING:
        # Validate no system message in continuation
        if system_message is not None:
            raise SystemMessageInContinuationError(
                "System messages are not allowed when continuing a conversation. "
                f"The system message for conversation {conversation_ctx.conversation_id} "
                "was set during creation and cannot be changed.",
            )

        if conversation_db is None:
            raise ValueError("Database connection required for continuing conversation")

        # Load conversation history
        try:
            conversation = await conversation_db.get_conversation(conversation_ctx.conversation_id)
        except Exception as e:
            raise ConversationNotFoundError(
                f"Conversation not found: {conversation_ctx.conversation_id}",
            ) from e

        # Build complete message list: [system] + [history] + [new messages]
        messages_for_llm = []

        # Add system message from conversation
        messages_for_llm.append({"role": "system", "content": conversation.system_message})

        # Add historical messages
        for msg in conversation.messages:
            messages_for_llm.append({"role": msg.role, "content": msg.content})

        # Add new user/assistant messages (filter out any system messages)
        user_messages = [m for m in request_messages if m.get("role") != "system"]
        messages_for_llm.extend(user_messages)

        return messages_for_llm

    # Should never reach here
    raise ValueError(f"Unknown conversation mode: {conversation_ctx.mode}")


async def store_conversation_messages(
    conversation_db: ConversationDB,
    conversation_id: UUID,
    request_messages: list[dict],
    response_content: str,
    response_metadata: dict | None = None,
    reasoning_events: list[dict] | None = None,
) -> None:
    """
    Store user messages + assistant response after streaming completes.

    Simple wrapper around append_messages that filters system messages
    and constructs the messages list. Assistant message includes usage and cost metadata.

    Args:
        conversation_db: Database connection
        conversation_id: UUID of conversation to append to
        request_messages: Original request messages (may include system message)
        response_content: Complete assistant response content
        response_metadata: Optional metadata dict with 'usage' and 'cost' fields
        reasoning_events: Optional list of reasoning event dicts collected during execution

    Examples:
        >>> request_msgs = [
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "Hello"}
        ... ]
        >>> metadata = {"usage": {...}, "cost": {...}}
        >>> events = [{"type": "planning", "step_iteration": 1, ...}]
        >>> await store_conversation_messages(db, conv_id, request_msgs, "Hi!", metadata, events)
        # Stores user message with empty metadata, assistant with usage/cost/reasoning_events
    """
    # Filter out system messages (only user/assistant should be stored)
    user_messages = [m for m in request_messages if m.get("role") != "system"]

    # Extract total_cost from nested metadata
    total_cost = None
    if response_metadata and 'cost' in response_metadata:
        total_cost = response_metadata['cost'].get('total_cost')

    # Add assistant response with metadata, total_cost, and reasoning_events
    assistant_message = {
        "role": "assistant",
        "content": response_content,
        "metadata": response_metadata or {},
        "total_cost": total_cost,
        "reasoning_events": reasoning_events,
    }

    # Store in database
    messages_to_store = [*user_messages, assistant_message]
    await conversation_db.append_messages(conversation_id, messages_to_store)
