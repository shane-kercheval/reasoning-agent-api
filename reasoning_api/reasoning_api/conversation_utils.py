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
from pydantic import BaseModel

from reasoning_api.database.conversation_db import ConversationDB
from reasoning_api.openai_protocol import extract_system_message
from reasoning_api.context_manager import ContextUtilizationMetadata


class UsageMetadata(BaseModel):
    """Token usage from LLM response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Provider-specific details (OpenAI, Anthropic vary in structure)
    completion_tokens_details: dict[str, int] | None = None
    prompt_tokens_details: dict[str, int] | None = None


class CostMetadata(BaseModel):
    """Cost breakdown from litellm.completion_cost()."""

    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    total_cost: float = 0.0


class ResponseMetadata(BaseModel):
    """
    Complete metadata accumulated during request execution.

    Aggregates usage, cost, and context info from LLM responses.
    """

    usage: UsageMetadata | None = None
    cost: CostMetadata | None = None
    model: str | None = None
    routing_path: str | None = None  # "passthrough", "reasoning", "orchestration"
    context_utilization: ContextUtilizationMetadata | None = None


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


def build_metadata_from_response(response: Any) -> ResponseMetadata:
    """
    Build ResponseMetadata from litellm response.

    Extracts usage, cost, and model information from a litellm response
    (streaming chunk or non-streaming response).

    Args:
        response: LiteLLM response object (chunk or full response)

    Returns:
        ResponseMetadata with usage, cost, and model fields

    Example:
        >>> metadata = build_metadata_from_response(chunk)
        >>> metadata.usage.prompt_tokens
        17
        >>> metadata.cost.total_cost
        0.000090
    """
    usage_model = None
    cost_model = None
    model = None

    # Extract usage
    usage_dict = extract_usage(response)
    if usage_dict:
        usage_model = UsageMetadata.model_validate(usage_dict)

    # Calculate cost
    cost_dict = calculate_cost(response)
    if cost_dict:
        cost_model = CostMetadata.model_validate(cost_dict)

    # Extract model
    if hasattr(response, 'model'):
        model = response.model

    return ResponseMetadata(usage=usage_model, cost=cost_model, model=model)


class ConversationError(Exception):
    """Base exception for conversation operations."""

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

    System messages are always taken from the request (never stored or retrieved
    from the database). This allows clients to change system instructions
    mid-conversation.

    Args:
        request_messages: Messages from incoming request
        conversation_ctx: Parsed conversation context
        conversation_db: Database connection (can be None for stateless mode)

    Returns:
        Complete list of messages to send to LLM (includes system message)

    Raises:
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

        Continuing conversation with system message:
            >>> ctx = ConversationContext(ConversationMode.CONTINUING, uuid)
            >>> messages = [
            ...     {"role": "system", "content": "You are helpful."},
            ...     {"role": "user", "content": "Hi"}
            ... ]
            >>> result = await build_llm_messages(messages, ctx, db)
            >>> result[0]["role"] == "system"  # System message from request
            True
    """
    # Stateless or new conversation: use request messages as-is
    if conversation_ctx.mode in (ConversationMode.STATELESS, ConversationMode.NEW):
        return request_messages

    # Continuing conversation
    if conversation_ctx.mode == ConversationMode.CONTINUING:
        if conversation_db is None:
            raise ValueError("Database connection required for continuing conversation")

        # Load conversation history
        try:
            conversation = await conversation_db.get_conversation(conversation_ctx.conversation_id)
        except Exception as e:
            raise ConversationNotFoundError(
                f"Conversation not found: {conversation_ctx.conversation_id}",
            ) from e

        # Build complete message list
        messages_for_llm = []

        # Extract system message from REQUEST (not from database)
        system_message = extract_system_message(request_messages)
        # Add system message from request if provided
        if system_message is not None:
            messages_for_llm.append({"role": "system", "content": system_message})

        # Add historical messages from database (user/assistant only, no system)
        for msg in conversation.messages:
            messages_for_llm.append({"role": msg.role, "content": msg.content})

        # Add new user/assistant messages (filter out system since we already added it)
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
    response_metadata: ResponseMetadata | None = None,
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
        response_metadata: Optional ResponseMetadata with usage and cost fields
        reasoning_events: Optional list of reasoning event dicts collected during execution

    Examples:
        >>> request_msgs = [
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "Hello"}
        ... ]
        >>> metadata = ResponseMetadata(usage=UsageMetadata(...), cost=CostMetadata(...))
        >>> events = [{"type": "planning", "step_iteration": 1, ...}]
        >>> await store_conversation_messages(db, conv_id, request_msgs, "Hi!", metadata, events)
        # Stores user message with empty metadata, assistant with usage/cost/reasoning_events
    """
    # Filter out system messages (only user/assistant should be stored)
    non_system_messages = [m for m in request_messages if m.get("role") != "system"]

    # Extract total_cost from ResponseMetadata model
    total_cost = None
    if response_metadata and response_metadata.cost:
        total_cost = response_metadata.cost.total_cost

    # Add assistant response with metadata, total_cost, and reasoning_events
    # Convert Pydantic model to dict for JSONB storage
    response_message = {
        "role": "assistant",
        "content": response_content,
        "metadata": response_metadata.model_dump() if response_metadata else {},
        "total_cost": total_cost,
        "reasoning_events": reasoning_events,
    }

    # Store in database
    messages_to_store = [*non_system_messages, response_message]
    await conversation_db.append_messages(conversation_id, messages_to_store)
