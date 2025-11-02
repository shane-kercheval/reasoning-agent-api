"""
Conversation utilities for handling conversation context and message storage.

Provides utilities for parsing conversation headers, building message lists for LLMs,
and storing conversation messages.
"""

from dataclasses import dataclass
from enum import Enum
from uuid import UUID

from api.database.conversation_db import ConversationDB
from api.openai_protocol import extract_system_message


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
        return ConversationContext(mode=ConversationMode.CONTINUING, conversation_id=conversation_id)
    except ValueError as e:
        raise InvalidConversationIDError(f"Invalid conversation ID format: {header_value}") from e


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
) -> None:
    """
    Store user messages + assistant response after streaming completes.

    Simple wrapper around append_messages that filters system messages
    and constructs the messages list.

    Args:
        conversation_db: Database connection
        conversation_id: UUID of conversation to append to
        request_messages: Original request messages (may include system message)
        response_content: Complete assistant response content

    Examples:
        >>> request_msgs = [
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "Hello"}
        ... ]
        >>> await store_conversation_messages(db, conv_id, request_msgs, "Hi there!")
        # Stores: [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]
    """  # noqa: E501
    # Filter out system messages (only user/assistant should be stored)
    user_messages = [m for m in request_messages if m.get("role") != "system"]
    # Add assistant response
    messages_to_store = [*user_messages, {"role": "assistant", "content": response_content}]
    # Store in database
    await conversation_db.append_messages(conversation_id, messages_to_store)
