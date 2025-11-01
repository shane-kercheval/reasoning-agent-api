"""
Conversation database layer using asyncpg for PostgreSQL access.

Provides CRUD operations for conversation storage with atomic sequence numbering
and connection pooling for optimal performance.
"""

import asyncpg
import json
from typing import Any
from uuid import UUID
from dataclasses import dataclass


@dataclass
class Message:
    """Represents a message in a conversation."""

    id: UUID
    conversation_id: UUID
    role: str
    content: str | None
    reasoning_events: dict[str, Any] | None
    tool_calls: dict[str, Any] | None
    metadata: dict[str, Any]
    created_at: str
    sequence_number: int


@dataclass
class Conversation:
    """Represents a conversation with its messages."""

    id: UUID
    user_id: UUID | None
    title: str | None
    system_message: str
    routing_mode: str | None
    created_at: str
    updated_at: str
    archived_at: str | None
    metadata: dict[str, Any]
    messages: list[Message]


class ConversationDB:
    """
    Database interface for conversation storage operations.

    Uses asyncpg connection pooling for efficient PostgreSQL access.
    All operations are async and handle errors gracefully.
    """

    def __init__(self, database_url: str) -> None:
        """
        Initialize the conversation database interface.

        Args:
            database_url: PostgreSQL connection URL in format:
                postgresql://user:password@host:port/database
        """
        self.database_url = database_url
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """
        Create connection pool to the database.

        Should be called once on application startup.
        """
        if self._pool is None:
            # Parse asyncpg URL from sqlalchemy URL format if needed
            db_url = self.database_url
            if db_url.startswith("postgresql+asyncpg://"):
                db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

            async def init_connection(conn: asyncpg.Connection) -> None:
                """Initialize connection with JSONB codec."""
                await conn.set_type_codec(
                    "jsonb",
                    encoder=json.dumps,
                    decoder=json.loads,
                    schema="pg_catalog",
                )

            self._pool = await asyncpg.create_pool(
                db_url,
                min_size=2,
                max_size=10,
                command_timeout=60,
                init=init_connection,
            )

    async def disconnect(self) -> None:
        """
        Close connection pool.

        Should be called on application shutdown.
        """
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def pool(self) -> asyncpg.Pool:
        """Get the connection pool, raising error if not connected."""
        if self._pool is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._pool

    async def create_conversation(
        self,
        messages: list[dict[str, Any]],
        system_message: str = "You are a helpful assistant.",
        routing_mode: str | None = None,
        title: str | None = None,
    ) -> UUID:
        """
        Create a new conversation with initial messages.

        Args:
            messages: Initial messages to store (user/assistant messages only, no system)
            system_message: System message for the conversation (stored in conversations table)
            routing_mode: Routing mode used for first request (for analytics)
            title: Optional conversation title (auto-generated from first message if None)

        Returns:
            UUID of the created conversation

        Raises:
            ValueError: If messages contain a system message
        """
        # Validate no system messages in messages list
        if any(msg.get("role") == "system" for msg in messages):
            raise ValueError(
                "System messages should not be in messages list. "
                "Pass system message via system_message parameter.",
            )

        async with self.pool.acquire() as conn, conn.transaction():
            # Create conversation
            conversation_id = await conn.fetchval(
                """
                    INSERT INTO conversations (system_message, routing_mode, title, metadata)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id
                    """,
                system_message,
                routing_mode,
                title,
                {},
            )

            # Insert messages if any
            if messages:
                await self._insert_messages(conn, conversation_id, messages, start_seq=1)

            return conversation_id

    async def get_conversation(self, conversation_id: UUID) -> Conversation:
        """
        Retrieve a conversation with all its messages.

        Args:
            conversation_id: UUID of the conversation to retrieve

        Returns:
            Conversation object with messages ordered by sequence number

        Raises:
            ValueError: If conversation not found
        """
        async with self.pool.acquire() as conn:
            # Fetch conversation
            conv_row = await conn.fetchrow(
                """
                SELECT id, user_id, title, system_message, routing_mode,
                       created_at, updated_at, archived_at, metadata
                FROM conversations
                WHERE id = $1
                """,
                conversation_id,
            )

            if conv_row is None:
                raise ValueError(f"Conversation {conversation_id} not found")

            # Fetch messages
            message_rows = await conn.fetch(
                """
                SELECT id, conversation_id, role, content, reasoning_events,
                       tool_calls, metadata, created_at, sequence_number
                FROM messages
                WHERE conversation_id = $1
                ORDER BY sequence_number ASC
                """,
                conversation_id,
            )

            # Convert to Message objects
            messages = [
                Message(
                    id=row["id"],
                    conversation_id=row["conversation_id"],
                    role=row["role"],
                    content=row["content"],
                    reasoning_events=row["reasoning_events"],
                    tool_calls=row["tool_calls"],
                    metadata=row["metadata"],
                    created_at=row["created_at"].isoformat(),
                    sequence_number=row["sequence_number"],
                )
                for row in message_rows
            ]

            return Conversation(
                id=conv_row["id"],
                user_id=conv_row["user_id"],
                title=conv_row["title"],
                system_message=conv_row["system_message"],
                routing_mode=conv_row["routing_mode"],
                created_at=conv_row["created_at"].isoformat(),
                updated_at=conv_row["updated_at"].isoformat(),
                archived_at=conv_row["archived_at"].isoformat() if conv_row["archived_at"] else None,  # noqa: E501
                metadata=conv_row["metadata"],
                messages=messages,
            )

    async def append_messages(
        self,
        conversation_id: UUID,
        messages: list[dict[str, Any]],
    ) -> None:
        """
        Append messages to an existing conversation with atomic sequence numbering.

        Uses FOR UPDATE lock on conversation row to prevent race conditions
        when multiple requests append to the same conversation concurrently.

        Args:
            conversation_id: UUID of the conversation to append to
            messages: List of messages to append (user/assistant messages)

        Raises:
            ValueError: If conversation not found or messages contain system message
        """
        # Validate no system messages
        if any(msg.get("role") == "system" for msg in messages):
            raise ValueError(
                "Cannot append system messages to conversation. "
                "System message is set once on creation.",
            )

        async with self.pool.acquire() as conn, conn.transaction():
            # Lock conversation row to prevent concurrent appends
            conv_exists = await conn.fetchval(
                "SELECT id FROM conversations WHERE id = $1 FOR UPDATE",
                conversation_id,
            )

            if conv_exists is None:
                raise ValueError(f"Conversation {conversation_id} not found")

            # Get next sequence number
            result = await conn.fetchrow(
                """
                    SELECT COALESCE(MAX(sequence_number), 0) + 1 as next_seq
                    FROM messages WHERE conversation_id = $1
                    """,
                conversation_id,
            )
            next_seq = result["next_seq"]

            # Insert messages
            await self._insert_messages(conn, conversation_id, messages, start_seq=next_seq)

            # Update conversation timestamp
            await conn.execute(
                "UPDATE conversations SET updated_at = NOW() WHERE id = $1",
                conversation_id,
            )

    async def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Conversation], int]:
        """
        List conversations with pagination.

        Args:
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip

        Returns:
            Tuple of (conversations list, total count)
        """
        async with self.pool.acquire() as conn:
            # Get total count
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM conversations WHERE archived_at IS NULL",
            )

            # Get conversations
            conv_rows = await conn.fetch(
                """
                SELECT id, user_id, title, system_message, routing_mode,
                       created_at, updated_at, archived_at, metadata
                FROM conversations
                WHERE archived_at IS NULL
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit,
                offset,
            )

            conversations = []
            for conv_row in conv_rows:
                # Fetch messages for each conversation
                message_rows = await conn.fetch(
                    """
                    SELECT id, conversation_id, role, content, reasoning_events,
                           tool_calls, metadata, created_at, sequence_number
                    FROM messages
                    WHERE conversation_id = $1
                    ORDER BY sequence_number ASC
                    """,
                    conv_row["id"],
                )

                messages = [
                    Message(
                        id=row["id"],
                        conversation_id=row["conversation_id"],
                        role=row["role"],
                        content=row["content"],
                        reasoning_events=row["reasoning_events"],
                        tool_calls=row["tool_calls"],
                        metadata=row["metadata"],
                        created_at=row["created_at"].isoformat(),
                        sequence_number=row["sequence_number"],
                    )
                    for row in message_rows
                ]

                conversations.append(
                    Conversation(
                        id=conv_row["id"],
                        user_id=conv_row["user_id"],
                        title=conv_row["title"],
                        system_message=conv_row["system_message"],
                        routing_mode=conv_row["routing_mode"],
                        created_at=conv_row["created_at"].isoformat(),
                        updated_at=conv_row["updated_at"].isoformat(),
                        archived_at=None,  # filtered out by query
                        metadata=conv_row["metadata"],
                        messages=messages,
                    ),
                )

            return conversations, total

    async def delete_conversation(self, conversation_id: UUID) -> bool:
        """
        Soft-delete a conversation by setting archived_at timestamp.

        Args:
            conversation_id: UUID of the conversation to delete

        Returns:
            True if conversation was deleted, False if not found
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE conversations
                SET archived_at = NOW()
                WHERE id = $1 AND archived_at IS NULL
                """,
                conversation_id,
            )

            # Check if any rows were updated
            return result.split()[-1] == "1"

    async def update_conversation_title(
        self,
        conversation_id: UUID,
        title: str,
    ) -> bool:
        """
        Update the title of a conversation.

        Args:
            conversation_id: UUID of the conversation to update
            title: New title for the conversation

        Returns:
            True if conversation was updated, False if not found
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE conversations
                SET title = $1, updated_at = NOW()
                WHERE id = $2 AND archived_at IS NULL
                """,
                title,
                conversation_id,
            )

            # Check if any rows were updated
            return result.split()[-1] == "1"

    async def _insert_messages(
        self,
        conn: asyncpg.Connection,
        conversation_id: UUID,
        messages: list[dict[str, Any]],
        start_seq: int,
    ) -> None:
        """
        Helper method to insert messages with sequential numbering.

        Args:
            conn: Database connection (within transaction)
            conversation_id: UUID of the conversation
            messages: Messages to insert
            start_seq: Starting sequence number
        """
        for i, message in enumerate(messages):
            await conn.execute(
                """
                INSERT INTO messages (
                    conversation_id, role, content, reasoning_events,
                    tool_calls, metadata, sequence_number
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                conversation_id,
                message["role"],
                message.get("content"),
                message.get("reasoning_events"),
                message.get("tool_calls"),
                message.get("metadata", {}),
                start_seq + i,
            )
