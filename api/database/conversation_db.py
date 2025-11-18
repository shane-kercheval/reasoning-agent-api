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
    reasoning_events: list[dict[str, Any]] | None
    metadata: dict[str, Any]
    total_cost: float | None
    created_at: str
    sequence_number: int


@dataclass
class Conversation:
    """Represents a conversation with its messages."""

    id: UUID
    user_id: UUID | None
    title: str | None
    system_message: str
    created_at: str
    updated_at: str
    archived_at: str | None
    metadata: dict[str, Any]
    messages: list[Message]
    message_count: int | None = None


@dataclass
class MessageSearchResult:
    """Represents a message search result with conversation context."""

    message_id: UUID
    conversation_id: UUID
    conversation_title: str | None
    role: str
    content: str | None
    snippet: str | None
    relevance: float
    created_at: str
    archived: bool


class ConversationDB:
    """
    Database interface for conversation storage operations.

    Uses asyncpg connection pooling for efficient PostgreSQL access.
    All operations are async and handle errors gracefully.
    """

    def __init__(
        self,
        database_url: str,
        _test_connection: asyncpg.Connection | None = None,
    ) -> None:
        """
        Initialize the conversation database interface.

        Args:
            database_url: PostgreSQL connection URL in format:
                postgresql://user:password@host:port/database
            _test_connection: Optional test connection for transaction rollback testing
                (internal use only, for integration tests)
        """
        self.database_url = database_url
        self._pool: asyncpg.Pool | None = None
        self._test_conn: asyncpg.Connection | None = _test_connection

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
        if self._test_conn is not None:
            # In test mode, we don't use the pool
            raise RuntimeError("Cannot use pool in test mode. Use _get_connection() instead.")
        if self._pool is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._pool

    def _get_connection(self) -> asyncpg.Connection | asyncpg.Pool:
        """
        Get database connection - either test connection or pool.

        This is an internal method used by all DB operations to get the right
        connection depending on whether we're in test mode or production mode.
        """
        if self._test_conn is not None:
            return self._test_conn
        if self._pool is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._pool

    async def _execute_with_connection(self, func: callable) -> Any:
        """
        Execute a function with a database connection (no transaction).

        In test mode: uses test connection directly
        In production: acquires connection from pool

        Args:
            func: Async function that takes a connection as argument
        """
        if self._test_conn is not None:
            # Test mode: use test connection directly
            return await func(self._test_conn)
        # Production mode: acquire connection from pool
        async with self._pool.acquire() as conn:
            return await func(conn)

    async def _execute_in_transaction(self, func: callable) -> Any:
        """
        Execute a function within a transaction context.

        In test mode: executes directly (already in transaction)
        In production: acquires connection and starts transaction

        Args:
            func: Async function that takes a connection as argument
        """
        if self._test_conn is not None:
            # Test mode: already in transaction, execute directly
            return await func(self._test_conn)
        # Production mode: acquire connection and start transaction
        async with self._pool.acquire() as conn, conn.transaction():
            return await func(conn)

    async def create_conversation(
        self,
        system_message: str | None = None,
        title: str | None = None,
    ) -> UUID:
        """
        Create a new conversation record with system message.

        This creates the conversation record only. User and assistant messages
        should be added via append_messages().

        Args:
            system_message: System message for the conversation (stored in conversations table)
            title: Optional conversation title

        Returns:
            UUID of the created conversation
        """
        async def _create(conn: asyncpg.Connection) -> UUID:
            # Create conversation with system message
            return await conn.fetchval(
                """
                    INSERT INTO conversations (system_message, title, metadata)
                    VALUES ($1, $2, $3)
                    RETURNING id
                    """,
                system_message,
                title,
                {},
            )


        return await self._execute_in_transaction(_create)

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
        async def _get(conn: asyncpg.Connection) -> Conversation:
            # Fetch conversation
            conv_row = await conn.fetchrow(
                """
                SELECT id, user_id, title, system_message,
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
                       metadata, total_cost, created_at, sequence_number
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
                    metadata=row["metadata"],
                    total_cost=float(row["total_cost"]) if row["total_cost"] is not None else None,
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
                created_at=conv_row["created_at"].isoformat(),
                updated_at=conv_row["updated_at"].isoformat(),
                archived_at=conv_row["archived_at"].isoformat() if conv_row["archived_at"] else None,  # noqa: E501
                metadata=conv_row["metadata"],
                messages=messages,
            )

        return await self._execute_with_connection(_get)

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

        async def _append(conn: asyncpg.Connection) -> None:
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
            # Use clock_timestamp() instead of NOW() to get actual current time
            # (NOW() returns transaction start time in PostgreSQL transactions)
            await conn.execute(
                "UPDATE conversations SET updated_at = clock_timestamp() WHERE id = $1",
                conversation_id,
            )

        await self._execute_in_transaction(_append)

    async def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
        archive_filter: str = "active",
    ) -> tuple[list[Conversation], int]:
        """
        List conversations with pagination, ordered by most recently updated.

        Returns conversation metadata with message count for efficient listing.
        Use get_conversation() to retrieve full message history.

        Args:
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip
            archive_filter: Filter by archive status - "active", "archived", or "all"

        Returns:
            Tuple of (conversations list, total count)
        """
        # Build archive filter condition
        archive_conditions = {
            "active": "c.archived_at IS NULL",
            "archived": "c.archived_at IS NOT NULL",
            "all": "TRUE",
        }

        if archive_filter not in archive_conditions:
            raise ValueError(
                f"Invalid archive_filter: {archive_filter}. "
                f"Must be one of: active, archived, all",
            )

        archive_where = archive_conditions[archive_filter]

        async def _list(conn: asyncpg.Connection) -> tuple[list[Conversation], int]:
            # Get total count
            total = await conn.fetchval(
                f"SELECT COUNT(*) FROM conversations c WHERE {archive_where}",
            )

            # Get conversations with message count (efficient - no message loading)
            conv_rows = await conn.fetch(
                f"""
                SELECT c.id, c.user_id, c.title, c.system_message,
                       c.created_at, c.updated_at, c.archived_at, c.metadata,
                       COUNT(m.id) as message_count
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE {archive_where}
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit,
                offset,
            )

            conversations = []
            for conv_row in conv_rows:
                conversations.append(
                    Conversation(
                        id=conv_row["id"],
                        user_id=conv_row["user_id"],
                        title=conv_row["title"],
                        system_message=conv_row["system_message"],
                        created_at=conv_row["created_at"].isoformat(),
                        updated_at=conv_row["updated_at"].isoformat(),
                        archived_at=conv_row["archived_at"].isoformat() if conv_row["archived_at"] else None,  # noqa: E501
                        metadata=conv_row["metadata"],
                        messages=[],  # Empty for list endpoint (use get_conversation for full history)  # noqa: E501
                        message_count=conv_row["message_count"],
                    ),
                )

            return conversations, total

        return await self._execute_with_connection(_list)

    async def delete_conversation(
        self, conversation_id: UUID, permanent: bool = False,
    ) -> bool:
        """
        Delete a conversation (soft or hard delete).

        Soft delete: Sets archived_at timestamp. Archived conversations are excluded
        from list_conversations but can still be retrieved by ID.

        Hard delete: Permanently removes conversation and all associated messages
        from the database. This operation cannot be undone.

        Args:
            conversation_id: UUID of the conversation to delete
            permanent: If True, permanently delete; if False, soft delete (default)

        Returns:
            True if conversation was deleted, False if not found
        """
        async def _delete(conn: asyncpg.Connection) -> bool:
            if permanent:
                # Hard delete: Remove conversation and messages (cascade)
                result = await conn.execute(
                    """
                    DELETE FROM conversations
                    WHERE id = $1
                    """,
                    conversation_id,
                )
            else:
                # Soft delete: Set archived_at timestamp
                result = await conn.execute(
                    """
                    UPDATE conversations
                    SET archived_at = NOW()
                    WHERE id = $1 AND archived_at IS NULL
                    """,
                    conversation_id,
                )

            # Check if any rows were affected
            return result.split()[-1] == "1"

        return await self._execute_with_connection(_delete)

    async def delete_messages_from_sequence(
        self,
        conversation_id: UUID,
        from_sequence: int,
    ) -> bool:
        """
        Delete message at sequence number and all subsequent messages.

        Args:
            conversation_id: UUID of the conversation
            from_sequence: Sequence number to start deleting from (inclusive)

        Returns:
            True if any messages were deleted, False if conversation/sequence not found
        """
        async def _delete(conn: asyncpg.Connection) -> bool:
            # Check if conversation exists and has message at that sequence
            check_query = """
                SELECT EXISTS(
                    SELECT 1 FROM messages
                    WHERE conversation_id = $1 AND sequence_number = $2
                )
            """
            exists = await conn.fetchval(check_query, conversation_id, from_sequence)

            if not exists:
                return False

            # Delete messages at and after sequence number
            delete_query = """
                DELETE FROM messages
                WHERE conversation_id = $1 AND sequence_number >= $2
            """
            await conn.execute(delete_query, conversation_id, from_sequence)

            # Update conversation timestamp
            update_query = """
                UPDATE conversations
                SET updated_at = clock_timestamp()
                WHERE id = $1
            """
            await conn.execute(update_query, conversation_id)

            return True

        return await self._execute_in_transaction(_delete)

    async def branch_conversation(
        self,
        source_conversation_id: UUID,
        branch_at_sequence: int,
    ) -> Conversation:
        """
        Create a new conversation by copying an existing one up to a sequence number.

        Creates a new conversation with the same system message and copies all messages
        up to and including the specified sequence number. The new conversation gets
        a title based on the source conversation and preserves all other fields
        (user_id, metadata) from the source.

        Args:
            source_conversation_id: UUID of the conversation to branch from
            branch_at_sequence: Copy messages with sequence_number <= this value

        Returns:
            Full Conversation object with copied messages

        Raises:
            ValueError: If source conversation or sequence number not found
        """
        async def _branch(conn: asyncpg.Connection) -> Conversation:
            # Get source conversation
            source_query = """
                SELECT system_message, title, user_id, metadata
                FROM conversations
                WHERE id = $1 AND archived_at IS NULL
            """
            source = await conn.fetchrow(source_query, source_conversation_id)

            if not source:
                raise ValueError(
                    f"Source conversation {source_conversation_id} not found or archived",
                )

            # Check if sequence number exists
            sequence_check = """
                SELECT EXISTS(
                    SELECT 1 FROM messages
                    WHERE conversation_id = $1 AND sequence_number = $2
                )
            """
            has_sequence = await conn.fetchval(
                sequence_check, source_conversation_id, branch_at_sequence,
            )

            if not has_sequence:
                raise ValueError(
                    f"Sequence number {branch_at_sequence} not found in conversation {source_conversation_id}",  # noqa: E501
                )

            # Create new conversation title
            source_title = source['title']
            new_title = f"Branch of '{source_title}'" if source_title else "Branch of Conversation"

            # Create new conversation (preserve user_id and metadata from source)
            create_query = """
                INSERT INTO conversations (system_message, title, user_id, metadata)
                VALUES ($1, $2, $3, $4)
                RETURNING id, user_id, system_message, title, created_at, updated_at,
                          archived_at, metadata
            """
            new_conv = await conn.fetchrow(
                create_query,
                source['system_message'],
                new_title,
                source['user_id'],
                source['metadata'],
            )

            new_conv_id = new_conv['id']

            # Copy messages up to and including branch_at_sequence
            copy_query = """
                INSERT INTO messages (
                    conversation_id, role, content, reasoning_events,
                    metadata, total_cost, sequence_number
                )
                SELECT
                    $1, role, content, reasoning_events,
                    metadata, total_cost, sequence_number
                FROM messages
                WHERE conversation_id = $2 AND sequence_number <= $3
                ORDER BY sequence_number
            """
            await conn.execute(copy_query, new_conv_id, source_conversation_id, branch_at_sequence)

            # Fetch copied messages for response
            messages_query = """
                SELECT
                    id, conversation_id, role, content, reasoning_events,
                    metadata, total_cost, created_at, sequence_number
                FROM messages
                WHERE conversation_id = $1
                ORDER BY sequence_number
            """
            messages = await conn.fetch(messages_query, new_conv_id)

            # Build Message objects
            message_objs = [
                Message(
                    id=msg['id'],
                    conversation_id=msg['conversation_id'],
                    role=msg['role'],
                    content=msg['content'],
                    reasoning_events=msg['reasoning_events'],
                    metadata=msg['metadata'],
                    total_cost=float(msg['total_cost']) if msg['total_cost'] is not None else None,
                    created_at=msg['created_at'].isoformat(),
                    sequence_number=msg['sequence_number'],
                )
                for msg in messages
            ]

            return Conversation(
                id=new_conv['id'],
                user_id=new_conv['user_id'],
                title=new_conv['title'],
                system_message=new_conv['system_message'],
                created_at=new_conv['created_at'].isoformat(),
                updated_at=new_conv['updated_at'].isoformat(),
                archived_at=(
                    new_conv['archived_at'].isoformat() if new_conv['archived_at'] else None
                ),
                metadata=new_conv['metadata'],
                messages=message_objs,
            )

        return await self._execute_in_transaction(_branch)

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
        async def _update(conn: asyncpg.Connection) -> bool:
            result = await conn.execute(
                """
                UPDATE conversations
                SET title = $1, updated_at = clock_timestamp()
                WHERE id = $2 AND archived_at IS NULL
                """,
                title,
                conversation_id,
            )

            # Check if any rows were updated
            return result.split()[-1] == "1"

        return await self._execute_with_connection(_update)

    async def search_messages(
        self,
        search_phrase: str,
        limit: int = 50,
        offset: int = 0,
        archive_filter: str = "active",
    ) -> tuple[list[MessageSearchResult], int]:
        """
        Search messages using PostgreSQL full-text search with pagination.

        Uses the content_search tsvector column with GIN index for fast search.
        Returns results ordered by relevance (ts_rank_cd) then recency.

        Args:
            search_phrase: Search query (case-insensitive, supports multi-word)
            limit: Maximum number of results to return
            offset: Number of results to skip (for pagination)
            archive_filter: Filter by archive status - "active", "archived", or "all"

        Returns:
            Tuple of (search results list, total matching count)
        """
        # Build archive filter condition
        archive_conditions = {
            "active": "c.archived_at IS NULL",
            "archived": "c.archived_at IS NOT NULL",
            "all": "TRUE",
        }

        if archive_filter not in archive_conditions:
            raise ValueError(
                f"Invalid archive_filter: {archive_filter}. "
                f"Must be one of: active, archived, all",
            )

        archive_where = archive_conditions[archive_filter]

        async def _search(conn: asyncpg.Connection) -> tuple[list[MessageSearchResult], int]:
            # Get total count of matching messages
            total = await conn.fetchval(
                f"""
                SELECT COUNT(*)
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE m.content_search @@ plainto_tsquery('english', $1)
                  AND {archive_where}
                """,
                search_phrase,
            )

            # Get paginated results with ranking and snippets
            rows = await conn.fetch(
                f"""
                SELECT
                    m.id as message_id,
                    m.conversation_id,
                    c.title as conversation_title,
                    m.role,
                    m.content,
                    m.created_at,
                    c.archived_at IS NOT NULL as archived,
                    ts_rank_cd(m.content_search, query, 32) as relevance,
                    ts_headline(
                        'english',
                        COALESCE(m.content, ''),
                        query,
                        'MaxWords=50, MinWords=25, MaxFragments=3'
                    ) as snippet
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id,
                     plainto_tsquery('english', $1) query
                WHERE m.content_search @@ query
                  AND {archive_where}
                ORDER BY relevance DESC, m.created_at DESC
                LIMIT $2 OFFSET $3
                """,
                search_phrase,
                limit,
                offset,
            )

            # Convert rows to MessageSearchResult objects
            results = [
                MessageSearchResult(
                    message_id=row["message_id"],
                    conversation_id=row["conversation_id"],
                    conversation_title=row["conversation_title"],
                    role=row["role"],
                    content=row["content"],
                    snippet=row["snippet"],
                    relevance=float(row["relevance"]),
                    created_at=row["created_at"].isoformat(),
                    archived=row["archived"],
                )
                for row in rows
            ]

            return results, total

        return await self._execute_with_connection(_search)

    async def _insert_messages(
        self,
        conn: asyncpg.Connection,
        conversation_id: UUID,
        messages: list[dict[str, Any]],
        start_seq: int,
    ) -> None:
        """
        Helper method to insert messages with sequential numbering using batch insert.

        Args:
            conn: Database connection (within transaction)
            conversation_id: UUID of the conversation
            messages: Messages to insert
            start_seq: Starting sequence number
        """
        # Prepare batch data for executemany
        batch_data = [
            (
                conversation_id,
                message["role"],
                message.get("content"),
                message.get("reasoning_events"),
                message.get("metadata", {}),
                message.get("total_cost"),
                start_seq + i,
            )
            for i, message in enumerate(messages)
        ]

        # Batch insert all messages
        await conn.executemany(
            """
            INSERT INTO messages (
                conversation_id, role, content, reasoning_events,
                metadata, total_cost, sequence_number
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            batch_data,
        )
