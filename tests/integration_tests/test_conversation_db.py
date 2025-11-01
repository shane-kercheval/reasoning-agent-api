"""
Integration tests for ConversationDB CRUD operations.

Tests the full database layer with a real PostgreSQL connection.
Requires postgres-reasoning container to be running on port 5434.

Run with pytest: uv run pytest tests/integration_tests/test_conversation_db.py -v
Or standalone: uv run python tests/integration_tests/test_conversation_db.py
"""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4, UUID

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from api.database import ConversationDB


DATABASE_URL = "postgresql://reasoning_user:reasoning_dev_password123@localhost:5434/reasoning"


async def cleanup_db(db: ConversationDB) -> None:
    """Clean up all test data."""
    async with db.pool.acquire() as conn:
        await conn.execute("DELETE FROM messages")
        await conn.execute("DELETE FROM conversations")


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_create_and_get_conversation() -> None:
    """Test creating and retrieving a conversation."""
    db = ConversationDB(DATABASE_URL)
    await db.connect()

    try:
        # Create conversation
        conv_id = await db.create_conversation(
            messages=[{"role": "user", "content": "Hello"}],
            system_message="You are helpful.",
            routing_mode="passthrough",
        )
        assert isinstance(conv_id, UUID)
        print(f"✓ Created conversation: {conv_id}")

        # Get conversation
        conv = await db.get_conversation(conv_id)
        assert conv.id == conv_id
        assert conv.system_message == "You are helpful."
        assert len(conv.messages) == 1
        assert conv.messages[0].content == "Hello"
        print(f"✓ Retrieved conversation with {len(conv.messages)} message(s)")

    finally:
        await cleanup_db(db)
        await db.disconnect()


@pytest.mark.asyncio
async def test_append_messages() -> None:
    """Test appending messages to a conversation."""
    db = ConversationDB(DATABASE_URL)
    await db.connect()

    try:
        # Create conversation
        conv_id = await db.create_conversation(
            messages=[{"role": "user", "content": "Hi"}],
        )

        # Append messages
        await db.append_messages(
            conv_id,
            [
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
            ],
        )

        # Verify
        conv = await db.get_conversation(conv_id)
        assert len(conv.messages) == 3
        assert conv.messages[0].sequence_number == 1
        assert conv.messages[1].sequence_number == 2
        assert conv.messages[2].sequence_number == 3
        print("✓ Appended messages with correct sequence numbers")

    finally:
        await cleanup_db(db)
        await db.disconnect()


@pytest.mark.asyncio
async def test_concurrent_appends() -> None:
    """Test concurrent message appends don't create duplicate sequence numbers."""
    db = ConversationDB(DATABASE_URL)
    await db.connect()

    try:
        # Create conversation
        conv_id = await db.create_conversation(messages=[])

        # Concurrent appends
        async def append_msg(num: int) -> None:
            await db.append_messages(
                conv_id,
                [{"role": "user", "content": f"Message {num}"}],
            )

        await asyncio.gather(*[append_msg(i) for i in range(5)])

        # Verify unique sequence numbers
        conv = await db.get_conversation(conv_id)
        assert len(conv.messages) == 5
        seq_nums = [msg.sequence_number for msg in conv.messages]
        assert seq_nums == [1, 2, 3, 4, 5]
        print(f"✓ Concurrent appends created unique sequence numbers: {seq_nums}")

    finally:
        await cleanup_db(db)
        await db.disconnect()


@pytest.mark.asyncio
async def test_list_and_delete() -> None:
    """Test listing and deleting conversations."""
    db = ConversationDB(DATABASE_URL)
    await db.connect()

    try:
        # Create 3 conversations
        await db.create_conversation(messages=[], title="First")
        id2 = await db.create_conversation(messages=[], title="Second")
        await db.create_conversation(messages=[], title="Third")

        # List all
        convs, total = await db.list_conversations()
        assert total == 3
        assert len(convs) == 3
        print(f"✓ Listed {total} conversations")

        # Delete one
        result = await db.delete_conversation(id2)
        assert result is True
        print(f"✓ Deleted conversation {id2}")

        # List again (should have 2 now)
        convs, total = await db.list_conversations()
        assert total == 2
        print(f"✓ After deletion, {total} conversations remain")

    finally:
        await cleanup_db(db)
        await db.disconnect()


@pytest.mark.asyncio
async def test_update_title() -> None:
    """Test updating conversation title."""
    db = ConversationDB(DATABASE_URL)
    await db.connect()

    try:
        # Create conversation
        conv_id = await db.create_conversation(messages=[], title="Old Title")

        # Update title
        result = await db.update_conversation_title(conv_id, "New Title")
        assert result is True

        # Verify
        conv = await db.get_conversation(conv_id)
        assert conv.title == "New Title"
        print(f"✓ Updated title to '{conv.title}'")

    finally:
        await cleanup_db(db)
        await db.disconnect()


# =============================================================================
# Error Cases - System Message Validation
# =============================================================================


@pytest.mark.asyncio
async def test_create_rejects_system_message_in_messages() -> None:
    """Test that system messages in messages list are rejected."""
    db = ConversationDB(DATABASE_URL)
    await db.connect()

    try:
        with pytest.raises(ValueError, match="System messages should not be in messages list"):
            await db.create_conversation(
                messages=[
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                ],
            )
    finally:
        await cleanup_db(db)
        await db.disconnect()


@pytest.mark.asyncio
async def test_append_rejects_system_message() -> None:
    """Test that system messages cannot be appended."""
    db = ConversationDB(DATABASE_URL)
    await db.connect()

    try:
        conv_id = await db.create_conversation(messages=[])

        with pytest.raises(ValueError, match="Cannot append system messages"):
            await db.append_messages(
                conv_id,
                [{"role": "system", "content": "New system message"}],
            )
    finally:
        await cleanup_db(db)
        await db.disconnect()


# =============================================================================
# Error Cases - Not Found
# =============================================================================


@pytest.mark.asyncio
async def test_get_conversation_not_found() -> None:
    """Test getting non-existent conversation raises error."""
    db = ConversationDB(DATABASE_URL)
    await db.connect()

    try:
        fake_id = uuid4()
        with pytest.raises(ValueError, match=f"Conversation {fake_id} not found"):
            await db.get_conversation(fake_id)
    finally:
        await db.disconnect()


@pytest.mark.asyncio
async def test_append_to_nonexistent_conversation() -> None:
    """Test appending to non-existent conversation raises error."""
    db = ConversationDB(DATABASE_URL)
    await db.connect()

    try:
        fake_id = uuid4()
        with pytest.raises(ValueError, match=f"Conversation {fake_id} not found"):
            await db.append_messages(fake_id, [{"role": "user", "content": "Hello"}])
    finally:
        await db.disconnect()


@pytest.mark.asyncio
async def test_delete_nonexistent_conversation() -> None:
    """Test deleting non-existent conversation returns False."""
    db = ConversationDB(DATABASE_URL)
    await db.connect()

    try:
        fake_id = uuid4()
        result = await db.delete_conversation(fake_id)
        assert result is False
    finally:
        await db.disconnect()


@pytest.mark.asyncio
async def test_update_title_nonexistent_conversation() -> None:
    """Test updating title of non-existent conversation returns False."""
    db = ConversationDB(DATABASE_URL)
    await db.connect()

    try:
        fake_id = uuid4()
        result = await db.update_conversation_title(fake_id, "New Title")
        assert result is False
    finally:
        await db.disconnect()


# =============================================================================
# Edge Cases - JSONB Fields
# =============================================================================


@pytest.mark.asyncio
async def test_message_with_reasoning_events() -> None:
    """Test storing and retrieving messages with reasoning events."""
    db = ConversationDB(DATABASE_URL)
    await db.connect()

    try:
        reasoning_events = [
            {"step": 1, "type": "PLANNING", "content": "Analyzing query"},
            {"step": 1, "type": "TOOL_EXECUTION_START", "tools": ["weather"]},
            {"step": 1, "type": "REASONING_COMPLETE"},
        ]

        conv_id = await db.create_conversation(
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "reasoning_events": reasoning_events,
                },
            ],
        )

        conv = await db.get_conversation(conv_id)
        assert len(conv.messages) == 2
        assert conv.messages[1].reasoning_events == reasoning_events
    finally:
        await cleanup_db(db)
        await db.disconnect()


@pytest.mark.asyncio
async def test_message_with_tool_calls() -> None:
    """Test storing and retrieving messages with tool calls."""
    db = ConversationDB(DATABASE_URL)
    await db.connect()

    try:
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            },
        ]

        conv_id = await db.create_conversation(
            messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
            ],
        )

        conv = await db.get_conversation(conv_id)
        assert conv.messages[0].tool_calls == tool_calls
        assert conv.messages[0].content is None
    finally:
        await cleanup_db(db)
        await db.disconnect()


@pytest.mark.asyncio
async def test_message_with_metadata() -> None:
    """Test storing and retrieving messages with custom metadata."""
    db = ConversationDB(DATABASE_URL)
    await db.connect()

    try:
        metadata = {
            "model": "gpt-4o",
            "tokens": 150,
            "cost": 0.002,
            "client_version": "1.0.0",
        }

        conv_id = await db.create_conversation(
            messages=[{"role": "user", "content": "Hello", "metadata": metadata}],
        )

        conv = await db.get_conversation(conv_id)
        assert conv.messages[0].metadata == metadata
    finally:
        await cleanup_db(db)
        await db.disconnect()


# =============================================================================
# Edge Cases - Pagination
# =============================================================================


@pytest.mark.asyncio
async def test_list_conversations_pagination() -> None:
    """Test pagination with limit and offset."""
    db = ConversationDB(DATABASE_URL)
    await db.connect()

    try:
        # Create 5 conversations
        ids = []
        for i in range(5):
            conv_id = await db.create_conversation(messages=[], title=f"Conv {i}")
            ids.append(conv_id)

        # Get first page (2 items)
        page1, total = await db.list_conversations(limit=2, offset=0)
        assert total == 5
        assert len(page1) == 2

        # Get second page (2 items)
        page2, total = await db.list_conversations(limit=2, offset=2)
        assert total == 5
        assert len(page2) == 2

        # Get third page (1 item)
        page3, total = await db.list_conversations(limit=2, offset=4)
        assert total == 5
        assert len(page3) == 1

        # Verify no duplicates across pages
        all_ids = {c.id for c in page1 + page2 + page3}
        assert len(all_ids) == 5
    finally:
        await cleanup_db(db)
        await db.disconnect()


@pytest.mark.asyncio
async def test_list_conversations_empty() -> None:
    """Test listing conversations when none exist."""
    db = ConversationDB(DATABASE_URL)
    await db.connect()

    try:
        convs, total = await db.list_conversations()
        assert total == 0
        assert len(convs) == 0
    finally:
        await cleanup_db(db)
        await db.disconnect()


# =============================================================================
# Edge Cases - Delete Idempotency
# =============================================================================


@pytest.mark.asyncio
async def test_delete_conversation_idempotent() -> None:
    """Test that deleting already-deleted conversation returns False."""
    db = ConversationDB(DATABASE_URL)
    await db.connect()

    try:
        conv_id = await db.create_conversation(messages=[], title="Test")

        # First delete succeeds
        result1 = await db.delete_conversation(conv_id)
        assert result1 is True

        # Second delete returns False (already archived)
        result2 = await db.delete_conversation(conv_id)
        assert result2 is False

        # Conversation still exists but is archived
        conv = await db.get_conversation(conv_id)
        assert conv.archived_at is not None
    finally:
        await cleanup_db(db)
        await db.disconnect()


@pytest.mark.asyncio
async def test_update_title_on_archived_conversation() -> None:
    """Test that archived conversations cannot have title updated."""
    db = ConversationDB(DATABASE_URL)
    await db.connect()

    try:
        conv_id = await db.create_conversation(messages=[], title="Original")

        # Archive conversation
        await db.delete_conversation(conv_id)

        # Try to update title (should fail)
        result = await db.update_conversation_title(conv_id, "New Title")
        assert result is False

        # Verify title unchanged
        conv = await db.get_conversation(conv_id)
        assert conv.title == "Original"
    finally:
        await cleanup_db(db)
        await db.disconnect()


# =============================================================================
# Edge Cases - Empty Conversations
# =============================================================================


@pytest.mark.asyncio
async def test_create_empty_conversation() -> None:
    """Test creating conversation with no initial messages."""
    db = ConversationDB(DATABASE_URL)
    await db.connect()

    try:
        conv_id = await db.create_conversation(
            messages=[],
            system_message="You are helpful.",
        )

        conv = await db.get_conversation(conv_id)
        assert len(conv.messages) == 0
        assert conv.system_message == "You are helpful."
    finally:
        await cleanup_db(db)
        await db.disconnect()


@pytest.mark.asyncio
async def test_list_excludes_archived() -> None:
    """Test that list_conversations excludes archived conversations."""
    db = ConversationDB(DATABASE_URL)
    await db.connect()

    try:
        # Create 3 conversations
        id1 = await db.create_conversation(messages=[], title="Active 1")
        id2 = await db.create_conversation(messages=[], title="To Archive")
        id3 = await db.create_conversation(messages=[], title="Active 2")

        # Archive one
        await db.delete_conversation(id2)

        # List should only show 2 active
        convs, total = await db.list_conversations()
        assert total == 2
        returned_ids = {c.id for c in convs}
        assert id1 in returned_ids
        assert id3 in returned_ids
        assert id2 not in returned_ids
    finally:
        await cleanup_db(db)
        await db.disconnect()
