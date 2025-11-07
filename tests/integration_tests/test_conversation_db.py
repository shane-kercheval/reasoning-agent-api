"""
Integration tests for ConversationDB CRUD operations using testcontainers.

Tests the full database layer with real PostgreSQL via testcontainers.
NO MANUAL SETUP REQUIRED - testcontainers handles everything!

Run with: pytest tests/integration_tests/test_conversation_db_testcontainers.py -v
"""

import asyncio
import pytest
from uuid import uuid4, UUID
from api.database import ConversationDB


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_create_and_get_conversation(conversation_db: ConversationDB) -> None:
    """Test creating and retrieving a conversation."""
    # Create conversation (conversation record only, no messages)
    conv_id = await conversation_db.create_conversation(
        system_message="You are helpful.",
    )
    assert isinstance(conv_id, UUID)

    # Append initial message
    await conversation_db.append_messages(
        conv_id,
        [{"role": "user", "content": "Hello"}],
    )

    # Get conversation
    conv = await conversation_db.get_conversation(conv_id)
    assert conv.id == conv_id
    assert conv.system_message == "You are helpful."
    assert len(conv.messages) == 1
    assert conv.messages[0].content == "Hello"


@pytest.mark.asyncio
async def test_append_messages(conversation_db: ConversationDB) -> None:
    """Test appending messages to a conversation."""
    # Create conversation (no messages)
    conv_id = await conversation_db.create_conversation()

    # Append first message
    await conversation_db.append_messages(
        conv_id,
        [{"role": "user", "content": "Hi"}],
    )

    # Append more messages
    await conversation_db.append_messages(
        conv_id,
        [
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ],
    )

    # Verify count and sequence numbers
    conv = await conversation_db.get_conversation(conv_id)
    assert len(conv.messages) == 3
    assert conv.messages[0].sequence_number == 1
    assert conv.messages[1].sequence_number == 2
    assert conv.messages[2].sequence_number == 3

    # Verify content is preserved correctly (batch insert test)
    assert conv.messages[0].role == "user"
    assert conv.messages[0].content == "Hi"
    assert conv.messages[1].role == "assistant"
    assert conv.messages[1].content == "Hello!"
    assert conv.messages[2].role == "user"
    assert conv.messages[2].content == "How are you?"


@pytest.mark.asyncio
async def test_concurrent_appends(conversation_db: ConversationDB) -> None:
    """
    Test concurrent message appends don't create duplicate sequence numbers.

    Note: In test mode with transaction rollback, we can't truly test concurrency
    because we're using a single connection. The test runs sequentially in that case,
    but still validates sequence numbering logic.
    """
    # Create conversation
    conv_id = await conversation_db.create_conversation()

    # Concurrent appends (will be sequential with test connection, concurrent in production)
    async def append_msg(num: int) -> None:
        await conversation_db.append_messages(
            conv_id,
            [{"role": "user", "content": f"Message {num}"}],
        )

    # In test mode (_test_conn is set), asyncpg can't handle concurrent ops on same connection
    # So we run sequentially. In production mode, this would be truly concurrent via pool.
    if conversation_db._test_conn is not None:
        # Sequential execution for test mode
        for i in range(5):
            await append_msg(i)
    else:
        # Concurrent execution for production mode
        await asyncio.gather(*[append_msg(i) for i in range(5)])

    # Verify unique sequence numbers
    conv = await conversation_db.get_conversation(conv_id)
    assert len(conv.messages) == 5
    seq_nums = [msg.sequence_number for msg in conv.messages]
    assert seq_nums == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_list_and_delete(conversation_db: ConversationDB) -> None:
    """Test listing and deleting conversations."""
    # Create 3 conversations
    await conversation_db.create_conversation(title="First")
    id2 = await conversation_db.create_conversation(title="Second")
    await conversation_db.create_conversation(title="Third")

    # List all
    convs, total = await conversation_db.list_conversations()
    assert total == 3
    assert len(convs) == 3

    # Delete one
    result = await conversation_db.delete_conversation(id2)
    assert result is True

    # List again (should have 2 now)
    convs, total = await conversation_db.list_conversations()
    assert total == 2


@pytest.mark.asyncio
async def test_update_title(conversation_db: ConversationDB) -> None:
    """Test updating conversation title."""
    # Create conversation
    conv_id = await conversation_db.create_conversation(title="Old Title")

    # Update title
    result = await conversation_db.update_conversation_title(conv_id, "New Title")
    assert result is True

    # Verify
    conv = await conversation_db.get_conversation(conv_id)
    assert conv.title == "New Title"


# =============================================================================
# Error Cases - System Message Validation
# =============================================================================


@pytest.mark.asyncio
async def test_append_rejects_system_message(conversation_db: ConversationDB) -> None:
    """Test that system messages cannot be appended."""
    conv_id = await conversation_db.create_conversation()

    with pytest.raises(ValueError, match="Cannot append system messages"):
        await conversation_db.append_messages(
            conv_id,
            [{"role": "system", "content": "New system message"}],
        )


# =============================================================================
# Error Cases - Not Found
# =============================================================================


@pytest.mark.asyncio
async def test_get_conversation_not_found(conversation_db: ConversationDB) -> None:
    """Test getting non-existent conversation raises error."""
    fake_id = uuid4()
    with pytest.raises(ValueError, match=f"Conversation {fake_id} not found"):
        await conversation_db.get_conversation(fake_id)


@pytest.mark.asyncio
async def test_append_to_nonexistent_conversation(conversation_db: ConversationDB) -> None:
    """Test appending to non-existent conversation raises error."""
    fake_id = uuid4()
    with pytest.raises(ValueError, match=f"Conversation {fake_id} not found"):
        await conversation_db.append_messages(fake_id, [{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
async def test_delete_nonexistent_conversation(conversation_db: ConversationDB) -> None:
    """Test deleting non-existent conversation returns False."""
    fake_id = uuid4()
    result = await conversation_db.delete_conversation(fake_id)
    assert result is False


@pytest.mark.asyncio
async def test_update_title_nonexistent_conversation(conversation_db: ConversationDB) -> None:
    """Test updating title of non-existent conversation returns False."""
    fake_id = uuid4()
    result = await conversation_db.update_conversation_title(fake_id, "New Title")
    assert result is False


# =============================================================================
# Edge Cases - JSONB Fields
# =============================================================================


@pytest.mark.asyncio
async def test_message_with_reasoning_events(conversation_db: ConversationDB) -> None:
    """Test storing and retrieving messages with reasoning events as array."""
    reasoning_events = [
        {"type": "iteration_start", "step_iteration": 1, "metadata": {"tools": []}},
        {"type": "planning", "step_iteration": 1, "metadata": {"thought": "Analyzing query", "tools_planned": ["weather"]}},  # noqa: E501
        {"type": "tool_execution_start", "step_iteration": 1, "metadata": {"tools": ["weather"]}},
        {"type": "tool_result", "step_iteration": 1, "metadata": {"tools": ["weather"], "tool_results": [{"tool_name": "weather", "success": True, "result": "Sunny"}]}},  # noqa: E501
        {"type": "iteration_complete", "step_iteration": 1, "metadata": {"tools": ["weather"], "had_tools": True}},  # noqa: E501
        {"type": "reasoning_complete", "step_iteration": 0, "metadata": {"tools": [], "total_steps": 1}},  # noqa: E501
    ]

    conv_id = await conversation_db.create_conversation()

    await conversation_db.append_messages(
        conv_id,
        [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "It's sunny today!",
                "reasoning_events": reasoning_events,
                "total_cost": 0.00015,
                "metadata": {
                    "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
                    "cost": {"total_cost": 0.00015, "prompt_cost": 0.0001, "completion_cost": 0.00005},  # noqa: E501
                },
            },
        ],
    )

    conv = await conversation_db.get_conversation(conv_id)
    assert len(conv.messages) == 2
    # Verify reasoning events are stored and retrieved correctly
    assert conv.messages[1].reasoning_events == reasoning_events
    assert len(conv.messages[1].reasoning_events) == 6
    # Verify each event has expected structure
    for event in conv.messages[1].reasoning_events:
        assert "type" in event
        assert "step_iteration" in event
        assert "metadata" in event
    # Verify total_cost is stored
    assert conv.messages[1].total_cost == 0.00015


@pytest.mark.asyncio
async def test_message_with_total_cost(conversation_db: ConversationDB) -> None:
    """Test storing and retrieving messages with total_cost."""
    conv_id = await conversation_db.create_conversation()

    # Test with explicit total_cost
    await conversation_db.append_messages(
        conv_id,
        [
            {
                "role": "assistant",
                "content": "Hello!",
                "total_cost": 0.000025,
                "metadata": {
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    "cost": {"total_cost": 0.000025, "prompt_cost": 0.000015, "completion_cost": 0.00001},  # noqa: E501
                },
            },
        ],
    )

    conv = await conversation_db.get_conversation(conv_id)
    assert conv.messages[0].total_cost == 0.000025
    assert conv.messages[0].metadata["cost"]["total_cost"] == 0.000025


@pytest.mark.asyncio
async def test_message_without_total_cost(conversation_db: ConversationDB) -> None:
    """Test storing and retrieving messages without total_cost (None)."""
    conv_id = await conversation_db.create_conversation()

    # Message without total_cost (e.g., user message or stateless mode)
    await conversation_db.append_messages(
        conv_id,
        [
            {
                "role": "user",
                "content": "Hello!",
                "metadata": {},
            },
        ],
    )

    conv = await conversation_db.get_conversation(conv_id)
    assert conv.messages[0].total_cost is None
    assert conv.messages[0].role == "user"


@pytest.mark.asyncio
async def test_message_with_metadata(conversation_db: ConversationDB) -> None:
    """Test storing and retrieving messages with custom metadata."""
    metadata = {
        "model": "gpt-4o",
        "tokens": 150,
        "cost": 0.002,
        "client_version": "1.0.0",
    }

    conv_id = await conversation_db.create_conversation()

    await conversation_db.append_messages(
        conv_id,
        [{"role": "user", "content": "Hello", "metadata": metadata}],
    )

    conv = await conversation_db.get_conversation(conv_id)
    assert conv.messages[0].metadata == metadata


# =============================================================================
# Edge Cases - Pagination
# =============================================================================


@pytest.mark.asyncio
async def test_list_conversations_pagination(conversation_db: ConversationDB) -> None:
    """Test pagination with limit and offset."""
    # Create 5 conversations
    ids = []
    for i in range(5):
        conv_id = await conversation_db.create_conversation(title=f"Conv {i}")
        ids.append(conv_id)

    # Get first page (2 items)
    page1, total = await conversation_db.list_conversations(limit=2, offset=0)
    assert total == 5
    assert len(page1) == 2

    # Get second page (2 items)
    page2, total = await conversation_db.list_conversations(limit=2, offset=2)
    assert total == 5
    assert len(page2) == 2

    # Get third page (1 item)
    page3, total = await conversation_db.list_conversations(limit=2, offset=4)
    assert total == 5
    assert len(page3) == 1

    # Verify no duplicates across pages
    all_ids = {c.id for c in page1 + page2 + page3}
    assert len(all_ids) == 5


@pytest.mark.asyncio
async def test_list_conversations_empty(conversation_db: ConversationDB) -> None:
    """Test listing conversations when none exist."""
    convs, total = await conversation_db.list_conversations()
    assert total == 0
    assert len(convs) == 0


# =============================================================================
# Edge Cases - Delete Idempotency
# =============================================================================


@pytest.mark.asyncio
async def test_delete_conversation_idempotent(conversation_db: ConversationDB) -> None:
    """Test that deleting already-deleted conversation returns False."""
    conv_id = await conversation_db.create_conversation(title="Test")

    # First delete succeeds
    result1 = await conversation_db.delete_conversation(conv_id)
    assert result1 is True

    # Second delete returns False (already archived)
    result2 = await conversation_db.delete_conversation(conv_id)
    assert result2 is False

    # Conversation still exists but is archived
    conv = await conversation_db.get_conversation(conv_id)
    assert conv.archived_at is not None


@pytest.mark.asyncio
async def test_update_title_on_archived_conversation(conversation_db: ConversationDB) -> None:
    """Test that archived conversations cannot have title updated."""
    conv_id = await conversation_db.create_conversation(title="Original")

    # Archive conversation
    await conversation_db.delete_conversation(conv_id)

    # Try to update title (should fail)
    result = await conversation_db.update_conversation_title(conv_id, "New Title")
    assert result is False

    # Verify title unchanged
    conv = await conversation_db.get_conversation(conv_id)
    assert conv.title == "Original"


# =============================================================================
# Edge Cases - Empty Conversations
# =============================================================================


@pytest.mark.asyncio
async def test_create_empty_conversation(conversation_db: ConversationDB) -> None:
    """Test creating conversation with no initial messages."""
    conv_id = await conversation_db.create_conversation(
        system_message="You are helpful.",
    )

    conv = await conversation_db.get_conversation(conv_id)
    assert len(conv.messages) == 0
    assert conv.system_message == "You are helpful."


@pytest.mark.asyncio
async def test_list_excludes_archived(conversation_db: ConversationDB) -> None:
    """Test that list_conversations excludes archived conversations."""
    # Create 3 conversations
    id1 = await conversation_db.create_conversation(title="Active 1")
    id2 = await conversation_db.create_conversation(title="To Archive")
    id3 = await conversation_db.create_conversation(title="Active 2")

    # Archive one
    await conversation_db.delete_conversation(id2)

    # List should only show 2 active
    convs, total = await conversation_db.list_conversations()
    assert total == 2
    returned_ids = {c.id for c in convs}
    assert id1 in returned_ids
    assert id3 in returned_ids
    assert id2 not in returned_ids
