"""
Integration tests for ConversationDB CRUD operations using testcontainers.

Tests the full database layer with real PostgreSQL via testcontainers.
NO MANUAL SETUP REQUIRED - testcontainers handles everything!

Run with: pytest tests/integration_tests/test_conversation_db_testcontainers.py -v
"""

import asyncio
import asyncpg
import pytest
from uuid import uuid4, UUID
from reasoning_api.database import ConversationDB


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_create_and_get_conversation(conversation_db: ConversationDB) -> None:
    """Test creating and retrieving a conversation."""
    # Create conversation (conversation record only, no messages)
    conv_id = await conversation_db.create_conversation()
    assert isinstance(conv_id, UUID)

    # Append initial message
    await conversation_db.append_messages(
        conv_id,
        [{"role": "user", "content": "Hello"}],
    )

    # Get conversation
    conv = await conversation_db.get_conversation(conv_id)
    assert conv.id == conv_id
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
    conv_id = await conversation_db.create_conversation()

    conv = await conversation_db.get_conversation(conv_id)
    assert len(conv.messages) == 0


@pytest.mark.asyncio
async def test_list_excludes_archived(conversation_db: ConversationDB) -> None:
    """Test that list_conversations excludes archived conversations by default."""
    # Create 3 conversations
    id1 = await conversation_db.create_conversation(title="Active 1")
    id2 = await conversation_db.create_conversation(title="To Archive")
    id3 = await conversation_db.create_conversation(title="Active 2")

    # Archive one
    await conversation_db.delete_conversation(id2)

    # List should only show 2 active (default: archive_filter="active")
    convs, total = await conversation_db.list_conversations()
    assert total == 2
    returned_ids = {c.id for c in convs}
    assert id1 in returned_ids
    assert id3 in returned_ids
    assert id2 not in returned_ids

    # Verify archived_at is None for active conversations
    for conv in convs:
        assert conv.archived_at is None


@pytest.mark.asyncio
async def test_list_conversations_archive_filter_active(conversation_db: ConversationDB) -> None:
    """Test archive_filter='active' explicitly returns only active conversations."""
    # Create conversations
    active_id = await conversation_db.create_conversation(title="Active")
    archived_id = await conversation_db.create_conversation(title="Archived")
    await conversation_db.delete_conversation(archived_id)

    # Explicitly filter for active
    convs, total = await conversation_db.list_conversations(archive_filter="active")
    assert total == 1
    assert len(convs) == 1
    assert convs[0].id == active_id
    assert convs[0].archived_at is None


@pytest.mark.asyncio
async def test_list_conversations_archive_filter_archived(conversation_db: ConversationDB) -> None:
    """Test archive_filter='archived' returns only archived conversations with archived_at."""
    # Create conversations
    await conversation_db.create_conversation(title="Active")
    archived_id = await conversation_db.create_conversation(title="Archived")
    await conversation_db.delete_conversation(archived_id)

    # Filter for archived only
    convs, total = await conversation_db.list_conversations(archive_filter="archived")
    assert total == 1
    assert len(convs) == 1
    assert convs[0].id == archived_id
    assert convs[0].archived_at is not None
    assert convs[0].title == "Archived"


@pytest.mark.asyncio
async def test_list_conversations_archive_filter_all(conversation_db: ConversationDB) -> None:
    """Test archive_filter='all' returns both active and archived conversations."""
    # Create conversations
    active_id = await conversation_db.create_conversation(title="Active")
    archived_id = await conversation_db.create_conversation(title="Archived")
    await conversation_db.delete_conversation(archived_id)

    # Get all conversations
    convs, total = await conversation_db.list_conversations(archive_filter="all")
    assert total == 2
    assert len(convs) == 2

    # Verify both are present with correct archived_at values
    conv_by_id = {c.id: c for c in convs}
    assert active_id in conv_by_id
    assert archived_id in conv_by_id
    assert conv_by_id[active_id].archived_at is None
    assert conv_by_id[archived_id].archived_at is not None


@pytest.mark.asyncio
async def test_list_conversations_invalid_archive_filter(conversation_db: ConversationDB) -> None:
    """Test that invalid archive_filter raises ValueError."""
    with pytest.raises(ValueError, match="Invalid archive_filter: invalid"):
        await conversation_db.list_conversations(archive_filter="invalid")


@pytest.mark.asyncio
async def test_list_conversations_archive_filter_with_pagination(
    conversation_db: ConversationDB,
) -> None:
    """Test archive filtering works correctly with pagination."""
    # Create 5 active and 3 archived conversations
    active_ids = []
    for i in range(5):
        conv_id = await conversation_db.create_conversation(title=f"Active {i}")
        active_ids.append(conv_id)

    archived_ids = []
    for i in range(3):
        conv_id = await conversation_db.create_conversation(title=f"Archived {i}")
        await conversation_db.delete_conversation(conv_id)
        archived_ids.append(conv_id)

    # Test pagination with active filter
    page1, total = await conversation_db.list_conversations(
        archive_filter="active",
        limit=2,
        offset=0,
    )
    assert total == 5
    assert len(page1) == 2

    # Test pagination with archived filter
    page1, total = await conversation_db.list_conversations(
        archive_filter="archived",
        limit=2,
        offset=0,
    )
    assert total == 3
    assert len(page1) == 2

    # Test pagination with all filter
    page1, total = await conversation_db.list_conversations(
        archive_filter="all",
        limit=3,
        offset=0,
    )
    assert total == 8
    assert len(page1) == 3


# =============================================================================
# Delete Messages from Sequence
# =============================================================================


@pytest.mark.asyncio
async def test_delete_messages_from_sequence__middle_message(
    conversation_db: ConversationDB,
) -> None:
    """Test deleting a middle message removes it and all subsequent messages."""
    # Create conversation with 5 messages
    conv_id = await conversation_db.create_conversation()
    await conversation_db.append_messages(
        conv_id,
        [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Message 3"},
        ],
    )

    # Delete from sequence 3 (should delete sequences 3, 4, 5)
    success = await conversation_db.delete_messages_from_sequence(conv_id, 3)
    assert success is True

    # Verify only messages 1 and 2 remain
    conv = await conversation_db.get_conversation(conv_id)
    assert len(conv.messages) == 2
    assert conv.messages[0].sequence_number == 1
    assert conv.messages[0].content == "Message 1"
    assert conv.messages[1].sequence_number == 2
    assert conv.messages[1].content == "Response 1"


@pytest.mark.asyncio
async def test_delete_messages_from_sequence__last_message(
    conversation_db: ConversationDB,
) -> None:
    """Test deleting the last message works correctly."""
    conv_id = await conversation_db.create_conversation()
    await conversation_db.append_messages(
        conv_id,
        [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
        ],
    )

    # Delete last message (sequence 2)
    success = await conversation_db.delete_messages_from_sequence(conv_id, 2)
    assert success is True

    # Verify only message 1 remains
    conv = await conversation_db.get_conversation(conv_id)
    assert len(conv.messages) == 1
    assert conv.messages[0].sequence_number == 1


@pytest.mark.asyncio
async def test_delete_messages_from_sequence__first_message(
    conversation_db: ConversationDB,
) -> None:
    """Test deleting the first message removes all messages."""
    conv_id = await conversation_db.create_conversation()
    await conversation_db.append_messages(
        conv_id,
        [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
        ],
    )

    # Delete from sequence 1 (should delete all messages)
    success = await conversation_db.delete_messages_from_sequence(conv_id, 1)
    assert success is True

    # Verify no messages remain
    conv = await conversation_db.get_conversation(conv_id)
    assert len(conv.messages) == 0


@pytest.mark.asyncio
async def test_delete_messages_from_sequence__nonexistent_conversation(
    conversation_db: ConversationDB,
) -> None:
    """Test deleting from non-existent conversation returns False."""
    fake_id = uuid4()
    success = await conversation_db.delete_messages_from_sequence(fake_id, 1)
    assert success is False


@pytest.mark.asyncio
async def test_delete_messages_from_sequence__nonexistent_sequence(
    conversation_db: ConversationDB,
) -> None:
    """Test deleting non-existent sequence number returns False."""
    conv_id = await conversation_db.create_conversation()
    await conversation_db.append_messages(
        conv_id,
        [{"role": "user", "content": "Message 1"}],
    )

    # Try to delete sequence 99 (doesn't exist)
    success = await conversation_db.delete_messages_from_sequence(conv_id, 99)
    assert success is False


# =============================================================================
# Branch Conversation
# =============================================================================


@pytest.mark.asyncio
async def test_branch_conversation__basic(conversation_db: ConversationDB) -> None:
    """Test branching a conversation creates independent copy."""
    # Create source conversation
    source_id = await conversation_db.create_conversation(
        title="Original Conversation",
    )
    await conversation_db.append_messages(
        source_id,
        [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
        ],
    )

    # Branch at sequence 2 (should copy messages 1 and 2)
    branched_conv = await conversation_db.branch_conversation(source_id, 2)

    # Verify new conversation has correct title
    assert branched_conv.title == "Branch of 'Original Conversation'"
    assert branched_conv.id != source_id

    # Verify correct messages copied
    assert len(branched_conv.messages) == 2
    assert branched_conv.messages[0].content == "Message 1"
    assert branched_conv.messages[0].sequence_number == 1
    assert branched_conv.messages[1].content == "Response 1"
    assert branched_conv.messages[1].sequence_number == 2

    # Verify source conversation unchanged
    source_conv = await conversation_db.get_conversation(source_id)
    assert len(source_conv.messages) == 4


@pytest.mark.asyncio
async def test_branch_conversation__no_title(conversation_db: ConversationDB) -> None:
    """Test branching conversation without title uses default."""
    source_id = await conversation_db.create_conversation()
    await conversation_db.append_messages(
        source_id,
        [{"role": "user", "content": "Message 1"}],
    )

    branched_conv = await conversation_db.branch_conversation(source_id, 1)
    assert branched_conv.title == "Branch of Conversation"


@pytest.mark.asyncio
async def test_branch_conversation__preserves_metadata(
    conversation_db: ConversationDB,
) -> None:
    """Test branching preserves metadata from source conversation."""
    # Create conversation with metadata
    source_id = await conversation_db.create_conversation(
        title="Original",
    )

    # Manually set metadata (would normally be set via update endpoint)
    # For this test, we'll create a new conversation with metadata directly via SQL
    async def set_metadata(conn: asyncpg.Connection) -> None:
        await conn.execute(
            "UPDATE conversations SET metadata = $1 WHERE id = $2",
            {"custom_field": "custom_value", "tags": ["tag1", "tag2"]},
            source_id,
        )

    await conversation_db._execute_with_connection(set_metadata)

    await conversation_db.append_messages(
        source_id,
        [{"role": "user", "content": "Message 1"}],
    )

    # Branch the conversation
    branched_conv = await conversation_db.branch_conversation(source_id, 1)

    # Verify metadata is preserved
    assert branched_conv.metadata == {"custom_field": "custom_value", "tags": ["tag1", "tag2"]}


@pytest.mark.asyncio
async def test_branch_conversation__independence(
    conversation_db: ConversationDB,
) -> None:
    """Test branched conversation is independent from source."""
    # Create source and branch
    source_id = await conversation_db.create_conversation(title="Original")
    await conversation_db.append_messages(
        source_id,
        [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
        ],
    )

    branched_conv = await conversation_db.branch_conversation(source_id, 2)
    branch_id = branched_conv.id

    # Add message to source
    await conversation_db.append_messages(
        source_id,
        [{"role": "user", "content": "New message in source"}],
    )

    # Add different message to branch
    await conversation_db.append_messages(
        branch_id,
        [{"role": "user", "content": "New message in branch"}],
    )

    # Verify they are independent
    source_conv = await conversation_db.get_conversation(source_id)
    branch_conv = await conversation_db.get_conversation(branch_id)

    assert len(source_conv.messages) == 3
    assert source_conv.messages[2].content == "New message in source"

    assert len(branch_conv.messages) == 3
    assert branch_conv.messages[2].content == "New message in branch"


@pytest.mark.asyncio
async def test_branch_conversation__nonexistent_conversation(
    conversation_db: ConversationDB,
) -> None:
    """Test branching non-existent conversation raises ValueError."""
    fake_id = uuid4()
    with pytest.raises(ValueError, match="not found"):
        await conversation_db.branch_conversation(fake_id, 1)


@pytest.mark.asyncio
async def test_branch_conversation__nonexistent_sequence(
    conversation_db: ConversationDB,
) -> None:
    """Test branching at non-existent sequence raises ValueError."""
    conv_id = await conversation_db.create_conversation()
    await conversation_db.append_messages(
        conv_id,
        [{"role": "user", "content": "Message 1"}],
    )

    # Try to branch at sequence 99 (doesn't exist)
    with pytest.raises(ValueError, match="Sequence number"):
        await conversation_db.branch_conversation(conv_id, 99)


@pytest.mark.asyncio
async def test_branch_conversation__archived_source(
    conversation_db: ConversationDB,
) -> None:
    """Test branching archived conversation raises ValueError."""
    conv_id = await conversation_db.create_conversation()
    await conversation_db.append_messages(
        conv_id,
        [{"role": "user", "content": "Message 1"}],
    )

    # Archive the conversation
    await conversation_db.delete_conversation(conv_id, permanent=False)

    # Try to branch archived conversation
    with pytest.raises(ValueError, match="not found or archived"):
        await conversation_db.branch_conversation(conv_id, 1)
