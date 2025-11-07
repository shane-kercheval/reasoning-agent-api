"""
Integration tests for conversation management REST endpoints (Milestone 9).

Tests the REST API for conversation CRUD operations with real database.
Uses testcontainers for real PostgreSQL database.

Run with: pytest tests/integration_tests/test_conversation_rest_api.py -v
"""

import asyncio

import pytest
import pytest_asyncio
from uuid import uuid4
from api.main import app
from api.database import ConversationDB
from api.dependencies import (
    get_conversation_db,
    get_tools,
    get_prompt_manager,
)
from api.prompt_manager import PromptManager
from httpx import AsyncClient, ASGITransport

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def client(conversation_db: ConversationDB) -> AsyncClient:
    """Create async test client with test database dependency override."""
    # Override dependencies to use test database
    app.dependency_overrides[get_conversation_db] = lambda: conversation_db
    app.dependency_overrides[get_tools] = lambda: []
    app.dependency_overrides[get_prompt_manager] = lambda: PromptManager()

    # Use ASGITransport with httpx.AsyncClient for proper async support
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    # Cleanup
    app.dependency_overrides.clear()


# =============================================================================
# GET /v1/conversations - List Conversations
# =============================================================================


@pytest.mark.asyncio
async def test_list_conversations__empty_database(client: AsyncClient) -> None:
    """Test listing conversations when database is empty."""
    response = await client.get("/v1/conversations")

    assert response.status_code == 200
    data = response.json()
    assert data["conversations"] == []
    assert data["total"] == 0
    assert data["limit"] == 50
    assert data["offset"] == 0


@pytest.mark.asyncio
async def test_list_conversations__returns_summaries(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test listing conversations returns summary data with message count."""
    # Create conversations with messages
    conv1_id = await conversation_db.create_conversation(
        title="First conversation",
        system_message="You are helpful.",
    )
    await conversation_db.append_messages(
        conv1_id,
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
    )

    conv2_id = await conversation_db.create_conversation(
        title="Second conversation",
        system_message="You are an expert.",
    )
    await conversation_db.append_messages(
        conv2_id,
        [{"role": "user", "content": "Question?"}],
    )

    response = await client.get("/v1/conversations")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["conversations"]) == 2

    # Verify summary structure (no full messages)
    for conv in data["conversations"]:
        assert "id" in conv
        assert "title" in conv
        assert "system_message" in conv
        assert "created_at" in conv
        assert "updated_at" in conv
        assert "message_count" in conv
        # Should NOT have messages field
        assert "messages" not in conv

    # Verify message counts
    conv_by_id = {c["id"]: c for c in data["conversations"]}
    assert conv_by_id[str(conv1_id)]["message_count"] == 2
    assert conv_by_id[str(conv2_id)]["message_count"] == 1


@pytest.mark.asyncio
async def test_list_conversations__ordered_by_updated_at_desc(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test conversations are ordered by updated_at DESC (most recent first)."""
    # Create first conversation and add message
    conv1_id = await conversation_db.create_conversation(title="First")
    await conversation_db.append_messages(
        conv1_id,
        [{"role": "user", "content": "Message 1"}],
    )

    # Small delay to ensure different timestamps
    await asyncio.sleep(0.01)

    # Create second conversation and add message
    conv2_id = await conversation_db.create_conversation(title="Second")
    await conversation_db.append_messages(
        conv2_id,
        [{"role": "user", "content": "Message 2"}],
    )

    # Small delay
    await asyncio.sleep(0.01)

    # Update conv1 (should move it to top)
    await conversation_db.append_messages(
        conv1_id,
        [{"role": "user", "content": "New message"}],
    )

    response = await client.get("/v1/conversations")

    assert response.status_code == 200
    data = response.json()
    ids = [c["id"] for c in data["conversations"]]

    # conv1 should be first (most recently updated)
    # conv2 should be second
    assert ids == [str(conv1_id), str(conv2_id)]


@pytest.mark.asyncio
async def test_list_conversations__pagination(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test pagination with limit and offset parameters."""
    # Create 5 conversations
    for i in range(5):
        await conversation_db.create_conversation(title=f"Conv {i}")

    # Page 1: limit=2, offset=0
    response = await client.get("/v1/conversations?limit=2&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 5
    assert len(data["conversations"]) == 2
    assert data["limit"] == 2
    assert data["offset"] == 0
    page1_ids = {c["id"] for c in data["conversations"]}

    # Page 2: limit=2, offset=2
    response = await client.get("/v1/conversations?limit=2&offset=2")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 5
    assert len(data["conversations"]) == 2
    assert data["limit"] == 2
    assert data["offset"] == 2
    page2_ids = {c["id"] for c in data["conversations"]}

    # No duplicates between pages
    assert len(page1_ids & page2_ids) == 0


@pytest.mark.asyncio
async def test_list_conversations__excludes_archived(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test that archived conversations are excluded from list."""
    # Create 3 conversations
    conv1_id = await conversation_db.create_conversation(title="Active 1")
    conv2_id = await conversation_db.create_conversation(title="To Delete")
    conv3_id = await conversation_db.create_conversation(title="Active 2")

    # Archive conv2
    await conversation_db.delete_conversation(conv2_id)

    response = await client.get("/v1/conversations")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    ids = {c["id"] for c in data["conversations"]}
    assert str(conv1_id) in ids
    assert str(conv3_id) in ids
    assert str(conv2_id) not in ids


@pytest.mark.asyncio
async def test_list_conversations__invalid_pagination_params(
    client: AsyncClient,
) -> None:
    """Test that invalid pagination parameters return 422."""
    # Negative offset
    response = await client.get("/v1/conversations?offset=-1")
    assert response.status_code == 422

    # Limit too large
    response = await client.get("/v1/conversations?limit=101")
    assert response.status_code == 422

    # Limit too small
    response = await client.get("/v1/conversations?limit=0")
    assert response.status_code == 422


# =============================================================================
# GET /v1/conversations/{id} - Get Single Conversation
# =============================================================================


@pytest.mark.asyncio
async def test_get_conversation__returns_full_details(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test getting conversation returns full details with all messages."""
    conv_id = await conversation_db.create_conversation(
        title="Test Conversation",
        system_message="You are a test assistant.",
    )
    await conversation_db.append_messages(
        conv_id,
        [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Second message"},
        ],
    )

    response = await client.get(f"/v1/conversations/{conv_id}")

    assert response.status_code == 200
    data = response.json()

    # Verify conversation metadata
    assert data["id"] == str(conv_id)
    assert data["title"] == "Test Conversation"
    assert data["system_message"] == "You are a test assistant."
    assert "created_at" in data
    assert "updated_at" in data

    # Verify messages
    assert len(data["messages"]) == 3
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][0]["content"] == "First message"
    assert data["messages"][0]["sequence_number"] == 1
    assert data["messages"][1]["role"] == "assistant"
    assert data["messages"][1]["content"] == "First response"
    assert data["messages"][1]["sequence_number"] == 2
    assert data["messages"][2]["role"] == "user"
    assert data["messages"][2]["content"] == "Second message"
    assert data["messages"][2]["sequence_number"] == 3


@pytest.mark.asyncio
async def test_get_conversation__not_found(client: AsyncClient) -> None:
    """Test getting non-existent conversation returns 404."""
    fake_id = uuid4()
    response = await client.get(f"/v1/conversations/{fake_id}")

    assert response.status_code == 404
    data = response.json()
    assert "conversation_not_found" in str(data)


@pytest.mark.asyncio
async def test_get_conversation__invalid_uuid(client: AsyncClient) -> None:
    """Test getting conversation with invalid UUID returns 422."""
    response = await client.get("/v1/conversations/not-a-uuid")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_conversation__archived_conversation(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test that archived conversations can still be retrieved by ID."""
    conv_id = await conversation_db.create_conversation(title="To Archive")
    await conversation_db.delete_conversation(conv_id)

    # Should still be retrievable by ID (soft delete)
    response = await client.get(f"/v1/conversations/{conv_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(conv_id)


# =============================================================================
# DELETE /v1/conversations/{id} - Delete Conversation
# =============================================================================


@pytest.mark.asyncio
async def test_delete_conversation__success(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test deleting conversation returns 204 No Content."""
    conv_id = await conversation_db.create_conversation(title="To Delete")

    response = await client.delete(f"/v1/conversations/{conv_id}")

    assert response.status_code == 204
    assert response.content == b""

    # Verify conversation is archived (excluded from list)
    list_response = await client.get("/v1/conversations")
    data = list_response.json()
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_delete_conversation__not_found(client: AsyncClient) -> None:
    """Test deleting non-existent conversation returns 404."""
    fake_id = uuid4()
    response = await client.delete(f"/v1/conversations/{fake_id}")

    assert response.status_code == 404
    data = response.json()
    assert "conversation_not_found" in str(data)


@pytest.mark.asyncio
async def test_delete_conversation__already_deleted(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test deleting already-deleted conversation returns 404."""
    conv_id = await conversation_db.create_conversation(title="Test")

    # First delete succeeds
    response = await client.delete(f"/v1/conversations/{conv_id}")
    assert response.status_code == 204

    # Second delete returns 404 (already archived)
    response = await client.delete(f"/v1/conversations/{conv_id}")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_conversation__invalid_uuid(client: AsyncClient) -> None:
    """Test deleting with invalid UUID returns 422."""
    response = await client.delete("/v1/conversations/not-a-uuid")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_delete_conversation__permanent_success(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test permanent delete removes conversation completely."""
    conv_id = await conversation_db.create_conversation(title="To Delete Permanently")

    # Hard delete
    response = await client.delete(f"/v1/conversations/{conv_id}?permanent=true")

    assert response.status_code == 204
    assert response.content == b""

    # Verify conversation is completely removed (not just archived)
    get_response = await client.get(f"/v1/conversations/{conv_id}")
    assert get_response.status_code == 404

    # Also excluded from list
    list_response = await client.get("/v1/conversations")
    data = list_response.json()
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_delete_conversation__permanent_not_found(client: AsyncClient) -> None:
    """Test permanent delete of non-existent conversation returns 404."""
    fake_id = uuid4()
    response = await client.delete(f"/v1/conversations/{fake_id}?permanent=true")

    assert response.status_code == 404
    data = response.json()
    assert "conversation_not_found" in str(data)


@pytest.mark.asyncio
async def test_delete_conversation__permanent_with_messages(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test permanent delete cascades to messages."""
    conv_id = await conversation_db.create_conversation(title="With Messages")

    # Add messages to the conversation
    await conversation_db.append_messages(
        conv_id,
        [
            {"role": "user", "content": "Test message 1"},
            {"role": "assistant", "content": "Test response 1"},
        ],
    )

    # Verify messages exist
    conv = await conversation_db.get_conversation(conv_id)
    assert conv is not None
    assert len(conv.messages) == 2

    # Hard delete
    response = await client.delete(f"/v1/conversations/{conv_id}?permanent=true")
    assert response.status_code == 204

    # Verify conversation and messages are completely removed
    get_response = await client.get(f"/v1/conversations/{conv_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_delete_conversation__permanent_after_soft_delete(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test permanent delete works on already soft-deleted conversation."""
    conv_id = await conversation_db.create_conversation(title="Test")

    # First soft delete
    response = await client.delete(f"/v1/conversations/{conv_id}")
    assert response.status_code == 204

    # Verify it's soft deleted (still retrievable)
    get_response = await client.get(f"/v1/conversations/{conv_id}")
    assert get_response.status_code == 200

    # Now permanent delete
    response = await client.delete(f"/v1/conversations/{conv_id}?permanent=true")
    assert response.status_code == 204

    # Verify it's completely removed
    get_response = await client.get(f"/v1/conversations/{conv_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_delete_conversation__soft_delete_remains_default(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test that soft delete is default behavior without query parameter."""
    conv_id = await conversation_db.create_conversation(title="Test Default")

    # Delete without permanent parameter (should be soft delete)
    response = await client.delete(f"/v1/conversations/{conv_id}")
    assert response.status_code == 204

    # Verify it's soft deleted (excluded from list but retrievable by ID)
    list_response = await client.get("/v1/conversations")
    assert list_response.json()["total"] == 0

    get_response = await client.get(f"/v1/conversations/{conv_id}")
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["id"] == str(conv_id)


@pytest.mark.asyncio
async def test_delete_conversation__permanent_false_is_soft_delete(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test that permanent=false explicitly requests soft delete."""
    conv_id = await conversation_db.create_conversation(title="Test Explicit Soft")

    # Explicitly request soft delete
    response = await client.delete(f"/v1/conversations/{conv_id}?permanent=false")
    assert response.status_code == 204

    # Verify it's soft deleted (still retrievable)
    get_response = await client.get(f"/v1/conversations/{conv_id}")
    assert get_response.status_code == 200


# =============================================================================
# PATCH /v1/conversations/{id} - Update Conversation Title
# =============================================================================


@pytest.mark.asyncio
async def test_update_conversation__success_with_valid_title(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test updating conversation title with valid string."""
    conv_id = await conversation_db.create_conversation(
        title="Original Title",
        system_message="You are helpful.",
    )

    # Get original updated_at timestamp
    original_conv = await conversation_db.get_conversation(conv_id)
    original_updated_at = original_conv.updated_at

    # Small delay to ensure timestamp changes
    await asyncio.sleep(0.01)

    # Update title
    response = await client.patch(
        f"/v1/conversations/{conv_id}",
        json={"title": "Updated Title"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(conv_id)
    assert data["title"] == "Updated Title"
    assert data["system_message"] == "You are helpful."
    assert "created_at" in data
    assert "updated_at" in data
    assert "message_count" in data

    # Verify updated_at timestamp changed
    assert data["updated_at"] != original_updated_at


@pytest.mark.asyncio
async def test_update_conversation__clear_title_with_null(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test clearing conversation title with null value."""
    conv_id = await conversation_db.create_conversation(
        title="Original Title",
        system_message="You are helpful.",
    )

    response = await client.patch(
        f"/v1/conversations/{conv_id}",
        json={"title": None},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(conv_id)
    assert data["title"] is None


@pytest.mark.asyncio
async def test_update_conversation__clear_title_with_empty_string(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test clearing conversation title with empty string (treated as null)."""
    conv_id = await conversation_db.create_conversation(
        title="Original Title",
        system_message="You are helpful.",
    )

    response = await client.patch(
        f"/v1/conversations/{conv_id}",
        json={"title": ""},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(conv_id)
    assert data["title"] is None


@pytest.mark.asyncio
async def test_update_conversation__trim_whitespace(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test that whitespace is trimmed from title."""
    conv_id = await conversation_db.create_conversation(
        title="Original",
        system_message="You are helpful.",
    )

    response = await client.patch(
        f"/v1/conversations/{conv_id}",
        json={"title": "  Trimmed Title  "},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Trimmed Title"


@pytest.mark.asyncio
async def test_update_conversation__whitespace_only_treated_as_null(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test that whitespace-only string is treated as null."""
    conv_id = await conversation_db.create_conversation(
        title="Original",
        system_message="You are helpful.",
    )

    response = await client.patch(
        f"/v1/conversations/{conv_id}",
        json={"title": "   "},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["title"] is None


@pytest.mark.asyncio
async def test_update_conversation__not_found(client: AsyncClient) -> None:
    """Test updating non-existent conversation returns 404."""
    fake_id = uuid4()
    response = await client.patch(
        f"/v1/conversations/{fake_id}",
        json={"title": "New Title"},
    )

    assert response.status_code == 404
    data = response.json()
    assert "conversation_not_found" in str(data)


@pytest.mark.asyncio
async def test_update_conversation__archived_conversation(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test updating archived conversation returns 404."""
    conv_id = await conversation_db.create_conversation(title="Test")
    await conversation_db.delete_conversation(conv_id)

    response = await client.patch(
        f"/v1/conversations/{conv_id}",
        json={"title": "Updated"},
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_conversation__title_exceeds_max_length(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test updating with title exceeding 200 characters returns 422."""
    conv_id = await conversation_db.create_conversation(title="Test")

    long_title = "a" * 201  # 201 characters
    response = await client.patch(
        f"/v1/conversations/{conv_id}",
        json={"title": long_title},
    )

    assert response.status_code == 422
    data = response.json()
    assert "200 characters" in str(data)


@pytest.mark.asyncio
async def test_update_conversation__invalid_uuid(client: AsyncClient) -> None:
    """Test updating with invalid UUID returns 422."""
    response = await client.patch(
        "/v1/conversations/not-a-uuid",
        json={"title": "New Title"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_update_conversation__extra_fields_rejected(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test that extra fields in request body are rejected."""
    conv_id = await conversation_db.create_conversation(title="Test")

    response = await client.patch(
        f"/v1/conversations/{conv_id}",
        json={"title": "New Title", "extra_field": "should fail"},
    )

    assert response.status_code == 422


# =============================================================================
# Error Cases - Database Unavailable
# =============================================================================


@pytest.mark.asyncio
async def test_endpoints__database_unavailable(client: AsyncClient) -> None:
    """Test that all endpoints return 503 when database is unavailable."""
    # Override to return None (simulates failed DB connection)
    app.dependency_overrides[get_conversation_db] = lambda: None

    # List conversations
    response = await client.get("/v1/conversations")
    assert response.status_code == 503
    assert "conversation_storage_unavailable" in str(response.json())

    # Get conversation
    fake_id = uuid4()
    response = await client.get(f"/v1/conversations/{fake_id}")
    assert response.status_code == 503

    # Delete conversation
    response = await client.delete(f"/v1/conversations/{fake_id}")
    assert response.status_code == 503

    # Update conversation
    response = await client.patch(
        f"/v1/conversations/{fake_id}",
        json={"title": "New Title"},
    )
    assert response.status_code == 503

    # Cleanup
    app.dependency_overrides.clear()
