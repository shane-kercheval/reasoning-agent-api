"""
Integration tests for message search functionality.

Tests both database layer (ConversationDB.search_messages) and REST API
(GET /v1/messages/search) with real PostgreSQL using testcontainers.

Run with: pytest tests/integration_tests/test_message_search.py -v
"""

import pytest
import pytest_asyncio
from uuid import UUID
from httpx import AsyncClient, ASGITransport
from api.main import app
from api.database import ConversationDB
from api.dependencies import (
    get_conversation_db,
    get_tools,
    get_prompt_manager,
)
from api.prompt_manager import PromptManager

pytestmark = pytest.mark.integration


# =============================================================================
# Database Layer Tests - ConversationDB.search_messages()
# =============================================================================


@pytest.mark.asyncio
async def test_search_messages__basic_search(conversation_db: ConversationDB) -> None:
    """Test basic message search returns matching results."""
    # Create conversation with searchable messages
    conv_id = await conversation_db.create_conversation(title="Weather Chat")
    await conversation_db.append_messages(
        conv_id,
        [
            {"role": "user", "content": "What is the weather like today?"},
            {"role": "assistant", "content": "The weather is sunny and warm."},
            {"role": "user", "content": "Should I bring an umbrella?"},
            {"role": "assistant", "content": "No, you won't need an umbrella today."},
        ],
    )

    # Search for "weather"
    results, total = await conversation_db.search_messages("weather")

    assert total == 2  # "weather" appears in 2 messages
    assert len(results) == 2
    assert all(r.conversation_id == conv_id for r in results)
    assert "weather" in results[0].content.lower()
    assert "weather" in results[1].content.lower()


@pytest.mark.asyncio
async def test_search_messages__case_insensitive(conversation_db: ConversationDB) -> None:
    """Test search is case-insensitive."""
    conv_id = await conversation_db.create_conversation()
    await conversation_db.append_messages(
        conv_id,
        [
            {"role": "user", "content": "Python is a great programming language"},
            {"role": "assistant", "content": "Yes, PYTHON is very popular"},
        ],
    )

    # All these searches should find both messages
    for query in ["python", "Python", "PYTHON", "PyThOn"]:
        _results, total = await conversation_db.search_messages(query)
        assert total == 2, f"Failed for query: {query}"


@pytest.mark.asyncio
async def test_search_messages__multi_word_query(conversation_db: ConversationDB) -> None:
    """Test multi-word search queries."""
    conv_id = await conversation_db.create_conversation()
    await conversation_db.append_messages(
        conv_id,
        [
            {"role": "user", "content": "Tell me about machine learning algorithms"},
            {"role": "assistant", "content": "Machine learning uses various algorithms"},
            {"role": "user", "content": "What about deep learning?"},
        ],
    )

    # Search for "machine learning"
    results, total = await conversation_db.search_messages("machine learning")

    assert total == 2  # Both messages with "machine learning"
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_messages__relevance_ranking(conversation_db: ConversationDB) -> None:
    """Test results are ordered by relevance score."""
    conv_id = await conversation_db.create_conversation()
    await conversation_db.append_messages(
        conv_id,
        [
            # This message has "weather" once
            {"role": "user", "content": "What is the weather today?"},
            # This message has "weather" three times (higher relevance)
            {"role": "assistant", "content": "The weather is great! Weather forecasts say weather will stay nice."},  # noqa: E501
        ],
    )

    results, total = await conversation_db.search_messages("weather")

    assert total == 2
    # Higher relevance (more mentions) should come first
    assert results[0].relevance > results[1].relevance
    assert results[0].content.count("weather") > results[1].content.count("weather")


@pytest.mark.asyncio
async def test_search_messages__snippet_highlighting(conversation_db: ConversationDB) -> None:
    """Test that snippets contain highlighted matches."""
    conv_id = await conversation_db.create_conversation()
    await conversation_db.append_messages(
        conv_id,
        [
            {"role": "user", "content": "What are the best practices for Python programming?"},
        ],
    )

    results, total = await conversation_db.search_messages("Python")

    assert total == 1
    # Snippet should contain the matched word with HTML highlighting
    assert results[0].snippet is not None
    assert "<b>" in results[0].snippet  # PostgreSQL ts_headline uses <b> tags
    assert "</b>" in results[0].snippet


@pytest.mark.asyncio
async def test_search_messages__pagination(conversation_db: ConversationDB) -> None:
    """Test pagination with limit and offset."""
    conv_id = await conversation_db.create_conversation()

    # Create 5 messages all containing "test"
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Test message number {i}"}
        for i in range(5)
    ]
    await conversation_db.append_messages(conv_id, messages)

    # Get first page (2 items)
    page1, total = await conversation_db.search_messages("test", limit=2, offset=0)
    assert total == 5
    assert len(page1) == 2

    # Get second page (2 items)
    page2, total = await conversation_db.search_messages("test", limit=2, offset=2)
    assert total == 5
    assert len(page2) == 2

    # Get third page (1 item)
    page3, total = await conversation_db.search_messages("test", limit=2, offset=4)
    assert total == 5
    assert len(page3) == 1

    # Verify no duplicates across pages
    all_ids = {r.message_id for r in page1 + page2 + page3}
    assert len(all_ids) == 5


@pytest.mark.asyncio
async def test_search_messages__archive_filter_active(conversation_db: ConversationDB) -> None:
    """Test archive_filter='active' excludes archived conversations."""
    # Create active conversation
    active_id = await conversation_db.create_conversation(title="Active")
    await conversation_db.append_messages(
        active_id,
        [{"role": "user", "content": "findme in active conversation"}],
    )

    # Create and archive conversation
    archived_id = await conversation_db.create_conversation(title="Archived")
    await conversation_db.append_messages(
        archived_id,
        [{"role": "user", "content": "findme in archived conversation"}],
    )
    await conversation_db.delete_conversation(archived_id)

    # Search with archive_filter="active" (default)
    results, total = await conversation_db.search_messages("findme", archive_filter="active")

    assert total == 1
    assert results[0].conversation_id == active_id
    assert results[0].archived is False


@pytest.mark.asyncio
async def test_search_messages__archive_filter_archived(conversation_db: ConversationDB) -> None:
    """Test archive_filter='archived' returns only archived conversations."""
    # Create active conversation
    active_id = await conversation_db.create_conversation(title="Active")
    await conversation_db.append_messages(
        active_id,
        [{"role": "user", "content": "findme in active conversation"}],
    )

    # Create and archive conversation
    archived_id = await conversation_db.create_conversation(title="Archived")
    await conversation_db.append_messages(
        archived_id,
        [{"role": "user", "content": "findme in archived conversation"}],
    )
    await conversation_db.delete_conversation(archived_id)

    # Search with archive_filter="archived"
    results, total = await conversation_db.search_messages("findme", archive_filter="archived")

    assert total == 1
    assert results[0].conversation_id == archived_id
    assert results[0].archived is True


@pytest.mark.asyncio
async def test_search_messages__archive_filter_all(conversation_db: ConversationDB) -> None:
    """Test archive_filter='all' returns both active and archived conversations."""
    # Create active conversation
    active_id = await conversation_db.create_conversation(title="Active")
    await conversation_db.append_messages(
        active_id,
        [{"role": "user", "content": "findme in active conversation"}],
    )

    # Create and archive conversation
    archived_id = await conversation_db.create_conversation(title="Archived")
    await conversation_db.append_messages(
        archived_id,
        [{"role": "user", "content": "findme in archived conversation"}],
    )
    await conversation_db.delete_conversation(archived_id)

    # Search with archive_filter="all"
    results, total = await conversation_db.search_messages("findme", archive_filter="all")

    assert total == 2
    conv_ids = {r.conversation_id for r in results}
    assert active_id in conv_ids
    assert archived_id in conv_ids


@pytest.mark.asyncio
async def test_search_messages__invalid_archive_filter(conversation_db: ConversationDB) -> None:
    """Test invalid archive_filter raises ValueError."""
    with pytest.raises(ValueError, match="Invalid archive_filter"):
        await conversation_db.search_messages("test", archive_filter="invalid")


@pytest.mark.asyncio
async def test_search_messages__no_matches(conversation_db: ConversationDB) -> None:
    """Test search with no matching results."""
    conv_id = await conversation_db.create_conversation()
    await conversation_db.append_messages(
        conv_id,
        [{"role": "user", "content": "Hello world"}],
    )

    results, total = await conversation_db.search_messages("nonexistent")

    assert total == 0
    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_messages__empty_database(conversation_db: ConversationDB) -> None:
    """Test search on empty database."""
    results, total = await conversation_db.search_messages("anything")

    assert total == 0
    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_messages__null_content_ignored(conversation_db: ConversationDB) -> None:
    """Test messages with null content are not included in search."""
    conv_id = await conversation_db.create_conversation()
    await conversation_db.append_messages(
        conv_id,
        [
            {"role": "user", "content": "Find this message"},
            {"role": "assistant", "content": None},  # Null content
        ],
    )

    results, total = await conversation_db.search_messages("message")

    assert total == 1
    assert results[0].content == "Find this message"


@pytest.mark.asyncio
async def test_search_messages__multiple_conversations(conversation_db: ConversationDB) -> None:
    """Test search across multiple conversations."""
    # Create 3 conversations with different content
    conv1_id = await conversation_db.create_conversation(title="Python Basics")
    await conversation_db.append_messages(
        conv1_id,
        [{"role": "user", "content": "How do I learn Python?"}],
    )

    conv2_id = await conversation_db.create_conversation(title="JavaScript Basics")
    await conversation_db.append_messages(
        conv2_id,
        [{"role": "user", "content": "How do I learn JavaScript?"}],
    )

    conv3_id = await conversation_db.create_conversation(title="Python Advanced")
    await conversation_db.append_messages(
        conv3_id,
        [{"role": "user", "content": "Advanced Python techniques"}],
    )

    # Search for "Python"
    results, total = await conversation_db.search_messages("Python")

    assert total == 2
    conv_ids = {r.conversation_id for r in results}
    assert conv1_id in conv_ids
    assert conv3_id in conv_ids
    assert conv2_id not in conv_ids


@pytest.mark.asyncio
async def test_search_messages__includes_conversation_metadata(
    conversation_db: ConversationDB,
) -> None:
    """Test search results include conversation title and metadata."""
    conv_id = await conversation_db.create_conversation(title="My Important Chat")
    await conversation_db.append_messages(
        conv_id,
        [{"role": "user", "content": "Searchable content here"}],
    )

    results, total = await conversation_db.search_messages("Searchable")

    assert total == 1
    assert results[0].conversation_title == "My Important Chat"
    assert results[0].conversation_id == conv_id
    assert isinstance(results[0].message_id, UUID)
    assert results[0].role == "user"
    assert results[0].archived is False


# =============================================================================
# REST API Tests - GET /v1/messages/search
# =============================================================================


@pytest_asyncio.fixture
async def client(conversation_db: ConversationDB) -> AsyncClient:
    """Create async test client with test database dependency override."""
    app.dependency_overrides[get_conversation_db] = lambda: conversation_db
    app.dependency_overrides[get_tools] = lambda: []
    app.dependency_overrides[get_prompt_manager] = lambda: PromptManager()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_api_search_messages__success(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test successful search via REST API."""
    # Create test data
    conv_id = await conversation_db.create_conversation(title="Test Chat")
    await conversation_db.append_messages(
        conv_id,
        [
            {"role": "user", "content": "Tell me about Python programming"},
            {"role": "assistant", "content": "Python is a versatile language"},
        ],
    )

    # Search via API
    response = await client.get("/v1/messages/search?q=Python")

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "results" in data
    assert "total" in data
    assert "limit" in data
    assert "offset" in data
    assert "query" in data

    # Verify data
    assert data["total"] == 2
    assert len(data["results"]) == 2
    assert data["query"] == "Python"
    assert data["limit"] == 50  # Default
    assert data["offset"] == 0  # Default

    # Verify result structure
    result = data["results"][0]
    assert "message_id" in result
    assert "conversation_id" in result
    assert "conversation_title" in result
    assert "role" in result
    assert "content" in result
    assert "snippet" in result
    assert "relevance" in result
    assert "created_at" in result
    assert "archived" in result

    assert result["conversation_title"] == "Test Chat"


@pytest.mark.asyncio
async def test_api_search_messages__pagination(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test pagination via REST API."""
    conv_id = await conversation_db.create_conversation()
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Test message {i}"}
        for i in range(5)
    ]
    await conversation_db.append_messages(conv_id, messages)

    # Page 1
    response = await client.get("/v1/messages/search?q=Test&limit=2&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 5
    assert len(data["results"]) == 2
    assert data["limit"] == 2
    assert data["offset"] == 0

    # Page 2
    response = await client.get("/v1/messages/search?q=Test&limit=2&offset=2")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 5
    assert len(data["results"]) == 2
    assert data["limit"] == 2
    assert data["offset"] == 2


@pytest.mark.asyncio
async def test_api_search_messages__archive_filter(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test archive_filter parameter via REST API."""
    # Create active conversation
    active_id = await conversation_db.create_conversation(title="Active")
    await conversation_db.append_messages(
        active_id,
        [{"role": "user", "content": "findme"}],
    )

    # Create and archive conversation
    archived_id = await conversation_db.create_conversation(title="Archived")
    await conversation_db.append_messages(
        archived_id,
        [{"role": "user", "content": "findme"}],
    )
    await conversation_db.delete_conversation(archived_id)

    # Test active (default)
    response = await client.get("/v1/messages/search?q=findme")
    assert response.status_code == 200
    assert response.json()["total"] == 1

    # Test archived
    response = await client.get("/v1/messages/search?q=findme&archive_filter=archived")
    assert response.status_code == 200
    assert response.json()["total"] == 1

    # Test all
    response = await client.get("/v1/messages/search?q=findme&archive_filter=all")
    assert response.status_code == 200
    assert response.json()["total"] == 2


@pytest.mark.asyncio
async def test_api_search_messages__missing_query_param(client: AsyncClient) -> None:
    """Test missing required 'q' parameter returns 422."""
    response = await client.get("/v1/messages/search")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_api_search_messages__empty_query(client: AsyncClient) -> None:
    """Test empty query string returns 422."""
    response = await client.get("/v1/messages/search?q=")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_api_search_messages__invalid_limit(client: AsyncClient) -> None:
    """Test invalid limit parameter returns 422."""
    # Limit too small
    response = await client.get("/v1/messages/search?q=test&limit=0")
    assert response.status_code == 422

    # Limit too large
    response = await client.get("/v1/messages/search?q=test&limit=101")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_api_search_messages__invalid_offset(client: AsyncClient) -> None:
    """Test invalid offset parameter returns 422."""
    response = await client.get("/v1/messages/search?q=test&offset=-1")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_api_search_messages__invalid_archive_filter(client: AsyncClient) -> None:
    """Test invalid archive_filter parameter returns 422."""
    response = await client.get("/v1/messages/search?q=test&archive_filter=invalid")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_api_search_messages__database_unavailable(client: AsyncClient) -> None:
    """Test 503 error when database is unavailable."""
    # Override to return None
    app.dependency_overrides[get_conversation_db] = lambda: None

    response = await client.get("/v1/messages/search?q=test")

    assert response.status_code == 503
    assert "conversation_storage_unavailable" in str(response.json())

    # Cleanup
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_api_search_messages__no_results(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test search with no matching results returns empty list."""
    conv_id = await conversation_db.create_conversation()
    await conversation_db.append_messages(
        conv_id,
        [{"role": "user", "content": "Hello world"}],
    )

    response = await client.get("/v1/messages/search?q=nonexistent")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert len(data["results"]) == 0


@pytest.mark.asyncio
async def test_api_search_messages__case_insensitive_query(
    client: AsyncClient,
    conversation_db: ConversationDB,
) -> None:
    """Test case-insensitive search via API."""
    conv_id = await conversation_db.create_conversation()
    await conversation_db.append_messages(
        conv_id,
        [{"role": "user", "content": "Python is great"}],
    )

    # All should return same result
    for query in ["python", "PYTHON", "PyThOn"]:
        response = await client.get(f"/v1/messages/search?q={query}")
        assert response.status_code == 200
        assert response.json()["total"] == 1
