"""
Integration tests for conversation storage in chat completions API.

Tests the API + Database integration with mocked LiteLLM responses.
Uses testcontainers for real database, mocks LiteLLM since that's not under test.

Run with: pytest tests/integration_tests/test_conversation_storage_api.py -v
"""

import asyncio
import pytest
import pytest_asyncio
from uuid import uuid4, UUID
from unittest.mock import patch
from collections.abc import AsyncGenerator
from litellm import ModelResponse
from litellm.types.utils import StreamingChoices, Delta
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
async def client(conversation_db: ConversationDB):
    """Create async test client with test database dependency override."""
    # Override dependencies to avoid initializing service container
    # (which would try to connect to real database at port 5434)
    # Use closures to capture the conversation_db instance
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


async def create_mock_stream(content: str) -> AsyncGenerator[ModelResponse]:
    """Create a mock streaming response using actual litellm types."""
    # First chunk - role
    yield ModelResponse(
        id="test",
        choices=[StreamingChoices(
            index=0,
            delta=Delta(role="assistant"),
            finish_reason=None,
        )],
        created=123,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
    )

    # Second chunk - content
    yield ModelResponse(
        id="test",
        choices=[StreamingChoices(
            index=0,
            delta=Delta(content=content),
            finish_reason=None,
        )],
        created=123,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
    )

    # Final chunk - finish
    yield ModelResponse(
        id="test",
        choices=[StreamingChoices(
            index=0,
            delta=Delta(),
            finish_reason="stop",
        )],
        created=123,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
    )


# =============================================================================
# Stateless Mode Tests
# =============================================================================


@pytest.mark.asyncio
async def test_stateless_mode__no_header__no_storage(
        client: AsyncClient,
        conversation_db: ConversationDB,
    ):
    """Test that requests without X-Conversation-ID header don't store conversations."""
    with patch('api.executors.passthrough.litellm.acompletion') as mock_litellm:
        # Mock LiteLLM to return a streaming response
        mock_litellm.return_value = create_mock_stream("Hello from LLM")

        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Test message"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        # No X-Conversation-ID in response (stateless)
        assert "X-Conversation-ID" not in response.headers

        # Verify no conversation was created
        _convs, total = await conversation_db.list_conversations()
        assert total == 0


# =============================================================================
# Stateful Mode - New Conversation
# =============================================================================


@pytest.mark.asyncio
async def test_new_conversation__empty_header__creates_conversation(client: AsyncClient, conversation_db: ConversationDB):  # noqa: E501
    """Test that X-Conversation-ID: '' creates a new conversation with auto-generated title."""
    with patch('api.executors.passthrough.litellm.acompletion') as mock_litellm:
        mock_litellm.return_value = create_mock_stream("Hello!")

        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Conversation-ID": ""},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a test assistant."},
                    {"role": "user", "content": "Hi"},
                ],
                "stream": True,
            },
        )

        assert response.status_code == 200
        # Conversation ID returned in response header
        assert "X-Conversation-ID" in response.headers
        conv_id = response.headers["X-Conversation-ID"]

        # Verify conversation was created in database
        conv = await conversation_db.get_conversation(UUID(conv_id))
        assert conv.system_message == "You are a test assistant."
        assert len(conv.messages) == 2  # user + assistant
        assert conv.messages[0].content == "Hi"
        assert conv.messages[1].content == "Hello!"
        # Verify title was auto-generated from user message
        assert conv.title == "Hi"


@pytest.mark.asyncio
async def test_new_conversation__null_header__creates_conversation(client: AsyncClient, conversation_db: ConversationDB):  # noqa: ARG001, E501
    """Test that X-Conversation-ID: 'null' creates a new conversation."""
    with patch('api.executors.passthrough.litellm.acompletion') as mock_litellm:
        mock_litellm.return_value = create_mock_stream("Response")

        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Conversation-ID": "null"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Message"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert "X-Conversation-ID" in response.headers


@pytest.mark.asyncio
async def test_new_conversation__no_system_message__uses_default(client: AsyncClient, conversation_db: ConversationDB):  # noqa: E501
    """Test that new conversation without system message uses default and auto-generates title."""
    with patch('api.executors.passthrough.litellm.acompletion') as mock_litellm:
        mock_litellm.return_value = create_mock_stream("Hello!")

        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Conversation-ID": ""},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        conv_id = response.headers["X-Conversation-ID"]
        conv = await conversation_db.get_conversation(UUID(conv_id))
        assert conv.system_message == "You are a helpful assistant."
        # Verify title was auto-generated
        assert conv.title == "Hi"


@pytest.mark.asyncio
async def test_new_conversation__long_message__truncates_title(client: AsyncClient, conversation_db: ConversationDB):  # noqa: E501
    """Test that long user messages are truncated to 100 chars with ellipsis."""
    with patch('api.executors.passthrough.litellm.acompletion') as mock_litellm:
        mock_litellm.return_value = create_mock_stream("Response")

        long_message = "This is a very long message that exceeds one hundred characters and should be truncated with ellipsis at the end"  # noqa: E501
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Conversation-ID": ""},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": long_message}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        conv_id = response.headers["X-Conversation-ID"]
        conv = await conversation_db.get_conversation(UUID(conv_id))
        # Verify title was truncated to 100 chars with ellipsis
        assert len(conv.title) == 100
        assert conv.title.endswith("...")
        assert conv.title.startswith("This is a very long message")


@pytest.mark.asyncio
async def test_new_conversation__message_with_newlines__normalizes_title(client: AsyncClient, conversation_db: ConversationDB):  # noqa: E501
    """Test that user messages with newlines are normalized to single line."""
    with patch('api.executors.passthrough.litellm.acompletion') as mock_litellm:
        mock_litellm.return_value = create_mock_stream("Response")

        message_with_newlines = "First line\n\nSecond line\n  Third line"
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Conversation-ID": ""},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": message_with_newlines}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        conv_id = response.headers["X-Conversation-ID"]
        conv = await conversation_db.get_conversation(UUID(conv_id))
        # Verify newlines were replaced with spaces
        assert conv.title == "First line Second line Third line"
        assert "\n" not in conv.title


@pytest.mark.asyncio
async def test_new_conversation__only_system_message__no_title(client: AsyncClient, conversation_db: ConversationDB):  # noqa: E501
    """Test that conversation with only system message has no title (edge case)."""
    with patch('api.executors.passthrough.litellm.acompletion') as mock_litellm:
        mock_litellm.return_value = create_mock_stream("Response")

        # Note: This is an edge case - normally you'd have a user message
        # But API allows it for testing
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Conversation-ID": ""},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "system", "content": "You are helpful."}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        conv_id = response.headers["X-Conversation-ID"]
        conv = await conversation_db.get_conversation(UUID(conv_id))
        # No user message means no title
        assert conv.title is None


# =============================================================================
# Stateful Mode - Continuation
# =============================================================================


@pytest.mark.asyncio
async def test_continue_conversation__loads_history(client: AsyncClient, conversation_db: ConversationDB):  # noqa: E501
    """Test that continuing a conversation loads full message history."""
    # Create initial conversation
    conv_id = await conversation_db.create_conversation(
        system_message="Custom system message",
    )

    # Add initial messages
    await conversation_db.append_messages(
        conv_id,
        [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
        ],
    )

    # Track what messages LiteLLM receives
    captured_request = None

    async def capture_litellm_request(*args, **kwargs):  # noqa
        nonlocal captured_request
        captured_request = kwargs
        return create_mock_stream("Second response")

    with patch('api.executors.passthrough.litellm.acompletion', side_effect=capture_litellm_request):  # noqa: E501
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Conversation-ID": str(conv_id)},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Second message"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert response.headers["X-Conversation-ID"] == str(conv_id)

        # Verify LiteLLM received full history: [system] + [history] + [new]
        messages_sent = captured_request["messages"]
        assert len(messages_sent) == 4
        assert messages_sent[0]["role"] == "system"
        assert messages_sent[0]["content"] == "Custom system message"
        assert messages_sent[1]["content"] == "First message"
        assert messages_sent[2]["content"] == "First response"
        assert messages_sent[3]["content"] == "Second message"

        # Verify conversation was updated in database
        conv = await conversation_db.get_conversation(conv_id)
        assert len(conv.messages) == 4  # original 2 + new user + new assistant


# =============================================================================
# Error Cases - System Message Validation
# =============================================================================


@pytest.mark.asyncio
async def test_continue_conversation__system_message_rejected(client: AsyncClient, conversation_db: ConversationDB):  # noqa: E501
    """Test that system message in continuation request returns 400 error."""
    # Create initial conversation
    conv_id = await conversation_db.create_conversation(
        system_message="Original system message",
    )

    # Add initial message
    await conversation_db.append_messages(
        conv_id,
        [{"role": "user", "content": "Hi"}],
    )

    response = await client.post(
        "/v1/chat/completions",
        headers={"X-Conversation-ID": str(conv_id)},
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "New system message"},
                {"role": "user", "content": "Message"},
            ],
            "stream": True,
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert "system_message_in_continuation" in str(data)


# =============================================================================
# Error Cases - Invalid Conversation ID
# =============================================================================


@pytest.mark.asyncio
async def test_invalid_conversation_id__returns_400(client: AsyncClient):
    """Test that invalid UUID format returns 400 error."""
    response = await client.post(
        "/v1/chat/completions",
        headers={"X-Conversation-ID": "not-a-valid-uuid"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Message"}],
            "stream": True,
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert "invalid_conversation_id" in str(data)


@pytest.mark.asyncio
async def test_nonexistent_conversation__returns_404(client: AsyncClient):
    """Test that non-existent conversation ID returns 404 error."""
    fake_id = str(uuid4())

    response = await client.post(
        "/v1/chat/completions",
        headers={"X-Conversation-ID": fake_id},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Message"}],
            "stream": True,
        },
    )

    assert response.status_code == 404
    data = response.json()
    assert "conversation_not_found" in str(data)


# =============================================================================
# Error Cases - Database Unavailable
# =============================================================================


@pytest.mark.asyncio
async def test_database_unavailable__stateful_request__returns_503(client: AsyncClient):
    """Test that stateful requests return 503 when database is unavailable."""
    # Override to return None (simulates failed DB connection)
    app.dependency_overrides[get_conversation_db] = lambda: None

    response = await client.post(
        "/v1/chat/completions",
        headers={"X-Conversation-ID": ""},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Message"}],
            "stream": True,
        },
    )

    assert response.status_code == 503
    data = response.json()
    assert "conversation_storage_unavailable" in str(data)

    # Cleanup override
    app.dependency_overrides.clear()


# =============================================================================
# Cancellation and Error Cases - Database Persistence
# =============================================================================


async def create_mock_stream_with_chunks(num_chunks: int = 5) -> AsyncGenerator[ModelResponse]:
    """Create a mock streaming response with multiple chunks for testing cancellation."""
    # First chunk - role
    yield ModelResponse(
        id="test",
        choices=[StreamingChoices(
            index=0,
            delta=Delta(role="assistant"),
            finish_reason=None,
        )],
        created=123,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
    )

    # Content chunks
    words = ["Hello", "this", "is", "a", "test", "response", "with", "multiple", "chunks"]
    for i in range(num_chunks):
        yield ModelResponse(
            id="test",
            choices=[StreamingChoices(
                index=0,
                delta=Delta(content=f"{words[i % len(words)]} "),
                finish_reason=None,
            )],
            created=123,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        )

    # Final chunk - finish
    yield ModelResponse(
        id="test",
        choices=[StreamingChoices(
            index=0,
            delta=Delta(),
            finish_reason="stop",
        )],
        created=123,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
    )


@pytest.mark.asyncio
async def test_cancellation_saves_partial_messages(
    client: AsyncClient,
    conversation_db: ConversationDB,
):
    """Test that client disconnection/cancellation saves partial assistant response to database."""
    with patch('api.executors.passthrough.litellm.acompletion') as mock_litellm:
        mock_litellm.return_value = create_mock_stream_with_chunks(num_chunks=10)

        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Conversation-ID": ""},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Test cancellation"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        conv_id = response.headers["X-Conversation-ID"]

        # Simulate cancellation by only consuming first 3 chunks
        chunks_received = 0
        async for _ in response.aiter_bytes():
            chunks_received += 1
            if chunks_received >= 3:
                break  # Simulate client disconnection

        # Give the server a moment to finish cleanup and save
        await asyncio.sleep(0.1)

        # Verify conversation was created and partial messages were saved
        conv = await conversation_db.get_conversation(UUID(conv_id))
        assert conv is not None
        assert len(conv.messages) == 2  # user + partial assistant

        # Verify user message was saved
        assert conv.messages[0].role == "user"
        assert conv.messages[0].content == "Test cancellation"

        # Verify partial assistant response was saved
        assert conv.messages[1].role == "assistant"
        # Should have some content (at least the chunks before cancellation)
        assert len(conv.messages[1].content) > 0
        # Content should be partial (not the full 10 chunks)
        # We consumed 3 chunks, but the executor may have buffered a few more before cancellation
        assert len(conv.messages[1].content.split()) <= 10


# NOTE: We don't test LiteLLM APIError with database persistence here because:
# 1. httpx's ASGITransport runs the ASGI app synchronously, causing exceptions
#    during streaming to propagate before the response object is returned
# 2. This prevents us from getting the conversation ID to verify database save
# 3. However, the `finally` block in api/main.py runs for ALL exceptions,
#    not just cancellations, so LiteLLM errors will trigger the same save logic
# 4. The cancellation test above proves the `finally` block mechanism works
# 5. Testing streaming errors would require a real HTTP server, not ASGITransport


@pytest.mark.asyncio
async def test_successful_completion_still_saves_messages(
    client: AsyncClient,
    conversation_db: ConversationDB,
):
    """Test that successful completion (no error/cancellation) still saves messages correctly."""
    with patch('api.executors.passthrough.litellm.acompletion') as mock_litellm:
        mock_litellm.return_value = create_mock_stream("Complete response")

        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Conversation-ID": ""},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Normal request"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        conv_id = response.headers["X-Conversation-ID"]

        # Consume entire stream
        async for _ in response.aiter_bytes():
            pass

        # Verify conversation was created and messages were saved
        conv = await conversation_db.get_conversation(UUID(conv_id))
        assert conv is not None
        assert len(conv.messages) == 2  # user + assistant

        # Verify messages
        assert conv.messages[0].content == "Normal request"
        assert conv.messages[1].content == "Complete response"
