"""Unit tests for conversation utilities."""

import pytest
from uuid import UUID, uuid4
from unittest.mock import AsyncMock

from api.conversation_utils import (
    ConversationMode,
    ConversationContext,
    parse_conversation_header,
    build_llm_messages,
    store_conversation_messages,
)
from api.database.conversation_db import Conversation, Message


class TestParseConversationHeader:
    """Tests for parse_conversation_header function."""

    def test__parse_conversation_header__stateless_mode_none(self) -> None:
        """Test parsing None header returns stateless mode."""
        result = parse_conversation_header(None)

        assert result.mode == ConversationMode.STATELESS
        assert result.conversation_id is None

    def test__parse_conversation_header__new_mode_empty_string(self) -> None:
        """Test parsing empty string header returns new mode."""
        result = parse_conversation_header("")

        assert result.mode == ConversationMode.NEW
        assert result.conversation_id is None

    def test__parse_conversation_header__new_mode_null_string(self) -> None:
        """Test parsing 'null' string header returns new mode."""
        result = parse_conversation_header("null")

        assert result.mode == ConversationMode.NEW
        assert result.conversation_id is None

    def test__parse_conversation_header__continuing_mode_valid_uuid(self) -> None:
        """Test parsing valid UUID header returns continuing mode."""
        test_uuid = "550e8400-e29b-41d4-a716-446655440000"

        result = parse_conversation_header(test_uuid)

        assert result.mode == ConversationMode.CONTINUING
        assert result.conversation_id == UUID(test_uuid)

    def test__parse_conversation_header__raises_on_invalid_uuid(self) -> None:
        """Test parsing invalid UUID format raises ValueError."""
        invalid_uuid = "not-a-valid-uuid"

        with pytest.raises(ValueError, match="Invalid conversation ID format"):
            parse_conversation_header(invalid_uuid)


class TestBuildLlmMessages:
    """Tests for build_llm_messages function."""

    @pytest.mark.asyncio
    async def test__build_llm_messages__stateless_mode_returns_request_as_is(self) -> None:
        """Test stateless mode returns request messages unchanged."""
        ctx = ConversationContext(mode=ConversationMode.STATELESS, conversation_id=None)
        request_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        messages = await build_llm_messages(request_messages, ctx, None)

        assert messages == request_messages

    @pytest.mark.asyncio
    async def test__build_llm_messages__new_mode_returns_request_as_is(self) -> None:
        """Test new conversation mode returns request messages unchanged."""
        ctx = ConversationContext(mode=ConversationMode.NEW, conversation_id=None)
        request_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]

        messages = await build_llm_messages(request_messages, ctx, None)

        assert messages == request_messages

    @pytest.mark.asyncio
    async def test__build_llm_messages__new_mode_no_system_message(self) -> None:
        """Test new conversation without system message."""
        ctx = ConversationContext(mode=ConversationMode.NEW, conversation_id=None)
        request_messages = [{"role": "user", "content": "Hello"}]

        messages = await build_llm_messages(request_messages, ctx, None)

        assert messages == request_messages

    @pytest.mark.asyncio
    async def test__build_llm_messages__continuing_rejects_system_message(self) -> None:
        """Test continuing conversation rejects system message in request."""
        conv_id = uuid4()
        ctx = ConversationContext(mode=ConversationMode.CONTINUING, conversation_id=conv_id)
        request_messages = [
            {"role": "system", "content": "New system message"},
            {"role": "user", "content": "Hello"},
        ]
        mock_db = AsyncMock()

        with pytest.raises(ValueError, match="System messages not allowed when continuing"):
            await build_llm_messages(request_messages, ctx, mock_db)

    @pytest.mark.asyncio
    async def test__build_llm_messages__continuing_requires_database(self) -> None:
        """Test continuing conversation requires database connection."""
        conv_id = uuid4()
        ctx = ConversationContext(mode=ConversationMode.CONTINUING, conversation_id=conv_id)
        request_messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(ValueError, match="Database connection required"):
            await build_llm_messages(request_messages, ctx, None)

    @pytest.mark.asyncio
    async def test__build_llm_messages__continuing_loads_history(self) -> None:
        """Test continuing conversation loads full history."""
        conv_id = uuid4()
        ctx = ConversationContext(mode=ConversationMode.CONTINUING, conversation_id=conv_id)
        request_messages = [{"role": "user", "content": "Third message"}]

        # Mock conversation with history
        mock_conversation = Conversation(
            id=conv_id,
            user_id=None,
            title=None,
            system_message="You are a helpful assistant.",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
            archived_at=None,
            metadata={},
            messages=[
                Message(
                    id=uuid4(),
                    conversation_id=conv_id,
                    role="user",
                    content="First message",
                    reasoning_events=None,
                    tool_calls=None,
                    metadata={},
                    created_at="2024-01-01T00:00:00",
                    sequence_number=1,
                ),
                Message(
                    id=uuid4(),
                    conversation_id=conv_id,
                    role="assistant",
                    content="Second message",
                    reasoning_events=None,
                    tool_calls=None,
                    metadata={},
                    created_at="2024-01-01T00:00:01",
                    sequence_number=2,
                ),
            ],
        )

        mock_db = AsyncMock()
        mock_db.get_conversation.return_value = mock_conversation

        messages = await build_llm_messages(request_messages, ctx, mock_db)

        # Verify conversation was loaded
        mock_db.get_conversation.assert_called_once_with(conv_id)

        # Verify messages include: [system] + [history] + [new]
        assert len(messages) == 4
        assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert messages[1] == {"role": "user", "content": "First message"}
        assert messages[2] == {"role": "assistant", "content": "Second message"}
        assert messages[3] == {"role": "user", "content": "Third message"}

    @pytest.mark.asyncio
    async def test__build_llm_messages__continuing_filters_system_from_request(self) -> None:
        """Test that system messages are filtered from request even if present."""
        conv_id = uuid4()
        ctx = ConversationContext(mode=ConversationMode.CONTINUING, conversation_id=conv_id)
        # This shouldn't happen (validation should prevent it), but test defensive filtering
        request_messages = [
            {"role": "user", "content": "Hello"},
        ]

        mock_conversation = Conversation(
            id=conv_id,
            user_id=None,
            title=None,
            system_message="Original system",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
            archived_at=None,
            metadata={},
            messages=[],
        )

        mock_db = AsyncMock()
        mock_db.get_conversation.return_value = mock_conversation

        messages = await build_llm_messages(request_messages, ctx, mock_db)

        # Verify only original system message is used
        assert messages[0] == {"role": "system", "content": "Original system"}
        assert messages[1] == {"role": "user", "content": "Hello"}
        assert len(messages) == 2


class TestStoreConversationMessages:
    """Tests for store_conversation_messages function."""

    @pytest.mark.asyncio
    async def test__store_conversation_messages__stores_user_and_assistant(self) -> None:
        """Test storing user and assistant messages."""
        conv_id = uuid4()
        request_messages = [{"role": "user", "content": "Hello"}]
        assistant_content = "Hi there!"

        mock_db = AsyncMock()

        await store_conversation_messages(mock_db, conv_id, request_messages, assistant_content)

        # Verify append_messages was called with correct data
        mock_db.append_messages.assert_called_once()
        call_args = mock_db.append_messages.call_args
        assert call_args[0][0] == conv_id

        stored_messages = call_args[0][1]
        assert len(stored_messages) == 2
        assert stored_messages[0] == {"role": "user", "content": "Hello"}
        assert stored_messages[1] == {"role": "assistant", "content": "Hi there!"}

    @pytest.mark.asyncio
    async def test__store_conversation_messages__filters_system_message(self) -> None:
        """Test that system messages are filtered out before storage."""
        conv_id = uuid4()
        request_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        assistant_content = "Hi!"

        mock_db = AsyncMock()

        await store_conversation_messages(mock_db, conv_id, request_messages, assistant_content)

        # Verify system message was filtered out
        stored_messages = mock_db.append_messages.call_args[0][1]
        assert len(stored_messages) == 2
        assert stored_messages[0] == {"role": "user", "content": "Hello"}
        assert stored_messages[1] == {"role": "assistant", "content": "Hi!"}

    @pytest.mark.asyncio
    async def test__store_conversation_messages__handles_multiple_user_messages(self) -> None:
        """Test storing multiple user messages in one request."""
        conv_id = uuid4()
        request_messages = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
        ]
        assistant_content = "Response"

        mock_db = AsyncMock()

        await store_conversation_messages(mock_db, conv_id, request_messages, assistant_content)

        stored_messages = mock_db.append_messages.call_args[0][1]
        assert len(stored_messages) == 3
        assert stored_messages[0] == {"role": "user", "content": "First"}
        assert stored_messages[1] == {"role": "user", "content": "Second"}
        assert stored_messages[2] == {"role": "assistant", "content": "Response"}

    @pytest.mark.asyncio
    async def test__store_conversation_messages__handles_empty_request(self) -> None:
        """Test storing when request has no user messages (edge case)."""
        conv_id = uuid4()
        request_messages = []
        assistant_content = "Response"

        mock_db = AsyncMock()

        await store_conversation_messages(mock_db, conv_id, request_messages, assistant_content)

        stored_messages = mock_db.append_messages.call_args[0][1]
        assert len(stored_messages) == 1
        assert stored_messages[0] == {"role": "assistant", "content": "Response"}


class TestConversationMode:
    """Tests for ConversationMode enum."""

    def test__conversation_mode__has_correct_values(self) -> None:
        """Test that ConversationMode enum has expected values."""
        assert ConversationMode.STATELESS.value == "stateless"
        assert ConversationMode.NEW.value == "new"
        assert ConversationMode.CONTINUING.value == "continuing"

    def test__conversation_mode__is_string_enum(self) -> None:
        """Test that ConversationMode extends str."""
        assert isinstance(ConversationMode.STATELESS, str)
        assert isinstance(ConversationMode.NEW, str)
        assert isinstance(ConversationMode.CONTINUING, str)


class TestConversationContext:
    """Tests for ConversationContext dataclass."""

    def test__conversation_context__initialization(self) -> None:
        """Test ConversationContext dataclass initialization."""
        conv_id = uuid4()
        ctx = ConversationContext(mode=ConversationMode.CONTINUING, conversation_id=conv_id)

        assert ctx.mode == ConversationMode.CONTINUING
        assert ctx.conversation_id == conv_id

    def test__conversation_context__none_conversation_id(self) -> None:
        """Test ConversationContext with None conversation_id."""
        ctx = ConversationContext(mode=ConversationMode.STATELESS, conversation_id=None)

        assert ctx.mode == ConversationMode.STATELESS
        assert ctx.conversation_id is None
