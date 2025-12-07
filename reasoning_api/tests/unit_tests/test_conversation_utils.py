"""Unit tests for conversation utilities."""

import pytest
from uuid import UUID, uuid4
from unittest.mock import AsyncMock

from reasoning_api.conversation_utils import (
    ConversationMode,
    ConversationContext,
    parse_conversation_header,
    build_llm_messages,
    store_conversation_messages,
    build_metadata_from_response,
    InvalidConversationIDError,
    ResponseMetadata,
    UsageMetadata,
    CostMetadata,
)
from reasoning_api.database.conversation_db import Conversation, Message


class TestBuildMetadataFromResponse:
    """Tests for build_metadata_from_response function."""

    def test__build_metadata__with_usage_and_cost(self) -> None:
        """Test building metadata from response with usage and cost."""
        # Mock a litellm response
        class MockUsage:
            prompt_tokens = 10
            completion_tokens = 5
            total_tokens = 15

            def model_dump(self) -> dict:
                return {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "total_tokens": self.total_tokens,
                }

        class MockResponse:
            usage = MockUsage()
            model = "gpt-4o-mini"

        response = MockResponse()
        metadata = build_metadata_from_response(response)

        assert metadata.usage is not None
        assert metadata.usage.prompt_tokens == 10
        assert metadata.usage.completion_tokens == 5
        assert metadata.usage.total_tokens == 15
        assert metadata.model == "gpt-4o-mini"
        # Cost may or may not be present depending on litellm config

    def test__build_metadata__no_usage(self) -> None:
        """Test building metadata when response has no usage."""
        class MockResponse:
            model = "gpt-4o-mini"

        response = MockResponse()
        metadata = build_metadata_from_response(response)

        assert metadata.usage is None
        assert metadata.model == "gpt-4o-mini"

    def test__build_metadata__no_model(self) -> None:
        """Test building metadata when response has no model."""
        class MockResponse:
            pass

        response = MockResponse()
        metadata = build_metadata_from_response(response)

        assert metadata.model is None
        assert metadata.usage is None

    def test__build_metadata__usage_details_with_none_values(self) -> None:
        """Test building metadata when token details contain None values.

        LiteLLM responses often have None values for unsupported token types
        like audio_tokens or image_tokens. This should not cause validation errors.
        """
        class MockUsage:
            prompt_tokens = 100
            completion_tokens = 50
            total_tokens = 150
            # Real-world LiteLLM response structure with None values
            completion_tokens_details = {
                "reasoning_tokens": 10,
                "audio_tokens": None,
                "text_tokens": None,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            }
            prompt_tokens_details = {
                "cached_tokens": 0,
                "audio_tokens": None,
                "text_tokens": None,
                "image_tokens": None,
            }

            def model_dump(self) -> dict:
                return {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "total_tokens": self.total_tokens,
                    "completion_tokens_details": self.completion_tokens_details,
                    "prompt_tokens_details": self.prompt_tokens_details,
                }

        class MockResponse:
            usage = MockUsage()
            model = "gpt-4o"

        response = MockResponse()
        metadata = build_metadata_from_response(response)

        assert metadata.usage is not None
        assert metadata.usage.prompt_tokens == 100
        assert metadata.usage.completion_tokens == 50
        assert metadata.usage.total_tokens == 150
        assert metadata.usage.completion_tokens_details == {
            "reasoning_tokens": 10,
            "audio_tokens": None,
            "text_tokens": None,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        }
        assert metadata.usage.prompt_tokens_details == {
            "cached_tokens": 0,
            "audio_tokens": None,
            "text_tokens": None,
            "image_tokens": None,
        }


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
        """Test parsing invalid UUID format raises InvalidConversationIDError."""
        invalid_uuid = "not-a-valid-uuid"

        with pytest.raises(
            InvalidConversationIDError,
            match="Invalid conversation ID format",
        ):
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
    async def test__build_llm_messages__continuing_requires_database(self) -> None:
        """Test continuing conversation requires database connection."""
        conv_id = uuid4()
        ctx = ConversationContext(mode=ConversationMode.CONTINUING, conversation_id=conv_id)
        request_messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(ValueError, match="Database connection required"):
            await build_llm_messages(request_messages, ctx, None)

    @pytest.mark.asyncio
    async def test__build_llm_messages__continuing_loads_history(self) -> None:
        """Test continuing conversation loads full history with system message from request."""
        conv_id = uuid4()
        ctx = ConversationContext(mode=ConversationMode.CONTINUING, conversation_id=conv_id)
        request_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Third message"},
        ]

        # Mock conversation with history (no system_message field)
        mock_conversation = Conversation(
            id=conv_id,
            user_id=None,
            title=None,
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
                    metadata={},
                    total_cost=None,
                    created_at="2024-01-01T00:00:00",
                    sequence_number=1,
                ),
                Message(
                    id=uuid4(),
                    conversation_id=conv_id,
                    role="assistant",
                    content="Second message",
                    reasoning_events=None,
                    metadata={},
                    total_cost=None,
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

        # Verify messages include: [system_from_request] + [history] + [new_user_messages]
        assert len(messages) == 4
        assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert messages[1] == {"role": "user", "content": "First message"}
        assert messages[2] == {"role": "assistant", "content": "Second message"}
        assert messages[3] == {"role": "user", "content": "Third message"}

    @pytest.mark.asyncio
    async def test__build_llm_messages__continuing_without_system_message(self) -> None:
        """Test continuing conversation works without system message in request."""
        conv_id = uuid4()
        ctx = ConversationContext(mode=ConversationMode.CONTINUING, conversation_id=conv_id)
        request_messages = [
            {"role": "user", "content": "Hello"},
        ]

        mock_conversation = Conversation(
            id=conv_id,
            user_id=None,
            title=None,
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
            archived_at=None,
            metadata={},
            messages=[],
        )

        mock_db = AsyncMock()
        mock_db.get_conversation.return_value = mock_conversation

        messages = await build_llm_messages(request_messages, ctx, mock_db)

        # Verify no system message when not provided in request
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello"}


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
        assert stored_messages[1] == {
            "role": "assistant",
            "content": "Hi there!",
            "metadata": {},
            "total_cost": None,
            "reasoning_events": None,
        }

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
        assert stored_messages[1] == {
            "role": "assistant",
            "content": "Hi!",
            "metadata": {},
            "total_cost": None,
            "reasoning_events": None,
        }

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
        assert stored_messages[2] == {
            "role": "assistant",
            "content": "Response",
            "metadata": {},
            "total_cost": None,
            "reasoning_events": None,
        }

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
        assert stored_messages[0] == {
            "role": "assistant",
            "content": "Response",
            "metadata": {},
            "total_cost": None,
            "reasoning_events": None,
        }

    @pytest.mark.asyncio
    async def test__store_conversation_messages__stores_metadata(self) -> None:
        """Test storing assistant message with usage and cost metadata."""
        conv_id = uuid4()
        request_messages = [{"role": "user", "content": "Hello"}]
        assistant_content = "Hi there!"
        metadata = ResponseMetadata(
            usage=UsageMetadata(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
            cost=CostMetadata(
                prompt_cost=0.00001,
                completion_cost=0.00002,
                total_cost=0.00003,
            ),
        )

        mock_db = AsyncMock()

        await store_conversation_messages(
            mock_db, conv_id, request_messages, assistant_content, metadata,
        )

        stored_messages = mock_db.append_messages.call_args[0][1]
        assert len(stored_messages) == 2
        assert stored_messages[0] == {"role": "user", "content": "Hello"}
        assert stored_messages[1] == {
            "role": "assistant",
            "content": "Hi there!",
            "metadata": metadata.model_dump(),  # Stored as dict in JSONB
            "total_cost": 0.00003,
            "reasoning_events": None,
        }


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
