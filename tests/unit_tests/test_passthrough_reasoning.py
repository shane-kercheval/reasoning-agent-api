"""Unit tests for PassthroughExecutor reasoning content buffering."""
import pytest
from unittest.mock import Mock, patch

from api.executors.passthrough import PassthroughExecutor
from api.openai_protocol import OpenAIChatRequest, parse_sse
from tests.fixtures.litellm_reasoning_responses import (
    create_anthropic_reasoning_chunks,
    create_openai_reasoning_chunks,
)


@pytest.fixture
def chat_request() -> OpenAIChatRequest:
    """Create a basic chat request for testing."""
    return OpenAIChatRequest(
        model="claude-3-7-sonnet-20250219",
        messages=[{"role": "user", "content": "What is 7 * 8?"}],
        stream=True,
    )


class TestPassthroughReasoningBuffering:
    """Tests for reasoning_content buffering and EXTERNAL_REASONING events."""

    @pytest.mark.asyncio
    async def test__reasoning_content__buffers_and_emits_event(
        self,
        chat_request: OpenAIChatRequest,
    ) -> None:
        """Test that reasoning_content chunks are buffered and converted to event."""
        # Create executor
        executor = PassthroughExecutor()

        # Mock litellm.acompletion to return Anthropic-style chunks with reasoning
        mock_chunks = create_anthropic_reasoning_chunks()

        async def mock_acompletion(*args, **kwargs):  # type: ignore[no-untyped-def]  # noqa: ANN202, ARG001, ANN002, ANN003
            """Mock async generator for litellm streaming."""
            for chunk in mock_chunks:
                yield chunk

        mock_path = "api.executors.passthrough.litellm.acompletion"
        with patch(mock_path, return_value=mock_acompletion()):
            # Collect all SSE chunks from executor
            sse_chunks = []
            async for sse_chunk in executor.execute_stream(chat_request):
                sse_chunks.append(sse_chunk)

        # Parse SSE chunks to JSON (filter out [DONE] marker)
        done_marker = "[DONE]\n\n"
        parsed_chunks = [
            parse_sse(chunk) for chunk in sse_chunks if not chunk.endswith(done_marker)
        ]

        # Should have: 1 reasoning_event chunk + 2 content chunks + 1 finish chunk
        assert len(parsed_chunks) == 4

        # First chunk should have reasoning_event
        first_chunk = parsed_chunks[0]
        assert "choices" in first_chunk
        assert "delta" in first_chunk["choices"][0]
        delta = first_chunk["choices"][0]["delta"]

        assert "reasoning_event" in delta
        reasoning_event = delta["reasoning_event"]

        # Verify reasoning event structure
        assert reasoning_event["type"] == "external_reasoning"
        assert reasoning_event["step_iteration"] == 1
        assert "metadata" in reasoning_event

        # Verify buffered reasoning text is complete
        metadata = reasoning_event["metadata"]
        assert "thought" in metadata
        assert metadata["thought"] == "To solve this problem, I need to multiply."

        # Remaining chunks should be regular content
        assert parsed_chunks[1]["choices"][0]["delta"]["content"] == "The answer"
        assert parsed_chunks[2]["choices"][0]["delta"]["content"] == " is 56."
        assert parsed_chunks[3]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test__no_reasoning_content__works_unchanged(
        self,
    ) -> None:
        """Test that models without reasoning_content work normally (OpenAI)."""
        # Create request for OpenAI model
        request = OpenAIChatRequest(
            model="o3-mini",
            messages=[{"role": "user", "content": "What is 7 * 8?"}],
            stream=True,
        )

        executor = PassthroughExecutor()

        # Mock litellm.acompletion to return OpenAI-style chunks (no reasoning_content)
        mock_chunks = create_openai_reasoning_chunks()

        async def mock_acompletion(*args, **kwargs):  # type: ignore[no-untyped-def]  # noqa: ANN202, ARG001, ANN002, ANN003
            """Mock async generator for litellm streaming."""
            for chunk in mock_chunks:
                yield chunk

        mock_path = "api.executors.passthrough.litellm.acompletion"
        with patch(mock_path, return_value=mock_acompletion()):
            # Collect all SSE chunks
            sse_chunks = []
            async for sse_chunk in executor.execute_stream(request):
                sse_chunks.append(sse_chunk)

        # Parse SSE chunks (filter out [DONE] marker)
        done_marker = "[DONE]\n\n"
        parsed_chunks = [
            parse_sse(chunk) for chunk in sse_chunks if not chunk.endswith(done_marker)
        ]

        # Should just have content chunks (no reasoning_event)
        assert len(parsed_chunks) == 3

        # Verify no reasoning_event in any chunk (should be None)
        for chunk in parsed_chunks:
            if chunk.get("choices"):
                delta = chunk["choices"][0]["delta"]
                assert delta.get("reasoning_event") is None

        # Verify regular content is present
        assert parsed_chunks[0]["choices"][0]["delta"]["content"] == "The answer"
        assert parsed_chunks[1]["choices"][0]["delta"]["content"] == " is 56."

    @pytest.mark.asyncio
    async def test__empty_reasoning_content__ignored(
        self,
        chat_request: OpenAIChatRequest,
    ) -> None:
        """Test that empty reasoning_content strings are ignored."""
        executor = PassthroughExecutor()

        # Create mock chunks with empty reasoning_content
        mock_chunk = Mock()
        mock_chunk.model_dump.return_value = {
            "id": "test",
            "created": 123,
            "model": "claude-3-7",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {
                    "reasoning_content": "",  # Empty string
                    "content": "Hello",
                    "role": "assistant",
                },
                "finish_reason": None,
            }],
        }
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta = Mock()
        mock_chunk.choices[0].delta.reasoning_content = ""
        mock_chunk.choices[0].delta.content = "Hello"
        mock_chunk.choices[0].delta.role = "assistant"

        async def mock_acompletion(*args, **kwargs):  # type: ignore[no-untyped-def]  # noqa: ANN202, ARG001, ANN002, ANN003
            """Mock async generator."""
            yield mock_chunk

        mock_path = "api.executors.passthrough.litellm.acompletion"
        with patch(mock_path, return_value=mock_acompletion()):
            sse_chunks = []
            async for chunk in executor.execute_stream(chat_request):
                sse_chunks.append(chunk)

        # Should just pass through content (no reasoning event)
        done_marker = "[DONE]\n\n"
        parsed = [parse_sse(c) for c in sse_chunks if not c.endswith(done_marker)]
        assert len(parsed) == 1
        assert parsed[0]["choices"][0]["delta"]["content"] == "Hello"
        # reasoning_event should be None (not present or null)
        assert parsed[0]["choices"][0]["delta"].get("reasoning_event") is None

    @pytest.mark.asyncio
    async def test__reasoning_event__only_sent_once(
        self,
        chat_request: OpenAIChatRequest,
    ) -> None:
        """Test that reasoning_event is only emitted once per stream."""
        executor = PassthroughExecutor()

        # Mock chunks with multiple transitions (edge case)
        chunks = []

        # Reasoning chunk
        chunk1 = Mock()
        chunk1.model_dump.return_value = {
            "id": "test",
            "created": 123,
            "model": "claude-3-7",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {"reasoning_content": "Think"},
                "finish_reason": None,
            }],
        }
        chunk1.choices = [Mock()]
        chunk1.choices[0].delta = Mock()
        chunk1.choices[0].delta.reasoning_content = "Think"
        chunk1.choices[0].delta.content = ""
        chunks.append(chunk1)

        # Content chunk 1
        chunk2 = Mock()
        chunk2.model_dump.return_value = {
            "id": "test",
            "created": 123,
            "model": "claude-3-7",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "A"}, "finish_reason": None}],
        }
        chunk2.choices = [Mock()]
        chunk2.choices[0].delta = Mock()
        chunk2.choices[0].delta.content = "A"
        chunks.append(chunk2)

        # Content chunk 2
        chunk3 = Mock()
        chunk3.model_dump.return_value = {
            "id": "test",
            "created": 123,
            "model": "claude-3-7",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "B"}, "finish_reason": None}],
        }
        chunk3.choices = [Mock()]
        chunk3.choices[0].delta = Mock()
        chunk3.choices[0].delta.content = "B"
        chunks.append(chunk3)

        async def mock_acompletion(*args, **kwargs):  # type: ignore[no-untyped-def]  # noqa: ANN202, ARG001, ANN002, ANN003
            """Mock async generator."""
            for c in chunks:
                yield c

        mock_path = "api.executors.passthrough.litellm.acompletion"
        with patch(mock_path, return_value=mock_acompletion()):
            sse_chunks = []
            async for chunk in executor.execute_stream(chat_request):
                sse_chunks.append(chunk)

        # Parse all chunks (filter out [DONE] marker)
        done_marker = "[DONE]\n\n"
        parsed = [parse_sse(c) for c in sse_chunks if not c.endswith(done_marker)]

        # Count reasoning_events (check if reasoning_event is not None)
        reasoning_event_count = sum(
            1 for chunk in parsed
            if "choices" in chunk
            and chunk["choices"]
            and chunk["choices"][0]["delta"].get("reasoning_event") is not None
        )

        # Should only have ONE reasoning_event
        assert reasoning_event_count == 1
