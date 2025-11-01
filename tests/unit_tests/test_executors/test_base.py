"""Unit tests for base executor functionality."""

import asyncio
import pytest
from typing import AsyncGenerator
from unittest.mock import Mock, AsyncMock
from opentelemetry import trace

from api.executors.base import BaseExecutor, extract_content_from_sse_chunk
from api.openai_protocol import OpenAIChatRequest


class TestExtractContentFromSseChunk:
    """Tests for extract_content_from_sse_chunk function."""

    def test__extract_content__valid_chunk(self) -> None:
        """Test extracting content from valid SSE chunk."""
        chunk = 'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'

        result = extract_content_from_sse_chunk(chunk)

        assert result == "Hello"

    def test__extract_content__done_marker(self) -> None:
        """Test extracting content from [DONE] marker returns None."""
        chunk = "data: [DONE]\n\n"

        result = extract_content_from_sse_chunk(chunk)

        assert result is None

    def test__extract_content__malformed_json(self) -> None:
        """Test malformed JSON returns None."""
        chunk = "data: {invalid json}\n\n"

        result = extract_content_from_sse_chunk(chunk)

        assert result is None

    def test__extract_content__missing_choices(self) -> None:
        """Test chunk missing choices field returns None."""
        chunk = 'data: {"id": "123", "model": "gpt-4"}\n\n'

        result = extract_content_from_sse_chunk(chunk)

        assert result is None

    def test__extract_content__missing_delta(self) -> None:
        """Test chunk missing delta field returns None."""
        chunk = 'data: {"choices": [{"index": 0}]}\n\n'

        result = extract_content_from_sse_chunk(chunk)

        assert result is None

    def test__extract_content__missing_content(self) -> None:
        """Test chunk with delta but no content returns None."""
        chunk = 'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n'

        result = extract_content_from_sse_chunk(chunk)

        assert result is None

    def test__extract_content__not_string(self) -> None:
        """Test non-string input returns None."""
        result = extract_content_from_sse_chunk(123)  # type: ignore

        assert result is None

    def test__extract_content__no_data_prefix(self) -> None:
        """Test chunk without 'data: ' prefix returns None."""
        chunk = '{"choices": [{"delta": {"content": "Hello"}}]}\n\n'

        result = extract_content_from_sse_chunk(chunk)

        assert result is None

    def test__extract_content__empty_string(self) -> None:
        """Test empty string returns None."""
        result = extract_content_from_sse_chunk("")

        assert result is None

    def test__extract_content__empty_content(self) -> None:
        """Test extracting empty content string."""
        chunk = 'data: {"choices": [{"delta": {"content": ""}}]}\n\n'

        result = extract_content_from_sse_chunk(chunk)

        # Empty string is still valid content
        assert result == ""

    def test__extract_content__multiline_content(self) -> None:
        """Test extracting content with newlines."""
        chunk = 'data: {"choices": [{"delta": {"content": "Line 1\\nLine 2"}}]}\n\n'

        result = extract_content_from_sse_chunk(chunk)

        assert result == "Line 1\nLine 2"


class ConcreteExecutor(BaseExecutor):
    """Concrete implementation of BaseExecutor for testing."""

    async def execute_stream(
        self,
        request: OpenAIChatRequest,
        parent_span: trace.Span | None = None,
        check_disconnected=None,
    ) -> AsyncGenerator[str, None]:
        """Simple implementation that yields test chunks."""
        self._reset_buffer()
        chunks = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
            'data: {"choices": [{"delta": {"content": " world"}}]}\n\n',
            "data: [DONE]\n\n",
        ]

        for chunk in chunks:
            self._buffer_chunk(chunk)
            await self._check_disconnection(check_disconnected)
            yield chunk


class TestBaseExecutor:
    """Tests for BaseExecutor base class."""

    def test__initialization(self) -> None:
        """Test BaseExecutor initializes with empty buffer."""
        executor = ConcreteExecutor()

        assert executor._content_buffer == []
        assert executor.get_buffered_content() == ""

    def test__reset_buffer(self) -> None:
        """Test _reset_buffer clears content buffer."""
        executor = ConcreteExecutor()
        executor._content_buffer = ["Hello", " world"]

        executor._reset_buffer()

        assert executor._content_buffer == []
        assert executor.get_buffered_content() == ""

    def test__buffer_chunk__valid_content(self) -> None:
        """Test _buffer_chunk adds content to buffer."""
        executor = ConcreteExecutor()
        chunk = 'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'

        executor._buffer_chunk(chunk)

        assert executor._content_buffer == ["Hello"]

    def test__buffer_chunk__no_content(self) -> None:
        """Test _buffer_chunk ignores chunks with no content."""
        executor = ConcreteExecutor()
        chunk = "data: [DONE]\n\n"

        executor._buffer_chunk(chunk)

        assert executor._content_buffer == []

    def test__get_buffered_content(self) -> None:
        """Test get_buffered_content joins buffer."""
        executor = ConcreteExecutor()
        executor._content_buffer = ["Hello", " ", "world"]

        result = executor.get_buffered_content()

        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test__check_disconnection__not_triggered(self) -> None:
        """Test _check_disconnection does nothing when client connected."""
        executor = ConcreteExecutor()
        check_disconnected = AsyncMock(return_value=False)

        # Should not raise
        await executor._check_disconnection(check_disconnected)

    @pytest.mark.asyncio
    async def test__check_disconnection__triggered(self) -> None:
        """Test _check_disconnection raises CancelledError when disconnected."""
        executor = ConcreteExecutor()
        check_disconnected = AsyncMock(return_value=True)

        with pytest.raises(asyncio.CancelledError, match="Client disconnected"):
            await executor._check_disconnection(check_disconnected)

    @pytest.mark.asyncio
    async def test__check_disconnection__none_callback(self) -> None:
        """Test _check_disconnection handles None callback gracefully."""
        executor = ConcreteExecutor()

        # Should not raise with None callback
        await executor._check_disconnection(None)

    @pytest.mark.asyncio
    async def test__execute_stream__buffers_content(self) -> None:
        """Test execute_stream implementation buffers content."""
        executor = ConcreteExecutor()
        request = OpenAIChatRequest(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test"}],
            stream=True,
        )

        chunks = []
        async for chunk in executor.execute_stream(request):
            chunks.append(chunk)

        # Verify chunks were yielded
        assert len(chunks) == 3
        assert 'data: {"choices"' in chunks[0]

        # Verify content was buffered (excluding [DONE])
        buffered = executor.get_buffered_content()
        assert buffered == "Hello world"

    @pytest.mark.asyncio
    async def test__execute_stream__checks_disconnection(self) -> None:
        """Test execute_stream checks for client disconnection."""
        executor = ConcreteExecutor()
        request = OpenAIChatRequest(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test"}],
            stream=True,
        )

        # Mock check that returns True after first chunk
        call_count = 0

        async def check_disconnected():
            nonlocal call_count
            call_count += 1
            return call_count > 1  # Disconnect after first chunk

        with pytest.raises(asyncio.CancelledError):
            async for chunk in executor.execute_stream(request, check_disconnected=check_disconnected):
                pass

        # Should have checked disconnection
        assert call_count >= 2
