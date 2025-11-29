"""Unit tests for base executor functionality."""

import asyncio
from collections.abc import AsyncGenerator, Callable
from unittest.mock import AsyncMock, Mock

import pytest
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace

from reasoning_api.executors.base import BaseExecutor
from reasoning_api.openai_protocol import (
    OpenAIChatRequest,
    OpenAIDelta,
    OpenAIStreamChoice,
    OpenAIStreamResponse,
)
from reasoning_api.reasoning_models import ReasoningEvent, ReasoningEventType


class ConcreteExecutor(BaseExecutor):
    """Concrete implementation of BaseExecutor for testing."""

    def __init__(
        self,
        parent_span: trace.Span | None = None,
        check_disconnected: Callable[[], bool] | None = None,
        test_responses: list[OpenAIStreamResponse] | None = None,
    ):
        """Initialize with optional test responses."""
        super().__init__(parent_span, check_disconnected)
        self.test_responses = test_responses or []

    async def _execute_stream(
        self,
        request: OpenAIChatRequest,  # noqa: ARG002
    ) -> AsyncGenerator[OpenAIStreamResponse]:
        """Simple implementation that yields test responses."""
        for response in self.test_responses:
            yield response

    def _set_span_attributes(
        self,
        request: OpenAIChatRequest,
        span: trace.Span,
    ) -> None:
        """Test implementation that sets test attributes."""
        span.set_attribute("test.model", request.model)


class TestBaseExecutor:
    """Tests for BaseExecutor base class."""

    def test__initialization(self) -> None:
        """Test BaseExecutor initializes with empty buffer and not executed."""
        executor = ConcreteExecutor()

        assert executor._content_buffer == []
        assert executor.get_buffered_content() == ""
        assert executor._executed is False

    def test__initialization_with_params(self) -> None:
        """Test BaseExecutor initializes with parent_span and check_disconnected."""
        mock_span = Mock(spec=trace.Span)
        check_disconnected = AsyncMock()
        executor = ConcreteExecutor(
            parent_span=mock_span,
            check_disconnected=check_disconnected,
        )

        assert executor._parent_span == mock_span
        assert executor._check_disconnected_callback == check_disconnected

    def test__get_buffered_content(self) -> None:
        """Test get_buffered_content joins buffer."""
        executor = ConcreteExecutor()
        executor._content_buffer = ["Hello", " ", "world"]

        result = executor.get_buffered_content()

        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test__execute_stream__single_use_enforcement(self) -> None:
        """Test executor can only be used once."""
        executor = ConcreteExecutor()
        request = OpenAIChatRequest(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test"}],
            stream=True,
        )

        # First call should work
        chunks = []
        async for chunk in executor.execute_stream(request):
            chunks.append(chunk)

        # Second call should raise RuntimeError
        with pytest.raises(RuntimeError, match="Executor can only be used once"):
            async for chunk in executor.execute_stream(request):
                pass

    @pytest.mark.asyncio
    async def test__execute_stream__buffers_content(self) -> None:
        """Test execute_stream buffers content from responses."""
        test_responses = [
            OpenAIStreamResponse(
                id="test-1",
                created=1234567890,
                model="gpt-4o-mini",
                choices=[
                    OpenAIStreamChoice(
                        index=0,
                        delta=OpenAIDelta(content="Hello"),
                        finish_reason=None,
                    ),
                ],
            ),
            OpenAIStreamResponse(
                id="test-1",
                created=1234567890,
                model="gpt-4o-mini",
                choices=[
                    OpenAIStreamChoice(
                        index=0,
                        delta=OpenAIDelta(content=" world"),
                        finish_reason=None,
                    ),
                ],
            ),
        ]
        executor = ConcreteExecutor(test_responses=test_responses)
        request = OpenAIChatRequest(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test"}],
            stream=True,
        )

        chunks = []
        async for chunk in executor.execute_stream(request):
            chunks.append(chunk)

        # Verify chunks were yielded (SSE formatted)
        assert len(chunks) == 3  # 2 content chunks + [DONE]
        assert 'data: {' in chunks[0]
        assert "[DONE]" in chunks[2]

        # Verify content was buffered
        buffered = executor.get_buffered_content()
        assert buffered == "Hello world"

    @pytest.mark.asyncio
    async def test__execute_stream__skips_reasoning_events(self) -> None:
        """Test execute_stream does not buffer reasoning events."""
        test_responses = [
            OpenAIStreamResponse(
                id="test-1",
                created=1234567890,
                model="gpt-4o-mini",
                choices=[
                    OpenAIStreamChoice(
                        index=0,
                        delta=OpenAIDelta(
                            reasoning_event=ReasoningEvent(
                                type=ReasoningEventType.PLANNING,
                                step_iteration=1,
                                metadata={"thought": "thinking..."},
                            ),
                        ),
                        finish_reason=None,
                    ),
                ],
            ),
            OpenAIStreamResponse(
                id="test-1",
                created=1234567890,
                model="gpt-4o-mini",
                choices=[
                    OpenAIStreamChoice(
                        index=0,
                        delta=OpenAIDelta(content="Final answer"),
                        finish_reason=None,
                    ),
                ],
            ),
        ]
        executor = ConcreteExecutor(test_responses=test_responses)
        request = OpenAIChatRequest(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test"}],
            stream=True,
        )

        chunks = []
        async for chunk in executor.execute_stream(request):
            chunks.append(chunk)

        # Verify both chunks were yielded
        assert len(chunks) == 3  # 2 responses + [DONE]

        # Verify only final answer was buffered (not reasoning event)
        buffered = executor.get_buffered_content()
        assert buffered == "Final answer"

    @pytest.mark.asyncio
    async def test__execute_stream__checks_disconnection(self) -> None:
        """Test execute_stream checks for client disconnection."""
        test_responses = [
            OpenAIStreamResponse(
                id="test-1",
                created=1234567890,
                model="gpt-4o-mini",
                choices=[
                    OpenAIStreamChoice(
                        index=0,
                        delta=OpenAIDelta(content="Hello"),
                        finish_reason=None,
                    ),
                ],
            ),
            OpenAIStreamResponse(
                id="test-1",
                created=1234567890,
                model="gpt-4o-mini",
                choices=[
                    OpenAIStreamChoice(
                        index=0,
                        delta=OpenAIDelta(content=" world"),
                        finish_reason=None,
                    ),
                ],
            ),
        ]

        # Mock check that returns True after first chunk
        call_count = 0

        async def check_disconnected() -> bool:
            nonlocal call_count
            call_count += 1
            return call_count > 1  # Disconnect after first chunk

        executor = ConcreteExecutor(
            check_disconnected=check_disconnected,
            test_responses=test_responses,
        )
        request = OpenAIChatRequest(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test"}],
            stream=True,
        )

        with pytest.raises(asyncio.CancelledError, match="Client disconnected"):
            async for chunk in executor.execute_stream(request):
                pass

        # Should have checked disconnection
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test__execute_stream__sets_span_attributes(self) -> None:
        """Test execute_stream sets span attributes via _set_span_attributes."""
        mock_span = Mock(spec=trace.Span)
        executor = ConcreteExecutor(parent_span=mock_span)
        request = OpenAIChatRequest(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test"}],
            stream=True,
        )

        chunks = []
        async for chunk in executor.execute_stream(request):
            chunks.append(chunk)

        # Verify _set_span_attributes was called
        mock_span.set_attribute.assert_called()
        # Check that test.model was set (from our ConcreteExecutor implementation)
        calls = [call[0] for call in mock_span.set_attribute.call_args_list]
        assert any("test.model" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test__execute_stream__sets_output_attribute(self) -> None:
        """Test execute_stream sets output attribute on parent span after streaming."""
        test_responses = [
            OpenAIStreamResponse(
                id="test-1",
                created=1234567890,
                model="gpt-4o-mini",
                choices=[
                    OpenAIStreamChoice(
                        index=0,
                        delta=OpenAIDelta(content="Hello world"),
                        finish_reason=None,
                    ),
                ],
            ),
        ]

        mock_span = Mock(spec=trace.Span)
        executor = ConcreteExecutor(
            parent_span=mock_span,
            test_responses=test_responses,
        )
        request = OpenAIChatRequest(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test"}],
            stream=True,
        )

        chunks = []
        async for chunk in executor.execute_stream(request):
            chunks.append(chunk)

        # Verify output attribute was set
        mock_span.set_attribute.assert_any_call(
            SpanAttributes.OUTPUT_VALUE,
            "Hello world",
        )

    @pytest.mark.asyncio
    async def test__execute_stream__no_output_if_no_content(self) -> None:
        """Test execute_stream doesn't set output attribute if no content."""
        # Response with no content
        test_responses = [
            OpenAIStreamResponse(
                id="test-1",
                created=1234567890,
                model="gpt-4o-mini",
                choices=[
                    OpenAIStreamChoice(
                        index=0,
                        delta=OpenAIDelta(role="assistant"),  # No content
                        finish_reason=None,
                    ),
                ],
            ),
        ]

        mock_span = Mock(spec=trace.Span)
        executor = ConcreteExecutor(
            parent_span=mock_span,
            test_responses=test_responses,
        )
        request = OpenAIChatRequest(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test"}],
            stream=True,
        )

        chunks = []
        async for chunk in executor.execute_stream(request):
            chunks.append(chunk)

        # Verify output attribute was NOT set (no content to buffer)
        output_calls = [
            call for call in mock_span.set_attribute.call_args_list
            if call[0][0] == SpanAttributes.OUTPUT_VALUE
        ]
        assert len(output_calls) == 0
