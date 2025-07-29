"""Integration tests for cancellation functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from opentelemetry import trace

from api.main import chat_completions
from api.openai_protocol import OpenAIChatRequest
from api.reasoning_agent import ReasoningAgent


class TestCancellation:
    """Test cancellation behavior in the reasoning agent API."""

    @pytest.fixture
    def mock_request(self) -> AsyncMock:
        """Create a mock HTTP request object."""
        request = AsyncMock(spec=Request)
        request.headers = {"user-agent": "test-client"}
        request.url = MagicMock()
        request.url.__str__.return_value = "http://test/v1/chat/completions"
        request.is_disconnected = AsyncMock(return_value=False)
        return request

    @pytest.fixture
    def chat_request(self) -> OpenAIChatRequest:
        """Create a sample chat request."""
        return OpenAIChatRequest(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Tell me a long story about space exploration"},
            ],
            stream=True,
            temperature=0.7,
        )

    @pytest.mark.asyncio
    async def test_client_disconnection_cancels_stream(
        self,
        mock_request: AsyncMock,
        chat_request: OpenAIChatRequest,
    ) -> None:
        """Test that client disconnection properly cancels the streaming response."""
        # Setup mock reasoning agent that yields chunks slowly
        mock_agent = AsyncMock(spec=ReasoningAgent)
        chunks_yielded = 0

        async def mock_stream(*args, **kwargs):
            nonlocal chunks_yielded
            for i in range(10):
                chunks_yielded += 1
                yield f"data: chunk {i}\n\n"
                await asyncio.sleep(0.1)  # Simulate slow generation

        mock_agent.execute_stream.return_value = mock_stream()

        # Simulate disconnection after 3 chunks
        disconnect_after = 3
        call_count = 0

        async def mock_is_disconnected():
            nonlocal call_count
            call_count += 1
            return call_count > disconnect_after

        mock_request.is_disconnected = mock_is_disconnected

        # Execute the endpoint
        with patch("api.main.verify_token", return_value=True):
            response = await chat_completions(
                request=chat_request,
                reasoning_agent=mock_agent,
                http_request=mock_request,
                _=True,
            )

        # Consume the streaming response
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)

        # Verify cancellation happened early
        assert len(chunks) <= disconnect_after + 1  # +1 for potential in-flight chunk
        assert chunks_yielded <= disconnect_after + 2  # Some tolerance for timing

    @pytest.mark.asyncio
    async def test_multiple_clients_independent_cancellation(
        self,
        mock_request: AsyncMock,
        chat_request: OpenAIChatRequest,
    ) -> None:
        """Test that cancelling one client doesn't affect others."""
        # Create two independent mock requests
        request_a = AsyncMock(spec=Request)
        request_a.headers = {"user-agent": "client-a"}
        request_a.url = MagicMock()
        request_a.url.__str__.return_value = "http://test/v1/chat/completions"
        request_a.is_disconnected = AsyncMock(return_value=False)

        request_b = AsyncMock(spec=Request)
        request_b.headers = {"user-agent": "client-b"}
        request_b.url = MagicMock()
        request_b.url.__str__.return_value = "http://test/v1/chat/completions"
        request_b.is_disconnected = AsyncMock(return_value=False)

        # Setup mock agents
        mock_agent_a = AsyncMock(spec=ReasoningAgent)
        mock_agent_b = AsyncMock(spec=ReasoningAgent)

        chunks_a = []
        chunks_b = []

        async def stream_a(*args, **kwargs):
            for i in range(10):
                chunks_a.append(i)
                yield f"data: A{i}\n\n"
                await asyncio.sleep(0.05)

        async def stream_b(*args, **kwargs):
            for i in range(10):
                chunks_b.append(i)
                yield f"data: B{i}\n\n"
                await asyncio.sleep(0.05)

        mock_agent_a.execute_stream.return_value = stream_a()
        mock_agent_b.execute_stream.return_value = stream_b()

        # Client A disconnects after 3 chunks
        a_disconnect_count = 0
        async def a_is_disconnected():
            nonlocal a_disconnect_count
            a_disconnect_count += 1
            return a_disconnect_count > 3

        request_a.is_disconnected = a_is_disconnected

        # Execute both requests concurrently
        with patch("api.main.verify_token", return_value=True):
            response_a_coro = chat_completions(
                request=chat_request,
                reasoning_agent=mock_agent_a,
                http_request=request_a,
                _=True,
            )
            response_b_coro = chat_completions(
                request=chat_request,
                reasoning_agent=mock_agent_b,
                http_request=request_b,
                _=True,
            )

            # Start both requests
            response_a, response_b = await asyncio.gather(
                response_a_coro,
                response_b_coro,
            )

        # Consume both streams concurrently
        async def consume_stream(response):
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
            return chunks

        results_a, results_b = await asyncio.gather(
            consume_stream(response_a),
            consume_stream(response_b),
            return_exceptions=True,
        )

        # Verify A was cancelled early but B completed
        assert len(chunks_a) <= 5  # Should stop early
        assert len(chunks_b) == 10  # Should complete all chunks

    @pytest.mark.asyncio
    async def test_cancellation_between_chunks(
        self,
        mock_request: AsyncMock,
        chat_request: OpenAIChatRequest,
    ) -> None:
        """Test cancellation detection between chunks."""
        mock_agent = AsyncMock(spec=ReasoningAgent)

        chunks_yielded = 0

        async def mock_stream(*args, **kwargs):
            nonlocal chunks_yielded

            # Yield multiple chunks
            for i in range(5):
                chunks_yielded += 1
                yield f'data: chunk {i}\n\n'
                # Small delay between chunks to allow cancellation detection
                await asyncio.sleep(0.01)

        mock_agent.execute_stream.return_value = mock_stream()

        # Disconnect after 2 chunks
        call_count = 0
        async def mock_is_disconnected():
            nonlocal call_count
            call_count += 1
            # Allow first 2 chunks, then disconnect
            return call_count > 2

        mock_request.is_disconnected = mock_is_disconnected

        # Execute
        with patch("api.main.verify_token", return_value=True):
            response = await chat_completions(
                request=chat_request,
                reasoning_agent=mock_agent,
                http_request=mock_request,
                _=True,
            )

        # Consume stream
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)

        # Verify cancellation happened after first few chunks
        assert len(chunks) <= 3  # Should stop after disconnect detection
        assert chunks_yielded <= 4  # Generator might yield 1-2 more before stopping

    @pytest.mark.asyncio
    async def test_span_status_on_cancellation(
        self,
        mock_request: AsyncMock,
        chat_request: OpenAIChatRequest,
    ) -> None:
        """Test that span status is set correctly on cancellation."""
        # Mock span to track status
        mock_span = MagicMock(spec=trace.Span)
        mock_span.is_recording.return_value = True

        with patch("api.main.tracer.start_span", return_value=mock_span):
            mock_agent = AsyncMock(spec=ReasoningAgent)

            async def mock_stream(*args, **kwargs):
                for i in range(5):
                    yield f"data: chunk {i}\n\n"
                    await asyncio.sleep(0.1)

            mock_agent.execute_stream.return_value = mock_stream()

            # Disconnect after first chunk
            call_count = 0
            async def mock_is_disconnected():
                nonlocal call_count
                call_count += 1
                return call_count > 1

            mock_request.is_disconnected = mock_is_disconnected

            # Execute
            with patch("api.main.verify_token", return_value=True):
                response = await chat_completions(
                    request=chat_request,
                    reasoning_agent=mock_agent,
                    http_request=mock_request,
                    _=True,
                )

            # Consume stream
            async for _ in response.body_iterator:
                pass

            # Verify span was ended with correct status
            mock_span.set_status.assert_called()
            status_calls = mock_span.set_status.call_args_list

            # Check that span was set with OK status
            found_ok_status = False
            for call in status_calls:
                status = call[0][0]
                if hasattr(status, 'status_code') and status.status_code == trace.StatusCode.OK:
                    found_ok_status = True
                    break

            assert found_ok_status, "Span should have OK status"

            # Check for cancellation attributes
            mock_span.set_attribute.assert_any_call("http.cancelled", True)
            mock_span.set_attribute.assert_any_call("cancellation.reason", "Client disconnected")


# Note: Integration tests with real OpenAI would require actual API key and fixtures
# These are commented out but show how they would be structured

# @pytest.mark.integration
# class TestCancellationWithRealOpenAI:
#     """Integration tests with real OpenAI API calls."""
#
#     @pytest.mark.asyncio
#     async def test_cancellation_during_real_openai_call(
#         self,
#         real_reasoning_agent: ReasoningAgent,
#     ) -> None:
#         """Test that cancellation interrupts actual OpenAI API calls."""
#         # Would test with real OpenAI API
#         pass
