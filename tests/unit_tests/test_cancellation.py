"""Unit tests for cancellation functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.responses import StreamingResponse

from api.main import chat_completions
from api.openai_protocol import OpenAIChatRequest
from api.reasoning_agent import ReasoningAgent


class TestCancellationUnit:
    """Unit tests for cancellation behavior."""

    @pytest.fixture
    def mock_request(self) -> AsyncMock:
        """Create a mock HTTP request."""
        request = AsyncMock(spec=Request)
        request.headers = {"user-agent": "test"}
        request.url = MagicMock()
        request.url.__str__.return_value = "http://test/v1/chat/completions"
        request.is_disconnected = AsyncMock(return_value=False)
        return request

    @pytest.fixture
    def chat_request(self) -> OpenAIChatRequest:
        """Create a test chat request."""
        return OpenAIChatRequest(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

    @pytest.mark.asyncio
    async def test_is_disconnected_called_per_chunk(
        self,
        mock_request: AsyncMock,
        chat_request: OpenAIChatRequest,
    ) -> None:
        """Test that is_disconnected is called before yielding each chunk."""
        mock_agent = AsyncMock(spec=ReasoningAgent)

        # Create a stream that yields multiple chunks
        async def mock_stream(*args, **kwargs):
            for i in range(5):
                yield f"data: chunk {i}\n\n"

        mock_agent.execute_stream.return_value = mock_stream()

        with patch("api.main.verify_token", return_value=True):
            response = await chat_completions(
                request=chat_request,
                reasoning_agent=mock_agent,
                http_request=mock_request,
                _=True,
            )

        # Consume all chunks
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)

        # Verify is_disconnected was called multiple times
        assert mock_request.is_disconnected.call_count >= 5
        assert len(chunks) == 5

    @pytest.mark.asyncio
    async def test_cancelled_error_propagation(
        self,
        mock_request: AsyncMock,
        chat_request: OpenAIChatRequest,
    ) -> None:
        """Test that CancelledError propagates correctly."""
        mock_agent = AsyncMock(spec=ReasoningAgent)

        # Stream that raises CancelledError
        async def mock_stream(*args, **kwargs):
            yield "data: chunk 1\n\n"
            await asyncio.sleep(0.01)
            raise asyncio.CancelledError("Test cancellation")

        mock_agent.execute_stream.return_value = mock_stream()

        # Never report disconnection
        mock_request.is_disconnected.return_value = False

        with patch("api.main.verify_token", return_value=True):
            response = await chat_completions(
                request=chat_request,
                reasoning_agent=mock_agent,
                http_request=mock_request,
                _=True,
            )

        # The stream should end gracefully even with CancelledError
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)

        # Should get only the first chunk before cancellation
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_non_streaming_not_affected(
        self,
        mock_request: AsyncMock,
    ) -> None:
        """Test that non-streaming requests are not affected by cancellation logic."""
        chat_request = OpenAIChatRequest(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,  # Non-streaming
        )

        mock_agent = AsyncMock(spec=ReasoningAgent)
        mock_agent.execute.return_value = {
            "id": "test",
            "choices": [{"message": {"content": "Response"}}],
        }

        # is_disconnected should never be called for non-streaming
        with patch("api.main.verify_token", return_value=True):
            response = await chat_completions(
                request=chat_request,
                reasoning_agent=mock_agent,
                http_request=mock_request,
                _=True,
            )

        # Verify non-streaming response
        assert not isinstance(response, StreamingResponse)
        assert mock_request.is_disconnected.call_count == 0

    @pytest.mark.asyncio
    async def test_early_disconnection(
        self,
        mock_request: AsyncMock,
        chat_request: OpenAIChatRequest,
    ) -> None:
        """Test disconnection on first chunk check."""
        mock_agent = AsyncMock(spec=ReasoningAgent)

        chunks_generated = []

        async def mock_stream(*args, **kwargs):
            for i in range(10):
                chunks_generated.append(i)
                yield f"data: chunk {i}\n\n"
                await asyncio.sleep(0.01)

        mock_agent.execute_stream.return_value = mock_stream()

        # Disconnect immediately
        mock_request.is_disconnected.return_value = True

        with patch("api.main.verify_token", return_value=True):
            response = await chat_completions(
                request=chat_request,
                reasoning_agent=mock_agent,
                http_request=mock_request,
                _=True,
            )

        # Should get no chunks due to immediate disconnection
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)

        assert len(chunks) == 0
        # Generator might yield 1-2 chunks before detection
        assert len(chunks_generated) <= 2

    @pytest.mark.asyncio
    async def test_disconnection_timing(
        self,
        mock_request: AsyncMock,
        chat_request: OpenAIChatRequest,
    ) -> None:
        """Test that disconnection is detected at the right time."""
        mock_agent = AsyncMock(spec=ReasoningAgent)

        async def mock_stream(*args, **kwargs):
            for i in range(10):
                yield f"data: chunk {i}\n\n"

        mock_agent.execute_stream.return_value = mock_stream()

        # Disconnect after 3 checks
        check_count = 0

        async def mock_is_disconnected():
            nonlocal check_count
            check_count += 1
            return check_count > 3

        mock_request.is_disconnected = mock_is_disconnected

        with patch("api.main.verify_token", return_value=True):
            response = await chat_completions(
                request=chat_request,
                reasoning_agent=mock_agent,
                http_request=mock_request,
                _=True,
            )

        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)

        # Should get approximately 3 chunks
        assert 2 <= len(chunks) <= 4  # Some tolerance for async timing

    @pytest.mark.asyncio
    async def test_concurrent_requests_isolation(
        self,
        chat_request: OpenAIChatRequest,
    ) -> None:
        """Test that concurrent requests don't interfere with each other."""
        # Create separate mock requests
        requests = []
        agents = []

        for i in range(3):
            req = AsyncMock(spec=Request)
            req.headers = {"user-agent": f"client-{i}"}
            req.url = MagicMock()
            req.url.__str__.return_value = "http://test/v1/chat/completions"
            req.is_disconnected = AsyncMock(return_value=False)
            requests.append(req)

            agent = AsyncMock(spec=ReasoningAgent)

            # Each agent yields different content
            async def make_stream(client_id=i):
                for j in range(5):
                    yield f"data: client{client_id}-chunk{j}\n\n"
                    await asyncio.sleep(0.01)

            agent.execute_stream.return_value = make_stream()
            agents.append(agent)

        # Client 1 disconnects early
        disconnect_count = 0

        async def client1_disconnect():
            nonlocal disconnect_count
            disconnect_count += 1
            return disconnect_count > 2

        requests[1].is_disconnected = client1_disconnect

        # Execute all requests concurrently
        with patch("api.main.verify_token", return_value=True):
            responses = await asyncio.gather(*[
                chat_completions(
                    request=chat_request,
                    reasoning_agent=agents[i],
                    http_request=requests[i],
                    _=True,
                )
                for i in range(3)
            ])

        # Consume all streams
        results = []
        for response in responses:
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
            results.append(chunks)

        # Verify isolation
        assert len(results[0]) == 5  # Client 0: full completion
        assert len(results[1]) <= 3  # Client 1: cancelled early
        assert len(results[2]) == 5  # Client 2: full completion

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
