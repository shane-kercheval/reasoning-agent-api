"""Unit tests for cancellation at API and agent interface levels."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from opentelemetry import trace

from api.main import chat_completions
from api.openai_protocol import OpenAIChatRequest
from api.reasoning_agent import ReasoningAgent


class TestCancellationPassthroughPath:
    """
    Test HTTP cancellation in passthrough (direct OpenAI) path.

    STABLE: These tests verify FastAPI/HTTP layer cancellation logic for passthrough routing.
    They will survive the ReasoningAgent → OrchestratorAgent refactor.

    Tests focus on:
    - HTTP request disconnection detection during streaming
    - API response streaming behavior with client disconnect handling
    - Multi-client isolation at API layer
    - OpenTelemetry span handling during cancellation

    Passthrough path checks disconnection INSIDE execute_passthrough_stream.
    """

    @pytest.fixture
    def mock_request(self) -> AsyncMock:
        """Create a mock HTTP request for passthrough routing (no header)."""
        request = AsyncMock(spec=Request)
        request.headers = {"user-agent": "test"}  # No routing header = passthrough
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

    @pytest.fixture
    def any_agent(self) -> AsyncMock:
        """Create a mock agent (not used in passthrough, but required by endpoint)."""
        return AsyncMock(spec=ReasoningAgent)

    @pytest.mark.asyncio
    async def test_disconnection_detection_per_chunk(
        self,
        mock_request: AsyncMock,
        chat_request: OpenAIChatRequest,
        any_agent: AsyncMock,
    ) -> None:
        """
        Test passthrough path disconnection detection before each chunk yield.

        STABLE: Tests FastAPI streaming response behavior, not agent details.
        """
        # Mock passthrough stream that checks disconnection
        async def mock_passthrough_stream(*args, check_disconnected=None, **kwargs):  # noqa
            for i in range(5):
                if check_disconnected and await check_disconnected():
                    raise asyncio.CancelledError("Client disconnected")
                yield f"data: chunk {i}\n\n"

        with patch("api.main.verify_token", return_value=True), \
             patch("api.main.execute_passthrough_stream", new=mock_passthrough_stream):
            response = await chat_completions(
                request=chat_request,
                reasoning_agent=any_agent,
                http_request=mock_request,
                _=True,
            )

            # Consume all chunks inside patch context
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)

        # Verify all chunks were yielded
        assert len(chunks) == 5

    @pytest.mark.asyncio
    async def test_early_disconnection(
        self,
        mock_request: AsyncMock,
        chat_request: OpenAIChatRequest,
        any_agent: AsyncMock,
    ) -> None:
        """
        Test passthrough path handles immediate disconnection.

        STABLE: Tests API endpoint behavior with early client disconnection.
        """
        chunks_generated = []

        async def mock_passthrough_stream(*args, check_disconnected=None, **kwargs):  # noqa
            for i in range(10):
                chunks_generated.append(i)
                if check_disconnected and await check_disconnected():
                    raise asyncio.CancelledError("Client disconnected")
                yield f"data: chunk {i}\n\n"
                await asyncio.sleep(0.01)

        # Disconnect immediately
        mock_request.is_disconnected.return_value = True

        with patch("api.main.verify_token", return_value=True), \
             patch("api.main.execute_passthrough_stream", new=mock_passthrough_stream):
            response = await chat_completions(
                request=chat_request,
                reasoning_agent=any_agent,
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
        any_agent: AsyncMock,
    ) -> None:
        """
        Test passthrough path disconnection detection timing.

        STABLE: Tests HTTP request disconnection detection precision.
        """
        async def mock_passthrough_stream(*args, check_disconnected=None, **kwargs):  # noqa
            for i in range(10):
                if check_disconnected and await check_disconnected():
                    raise asyncio.CancelledError("Client disconnected")
                yield f"data: chunk {i}\n\n"

        # Disconnect after 3 checks
        check_count = 0

        async def mock_is_disconnected():  # noqa: ANN202
            nonlocal check_count
            check_count += 1
            return check_count > 3

        mock_request.is_disconnected = mock_is_disconnected

        with patch("api.main.verify_token", return_value=True), \
             patch("api.main.execute_passthrough_stream", new=mock_passthrough_stream):
            response = await chat_completions(
                request=chat_request,
                reasoning_agent=any_agent,
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
        """
        Test passthrough path isolates concurrent client requests.

        STABLE: Tests FastAPI concurrent request handling, not agent logic.
        """
        # Create separate mock requests
        requests = []
        agents = []

        for i in range(3):
            req = AsyncMock(spec=Request)
            req.headers = {"user-agent": f"client-{i}"}  # No routing header = passthrough
            req.url = MagicMock()
            req.url.__str__.return_value = "http://test/v1/chat/completions"
            req.is_disconnected = AsyncMock(return_value=False)
            requests.append(req)

            # Agent not used in passthrough, but required
            agent = AsyncMock(spec=ReasoningAgent)
            agents.append(agent)

        # Client 1 disconnects early
        disconnect_count = 0

        async def client1_disconnect():  # noqa: ANN202
            nonlocal disconnect_count
            disconnect_count += 1
            return disconnect_count > 2

        requests[1].is_disconnected = client1_disconnect

        # Create mock streams for each client
        async def make_stream_for_client(client_id: int, *args, check_disconnected=None, **kwargs):  # noqa
            for j in range(5):
                if check_disconnected and await check_disconnected():
                    raise asyncio.CancelledError("Client disconnected")
                yield f"data: client{client_id}-chunk{j}\n\n"
                await asyncio.sleep(0.01)

        # Mock passthrough to return different streams
        call_index = 0
        def mock_passthrough_side_effect(*args, **kwargs):  # noqa
            nonlocal call_index
            client_id = call_index
            call_index += 1
            return make_stream_for_client(client_id, *args, **kwargs)

        with patch("api.main.verify_token", return_value=True), \
             patch("api.main.execute_passthrough_stream", side_effect=mock_passthrough_side_effect):  # noqa: E501
            responses = await asyncio.gather(*[
                chat_completions(
                    request=chat_request,
                    reasoning_agent=agents[i],
                    http_request=requests[i],
                    _=True,
                )
                for i in range(3)
            ])

            # Consume all streams inside patch context
            results = []
            for response in responses:
                chunks = []
                async for chunk in response.body_iterator:
                    chunks.append(chunk)
                results.append(chunks)

        # Verify API layer isolation
        assert len(results[0]) == 5  # Client 0: full completion
        assert len(results[1]) <= 3  # Client 1: cancelled early
        assert len(results[2]) == 5  # Client 2: full completion

    @pytest.mark.asyncio
    async def test_span_status_on_cancellation(
        self,
        mock_request: AsyncMock,
        chat_request: OpenAIChatRequest,
        any_agent: AsyncMock,
    ) -> None:
        """
        Test passthrough path OpenTelemetry span handling during cancellation.

        STABLE: Tests observability instrumentation, not agent implementation.
        """
        # Mock span to track status
        mock_span = MagicMock(spec=trace.Span)
        mock_span.is_recording.return_value = True

        async def mock_passthrough_stream(*args, check_disconnected=None, **kwargs):  # noqa
            for i in range(5):
                if check_disconnected and await check_disconnected():
                    raise asyncio.CancelledError("Client disconnected")
                yield f"data: chunk {i}\n\n"
                await asyncio.sleep(0.1)

        # Disconnect after first chunk
        call_count = 0
        async def mock_is_disconnected():  # noqa: ANN202
            nonlocal call_count
            call_count += 1
            return call_count > 1

        mock_request.is_disconnected = mock_is_disconnected

        with patch("api.main.tracer.start_span", return_value=mock_span):
            with patch("api.main.verify_token", return_value=True), \
                 patch("api.main.execute_passthrough_stream", new=mock_passthrough_stream):
                response = await chat_completions(
                    request=chat_request,
                    reasoning_agent=any_agent,
                    http_request=mock_request,
                    _=True,
                )

                # Consume stream inside patch context
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


class TestCancellationReasoningPath:
    """
    Test HTTP cancellation in reasoning agent path.

    STABLE: These tests verify FastAPI/HTTP layer cancellation logic for reasoning routing.
    They will survive the ReasoningAgent → OrchestratorAgent refactor.

    Tests focus on:
    - HTTP request disconnection detection during streaming
    - API response streaming behavior with client disconnect handling
    - Multi-client isolation at API layer
    - OpenTelemetry span handling during cancellation

    Reasoning path checks disconnection in WRAPPER in main.py (before yielding to client).
    """

    @pytest.fixture
    def mock_request(self) -> AsyncMock:
        """Create a mock HTTP request for reasoning routing."""
        request = AsyncMock(spec=Request)
        request.headers = {"user-agent": "test", "x-routing-mode": "reasoning"}
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

    @pytest.fixture
    def reasoning_agent(self) -> AsyncMock:
        """Create a mock reasoning agent."""
        return AsyncMock(spec=ReasoningAgent)

    @pytest.mark.asyncio
    async def test_disconnection_detection_per_chunk(
        self,
        mock_request: AsyncMock,
        chat_request: OpenAIChatRequest,
        reasoning_agent: AsyncMock,
    ) -> None:
        """
        Test reasoning path disconnection detection before each chunk yield.

        STABLE: Tests FastAPI streaming response behavior, not agent details.
        Note: Disconnection check happens in wrapper (main.py), not in agent.
        """
        # Mock agent stream - no disconnection checking needed here
        # (checking happens in main.py wrapper)
        async def mock_agent_stream(*args, **kwargs):  # noqa
            for i in range(5):
                yield f"data: chunk {i}\n\n"

        reasoning_agent.execute_stream = mock_agent_stream

        with patch("api.main.verify_token", return_value=True):
            response = await chat_completions(
                request=chat_request,
                reasoning_agent=reasoning_agent,
                http_request=mock_request,
                _=True,
            )

            # Consume all chunks
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)

        # Verify all chunks were yielded
        assert len(chunks) == 5

    @pytest.mark.asyncio
    async def test_early_disconnection(
        self,
        mock_request: AsyncMock,
        chat_request: OpenAIChatRequest,
        reasoning_agent: AsyncMock,
    ) -> None:
        """
        Test reasoning path handles immediate disconnection.

        STABLE: Tests API endpoint behavior with early client disconnection.
        """
        chunks_generated = []

        async def mock_agent_stream(*args, **kwargs):  # noqa
            for i in range(10):
                chunks_generated.append(i)
                yield f"data: chunk {i}\n\n"
                await asyncio.sleep(0.01)

        reasoning_agent.execute_stream = mock_agent_stream

        # Disconnect immediately
        mock_request.is_disconnected.return_value = True

        with patch("api.main.verify_token", return_value=True):
            response = await chat_completions(
                request=chat_request,
                reasoning_agent=reasoning_agent,
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
        reasoning_agent: AsyncMock,
    ) -> None:
        """
        Test reasoning path disconnection detection timing.

        STABLE: Tests HTTP request disconnection detection precision.
        """
        async def mock_agent_stream(*args, **kwargs):  # noqa
            for i in range(10):
                yield f"data: chunk {i}\n\n"

        reasoning_agent.execute_stream = mock_agent_stream

        # Disconnect after 3 checks
        check_count = 0

        async def mock_is_disconnected():  # noqa: ANN202
            nonlocal check_count
            check_count += 1
            return check_count > 3

        mock_request.is_disconnected = mock_is_disconnected

        with patch("api.main.verify_token", return_value=True):
            response = await chat_completions(
                request=chat_request,
                reasoning_agent=reasoning_agent,
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
        """
        Test reasoning path isolates concurrent client requests.

        STABLE: Tests FastAPI concurrent request handling, not agent logic.
        """
        # Create separate mock requests
        requests = []
        agents = []

        for i in range(3):
            req = AsyncMock(spec=Request)
            req.headers = {"user-agent": f"client-{i}", "x-routing-mode": "reasoning"}
            req.url = MagicMock()
            req.url.__str__.return_value = "http://test/v1/chat/completions"
            req.is_disconnected = AsyncMock(return_value=False)
            requests.append(req)

            agent = AsyncMock(spec=ReasoningAgent)

            # Each agent yields different content
            async def make_stream(client_id=i, *args, **kwargs):  # noqa
                for j in range(5):
                    yield f"data: client{client_id}-chunk{j}\n\n"
                    await asyncio.sleep(0.01)

            # Direct assignment so it works as an async generator function
            agent.execute_stream = make_stream
            agents.append(agent)

        # Client 1 disconnects early
        disconnect_count = 0

        async def client1_disconnect():  # noqa: ANN202
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

        # Verify API layer isolation
        assert len(results[0]) == 5  # Client 0: full completion
        assert len(results[1]) <= 3  # Client 1: cancelled early
        assert len(results[2]) == 5  # Client 2: full completion

    @pytest.mark.asyncio
    async def test_span_status_on_cancellation(
        self,
        mock_request: AsyncMock,
        chat_request: OpenAIChatRequest,
        reasoning_agent: AsyncMock,
    ) -> None:
        """
        Test reasoning path OpenTelemetry span handling during cancellation.

        STABLE: Tests observability instrumentation, not agent implementation.
        """
        # Mock span to track status
        mock_span = MagicMock(spec=trace.Span)
        mock_span.is_recording.return_value = True

        async def mock_agent_stream(*args, **kwargs):  # noqa
            for i in range(5):
                yield f"data: chunk {i}\n\n"
                await asyncio.sleep(0.1)

        reasoning_agent.execute_stream = mock_agent_stream

        # Disconnect after first chunk
        call_count = 0
        async def mock_is_disconnected():  # noqa: ANN202
            nonlocal call_count
            call_count += 1
            return call_count > 1

        mock_request.is_disconnected = mock_is_disconnected

        with patch("api.main.tracer.start_span", return_value=mock_span):
            with patch("api.main.verify_token", return_value=True):
                response = await chat_completions(
                    request=chat_request,
                    reasoning_agent=reasoning_agent,
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
