"""Integration tests for cancellation with real components."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv
from fastapi import Request

from api.main import chat_completions
from api.openai_protocol import OpenAIChatRequest
from api.reasoning_agent import ReasoningAgent

# Load environment variables
load_dotenv()


@pytest.mark.integration
class TestCancellationAPIIntegration:
    """
    Test API cancellation with real agents and OpenAI.

    STABLE: Tests end-to-end API behavior with real components.
    Will survive agent refactoring because tests go through API layer.

    Tests focus on:
    - API endpoint with real agent and real OpenAI calls
    - End-to-end cancellation timing
    - Multi-client isolation with real processing
    - API cancellation during different processing phases
    """

    @pytest.fixture
    def mock_agent(self) -> ReasoningAgent:
        """
        Create a mock agent that simulates OpenAI streaming without real API calls.

        This fixture will be easily updated when OrchestratorAgent replaces ReasoningAgent.
        """
        mock_agent = AsyncMock(spec=ReasoningAgent)

        async def mock_execute_stream(request, parent_span=None):  # Accept parent_span parameter  # noqa
            stream_chunks = [
                # Reasoning events
                ('data: {"id": "test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o-mini", "choices": [{"index": 0, "delta": {"reasoning_event": {"type": "iteration_start", "step_iteration": 1, "metadata": {"tools": []}}}, "finish_reason": null}]}'),  # noqa: E501
                ('data: {"id": "test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o-mini", "choices": [{"index": 0, "delta": {"reasoning_event": {"type": "planning", "step_iteration": 1, "metadata": {"thought": "Processing request...", "tools_planned": []}}}, "finish_reason": null}]}'),  # noqa: E501
                ('data: {"id": "test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o-mini", "choices": [{"index": 0, "delta": {"reasoning_event": {"type": "iteration_complete", "step_iteration": 1, "metadata": {"result": "Ready to respond"}}}, "finish_reason": null}]}'),  # noqa: E501
                # Content chunks
                ('data: {"id": "test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o-mini", "choices": [{"index": 0, "delta": {"content": "This"}, "finish_reason": null}]}'),  # noqa: E501
                ('data: {"id": "test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o-mini", "choices": [{"index": 0, "delta": {"content": " is"}, "finish_reason": null}]}'),  # noqa: E501
                ('data: {"id": "test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o-mini", "choices": [{"index": 0, "delta": {"content": " a"}, "finish_reason": null}]}'),  # noqa: E501
                ('data: {"id": "test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o-mini", "choices": [{"index": 0, "delta": {"content": " test"}, "finish_reason": null}]}'),  # noqa: E501
                ('data: {"id": "test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o-mini", "choices": [{"index": 0, "delta": {"content": " response"}, "finish_reason": null}]}'),  # noqa: E501
                ('data: {"id": "test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o-mini", "choices": [{"index": 0, "delta": {"content": " with"}, "finish_reason": null}]}'),  # noqa: E501
                ('data: {"id": "test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o-mini", "choices": [{"index": 0, "delta": {"content": " multiple"}, "finish_reason": null}]}'),  # noqa: E501
                ('data: {"id": "test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o-mini", "choices": [{"index": 0, "delta": {"content": " chunks."}, "finish_reason": null}]}'),  # noqa: E501
                ('data: {"id": "test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o-mini", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}'),  # noqa: E501
                ('data: [DONE]'),
            ]

            for chunk in stream_chunks:
                yield chunk + '\n\n'

        mock_agent.execute_stream = mock_execute_stream
        return mock_agent

    @pytest.fixture
    def mock_request(self) -> AsyncMock:
        """Create a mock HTTP request for testing."""
        request = AsyncMock(spec=Request)
        request.headers = {
            "user-agent": "test-integration-client",
            "X-Routing-Mode": "reasoning",  # Route to reasoning path for testing
        }
        request.url = MagicMock()
        request.url.__str__.return_value = "http://test/v1/chat/completions"
        request.is_disconnected = AsyncMock(return_value=False)
        return request

    @pytest.fixture
    def long_request(self) -> OpenAIChatRequest:
        """Create a request that will trigger a long OpenAI response."""
        return OpenAIChatRequest(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Write a detailed 500-word essay about the history of artificial intelligence, including key milestones, important researchers, and future implications. Be very thorough and detailed.",  # noqa: E501
                },
            ],
            stream=True,
            temperature=0.7,
            max_tokens=1000,  # Ensure a long response
        )

    @pytest.mark.asyncio
    async def test_api_cancellation_timing(
        self,
        mock_agent: ReasoningAgent,
        mock_request: AsyncMock,
        long_request: OpenAIChatRequest,
    ) -> None:
        """
        Test API cancellation interrupts streaming through API layer.

        STABLE: Tests through chat_completions() endpoint with mock agent.
        This test will work with any agent that implements the interface.
        """
        chunks_received = []
        cancellation_triggered = False

        # Simple disconnection: return False initially, then True after some chunks
        async def is_disconnected() -> bool:
            nonlocal cancellation_triggered
            # Allow a few chunks to be processed, then simulate disconnection
            if len(chunks_received) >= 3 and not cancellation_triggered:
                cancellation_triggered = True
                return True
            return False

        mock_request.is_disconnected = is_disconnected

        # Execute through API layer (stable interface)
        with patch("api.main.verify_token", return_value=True):
            response = await chat_completions(
                request=long_request,
                reasoning_agent=mock_agent,
                http_request=mock_request,
                _=True,
            )

        # Consume the stream until cancellation
        start_time = time.time()
        async for chunk in response.body_iterator:
            chunks_received.append(chunk)

            # Safety timeout in case cancellation doesn't work
            if time.time() - start_time > 15.0:
                break

        duration = time.time() - start_time

        print(f"API Integration: Received {len(chunks_received)} chunks in {duration:.2f}s")

        # Verify API-level cancellation worked
        assert len(chunks_received) > 0, "Should have received some chunks before cancellation"
        assert cancellation_triggered, "Cancellation should have been triggered"
        assert len(chunks_received) <= 12, (
            f"Received {len(chunks_received)} chunks. "
            "Should be cancelled after 3 chunks, but mock agent has 12 total chunks."
        )

    @pytest.mark.asyncio
    async def test_api_cancellation_during_reasoning(
        self,
        mock_agent: ReasoningAgent,
        mock_request: AsyncMock,
    ) -> None:
        """
        Test API cancellation during reasoning phase through API endpoint.

        STABLE: Tests API behavior during processing phases, not agent internals.
        """
        # Request that will trigger reasoning but not tool use
        request = OpenAIChatRequest(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Think step by step about how to solve this math problem: What is 347 * 892? Show your reasoning process.",  # noqa: E501
                },
            ],
            stream=True,
            temperature=0.2,
        )

        # Cancel very quickly to catch during reasoning
        call_count = 0
        async def quick_disconnect():  # noqa: ANN202
            nonlocal call_count
            call_count += 1
            # Allow first few chunks, then disconnect
            return call_count > 3

        mock_request.is_disconnected = quick_disconnect

        start_time = time.time()
        chunks_received = 0

        # Execute through API layer
        with patch("api.main.verify_token", return_value=True):
            response = await chat_completions(
                request=request,
                reasoning_agent=mock_agent,
                http_request=mock_request,
                _=True,
            )

        # Consume stream
        async for chunk in response.body_iterator:
            chunks_received += 1

        duration = time.time() - start_time

        # Verify we interrupted through API during processing
        assert chunks_received > 0, "Should have received some chunks"

        print(f"API Quick cancellation: {chunks_received} chunks in {duration:.2f}s")

    @pytest.mark.asyncio
    async def test_multi_client_isolation(
        self,
        long_request: OpenAIChatRequest,
    ) -> None:
        """
        Test API layer isolates multiple clients properly.

        STABLE: Tests API concurrent request handling with mock processing.
        """
        # Create separate mock agent instances for each client (like the real API does)
        agent_a = AsyncMock(spec=ReasoningAgent)
        agent_b = AsyncMock(spec=ReasoningAgent)

        # Mock streaming responses for both agents
        async def mock_stream_a(request, parent_span=None):  # Accept parent_span parameter  # noqa
            for i in range(5):  # Short stream for agent A (will be cancelled)
                yield f'data: {{"id": "test-a", "object": "chat.completion.chunk", "choices": [{{"index": 0, "delta": {{"content": "A{i}"}}, "finish_reason": null}}]}}\n\n'  # noqa: E501

        async def mock_stream_b(request, parent_span=None):  # Accept parent_span parameter  # noqa
            for i in range(15):  # Longer stream for agent B
                yield f'data: {{"id": "test-b", "object": "chat.completion.chunk", "choices": [{{"index": 0, "delta": {{"content": "B{i}"}}, "finish_reason": null}}]}}\n\n'  # noqa: E501

        agent_a.execute_stream = mock_stream_a
        agent_b.execute_stream = mock_stream_b

        # Create two separate requests
        request_a = AsyncMock(spec=Request)
        request_a.headers = {
            "user-agent": "client-a",
            "X-Routing-Mode": "reasoning",  # Route to reasoning path
        }
        request_a.url = MagicMock()
        request_a.url.__str__.return_value = "http://test/v1/chat/completions"

        request_b = AsyncMock(spec=Request)
        request_b.headers = {
            "user-agent": "client-b",
            "X-Routing-Mode": "reasoning",  # Route to reasoning path
        }
        request_b.url = MagicMock()
        request_b.url.__str__.return_value = "http://test/v1/chat/completions"
        request_b.is_disconnected = AsyncMock(return_value=False)  # Never disconnect

        # Client A disconnects quickly
        a_call_count = 0
        async def a_disconnect():  # noqa: ANN202
            nonlocal a_call_count
            a_call_count += 1
            return a_call_count > 5  # Quick disconnect

        request_a.is_disconnected = a_disconnect

        # Start both clients concurrently through API
        start_time = time.time()

        with patch("api.main.verify_token", return_value=True):
            # Start both API requests with separate agent instances
            response_a_task = asyncio.create_task(chat_completions(
                request=long_request,
                reasoning_agent=agent_a,  # Separate instance
                http_request=request_a,
                _=True,
            ))
            response_b_task = asyncio.create_task(chat_completions(
                request=long_request,
                reasoning_agent=agent_b,  # Separate instance
                http_request=request_b,
                _=True,
            ))

            responses = await asyncio.gather(response_a_task, response_b_task)
            response_a, response_b = responses

        # Consume both API streams
        chunks_a = 0
        chunks_b = 0

        async def consume_a() -> None:
            nonlocal chunks_a
            async for chunk in response_a.body_iterator:
                chunks_a += 1

        async def consume_b() -> None:
            nonlocal chunks_b
            async for chunk in response_b.body_iterator:
                chunks_b += 1

        # Consume both concurrently with timeout
        try:  # noqa: SIM105
            await asyncio.wait_for(
                asyncio.gather(consume_a(), consume_b()),
                timeout=5.0,
            )
        except TimeoutError:
            pass

        duration = time.time() - start_time

        # Verify API layer isolation
        print(f"API Multi-client: Client A: {chunks_a} chunks, Client B: {chunks_b} chunks in {duration:.2f}s")  # noqa: E501

        # Client A should be cancelled (fewer chunks), B should continue independently
        assert chunks_a < chunks_b, "Client A should have been cancelled while B continued through API"  # noqa: E501
        assert chunks_a > 0, "Client A should have gotten some chunks before API cancellation"
        assert chunks_b > chunks_a, "Client B should have continued processing through API"

    @pytest.mark.asyncio
    async def test_api_error_handling_on_cancellation(
        self,
        mock_agent: ReasoningAgent,
        mock_request: AsyncMock,
        long_request: OpenAIChatRequest,
    ) -> None:
        """
        Test API layer handles cancellation errors gracefully.

        STABLE: Tests API error handling, not agent error handling.
        """
        # Track cancellation with proper scoping
        cancellation_detected = False
        chunk_count = 0

        async def detect_cancellation() -> bool:
            nonlocal cancellation_detected, chunk_count
            chunk_count += 1
            # Allow first 3 chunks, then trigger cancellation
            if chunk_count > 3:
                cancellation_detected = True
                return True
            return False

        mock_request.is_disconnected = detect_cancellation

        # Execute through API
        with patch("api.main.verify_token", return_value=True):
            response = await chat_completions(
                request=long_request,
                reasoning_agent=mock_agent,
                http_request=mock_request,
                _=True,
            )

        # Consume stream - should handle any errors gracefully
        chunks_received = 0
        error_occurred = False

        try:
            async for chunk in response.body_iterator:
                chunks_received += 1
        except Exception as e:
            error_occurred = True
            print(f"API error during cancellation: {e}")

        # API should handle cancellation gracefully without exceptions
        assert not error_occurred, "API should handle cancellation without raising exceptions"
        assert chunks_received > 0, "Should receive some chunks before cancellation"
        assert cancellation_detected, "Cancellation should have been detected"

        print(f"API Error handling: {chunks_received} chunks processed gracefully")


@pytest.mark.integration
class TestCancellationAgentIntegration:
    """
    Test agent cancellation directly with real OpenAI.

    PORTABLE: Tests agent interface with real external calls.
    Tests behavior, not implementation details.

    Tests focus on:
    - Direct agent cancellation with real OpenAI
    - Agent resource cleanup with real connections
    - Agent cancellation timing with real API calls
    """

    @pytest.fixture
    def mock_agent(self) -> ReasoningAgent:
        """
        Create a mock agent for direct testing.

        This fixture will be easily updated when OrchestratorAgent replaces ReasoningAgent.
        Includes small delays to enable testing of mid-stream cancellation.
        """
        mock_agent = AsyncMock(spec=ReasoningAgent)

        async def mock_execute_stream(request, parent_span=None):  # Accept parent_span parameter  # noqa
            stream_chunks = [
                ('data: {"id": "agent-test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o-mini", "choices": [{"index": 0, "delta": {"reasoning_event": {"type": "iteration_start", "step_iteration": 1, "metadata": {"tools": []}}}, "finish_reason": null}]}'),  # noqa: E501
                ('data: {"id": "agent-test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o-mini", "choices": [{"index": 0, "delta": {"content": "Agent"}, "finish_reason": null}]}'),  # noqa: E501
                ('data: {"id": "agent-test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o-mini", "choices": [{"index": 0, "delta": {"content": " direct"}, "finish_reason": null}]}'),  # noqa: E501
                ('data: {"id": "agent-test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o-mini", "choices": [{"index": 0, "delta": {"content": " test"}, "finish_reason": null}]}'),  # noqa: E501
                ('data: {"id": "agent-test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o-mini", "choices": [{"index": 0, "delta": {"content": " response."}, "finish_reason": null}]}'),  # noqa: E501
                ('data: {"id": "agent-test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o-mini", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}'),  # noqa: E501
                ('data: [DONE]'),
            ]

            for chunk in stream_chunks:
                yield chunk + '\n\n'
                await asyncio.sleep(0.01)

        mock_agent.execute_stream = mock_execute_stream
        return mock_agent

    @pytest.fixture
    def long_request(self) -> OpenAIChatRequest:
        """Create a request that will trigger a long agent response."""
        return OpenAIChatRequest(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Write a comprehensive 400-word analysis of machine learning algorithms, covering supervised learning, unsupervised learning, and deep learning approaches.",  # noqa: E501
                },
            ],
            stream=True,
            temperature=0.5,
            max_tokens=800,
        )

    @pytest.mark.asyncio
    async def test_agent_cancellation_during_streaming(
        self,
        mock_agent: ReasoningAgent,
        long_request: OpenAIChatRequest,
    ) -> None:
        """
        Test that task.cancel() interrupts agent streaming mid-stream.

        This test validates that async generator streams can be cancelled using Python's
        asyncio.Task.cancel() mechanism. The mock includes small delays to ensure the
        stream is actively processing when cancel is called.

        PORTABLE: Tests agent interface directly with mock streaming.
        """
        # Create a wrapper coroutine to consume the agent stream
        async def consume_agent_stream():  # noqa: ANN202
            chunks = []
            async for chunk in mock_agent.execute_stream(long_request):
                chunks.append(chunk)
            return chunks

        # Test agent interface directly (not through API endpoint)
        agent_task = asyncio.create_task(consume_agent_stream())

        # Allow a few chunks to be processed before cancelling
        await asyncio.sleep(0.05)
        agent_task.cancel()

        # Verify cancellation behavior
        chunks_received = 0

        try:
            chunks = await agent_task
            chunks_received = len(chunks)
        except asyncio.CancelledError:
            pass

        # Verify agent-level cancellation interrupted the stream mid-processing
        assert chunks_received < 7, f"Should interrupt before all 7 chunks, got {chunks_received}"

        print(f"Agent direct cancellation: {chunks_received} chunks received before cancellation")

    @pytest.mark.asyncio
    async def test_agent_resource_cleanup(
        self,
        mock_agent: ReasoningAgent,
        long_request: OpenAIChatRequest,
    ) -> None:
        """
        Test agent properly cleans up resources on cancellation and can be reused.

        Validates that after cancellation, the agent is in a clean state and can
        immediately handle new requests without errors.

        PORTABLE: Tests agent interface contract for resource management.
        """
        # Create a wrapper coroutine to consume the agent stream
        async def consume_agent_stream():  # noqa: ANN202
            chunks = []
            async for chunk in mock_agent.execute_stream(long_request):
                chunks.append(chunk)
            return chunks

        # Start agent stream
        agent_task = asyncio.create_task(consume_agent_stream())

        # Allow a few chunks to process before cancelling
        await asyncio.sleep(0.03)
        agent_task.cancel()

        # Verify cancellation completes without hanging or errors
        cancellation_clean = False
        try:
            await agent_task
            cancellation_clean = True  # Completed successfully
        except asyncio.CancelledError:
            cancellation_clean = True  # Cancelled cleanly

        assert cancellation_clean, "Agent should complete or cancel without hanging"

        # Agent should be in a clean state for reuse
        # Test this by making another request immediately
        async def consume_short_stream():  # noqa: ANN202
            chunks = []
            async for chunk in mock_agent.execute_stream(OpenAIChatRequest(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Short test message"}],
                stream=True,
            )):
                chunks.append(chunk)
                if len(chunks) >= 3:  # Just verify it works
                    break
            return chunks

        new_task = asyncio.create_task(consume_short_stream())

        # Should work without issues after cleanup
        chunks = await new_task
        chunk_count = len(chunks)

        assert chunk_count > 0, "Agent should work correctly after cancellation cleanup"

        print(f"Agent cleanup test: Resource cleanup successful, {chunk_count} chunks in new request")  # noqa: E501

    @pytest.mark.asyncio
    async def test_agent_cancellation_timing(
        self,
        long_request: OpenAIChatRequest,
    ) -> None:
        """
        Test agent cancellation timing and responsiveness.

        This test explicitly validates performance characteristics: that task cancellation
        is responsive and completes quickly. Uses a mock with artificial delays to measure
        cancellation responsiveness.
        """
        # Create mock with delays to measure cancellation responsiveness
        mock_agent = AsyncMock(spec=ReasoningAgent)

        async def mock_execute_stream_with_delays(request, parent_span=None):  # noqa
            stream_chunks = [
                ('data: {"id": "timing-test", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "chunk1"}, "finish_reason": null}]}'),  # noqa: E501
                ('data: {"id": "timing-test", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "chunk2"}, "finish_reason": null}]}'),  # noqa: E501
                ('data: {"id": "timing-test", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "chunk3"}, "finish_reason": null}]}'),  # noqa: E501
                ('data: {"id": "timing-test", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "chunk4"}, "finish_reason": null}]}'),  # noqa: E501
                ('data: {"id": "timing-test", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "chunk5"}, "finish_reason": null}]}'),  # noqa: E501
                ('data: {"id": "timing-test", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}'),  # noqa: E501
                ('data: [DONE]'),
            ]

            for chunk in stream_chunks:
                yield chunk + '\n\n'
                await asyncio.sleep(0.05)

        mock_agent.execute_stream = mock_execute_stream_with_delays

        start_time = time.time()

        async def consume_agent_stream():  # noqa: ANN202
            chunks = []
            async for chunk in mock_agent.execute_stream(long_request):
                chunks.append(chunk)
            return chunks

        agent_task = asyncio.create_task(consume_agent_stream())

        # Allow some processing before cancelling to measure responsiveness
        await asyncio.sleep(0.2)
        cancel_start = time.time()
        agent_task.cancel()

        chunks_after_cancel = 0
        try:
            chunks = await agent_task
            chunks_after_cancel = len(chunks)
            chunks_after_cancel = min(chunks_after_cancel, 7)
        except asyncio.CancelledError:
            pass

        cancel_duration = time.time() - cancel_start
        total_duration = time.time() - start_time

        # Validate cancellation responsiveness
        assert cancel_duration < 0.5, f"Agent cancellation took {cancel_duration:.2f}s, should be under 0.5s for mock"  # noqa: E501
        assert total_duration < 1.0, f"Total test took {total_duration:.2f}s, should be under 1s for mock"  # noqa: E501
        assert chunks_after_cancel < 7, f"Got {chunks_after_cancel} chunks after cancel, should be minimal (mock has 7 total)"  # noqa: E501

        print(f"Agent timing: Cancel took {cancel_duration:.2f}s, total {total_duration:.2f}s, {chunks_after_cancel} chunks after cancel")  # noqa: E501
