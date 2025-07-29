"""Integration tests for cancellation with real components."""

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv
from fastapi import Request

from api.main import chat_completions
from api.openai_protocol import OpenAIChatRequest
from api.reasoning_agent import ReasoningAgent
from tests.fixtures.agents import create_reasoning_agent

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
    def real_agent(self) -> ReasoningAgent:
        """
        Create a real agent that makes actual OpenAI API calls.

        This fixture will be easily updated when OrchestratorAgent replaces ReasoningAgent.
        """
        # Skip if no OpenAI key available
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not available for integration testing")

        return create_reasoning_agent(
            tools=[],  # No tools to keep it simple
            base_url="https://api.openai.com/v1",
            api_key=api_key,
        )

    @pytest.fixture
    def mock_request(self) -> AsyncMock:
        """Create a mock HTTP request for testing."""
        request = AsyncMock(spec=Request)
        request.headers = {"user-agent": "test-integration-client"}
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
                    "content": "Write a detailed 500-word essay about the history of artificial intelligence, including key milestones, important researchers, and future implications. Be very thorough and detailed.",
                },
            ],
            stream=True,
            temperature=0.7,
            max_tokens=1000,  # Ensure a long response
        )

    @pytest.mark.asyncio
    async def test_real_api_cancellation_timing(
        self,
        real_agent: ReasoningAgent,
        mock_request: AsyncMock,
        long_request: OpenAIChatRequest,
    ) -> None:
        """
        Test API cancellation interrupts real OpenAI calls through API layer.

        STABLE: Tests through chat_completions() endpoint with real agent.
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
                reasoning_agent=real_agent,
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
        assert len(chunks_received) < 50, (
            f"Received {len(chunks_received)} chunks. "
            "Too many chunks suggests API cancellation didn't work properly."
        )
        assert 1.0 < duration < 10.0, (
            f"Duration {duration:.2f}s seems wrong. "
            "Should be a few seconds for partial processing + cancellation."
        )

    @pytest.mark.asyncio
    async def test_real_api_cancellation_during_reasoning(
        self,
        real_agent: ReasoningAgent,
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
                    "content": "Think step by step about how to solve this math problem: What is 347 * 892? Show your reasoning process.",
                },
            ],
            stream=True,
            temperature=0.2,
        )

        # Cancel very quickly to catch during reasoning
        call_count = 0
        async def quick_disconnect():
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
                reasoning_agent=real_agent,
                http_request=mock_request,
                _=True,
            )

        # Consume stream
        async for chunk in response.body_iterator:
            chunks_received += 1

        duration = time.time() - start_time

        # Verify we interrupted through API during processing
        assert chunks_received > 0, "Should have received some chunks"
        assert duration < 5.0, "Should have cancelled quickly through API"

        print(f"API Quick cancellation: {chunks_received} chunks in {duration:.2f}s")

    @pytest.mark.asyncio
    async def test_real_multi_client_isolation(
        self,
        real_agent: ReasoningAgent,
        long_request: OpenAIChatRequest,
    ) -> None:
        """
        Test API layer isolates multiple real clients properly.

        STABLE: Tests API concurrent request handling with real processing.
        """
        # Create two separate requests
        request_a = AsyncMock(spec=Request)
        request_a.headers = {"user-agent": "client-a"}
        request_a.url = MagicMock()
        request_a.url.__str__.return_value = "http://test/v1/chat/completions"

        request_b = AsyncMock(spec=Request)
        request_b.headers = {"user-agent": "client-b"}
        request_b.url = MagicMock()
        request_b.url.__str__.return_value = "http://test/v1/chat/completions"
        request_b.is_disconnected = AsyncMock(return_value=False)  # Never disconnect

        # Client A disconnects quickly
        a_call_count = 0
        async def a_disconnect():
            nonlocal a_call_count
            a_call_count += 1
            return a_call_count > 5  # Quick disconnect

        request_a.is_disconnected = a_disconnect

        # Start both clients concurrently through API
        start_time = time.time()

        with patch("api.main.verify_token", return_value=True):
            # Start both API requests
            response_a_task = asyncio.create_task(chat_completions(
                request=long_request,
                reasoning_agent=real_agent,
                http_request=request_a,
                _=True,
            ))
            response_b_task = asyncio.create_task(chat_completions(
                request=long_request,
                reasoning_agent=real_agent,
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
                # Stop B after getting reasonable amount to avoid long test
                if chunks_b > 20:
                    break

        # Consume both concurrently with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(consume_a(), consume_b()),
                timeout=10.0,
            )
        except TimeoutError:
            pass  # Expected for long responses

        duration = time.time() - start_time

        # Verify API layer isolation
        print(f"API Multi-client: Client A: {chunks_a} chunks, Client B: {chunks_b} chunks in {duration:.2f}s")

        # A should be cancelled (fewer chunks), B should continue (more chunks)
        assert chunks_a < chunks_b, "Client A should have been cancelled while B continued through API"
        assert chunks_a > 0, "Client A should have gotten some chunks before API cancellation"
        assert chunks_b > 10, "Client B should have continued processing through API"

    @pytest.mark.asyncio
    async def test_real_api_error_handling_on_cancellation(
        self,
        real_agent: ReasoningAgent,
        mock_request: AsyncMock,
        long_request: OpenAIChatRequest,
    ) -> None:
        """
        Test API layer handles cancellation errors gracefully.

        STABLE: Tests API error handling, not agent error handling.
        """
        # Track cancellation
        cancellation_detected = False

        async def detect_cancellation() -> bool:
            nonlocal cancellation_detected
            # Allow a few chunks to be processed first
            if chunks_received >= 3:
                cancellation_detected = True
                return True
            return False

        mock_request.is_disconnected = detect_cancellation

        # Execute through API
        with patch("api.main.verify_token", return_value=True):
            response = await chat_completions(
                request=long_request,
                reasoning_agent=real_agent,
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
    def real_agent(self) -> ReasoningAgent:
        """
        Create a real agent for direct testing.

        This fixture will be easily updated when OrchestratorAgent replaces ReasoningAgent.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not available for integration testing")

        return create_reasoning_agent(
            tools=[],
            base_url="https://api.openai.com/v1",
            api_key=api_key,
        )

    @pytest.fixture
    def long_request(self) -> OpenAIChatRequest:
        """Create a request that will trigger a long agent response."""
        return OpenAIChatRequest(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Write a comprehensive 400-word analysis of machine learning algorithms, covering supervised learning, unsupervised learning, and deep learning approaches.",
                },
            ],
            stream=True,
            temperature=0.5,
            max_tokens=800,
        )

    @pytest.mark.asyncio
    async def test_real_agent_cancellation_during_openai_calls(
        self,
        real_agent: ReasoningAgent,
        long_request: OpenAIChatRequest,
    ) -> None:
        """
        Test agent cancellation interrupts real OpenAI API calls directly.

        PORTABLE: Tests agent interface directly with real external API.
        """
        # Test agent interface directly (not through API endpoint)
        agent_task = asyncio.create_task(
            real_agent.execute_stream(long_request),
        )

        # Let it start making real OpenAI calls, then cancel
        await asyncio.sleep(1.0)
        start_cancel_time = time.time()
        agent_task.cancel()

        # Try to consume the cancelled stream
        chunks_received = 0
        cancellation_was_fast = False

        try:
            async for chunk in agent_task:
                chunks_received += 1
                # Should not receive many chunks after cancellation
                if chunks_received > 5:
                    break
        except asyncio.CancelledError:
            cancel_duration = time.time() - start_cancel_time
            cancellation_was_fast = cancel_duration < 5.0

        # Verify agent-level cancellation
        assert cancellation_was_fast, "Agent should cancel real OpenAI calls quickly"
        assert chunks_received < 10, "Should not receive many chunks after agent cancellation"

        print(f"Agent direct cancellation: {chunks_received} chunks, cancelled quickly: {cancellation_was_fast}")

    @pytest.mark.asyncio
    async def test_real_agent_resource_cleanup(
        self,
        real_agent: ReasoningAgent,
        long_request: OpenAIChatRequest,
    ) -> None:
        """
        Test agent properly cleans up OpenAI connections on cancellation.

        PORTABLE: Tests agent interface contract for resource management.
        """
        # Start agent stream that will make real OpenAI calls
        agent_task = asyncio.create_task(
            real_agent.execute_stream(long_request),
        )

        # Let it establish connections, then cancel
        await asyncio.sleep(0.5)
        agent_task.cancel()

        # Verify cancellation is clean
        cleanup_successful = False
        try:
            async for chunk in agent_task:
                pass
        except asyncio.CancelledError:
            cleanup_successful = True

        assert cleanup_successful, "Agent should clean up resources and raise CancelledError"

        # Give time for any cleanup
        await asyncio.sleep(0.1)

        # Agent should be in a clean state for reuse
        # Test this by making another request immediately
        new_task = asyncio.create_task(
            real_agent.execute_stream(OpenAIChatRequest(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Short test message"}],
                stream=True,
            )),
        )

        # Should work without issues after cleanup
        chunk_count = 0
        async for chunk in new_task:
            chunk_count += 1
            if chunk_count >= 3:  # Just verify it works
                break

        assert chunk_count > 0, "Agent should work correctly after cancellation cleanup"

        print(f"Agent cleanup test: Resource cleanup successful, {chunk_count} chunks in new request")

    @pytest.mark.asyncio
    async def test_real_agent_cancellation_timing(
        self,
        real_agent: ReasoningAgent,
        long_request: OpenAIChatRequest,
    ) -> None:
        """
        Test agent cancellation timing with real processing.

        PORTABLE: Tests agent interface performance characteristics.
        """
        start_time = time.time()

        # Start agent with real OpenAI processing
        agent_task = asyncio.create_task(
            real_agent.execute_stream(long_request),
        )

        # Let it process for a bit, then cancel
        await asyncio.sleep(1.5)
        cancel_start = time.time()
        agent_task.cancel()

        # Measure cancellation speed
        chunks_after_cancel = 0
        try:
            async for chunk in agent_task:
                chunks_after_cancel += 1
                # Safety limit
                if chunks_after_cancel > 10:
                    break
        except asyncio.CancelledError:
            pass

        cancel_duration = time.time() - cancel_start
        total_duration = time.time() - start_time

        # Verify agent cancellation timing
        assert cancel_duration < 3.0, f"Agent cancellation took {cancel_duration:.2f}s, should be under 3s"
        assert total_duration < 8.0, f"Total test took {total_duration:.2f}s, should be under 8s"
        assert chunks_after_cancel < 5, f"Got {chunks_after_cancel} chunks after cancel, should be minimal"

        print(f"Agent timing: Cancel took {cancel_duration:.2f}s, total {total_duration:.2f}s, {chunks_after_cancel} chunks after cancel")
