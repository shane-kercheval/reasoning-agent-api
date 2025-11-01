"""
Integration tests for API cancellation with real executors and mocked LiteLLM.

Tests use REAL PassthroughExecutor and ReasoningAgent with mocked LiteLLM responses.
This ensures we test actual cancellation logic without expensive OpenAI API calls.

Key principle: Mock external dependencies (LiteLLM), NOT our own code (executors).
"""

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from opentelemetry import trace
from litellm import ModelResponse
from litellm.types.utils import StreamingChoices, Delta

from api.main import chat_completions
from api.openai_protocol import OpenAIChatRequest


# ============================================================================
# Fixtures: Create real executors with mocked LiteLLM
# ============================================================================

@pytest.fixture
def mock_tools():
    """Mock tools for ReasoningAgent."""
    return []


@pytest.fixture
def mock_prompt_manager():
    """Mock prompt manager for ReasoningAgent."""
    prompt_manager = AsyncMock()
    prompt_manager.get_prompt.return_value = "You are a helpful assistant."
    return prompt_manager


def create_mock_llm_stream(num_chunks: int = 10) -> AsyncGenerator:
    """
    Create a mock LiteLLM streaming response using proper LiteLLM types.

    Yields realistic ModelResponse chunks with delays to allow cancellation testing.
    """
    async def stream():
        # Yield content chunks
        words = ["This", "is", "a", "test", "response", "with", "multiple", "chunks", "for", "testing"]
        for i in range(num_chunks):
            word = words[i % len(words)]
            yield ModelResponse(
                id="chatcmpl-test",
                object="chat.completion.chunk",
                created=1234567890,
                model="gpt-4o-mini",
                choices=[StreamingChoices(
                    index=0,
                    delta=Delta(content=f"{word} "),
                    finish_reason=None,
                )],
            )
            await asyncio.sleep(0.01)  # Small delay to allow disconnection checks

        # Yield final chunk
        yield ModelResponse(
            id="chatcmpl-test",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-4o-mini",
            choices=[StreamingChoices(
                index=0,
                delta=Delta(),
                finish_reason="stop",
            )],
        )

    return stream()


# ============================================================================
# Parameterized Test Class: Tests both Passthrough and Reasoning executors
# ============================================================================

@pytest.mark.integration
class TestExecutorCancellation:
    """
    Test cancellation for both PassthroughExecutor and ReasoningAgent.

    Uses pytest.mark.parametrize to run same tests against both executors.
    Tests REAL cancellation logic with mocked LiteLLM (no API calls).
    """

    @pytest.fixture(params=["passthrough", "reasoning"])
    def executor_mode(self, request):
        """Parametrize tests to run for both executor types."""
        return request.param

    @pytest.fixture
    def mock_request(self, executor_mode):
        """Create mock HTTP request with appropriate routing."""
        request = AsyncMock(spec=Request)
        request.headers = {
            "user-agent": "test-client",
            "x-routing-mode": executor_mode,  # Routes to correct executor
        }
        request.url = MagicMock()
        request.url.__str__.return_value = "http://test/v1/chat/completions"
        request.is_disconnected = AsyncMock(return_value=False)
        return request

    @pytest.fixture
    def chat_request(self):
        """Create a test chat request."""
        return OpenAIChatRequest(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test message"}],
            stream=True,
        )

    @pytest.mark.asyncio
    async def test_disconnection_stops_stream(
        self,
        executor_mode: str,
        mock_request: AsyncMock,
        chat_request: OpenAIChatRequest,
        mock_tools: list,
        mock_prompt_manager: AsyncMock,
    ):
        """
        Test that client disconnection stops the stream.

        High-value test: Verifies REAL disconnection checking logic works.
        """
        chunks_received = []

        # Disconnect after 3 chunks
        call_count = 0
        async def is_disconnected():
            nonlocal call_count
            call_count += 1
            return call_count > 3

        mock_request.is_disconnected = is_disconnected

        # Mock LiteLLM to return streaming response
        patch_target = (
            "api.executors.passthrough.litellm.acompletion"
            if executor_mode == "passthrough"
            else "api.executors.reasoning_agent.litellm.acompletion"
        )

        with patch("api.main.verify_token", return_value=True), \
             patch(patch_target, return_value=create_mock_llm_stream(num_chunks=20)):

            response = await chat_completions(
                request=chat_request,
                tools=mock_tools,
                prompt_manager=mock_prompt_manager,
                conversation_db=None,
                http_request=mock_request,
                _=True,
            )

            # Consume stream - should stop due to disconnection
            async for chunk in response.body_iterator:
                chunks_received.append(chunk)

        # Verify disconnection stopped the stream
        assert len(chunks_received) > 0, "Should receive some chunks before disconnection"
        assert len(chunks_received) < 15, (
            f"Received {len(chunks_received)} chunks. Should stop around 3-5 chunks due to disconnection. "
            f"If this fails, {executor_mode} executor isn't checking disconnection!"
        )
        assert call_count >= 3, "Should have checked disconnection multiple times"

    @pytest.mark.asyncio
    async def test_disconnection_timing_precision(
        self,
        executor_mode: str,
        mock_request: AsyncMock,
        chat_request: OpenAIChatRequest,
        mock_tools: list,
        mock_prompt_manager: AsyncMock,
    ):
        """
        Test that disconnection is detected promptly (not delayed).

        High-value test: Ensures disconnection checking happens per-chunk.
        """
        chunks_received = []
        disconnect_after = 5

        call_count = 0
        async def is_disconnected():
            nonlocal call_count
            call_count += 1
            return call_count > disconnect_after

        mock_request.is_disconnected = is_disconnected

        patch_target = (
            "api.executors.passthrough.litellm.acompletion"
            if executor_mode == "passthrough"
            else "api.executors.reasoning_agent.litellm.acompletion"
        )

        with patch("api.main.verify_token", return_value=True), \
             patch(patch_target, return_value=create_mock_llm_stream(num_chunks=30)):

            response = await chat_completions(
                request=chat_request,
                tools=mock_tools,
                prompt_manager=mock_prompt_manager,
                conversation_db=None,
                http_request=mock_request,
                _=True,
            )

            async for chunk in response.body_iterator:
                chunks_received.append(chunk)

        # Verify disconnection detected within reasonable tolerance
        assert disconnect_after - 2 <= len(chunks_received) <= disconnect_after + 3, (
            f"Disconnected after {disconnect_after} checks but received {len(chunks_received)} chunks. "
            f"Disconnection detection should be precise (±2 chunks tolerance)."
        )

    @pytest.mark.asyncio
    async def test_concurrent_requests_isolation(
        self,
        executor_mode: str,
        chat_request: OpenAIChatRequest,
        mock_tools: list,
        mock_prompt_manager: AsyncMock,
    ):
        """
        Test that concurrent requests are isolated (one cancels, other continues).

        High-value test: Prevents shared state bugs causing cascading cancellations.
        """
        # Create 3 separate mock requests
        requests = []
        for i in range(3):
            req = AsyncMock(spec=Request)
            req.headers = {
                "user-agent": f"client-{i}",
                "x-routing-mode": executor_mode,
            }
            req.url = MagicMock()
            req.url.__str__.return_value = "http://test/v1/chat/completions"
            req.is_disconnected = AsyncMock(return_value=False)
            requests.append(req)

        # Client 1 disconnects early
        disconnect_count = 0
        async def client1_disconnect():
            nonlocal disconnect_count
            disconnect_count += 1
            return disconnect_count > 2

        requests[1].is_disconnected = client1_disconnect

        patch_target = (
            "api.executors.passthrough.litellm.acompletion"
            if executor_mode == "passthrough"
            else "api.executors.reasoning_agent.litellm.acompletion"
        )

        # Execute all requests concurrently
        # IMPORTANT: Return a function that creates NEW generators for each call
        with patch("api.main.verify_token", return_value=True), \
             patch(patch_target, side_effect=lambda *args, **kwargs: create_mock_llm_stream(num_chunks=15)):

            responses = await asyncio.gather(*[
                chat_completions(
                    request=chat_request,
                    tools=mock_tools,
                    prompt_manager=mock_prompt_manager,
                    conversation_db=None,
                    http_request=requests[i],
                    _=True,
                )
                for i in range(3)
            ])

            # Consume all streams (must be inside patch context)
            results = []
            for response in responses:
                chunks = []
                async for chunk in response.body_iterator:
                    chunks.append(chunk)
                results.append(chunks)

        # Verify isolation: Client 0 and 2 get full response, Client 1 is cancelled
        assert len(results[0]) >= 10, f"Client 0 should complete fully, got {len(results[0])} chunks"
        assert len(results[1]) <= 5, f"Client 1 should be cancelled early, got {len(results[1])} chunks"
        assert len(results[2]) >= 10, f"Client 2 should complete fully, got {len(results[2])} chunks"

    @pytest.mark.asyncio
    async def test_span_marked_on_cancellation(
        self,
        executor_mode: str,
        mock_request: AsyncMock,
        chat_request: OpenAIChatRequest,
        mock_tools: list,
        mock_prompt_manager: AsyncMock,
    ):
        """
        Test that OpenTelemetry spans are properly marked on cancellation.

        Medium-value test: Ensures observability during cancellations.
        """
        # Create a properly configured mock span with context manager support
        mock_span = MagicMock(spec=trace.Span)
        mock_span.is_recording.return_value = True
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)

        # Create a mock span context with proper trace/span IDs
        mock_span_context = MagicMock()
        mock_span_context.trace_id = 123456789  # Real integer, not a mock
        mock_span_context.span_id = 987654321  # Real integer, not a mock
        mock_span_context.trace_flags = 1  # Real integer
        mock_span.get_span_context.return_value = mock_span_context

        # Disconnect after first chunk (call_count >= 2 means on 2nd check)
        call_count = 0
        async def is_disconnected():
            nonlocal call_count
            call_count += 1
            return call_count >= 2  # Return True on 2nd check, not 3rd

        mock_request.is_disconnected = is_disconnected

        patch_target = (
            "api.executors.passthrough.litellm.acompletion"
            if executor_mode == "passthrough"
            else "api.executors.reasoning_agent.litellm.acompletion"
        )

        with patch("api.main.tracer.start_span", return_value=mock_span), \
             patch("api.main.verify_token", return_value=True), \
             patch(patch_target, return_value=create_mock_llm_stream(num_chunks=10)):

            response = await chat_completions(
                request=chat_request,
                tools=mock_tools,
                prompt_manager=mock_prompt_manager,
                conversation_db=None,
                http_request=mock_request,
                _=True,
            )

            async for _ in response.body_iterator:
                pass

        # Verify span was marked with cancellation
        mock_span.set_attribute.assert_any_call("http.cancelled", True)
        mock_span.set_attribute.assert_any_call("cancellation.reason", "Client disconnected")


# ============================================================================
# ReasoningAgent-Specific Tests
# ============================================================================

@pytest.mark.integration
class TestReasoningAgentCancellationSpecifics:
    """
    Tests specific to ReasoningAgent cancellation behavior.

    These tests cover reasoning-specific scenarios that don't apply to passthrough.
    """

    @pytest.fixture
    def reasoning_request(self):
        """Create mock request for reasoning path."""
        request = AsyncMock(spec=Request)
        request.headers = {
            "user-agent": "test-client",
            "x-routing-mode": "reasoning",
        }
        request.url = MagicMock()
        request.url.__str__.return_value = "http://test/v1/chat/completions"
        request.is_disconnected = AsyncMock(return_value=False)
        return request

    @pytest.mark.asyncio
    async def test_cancellation_during_reasoning_loop(
        self,
        reasoning_request: AsyncMock,
        mock_tools: list,
        mock_prompt_manager: AsyncMock,
    ):
        """
        Test that cancellation stops the reasoning loop (not just final synthesis).

        ReasoningAgent-specific: Tests multi-iteration reasoning loop cancellation.
        """
        chunks_received = []

        # Disconnect quickly to catch during reasoning
        call_count = 0
        async def is_disconnected():
            nonlocal call_count
            call_count += 1
            return call_count > 2

        reasoning_request.is_disconnected = is_disconnected

        # Mock LiteLLM to return multi-iteration response
        async def multi_iteration_stream():
            # Iteration 1
            yield ModelResponse(
                id="test",
                object="chat.completion.chunk",
                created=123,
                model="gpt-4o-mini",
                choices=[StreamingChoices(
                    index=0,
                    delta=Delta(content="Thinking..."),
                    finish_reason=None,
                )],
            )
            await asyncio.sleep(0.01)

            # Iteration 2
            yield ModelResponse(
                id="test",
                object="chat.completion.chunk",
                created=123,
                model="gpt-4o-mini",
                choices=[StreamingChoices(
                    index=0,
                    delta=Delta(content="More thinking..."),
                    finish_reason=None,
                )],
            )
            await asyncio.sleep(0.01)

            # Final synthesis (should not reach here if cancelled)
            for word in ["Final", "answer", "here"]:
                yield ModelResponse(
                    id="test",
                    object="chat.completion.chunk",
                    created=123,
                    model="gpt-4o-mini",
                    choices=[StreamingChoices(
                        index=0,
                        delta=Delta(content=f"{word} "),
                        finish_reason=None,
                    )],
                )
                await asyncio.sleep(0.01)

        with patch("api.main.verify_token", return_value=True), \
             patch("api.executors.reasoning_agent.litellm.acompletion",
                   return_value=multi_iteration_stream()):

            response = await chat_completions(
                request=OpenAIChatRequest(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Think step by step"}],
                    stream=True,
                ),
                tools=mock_tools,
                prompt_manager=mock_prompt_manager,
                conversation_db=None,
                http_request=reasoning_request,
                _=True,
            )

            async for chunk in response.body_iterator:
                chunks_received.append(chunk)

        # Should be cancelled during reasoning, not reach final synthesis
        assert len(chunks_received) > 0, "Should receive some chunks"
        assert len(chunks_received) < 8, (
            f"Received {len(chunks_received)} chunks. Should be cancelled during reasoning loop. "
            "If this fails, reasoning agent isn't checking disconnection during iterations!"
        )
