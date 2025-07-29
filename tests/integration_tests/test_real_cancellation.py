"""Real integration tests for cancellation with actual OpenAI API calls."""

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
from api.prompt_manager import PromptManager
from tests.fixtures.agents import create_reasoning_agent, create_mock_prompt_manager

# Load environment variables
load_dotenv()


@pytest.mark.integration
class TestRealCancellation:
    """Integration tests with real OpenAI API calls to verify cancellation works."""

    @pytest.fixture
    def real_reasoning_agent(self) -> ReasoningAgent:
        """Create a real reasoning agent that makes actual OpenAI API calls."""
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
                    "content": "Write a detailed 500-word essay about the history of artificial intelligence, including key milestones, important researchers, and future implications. Be very thorough and detailed."
                }
            ],
            stream=True,
            temperature=0.7,
            max_tokens=1000,  # Ensure a long response
        )

    @pytest.mark.asyncio
    async def test_cancellation_interrupts_real_openai_calls(
        self,
        real_reasoning_agent: ReasoningAgent,
        mock_request: AsyncMock,
        long_request: OpenAIChatRequest,
    ) -> None:
        """
        Test that cancellation actually interrupts real OpenAI API calls.
        
        This test validates that:
        - Real reasoning process starts (we get chunks)
        - Cancellation stops the reasoning process when client disconnects
        - The interruption is clean and handled properly
        """
        chunks_received = []
        cancellation_triggered = False
        
        # Simple disconnection: return False initially, then True after some chunks
        async def is_disconnected():
            nonlocal cancellation_triggered
            # Allow a few chunks to be processed, then simulate disconnection
            if len(chunks_received) >= 3 and not cancellation_triggered:
                cancellation_triggered = True
                return True
            return False
        
        mock_request.is_disconnected = is_disconnected
        
        # Execute the request
        with patch("api.main.verify_token", return_value=True):
            response = await chat_completions(
                request=long_request,
                reasoning_agent=real_reasoning_agent,
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
        
        print(f"Received {len(chunks_received)} chunks in {duration:.2f}s")
        print(f"Cancellation triggered: {cancellation_triggered}")
        
        # Verify the test worked properly
        assert len(chunks_received) > 0, "Should have received some chunks before cancellation"
        assert cancellation_triggered, "Cancellation should have been triggered"
        
        # The key test: we should get a reasonable number of chunks but not a complete response
        # A complete 500-word essay would generate many more chunks
        assert len(chunks_received) < 50, (
            f"Received {len(chunks_received)} chunks. "
            "Too many chunks suggests cancellation didn't work properly."
        )
        
        # Duration should be reasonable - not too fast (no processing) or too slow (no cancellation)
        assert 1.0 < duration < 10.0, (
            f"Duration {duration:.2f}s seems wrong. "
            "Should be a few seconds for partial processing + cancellation."
        )

    @pytest.mark.asyncio
    async def test_cancellation_during_reasoning_phase(
        self,
        real_reasoning_agent: ReasoningAgent,
        mock_request: AsyncMock,
    ) -> None:
        """Test cancellation during the reasoning phase specifically."""
        # Request that will trigger reasoning but not tool use
        request = OpenAIChatRequest(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Think step by step about how to solve this math problem: What is 347 * 892? Show your reasoning process."
                }
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
        planning_events_received = 0
        
        # Execute
        with patch("api.main.verify_token", return_value=True):
            response = await chat_completions(
                request=request,
                reasoning_agent=real_reasoning_agent,
                http_request=mock_request,
                _=True,
            )
        
        # Consume stream
        async for chunk in response.body_iterator:
            chunks_received += 1
            if "planning" in chunk.lower():
                planning_events_received += 1
        
        duration = time.time() - start_time
        
        # Verify we interrupted during reasoning phase
        assert chunks_received > 0, "Should have received some chunks"
        assert duration < 5.0, "Should have cancelled quickly during reasoning"
        
        print(f"Quick cancellation: {chunks_received} chunks in {duration:.2f}s")

    @pytest.mark.asyncio
    async def test_multiple_real_clients_isolation(
        self,
        real_reasoning_agent: ReasoningAgent,
        long_request: OpenAIChatRequest,
    ) -> None:
        """Test that cancelling one real client doesn't affect another."""
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
        
        # Start both clients concurrently
        start_time = time.time()
        
        with patch("api.main.verify_token", return_value=True):
            # Start both requests
            response_a_task = asyncio.create_task(chat_completions(
                request=long_request,
                reasoning_agent=real_reasoning_agent,
                http_request=request_a,
                _=True,
            ))
            response_b_task = asyncio.create_task(chat_completions(
                request=long_request,
                reasoning_agent=real_reasoning_agent,
                http_request=request_b,
                _=True,
            ))
            
            responses = await asyncio.gather(response_a_task, response_b_task)
            response_a, response_b = responses
        
        # Consume both streams
        chunks_a = 0
        chunks_b = 0
        
        async def consume_a():
            nonlocal chunks_a
            async for chunk in response_a.body_iterator:
                chunks_a += 1
        
        async def consume_b():
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
                timeout=10.0
            )
        except asyncio.TimeoutError:
            pass  # Expected for long responses
        
        duration = time.time() - start_time
        
        # Verify isolation
        print(f"Client A: {chunks_a} chunks, Client B: {chunks_b} chunks in {duration:.2f}s")
        
        # A should be cancelled (fewer chunks)
        # B should continue (more chunks)  
        assert chunks_a < chunks_b, "Client A should have been cancelled while B continued"
        assert chunks_a > 0, "Client A should have gotten some chunks before cancellation"
        assert chunks_b > 10, "Client B should have continued processing"