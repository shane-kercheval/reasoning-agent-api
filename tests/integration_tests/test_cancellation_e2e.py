"""
True end-to-end cancellation tests with real HTTP server and clients.

These tests start a real HTTP server and make real HTTP requests to test
cancellation behavior without mocks. They test actual client disconnection
scenarios that would occur in production.
"""

import asyncio
import json
import os
import socket
import subprocess
import time
from collections.abc import AsyncGenerator
from typing import Dict, List, Optional

import httpx
import pytest
import pytest_asyncio
from dotenv import load_dotenv

load_dotenv()


class SSEStreamParser:
    """Proper Server-Sent Events (SSE) stream parser with buffering."""
    
    def __init__(self):
        self._buffer = ""
    
    def parse_raw_chunk(self, raw_chunk: str) -> List[Dict]:
        """Parse raw HTTP chunk into complete SSE events."""
        # Add to buffer
        self._buffer += raw_chunk
        
        # Extract complete SSE events (terminated by \n\n)
        events = []
        while '\n\n' in self._buffer:
            event_text, self._buffer = self._buffer.split('\n\n', 1)
            
            # Parse the SSE event
            parsed_event = self._parse_sse_event(event_text.strip())
            if parsed_event:
                events.append(parsed_event)
        
        return events
    
    def _parse_sse_event(self, event_text: str) -> Optional[Dict]:
        """Parse a single complete SSE event."""
        if not event_text:
            return None
            
        # Handle SSE format: "data: {json}"
        if event_text.startswith('data: '):
            data_part = event_text[6:].strip()
            if data_part == '[DONE]':
                return {'done': True}
            try:
                return json.loads(data_part)
            except json.JSONDecodeError:
                return None
        return None
    
    @staticmethod
    def is_reasoning_chunk(parsed_chunk: Dict) -> bool:
        """Check if this chunk contains reasoning events."""
        if not parsed_chunk or parsed_chunk.get('done'):
            return False
        choices = parsed_chunk.get('choices', [])
        if not choices:
            return False
        delta = choices[0].get('delta', {})
        return 'reasoning_event' in delta
    
    @staticmethod
    def is_content_chunk(parsed_chunk: Dict) -> bool:
        """Check if this chunk contains actual content."""
        if not parsed_chunk or parsed_chunk.get('done'):
            return False
        choices = parsed_chunk.get('choices', [])
        if not choices:
            return False
        delta = choices[0].get('delta', {})
        return 'content' in delta and delta['content'] is not None


class InterleavedSSEReader:
    """Reads SSE events from two streams in interleaved fashion."""
    
    def __init__(self, response_a, response_b):
        self.response_a = response_a
        self.response_b = response_b
        self.parser_a = SSEStreamParser()
        self.parser_b = SSEStreamParser()
        self.iter_a = response_a.aiter_text().__aiter__()
        self.iter_b = response_b.aiter_text().__aiter__()
        self.events_a = []
        self.events_b = []
        self.done_a = False
        self.done_b = False
    
    async def read_next_event_from_a(self) -> Optional[Dict]:
        """Read next complete SSE event from stream A."""
        if self.done_a:
            return None
            
        # If we have buffered events, return one
        if self.events_a:
            return self.events_a.pop(0)
        
        # Try to get more raw data and parse events
        try:
            raw_chunk = await asyncio.wait_for(self.iter_a.__anext__(), timeout=5.0)
            new_events = self.parser_a.parse_raw_chunk(raw_chunk)
            self.events_a.extend(new_events)
            
            if self.events_a:
                return self.events_a.pop(0)
        except (StopAsyncIteration, asyncio.TimeoutError):
            self.done_a = True
        
        return None
    
    async def read_next_event_from_b(self) -> Optional[Dict]:
        """Read next complete SSE event from stream B."""
        if self.done_b:
            return None
            
        # If we have buffered events, return one
        if self.events_b:
            return self.events_b.pop(0)
        
        # Try to get more raw data and parse events
        try:
            raw_chunk = await asyncio.wait_for(self.iter_b.__anext__(), timeout=5.0)
            new_events = self.parser_b.parse_raw_chunk(raw_chunk)
            self.events_b.extend(new_events)
            
            if self.events_b:
                return self.events_b.pop(0)
        except (StopAsyncIteration, asyncio.TimeoutError):
            self.done_b = True
        
        return None


@pytest.mark.integration
class TestCancellationE2E:
    """True end-to-end cancellation tests with real HTTP server."""

    @pytest_asyncio.fixture
    async def real_server_url(self) -> AsyncGenerator[str]:
        """Start real HTTP server and return its URL."""
        # Skip if no OpenAI key available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not available for integration testing")
            
        # Find free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]

        # Start real server process with auth disabled for testing
        server_process = subprocess.Popen(
            [
                "uv", "run", "uvicorn", "api.main:app",
                "--host", "127.0.0.1", "--port", str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={
                **os.environ,
                "REQUIRE_AUTH": "false",  # Disable auth for integration test
            },
        )

        # Wait for server to be ready
        server_url = f"http://localhost:{port}"
        for _ in range(50):
            try:
                async with httpx.AsyncClient() as test_client:
                    response = await test_client.get(f"{server_url}/health")
                    if response.status_code == 200:
                        break
            except Exception:
                pass
            await asyncio.sleep(0.1)
        else:
            server_process.terminate()
            raise RuntimeError("Server failed to start")

        try:
            yield server_url
        finally:
            server_process.terminate()
            server_process.wait()

    async def _read_next_chunk_with_timeout(
        self, 
        response: httpx.Response, 
        timeout: float = 2.0
    ) -> Optional[str]:
        """Read next chunk from response stream with timeout."""
        try:
            async for chunk in response.aiter_text():
                if chunk.strip():  # Skip empty chunks
                    return chunk
            return None
        except asyncio.TimeoutError:
            return None

    @pytest.mark.asyncio
    async def test_concurrent_client_cancellation_during_reasoning(
        self, real_server_url: str
    ) -> None:
        """
        Test that cancelling Client A during reasoning phase doesn't affect Client B.
        
        This test uses interleaved chunk reading to ensure Client A cancels while
        Client B is still actively processing reasoning events.
        """
        # Request that triggers reasoning before OpenAI response
        request_data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user", 
                    "content": "Write a creative 20-line poem about space exploration, with each line being at least 10 words long. Be very detailed and poetic."
                }
            ],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 500,
        }

        # Create two separate HTTP clients (real isolation)
        async with httpx.AsyncClient(timeout=30.0) as client_a, \
                   httpx.AsyncClient(timeout=30.0) as client_b:
            
            # Start both requests concurrently
            response_a = await client_a.post(
                f"{real_server_url}/v1/chat/completions", 
                json=request_data
            )
            response_b = await client_b.post(
                f"{real_server_url}/v1/chat/completions", 
                json=request_data
            )
            
            assert response_a.status_code == 200
            assert response_b.status_code == 200

        # Use proper interleaved SSE reading for deterministic cancellation
        reader = InterleavedSSEReader(response_a, response_b)
        
        events_a: List[Dict] = []
        events_b: List[Dict] = []
        reasoning_chunks_a = 0
        reasoning_chunks_b = 0
        
        # Interleaved reading: A, B, A, B, A, B...
        for round_num in range(20):  # Maximum rounds to prevent infinite loop
            
            # Read one event from Client A
            event_a = await reader.read_next_event_from_a()
            if event_a:
                events_a.append(event_a)
                if SSEStreamParser.is_reasoning_chunk(event_a):
                    reasoning_chunks_a += 1
            
            # Read one event from Client B  
            event_b = await reader.read_next_event_from_b()
            if event_b:
                events_b.append(event_b)
                if SSEStreamParser.is_reasoning_chunk(event_b):
                    reasoning_chunks_b += 1
            
            # DETERMINISTIC CANCELLATION: Cancel A after exactly 3 reasoning chunks
            if reasoning_chunks_a >= 3:
                print(f"Cancelling Client A after exactly {reasoning_chunks_a} reasoning chunks")
                await client_a.aclose()  # REAL CLIENT DISCONNECTION
                break
            
            # Safety: if both streams are done, exit
            if reader.done_a and reader.done_b:
                break
        
        # Continue reading Client B to prove it wasn't affected by A's cancellation
        b_events_after_a_cancel = 0
        for _ in range(15):  # Read up to 15 more events from B
            event_b = await reader.read_next_event_from_b()
            if event_b:
                events_b.append(event_b)
                b_events_after_a_cancel += 1
                if SSEStreamParser.is_reasoning_chunk(event_b):
                    reasoning_chunks_b += 1
            else:
                break  # Stream B is done

        # Verify the test results
        print(f"Client A: {len(events_a)} total events, {reasoning_chunks_a} reasoning chunks")
        print(f"Client B: {len(events_b)} total events, {reasoning_chunks_b} reasoning chunks") 
        print(f"Client B events after A cancelled: {b_events_after_a_cancel}")
        
        # Test assertions - deterministic expectations
        assert reasoning_chunks_a == 3, f"Client A should have been cancelled after exactly 3 reasoning chunks, got {reasoning_chunks_a}"
        assert len(events_a) >= 3, f"Client A should have received at least 3 events before cancellation, got {len(events_a)}"
        assert len(events_b) > len(events_a), f"Client B ({len(events_b)}) should have more events than cancelled Client A ({len(events_a)})"
        assert reasoning_chunks_b > reasoning_chunks_a, f"Client B should have processed more reasoning chunks ({reasoning_chunks_b}) than cancelled Client A ({reasoning_chunks_a})"
        assert b_events_after_a_cancel > 0, f"Client B should have continued processing after Client A cancelled, got {b_events_after_a_cancel} additional events"

    @pytest.mark.asyncio
    async def test_concurrent_client_cancellation_during_content_streaming(
        self, real_server_url: str
    ) -> None:
        """
        Test that cancelling Client A during OpenAI content streaming doesn't affect Client B.
        
        This test specifically cancels Client A while it's receiving actual content chunks
        (not reasoning events) to test isolation during the OpenAI response phase.
        """
        # Request that will generate substantial content
        request_data = {
            "model": "gpt-4o-mini", 
            "messages": [
                {
                    "role": "user",
                    "content": "Write a detailed 25-line story about a magical forest, with each line being at least 12 words long. Include vivid descriptions and dialogue."
                }
            ],
            "stream": True,
            "temperature": 0.8,
            "max_tokens": 600,
        }

        # Create two separate HTTP clients
        async with httpx.AsyncClient(timeout=30.0) as client_a, \
                   httpx.AsyncClient(timeout=30.0) as client_b:
            
            # Start both requests
            response_a = await client_a.post(
                f"{real_server_url}/v1/chat/completions", 
                json=request_data
            )
            response_b = await client_b.post(
                f"{real_server_url}/v1/chat/completions", 
                json=request_data
            )
            
            assert response_a.status_code == 200
            assert response_b.status_code == 200

        # Use proper interleaved SSE reading for deterministic cancellation
        reader = InterleavedSSEReader(response_a, response_b)
        
        events_a: List[Dict] = []
        events_b: List[Dict] = []
        content_chunks_a = 0
        content_chunks_b = 0
        
        # Interleaved reading: A, B, A, B, A, B...
        for round_num in range(30):  # Maximum rounds to prevent infinite loop
            
            # Read one event from Client A
            event_a = await reader.read_next_event_from_a()
            if event_a:
                events_a.append(event_a)
                if SSEStreamParser.is_content_chunk(event_a):
                    content_chunks_a += 1
            
            # Read one event from Client B  
            event_b = await reader.read_next_event_from_b()
            if event_b:
                events_b.append(event_b)
                if SSEStreamParser.is_content_chunk(event_b):
                    content_chunks_b += 1
            
            # DETERMINISTIC CANCELLATION: Cancel A after exactly 5 content chunks
            if content_chunks_a >= 5:
                print(f"Cancelling Client A after exactly {content_chunks_a} content chunks")
                await client_a.aclose()  # REAL CLIENT DISCONNECTION
                break
            
            # Safety: if both streams are done, exit
            if reader.done_a and reader.done_b:
                break
        
        # Continue reading Client B to prove it wasn't affected by A's cancellation
        b_events_after_a_cancel = 0
        for _ in range(20):  # Read up to 20 more events from B
            event_b = await reader.read_next_event_from_b()
            if event_b:
                events_b.append(event_b)
                b_events_after_a_cancel += 1
                if SSEStreamParser.is_content_chunk(event_b):
                    content_chunks_b += 1
            else:
                break  # Stream B is done

        # Verify results
        print(f"Client A: {len(events_a)} total events, {content_chunks_a} content chunks")
        print(f"Client B: {len(events_b)} total events, {content_chunks_b} content chunks")
        print(f"Client B events after A cancelled: {b_events_after_a_cancel}")
        
        # Test assertions - deterministic expectations
        assert content_chunks_a == 5, f"Client A should have been cancelled after exactly 5 content chunks, got {content_chunks_a}"
        assert len(events_a) >= 5, f"Client A should have received at least 5 events before cancellation, got {len(events_a)}"
        assert len(events_b) > len(events_a), f"Client B ({len(events_b)}) should have more events than cancelled Client A ({len(events_a)})"
        assert content_chunks_b > content_chunks_a, f"Client B should have processed more content chunks ({content_chunks_b}) than cancelled Client A ({content_chunks_a})"
        assert b_events_after_a_cancel > 0, f"Client B should have continued processing after Client A cancelled, got {b_events_after_a_cancel} additional events"