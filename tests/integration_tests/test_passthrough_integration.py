"""
Integration tests for passthrough path (direct OpenAI API).

These tests validate the end-to-end passthrough path with real OpenAI API calls:
- Request routing (passthrough rules, headers)
- Direct OpenAI API calls (streaming and non-streaming)
- OpenAI error forwarding
- Tracing integration

Requires OPENAI_API_KEY environment variable.
"""

import os
import json

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from dotenv import load_dotenv

from api.main import app
from api.dependencies import service_container
from tests.conftest import ReasoningAgentStreamingCollector

load_dotenv()

# Mark all tests as integration tests and async
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY required for integration tests",
    ),
]


@pytest_asyncio.fixture(scope="module", autouse=True)
async def setup_teardown():
    """Initialize and cleanup service container for all tests."""
    await service_container.initialize()
    yield
    await service_container.cleanup()


@pytest_asyncio.fixture
async def client() -> AsyncClient: # type: ignore
    """Create async HTTP client for API testing."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


class TestPassthroughStreaming:
    """Test passthrough path with streaming responses."""

    async def test_streaming_with_passthrough_rule(self, client: AsyncClient) -> None:
        """Streaming request with response_format should use passthrough path."""
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Count from 1 to 3"}],
                "stream": True,
                "max_tokens": 50,
                "response_format": {"type": "json_object"},
            },
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Collect all SSE chunks
        collector = ReasoningAgentStreamingCollector()
        await collector.process(response.aiter_lines())

        # Validate streaming response structure
        assert len(collector.all_chunks) > 0
        assert len(collector.content) > 0

    async def test_streaming_with_header_override(self, client: AsyncClient) -> None:
        """Streaming with X-Routing-Mode: passthrough header should work."""
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": True,
                "max_tokens": 20,
            },
            headers={"X-Routing-Mode": "passthrough"},
        )

        assert response.status_code == 200

        # Collect chunks
        chunks = []
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data_part = line[6:].strip()
                if data_part == "[DONE]":
                    break
                chunks.append(json.loads(data_part))

        assert len(chunks) > 0
        assert chunks[0]["object"] == "chat.completion.chunk"

    async def test_openai_error_forwarding(self, client: AsyncClient) -> None:
        """
        OpenAI API errors during streaming cause exception.

        In streaming-only architecture, validation errors (like invalid model) occur
        during the first iteration of the stream, after 200 headers are sent.
        The stream aborts and raises an exception instead of returning error codes.
        """
        # Streaming-only: errors during stream generation raise exceptions
        with pytest.raises(Exception):  # OpenAI SDK raises BadRequestError/NotFoundError  # noqa: E501, PT011, PT012
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "invalid-model-name-that-does-not-exist",
                    "messages": [{"role": "user", "content": "Test"}],
                    "response_format": {"type": "json_object"},  # Force passthrough
                    "stream": True,
                },
            )
            # Force stream consumption to trigger the error
            async for _ in response.aiter_lines():
                pass


class TestOrchestrationPathStub:
    """Test orchestration path returns 501 Not Implemented stub."""

    async def test_orchestration_query_returns_501(self, client: AsyncClient) -> None:
        """Orchestration query with explicit header should return 501."""
        # Explicitly route to orchestration to test 501 stub
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{
                    "role": "user",
                    "content": "Research renewable energy and create investment strategy",
                }],
                "stream": True,
            },
            headers={"X-Routing-Mode": "orchestration"},
        )

        # Orchestration is not yet implemented, should return 501
        # If getting 500, there's an internal error - check response
        if response.status_code == 500:
            print(f"Got 500 error: {response.json()}")
        assert response.status_code == 501
        data = response.json()
        assert "detail" in data
        assert "error" in data["detail"]


class TestRequestValidation:
    """Test request validation and error handling."""

    async def test_invalid_request_returns_422(self, client: AsyncClient) -> None:
        """Invalid request should return 422 validation error."""
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                # Missing required 'messages' field
                "response_format": {"type": "json_object"},
                "stream": True,
            },
        )

        assert response.status_code == 422  # Validation error


class TestAuthenticationIntegration:
    """Test authentication with routing (if auth enabled)."""

    async def test_passthrough_requires_auth_if_enabled(self) -> None:
        """Passthrough path should respect authentication settings."""
        # This test depends on REQUIRE_AUTH setting
        # If auth is disabled, endpoint works without token
        # If auth is enabled, endpoint requires token
        # Test documents expected behavior

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Try without auth header
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "response_format": {"type": "json_object"},
                    "stream": True,
                },
            )

            # If REQUIRE_AUTH=false, should succeed (200)
            # If REQUIRE_AUTH=true, should fail (401)
            assert response.status_code in (200, 401)
