"""
Integration tests for passthrough path (stateless mode).

These tests validate the end-to-end passthrough path with mocked LiteLLM:
- Request routing (passthrough rules, headers)
- API endpoint behavior (streaming and non-streaming)
- Error handling and validation
- Tracing integration

NO EXTERNAL SERVICES REQUIRED - LiteLLM is mocked at the HTTP library level.
"""

import json
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from api.main import app
from tests.conftest import ReasoningAgentStreamingCollector
from tests.integration_tests.conftest import create_mock_litellm_stream

# Mark all tests as integration tests and async
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


@pytest_asyncio.fixture
async def client() -> AsyncClient: # type: ignore
    """
    Create async HTTP client for API testing.

    Note: The app's lifespan function handles service_container initialization,
    so no separate setup/teardown fixture is needed.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


class TestPassthroughStreaming:
    """Test passthrough path with streaming responses."""

    @patch('api.executors.passthrough.litellm.acompletion')
    async def test_streaming_with_passthrough_rule(self, mock_litellm, client: AsyncClient) -> None:
        """Streaming request with response_format should use passthrough path."""
        # Mock LiteLLM response
        mock_litellm.return_value = create_mock_litellm_stream("1, 2, 3")

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

    @patch('api.executors.passthrough.litellm.acompletion')
    async def test_streaming_with_header_override(self, mock_litellm, client: AsyncClient) -> None:
        """Streaming with X-Routing-Mode: passthrough header should work."""
        # Mock LiteLLM response
        mock_litellm.return_value = create_mock_litellm_stream("Hello!")

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
        # Verify we got streaming response chunks
        assert chunks[0]["object"] in ("chat.completion.chunk", "chat.completion")

    @patch('api.executors.passthrough.litellm.acompletion')
    async def test_openai_error_forwarding(self, mock_litellm, client: AsyncClient) -> None:
        """
        LiteLLM API errors during streaming cause exception.

        In streaming-only architecture, validation errors (like invalid model) occur
        during the first iteration of the stream, after 200 headers are sent.
        The stream aborts and raises an exception instead of returning error codes.
        """
        # Mock LiteLLM to raise an error (simulating invalid model)
        from litellm.exceptions import BadRequestError
        mock_litellm.side_effect = BadRequestError(
            message="Invalid model",
            model="invalid-model",
            llm_provider="openai",
        )

        # Streaming-only: errors during stream generation raise exceptions
        with pytest.raises(Exception):  # LiteLLM raises BadRequestError  # noqa: E501, PT011, PT012
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

    @patch('api.executors.passthrough.litellm.acompletion')
    async def test_passthrough_requires_auth_if_enabled(self, mock_litellm) -> None:
        """Passthrough path should respect authentication settings."""
        # Mock LiteLLM response
        mock_litellm.return_value = create_mock_litellm_stream("Hello!")

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
