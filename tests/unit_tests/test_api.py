"""
API endpoint tests for the FastAPI application - Refactored with Dependency Injection.

Tests the FastAPI routes using clean dependency injection patterns instead of
complex global state mocking. This demonstrates the power of FastAPI's DI system.
"""

import asyncio
import os
import socket
import subprocess
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from api.main import app
from api.openai_protocol import (
    SSE_DONE,
    OpenAIStreamingResponseBuilder,
)
from api.dependencies import get_reasoning_agent
from api.main import list_tools
# ToolInfo removed - using new Tool abstraction
from api.auth import verify_token
from api.tools import function_to_tool

load_dotenv()


@pytest.fixture(autouse=True)
def disable_auth_for_tests():
    """
    Disable authentication for existing tests.

    This fixture automatically disables authentication for all tests in this file
    to maintain backward compatibility. Tests that specifically want to test
    authentication should override this behavior.
    """
    # Override the verify_token dependency to always return True
    app.dependency_overrides[verify_token] = lambda: True
    yield
    # Clean up
    if verify_token in app.dependency_overrides:
        del app.dependency_overrides[verify_token]


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test__health_endpoint__returns_healthy_status(self) -> None:
        """Test that health endpoint returns healthy status."""
        with TestClient(app) as client:
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data


class TestModelsEndpoint:
    """Test models listing endpoint."""

    def test__models_endpoint__returns_available_models(self) -> None:
        """Test that models endpoint returns available models."""
        with TestClient(app) as client:
            response = client.get("/v1/models")

            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "list"
            assert len(data["data"]) >= 2  # At least gpt-4o, gpt-4o-mini

            model_ids = [model["id"] for model in data["data"]]
            assert "gpt-4o" in model_ids
            assert "gpt-4o-mini" in model_ids


class TestChatCompletionsEndpoint:
    """Test chat completions endpoint."""

    def test__streaming_chat_completion__success(self) -> None:
        """Test successful streaming chat completion."""
        # Mock the streaming response using the builder
        async def mock_stream_generator(*args, **kwargs) -> AsyncGenerator[str]:  # noqa: ARG001, ANN002, ANN003
            stream = (
                OpenAIStreamingResponseBuilder()
                .chunk("chatcmpl-test", "gpt-4o", delta_content="Analyzing request...")
                .chunk("chatcmpl-test", "gpt-4o", delta_content="Hello")
                .chunk("chatcmpl-test", "gpt-4o", finish_reason="stop")
                .done()
                .build()
            )
            for chunk in stream.split('\n\n')[:-1]:  # Split and remove last empty element
                if chunk.strip():
                    yield chunk + "\n\n"

        # Create mock reasoning agent with streaming support
        # The reasoning path calls execute_stream() which returns an async generator
        # Use side_effect to make the mock return an async generator when called
        mock_agent = AsyncMock()
        mock_agent.execute_stream = mock_stream_generator

        # Use FastAPI dependency override
        app.dependency_overrides[get_reasoning_agent] = lambda: mock_agent

        try:
            with TestClient(app) as client:
                request_data = {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "stream": True,
                }

                response = client.post(
                    "/v1/chat/completions",
                    json=request_data,
                    headers={"X-Routing-Mode": "reasoning"},
                )

                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

                # Check that we get streaming data
                content = response.content.decode()
                # Check that we get reasoning step content
                assert "Analyzing request..." in content
                assert SSE_DONE in content
        finally:
            app.dependency_overrides.clear()

    def test__chat_completion__forwards_openai_errors(self) -> None:
        """
        Test that OpenAI API errors during streaming cause stream failure.

        In streaming-only architecture, once StreamingResponse is returned with 200 status,
        errors during streaming cause the stream to abort rather than return HTTP error codes.
        This matches real-world behavior where network/API errors during streaming result
        in incomplete streams.
        """
        # Mock agent to raise HTTPStatusError during streaming
        mock_agent = AsyncMock()

        # Create a mock HTTPStatusError
        error_response_json = {
            "error": {
                "message": "Invalid API key provided",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            },
        }

        mock_response = httpx.Response(401, json=error_response_json)
        mock_error = httpx.HTTPStatusError(
            "HTTP 401",
            request=httpx.Request("POST", "test"),
            response=mock_response,
        )

        # Make execute_stream raise the error
        async def mock_stream_with_error(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202, ARG001
            raise mock_error
            yield  # Never reached but needed for generator syntax

        mock_agent.execute_stream = mock_stream_with_error

        app.dependency_overrides[get_reasoning_agent] = lambda: mock_agent

        try:
            with TestClient(app) as client:
                request_data = {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "stream": True,
                }

                # With streaming-only architecture, errors during streaming cause exception
                # rather than returning HTTP error codes (since 200 headers already sent)
                with pytest.raises((httpx.HTTPStatusError, Exception)):  # noqa: PT012
                    response = client.post(
                        "/v1/chat/completions",
                        json=request_data,
                        headers={"X-Routing-Mode": "reasoning"},
                    )
                    # Force consumption of stream to trigger the error
                    _ = response.content
        finally:
            app.dependency_overrides.clear()

    def test__chat_completion__handles_invalid_request_data(self) -> None:
        """Test that invalid request data returns proper error."""
        with TestClient(app) as client:
            # Missing required fields
            request_data = {
                "messages": [{"role": "user", "content": "Hello!"}],
                # Missing 'model' field
            }

            response = client.post("/v1/chat/completions", json=request_data)

            assert response.status_code == 422  # Validation error
            data = response.json()
            assert "detail" in data

    def test__chat_completion__handles_invalid_message_role(self) -> None:
        """Test that invalid message role returns proper error."""
        with TestClient(app) as client:
            request_data = {
                "model": "gpt-4o",
                "messages": [{"role": "invalid_role", "content": "Hello!"}],
            }

            response = client.post("/v1/chat/completions", json=request_data)

            assert response.status_code == 422  # Validation error

    def test__chat_completion__handles_internal_server_errors(self) -> None:
        """
        Test that internal server errors during streaming cause stream failure.

        Similar to OpenAI errors, internal errors during streaming cause stream abortion
        rather than returning HTTP 500 codes, since headers are already sent.
        """
        mock_agent = AsyncMock()

        # Make execute_stream raise an internal error
        async def mock_stream_with_error(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202, ARG001
            raise Exception("Internal error")
            yield  # Never reached but needed for generator syntax

        mock_agent.execute_stream = mock_stream_with_error

        app.dependency_overrides[get_reasoning_agent] = lambda: mock_agent
        try:
            with TestClient(app) as client:
                request_data = {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "stream": True,
                }

                # Errors during streaming cause exception rather than HTTP error responses
                with pytest.raises(Exception):  # noqa: PT011, PT012
                    response = client.post(
                        "/v1/chat/completions",
                        json=request_data,
                        headers={"X-Routing-Mode": "reasoning"},
                    )
                    # Force consumption of stream to trigger the error
                    _ = response.content
        finally:
            app.dependency_overrides.clear()


class TestToolsEndpoint:
    """Test tools endpoint."""

    @pytest.mark.asyncio
    async def test__tools_endpoint__with_tools_available(self) -> None:
        """Test tools endpoint logic when tools are available."""
        # Create mock tools
        def mock_web_search(query: str) -> dict:
            return {"query": query, "results": ["result1", "result2"]}

        def mock_weather_api(location: str) -> dict:
            return {"location": location, "temperature": "20°C"}

        mock_tools = [
            function_to_tool(mock_web_search, description="Search the web"),
            function_to_tool(mock_weather_api, description="Get weather"),
        ]

        # Call the endpoint function directly with mock tools (bypass auth for test)
        result = await list_tools(tools=mock_tools, _=True)

        assert "tools" in result
        expected_tools = ["mock_web_search", "mock_weather_api"]
        assert sorted(result["tools"]) == sorted(expected_tools)

    @pytest.mark.asyncio
    async def test__tools_endpoint__without_tools(self) -> None:
        """Test tools endpoint logic when no tools are available."""
        # Call the endpoint function directly with empty tools list (bypass auth for test)
        result = await list_tools(tools=[], _=True)

        assert result == {"tools": []}


class TestCORSAndMiddleware:
    """Test CORS and middleware functionality."""

    def test__cors_headers__are_present(self) -> None:
        """Test that CORS headers are properly set."""
        with TestClient(app) as client:
            response = client.options("/v1/models", headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            })

            # Should allow the request
            assert "access-control-allow-origin" in response.headers

    def test__preflight_requests__are_handled(self) -> None:
        """Test that preflight requests are handled correctly."""
        with TestClient(app) as client:
            response = client.options("/v1/chat/completions", headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            })

            assert response.status_code == 200
            assert "access-control-allow-methods" in response.headers


class TestOpenAICompatibility:
    """
    Test OpenAI API compatibility.

    Note: Non-streaming compatibility tests were removed as the API now only streams.
    Streaming compatibility is tested in
    TestChatCompletionsEndpoint.test__streaming_chat_completion__success.
    """


@pytest.mark.integration
class TestOpenAISDKCompatibility:
    """Test that our API works with the official OpenAI SDK."""

    @pytest_asyncio.fixture
    async def openai_client(self) -> AsyncGenerator[AsyncOpenAI]:
        """Start real server and return OpenAI SDK client pointing to it."""
        # Find free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]

        # Start real server process with auth disabled for testing
        server_process = subprocess.Popen(  # noqa: ASYNC220
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
        for _ in range(50):
            try:
                async with httpx.AsyncClient() as test_client:
                    response = await test_client.get(f"http://localhost:{port}/health")
                    if response.status_code == 200:
                        break
            except Exception:
                pass
            await asyncio.sleep(0.1)
        else:
            server_process.terminate()
            raise RuntimeError("Server failed to start")

        # Create OpenAI SDK client pointing to our real server
        client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=f"http://localhost:{port}/v1",
        )

        try:
            yield client
        finally:
            server_process.terminate()
            server_process.wait()

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set",
    )
    @pytest.mark.asyncio
    async def test__openai_sdk_streaming_chat_completion(
        self, openai_client: AsyncOpenAI,
    ) -> None:
        """
        Test streaming chat completion using OpenAI SDK with reasoning path.

        Routes to reasoning path to get reasoning events injected into the stream.
        """
        client = openai_client
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Repeat back exactly 'Hello from streaming test!'",
                },
            ],
            max_tokens=50,
            temperature=0.0,
            stream=True,
            extra_headers={"X-Routing-Mode": "reasoning"},
        )

        chunks = []
        content_parts = []
        reasoning_events = []

        async for chunk in stream:
            chunks.append(chunk)

            # Validate chunk structure
            assert chunk.id.startswith("chatcmpl-")
            assert chunk.object == "chat.completion.chunk"
            assert chunk.model.startswith("gpt-4o-mini")

            # OpenAI can return chunks with empty choices (just usage data)
            if not chunk.choices:
                continue

            assert len(chunk.choices) == 1

            choice = chunk.choices[0]
            assert choice.index == 0

            # Check for reasoning events in delta
            if hasattr(choice.delta, 'reasoning_event') and choice.delta.reasoning_event:
                reasoning_events.append(choice.delta.reasoning_event)

            # Collect actual content for final response
            if choice.delta.content:
                content_parts.append(choice.delta.content)

            # Check for finish reason in final chunks
            if choice.finish_reason:
                assert choice.finish_reason == "stop"

        # Validate we received chunks
        assert len(chunks) > 1, "Should receive multiple chunks in streaming mode"

        # Validate reasoning events were injected
        assert len(reasoning_events) > 0, "Should have reasoning events in delta"

        # Validate reasoning event structure (OpenAI SDK deserializes as dict)
        first_reasoning_event = reasoning_events[0]
        if hasattr(first_reasoning_event, 'type'):
            # Pydantic model access
            assert first_reasoning_event.type
            assert first_reasoning_event.step_iteration
            # No status field in new architecture
            assert first_reasoning_event.metadata
        else:
            # Dictionary access (OpenAI SDK deserialization)
            assert 'type' in first_reasoning_event
            assert 'step_iteration' in first_reasoning_event
            # No status field in new architecture
            assert 'metadata' in first_reasoning_event

        # Validate content was received
        full_content = "".join(content_parts)
        assert "hello from streaming test" in full_content.lower()

        # Validate that usage information is present (if available)
        usage_chunks = [chunk for chunk in chunks if chunk.usage is not None]
        if usage_chunks:
            # If usage is provided, validate it
            assert usage_chunks[0].usage.total_tokens > 0


class TestOpenAISDKCompatibilityUnit:
    """
    Unit tests for OpenAI SDK compatibility using TestClient.

    Note: Non-streaming SDK compatibility tests were removed as the API now only streams.
    SDK streaming compatibility is tested in
    TestOpenAISDKCompatibility.test__openai_sdk_streaming_chat_completion.
    """

    def test__models_endpoint_sdk_compatibility(self) -> None:
        """Test that models endpoint returns SDK-compatible format."""
        with TestClient(app) as client:
            response = client.get("/v1/models")

            assert response.status_code == 200
            data = response.json()

            # Should match OpenAI SDK expectations
            assert data["object"] == "list"
            assert "data" in data
            assert len(data["data"]) >= 2

            # Each model should have SDK-expected fields
            for model in data["data"]:
                assert "id" in model
                assert "object" in model
                assert "created" in model
                assert "owned_by" in model
                assert model["object"] == "model"

