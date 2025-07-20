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
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChoice,
    OpenAIMessage,
    MessageRole,
    OpenAIUsage,
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

    def test__non_streaming_chat_completion__success(self) -> None:
        """Test successful non-streaming chat completion."""
        # Create a mock reasoning agent
        mock_agent = AsyncMock()
        mock_response = OpenAIChatResponse(
            id="chatcmpl-test123",
            object="chat.completion",
            created=1234567890,
            model="gpt-4o",
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIMessage(
                        role=MessageRole.ASSISTANT,
                        content="Hello! How can I help you today?",
                    ),
                    finish_reason="stop",
                ),
            ],
            usage=OpenAIUsage(prompt_tokens=10, completion_tokens=8, total_tokens=18),
        )
        mock_agent.execute.return_value = mock_response

        # Use FastAPI dependency override - much cleaner!
        app.dependency_overrides[get_reasoning_agent] = lambda: mock_agent

        try:
            with TestClient(app) as client:
                request_data = {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "user", "content": "Hello!"},
                    ],
                    "temperature": 0.7,
                }

                response = client.post("/v1/chat/completions", json=request_data)

                assert response.status_code == 200
                data = response.json()
                assert data["id"] == "chatcmpl-test123"
                assert data["model"] == "gpt-4o"
                assert len(data["choices"]) == 1
                assert data["choices"][0]["message"]["content"] == "Hello! How can I help you today?"  # noqa: E501
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()

    def test__streaming_chat_completion__success(self) -> None:
        """Test successful streaming chat completion."""
        # Mock the streaming response using the builder
        async def mock_stream(request: OpenAIChatRequest) -> AsyncGenerator[str]:  # noqa: ARG001
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
        mock_agent = AsyncMock()
        mock_agent.execute_stream = mock_stream

        # Use FastAPI dependency override
        app.dependency_overrides[get_reasoning_agent] = lambda: mock_agent

        try:
            with TestClient(app) as client:
                request_data = {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "stream": True,
                }

                response = client.post("/v1/chat/completions", json=request_data)

                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

                # Check that we get streaming data
                content = response.content.decode()
                # Check that we get reasoning step content
                assert "Analyzing request..." in content
                assert "data: [DONE]" in content
        finally:
            app.dependency_overrides.clear()

    def test__chat_completion__forwards_openai_errors(self) -> None:
        """Test that OpenAI API errors are properly forwarded."""
        # Mock agent to raise HTTPStatusError
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
        mock_agent.execute.side_effect = mock_error

        app.dependency_overrides[get_reasoning_agent] = lambda: mock_agent

        try:
            with TestClient(app) as client:
                request_data = {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello!"}],
                }

                response = client.post("/v1/chat/completions", json=request_data)

                assert response.status_code == 401
                data = response.json()
                # OpenAI error should be in detail field when returned via HTTPException
                assert "detail" in data
                assert "error" in data["detail"]
                assert data["detail"]["error"]["message"] == "Invalid API key provided"
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
        """Test that internal server errors are handled gracefully."""
        mock_agent = AsyncMock()
        mock_agent.execute.side_effect = Exception("Internal error")

        app.dependency_overrides[get_reasoning_agent] = lambda: mock_agent
        try:
            with TestClient(app) as client:
                request_data = {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello!"}],
                }

                response = client.post("/v1/chat/completions", json=request_data)

                assert response.status_code == 500
                data = response.json()
                # Internal error should be in detail field when returned via HTTPException
                assert "detail" in data
                assert "error" in data["detail"]
                assert data["detail"]["error"]["type"] == "internal_server_error"
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
    """Test OpenAI API compatibility."""

    def test__request_format__matches_openai_exactly(self) -> None:
        """Test that our request format matches OpenAI's expectations."""
        # Mock a simple successful response
        mock_agent = AsyncMock()
        mock_response = OpenAIChatResponse(
            id="chatcmpl-test",
            object="chat.completion",
            created=1234567890,
            model="gpt-4o",
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIMessage(role=MessageRole.ASSISTANT, content="test"),
                    finish_reason="stop",
                ),
            ],
            usage=OpenAIUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        mock_agent.execute.return_value = mock_response

        app.dependency_overrides[get_reasoning_agent] = lambda: mock_agent

        try:
            with TestClient(app) as client:
                request_data = {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "You are helpful"},
                        {"role": "user", "content": "Hello!"},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 100,
                    "top_p": 0.9,
                }

                response = client.post("/v1/chat/completions", json=request_data)

                assert response.status_code == 200

                # Verify the mock was called with the request - our API received it
                mock_agent.execute.assert_called_once()
                # Get the ChatCompletionRequest
                call_args = mock_agent.execute.call_args[0][0]
                assert call_args.model == "gpt-4o"
                assert call_args.temperature == 0.7
                assert call_args.max_tokens == 100
                assert call_args.stream is False  # Should be explicitly set
                assert len(call_args.messages) == 2
        finally:
            app.dependency_overrides.clear()

    def test__response_format__matches_openai_exactly(self) -> None:
        """Test that our response format matches OpenAI's exactly."""
        # This test ensures we don't modify the response structure
        # Mock a response that looks exactly like OpenAI's
        mock_response = OpenAIChatResponse(
            id="chatcmpl-real-openai-id",
            object="chat.completion",
            created=1234567890,
            model="gpt-4o",
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIMessage(role=MessageRole.ASSISTANT, content="OpenAI response"),
                    finish_reason="stop",
                ),
            ],
            usage=OpenAIUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        mock_agent = AsyncMock()
        mock_agent.execute.return_value = mock_response

        app.dependency_overrides[get_reasoning_agent] = lambda: mock_agent

        try:
            with TestClient(app) as client:
                request_data = {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello!"}],
                }

                response = client.post("/v1/chat/completions", json=request_data)

                assert response.status_code == 200
                data = response.json()

                # Should match OpenAI format exactly
                assert data["id"] == "chatcmpl-real-openai-id"
                assert data["object"] == "chat.completion"
                assert data["model"] == "gpt-4o"
                assert "choices" in data
                assert "usage" in data
                assert data["usage"]["total_tokens"] == 15
        finally:
            app.dependency_overrides.clear()


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
    async def test__openai_sdk_non_streaming_chat_completion(
        self, openai_client: AsyncOpenAI,
    ) -> None:
        """Test non-streaming chat completion using OpenAI SDK."""
        client = openai_client
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello from OpenAI SDK integration test!'"},
            ],
            max_tokens=50,
            temperature=0.0,
        )

        # Validate response structure
        assert response.id.startswith("chatcmpl-")
        assert response.object == "chat.completion"
        assert response.model.startswith("gpt-4o-mini")
        assert len(response.choices) == 1
        assert response.choices[0].message.role == "assistant"
        assert "Hello from OpenAI SDK integration test" in response.choices[0].message.content
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.total_tokens > 0

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set",
    )
    @pytest.mark.asyncio
    async def test__openai_sdk_streaming_chat_completion(
        self, openai_client: AsyncOpenAI,
    ) -> None:
        """Test streaming chat completion using OpenAI SDK."""
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
            assert first_reasoning_event.step_id
            assert first_reasoning_event.status
            assert first_reasoning_event.metadata
        else:
            # Dictionary access (OpenAI SDK deserialization)
            assert 'type' in first_reasoning_event
            assert 'step_id' in first_reasoning_event
            assert 'status' in first_reasoning_event
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
    """Unit tests for OpenAI SDK compatibility using TestClient."""

    def test__sdk_like_request_structure(self) -> None:
        """Test that our API accepts SDK-like requests."""
        mock_agent = AsyncMock()
        mock_response = OpenAIChatResponse(
            id="chatcmpl-test",
            object="chat.completion",
            created=1234567890,
            model="gpt-4o-mini",
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIMessage(role=MessageRole.ASSISTANT, content="Hello there!"),
                    finish_reason="stop",
                ),
            ],
            usage=OpenAIUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        mock_agent.execute.return_value = mock_response

        app.dependency_overrides[get_reasoning_agent] = lambda: mock_agent

        try:
            with TestClient(app) as client:
                # This mimics what the OpenAI SDK would send
                request_data = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Hello!"},
                    ],
                    "max_tokens": 50,
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "stream": False,
                }

                response = client.post("/v1/chat/completions", json=request_data)

                assert response.status_code == 200
                data = response.json()

                # Should match OpenAI SDK expectations
                assert data["id"] == "chatcmpl-test"
                assert data["object"] == "chat.completion"
                assert data["choices"][0]["message"]["role"] == "assistant"
                assert data["choices"][0]["message"]["content"] == "Hello there!"
                assert data["usage"]["total_tokens"] == 15
        finally:
            app.dependency_overrides.clear()

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

