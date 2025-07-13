"""
API endpoint tests for the FastAPI application - Refactored with Dependency Injection.

Tests the FastAPI routes using clean dependency injection patterns instead of
complex global state mocking. This demonstrates the power of FastAPI's DI system.
"""

import asyncio
import json
import os
import socket
import subprocess
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from unittest.mock import patch
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from api.main import app
from api.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChatMessage,
    MessageRole,
    Usage,
)
from api.dependencies import get_reasoning_agent
from api.main import list_tools
from api.mcp import ToolInfo
from api.auth import verify_token

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
        mock_response = ChatCompletionResponse(
            id="chatcmpl-test123",
            object="chat.completion",
            created=1234567890,
            model="gpt-4o",
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="Hello! How can I help you today?",
                    ),
                    finish_reason="stop",
                ),
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=8, total_tokens=18),
        )
        mock_agent.process_chat_completion.return_value = mock_response

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
        # Mock the streaming response
        async def mock_stream(request: ChatCompletionRequest) -> AsyncGenerator[str]:  # noqa: ARG001
            yield "data: " + json.dumps({
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "delta": {"content": "Analyzing request..."},
                    "finish_reason": None,
                }],
            }) + "\n\n"
            yield "data: " + json.dumps({
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": None,
                }],
            }) + "\n\n"
            # Final chunk with finish_reason set
            yield "data: " + json.dumps({
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }) + "\n\n"
            yield "data: [DONE]\n\n"

        # Create mock reasoning agent with streaming support
        mock_agent = AsyncMock()
        mock_agent.process_chat_completion_stream = mock_stream

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
        mock_agent.process_chat_completion.side_effect = mock_error

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
        mock_agent.process_chat_completion.side_effect = Exception("Internal error")

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
    async def test__tools_endpoint__with_mcp_manager(self) -> None:
        """Test tools endpoint logic when MCP manager has tools available."""
        # Mock MCP manager with tools
        mock_manager = AsyncMock()
        mock_tools = [
            ToolInfo(
                server_name="test_server",
                tool_name="web_search",
                description="Search the web",
                input_schema={},
            ),
            ToolInfo(
                server_name="test_server",
                tool_name="weather_api",
                description="Get weather",
                input_schema={},
            ),
        ]
        mock_manager.get_available_tools = AsyncMock(return_value=mock_tools)

        # Mock the get_mcp_manager function directly
        with patch('api.main.get_mcp_manager') as mock_get_manager:
            mock_get_manager.return_value = mock_manager

            # Call the endpoint function directly (bypass auth for test)
            result = await list_tools(_=True)

            assert "test_server" in result
            assert result["test_server"] == ["web_search", "weather_api"]

    @pytest.mark.asyncio
    async def test__tools_endpoint__without_mcp_manager(self) -> None:
        """Test tools endpoint logic when MCP manager has no tools available."""
        # Mock MCP manager with no tools
        mock_manager = AsyncMock()
        mock_manager.get_available_tools = AsyncMock(return_value=[])

        # Mock the get_mcp_manager function directly
        with patch('api.main.get_mcp_manager') as mock_get_manager:
            mock_get_manager.return_value = mock_manager

            # Call the endpoint function directly (bypass auth for test)
            result = await list_tools(_=True)

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
        mock_response = ChatCompletionResponse(
            id="chatcmpl-test",
            object="chat.completion",
            created=1234567890,
            model="gpt-4o",
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role=MessageRole.ASSISTANT, content="test"),
                    finish_reason="stop",
                ),
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        mock_agent.process_chat_completion.return_value = mock_response

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
                mock_agent.process_chat_completion.assert_called_once()
                # Get the ChatCompletionRequest
                call_args = mock_agent.process_chat_completion.call_args[0][0]
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
        mock_response = ChatCompletionResponse(
            id="chatcmpl-real-openai-id",
            object="chat.completion",
            created=1234567890,
            model="gpt-4o",
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role=MessageRole.ASSISTANT, content="OpenAI response"),
                    finish_reason="stop",
                ),
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        mock_agent = AsyncMock()
        mock_agent.process_chat_completion.return_value = mock_response

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


class TestOpenAISDKCompatibilityUnit:
    """Unit tests for OpenAI SDK compatibility using TestClient."""

    def test__sdk_like_request_structure(self) -> None:
        """Test that our API accepts SDK-like requests."""
        mock_agent = AsyncMock()
        mock_response = ChatCompletionResponse(
            id="chatcmpl-test",
            object="chat.completion",
            created=1234567890,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role=MessageRole.ASSISTANT, content="Hello there!"),
                    finish_reason="stop",
                ),
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        mock_agent.process_chat_completion.return_value = mock_response

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

