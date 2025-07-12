"""
API endpoint tests for the FastAPI application.

Tests the FastAPI routes with mocked ReasoningAgent to verify
proper integration and error handling.
"""

import json
from collections.abc import AsyncGenerator

import respx
import httpx
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import asyncio

from api.main import app
from api.models import ChatCompletionResponse, Choice, ChatMessage, Usage, MessageRole
from api.reasoning_agent import ReasoningAgent
from api.config import settings
from typing import Never


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

    @respx.mock
    def test__non_streaming_chat_completion__success(self) -> None:
        """Test successful non-streaming chat completion."""
        # Create a fresh HTTP client for this test
        test_client = httpx.AsyncClient()

        # Patch the global reasoning agent with a fresh one
        test_reasoning_agent = ReasoningAgent(
            base_url=settings.reasoning_agent_base_url,
            api_key=settings.openai_api_key,
            http_client=test_client,
            mcp_client=None,
        )

        patcher = patch('api.main.reasoning_agent', test_reasoning_agent)
        patcher.start()

        try:
            # Mock OpenAI API response
            mock_openai_response = {
                "id": "chatcmpl-test123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello! How can I help you today?",
                        },
                        "finish_reason": "stop",
                    },
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18,
                },
            }

            respx.post("https://api.openai.com/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=mock_openai_response),
            )

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
            patcher.stop()
            # Clean up the test client
            asyncio.get_event_loop().run_until_complete(test_client.aclose())

    def test__streaming_chat_completion__success(self) -> None:
        """Test successful streaming chat completion."""
        # Mock the streaming response
        async def mock_stream() -> AsyncGenerator[str]:
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

        # Patch the method and ensure it cleans up properly
        patcher = patch('api.main.reasoning_agent')
        mock_agent = patcher.start()
        try:
            mock_agent.process_chat_completion_stream.return_value = mock_stream()

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
            patcher.stop()

    @respx.mock
    def test__chat_completion__forwards_openai_errors(self) -> None:
        """Test that OpenAI API errors are properly forwarded."""
        # Create a fresh HTTP client for this test to avoid closure issues
        test_client = httpx.AsyncClient()

        # Patch the global reasoning agent with a fresh one
        test_reasoning_agent = ReasoningAgent(
            base_url=settings.reasoning_agent_base_url,
            api_key=settings.openai_api_key,
            http_client=test_client,
            mcp_client=None,
        )

        patcher = patch('api.main.reasoning_agent', test_reasoning_agent)
        patcher.start()

        try:
            # Mock OpenAI API error
            error_response = {
                "error": {
                    "message": "Invalid API key provided",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                },
            }

            respx.post("https://api.openai.com/v1/chat/completions").mock(
                return_value=httpx.Response(401, json=error_response),
            )

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
            patcher.stop()
            # Clean up the test client
            asyncio.get_event_loop().run_until_complete(test_client.aclose())

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
        # Patch and ensure proper cleanup
        patcher = patch('api.main.reasoning_agent')
        mock_agent = patcher.start()
        try:
            # Mock an internal error - make it async
            async def async_error() -> Never:
                raise Exception("Internal error")
            mock_agent.process_chat_completion.return_value = async_error()

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
            patcher.stop()


class TestToolsEndpoint:
    """Test tools endpoint."""

    def test__tools_endpoint__with_mcp_client(self) -> None:
        """Test tools endpoint when MCP client is available."""
        with patch('api.main.mcp_client') as mock_mcp:
            # Mock the async methods properly
            mock_mcp.list_tools = AsyncMock(return_value={
                "test_server": ["web_search", "weather_api"],
            })
            mock_mcp.close = AsyncMock()

            with TestClient(app) as client:
                response = client.get("/tools")

                assert response.status_code == 200
                data = response.json()
                assert "test_server" in data

    def test__tools_endpoint__without_mcp_client(self) -> None:
        """Test tools endpoint when MCP client is not available."""
        with patch('api.main.mcp_client', None):
            with TestClient(app) as client:
                response = client.get("/tools")

                assert response.status_code == 200
                data = response.json()
                assert data == {"tools": []}


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

    @respx.mock
    def test__request_format__matches_openai_exactly(self) -> None:
        """Test that our request format matches OpenAI's expectations."""
        # Create a fresh HTTP client for this test
        test_client = httpx.AsyncClient()

        # Patch the global reasoning agent with a fresh one
        test_reasoning_agent = ReasoningAgent(
            base_url=settings.reasoning_agent_base_url,
            api_key=settings.openai_api_key,
            http_client=test_client,
            mcp_client=None,
        )

        patcher = patch('api.main.reasoning_agent', test_reasoning_agent)
        patcher.start()

        try:
            # Capture the actual request sent to OpenAI
            mock_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
                return_value=httpx.Response(200, json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "test"},
                            "finish_reason": "stop",
                        },
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }),
            )

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
                assert mock_route.called

                # Verify the request sent to OpenAI matches our input exactly
                openai_request = json.loads(mock_route.calls[0].request.content)
                assert openai_request["model"] == "gpt-4o"
                assert openai_request["temperature"] == 0.7
                assert openai_request["max_tokens"] == 100
                assert openai_request["stream"] is False  # Should be explicitly set
                assert len(openai_request["messages"]) == 2
        finally:
            patcher.stop()
            # Clean up the test client
            asyncio.get_event_loop().run_until_complete(test_client.aclose())

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

        patcher = patch('api.main.reasoning_agent')
        mock_agent = patcher.start()
        try:
            # Make the mock return an awaitable
            async def async_mock() -> ChatCompletionResponse:
                return mock_response
            mock_agent.process_chat_completion.return_value = async_mock()

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
            patcher.stop()
