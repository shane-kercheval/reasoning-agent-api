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
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from openai import AsyncOpenAI

from reasoning_api.auth import verify_token
from reasoning_api.config import settings
from reasoning_api.dependencies import get_conversation_db, get_prompt_manager, get_tools
from reasoning_api.main import app
from reasoning_api.openai_protocol import SSE_DONE
from reasoning_api.prompt_manager import PromptManager
from tests.integration_tests.litellm_mocks import mock_direct_answer

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

    def test__models_endpoint__returns_available_models(self, respx_mock) -> None:  # type: ignore[misc]  # noqa: ANN001
        """Test that models endpoint returns available models from LiteLLM."""
        # Mock LiteLLM's /v1/model/info endpoint response
        respx_mock.get(f"{settings.llm_base_url}/v1/model/info").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "model_name": "gpt-4o",
                            "model_info": {
                                "litellm_provider": "openai",
                                "max_input_tokens": 128000,
                                "max_output_tokens": 16384,
                                "input_cost_per_token": 0.0000025,
                                "output_cost_per_token": 0.00001,
                                "supports_reasoning": False,
                                "supports_response_schema": True,
                                "supports_vision": True,
                                "supports_function_calling": True,
                                "supports_web_search": None,
                            },
                        },
                        {
                            "model_name": "gpt-4o-mini",
                            "model_info": {
                                "litellm_provider": "openai",
                                "max_input_tokens": 128000,
                                "max_output_tokens": 16384,
                                "input_cost_per_token": 0.00000015,
                                "output_cost_per_token": 0.0000006,
                                "supports_reasoning": False,
                                "supports_response_schema": True,
                                "supports_vision": True,
                                "supports_function_calling": True,
                                "supports_web_search": None,
                            },
                        },
                        {
                            "model_name": "claude-3-5-sonnet-20241022",
                            "model_info": {
                                "litellm_provider": "anthropic",
                                "max_input_tokens": 200000,
                                "max_output_tokens": 8192,
                                "input_cost_per_token": 0.000003,
                                "output_cost_per_token": 0.000015,
                                "supports_reasoning": False,
                                "supports_response_schema": True,
                                "supports_vision": True,
                                "supports_function_calling": True,
                                "supports_web_search": None,
                            },
                        },
                    ],
                },
            ),
        )

        with TestClient(app) as client:
            response = client.get("/v1/models")

            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "list"
            assert len(data["data"]) == 3

            model_ids = [model["id"] for model in data["data"]]
            assert "gpt-4o" in model_ids
            assert "gpt-4o-mini" in model_ids
            assert "claude-3-5-sonnet-20241022" in model_ids

            # Verify owned_by is set correctly from litellm_provider
            models_by_id = {model["id"]: model for model in data["data"]}
            assert models_by_id["gpt-4o"]["owned_by"] == "openai"
            assert models_by_id["claude-3-5-sonnet-20241022"]["owned_by"] == "anthropic"

            # Verify new fields are present
            assert models_by_id["gpt-4o"]["max_input_tokens"] == 128000
            assert models_by_id["gpt-4o"]["max_output_tokens"] == 16384
            assert models_by_id["gpt-4o"]["input_cost_per_token"] == 0.0000025
            assert models_by_id["gpt-4o"]["supports_vision"] is True

    def test__models_endpoint__includes_supports_reasoning_field(self, respx_mock) -> None:  # type: ignore[misc]  # noqa: ANN001
        """Test that models endpoint includes supports_reasoning field."""
        # Mock LiteLLM's /v1/model/info endpoint response
        respx_mock.get(f"{settings.llm_base_url}/v1/model/info").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "model_name": "o3-mini",
                            "model_info": {
                                "litellm_provider": "openai",
                                "max_input_tokens": 200000,
                                "max_output_tokens": 100000,
                                "input_cost_per_token": 0.000001,
                                "output_cost_per_token": 0.000004,
                                "supports_reasoning": True,
                                "supports_response_schema": True,
                                "supports_vision": True,
                                "supports_function_calling": True,
                                "supports_web_search": None,
                            },
                        },
                        {
                            "model_name": "gpt-4o",
                            "model_info": {
                                "litellm_provider": "openai",
                                "max_input_tokens": 128000,
                                "max_output_tokens": 16384,
                                "input_cost_per_token": 0.0000025,
                                "output_cost_per_token": 0.00001,
                                "supports_reasoning": False,
                                "supports_response_schema": True,
                                "supports_vision": True,
                                "supports_function_calling": True,
                                "supports_web_search": None,
                            },
                        },
                        {
                            "model_name": "claude-3-7-sonnet-20250219",
                            "model_info": {
                                "litellm_provider": "anthropic",
                                "max_input_tokens": 200000,
                                "max_output_tokens": 64000,
                                "input_cost_per_token": 0.000003,
                                "output_cost_per_token": 0.000015,
                                "supports_reasoning": True,
                                "supports_response_schema": True,
                                "supports_vision": True,
                                "supports_function_calling": True,
                                "supports_web_search": None,
                            },
                        },
                    ],
                },
            ),
        )

        with TestClient(app) as client:
            response = client.get("/v1/models")

            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "list"
            assert len(data["data"]) == 3

            # Find specific models and check supports_reasoning
            models_by_id = {model["id"]: model for model in data["data"]}

            # Reasoning models should have supports_reasoning=True
            assert models_by_id["o3-mini"]["supports_reasoning"] is True
            assert models_by_id["claude-3-7-sonnet-20250219"]["supports_reasoning"] is True

            # Non-reasoning models should have supports_reasoning=False
            assert models_by_id["gpt-4o"]["supports_reasoning"] is False

    def test__models_endpoint__forwards_litellm_errors(self, respx_mock) -> None:  # type: ignore[misc]  # noqa: ANN001
        """Test that LiteLLM errors are properly forwarded."""
        # Mock LiteLLM returning 503 Service Unavailable
        respx_mock.get(f"{settings.llm_base_url}/v1/model/info").mock(
            return_value=httpx.Response(
                503,
                json={
                    "error": {
                        "message": "Service temporarily unavailable",
                        "type": "service_error",
                    },
                },
            ),
        )

        with TestClient(app) as client:
            response = client.get("/v1/models")

            # Should forward the 503 status
            assert response.status_code == 503
            data = response.json()
            # FastAPI wraps in detail
            assert "detail" in data
            assert "error" in data["detail"]
            assert data["detail"]["error"]["type"] == "upstream_error"

    def test__models_endpoint__handles_connection_errors(self, respx_mock) -> None:  # type: ignore[misc]  # noqa: ANN001
        """Test that connection errors return 503."""
        # Mock connection error
        respx_mock.get(f"{settings.llm_base_url}/v1/model/info").mock(
            side_effect=httpx.ConnectError("Connection refused"),
        )

        with TestClient(app) as client:
            response = client.get("/v1/models")

            # Should return 503 Service Unavailable
            assert response.status_code == 503
            data = response.json()
            # FastAPI wraps in detail
            assert "detail" in data
            assert "error" in data["detail"]
            assert data["detail"]["error"]["type"] == "service_unavailable"


class TestChatCompletionsEndpoint:
    """Test chat completions endpoint."""

    def test__streaming_chat_completion__success(self) -> None:
        """Test successful streaming chat completion with reasoning mode."""
        # Mock dependencies
        mock_tools = []
        mock_prompt_manager = AsyncMock()
        mock_prompt_manager.get_prompt.return_value = "You are a helpful assistant."

        # Use FastAPI dependency override
        app.dependency_overrides[get_tools] = lambda: mock_tools
        app.dependency_overrides[get_prompt_manager] = lambda: mock_prompt_manager

        # Mock LiteLLM (the external dependency) instead of business logic
        # Configure it to return a simple reasoning step followed by streaming response
        mock_litellm = mock_direct_answer("Hello! How can I help you today?")

        with patch("reasoning_api.executors.reasoning_agent.litellm.acompletion", side_effect=mock_litellm):
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
                    # Should contain the mocked response content
                    assert "Hello" in content or "help" in content
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

        # Mock litellm.acompletion to raise the error
        async def mock_litellm_with_error(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202, ARG001
            raise mock_error

        try:
            with patch(
                "reasoning_api.executors.reasoning_agent.litellm.acompletion",
                side_effect=mock_litellm_with_error,
            ), TestClient(app, raise_server_exceptions=False) as client:
                request_data = {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "stream": True,
                }

                # Make request with reasoning mode to trigger the error
                response = client.post(
                    "/v1/chat/completions",
                    json=request_data,
                    headers={"X-Routing-Mode": "reasoning"},
                )

                # Streaming errors result in either:
                # 1. 500 error (if error before streaming starts)
                # 2. 200 with incomplete stream (if error during streaming)
                assert response.status_code in [200, 500]

                # If status is 200, stream should be incomplete/empty
                if response.status_code == 200:
                    content = response.text
                    assert content == "" or len(content) < 100  # Incomplete stream
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
        # Mock litellm.acompletion to raise an internal error
        async def mock_litellm_with_error(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202, ARG001
            raise Exception("Internal server error during LLM call")

        try:
            with patch(
                "reasoning_api.executors.reasoning_agent.litellm.acompletion",
                side_effect=mock_litellm_with_error,
            ), TestClient(app, raise_server_exceptions=False) as client:
                request_data = {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "stream": True,
                }

                # Make request with reasoning mode to trigger the error
                response = client.post(
                    "/v1/chat/completions",
                    json=request_data,
                    headers={"X-Routing-Mode": "reasoning"},
                )

                # Streaming errors result in either:
                # 1. 500 error (if error before streaming starts)
                # 2. 200 with incomplete stream (if error during streaming)
                assert response.status_code in [200, 500]

                # If status is 200, stream should be incomplete/empty
                if response.status_code == 200:
                    content = response.text
                    assert content == "" or len(content) < 100  # Incomplete stream
        finally:
            app.dependency_overrides.clear()


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
@pytest.mark.e2e
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
                "uv", "run", "uvicorn", "reasoning_api.main:app",
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

    @pytest.mark.asyncio
    async def test__openai_sdk_streaming_chat_completion(self) -> None:
        """
        Test streaming chat completion using OpenAI SDK.

        Uses ASGI transport to test OpenAI SDK compatibility without subprocess.
        Tests passthrough mode with mocked LiteLLM to validate SDK streaming works.
        """
        # Create mock tools and dependencies (empty tools list is fine for this test)
        app.dependency_overrides[get_tools] = lambda: []
        mock_prompt_manager = AsyncMock(spec=PromptManager)
        mock_prompt_manager.get_prompt.return_value = "Test reasoning system prompt"
        app.dependency_overrides[get_prompt_manager] = lambda: mock_prompt_manager
        app.dependency_overrides[get_conversation_db] = lambda: None  # Stateless mode

        # Create mock LiteLLM chunks with model_dump() that returns correct format
        def create_mock_chunk(chunk_data: dict) -> MagicMock:
            """Create a mock chunk that returns correct dict from model_dump()."""
            mock = MagicMock()
            mock.model_dump.return_value = chunk_data
            # Also set attributes for usage tracking in passthrough
            mock.usage = chunk_data.get('usage')
            return mock

        # Create mock LiteLLM streaming response
        async def mock_litellm_stream() -> AsyncGenerator[MagicMock]:
            # First chunk: role
            yield create_mock_chunk({
                "id": "chatcmpl-test123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "gpt-4o-mini",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }],
            })
            # Second chunk: content
            yield create_mock_chunk({
                "id": "chatcmpl-test123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "gpt-4o-mini",
                "choices": [{
                    "index": 0,
                    "delta": {"content": "Test content"},
                    "finish_reason": None,
                }],
            })
            # Final chunk: finish
            yield create_mock_chunk({
                "id": "chatcmpl-test123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "gpt-4o-mini",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            })

        try:
            # Patch LiteLLM for passthrough path
            with patch('reasoning_api.executors.passthrough.litellm.acompletion') as mock_acompletion:
                # Configure mock to return our stream
                mock_acompletion.return_value = mock_litellm_stream()

                # Create ASGI transport client
                async with AsyncClient(
                    transport=ASGITransport(app=app),
                    base_url="http://test",
                ) as httpx_client:
                    # Create OpenAI SDK client using ASGI transport
                    client = AsyncOpenAI(
                        api_key="test-key",
                        base_url="http://test/v1",
                        http_client=httpx_client,
                    )

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
                        extra_headers={"X-Routing-Mode": "passthrough"},
                    )

                    chunks = []
                    content_parts = []
                    reasoning_events = []

                    async for chunk in stream:
                        chunks.append(chunk)

                        # Validate chunk structure
                        assert chunk.id == "chatcmpl-test123"
                        assert chunk.object == "chat.completion.chunk"
                        assert chunk.model == "gpt-4o-mini"

                        # OpenAI can return chunks with empty choices (just usage data)
                        if not chunk.choices:
                            continue

                        assert len(chunk.choices) == 1

                        choice = chunk.choices[0]
                        assert choice.index == 0

                        # Check for reasoning events in delta
                        if (
                            hasattr(choice.delta, 'reasoning_event')
                            and choice.delta.reasoning_event
                        ):
                            reasoning_events.append(choice.delta.reasoning_event)

                        # Collect actual content for final response
                        if choice.delta.content:
                            content_parts.append(choice.delta.content)

                        # Check for finish reason in final chunks
                        if choice.finish_reason:
                            assert choice.finish_reason == "stop"

                    # Validate we received chunks
                    assert len(chunks) > 1, "Should receive multiple chunks in streaming mode"

                    # Validate content was received
                    full_content = "".join(content_parts)
                    assert full_content == "Test content"  # From our mock stream

        finally:
            # Cleanup dependency overrides
            app.dependency_overrides.clear()
