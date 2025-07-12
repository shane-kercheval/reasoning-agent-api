"""
Comprehensive tests for the ReasoningAgent class.

Tests the ReasoningAgent proxy functionality, dependency injection,
error handling, and OpenAI API compatibility.
"""

import json
from typing import Any

import pytest
import httpx
import respx

from api.reasoning_agent import ReasoningAgent
from api.models import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageRole
from tests.conftest import OPENAI_TEST_MODEL


class TestReasoningAgentInitialization:
    """Test ReasoningAgent initialization and configuration."""

    @pytest.mark.asyncio
    async def test__init__sets_attributes_correctly(self) -> None:
        """Test that __init__ sets all attributes correctly."""
        async with httpx.AsyncClient() as client:
            agent = ReasoningAgent(
                base_url="https://api.openai.com/v1",
                api_key="test-key",
                http_client=client,
                mcp_client=None,
            )

            assert agent.base_url == "https://api.openai.com/v1"
            assert agent.api_key == "test-key"
            assert agent.http_client == client
            assert agent.mcp_client is None

    @pytest.mark.asyncio
    async def test__init__strips_trailing_slash_from_base_url(self) -> None:
        """Test that trailing slash is stripped from base_url."""
        async with httpx.AsyncClient() as client:
            agent = ReasoningAgent(
                base_url="https://api.openai.com/v1/",
                api_key="test-key",
                http_client=client,
            )

            assert agent.base_url == "https://api.openai.com/v1"

    @pytest.mark.asyncio
    async def test__init__sets_authorization_header_if_missing(self) -> None:
        """Test that Authorization header is set if not present."""
        async with httpx.AsyncClient() as client:
            ReasoningAgent(
                base_url="https://api.openai.com/v1",
                api_key="test-key",
                http_client=client,
            )

            assert client.headers["Authorization"] == "Bearer test-key"

    @pytest.mark.asyncio
    async def test__init__preserves_existing_authorization_header(self) -> None:
        """Test that existing Authorization header is preserved."""
        async with httpx.AsyncClient(headers={"Authorization": "Bearer existing-key"}) as client:
            ReasoningAgent(
                base_url="https://api.openai.com/v1",
                api_key="test-key",
                http_client=client,
            )

            assert client.headers["Authorization"] == "Bearer existing-key"


class TestProcessChatCompletion:
    """Test non-streaming chat completion processing."""

    @pytest.mark.asyncio
    @respx.mock
    async def test__process_chat_completion__success(
        self,
        reasoning_agent: ReasoningAgent,
        sample_chat_request: ChatCompletionRequest,
        mock_openai_response: dict[str, Any],
    ) -> None:
        """Test successful non-streaming chat completion."""
        # Mock OpenAI API response
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_openai_response),
        )

        result = await reasoning_agent.process_chat_completion(sample_chat_request)

        assert isinstance(result, ChatCompletionResponse)
        assert result.id == "chatcmpl-test123"
        assert result.model == "gpt-4o"
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "This is a test response from OpenAI."

    @pytest.mark.asyncio
    @respx.mock
    async def test__process_chat_completion__forwards_request_correctly(
        self,
        reasoning_agent: ReasoningAgent,
        sample_chat_request: ChatCompletionRequest,
    ) -> None:
        """Test that request is forwarded to OpenAI with correct payload."""
        mock_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "test"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }),
        )

        await reasoning_agent.process_chat_completion(sample_chat_request)

        # Check that request was made correctly
        assert mock_route.called
        request_data = json.loads(mock_route.calls[0].request.content)
        assert request_data["model"] == "gpt-4o"
        assert request_data["stream"] is False
        assert len(request_data["messages"]) == 1
        assert request_data["messages"][0]["content"] == "What's the weather in Paris?"

    @pytest.mark.asyncio
    @respx.mock
    async def test__process_chat_completion__handles_openai_error(
        self,
        reasoning_agent: ReasoningAgent,
        sample_chat_request: ChatCompletionRequest,
        mock_openai_error_response: dict[str, Any],
    ) -> None:
        """Test that OpenAI API errors are properly raised."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(401, json=mock_openai_error_response),
        )

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await reasoning_agent.process_chat_completion(sample_chat_request)

        assert exc_info.value.response.status_code == 401

    @pytest.mark.asyncio
    @respx.mock
    async def test__process_chat_completion__handles_different_models(
        self, reasoning_agent: ReasoningAgent,
    ) -> None:
        """Test that different models are handled correctly."""
        request = ChatCompletionRequest(
            model=OPENAI_TEST_MODEL,
            messages=[ChatMessage(role=MessageRole.USER, content="Test")],
        )

        mock_response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": OPENAI_TEST_MODEL,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "test"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response),
        )

        result = await reasoning_agent.process_chat_completion(request)
        assert result.model == OPENAI_TEST_MODEL


class TestProcessChatCompletionStream:
    """Test streaming chat completion processing."""

    @pytest.mark.asyncio
    @respx.mock
    async def test__process_chat_completion_stream__includes_reasoning_steps(
        self,
        reasoning_agent: ReasoningAgent,
        sample_streaming_request: ChatCompletionRequest,
        mock_openai_streaming_chunks: list[str],
    ) -> None:
        """Test that streaming includes reasoning steps before OpenAI response."""
        # Mock streaming response
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                text="\n".join(mock_openai_streaming_chunks),
                headers={"content-type": "text/plain"},
            ),
        )

        chunks = []
        async for chunk in reasoning_agent.process_chat_completion_stream(sample_streaming_request):  # noqa: E501
            chunks.append(chunk)

        # Should have reasoning steps + separator + OpenAI chunks + [DONE]
        assert len(chunks) > 5  # At least 3 reasoning + separator + some OpenAI chunks + [DONE]

        # Check that we have reasoning steps
        reasoning_chunks = [c for c in chunks if "🔍" in c or "🤔" in c or "✅" in c]
        assert len(reasoning_chunks) >= 3

        # Check that we have the separator
        separator_chunks = [c for c in chunks if "---" in c]
        assert len(separator_chunks) >= 1

        # Check final chunk
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    @respx.mock
    async def test__process_chat_completion_stream__forwards_openai_chunks(
        self,
        reasoning_agent: ReasoningAgent,
        sample_streaming_request: ChatCompletionRequest,
        mock_openai_streaming_chunks: list[str],
    ) -> None:
        """Test that OpenAI chunks are properly forwarded with modified IDs."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                text="\n".join(mock_openai_streaming_chunks),
                headers={"content-type": "text/plain"},
            ),
        )

        chunks = []
        async for chunk in reasoning_agent.process_chat_completion_stream(sample_streaming_request):  # noqa: E501
            if chunk.startswith("data: {") and "chatcmpl-" in chunk:
                chunks.append(chunk)

        # Should have OpenAI-style chunks with modified completion IDs
        assert len(chunks) > 0
        for chunk in chunks:
            if "chatcmpl-" in chunk:
                # Extract JSON from chunk
                chunk_data = json.loads(chunk[6:])  # Remove "data: " prefix
                # ID should be modified to our format, not the original "chatcmpl-test123"
                assert chunk_data["id"].startswith("chatcmpl-")
                assert chunk_data["id"] != "chatcmpl-test123"

    @pytest.mark.asyncio
    @respx.mock
    async def test__process_chat_completion_stream__handles_streaming_errors(
        self,
        reasoning_agent: ReasoningAgent,
        sample_streaming_request: ChatCompletionRequest,
    ) -> None:
        """Test that streaming errors are properly handled."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(401, json={"error": {"message": "Unauthorized"}}),
        )

        with pytest.raises(httpx.HTTPStatusError):
            async for _ in reasoning_agent.process_chat_completion_stream(sample_streaming_request):  # noqa: E501
                pass


class TestEnhanceRequest:
    """Test request enhancement functionality."""

    @pytest.mark.asyncio
    async def test__enhance_request__returns_original_for_now(
        self,
        reasoning_agent: ReasoningAgent,
        sample_chat_request: ChatCompletionRequest,
    ) -> None:
        """Test that _enhance_request currently returns the original request."""
        result = await reasoning_agent._enhance_request(sample_chat_request)
        assert result == sample_chat_request
        assert result is sample_chat_request  # Should be the same object for now


class TestIntegration:
    """Integration tests with different configurations."""

    @pytest.mark.asyncio
    @respx.mock
    async def test__reasoning_agent_without_mcp__works_correctly(
        self,
        reasoning_agent_no_mcp: ReasoningAgent,
        sample_chat_request: ChatCompletionRequest,
        mock_openai_response: dict[str, Any],
    ) -> None:
        """Test that ReasoningAgent works without MCP client."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_openai_response),
        )

        result = await reasoning_agent_no_mcp.process_chat_completion(sample_chat_request)
        assert isinstance(result, ChatCompletionResponse)
        assert result.id == "chatcmpl-test123"

    @pytest.mark.asyncio
    async def test__different_base_urls__work_correctly(self) -> None:
        """Test that different base URLs work correctly."""
        async with httpx.AsyncClient() as client:
            agent = ReasoningAgent(
                base_url="https://custom-api.example.com/v1",
                api_key="test-key",
                http_client=client,
            )

            assert agent.base_url == "https://custom-api.example.com/v1"
