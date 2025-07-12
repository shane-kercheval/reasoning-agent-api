"""
Reasoning agent tests for the stubbed implementation.

Tests the basic stub functionality of the reasoning agent.
"""

import pytest

from api.mcp_client import MCPClient
from api.reasoning_agent import ReasoningAgent
from api.models import ChatCompletionRequest, ChatMessage, MessageRole


@pytest.mark.asyncio
async def test_process_request_returns_unchanged(mock_mcp_client: MCPClient):
    """Test that process_request returns the request unchanged."""
    agent = ReasoningAgent(mock_mcp_client)

    request = ChatCompletionRequest(
        model="gpt-4o",
        messages=[
            ChatMessage(role=MessageRole.USER, content="Hello, world!"),
        ],
    )

    result = await agent.process_request(request)

    # Should return the exact same request
    assert result == request
    assert result.messages == request.messages
    assert result.model == request.model


@pytest.mark.asyncio
async def test_process_streaming_request_basic_flow(mock_mcp_client: MCPClient):
    """Test that streaming request provides basic progress updates."""
    agent = ReasoningAgent(mock_mcp_client)

    request = ChatCompletionRequest(
        model="gpt-4o",
        messages=[
            ChatMessage(role=MessageRole.USER, content="Test message"),
        ],
        stream=True,
    )

    updates = []
    async for update in agent.process_streaming_request(request):
        updates.append(update)

    # Should have exactly 3 updates: processing, ready, enhanced_request
    assert len(updates) == 3

    # Check first update
    assert updates[0]["type"] == "reasoning_step"
    assert "Processing request" in updates[0]["content"]

    # Check second update
    assert updates[1]["type"] == "reasoning_step"
    assert "Ready to generate response" in updates[1]["content"]

    # Check final update
    assert updates[2]["type"] == "enhanced_request"
    assert updates[2]["request"] == request


@pytest.mark.asyncio
async def test_different_request_types_unchanged(mock_mcp_client: MCPClient):
    """Test that different request types are returned unchanged."""
    agent = ReasoningAgent(mock_mcp_client)

    # Test with different parameters
    request = ChatCompletionRequest(
        model="gpt-3.5-turbo",
        messages=[
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(role=MessageRole.USER, content="What's 2+2?"),
        ],
        temperature=0.5,
        max_tokens=100,
    )

    result = await agent.process_request(request)

    # Should be identical
    assert result.model == "gpt-3.5-turbo"
    assert len(result.messages) == 2
    assert result.temperature == 0.5
    assert result.max_tokens == 100


@pytest.mark.asyncio
async def test_mcp_client_not_used_in_stub(mock_mcp_client: MCPClient):
    """Test that the MCP client is not used in the stub implementation."""
    agent = ReasoningAgent(mock_mcp_client)

    request = ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role=MessageRole.USER, content="Test")],
    )

    # Process request
    await agent.process_request(request)

    # MCP client should not have been called
    mock_mcp_client.call_tool.assert_not_called()
    mock_mcp_client.list_tools.assert_not_called()
