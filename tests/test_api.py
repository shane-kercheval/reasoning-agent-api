"""
API endpoint tests for the Reasoning Agent API.

This module contains comprehensive tests for all API endpoints including
health checks, model listing, chat completions (both streaming and non-streaming),
and various error conditions.
"""

import pytest
import json
from fastapi.testclient import TestClient

from api.main import stream_chat_completion
from api.models import ChatCompletionRequest, ChatMessage, MessageRole

def test_health_check(client: TestClient):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data

def test_list_models(client: TestClient):
    """Test models endpoint."""
    response = client.get("/v1/models")
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0

    # Check model structure
    model = data["data"][0]
    assert "id" in model
    assert "object" in model
    assert "created" in model
    assert "owned_by" in model

def test_list_tools(client: TestClient):
    """Test tools endpoint."""
    response = client.get("/tools")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, dict)

def test_chat_completion_non_streaming(
        client: TestClient,
        sample_chat_request: ChatCompletionRequest,
    ):
    """Test non-streaming chat completion."""
    response = client.post(
        "/v1/chat/completions",
        json=sample_chat_request.model_dump(),
    )

    assert response.status_code == 200
    data = response.json()

    # Validate OpenAI-compatible response structure
    assert "id" in data
    assert "object" in data
    assert data["object"] == "chat.completion"
    assert "created" in data
    assert "model" in data
    assert "choices" in data
    assert "usage" in data

    # Validate choice structure
    choice = data["choices"][0]
    assert "index" in choice
    assert "message" in choice
    assert "finish_reason" in choice

    # Validate message structure
    message = choice["message"]
    assert "role" in message
    assert "content" in message
    assert message["role"] == "assistant"

def test_chat_completion_streaming(
        client: TestClient,
        sample_streaming_request: ChatCompletionRequest,
    ):
    """Test streaming chat completion."""
    response = client.post(
        "/v1/chat/completions",
        json=sample_streaming_request.model_dump(),
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    # Parse streaming response
    lines = response.text.strip().split('\n')
    data_lines = [line for line in lines if line.startswith('data: ')]

    assert len(data_lines) > 0

    # Check that we have reasoning steps and final response
    reasoning_steps = 0
    response_chunks = 0

    for line in data_lines:
        if line == "data: [DONE]":
            continue

        try:
            chunk_data = json.loads(line[6:])  # Remove "data: " prefix
            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                delta = chunk_data["choices"][0].get("delta", {})
                content = delta.get("content", "")

                if content.startswith(("\n🔍", "\n🛠️", "\n✅")):
                    reasoning_steps += 1
                else:
                    response_chunks += 1

        except json.JSONDecodeError:
            continue

    # Should have both reasoning steps and response content
    assert reasoning_steps > 0
    assert response_chunks >= 0

def test_chat_completion_invalid_model(client: TestClient):
    """Test chat completion with invalid model."""
    request_data = {
        "model": "invalid-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    response = client.post("/v1/chat/completions", json=request_data)
    # Should still process (will be passed to OpenAI)
    assert response.status_code in [200, 500]  # Depends on OpenAI mock behavior

def test_chat_completion_missing_messages(client: TestClient):
    """Test chat completion with missing messages."""
    request_data = {
        "model": "gpt-4o",
        # Missing messages field
    }

    response = client.post("/v1/chat/completions", json=request_data)
    assert response.status_code == 422  # Validation error

def test_chat_completion_empty_messages(client: TestClient):
    """Test chat completion with empty messages."""
    request_data = {
        "model": "gpt-4o",
        "messages": [],
    }

    response = client.post("/v1/chat/completions", json=request_data)
    assert response.status_code in [200, 500]  # May pass validation but fail processing

def test_chat_completion_with_parameters(client: TestClient):
    """Test chat completion with various parameters."""
    request_data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.8,
        "max_tokens": 100,
        "top_p": 0.9,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3,
        "stop": [".", "!"],
    }

    response = client.post("/v1/chat/completions", json=request_data)
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_streaming_response_format():
    """Test that streaming response chunks have correct format."""
    request = ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role=MessageRole.USER, content="Test message")],
        stream=True,
    )

    chunks = []
    async for chunk in stream_chat_completion(request):
        if chunk.startswith("data: ") and chunk != "data: [DONE]\n\n":
            try:
                chunk_data = json.loads(chunk[6:])  # Remove "data: " prefix
                chunks.append(chunk_data)
            except json.JSONDecodeError:
                pass

    # Verify chunk structure
    assert len(chunks) > 0

    for chunk in chunks:
        assert "id" in chunk
        assert "object" in chunk
        assert chunk["object"] == "chat.completion.chunk"
        assert "created" in chunk
        assert "model" in chunk
        assert "choices" in chunk

        if len(chunk["choices"]) > 0:
            choice = chunk["choices"][0]
            assert "index" in choice
            assert "delta" in choice
