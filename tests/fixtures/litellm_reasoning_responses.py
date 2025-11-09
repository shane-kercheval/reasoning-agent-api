"""Mock LiteLLM responses for reasoning models based on real API responses."""
from unittest.mock import Mock


def create_anthropic_reasoning_chunks() -> list[Mock]:
    """
    Create mock Anthropic reasoning response chunks.

    Based on real response structure from claude-3-7-sonnet-20250219.
    Mimics the pattern: reasoning_content chunks → content chunks
    """
    chunks = []

    # Reasoning phase - chunk 1
    chunk1 = Mock()
    chunk1.model_dump.return_value = {
        "id": "chatcmpl-test123",
        "created": 1234567890,
        "model": "claude-3-7-sonnet-20250219",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "reasoning_content": "To solve",
                "content": "",
                "role": "assistant",
            },
            "finish_reason": None,
        }],
        "usage": None,
    }
    # Add reasoning_content as attribute for direct access
    chunk1.choices = [Mock()]
    chunk1.choices[0].delta = Mock()
    chunk1.choices[0].delta.reasoning_content = "To solve"
    chunk1.choices[0].delta.content = ""
    chunk1.choices[0].delta.role = "assistant"
    chunks.append(chunk1)

    # Reasoning phase - chunk 2
    chunk2 = Mock()
    chunk2.model_dump.return_value = {
        "id": "chatcmpl-test123",
        "created": 1234567890,
        "model": "claude-3-7-sonnet-20250219",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "reasoning_content": " this problem",
                "content": "",
                "role": None,
            },
            "finish_reason": None,
        }],
        "usage": None,
    }
    chunk2.choices = [Mock()]
    chunk2.choices[0].delta = Mock()
    chunk2.choices[0].delta.reasoning_content = " this problem"
    chunk2.choices[0].delta.content = ""
    chunk2.choices[0].delta.role = None
    chunks.append(chunk2)

    # Reasoning phase - chunk 3
    chunk3 = Mock()
    chunk3.model_dump.return_value = {
        "id": "chatcmpl-test123",
        "created": 1234567890,
        "model": "claude-3-7-sonnet-20250219",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "reasoning_content": ", I need to multiply.",
                "content": "",
                "role": None,
            },
            "finish_reason": None,
        }],
        "usage": None,
    }
    chunk3.choices = [Mock()]
    chunk3.choices[0].delta = Mock()
    chunk3.choices[0].delta.reasoning_content = ", I need to multiply."
    chunk3.choices[0].delta.content = ""
    chunk3.choices[0].delta.role = None
    chunks.append(chunk3)

    # Content phase - chunk 1 (transition point)
    chunk4 = Mock()
    chunk4.model_dump.return_value = {
        "id": "chatcmpl-test123",
        "created": 1234567890,
        "model": "claude-3-7-sonnet-20250219",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "content": "The answer",
                "role": None,
            },
            "finish_reason": None,
        }],
        "usage": None,
    }
    chunk4.choices = [Mock()]
    chunk4.choices[0].delta = Mock()
    # No reasoning_content attribute in content chunks
    chunk4.choices[0].delta.content = "The answer"
    chunk4.choices[0].delta.role = None
    chunks.append(chunk4)

    # Content phase - chunk 2
    chunk5 = Mock()
    chunk5.model_dump.return_value = {
        "id": "chatcmpl-test123",
        "created": 1234567890,
        "model": "claude-3-7-sonnet-20250219",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "content": " is 56.",
                "role": None,
            },
            "finish_reason": None,
        }],
        "usage": None,
    }
    chunk5.choices = [Mock()]
    chunk5.choices[0].delta = Mock()
    chunk5.choices[0].delta.content = " is 56."
    chunk5.choices[0].delta.role = None
    chunks.append(chunk5)

    # Finish chunk
    chunk6 = Mock()
    chunk6.model_dump.return_value = {
        "id": "chatcmpl-test123",
        "created": 1234567890,
        "model": "claude-3-7-sonnet-20250219",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
        "usage": None,
    }
    chunk6.choices = [Mock()]
    chunk6.choices[0].delta = Mock()
    chunk6.choices[0].delta.content = None
    chunk6.choices[0].delta.role = None
    chunk6.choices[0].finish_reason = "stop"
    chunks.append(chunk6)

    return chunks


def create_openai_reasoning_chunks() -> list[Mock]:
    """
    Create mock OpenAI o3-mini response chunks.

    Based on real response: NO reasoning_content field, only content.
    """
    chunks = []

    # Content chunk 1
    chunk1 = Mock()
    chunk1.model_dump.return_value = {
        "id": "chatcmpl-openai123",
        "created": 1234567890,
        "model": "o3-mini",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "content": "The answer",
                "role": "assistant",
            },
            "finish_reason": None,
        }],
        "usage": None,
    }
    chunk1.choices = [Mock()]
    chunk1.choices[0].delta = Mock()
    chunk1.choices[0].delta.content = "The answer"
    chunk1.choices[0].delta.role = "assistant"
    chunks.append(chunk1)

    # Content chunk 2
    chunk2 = Mock()
    chunk2.model_dump.return_value = {
        "id": "chatcmpl-openai123",
        "created": 1234567890,
        "model": "o3-mini",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "content": " is 56.",
                "role": None,
            },
            "finish_reason": None,
        }],
        "usage": None,
    }
    chunk2.choices = [Mock()]
    chunk2.choices[0].delta = Mock()
    chunk2.choices[0].delta.content = " is 56."
    chunk2.choices[0].delta.role = None
    chunks.append(chunk2)

    # Finish chunk
    chunk3 = Mock()
    chunk3.model_dump.return_value = {
        "id": "chatcmpl-openai123",
        "created": 1234567890,
        "model": "o3-mini",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
        "usage": None,
    }
    chunk3.choices = [Mock()]
    chunk3.choices[0].delta = Mock()
    chunk3.choices[0].delta.content = None
    chunk3.choices[0].delta.role = None
    chunk3.choices[0].finish_reason = "stop"
    chunks.append(chunk3)

    return chunks
