"""
OpenAI Test Helper Functions.

These convenience functions are only used for testing and should not be
imported into production code. They provide simple ways to create common
OpenAI request/response patterns for test cases.
"""

import time
from typing import Any

from api.openai_protocol import (
    OpenAIRequestBuilder,
    OpenAIResponseBuilder,
    OpenAIStreamingResponseBuilder,
)


def create_simple_chat_request(
    model: str, user_message: str, system_message: str | None = None,
) -> dict[str, Any]:
    """Create a simple chat request for testing."""
    builder = OpenAIRequestBuilder().model(model)

    if system_message:
        builder.message("system", system_message)

    builder.message("user", user_message)

    return builder.build().model_dump(exclude_unset=True)


def create_simple_chat_response(completion_id: str, model: str, content: str) -> dict[str, Any]:
    """Create a simple chat response for testing."""
    return (
        OpenAIResponseBuilder()
        .id(completion_id)
        .model(model)
        .created(int(time.time()))
        .choice(0, "assistant", content)
        .usage(50, 25)
        .build()
        .model_dump()
    )


def create_streaming_chunks(completion_id: str, model: str, content: str) -> str:
    """Create streaming chunks for a simple response for testing."""
    words = content.split()

    builder = OpenAIStreamingResponseBuilder()

    # First chunk with role
    builder.chunk(completion_id, model, delta_role="assistant")

    # Content chunks
    for word in words:
        builder.chunk(completion_id, model, delta_content=word + " ")

    # Final chunk with finish reason
    builder.chunk(completion_id, model, finish_reason="stop")

    return builder.done().build()
