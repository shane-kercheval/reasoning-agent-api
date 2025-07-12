"""
Pydantic models for the Reasoning Agent API.

This module defines data models for OpenAI-compatible chat completion requests
and responses, including support for streaming and non-streaming formats.
"""

from pydantic import BaseModel
from typing import Literal
from enum import Enum

class MessageRole(str, Enum):
    """Enumeration for message roles in chat completions."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class ChatMessage(BaseModel):
    """Model representing a chat message in the completion request."""

    role: MessageRole
    content: str

class ChatCompletionRequest(BaseModel):
    """Model for OpenAI-compatible chat completion requests."""

    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    n: int | None = 1
    stream: bool | None = False
    stop: str | list[str] | None = None
    presence_penalty: float | None = 0
    frequency_penalty: float | None = 0
    logit_bias: dict[str, float] | None = None
    user: str | None = None

class Usage(BaseModel):
    """Model for usage statistics in chat completions."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Choice(BaseModel):
    """Model representing a choice in the chat completion response."""

    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "content_filter"] | None = None

class ChatCompletionResponse(BaseModel):
    """Model for OpenAI-compatible chat completion responses."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage

# Streaming models
class Delta(BaseModel):
    """Model representing a delta in streaming responses."""

    role: MessageRole | None = None
    content: str | None = None

class StreamChoice(BaseModel):
    """Model representing a choice in streaming chat completions."""

    index: int
    delta: Delta
    finish_reason: Literal["stop", "length", "content_filter"] | None = None

class ChatCompletionStreamResponse(BaseModel):
    """Model for OpenAI-compatible streaming chat completion responses."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]

class ModelInfo(BaseModel):
    """Model information for OpenAI-compatible models."""

    id: str
    object: str = "model"
    created: int
    owned_by: str

class ModelsResponse(BaseModel):
    """Response model for listing available models."""

    object: str = "list"
    data: list[ModelInfo]

class ErrorDetail(BaseModel):
    """Model for error details in API responses."""

    message: str
    type: str
    param: str | None = None
    code: str | None = None

class ErrorResponse(BaseModel):
    """Model for error responses in the API."""

    error: ErrorDetail
