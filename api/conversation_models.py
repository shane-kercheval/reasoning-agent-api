"""
Pydantic models for conversation management REST API.

Provides request/response models for conversation CRUD endpoints.
"""

from uuid import UUID
from pydantic import BaseModel, ConfigDict, field_validator


class MessageResponse(BaseModel):
    """Message model for API responses."""

    model_config = ConfigDict(extra='forbid')

    id: UUID
    conversation_id: UUID
    role: str
    content: str | None
    reasoning_events: dict | None = None
    tool_calls: dict | None = None
    metadata: dict
    created_at: str
    sequence_number: int


class ConversationSummary(BaseModel):
    """Conversation summary for list endpoint (no messages)."""

    model_config = ConfigDict(extra='forbid')

    id: UUID
    title: str | None
    system_message: str
    created_at: str
    updated_at: str
    message_count: int


class ConversationListResponse(BaseModel):
    """Response model for GET /v1/conversations."""

    model_config = ConfigDict(extra='forbid')

    conversations: list[ConversationSummary]
    total: int
    limit: int
    offset: int


class ConversationDetail(BaseModel):
    """Full conversation with messages for detail endpoint."""

    model_config = ConfigDict(extra='forbid')

    id: UUID
    title: str | None
    system_message: str
    created_at: str
    updated_at: str
    messages: list[MessageResponse]


class UpdateConversationRequest(BaseModel):
    """Request model for PATCH /v1/conversations/{id}."""

    model_config = ConfigDict(extra='forbid')

    title: str | None

    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str | None) -> str | None:
        """Validate and normalize conversation title."""
        if v == "":
            # Treat empty string as null (clear the title)
            return None
        if v is not None:
            # Trim whitespace
            v = v.strip()
            # Check max length
            if len(v) > 200:
                raise ValueError("Title must not exceed 200 characters")
            # After trimming, if it's empty, treat as null
            if v == "":
                return None
        return v
