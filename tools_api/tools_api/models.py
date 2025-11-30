"""Pydantic models for tools-api requests and responses."""

from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Structured tool execution result."""

    success: bool = Field(description="Whether execution was successful")
    result: Any = Field(description="Tool-specific data structure")
    error: str | None = Field(default=None, description="Error message if execution failed")
    execution_time_ms: float = Field(description="Execution time in milliseconds")


class PromptResult(BaseModel):
    """Structured prompt rendering result."""

    success: bool = Field(description="Whether rendering was successful")
    content: str = Field(description="Rendered prompt content")
    error: str | None = Field(default=None, description="Error message if rendering failed")


class ToolDefinition(BaseModel):
    """Tool metadata for discovery."""

    name: str = Field(description="Tool name (used in API endpoint)")
    description: str = Field(description="Human-readable description")
    parameters: dict[str, Any] = Field(description="JSON Schema for parameters")
    output_schema: dict[str, Any] = Field(description="JSON Schema for tool output")
    category: str | None = Field(default=None, description="Tool category for organization")
    tags: list[str] = Field(default_factory=list, description="Semantic tags for categorization")


class PromptInfo(BaseModel):
    """Prompt metadata for discovery."""

    name: str = Field(description="Prompt name (used in API endpoint)")
    description: str = Field(description="Human-readable description")
    arguments: list[dict[str, Any]] = Field(
        description="[{'name': '...', 'required': bool, 'description': '...'}]",
    )
    category: str | None = Field(default=None, description="Prompt category for organization")
    tags: list[str] = Field(default_factory=list, description="Semantic tags for categorization")
