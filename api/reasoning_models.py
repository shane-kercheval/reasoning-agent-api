"""Pydantic models for reasoning agent functionality."""
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ReasoningAction(str, Enum):
    """Actions the reasoning agent can take."""

    CONTINUE_THINKING = "continue_thinking"
    USE_TOOLS = "use_tools"
    FINISHED = "finished"


class ToolRequest(BaseModel):
    """Request to execute a specific tool."""

    server_name: str = Field(description="Name of the MCP server hosting the tool")
    tool_name: str = Field(description="Name of the tool to execute")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool arguments",
        json_schema_extra={"additionalProperties": False},
    )
    reasoning: str = Field(description="Explanation of why this tool is needed")

    model_config = ConfigDict(extra="forbid")


class ReasoningStep(BaseModel):
    """A single step in the reasoning process."""

    thought: str = Field(description="Current thinking/analysis")
    next_action: ReasoningAction = Field(description="What to do next")
    tools_to_use: list[ToolRequest] = Field(default_factory=list, description="Tools to execute if action is USE_TOOLS")  # noqa: E501
    parallel_execution: bool = Field(default=False, description="Whether tools can be executed in parallel")  # noqa: E501

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "thought": "I need to search for population data for both cities",
                "next_action": "use_tools",
                "tools_to_use": [
                    {
                        "server_name": "web_search",
                        "tool_name": "search",
                        "arguments": {"query": "Tokyo population 2024"},
                        "reasoning": "Get current population data for Tokyo",
                    },
                    {
                        "server_name": "web_search",
                        "tool_name": "search",
                        "arguments": {"query": "New York City population 2024"},
                        "reasoning": "Get current population data for NYC",
                    },
                ],
                "parallel_execution": True,
            },
        },
    )

    @classmethod
    def openai_schema(cls):
        """Generate OpenAI-compatible schema."""
        # Just use the standard schema - Pydantic already handles
        # required fields correctly by excluding those with defaults
        return cls.model_json_schema()


class ReasoningEventType(str, Enum):
    """Types of reasoning events for streaming."""

    REASONING_STEP = "reasoning_step"
    TOOL_EXECUTION = "tool_execution"
    TOOL_RESULT = "tool_result"
    PARALLEL_TOOLS = "parallel_tools"
    SYNTHESIS = "synthesis"
    ERROR = "error"


class ReasoningEventStatus(str, Enum):
    """Status of a reasoning event."""

    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ReasoningEvent(BaseModel):
    """Metadata for reasoning events in streaming responses."""

    type: ReasoningEventType = Field(description="Type of reasoning event")
    step_id: str = Field(description="Unique identifier for the step (e.g., '1', '2a', '2b')")
    tool_name: str | None = Field(default=None, description="Name of tool being executed")
    tools: list[str] | None = Field(default=None, description="List of tools for parallel execution")  # noqa: E501
    status: ReasoningEventStatus = Field(description="Current status of the event")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional event-specific data")  # noqa: E501
    error: str | None = Field(default=None, description="Error message if status is FAILED")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "tool_result",
                "step_id": "2a",
                "tool_name": "web_search",
                "status": "completed",
                "metadata": {
                    "population": "37400000",
                    "source": "official_statistics",
                    "year": "2024",
                },
            },
        },
    )


class ToolResult(BaseModel):
    """Result from a tool execution."""

    server_name: str = Field(description="MCP server that executed the tool")
    tool_name: str = Field(description="Name of the executed tool")
    success: bool = Field(description="Whether execution was successful")
    result: Any | None = Field(default=None, description="Tool execution result")
    error: str | None = Field(default=None, description="Error message if execution failed")
    execution_time_ms: float = Field(description="Execution time in milliseconds")


class MCPServerConfig(BaseModel):
    """Configuration for an HTTP-based MCP server."""

    name: str = Field(description="Unique name for the server")
    url: str = Field(description="HTTP URL for the MCP server")
    auth_env_var: str | None = Field(default=None, description="Environment variable containing auth token")  # noqa: E501
    enabled: bool = Field(default=True, description="Whether this server is enabled")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "web_search",
                    "url": "https://mcp-web-search.example.com",
                    "auth_env_var": "WEB_SEARCH_API_KEY",
                    "enabled": True,
                },
                {
                    "name": "local_tools",
                    "url": "http://localhost:8001",
                    "enabled": True,
                },
            ],
        },
    )


class MCPServersConfig(BaseModel):
    """Root configuration for all MCP servers."""

    servers: list[MCPServerConfig] = Field(description="List of MCP server configurations")

    def get_enabled_servers(self) -> list[MCPServerConfig]:
        """Get only enabled servers."""
        return [server for server in self.servers if server.enabled]


class ToolInfo(BaseModel):
    """Information about an available tool."""

    server_name: str = Field(description="MCP server hosting this tool")
    tool_name: str = Field(description="Name of the tool")
    description: str = Field(description="What the tool does")
    input_schema: dict[str, Any] = Field(description="JSON schema for tool inputs")
