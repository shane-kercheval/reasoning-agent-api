"""Pydantic models for reasoning agent functionality."""
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ReasoningAction(str, Enum):
    """
    Actions available at each step of the reasoning process.

    Choose based on whether you need more analysis, external tools, or are ready to answer.
    """

    CONTINUE_THINKING = "continue_thinking"  # Need more analysis before proceeding
    USE_TOOLS = "use_tools"  # Need to call external tools for information or actions
    FINISHED = "finished"  # Ready to provide final answer to user


class ToolPrediction(BaseModel):
    """
    Represents a specific tool that should be executed as part of a reasoning step.

    Use this when you need to call external tools to gather information, perform
    calculations, or take actions. Each tool prediction should have clear reasoning
    for why that specific tool is needed at this point in the reasoning process.

    We use JSON mode instead of OpenAI structured outputs or function calling because:

    1. **Flexible Tool Arguments**: We need dict[str, Any] to work with any tool (MCP,
       function-based) at runtime. OpenAI structured outputs requires all objects to have
       additionalProperties: false, which makes dict[str, Any] impossible (the 'arguments'
       field would generate "additionalProperties": true and fail validation).

    2. **Tool Agnostic Design**: We want to work with any tool signature without
       pre-defining all possible argument schemas. JSON mode gives us this flexibility.

    3. **Architecture Requirements**: Our reasoning agent needs to predict reasoning steps
       AND tools in one response. OpenAI function calling assumes simple request -> tool
       execution, but we need multi-step reasoning planning.

    IMPORTANT: When using JSON mode (response_format={"type": "json_object"}), the LLM
    receives NO schema or field descriptions. You must include the complete JSON schema
    with all Field() descriptions in the system prompt so the model knows what to generate.
    """

    tool_name: str = Field(
        description="Exact name of the tool to execute (must match available tool names)",
    )
    arguments: dict[str, Any] = Field(
        description="Arguments to pass to the tool as key-value pairs with proper types",
    )
    reasoning: str = Field(
        description="Brief explanation of why this specific tool is needed right now",
    )

    model_config = ConfigDict(
        extra="forbid",
    )


class ReasoningStep(BaseModel):
    """
    A single step in the AI reasoning process for solving user requests.

    This represents one iteration of thinking, where you analyze the current situation,
    decide what to do next, and optionally specify tools to help you. Each step should
    build logically on previous steps and move toward solving the user's request.
    """

    thought: str = Field(
        description="Your current analysis and thinking about the user's request and what you've learned so far",  # noqa: E501
    )
    next_action: ReasoningAction = Field(
        description="What you should do next: continue_thinking, use_tools, or finish with final answer",  # noqa: E501
    )
    tools_to_use: list[ToolPrediction] = Field(
        default_factory=list,
        description="List of tools to execute if next_action is USE_TOOLS (empty if continuing thinking or finishing). All tools will be executed in asynchronously if concurrent_execution is True.",  # noqa: E501
    )
    concurrent_execution: bool = Field(
        default=False,
        description="Set to true if the tools can run concurrently, false if they must run in sequence",  # noqa: E501
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "thought": "The user is asking about population comparison. I need to get current data for both cities to provide accurate information.",  # noqa: E501
                    "next_action": "use_tools",
                    "tools_to_use": [
                        {
                            "tool_name": "search",
                            "arguments": {"query": "Tokyo population 2024"},
                            "reasoning": "Get current population data for Tokyo",
                        },
                        {
                            "tool_name": "search",
                            "arguments": {"query": "New York City population 2024"},
                            "reasoning": "Get current population data for NYC",
                        },
                    ],
                    "concurrent_execution": True,
                },
                {
                    "thought": "I have gathered the population data for both cities. Now I need to analyze and compare the numbers to answer the user's question.",  # noqa: E501
                    "next_action": "continue_thinking",
                    "tools_to_use": [],
                    "concurrent_execution": False,
                },
                {
                    "thought": "Based on my analysis of the population data, I now have enough information to provide a comprehensive answer comparing Tokyo and NYC populations.",  # noqa: E501
                    "next_action": "finished",
                    "tools_to_use": [],
                    "concurrent_execution": False,
                },
            ],
        },
    )


class ReasoningEventType(str, Enum):
    """
    Types of reasoning events for streaming.

    Each event type corresponds to specific reasoning process events:
    - ITERATION_START: Beginning of a reasoning step (ReasoningAgent)
    - PLANNING: Generated reasoning plan with thought and tool decisions (ReasoningAgent)
    - TOOL_EXECUTION_START: Starting tool execution (ReasoningAgent)
    - TOOL_RESULT: Tool execution completed with results (ReasoningAgent)
    - ITERATION_COMPLETE: Reasoning step finished (ReasoningAgent)
    - REASONING_COMPLETE: Final response synthesis completed (ReasoningAgent)
    - EXTERNAL_REASONING: Model's native reasoning (Anthropic/DeepSeek/etc.)
    - ERROR: Error occurred during reasoning or tool execution
    """

    ITERATION_START = "iteration_start"
    PLANNING = "planning"
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_RESULT = "tool_result"
    ITERATION_COMPLETE = "iteration_complete"
    REASONING_COMPLETE = "reasoning_complete"
    EXTERNAL_REASONING = "external_reasoning"
    ERROR = "error"


class ReasoningEvent(BaseModel):
    """Metadata for reasoning events in streaming responses."""

    type: ReasoningEventType = Field(description="Type of reasoning event")
    step_iteration: int = Field(description="Iteration number of the reasoning step (1, 2, 3, etc.)")  # noqa: E501
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional event-specific data including tools")  # noqa: E501
    error: str | None = Field(default=None, description="Error message if event type is ERROR")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": ReasoningEventType.TOOL_RESULT.value,
                "step_iteration": 1,
                "metadata": {
                    "tools": ["web_search"],
                    "results": ["Search completed successfully"],
                },
            },
        },
    )


class ToolInfo(BaseModel):
    """Information about an available tool."""

    tool_name: str = Field(description="Name of the tool")
    description: str = Field(description="What the tool does")
    input_schema: dict[str, Any] = Field(description="JSON schema for tool inputs")
