"""
Generic tool abstraction for the reasoning agent.

This module provides a tool-source agnostic interface that allows the reasoning agent
to work with any callable tool, regardless of whether it's from MCP, local functions,
or other sources.
"""

import asyncio
import inspect
import time
from typing import Any
from collections.abc import Callable
import logging

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


class ToolResult(BaseModel):
    """Result from a tool execution."""

    tool_name: str = Field(description="Name of the tool that was executed")
    success: bool = Field(description="Whether execution was successful")
    result: Any | None = Field(default=None, description="Tool execution result")
    error: str | None = Field(default=None, description="Error message if execution failed")
    execution_time_ms: float = Field(description="Execution time in milliseconds")

    model_config = ConfigDict(extra="forbid")


class Tool(BaseModel):
    """
    Generic tool interface that abstracts the underlying implementation.

    This allows the reasoning agent to be tool-source agnostic - it can work with
    MCP tools, local functions, API calls, or any other callable without knowing
    the specific implementation details.
    """

    name: str = Field(description="Unique name for the tool")
    description: str = Field(description="Human-readable description of what the tool does")
    input_schema: dict[str, Any] = Field(description="JSON schema for tool input parameters")
    function: Callable = Field(exclude=True, description="The underlying callable function")

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,  # Allow Callable type
    )

    async def __call__(self, **kwargs) -> ToolResult:  # noqa: ANN003
        """
        Execute the tool with the given arguments.

        Args:
            **kwargs: Arguments to pass to the tool function

        Returns:
            ToolResult with success status, result data, and execution metrics
        """
        start_time = time.time()

        try:
            # Validate inputs against schema if possible
            # Note: Full JSON schema validation would require jsonschema library
            # For now, we'll do basic validation
            self._validate_inputs(kwargs)

            # Execute the function (handle both sync and async)
            if asyncio.iscoroutinefunction(self.function):
                result = await self.function(**kwargs)
            else:
                # Run sync function in thread pool to avoid blocking
                result = await asyncio.to_thread(self.function, **kwargs)

            execution_time = (time.time() - start_time) * 1000

            logger.debug(f"Tool '{self.name}' executed successfully in {execution_time:.2f}ms")

            return ToolResult(
                tool_name=self.name,
                success=True,
                result=result,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Tool '{self.name}' failed: {e!s}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def _validate_inputs(self, kwargs: dict[str, Any]) -> None:
        """
        Basic input validation against the schema.

        This is a simplified validation - for production use, consider
        using jsonschema library for full JSON schema validation.
        """
        schema_properties = self.input_schema.get("properties", {})
        required_fields = self.input_schema.get("required", [])

        # Check required fields
        for field in required_fields:
            if field not in kwargs:
                raise ValueError(f"Missing required parameter: {field}")

        # Check for unexpected fields (if additionalProperties is False)
        if not self.input_schema.get("additionalProperties", True):
            unexpected = set(kwargs.keys()) - set(schema_properties.keys())
            if unexpected:
                raise ValueError(f"Unexpected parameters: {', '.join(unexpected)}")

    def to_dict(self) -> dict[str, Any]:
        """Convert tool to dictionary for serialization (excluding function)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def function_to_tool(
    func: Callable,
    name: str | None = None,
    description: str | None = None,
    input_schema: dict[str, Any] | None = None,
) -> Tool:
    """
    Convert a Python function to a Tool object.

    Args:
        func: The function to convert
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
        input_schema: JSON schema for inputs (auto-generated if not provided)

    Returns:
        Tool object wrapping the function

    Example:
        def add_numbers(a: int, b: int) -> int:
            '''Add two numbers together.'''
            return a + b

        tool = function_to_tool(add_numbers)
    """
    tool_name = name or func.__name__
    tool_description = description or (func.__doc__ or "No description available").strip()

    # Auto-generate basic schema if not provided
    if input_schema is None:
        input_schema = _generate_schema_from_function(func)

    return Tool(
        name=tool_name,
        description=tool_description,
        input_schema=input_schema,
        function=func,
    )


def _generate_schema_from_function(func: Callable) -> dict[str, Any]:
    """
    Generate a schema from function signature using actual Python type hints.

    With structured outputs, we pass the actual type annotations to OpenAI
    instead of converting them to JSON schema types like 'integer' or 'boolean'.
    OpenAI can understand Python types directly.
    """
    sig = inspect.signature(func)
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        # Skip special parameters
        if param_name in ('self', 'cls', 'args', 'kwargs'):
            continue

        # Determine if required (no default value)
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

        # Use actual Python type hints - OpenAI can understand these directly
        if param.annotation != inspect.Parameter.empty:
            # Convert type annotation to string representation
            type_hint = str(param.annotation).replace("<class '", "").replace("'>", "")
            # Handle typing module types (e.g., typing.Union, typing.Optional)
            if hasattr(param.annotation, '__module__') and param.annotation.__module__ == 'typing':
                type_hint = str(param.annotation)
            properties[param_name] = {"type": type_hint}
        else:
            # No type hint provided - default to any
            properties[param_name] = {"type": "Any"}

        # Add default value info if present
        if param.default != inspect.Parameter.empty:
            properties[param_name]["default"] = param.default

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def format_tool_for_prompt(tool: Tool) -> str:
    """
    Format a single tool with its schema for inclusion in LLM prompts.

    This ensures the LLM sees parameter names, types, and whether they're required,
    preventing it from guessing parameter names based on common conventions.

    Args:
        tool: Tool object to format

    Returns:
        Formatted string showing tool name, description, and parameter schema

    Example:
        >>> tool = Tool(name="read_file", description="Read a file",
        ...             input_schema={"properties": {"path": {"type": "string"}},
        ...                          "required": ["path"]}, function=lambda: None)
        >>> print(format_tool_for_prompt(tool))
        - read_file
          Description: Read a file
          Parameters:
            - path (string) [REQUIRED]
    """
    schema = tool.input_schema
    params_info = []

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for param_name, param_info in properties.items():
        param_type = param_info.get("type", "any")
        is_required = param_name in required
        req_marker = "[REQUIRED]" if is_required else "[OPTIONAL]"

        # Include default value if present
        default_value = param_info.get("default")
        default_text = f" (default: {default_value})" if default_value is not None else ""

        params_info.append(f"    - {param_name} ({param_type}) {req_marker}{default_text}")

    params_text = "\n".join(params_info) if params_info else "    No parameters"

    return f"""- {tool.name}
  Description: {tool.description}
  Parameters:
{params_text}"""


def format_tools_for_prompt(tools: list[Tool]) -> str:
    r"""
    Format multiple tools for inclusion in LLM prompts.

    Includes tool names, descriptions, and complete parameter schemas so the LLM
    knows exactly what parameters to use (preventing guessing like 'file_path' vs 'path').

    Args:
        tools: List of Tool objects to format

    Returns:
        Formatted string with all tools, ready to include in prompt

    Example:
        >>> tools = [Tool(...), Tool(...)]
        >>> prompt = f"Available tools:\\n\\n{format_tools_for_prompt(tools)}"
    """
    if not tools:
        return "No tools are currently available."

    return "\n\n".join([
        format_tool_for_prompt(tool) for tool in tools
    ])

