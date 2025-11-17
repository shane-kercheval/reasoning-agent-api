"""
Generic prompt abstraction for MCP prompts.

This module provides a prompt-source agnostic interface that allows the API
to work with MCP prompts from any server, providing reusable message templates
with parameters for LLM consumption.
"""

import asyncio
from typing import Any
from collections.abc import Callable
import logging

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


class PromptResult(BaseModel):
    """Result from prompt execution."""

    prompt_name: str = Field(description="Name of the prompt that was executed")
    success: bool = Field(description="Whether execution was successful")
    messages: list[dict[str, Any]] | None = Field(
        default=None,
        description="MCP PromptMessage format (role and content)",
    )
    error: str | None = Field(default=None, description="Error message if execution failed")

    model_config = ConfigDict(extra="forbid")


class Prompt(BaseModel):
    """
    Generic prompt interface abstracting MCP prompts.

    Similar to Tool, but for prompt templates instead of actions.
    Prompts return message sequences for LLM consumption.
    """

    name: str = Field(description="Unique name for the prompt")
    description: str = Field(description="Human-readable description of what the prompt does")
    arguments: list[dict[str, Any]] = Field(
        description="Argument specifications (name, required, description)",
    )
    function: Callable = Field(exclude=True, description="The underlying callable function")

    # MCP metadata fields
    server_name: str | None = Field(
        default=None,
        description="Source MCP server name (e.g., 'meta')",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Semantic tags for prompt categorization (e.g., ['automation', 'templates'])",
    )
    mcp_name: str | None = Field(
        default=None,
        exclude=True,
        description=(
            "Original MCP prompt name used for internal calls "
            "(excluded from API responses)"
        ),
    )

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,  # Allow Callable type
    )

    async def __call__(self, **kwargs) -> PromptResult:  # noqa: ANN003
        """
        Execute the prompt with the given arguments.

        Args:
            **kwargs: Arguments to pass to the prompt function

        Returns:
            PromptResult with success status and messages

        Raises:
            ValueError: If validation fails (missing/unexpected arguments)
        """
        # Validate inputs against argument specifications
        # ValueError is raised for validation errors and propagated to caller
        self._validate_inputs(kwargs)

        try:
            # Execute the function (handle both sync and async)
            if asyncio.iscoroutinefunction(self.function):
                result = await self.function(**kwargs)
            else:
                # Run sync function in thread pool to avoid blocking
                result = await asyncio.to_thread(self.function, **kwargs)

            logger.debug(f"Prompt '{self.name}' executed successfully")

            return result

        except Exception as e:
            return PromptResult(
                prompt_name=self.name,
                success=False,
                error=f"Prompt '{self.name}' failed: {e!s}",
            )

    def _validate_inputs(self, kwargs: dict[str, Any]) -> None:
        """
        Basic input validation against the argument specifications.

        Args:
            kwargs: Provided arguments to validate

        Raises:
            ValueError: If required arguments are missing or unexpected arguments provided
        """
        # Build set of required argument names
        required_args = {
            arg["name"] for arg in self.arguments if arg.get("required", False)
        }

        # Build set of all valid argument names
        valid_args = {arg["name"] for arg in self.arguments}

        # Check required arguments
        for arg_name in required_args:
            if arg_name not in kwargs:
                raise ValueError(f"Missing required argument: {arg_name}")

        # Check for unexpected arguments
        unexpected = set(kwargs.keys()) - valid_args
        if unexpected:
            raise ValueError(f"Unexpected arguments: {', '.join(unexpected)}")

    def to_dict(self) -> dict[str, Any]:
        """Convert prompt to dictionary for serialization (excluding function and mcp_name)."""
        result = {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
        }
        if self.server_name is not None:
            result["server_name"] = self.server_name
        if self.tags:
            result["tags"] = self.tags
        return result


def format_prompt_for_display(prompt: Prompt) -> str:
    """
    Format a single prompt with its arguments for display to users.

    Args:
        prompt: Prompt object to format

    Returns:
        Formatted string showing prompt name, description, and argument specifications

    Example:
        >>> prompt = Prompt(
        ...     name="ask_question",
        ...     description="Generate a question about a topic",
        ...     arguments=[
        ...         {"name": "topic", "required": True, "description": "Topic to ask about"}
        ...     ],
        ...     function=lambda: None
        ... )
        >>> print(format_prompt_for_display(prompt))
        ## ask_question
        <BLANKLINE>
        ### Description
        <BLANKLINE>
        Generate a question about a topic
        <BLANKLINE>
        ### Arguments
        <BLANKLINE>
        #### Required
        <BLANKLINE>
        - `topic`: Topic to ask about
    """
    # Separate required and optional arguments
    required_args = []
    optional_args = []

    for arg in prompt.arguments:
        arg_name = arg.get("name", "")
        arg_desc = arg.get("description", "")
        is_required = arg.get("required", False)

        # Build argument line
        arg_line = f"- `{arg_name}`"
        if arg_desc:
            arg_line += f": {arg_desc}"

        if is_required:
            required_args.append(arg_line)
        else:
            optional_args.append(arg_line)

    # Build arguments section
    args_text = ""
    if required_args:
        args_text += "#### Required\n\n" + "\n".join(required_args)
    if optional_args:
        if args_text:
            args_text += "\n\n"
        args_text += "#### Optional\n\n" + "\n".join(optional_args)
    if not args_text:
        args_text = "No arguments."

    return \
f"""## {prompt.name}

### Description

{prompt.description}

### Arguments

{args_text}"""


def format_prompts_for_display(prompts: list[Prompt]) -> str:
    """
    Format multiple prompts for display to users.

    Args:
        prompts: List of Prompt objects to format

    Returns:
        Formatted string with all prompts, ready for display
    """
    if not prompts:
        return "No prompts are currently available."

    # Add clear separator between prompts for better visual parsing
    return "\n\n---\n\n".join([
        format_prompt_for_display(prompt) for prompt in prompts
    ])
