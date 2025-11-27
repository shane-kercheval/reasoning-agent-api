"""Tests for base tool and prompt abstractions."""

import pytest

from tools_api.services.tools.example_tool import EchoTool
from tools_api.services.prompts.example_prompt import GreetingPrompt


@pytest.mark.asyncio
async def test_tool_execution_success() -> None:
    """Test successful tool execution."""
    tool = EchoTool()
    result = await tool(message="hello world")

    assert result.success is True
    assert result.result["echo"] == "hello world"
    assert result.result["length"] == 11
    assert result.result["reversed"] == "dlrow olleh"
    assert result.error is None
    assert result.execution_time_ms > 0


@pytest.mark.asyncio
async def test_tool_execution_error_handling() -> None:
    """Test tool error handling."""
    tool = EchoTool()
    # Missing required parameter should cause error
    result = await tool()

    assert result.success is False
    assert result.result is None
    assert result.error is not None
    assert "message" in result.error.lower() or "required" in result.error.lower()
    assert result.execution_time_ms >= 0


@pytest.mark.asyncio
async def test_tool_metadata() -> None:
    """Test tool metadata properties."""
    tool = EchoTool()

    assert tool.name == "echo"
    assert tool.description == "Echo the input message back with metadata"
    assert tool.tags == ["example", "test"]
    assert "message" in tool.parameters["properties"]
    assert "message" in tool.parameters["required"]


@pytest.mark.asyncio
async def test_prompt_rendering_success() -> None:
    """Test successful prompt rendering."""
    prompt = GreetingPrompt()
    result = await prompt(name="Alice", formal=False)

    assert result.success is True
    assert "Alice" in result.content
    assert "Hey" in result.content
    assert result.error is None


@pytest.mark.asyncio
async def test_prompt_rendering_formal() -> None:
    """Test formal prompt rendering."""
    prompt = GreetingPrompt()
    result = await prompt(name="Bob", formal=True)

    assert result.success is True
    assert "Bob" in result.content
    assert "Good day" in result.content


@pytest.mark.asyncio
async def test_prompt_error_handling() -> None:
    """Test prompt error handling."""
    prompt = GreetingPrompt()
    # Missing required parameter
    result = await prompt()

    assert result.success is False
    assert result.content == ""
    assert result.error is not None


@pytest.mark.asyncio
async def test_prompt_metadata() -> None:
    """Test prompt metadata properties."""
    prompt = GreetingPrompt()

    assert prompt.name == "greeting"
    assert prompt.description == "Generate a greeting message"
    assert prompt.tags == ["example", "test"]
    assert len(prompt.arguments) == 2
    assert prompt.arguments[0]["name"] == "name"
    assert prompt.arguments[0]["required"] is True
    assert prompt.arguments[1]["name"] == "formal"
    assert prompt.arguments[1]["required"] is False
