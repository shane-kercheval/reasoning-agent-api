"""Tests for tool and prompt registries."""

import pytest

from tools_api.services.registry import ToolRegistry, PromptRegistry
from tools_api.services.tools.example_tool import EchoTool
from tools_api.services.prompts.example_prompt import GreetingPrompt


@pytest.fixture(autouse=True)
def clear_registries():
    """Clear registries before each test."""
    ToolRegistry.clear()
    PromptRegistry.clear()
    yield
    ToolRegistry.clear()
    PromptRegistry.clear()


def test_tool_registry_register() -> None:
    """Test registering a tool."""
    tool = EchoTool()
    ToolRegistry.register(tool)

    retrieved = ToolRegistry.get("echo")
    assert retrieved is tool
    assert retrieved.name == "echo"


def test_tool_registry_duplicate_error() -> None:
    """Test that registering duplicate tool raises error."""
    tool1 = EchoTool()
    tool2 = EchoTool()

    ToolRegistry.register(tool1)
    with pytest.raises(ValueError, match="already registered"):
        ToolRegistry.register(tool2)


def test_tool_registry_get_nonexistent() -> None:
    """Test getting non-existent tool returns None."""
    result = ToolRegistry.get("nonexistent")
    assert result is None


def test_tool_registry_list() -> None:
    """Test listing all tools."""
    tool = EchoTool()
    ToolRegistry.register(tool)

    tools = ToolRegistry.list()
    assert len(tools) == 1
    assert tools[0] is tool


def test_tool_registry_list_empty() -> None:
    """Test listing tools when registry is empty."""
    tools = ToolRegistry.list()
    assert tools == []


def test_tool_registry_clear() -> None:
    """Test clearing the registry."""
    tool = EchoTool()
    ToolRegistry.register(tool)

    ToolRegistry.clear()

    assert ToolRegistry.get("echo") is None
    assert ToolRegistry.list() == []


def test_prompt_registry_register() -> None:
    """Test registering a prompt."""
    prompt = GreetingPrompt()
    PromptRegistry.register(prompt)

    retrieved = PromptRegistry.get("greeting")
    assert retrieved is prompt
    assert retrieved.name == "greeting"


def test_prompt_registry_duplicate_error() -> None:
    """Test that registering duplicate prompt raises error."""
    prompt1 = GreetingPrompt()
    prompt2 = GreetingPrompt()

    PromptRegistry.register(prompt1)
    with pytest.raises(ValueError, match="already registered"):
        PromptRegistry.register(prompt2)


def test_prompt_registry_get_nonexistent() -> None:
    """Test getting non-existent prompt returns None."""
    result = PromptRegistry.get("nonexistent")
    assert result is None


def test_prompt_registry_list() -> None:
    """Test listing all prompts."""
    prompt = GreetingPrompt()
    PromptRegistry.register(prompt)

    prompts = PromptRegistry.list()
    assert len(prompts) == 1
    assert prompts[0] is prompt


def test_prompt_registry_list_empty() -> None:
    """Test listing prompts when registry is empty."""
    prompts = PromptRegistry.list()
    assert prompts == []


def test_prompt_registry_clear() -> None:
    """Test clearing the registry."""
    prompt = GreetingPrompt()
    PromptRegistry.register(prompt)

    PromptRegistry.clear()

    assert PromptRegistry.get("greeting") is None
    assert PromptRegistry.list() == []
