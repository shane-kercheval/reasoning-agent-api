"""
Unit tests for MCP server handler functions.

Tests the MCP handler functions directly without transport layer.
"""

import json
from typing import Any

from mcp import types
from pydantic import BaseModel, Field
import pytest

from tools_api.mcp_server import call_tool, get_prompt, list_prompts, list_tools
from tools_api.services.base import BasePrompt, BaseTool
from tools_api.services.registry import PromptRegistry, ToolRegistry


# =============================================================================
# Result Models for Mock Tools
# =============================================================================


class MockToolResult(BaseModel):
    """Result model for MockTool."""

    output: str = Field(description="The processed output")


class MockFailingToolResult(BaseModel):
    """Result model for MockFailingTool (never returned since it always fails)."""

    pass


# =============================================================================
# Test Fixtures - Mock Tools and Prompts
# =============================================================================


class MockTool(BaseTool):
    """Mock tool for testing."""

    @property
    def name(self) -> str:  # noqa: D102
        return "mock_tool"

    @property
    def description(self) -> str:  # noqa: D102
        return "A mock tool for testing"

    @property
    def parameters(self) -> dict[str, Any]:  # noqa: D102
        return {
            "type": "object",
            "properties": {
                "input_text": {"type": "string", "description": "Test input"},
            },
            "required": ["input_text"],
        }

    @property
    def result_model(self) -> type[BaseModel]:  # noqa: D102
        return MockToolResult

    async def _execute(self, input_text: str) -> MockToolResult:
        return MockToolResult(output=f"processed: {input_text}")


class MockFailingTool(BaseTool):
    """Mock tool that always fails."""

    @property
    def name(self) -> str:  # noqa: D102
        return "failing_tool"

    @property
    def description(self) -> str:  # noqa: D102
        return "A tool that always fails"

    @property
    def parameters(self) -> dict[str, Any]:  # noqa: D102
        return {"type": "object", "properties": {}}

    @property
    def result_model(self) -> type[BaseModel]:  # noqa: D102
        return MockFailingToolResult

    async def _execute(self) -> MockFailingToolResult:
        raise RuntimeError("Tool execution failed")


class MockPrompt(BasePrompt):
    """Mock prompt for testing."""

    @property
    def name(self) -> str:  # noqa: D102
        return "mock_prompt"

    @property
    def description(self) -> str:  # noqa: D102
        return "A mock prompt for testing"

    @property
    def arguments(self) -> list[dict[str, Any]]:  # noqa: D102
        return [
            {"name": "name", "required": True, "description": "User name"},
            {"name": "formal", "required": False, "description": "Use formal greeting"},
        ]

    async def render(self, name: str, formal: bool = False) -> str:  # noqa: D102
        if formal:
            return f"Good day, {name}."
        return f"Hello, {name}!"


class MockFailingPrompt(BasePrompt):
    """Mock prompt that always fails."""

    @property
    def name(self) -> str:  # noqa: D102
        return "failing_prompt"

    @property
    def description(self) -> str:  # noqa: D102
        return "A prompt that always fails"

    @property
    def arguments(self) -> list[dict[str, Any]]:  # noqa: D102
        return []

    async def render(self) -> str:  # noqa: D102
        raise RuntimeError("Prompt rendering failed")


@pytest.fixture(autouse=True)
def clear_registries() -> None:
    """Clear registries before and after each test."""
    ToolRegistry.clear()
    PromptRegistry.clear()
    yield
    ToolRegistry.clear()
    PromptRegistry.clear()


@pytest.fixture
def mock_tool() -> MockTool:
    """Create a mock tool instance."""
    return MockTool()


@pytest.fixture
def mock_failing_tool() -> MockFailingTool:
    """Create a mock failing tool instance."""
    return MockFailingTool()


@pytest.fixture
def mock_prompt() -> MockPrompt:
    """Create a mock prompt instance."""
    return MockPrompt()


@pytest.fixture
def mock_failing_prompt() -> MockFailingPrompt:
    """Create a mock failing prompt instance."""
    return MockFailingPrompt()


# =============================================================================
# list_tools Tests
# =============================================================================


class MockToolWithCategory(BaseTool):
    """Mock tool with a category for testing."""

    @property
    def name(self) -> str:  # noqa: D102
        return "categorized_tool"

    @property
    def description(self) -> str:  # noqa: D102
        return "A tool with a category"

    @property
    def parameters(self) -> dict[str, Any]:  # noqa: D102
        return {"type": "object", "properties": {}}

    @property
    def result_model(self) -> type[BaseModel]:  # noqa: D102
        return MockToolResult

    @property
    def category(self) -> str | None:  # noqa: D102
        return "test-category"

    async def _execute(self) -> MockToolResult:
        return MockToolResult(output="categorized result")


class TestListTools:
    """Tests for the list_tools MCP handler."""

    @pytest.mark.asyncio
    async def test_empty_registry_returns_empty_list(self) -> None:
        """list_tools returns empty list when no tools registered."""
        result = await list_tools()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_registered_tools(self, mock_tool: MockTool) -> None:
        """list_tools returns all registered tools as MCP Tool types."""
        ToolRegistry.register(mock_tool)

        result = await list_tools()

        assert len(result) == 1
        assert isinstance(result[0], types.Tool)
        assert result[0].name == "mock_tool"
        # No category, so description is unchanged
        assert result[0].description == "A mock tool for testing"
        assert result[0].inputSchema == mock_tool.parameters

    @pytest.mark.asyncio
    async def test_tool_with_category_has_prefixed_description(self) -> None:
        """list_tools prefixes description with category when present."""
        tool = MockToolWithCategory()
        ToolRegistry.register(tool)

        result = await list_tools()

        assert len(result) == 1
        assert result[0].name == "categorized_tool"
        assert result[0].description == "[test-category] A tool with a category"

    @pytest.mark.asyncio
    async def test_returns_multiple_tools(
        self, mock_tool: MockTool, mock_failing_tool: MockFailingTool,
    ) -> None:
        """list_tools returns all registered tools."""
        ToolRegistry.register(mock_tool)
        ToolRegistry.register(mock_failing_tool)

        result = await list_tools()

        assert len(result) == 2
        tool_names = {t.name for t in result}
        assert tool_names == {"mock_tool", "failing_tool"}


# =============================================================================
# call_tool Tests
# =============================================================================


class TestCallTool:
    """Tests for the call_tool MCP handler."""

    @pytest.mark.asyncio
    async def test_tool_not_found_raises_error(self) -> None:
        """call_tool raises ValueError for non-existent tool."""
        with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
            await call_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_successful_execution_returns_text_content(
        self, mock_tool: MockTool,
    ) -> None:
        """call_tool returns TextContent with JSON result."""
        ToolRegistry.register(mock_tool)

        result = await call_tool("mock_tool", {"input_text": "test"})

        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)
        assert result[0].type == "text"

        # Verify JSON content
        parsed = json.loads(result[0].text)
        assert parsed == {"output": "processed: test"}

    @pytest.mark.asyncio
    async def test_tool_failure_raises_exception(
        self, mock_failing_tool: MockFailingTool,
    ) -> None:
        """call_tool raises exception when tool execution fails."""
        ToolRegistry.register(mock_failing_tool)

        with pytest.raises(Exception, match="Tool execution failed"):
            await call_tool("failing_tool", {})

    @pytest.mark.asyncio
    async def test_returns_pydantic_result_as_json(self) -> None:
        """call_tool returns Pydantic model result as JSON text."""

        class StringContentResult(BaseModel):
            """Result model with string content."""

            content: str = Field(description="The content string")

        class StringResultTool(BaseTool):
            @property
            def name(self) -> str:
                return "string_tool"

            @property
            def description(self) -> str:
                return "Returns a string content result"

            @property
            def parameters(self) -> dict[str, Any]:
                return {"type": "object", "properties": {}}

            @property
            def result_model(self) -> type[BaseModel]:
                return StringContentResult

            async def _execute(self) -> StringContentResult:
                return StringContentResult(content="plain string result")

        ToolRegistry.register(StringResultTool())
        result = await call_tool("string_tool", {})

        assert len(result) == 1
        parsed = json.loads(result[0].text)
        assert parsed == {"content": "plain string result"}


# =============================================================================
# list_prompts Tests
# =============================================================================


class TestListPrompts:
    """Tests for the list_prompts MCP handler."""

    @pytest.mark.asyncio
    async def test_empty_registry_returns_empty_list(self) -> None:
        """list_prompts returns empty list when no prompts registered."""
        result = await list_prompts()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_registered_prompts(self, mock_prompt: MockPrompt) -> None:
        """list_prompts returns all registered prompts as MCP Prompt types."""
        PromptRegistry.register(mock_prompt)

        result = await list_prompts()

        assert len(result) == 1
        assert isinstance(result[0], types.Prompt)
        assert result[0].name == "mock_prompt"
        assert result[0].description == "A mock prompt for testing"

        # Verify arguments mapping
        assert len(result[0].arguments) == 2
        arg_names = {a.name for a in result[0].arguments}
        assert arg_names == {"name", "formal"}

        # Check required flag
        name_arg = next(a for a in result[0].arguments if a.name == "name")
        formal_arg = next(a for a in result[0].arguments if a.name == "formal")
        assert name_arg.required is True
        assert formal_arg.required is False

    @pytest.mark.asyncio
    async def test_returns_multiple_prompts(
        self, mock_prompt: MockPrompt, mock_failing_prompt: MockFailingPrompt,
    ) -> None:
        """list_prompts returns all registered prompts."""
        PromptRegistry.register(mock_prompt)
        PromptRegistry.register(mock_failing_prompt)

        result = await list_prompts()

        assert len(result) == 2
        prompt_names = {p.name for p in result}
        assert prompt_names == {"mock_prompt", "failing_prompt"}


# =============================================================================
# get_prompt Tests
# =============================================================================


class TestGetPrompt:
    """Tests for the get_prompt MCP handler."""

    @pytest.mark.asyncio
    async def test_prompt_not_found_raises_error(self) -> None:
        """get_prompt raises ValueError for non-existent prompt."""
        with pytest.raises(ValueError, match="Prompt 'nonexistent' not found"):
            await get_prompt("nonexistent", {})

    @pytest.mark.asyncio
    async def test_successful_render_returns_prompt_result(
        self, mock_prompt: MockPrompt,
    ) -> None:
        """get_prompt returns GetPromptResult with rendered content."""
        PromptRegistry.register(mock_prompt)

        result = await get_prompt("mock_prompt", {"name": "World"})

        assert isinstance(result, types.GetPromptResult)
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert isinstance(result.messages[0].content, types.TextContent)
        assert result.messages[0].content.text == "Hello, World!"

    @pytest.mark.asyncio
    async def test_render_with_optional_args(self, mock_prompt: MockPrompt) -> None:
        """get_prompt handles optional arguments correctly."""
        PromptRegistry.register(mock_prompt)

        result = await get_prompt("mock_prompt", {"name": "World", "formal": True})

        assert result.messages[0].content.text == "Good day, World."

    @pytest.mark.asyncio
    async def test_render_with_none_arguments(self, mock_prompt: MockPrompt) -> None:
        """get_prompt handles None arguments dict."""
        PromptRegistry.register(mock_prompt)

        # Should fail because 'name' is required
        with pytest.raises(Exception, match="name"):
            await get_prompt("mock_prompt", None)

    @pytest.mark.asyncio
    async def test_prompt_failure_raises_exception(
        self, mock_failing_prompt: MockFailingPrompt,
    ) -> None:
        """get_prompt raises exception when prompt rendering fails."""
        PromptRegistry.register(mock_failing_prompt)

        with pytest.raises(Exception, match="Prompt rendering failed"):
            await get_prompt("failing_prompt", {})
