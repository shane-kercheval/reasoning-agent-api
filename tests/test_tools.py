"""
Tests for the generic tool abstraction.

This module tests the Tool class and related utilities to ensure they work
correctly with both sync and async functions, handle errors properly, and
provide a consistent interface for the reasoning agent.
"""

import asyncio
from typing import Any
import pytest
import time

from api.tools import (
    Tool,
    ToolResult,
    function_to_tool,
)
from tests.fixtures.tools import weather_tool, search_tool



class TestTool:
    """Test the Tool class functionality."""

    def test_tool_creation(self):
        """Test basic Tool object creation."""
        def simple_func(x: int) -> int:
            return x * 2

        tool = Tool(
            name="multiply",
            description="Multiply by 2",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            function=simple_func,
        )

        assert tool.name == "multiply"
        assert tool.description == "Multiply by 2"
        assert tool.input_schema["required"] == ["x"]

    @pytest.mark.asyncio
    async def test_sync_function_execution(self):
        """Test Tool execution with synchronous function."""
        def add_numbers(a: int, b: int) -> int:
            return a + b

        tool = Tool(
            name="add",
            description="Add two numbers",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
            function=add_numbers,
        )

        result = await tool(a=5, b=3)

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.result == 8
        assert result.error is None
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_async_function_execution(self):
        """Test Tool execution with asynchronous function."""
        async def async_multiply(x: float, y: float) -> float:
            await asyncio.sleep(0.01)  # Simulate async work
            return x * y

        tool = Tool(
            name="async_multiply",
            description="Multiply two numbers asynchronously",
            input_schema={
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                },
                "required": ["x", "y"],
            },
            function=async_multiply,
        )

        result = await tool(x=2.5, y=4.0)

        assert result.success is True
        assert result.result == 10.0
        assert result.execution_time_ms >= 10  # Should take at least 10ms due to sleep

    @pytest.mark.asyncio
    async def test_function_exception_handling(self):
        """Test Tool handles exceptions properly."""
        def failing_function(should_fail: bool) -> str:
            if should_fail:
                raise ValueError("This function intentionally failed")
            return "success"

        tool = Tool(
            name="failing_tool",
            description="A tool that can fail",
            input_schema={
                "type": "object",
                "properties": {"should_fail": {"type": "boolean"}},
                "required": ["should_fail"],
            },
            function=failing_function,
        )

        # Test failure case
        result = await tool(should_fail=True)
        assert result.success is False
        assert "intentionally failed" in result.error
        assert result.result is None
        assert result.execution_time_ms > 0

        # Test success case
        result = await tool(should_fail=False)
        assert result.success is True
        assert result.result == "success"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_input_validation_required_fields(self):
        """Test that missing required fields are caught."""
        def simple_func(required_param: str) -> str:
            return f"Got: {required_param}"

        tool = Tool(
            name="test_tool",
            description="Test required validation",
            input_schema={
                "type": "object",
                "properties": {"required_param": {"type": "string"}},
                "required": ["required_param"],
            },
            function=simple_func,
        )

        # Should fail without required parameter
        result = await tool()
        assert result.success is False
        assert "Missing required parameter: required_param" in result.error

    @pytest.mark.asyncio
    async def test_input_validation_additional_properties(self):
        """Test validation of additional properties."""
        def simple_func(param: str) -> str:
            return param

        tool = Tool(
            name="strict_tool",
            description="Tool with strict schema",
            input_schema={
                "type": "object",
                "properties": {"param": {"type": "string"}},
                "required": ["param"],
                "additionalProperties": False,
            },
            function=simple_func,
        )

        # Should fail with unexpected parameter
        result = await tool(param="valid", unexpected="invalid")
        assert result.success is False
        assert "Unexpected parameters: unexpected" in result.error

    def test_to_dict(self):
        """Test Tool serialization to dictionary."""
        def dummy_func() -> None:
            pass

        tool = Tool(
            name="test",
            description="Test tool",
            input_schema={"type": "object"},
            function=dummy_func,
        )

        result = tool.to_dict()

        assert result == {
            "name": "test",
            "description": "Test tool",
            "input_schema": {"type": "object"},
        }
        # Function should not be included in serialization
        assert "function" not in result


class TestFunctionToTool:
    """Test the function_to_tool utility."""

    def test_basic_function_conversion(self):
        """Test converting a basic function to Tool."""
        def greet(name: str) -> str:
            """Greet a person by name."""
            return f"Hello, {name}!"

        tool = function_to_tool(greet)

        assert tool.name == "greet"
        assert tool.description == "Greet a person by name."
        assert "name" in tool.input_schema["properties"]
        assert tool.input_schema["required"] == ["name"]

    def test_function_with_defaults(self):
        """Test function with default parameters."""
        def calculate(x: int, multiplier: int = 2) -> int:
            """Calculate x * multiplier."""
            return x * multiplier

        tool = function_to_tool(calculate)

        # Only x should be required (multiplier has default)
        assert tool.input_schema["required"] == ["x"]
        assert "x" in tool.input_schema["properties"]
        assert "multiplier" in tool.input_schema["properties"]

    def test_custom_name_and_description(self):
        """Test overriding name and description."""
        def internal_func() -> None:
            """Internal docstring."""
            pass

        tool = function_to_tool(
            internal_func,
            name="public_name",
            description="Public description",
        )

        assert tool.name == "public_name"
        assert tool.description == "Public description"

    def test_custom_schema(self):
        """Test providing custom input schema."""
        def simple_func(data):  # noqa: ANN001, ANN202
            return data

        custom_schema = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "pattern": "^[A-Z]+$",
                },
            },
            "required": ["data"],
        }

        tool = function_to_tool(simple_func, input_schema=custom_schema)

        assert tool.input_schema == custom_schema

    def test_type_inference(self):
        """Test automatic type inference from annotations."""
        def typed_function(
            text: str,  # noqa: ARG001
            count: int,  # noqa: ARG001
            rate: float,  # noqa: ARG001
            active: bool,  # noqa: ARG001
            items: list,  # noqa: ARG001
            config: dict,  # noqa: ARG001
        ) -> str:
            return "result"

        tool = function_to_tool(typed_function)

        props = tool.input_schema["properties"]
        assert props["text"]["type"] == "str"
        assert props["count"]["type"] == "int"
        assert props["rate"]["type"] == "float"
        assert props["active"]["type"] == "bool"
        assert props["items"]["type"] == "list"
        assert props["config"]["type"] == "dict"

    @pytest.mark.asyncio
    async def test_converted_tool_execution(self):
        """Test that converted tool actually works."""
        def multiply_and_format(number: int, factor: int = 10) -> str:
            """Multiply number by factor and format as string."""
            return f"Result: {number * factor}"

        tool = function_to_tool(multiply_and_format)
        result = await tool(number=5, factor=3)

        assert result.success is True
        assert result.result == "Result: 15"


class TestMockTools:
    """Test mock tool utilities."""

    @pytest.mark.asyncio
    async def test_mock_weather_tool(self):
        """Test the mock weather tool."""
        tool = function_to_tool(weather_tool)

        result = await tool(location="New York")
        assert result.success is True
        assert result.result["location"] == "New York"
        assert "temperature" in result.result
        assert "condition" in result.result

    @pytest.mark.asyncio
    async def test_mock_search_tool(self):
        """Test the mock search tool."""
        tool =  function_to_tool(search_tool)

        result = await tool(query="test search", num_results=3)
        assert result.success is True
        assert result.result["query"] == "test search"
        assert len(result.result["results"]) == 3
        assert result.result["total_results"] == 3

    @pytest.mark.asyncio
    async def test_mock_search_tool_default_params(self):
        """Test mock search tool with default parameters."""
        tool =  function_to_tool(search_tool)

        result = await tool(query="default test")
        assert result.success is True
        assert len(result.result["results"]) == 5  # Default num_results


class TestToolPerformance:
    """Test Tool performance characteristics."""

    @pytest.mark.asyncio
    async def test_async_function_concurrency(self):
        """Test async functions run concurrently."""
        async def async_delay_task(delay: float) -> str:
            await asyncio.sleep(delay)
            return f"Completed after {delay}s"

        tool = function_to_tool(async_delay_task)
        # Run concurrent async tasks
        start_time = time.time()
        tasks = [tool(delay=0.1) for _ in range(100)]  # 100ms each
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        # All should succeed
        assert all(r.success for r in results)
        # Should take roughly 100ms concurrently, not 100 * 100ms = 10 seconds
        elapsed = (end_time - start_time)
        assert elapsed < 0.5  # Should be less than 500ms total


class TestComplexTypeHints:
    """Test function_to_tool with complex Python type hints."""

    def test_union_types(self):
        """Test Union type hints are preserved."""
        def flexible_func(value: int | float | str) -> str:
            """Function that accepts multiple types."""
            return str(value)

        tool = function_to_tool(flexible_func)

        assert tool.input_schema["properties"]["value"]["type"] == "int | float | str"
        assert tool.input_schema["required"] == ["value"]

    def test_optional_types(self):
        """Test Optional type hints."""
        def optional_func(required: str, optional: int | None = None) -> str:
            """Function with optional parameter."""
            return f"{required}-{optional}"

        tool = function_to_tool(optional_func)

        assert tool.input_schema["properties"]["required"]["type"] == "str"
        assert tool.input_schema["properties"]["optional"]["type"] == "int | None"
        assert tool.input_schema["properties"]["optional"]["default"] is None
        assert tool.input_schema["required"] == ["required"]

    def test_list_types(self):
        """Test List type hints."""
        def list_func(items: list[str], numbers: list[int]) -> int:
            """Function with list parameters."""
            return len(items) + len(numbers)

        tool = function_to_tool(list_func)

        assert tool.input_schema["properties"]["items"]["type"] == "list[str]"
        assert tool.input_schema["properties"]["numbers"]["type"] == "list[int]"
        assert set(tool.input_schema["required"]) == {"items", "numbers"}

    def test_dict_types(self):
        """Test Dict type hints."""
        def dict_func(config: dict[str, Any], mapping: dict[str, int]) -> str:  # noqa: ARG001
            """Function with dict parameters."""
            return str(config)

        tool = function_to_tool(dict_func)

        assert tool.input_schema["properties"]["config"]["type"] == "dict[str, typing.Any]"
        assert tool.input_schema["properties"]["mapping"]["type"] == "dict[str, int]"

    def test_complex_nested_types(self):
        """Test complex nested type hints."""
        def complex_func(
            data: list[dict[str, int | str]] | None = None,
            callback: str | None = "default",  # noqa: ARG001
        ) -> bool:
            """Function with very complex type hints."""
            return data is not None

        tool = function_to_tool(complex_func)

        # Should preserve the complex type structure
        data_type = tool.input_schema["properties"]["data"]["type"]
        assert "list" in data_type
        assert "dict" in data_type
        assert "|" in data_type  # Union syntax
        assert tool.input_schema["properties"]["data"]["default"] is None
        assert tool.input_schema["properties"]["callback"]["default"] == "default"
        assert tool.input_schema["required"] == []  # All have defaults

    def test_no_type_hints(self):
        """Test functions without type hints."""
        def no_hints_func(param1, param2="default") -> str:  # noqa: ANN001
            """Function without type hints."""
            return f"{param1}-{param2}"

        tool = function_to_tool(no_hints_func)

        assert tool.input_schema["properties"]["param1"]["type"] == "Any"
        assert tool.input_schema["properties"]["param2"]["type"] == "Any"
        assert tool.input_schema["properties"]["param2"]["default"] == "default"
        assert tool.input_schema["required"] == ["param1"]
