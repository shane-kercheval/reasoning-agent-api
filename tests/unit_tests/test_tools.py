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
import pathlib
from api.tools import (
    Tool,
    ToolResult,
    function_to_tool,
    format_tool_for_prompt,
    format_tools_for_prompt,
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


class TestToolFormatting:
    """Test tool formatting functions for LLM prompts."""

    def test_format_tool_with_required_params(self):
        """Test formatting a tool with required parameters."""
        def dummy_func() -> None:
            pass

        tool = Tool(
            name="read_file",
            description="Read a file from disk",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
            function=dummy_func,
        )

        result = format_tool_for_prompt(tool)

        # Check all essential parts are present
        assert "## read_file" in result
        assert "### Description" in result
        assert "Read a file from disk" in result
        assert "### Parameters" in result
        assert "#### Required" in result
        assert "`path` (string)" in result

    def test_format_tool_with_optional_params(self):
        """Test formatting a tool with optional parameters."""
        def dummy_func() -> None:
            pass

        tool = Tool(
            name="search",
            description="Search for items",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "number"},
                },
                "required": ["query"],
            },
            function=dummy_func,
        )

        result = format_tool_for_prompt(tool)

        assert "#### Required" in result
        assert "`query` (string)" in result
        assert "#### Optional" in result
        assert "`limit` (number)" in result

    def test_format_tool_with_defaults(self):
        """Test formatting a tool with default values."""
        def dummy_func() -> None:
            pass

        tool = Tool(
            name="config_tool",
            description="Configure settings",
            input_schema={
                "type": "object",
                "properties": {
                    "timeout": {"type": "number", "default": 30},
                    "retry": {"type": "boolean", "default": True},
                },
                "required": [],
            },
            function=dummy_func,
        )

        result = format_tool_for_prompt(tool)

        assert "#### Optional" in result
        assert "`timeout` (number - Default: `30`)" in result
        assert "`retry` (boolean - Default: `True`)" in result

    def test_format_tool_no_params(self):
        """Test formatting a tool with no parameters."""
        def dummy_func() -> None:
            pass

        tool = Tool(
            name="get_time",
            description="Get current time",
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
            function=dummy_func,
        )

        result = format_tool_for_prompt(tool)

        assert "## get_time" in result
        assert "### Description" in result
        assert "Get current time" in result
        assert "No parameters." in result

    def test_format_tool_prevents_parameter_guessing(self):
        """
        Test that formatted output includes exact parameter names.

        This is critical - the LLM should see 'path' not guess 'file_path'.
        """
        def dummy_func() -> None:
            pass

        tool = Tool(
            name="filesystem_read_text_file",
            description="Read the complete contents of a file from the file system as text.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "tail": {"type": "number", "default": None},
                    "head": {"type": "number", "default": None},
                },
                "required": ["path"],
            },
            function=dummy_func,
        )

        result = format_tool_for_prompt(tool)

        # Must include exact parameter name 'path', not 'file_path'
        assert "`path` (string)" in result
        assert "#### Required" in result
        # Must NOT encourage guessing
        assert "file_path" not in result.lower()
        # Should show optional params
        assert "#### Optional" in result
        assert "`tail` (number)" in result
        assert "`head` (number)" in result

    def test_format_multiple_tools(self):
        """Test formatting multiple tools."""
        def dummy_func() -> None:
            pass

        tool1 = Tool(
            name="tool_one",
            description="First tool",
            input_schema={
                "type": "object",
                "properties": {"param1": {"type": "string"}},
                "required": ["param1"],
            },
            function=dummy_func,
        )

        tool2 = Tool(
            name="tool_two",
            description="Second tool",
            input_schema={
                "type": "object",
                "properties": {"param2": {"type": "number"}},
                "required": ["param2"],
            },
            function=dummy_func,
        )

        result = format_tools_for_prompt([tool1, tool2])

        # Both tools should be present
        assert "## tool_one" in result
        assert "## tool_two" in result
        assert "First tool" in result
        assert "Second tool" in result
        assert "`param1` (string)" in result
        assert "`param2` (number)" in result
        assert "---" in result  # Separator between tools

    def test_format_empty_tools_list(self):
        """Test formatting with no tools."""
        result = format_tools_for_prompt([])

        assert result == "No tools are currently available."

    def test_format_tool_complex_schema(self):
        """Test formatting a tool with complex parameter types."""
        def dummy_func() -> None:
            pass

        tool = Tool(
            name="complex_tool",
            description="Tool with complex types",
            input_schema={
                "type": "object",
                "properties": {
                    "items": {"type": "list[str]"},
                    "config": {"type": "dict[str, int]"},
                    "value": {"type": "int | float | str"},
                },
                "required": ["items"],
            },
            function=dummy_func,
        )

        result = format_tool_for_prompt(tool)

        # Complex types should be preserved
        assert "#### Required" in result
        assert "`items` (list[str])" in result
        assert "#### Optional" in result
        assert "`config` (dict[str, int])" in result
        assert "`value` (int | float | str)" in result

    def test_format_preserves_all_schema_information(self):
        """
        Test that formatted output preserves all critical schema information.

        This test ensures the LLM has everything it needs to make correct tool calls.
        """
        def dummy_func() -> None:
            pass

        tool = Tool(
            name="github_api",
            description="Make GitHub API calls",
            input_schema={
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string"},
                    "method": {"type": "string", "default": "GET"},
                    "headers": {"type": "dict", "default": {}},
                    "timeout": {"type": "number", "default": 30},
                },
                "required": ["endpoint"],
            },
            function=dummy_func,
        )

        result = format_tool_for_prompt(tool)

        # All information should be present
        assert "github_api" in result
        assert "Make GitHub API calls" in result
        assert "#### Required" in result
        assert "`endpoint` (string)" in result
        assert "#### Optional" in result
        assert "`method` (string - Default: `\"GET\"`)" in result
        assert "`headers` (dict - Default: `{}`)" in result
        assert "`timeout` (number - Default: `30`)" in result

    def test_formatted_output_structure(self):
        """Test the overall structure of formatted output."""
        def dummy_func() -> None:
            pass

        tool = Tool(
            name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {"param": {"type": "string"}},
                "required": ["param"],
            },
            function=dummy_func,
        )

        result = format_tool_for_prompt(tool)

        # Should have clear structure with markdown headers
        lines = result.split("\n")
        assert lines[0].startswith("## test_tool")
        assert "### Description" in result
        assert "### Parameters" in result
        assert "#### Required" in result
        assert "`param`" in result

    def test_complex_tools_formatting_snapshot(self) -> None:
        """
        Test complex tool formatting and save output to artifacts for visual inspection.

        This test creates a comprehensive set of tools with various parameter types,
        descriptions, defaults, and complexity levels. The formatted output is saved
        to tests/artifacts/tool_formatting_output.md for:
        1. Visual inspection of formatting quality
        2. Git diff tracking when formatting logic changes
        3. Documentation of expected output format
        """
        def dummy_func() -> None:
            pass

        # Tool 1: Database query tool with complex parameters
        database_query_tool = Tool(
            name="execute_database_query",
            description=(
                "Execute a SQL query against the database with support for parameterized "
                "queries, transactions, and result formatting. Returns query results as "
                "structured data with metadata about execution time and row counts."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute (supports parameterized queries with $1, $2, etc.)",  # noqa: E501
                    },
                    "parameters": {
                        "type": "list[Any]",
                        "description": "Optional list of parameters to bind to the query",
                        "default": [],
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Maximum execution time before query is cancelled",
                        "default": 30,
                    },
                    "return_format": {
                        "type": "string",
                        "description": "Format for results: 'dict', 'tuple', or 'dataframe'",
                        "default": "dict",
                    },
                    "use_transaction": {
                        "type": "boolean",
                        "description": "Whether to wrap query in a transaction",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
            function=dummy_func,
        )

        # Tool 2: File system operations with union types
        file_operations_tool = Tool(
            name="filesystem_operation",
            description=(
                "Perform various filesystem operations including reading, writing, moving, "
                "copying, and deleting files. Supports both text and binary modes with "
                "automatic encoding detection and path validation."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "Operation type: 'read', 'write', 'move', 'copy', 'delete', 'stat'",  # noqa: E501
                    },
                    "path": {
                        "type": "string",
                        "description": "Target file or directory path (supports absolute and relative paths)",  # noqa: E501
                    },
                    "content": {
                        "type": "str | bytes | None",
                        "description": "Content to write (required for 'write' operation)",
                        "default": None,
                    },
                    "destination": {
                        "type": "str | None",
                        "description": "Destination path for 'move' or 'copy' operations",
                        "default": None,
                    },
                    "encoding": {
                        "type": "str | None",
                        "description": "Text encoding for read/write operations (default: 'utf-8')",  # noqa: E501
                        "default": "utf-8",
                    },
                    "create_dirs": {
                        "type": "boolean",
                        "description": "Automatically create parent directories if they don't exist",  # noqa: E501
                        "default": True,
                    },
                    "follow_symlinks": {
                        "type": "boolean",
                        "description": "Whether to follow symbolic links during operations",
                        "default": True,
                    },
                },
                "required": ["operation", "path"],
            },
            function=dummy_func,
        )

        # Tool 3: API client with nested structures
        api_client_tool = Tool(
            name="http_api_request",
            description=(
                "Make HTTP API requests with full control over headers, authentication, "
                "request body, timeouts, and retry behavior. Supports all HTTP methods "
                "and returns structured response data with status codes and headers."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Target URL (must be valid HTTP/HTTPS endpoint)",
                    },
                    "method": {
                        "type": "string",
                        "description": "HTTP method: GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS",
                        "default": "GET",
                    },
                    "headers": {
                        "type": "dict[str, str]",
                        "description": "HTTP headers as key-value pairs (e.g., {'Authorization': 'Bearer token'})",  # noqa: E501
                        "default": {},
                    },
                    "body": {
                        "type": "dict[str, Any] | str | bytes | None",
                        "description": "Request body (automatically serialized based on Content-Type header)",  # noqa: E501
                        "default": None,
                    },
                    "query_params": {
                        "type": "dict[str, str | int | bool]",
                        "description": "URL query parameters to append to the URL",
                        "default": {},
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Request timeout in seconds (applies to both connect and read)",  # noqa: E501
                        "default": 30.0,
                    },
                    "retry_config": {
                        "type": "dict[str, int]",
                        "description": (
                            "Retry configuration with keys: 'max_attempts', 'backoff_factor', "
                            "'retry_on_status' (comma-separated status codes)"
                        ),
                        "default": {"max_attempts": 3, "backoff_factor": 2},
                    },
                    "verify_ssl": {
                        "type": "boolean",
                        "description": "Whether to verify SSL certificates (disable for self-signed certs)",  # noqa: E501
                        "default": True,
                    },
                    "follow_redirects": {
                        "type": "boolean",
                        "description": "Automatically follow HTTP redirects (3xx status codes)",
                        "default": True,
                    },
                },
                "required": ["url"],
            },
            function=dummy_func,
        )

        # Tool 4: Data transformation with complex types
        data_transform_tool = Tool(
            name="transform_data",
            description=(
                "Transform and manipulate structured data with support for filtering, "
                "mapping, aggregation, and joins. Works with lists, dicts, and nested "
                "structures to produce clean, formatted output."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "list[dict[str, Any]] | dict[str, Any]",
                        "description": "Input data to transform (supports both list of objects and single object)",  # noqa: E501
                    },
                    "operations": {
                        "type": "list[dict[str, str | int | float]]",
                        "description": (
                            "Ordered list of transformation operations to apply. Each operation "
                            "has 'type' and operation-specific parameters."
                        ),
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output format: 'json', 'csv', 'xml', 'yaml', or 'pretty'",
                        "default": "json",
                    },
                    "filter_nulls": {
                        "type": "boolean",
                        "description": "Remove null/None values from output",
                        "default": False,
                    },
                    "sort_keys": {
                        "type": "boolean",
                        "description": "Sort object keys alphabetically in output",
                        "default": False,
                    },
                },
                "required": ["data", "operations"],
            },
            function=dummy_func,
        )

        # Tool 5: Simple tool with no parameters
        health_check_tool = Tool(
            name="system_health_check",
            description=(
                "Perform a comprehensive health check of all system components including "
                "database connectivity, external API availability, disk space, memory usage, "
                "and service status. Returns detailed diagnostics and recommendations."
            ),
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
            function=dummy_func,
        )

        # Format all tools
        tools = [
            database_query_tool,
            file_operations_tool,
            api_client_tool,
            data_transform_tool,
            health_check_tool,
        ]

        formatted_output = format_tools_for_prompt(tools)

        # Write to artifacts directory
        artifacts_dir = pathlib.Path(__file__).parent.parent / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        output_file = artifacts_dir / "tool_formatting_output.md"
        output_file.write_text(formatted_output)

        # Basic assertions to ensure the test is working
        assert "execute_database_query" in formatted_output
        assert "filesystem_operation" in formatted_output
        assert "http_api_request" in formatted_output
        assert "transform_data" in formatted_output
        assert "system_health_check" in formatted_output
        assert "No parameters." in formatted_output  # health check has no params
        assert "#### Required" in formatted_output
        assert "#### Optional" in formatted_output
        assert "Default:" in formatted_output
        assert "SQL query to execute" in formatted_output  # Check descriptions are included
