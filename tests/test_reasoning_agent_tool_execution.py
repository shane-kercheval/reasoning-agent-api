"""
Test cases for the ReasoningAgent's tool execution methods, including sequential and concurrent
execution.
"""
import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock
import pytest
import httpx
from api.reasoning_agent import ReasoningAgent
from api.prompt_manager import PromptManager
from api.reasoning_models import ToolPrediction
from api.tools import function_to_tool

class TestSequentialToolExecution:
    """Test the _execute_tools method."""

    @pytest.fixture
    def reasoning_agent(self):
        """Create a reasoning agent with mock tools for testing."""
        http_client = httpx.AsyncClient()
        mock_prompt_manager = AsyncMock()

        # Create tools for testing
        def weather_func(location: str) -> dict:
            return {"location": location, "temperature": "22°C", "condition": "Sunny"}

        def search_func(query: str) -> dict:
            return {"query": query, "results": ["result1", "result2"]}

        def failing_func(should_fail: bool = True) -> dict:
            if should_fail:
                raise ValueError("Tool intentionally failed")
            return {"success": True}

        async def async_delay_task(delay: float) -> str:
            await asyncio.sleep(delay)
            return f"Completed after {delay}s"

        tools = [
            function_to_tool(weather_func, name="get_weather", description="Get weather"),
            function_to_tool(search_func, name="search_web", description="Search web"),
            function_to_tool(failing_func, name="failing_tool", description="Tool that can fail"),
            function_to_tool(async_delay_task, name="async_delay", description="Async delay task"),
        ]

        return ReasoningAgent(
            base_url="http://test",
            api_key="test-key",
            http_client=http_client,
            tools=tools,
            prompt_manager=mock_prompt_manager,
        )

    @pytest.mark.asyncio
    async def test_execute_single_tool_success(self, reasoning_agent: ReasoningAgent):
        """Test executing a single tool successfully."""
        prediction = ToolPrediction(
            tool_name="get_weather",
            arguments={"location": "Tokyo"},
            reasoning="Need weather data",
        )

        results = await reasoning_agent._execute_tools_sequentially([prediction])

        assert len(results) == 1
        result = results[0]
        assert result.success is True
        assert result.tool_name == "get_weather"
        assert result.result["location"] == "Tokyo"
        assert result.result["temperature"] == "22°C"
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_single_tool_failure(self, reasoning_agent: ReasoningAgent):
        """Test executing a tool that fails."""
        prediction = ToolPrediction(
            tool_name="failing_tool",
            arguments={"should_fail": True},
            reasoning="Test failure",
        )

        results = await reasoning_agent._execute_tools_sequentially([prediction])

        assert len(results) == 1
        result = results[0]
        assert result.success is False
        assert result.tool_name == "failing_tool"
        assert "intentionally failed" in result.error
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_multiple_tools_sequential(self, reasoning_agent: ReasoningAgent):
        """Test executing multiple tools sequentially."""
        predictions = [
            ToolPrediction(
                    tool_name="get_weather",
                arguments={"location": "Tokyo"},
                reasoning="Need weather",
            ),
            ToolPrediction(
                    tool_name="search_web",
                arguments={"query": "Tokyo weather"},
                reasoning="Need more info",
            ),
        ]

        results = await reasoning_agent._execute_tools_sequentially(predictions)

        assert len(results) == 2

        # Weather result
        weather_result = results[0]
        assert weather_result.success is True
        assert weather_result.tool_name == "get_weather"
        assert weather_result.result["location"] == "Tokyo"

        # Search result
        search_result = results[1]
        assert search_result.success is True
        assert search_result.tool_name == "search_web"
        assert search_result.result["query"] == "Tokyo weather"

    @pytest.mark.asyncio
    async def test_execute_mixed_success_failure(self, reasoning_agent: ReasoningAgent):
        """Test executing tools with mixed success and failure."""
        predictions = [
            ToolPrediction(
                    tool_name="get_weather",
                arguments={"location": "Tokyo"},
                reasoning="Need weather",
            ),
            ToolPrediction(
                    tool_name="failing_tool",
                arguments={"should_fail": True},
                reasoning="Test failure",
            ),
        ]

        results = await reasoning_agent._execute_tools_sequentially(predictions)

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False
        assert "intentionally failed" in results[1].error

    @pytest.mark.asyncio
    async def test_execute_empty_list(self, reasoning_agent: ReasoningAgent):
        """Test executing an empty list of tools."""
        results = await reasoning_agent._execute_tools_sequentially([])
        assert results == []

        results = await reasoning_agent._execute_tools_concurrently([])
        assert results == []

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, reasoning_agent: ReasoningAgent):
        """Test executing a tool that doesn't exist."""
        prediction = ToolPrediction(
            tool_name="unknown_tool",
            arguments={},
            reasoning="Test unknown tool",
        )

        results = await reasoning_agent._execute_tools_sequentially([prediction])

        assert len(results) == 1
        result = results[0]
        assert result.success is False
        assert result.tool_name == "unknown_tool"
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_get_available_tools(self, reasoning_agent: ReasoningAgent):
        """Test getting available tools."""
        tools = await reasoning_agent.get_available_tools()

        assert len(tools) == 4
        tool_names = {tool.name for tool in tools}
        assert tool_names == {"get_weather", "search_web", "failing_tool", "async_delay"}

    @pytest.mark.asyncio
    async def test_execute_tool_directly(self, reasoning_agent: ReasoningAgent):
        """Test executing a tool directly by name."""
        prediction = ToolPrediction(
            tool_name="get_weather",
            arguments={"location": "Paris"},
            reasoning="Direct tool execution test",
        )
        results = await reasoning_agent._execute_tools_sequentially([prediction])
        result = results[0]

        assert result.success is True
        assert result.tool_name == "get_weather"
        assert result.result["location"] == "Paris"

    @pytest.mark.asyncio
    async def test_execute_tool_directly_unknown(self, reasoning_agent: ReasoningAgent):
        """Test executing unknown tool directly."""
        prediction = ToolPrediction(
            tool_name="unknown",
            arguments={},
            reasoning="Unknown tool test",
        )
        results = await reasoning_agent._execute_tools_sequentially([prediction])
        result = results[0]

        assert result.success is False
        assert result.tool_name == "unknown"
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_tool_execution_preserves_types(self, reasoning_agent: ReasoningAgent):
        """Test that tool execution preserves result data types."""
        def complex_tool(data: dict) -> dict:
            return {
                "string": "test",
                "number": 42,
                "float": 3.14,
                "bool": True,
                "list": [1, 2, 3],
                "dict": {"nested": "value"},
                "original": data,
            }

        # Add complex tool temporarily
        complex_tool_obj = function_to_tool(complex_tool, name="complex")
        reasoning_agent.tools["complex"] = complex_tool_obj

        prediction = ToolPrediction(
            tool_name="complex",
            arguments={"data": {"input": "test"}},
            reasoning="Complex type preservation test",
        )
        results = await reasoning_agent._execute_tools_sequentially([prediction])
        result = results[0]

        assert result.success is True
        assert isinstance(result.result["string"], str)
        assert isinstance(result.result["number"], int)
        assert isinstance(result.result["float"], float)
        assert isinstance(result.result["bool"], bool)
        assert isinstance(result.result["list"], list)
        assert isinstance(result.result["dict"], dict)
        assert result.result["original"]["input"] == "test"

    @pytest.mark.asyncio
    async def test_execute_tool_with_sync_and_async_tools_sequentially(self, reasoning_agent: ReasoningAgent):  # noqa: E501
        predictions = [
            ToolPrediction(
                tool_name="get_weather",
                arguments={"location": "Berlin"},
                reasoning="Need weather data for Berlin",
            ),
            ToolPrediction(
                tool_name="async_delay",
                arguments={"delay": 0.01},
                reasoning="Testing async delay tool",
            ),
        ]
        results = await reasoning_agent._execute_tools_sequentially(predictions)
        assert len(results) == 2
        # Check weather result
        weather_result = results[0]
        assert weather_result.success is True
        assert weather_result.tool_name == "get_weather"
        assert weather_result.result["location"] == "Berlin"
        # Check async delay result
        delay_result = results[1]
        assert delay_result.success is True
        assert delay_result.tool_name == "async_delay"
        assert delay_result.result == "Completed after 0.01s"


class TestConcurrentToolExecution:
    """Test the concurrent tool execution functionality."""

    @pytest.fixture
    def reasoning_agent(self):
        """Create a reasoning agent with mock tools for testing."""
        http_client = httpx.AsyncClient()
        mock_prompt_manager = AsyncMock()
        # Create tools for testing
        def weather_func(location: str) -> dict:
            return {"location": location, "temperature": "22°C", "condition": "Sunny"}
        def search_func(query: str) -> dict:
            return {"query": query, "results": ["result1", "result2"]}
        def failing_func(should_fail: bool = True) -> dict:
            if should_fail:
                raise ValueError("Tool intentionally failed")
            return {"success": True}
        async def async_delay_task(delay: float) -> str:
            await asyncio.sleep(delay)
            return f"Completed after {delay}s"
        tools = [
            function_to_tool(weather_func, name="get_weather", description="Get weather"),
            function_to_tool(search_func, name="search_web", description="Search web"),
            function_to_tool(failing_func, name="failing_tool", description="Tool that can fail"),
            function_to_tool(async_delay_task, name="async_delay", description="Async delay task"),
        ]
        return ReasoningAgent(
            base_url="http://test",
            api_key="test-key",
            http_client=http_client,
            tools=tools,
            prompt_manager=mock_prompt_manager,
        )

    @pytest.mark.asyncio
    async def test_execute_tools_concurrently_known_tools(self, reasoning_agent: ReasoningAgent):
        """Test parallel execution with known tools."""
        # Create tool predictions for known tools
        predictions = [
            ToolPrediction(
                tool_name="get_weather",
                arguments={"location": "Tokyo"},
                reasoning="Need weather for Tokyo",
            ),
            ToolPrediction(
                tool_name="search_web",
                arguments={"query": "weather Tokyo"},
                reasoning="Need to search for weather",
            ),
        ]

        # Execute tools in parallel
        results = await reasoning_agent._execute_tools_concurrently(predictions)

        # Verify results
        assert len(results) == 2

        # Check weather result
        weather_result = results[0]
        assert weather_result.tool_name == "get_weather"
        assert weather_result.success
        assert weather_result.result["location"] == "Tokyo"
        assert weather_result.result["temperature"] == "22°C"

        # Check search result
        search_result = results[1]
        assert search_result.tool_name == "search_web"
        assert search_result.success
        assert search_result.result["query"] == "weather Tokyo"

    @pytest.mark.asyncio
    async def test_execute_tools_concurrently_unknown_tools(self, reasoning_agent: ReasoningAgent):
        """Test parallel execution with unknown tools."""
        # Create tool predictions for unknown tools
        predictions = [
            ToolPrediction(
                tool_name="unknown_tool1",
                arguments={"param": "value1"},
                reasoning="Testing unknown tool",
            ),
            ToolPrediction(
                tool_name="unknown_tool2",
                arguments={"param": "value2"},
                reasoning="Testing another unknown tool",
            ),
        ]

        # Execute tools in parallel
        results = await reasoning_agent._execute_tools_concurrently(predictions)

        # Verify results
        assert len(results) == 2

        # Check first unknown tool result
        result1 = results[0]
        assert result1.tool_name == "unknown_tool1"
        assert not result1.success
        assert "Tool 'unknown_tool1' not found" in result1.error

        # Check second unknown tool result
        result2 = results[1]
        assert result2.tool_name == "unknown_tool2"
        assert not result2.success
        assert "Tool 'unknown_tool2' not found" in result2.error

    @pytest.mark.asyncio
    async def test_execute_tools_concurrently_mixed_known_unknown(self, reasoning_agent: ReasoningAgent):  # noqa: E501
        """Test parallel execution with mix of known and unknown tools."""
        # Create tool predictions mixing known and unknown tools
        predictions = [
            ToolPrediction(
                tool_name="get_weather",
                arguments={"location": "Paris"},
                reasoning="Need weather for Paris",
            ),
            ToolPrediction(
                tool_name="unknown_tool",
                arguments={"param": "value"},
                reasoning="Testing unknown tool",
            ),
            ToolPrediction(
                tool_name="search_web",
                arguments={"query": "Paris weather"},
                reasoning="Need to search for weather",
            ),
        ]

        # Execute tools in parallel
        results = await reasoning_agent._execute_tools_concurrently(predictions)

        # Verify results
        assert len(results) == 3

        # Check weather result (success)
        weather_result = results[0]
        assert weather_result.tool_name == "get_weather"
        assert weather_result.success
        assert weather_result.result["location"] == "Paris"

        # Check unknown tool result (failure)
        unknown_result = results[1]
        assert unknown_result.tool_name == "unknown_tool"
        assert not unknown_result.success
        assert "Tool 'unknown_tool' not found" in unknown_result.error

        # Check search result (success)
        search_result = results[2]
        assert search_result.tool_name == "search_web"
        assert search_result.success
        assert search_result.result["query"] == "Paris weather"

    @pytest.mark.asyncio
    async def test_execute_tools_concurrently_empty_list(self, reasoning_agent: ReasoningAgent):
        """Test parallel execution with empty tool predictions list."""
        # Execute with empty list
        results = await reasoning_agent._execute_tools_concurrently([])

        # Verify empty results
        assert len(results) == 0
        assert results == []

    @pytest.mark.asyncio
    async def test_execute_tool_with_sync_and_async_tools_parallel(self, reasoning_agent: ReasoningAgent):  # noqa: E501
        """Test executing both sync and async tools in parallel."""
        predictions = [
            ToolPrediction(
                tool_name="get_weather",
                arguments={"location": "Berlin"},
                reasoning="Need weather data for Berlin",
            ),
            ToolPrediction(
                tool_name="async_delay",
                arguments={"delay": 0.01},
                reasoning="Testing async delay tool",
            ),
        ]
        results = await reasoning_agent._execute_tools_concurrently(predictions)
        assert len(results) == 2
        # Check weather result
        weather_result = results[0]
        assert weather_result.success is True
        assert weather_result.tool_name == "get_weather"
        assert weather_result.result["location"] == "Berlin"
        # Check async delay result
        delay_result = results[1]
        assert delay_result.success is True
        assert delay_result.tool_name == "async_delay"
        assert delay_result.result == "Completed after 0.01s"

    @pytest.mark.asyncio
    async def test_execute_tools_concurrent_execution_time(self, reasoning_agent: ReasoningAgent):
        """Test that parallel execution is actually faster than sequential."""
        # Create multiple tool predictions
        predictions = [
            ToolPrediction(
                tool_name="async_delay",
                arguments={"delay": 0.1},
                reasoning=f"Delay task {i}",
            )
            for i in range(100)
        ]

        # Test parallel (concurrent/async) execution time
        start_time = time.time()
        results = await reasoning_agent._execute_tools_concurrently(predictions)
        end_time = time.time()
        # Should take roughly 100ms concurrently, not 100 * 100ms = 10 seconds
        elapsed = (end_time - start_time)
        assert elapsed < 0.5  # Should be less than 500ms total
        assert len(results) == 100
        assert all(result.success for result in results)


class TestToolExecutionEdgeCases:
    """Test edge cases in tool execution with complex types."""

    @pytest.fixture
    def complex_tool_agent(self):
        """Agent with tools that have complex type signatures."""
        http_client = httpx.AsyncClient()
        mock_prompt_manager = AsyncMock(spec=PromptManager)

        def complex_analysis(
            data: list[dict[str, int | float]],
            threshold: float | None = None,
            filters: list[str] | None = None,
            options: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """Complex function with nested types."""
            if options is None:
                options = {}
            return {
                "processed": len(data),
                "threshold": threshold,
                "filters": filters or [],
                "options": options,
            }

        def timeout_func(duration: int) -> str:
            """Function that can timeout."""
            time.sleep(duration)
            return f"Slept for {duration} seconds"

        async def async_timeout_func(duration: float) -> str:
            """Async function that can timeout."""
            await asyncio.sleep(duration)
            return f"Async slept for {duration} seconds"

        tools = [
            function_to_tool(complex_analysis, name="analyze_data"),
            function_to_tool(timeout_func, name="sync_timeout"),
            function_to_tool(async_timeout_func, name="async_timeout"),
        ]

        return ReasoningAgent(
            base_url="http://test",
            api_key="test-key",
            http_client=http_client,
            tools=tools,
            prompt_manager=mock_prompt_manager,
        )

    @pytest.mark.asyncio
    async def test_complex_type_execution(self, complex_tool_agent: ReasoningAgent):
        """Test execution with complex nested types."""
        prediction = ToolPrediction(
            tool_name="analyze_data",
            arguments={
                "data": [{"value": 10}, {"value": 20.5}],
                "threshold": 15.0,
                "filters": ["active", "recent"],
                "options": {"strict": True, "format": "json"},
            },
            reasoning="Test complex type handling",
        )

        results = await complex_tool_agent._execute_tools_sequentially([prediction])

        assert len(results) == 1
        result = results[0]
        assert result.success is True
        assert result.result["processed"] == 2
        assert result.result["threshold"] == 15.0
        assert result.result["filters"] == ["active", "recent"]
        assert result.result["options"]["strict"] is True

    @pytest.mark.asyncio
    async def test_optional_parameters_with_none(self, complex_tool_agent: ReasoningAgent):
        """Test execution with None values for optional parameters."""
        prediction = ToolPrediction(
            tool_name="analyze_data",
            arguments={
                "data": [{"value": 1}],
                "threshold": None,
                "filters": None,
            },
            reasoning="Test None handling",
        )

        results = await complex_tool_agent._execute_tools_sequentially([prediction])

        assert len(results) == 1
        result = results[0]
        assert result.success is True
        assert result.result["threshold"] is None
        assert result.result["filters"] == []  # Default handling

    @pytest.mark.asyncio
    async def test_sync_function_timeout_protection(self, complex_tool_agent: ReasoningAgent):
        """Test that sync functions don't block indefinitely."""
        prediction = ToolPrediction(
            tool_name="sync_timeout",
            arguments={"duration": 2},  # 2 second sleep
            reasoning="Test sync timeout protection",
        )

        # Should complete without hanging due to asyncio.to_thread
        start_time = asyncio.get_event_loop().time()
        results = await complex_tool_agent._execute_tools_sequentially([prediction])
        end_time = asyncio.get_event_loop().time()

        assert len(results) == 1
        result = results[0]
        assert result.success is True
        assert "Slept for 2 seconds" in result.result
        # Should take approximately 2+ seconds but not block the event loop
        assert end_time - start_time >= 2.0
