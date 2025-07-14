"""
Direct unit tests for the _execute_tools method.

This module tests the core tool execution logic to ensure it returns
exactly what we expect for both successful and failed tool executions,
in both parallel and sequential modes.
"""

import pytest
from unittest.mock import AsyncMock
import httpx

from api.reasoning_agent import ReasoningAgent
from api.mcp import ToolResult, ToolRequest
from api.reasoning_models import ToolPrediction


class TestExecuteTools:
    """Test the _execute_tools method directly."""

    @pytest.fixture
    def reasoning_agent(self):
        """Create a reasoning agent with mocked dependencies."""
        http_client = httpx.AsyncClient()
        mock_mcp_manager = AsyncMock()
        mock_prompt_manager = AsyncMock()

        return ReasoningAgent(
            base_url="http://test",
            api_key="test-key",
            http_client=http_client,
            mcp_manager=mock_mcp_manager,
            prompt_manager=mock_prompt_manager,
        )

    @pytest.fixture
    def sample_tool_prediction(self):
        """Create a sample tool prediction."""
        return ToolPrediction(
            server_name="demo_tools",
            tool_name="get_weather",
            arguments={"location": "Tokyo"},
            reasoning="Need weather data for Tokyo",
        )

    @pytest.fixture
    def sample_tool_result_success(self):
        """Create a successful tool result."""
        return ToolResult(
            server_name="demo_tools",
            tool_name="get_weather",
            success=True,
            result={"location": "Tokyo", "temperature": "22°C", "condition": "Sunny"},
            execution_time_ms=150.0,
        )

    @pytest.fixture
    def sample_tool_result_failure(self):
        """Create a failed tool result."""
        return ToolResult(
            server_name="demo_tools",
            tool_name="get_weather",
            success=False,
            error="Connection timeout",
            execution_time_ms=100.0,
        )

    @pytest.mark.asyncio
    async def test_execute_tools_sequential_single_success(
        self,
        reasoning_agent: ReasoningAgent,
        sample_tool_prediction: ToolPrediction,
        sample_tool_result_success: ToolResult,
    ):
        """Test executing a single tool sequentially with success."""
        # Setup mock
        reasoning_agent.mcp_manager.execute_tool.return_value = sample_tool_result_success

        # Execute
        results = await reasoning_agent._execute_tools([sample_tool_prediction], parallel=False)

        # Verify
        assert len(results) == 1
        assert results[0] == sample_tool_result_success
        assert results[0].success is True
        assert results[0].tool_name == "get_weather"
        assert results[0].result["location"] == "Tokyo"

        # Verify the MCP manager was called correctly
        reasoning_agent.mcp_manager.execute_tool.assert_called_once()
        call_args = reasoning_agent.mcp_manager.execute_tool.call_args[0][0]
        assert isinstance(call_args, ToolRequest)
        assert call_args.tool_name == "get_weather"
        assert call_args.arguments == {"location": "Tokyo"}

    @pytest.mark.asyncio
    async def test_execute_tools_sequential_single_failure(
        self,
        reasoning_agent: ReasoningAgent,
        sample_tool_prediction: ToolPrediction,
        sample_tool_result_failure: ToolResult,
    ):
        """Test executing a single tool sequentially with failure."""
        # Setup mock
        reasoning_agent.mcp_manager.execute_tool.return_value = sample_tool_result_failure

        # Execute
        results = await reasoning_agent._execute_tools([sample_tool_prediction], parallel=False)

        # Verify
        assert len(results) == 1
        assert results[0] == sample_tool_result_failure
        assert results[0].success is False
        assert results[0].tool_name == "get_weather"
        assert results[0].error == "Connection timeout"

    @pytest.mark.asyncio
    async def test_execute_tools_sequential_multiple_tools(self, reasoning_agent: ReasoningAgent):
        """Test executing multiple tools sequentially."""
        # Create multiple tool predictions
        tool_predictions = [
            ToolPrediction(
                server_name="demo_tools",
                tool_name="get_weather",
                arguments={"location": "Tokyo"},
                reasoning="Need Tokyo weather",
            ),
            ToolPrediction(
                server_name="demo_tools",
                tool_name="search_web",
                arguments={"query": "weather Tokyo"},
                reasoning="Need more weather info",
            ),
        ]

        # Create corresponding results
        weather_result = ToolResult(
            server_name="demo_tools",
            tool_name="get_weather",
            success=True,
            result={"location": "Tokyo", "temp": "22°C"},
            execution_time_ms=150.0,
        )

        search_result = ToolResult(
            server_name="demo_tools",
            tool_name="search_web",
            success=True,
            result={"query": "weather Tokyo", "results": ["result1"]},
            execution_time_ms=200.0,
        )

        # Setup mock to return different results for each call
        reasoning_agent.mcp_manager.execute_tool.side_effect = [weather_result, search_result]

        # Execute
        results = await reasoning_agent._execute_tools(tool_predictions, parallel=False)

        # Verify
        assert len(results) == 2
        assert results[0] == weather_result
        assert results[1] == search_result
        assert reasoning_agent.mcp_manager.execute_tool.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_tools_parallel_single_success(
        self,
        reasoning_agent: ReasoningAgent,
        sample_tool_prediction: ToolPrediction,
        sample_tool_result_success: ToolResult,
    ):
        """Test executing a single tool in parallel mode."""
        # Setup mock
        reasoning_agent.mcp_manager.execute_tools_parallel.return_value = [sample_tool_result_success]  # noqa: E501

        # Execute
        results = await reasoning_agent._execute_tools([sample_tool_prediction], parallel=True)

        # Verify
        assert len(results) == 1
        assert results[0] == sample_tool_result_success

        # Verify parallel execution was called
        reasoning_agent.mcp_manager.execute_tools_parallel.assert_called_once()
        call_args = reasoning_agent.mcp_manager.execute_tools_parallel.call_args[0][0]
        assert len(call_args) == 1
        assert isinstance(call_args[0], ToolRequest)
        assert call_args[0].tool_name == "get_weather"

    @pytest.mark.asyncio
    async def test_execute_tools_parallel_multiple_tools(self, reasoning_agent: ReasoningAgent):
        """Test executing multiple tools in parallel."""
        # Create multiple tool predictions
        tool_predictions = [
            ToolPrediction(
                server_name="demo_tools",
                tool_name="get_weather",
                arguments={"location": "Tokyo"},
                reasoning="Need Tokyo weather",
            ),
            ToolPrediction(
                server_name="demo_tools",
                tool_name="search_web",
                arguments={"query": "weather"},
                reasoning="Need search results",
            ),
        ]

        # Create expected results
        expected_results = [
            ToolResult(
                server_name="demo_tools",
                tool_name="get_weather",
                success=True,
                result={"temp": "22°C"},
                execution_time_ms=100.0,
            ),
            ToolResult(
                server_name="demo_tools",
                tool_name="search_web",
                success=True,
                result={"results": ["result1"]},
                execution_time_ms=200.0,
            ),
        ]

        # Setup mock
        reasoning_agent.mcp_manager.execute_tools_parallel.return_value = expected_results

        # Execute
        results = await reasoning_agent._execute_tools(tool_predictions, parallel=True)

        # Verify
        assert len(results) == 2
        assert results == expected_results

        # Verify parallel execution was called with both tools
        reasoning_agent.mcp_manager.execute_tools_parallel.assert_called_once()
        call_args = reasoning_agent.mcp_manager.execute_tools_parallel.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0].tool_name == "get_weather"
        assert call_args[1].tool_name == "search_web"

    @pytest.mark.asyncio
    async def test_execute_tools_mixed_success_failure(self, reasoning_agent: ReasoningAgent):
        """Test executing tools with mixed success and failure results."""
        tool_predictions = [
            ToolPrediction(
                server_name="demo_tools",
                tool_name="get_weather",
                arguments={"location": "Tokyo"},
                reasoning="Weather check",
            ),
            ToolPrediction(
                server_name="demo_tools",
                tool_name="failing_tool",
                arguments={"should_fail": True},
                reasoning="Test failure",
            ),
        ]

        success_result = ToolResult(
            server_name="demo_tools",
            tool_name="get_weather",
            success=True,
            result={"temp": "22°C"},
            execution_time_ms=100.0,
        )

        failure_result = ToolResult(
            server_name="demo_tools",
            tool_name="failing_tool",
            success=False,
            error="Tool failed intentionally",
            execution_time_ms=50.0,
        )

        # Test sequential execution
        reasoning_agent.mcp_manager.execute_tool.side_effect = [success_result, failure_result]

        results = await reasoning_agent._execute_tools(tool_predictions, parallel=False)

        # Verify mixed results
        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False
        assert results[0].tool_name == "get_weather"
        assert results[1].tool_name == "failing_tool"
        assert results[1].error == "Tool failed intentionally"

    @pytest.mark.asyncio
    async def test_execute_tools_empty_list(self, reasoning_agent: ReasoningAgent):
        """Test executing an empty list of tools."""
        results = await reasoning_agent._execute_tools([], parallel=False)
        assert results == []

        # Setup mock for parallel execution with empty list
        reasoning_agent.mcp_manager.execute_tools_parallel.return_value = []

        results = await reasoning_agent._execute_tools([], parallel=True)
        assert results == []

    @pytest.mark.asyncio
    async def test_tool_prediction_to_mcp_request_conversion(self, reasoning_agent: ReasoningAgent):  # noqa: E501
        """Test that ToolPrediction objects are correctly converted to MCP ToolRequest objects."""
        tool_prediction = ToolPrediction(
            server_name="demo_tools",
            tool_name="get_weather",
            arguments={"location": "Tokyo", "units": "metric"},
            reasoning="User requested Tokyo weather in metric units",
        )

        # Mock the MCP manager to capture the converted request
        mock_result = ToolResult(
            server_name="demo_tools",
            tool_name="get_weather",
            success=True,
            result={"temp": "22°C"},
            execution_time_ms=100.0,
        )
        reasoning_agent.mcp_manager.execute_tool.return_value = mock_result

        # Execute
        await reasoning_agent._execute_tools([tool_prediction], parallel=False)

        # Verify the conversion happened correctly
        reasoning_agent.mcp_manager.execute_tool.assert_called_once()
        converted_request = reasoning_agent.mcp_manager.execute_tool.call_args[0][0]

        assert isinstance(converted_request, ToolRequest)
        assert converted_request.server_name == "demo_tools"
        assert converted_request.tool_name == "get_weather"
        assert converted_request.arguments == {"location": "Tokyo", "units": "metric"}
        # Note: reasoning field is not part of ToolRequest, it's only in ToolPrediction

    @pytest.mark.asyncio
    async def test_execute_tools_preserves_result_types(self, reasoning_agent: ReasoningAgent):
        """Test that _execute_tools preserves exact ToolResult types and doesn't modify them."""
        # Create a tool result with various data types
        complex_result = ToolResult(
            server_name="demo_tools",
            tool_name="complex_tool",
            success=True,
            result={
                "string_data": "test",
                "number_data": 42,
                "float_data": 3.14,
                "bool_data": True,
                "list_data": [1, 2, 3],
                "nested_dict": {"key": "value"},
            },
            execution_time_ms=123.45,
        )

        tool_prediction = ToolPrediction(
            server_name="demo_tools",
            tool_name="complex_tool",
            arguments={},
            reasoning="Test",
        )

        reasoning_agent.mcp_manager.execute_tool.return_value = complex_result

        results = await reasoning_agent._execute_tools([tool_prediction], parallel=False)

        # Verify the result is returned exactly as provided by MCP manager
        assert len(results) == 1
        returned_result = results[0]
        assert returned_result is complex_result  # Same object reference
        assert returned_result.result["string_data"] == "test"
        assert returned_result.result["number_data"] == 42
        assert returned_result.result["float_data"] == 3.14
        assert returned_result.result["bool_data"] is True
        assert returned_result.result["list_data"] == [1, 2, 3]
        assert returned_result.result["nested_dict"]["key"] == "value"
        assert returned_result.execution_time_ms == 123.45
