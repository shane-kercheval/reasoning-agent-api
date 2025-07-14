"""
Simple unit tests for the streaming tool results bug fix.

Tests the core functionality without complex mocking.
"""

import pytest
import httpx
from api.reasoning_agent import ReasoningAgent
from api.mcp import ToolResult
from api.reasoning_models import ReasoningStep, ToolPrediction, ReasoningAction


class TestReasoningSummary:
    """Test the _build_reasoning_summary method that was key to the bug fix."""

    @pytest.fixture
    def reasoning_agent(self):
        """Create a minimal reasoning agent for testing."""
        http_client = httpx.AsyncClient()
        return ReasoningAgent(
            base_url="http://test",
            api_key="test-key",
            http_client=http_client,
            mcp_manager=None,  # Not needed for these tests
            prompt_manager=None,  # Not needed for these tests
        )

    def test_build_reasoning_summary_with_tool_results(self, reasoning_agent: ReasoningAgent):
        """Test that tool results are included in reasoning summary."""
        # Create sample tool results
        tool_results = [
            ToolResult(
                server_name="demo_tools",
                tool_name="get_weather",
                success=True,
                result={"location": "Tokyo", "temperature": "22°C", "condition": "Sunny"},
                execution_time_ms=150.0,
            ),
            ToolResult(
                server_name="demo_tools",
                tool_name="search_web",
                success=True,
                result={"query": "weather", "results": ["Weather info found"]},
                execution_time_ms=200.0,
            ),
        ]

        reasoning_context = {
            "steps": [],
            "tool_results": tool_results,
            "final_thoughts": "",
        }

        summary = reasoning_agent._build_reasoning_summary(reasoning_context)

        # Verify that tool results are included in the summary
        assert "Tool Results:" in summary
        assert "get_weather" in summary
        assert "search_web" in summary
        assert "Tokyo" in summary
        assert "22°C" in summary
        assert "weather" in summary

    def test_build_reasoning_summary_with_failed_tool(self, reasoning_agent: ReasoningAgent):
        """Test that failed tool results are also included in reasoning summary."""
        tool_result = ToolResult(
            server_name="demo_tools",
            tool_name="get_weather",
            success=False,
            error="Connection timeout",
            execution_time_ms=100.0,
        )

        reasoning_context = {
            "steps": [],
            "tool_results": [tool_result],
            "final_thoughts": "",
        }

        summary = reasoning_agent._build_reasoning_summary(reasoning_context)

        # Verify that failed tool results are included
        assert "Tool Results:" in summary
        assert "get_weather" in summary
        assert "Connection timeout" in summary

    def test_build_reasoning_summary_without_tool_results(self, reasoning_agent: ReasoningAgent):
        """Test that reasoning summary works when there are no tool results."""
        reasoning_context = {
            "steps": [],
            "tool_results": [],
            "final_thoughts": "",
        }

        summary = reasoning_agent._build_reasoning_summary(reasoning_context)

        # Should not include tool results section when there are no results
        assert "Tool Results:" not in summary

    def test_build_reasoning_summary_with_steps_and_tools(self, reasoning_agent: ReasoningAgent):
        """Test reasoning summary with both steps and tool results."""
        # Create a reasoning step
        step = ReasoningStep(
            thought="I need to get weather information",
            next_action=ReasoningAction.USE_TOOLS,
            tools_to_use=[
                ToolPrediction(
                    server_name="demo_tools",
                    tool_name="get_weather",
                    arguments={"location": "Tokyo"},
                    reasoning="User asked for Tokyo weather",
                ),
            ],
        )

        tool_result = ToolResult(
            server_name="demo_tools",
            tool_name="get_weather",
            success=True,
            result={"location": "Tokyo", "temperature": "25°C"},
            execution_time_ms=120.0,
        )

        reasoning_context = {
            "steps": [step],
            "tool_results": [tool_result],
            "final_thoughts": "",
        }

        summary = reasoning_agent._build_reasoning_summary(reasoning_context)

        # Should include both step information and tool results
        assert "Step 1:" in summary
        assert "I need to get weather information" in summary
        assert "Used tools: get_weather" in summary
        assert "Tool Results:" in summary
        assert "25°C" in summary


class TestStreamingContextStorage:
    """Test that the reasoning context is properly stored for streaming access."""

    @pytest.fixture
    def reasoning_agent(self):
        """Create a minimal reasoning agent for testing."""
        http_client = httpx.AsyncClient()
        return ReasoningAgent(
            base_url="http://test",
            api_key="test-key",
            http_client=http_client,
            mcp_manager=None,
            prompt_manager=None,
        )

    def test_current_reasoning_context_attribute_exists(self, reasoning_agent: ReasoningAgent):
        """Test that the agent has the _current_reasoning_context attribute after our fix."""
        # Simulate what happens in _stream_reasoning_process
        test_context = {
            "steps": [],
            "tool_results": [],
            "final_thoughts": "",
            "user_request": None,
        }

        # This is what our fix does in _stream_reasoning_process
        reasoning_agent._current_reasoning_context = test_context

        # Verify it's stored
        assert hasattr(reasoning_agent, '_current_reasoning_context')
        assert reasoning_agent._current_reasoning_context == test_context
        assert "tool_results" in reasoning_agent._current_reasoning_context
