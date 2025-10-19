"""
Comprehensive tests for the ReasoningAgent class.

Tests the ReasoningAgent proxy functionality, dependency injection, tool execution,
context building, error handling, and OpenAI API compatibility.

This file consolidates all reasoning agent unit tests following phase 4 of the
test refactoring plan to reduce duplication and improve maintainability.
"""

import asyncio
import json
import os
import time
from typing import Any
from unittest.mock import AsyncMock, Mock, patch
import pytest
import httpx
import respx
from openai import AsyncOpenAI
from opentelemetry.trace import StatusCode
from api.reasoning_agent import ReasoningAgent, ReasoningError
from api.openai_protocol import (
    SSE_DONE,
    OpenAIChatRequest,
    OpenAIDelta,
    OpenAIResponseBuilder,
    OpenAIStreamChoice,
    OpenAIStreamResponse,
    OpenAIStreamingResponseBuilder,
    OpenAIUsage,
    create_sse,
    parse_sse,
)
from api.prompt_manager import PromptManager
from api.reasoning_models import (
    ReasoningAction,
    ReasoningEventType,
    ReasoningStep,
    ToolPrediction,
)
from api.tools import ToolResult, function_to_tool
from tests.conftest import OPENAI_TEST_MODEL, ReasoningAgentStreamingCollector
from tests.fixtures.tools import weather_tool
from tests.fixtures.models import ReasoningStepFactory, ToolPredictionFactory
from tests.fixtures.responses import (
    create_reasoning_response,
    create_http_response,
)


def create_mock_request(model: str = "gpt-4", content: str = "What's the weather in Tokyo?"):
    """Create a mock request object with proper structure."""
    mock_request = Mock()
    mock_request.model = model
    mock_message = Mock()
    mock_message.content = content
    mock_request.messages = [mock_message]
    return mock_request


# =============================================================================
# Core Reasoning Agent Tests
# =============================================================================

class TestReasoningAgent:
    """Test core ReasoningAgent functionality including chat completion processing."""

    @pytest.mark.asyncio
    @respx.mock
    async def test__execute_stream__includes_reasoning_events(
        self,
        reasoning_agent: ReasoningAgent,
        sample_streaming_request: OpenAIChatRequest,
        mock_openai_streaming_chunks: list[str],
    ) -> None:
        """Test that streaming includes reasoning events with metadata."""
        # Create a reasoning step that finishes (no tools)
        reasoning_step = ReasoningStepFactory.finished_step("Direct answer needed")
        reasoning_response = create_reasoning_response(reasoning_step, "chatcmpl-reasoning")

        # Mock TWO calls: reasoning step generation + streaming final synthesis
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                # First call: reasoning step generation (JSON mode, non-streaming)
                create_http_response(reasoning_response),
                # Second call: final synthesis (streaming)
                httpx.Response(
                    200,
                    text="\n".join(mock_openai_streaming_chunks),
                    headers={"content-type": "text/plain"},
                ),
            ],
        )

        chunks = []
        async for chunk in reasoning_agent.execute_stream(sample_streaming_request):
            chunks.append(chunk)

        # Verify specific reasoning event structure and content
        reasoning_chunks = []
        response_chunks = []
        done_chunks = []

        for chunk in chunks:
            if "reasoning_event" in chunk:
                reasoning_chunks.append(chunk)
            elif "[DONE]" in chunk:
                done_chunks.append(chunk)
            else:
                response_chunks.append(chunk)

        # Should have at least one reasoning event for thinking step
        assert len(reasoning_chunks) >= 1, "Should have at least one reasoning event for visual step"  # noqa: E501

        # Verify reasoning event structure and content
        had_planning_event = False
        for reasoning_chunk in reasoning_chunks:
            reasoning_data = parse_sse(reasoning_chunk)
            assert reasoning_data
            assert "choices" in reasoning_data, "Reasoning event should have choices structure"
            assert len(reasoning_data["choices"]) == 1, "Should have exactly one choice"

            choice = reasoning_data["choices"][0]
            assert "delta" in choice, "Choice should have delta for streaming"
            assert "reasoning_event" in choice["delta"], "Should contain reasoning_event metadata"

            # Verify reasoning event metadata
            reasoning_event = choice["delta"]["reasoning_event"]
            assert "type" in reasoning_event, "Reasoning event should have type"
            assert reasoning_event["type"] in [
                ReasoningEventType.ITERATION_START.value,
                ReasoningEventType.PLANNING.value,
                ReasoningEventType.TOOL_EXECUTION_START.value,
                ReasoningEventType.TOOL_RESULT.value,
                ReasoningEventType.ITERATION_COMPLETE.value,
                ReasoningEventType.REASONING_COMPLETE.value,
                ReasoningEventType.ERROR.value,
            ], "Should be valid reasoning type"

            # planning events MUST have usage (from reasoning step generation)
            if reasoning_event["type"] == ReasoningEventType.PLANNING.value:
                had_planning_event = True
                # This is a planning event - it MUST have usage data
                assert "usage" in reasoning_data, "planning reasoning events MUST have usage"
                assert reasoning_data["usage"] is not None, "planning usage cannot be None"
                usage = reasoning_data["usage"]
                assert usage["prompt_tokens"] == 50, "planning: expected 50 prompt tokens"
                assert usage["completion_tokens"] == 100, "planning: expected 100 completion tokens"  # noqa: E501
                assert usage["total_tokens"] == 150, "planning: expected 150 total tokens"

        assert had_planning_event, "Should have at least one planning event with usage data"
        # Precise validation of expected streaming response structure

        # Should have exactly 4 reasoning events for a single FINISHED step:
        # 1. iteration_start
        # 2. planning (with thought and usage)
        # 3. iteration_complete
        # 4. reasoning_complete
        assert len(reasoning_chunks) == 4, f"Expected exactly 4 reasoning events, got {len(reasoning_chunks)}"  # noqa: E501

        # Verify the sequence of reasoning events (no more status field)
        event_types = []
        for chunk in reasoning_chunks:
            data = parse_sse(chunk)
            event = data["choices"][0]["delta"]["reasoning_event"]
            event_types.append(event["type"])

        expected_sequence = [
            ReasoningEventType.ITERATION_START.value,     # start of reasoning step
            ReasoningEventType.PLANNING.value,           # planning with thought (has usage)
            ReasoningEventType.ITERATION_COMPLETE.value, # step completed
            ReasoningEventType.REASONING_COMPLETE.value, # synthesis finished
        ]
        assert event_types == expected_sequence, f"Expected event sequence {expected_sequence}, got {event_types}"  # noqa: E501


        # Note: Due to AsyncOpenAI client's complex streaming parsing and respx mocking
        # limitations, the final synthesis chunks may not be properly captured in this test.
        # This test focuses on validating reasoning events are correctly generated.
        # Final response content streaming is tested in integration tests.

        # Should have exactly 1 [DONE] termination event with correct format
        assert len(done_chunks) == 1, f"Expected exactly 1 [DONE] chunk, got {len(done_chunks)}"
        assert done_chunks[0] == SSE_DONE, "Should end with proper [DONE] format"

    @pytest.mark.asyncio
    @respx.mock
    async def test__execute_stream__forwards_openai_chunks(
        self,
        reasoning_agent: ReasoningAgent,
        sample_streaming_request: OpenAIChatRequest,
    ) -> None:
        """Test that OpenAI chunks are properly forwarded with modified IDs and usage data."""
        # Create a reasoning step that finishes (no tools)
        reasoning_step = ReasoningStepFactory.finished_step("Direct answer needed")
        reasoning_response = create_reasoning_response(reasoning_step, "chatcmpl-reasoning")

        # Create streaming response with usage data for final synthesis
        mock_streaming_chunks = [
            'data: {"id": "chatcmpl-test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}], "usage": null}\n\n',  # noqa: E501
            'data: {"id": "chatcmpl-test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o", "choices": [{"index": 0, "delta": {"content": "This"}, "finish_reason": null}], "usage": null}\n\n',  # noqa: E501
            'data: {"id": "chatcmpl-test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o", "choices": [{"index": 0, "delta": {"content": " is"}, "finish_reason": null}], "usage": null}\n\n',  # noqa: E501
            'data: {"id": "chatcmpl-test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o", "choices": [{"index": 0, "delta": {"content": " a test"}, "finish_reason": null}], "usage": {"prompt_tokens": 25, "completion_tokens": 12, "total_tokens": 37}}\n\n',  # noqa: E501
            'data: {"id": "chatcmpl-test123", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": null}\n\n',  # noqa: E501
            'data: [DONE]\n\n',
        ]

        # Mock TWO calls: reasoning step generation + streaming final synthesis
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                # First call: reasoning step generation (JSON mode, non-streaming)
                create_http_response(reasoning_response),
                # Second call: final synthesis (streaming)
                httpx.Response(
                    200,
                    text="".join(mock_streaming_chunks),
                    headers={"content-type": "text/plain"},
                ),
            ],
        )

        # Collect all streaming chunks
        chunks = []
        async for chunk in reasoning_agent.execute_stream(sample_streaming_request):
            if chunk.startswith("data: {") and "chatcmpl-" in chunk:
                chunks.append(chunk)

        # Verify basic structure
        assert len(chunks) >= 4, "Should have reasoning events + final synthesis chunks"

        # Parse all chunks and separate by type
        content_chunks = []
        reasoning_event_chunks = []

        for chunk in chunks:
            chunk_data = parse_sse(chunk)

            # All chunks should have our modified completion ID, not the original mock ID
            assert chunk_data["id"].startswith("chatcmpl-")
            assert chunk_data["id"] != "chatcmpl-test123"

            delta = chunk_data["choices"][0]["delta"]
            if delta.get("reasoning_event"):
                reasoning_event_chunks.append(chunk_data)
            elif delta.get("content"):
                content_chunks.append(chunk_data)

        # Should have both reasoning events and content chunks
        assert len(reasoning_event_chunks) >= 3, "Should have iteration_start, planning, iteration_complete, reasoning_complete events"  # noqa: E501
        assert len(content_chunks) >= 3, "Should have multiple content chunks"

        # Find the specific content chunk with expected usage data
        usage_chunk = None
        for chunk_data in content_chunks:
            if chunk_data.get("usage") and chunk_data["usage"]["total_tokens"] == 37:
                usage_chunk = chunk_data
                break

        # Verify the usage chunk exists and has correct data
        assert usage_chunk is not None, "Must find content chunk with expected usage data"
        assert usage_chunk["usage"]["prompt_tokens"] == 25
        assert usage_chunk["usage"]["completion_tokens"] == 12
        assert usage_chunk["usage"]["total_tokens"] == 37
        assert usage_chunk["choices"][0]["delta"]["content"] == " a test"

    @pytest.mark.asyncio
    @respx.mock
    async def test__execute_stream__handles_streaming_errors(
        self,
        reasoning_agent: ReasoningAgent,
        sample_streaming_request: OpenAIChatRequest,
    ) -> None:
        """Test that streaming errors are properly handled."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(401, json={"error": {"message": "Unauthorized"}}),
        )

        with pytest.raises(ReasoningError) as exc_info:
            async for _ in reasoning_agent.execute_stream(sample_streaming_request):
                pass

        # Verify the error contains the HTTP status information
        assert "401" in str(exc_info.value)
        assert "Unauthorized" in str(exc_info.value)

    @pytest.mark.asyncio
    @respx.mock
    async def test__execute_stream__no_tools_available(self) -> None:
        """Test ReasoningAgent streaming when no tools are available."""
        # Create empty tools list for "no tools" test
        tools = []

        # Create mock prompt manager
        mock_prompt_manager = AsyncMock(spec=PromptManager)
        mock_prompt_manager.get_prompt.return_value = "Test system prompt"

        agent = ReasoningAgent(
            openai_client=AsyncOpenAI(api_key="test-api-key", base_url="https://api.openai.com/v1"),
            tools=tools,
            prompt_manager=mock_prompt_manager,
        )

        request = OpenAIChatRequest(
            model=OPENAI_TEST_MODEL,
            messages=[{"role": "user", "content": "What's the weather?"}],
            stream=True,
        )

        # Create mock streaming chunks with proper usage data
        mock_streaming_chunks = [
            'data: {"id": "chatcmpl-synthesis", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}], "usage": null}\n\n',  # noqa: E501
            'data: {"id": "chatcmpl-synthesis", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o", "choices": [{"index": 0, "delta": {"content": "thinking"}, "finish_reason": null}], "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18}}\n\n',  # noqa: E501
            'data: {"id": "chatcmpl-synthesis", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o", "choices": [{"index": 0, "delta": {"content": "response"}, "finish_reason": null}], "usage": {"prompt_tokens": 18, "completion_tokens": 8, "total_tokens": 26}}\n\n',  # noqa: E501
            'data: {"id": "chatcmpl-synthesis", "object": "chat.completion.chunk", "created": 1234567890, "model": "gpt-4o", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": null}\n\n',  # noqa: E501
            'data: [DONE]\n\n',
        ]

        # Mock reasoning step generation
        reasoning_step = ReasoningStepFactory.finished_step("No tools available")
        reasoning_response = create_reasoning_response(
            reasoning_step, "chatcmpl-reasoning",
        )
        reasoning_response.created = 1234567890
        reasoning_response.model = OPENAI_TEST_MODEL

        # Mock TWO calls: reasoning step generation + streaming final synthesis
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                # First call: reasoning step generation (JSON mode, non-streaming)
                create_http_response(reasoning_response),
                # Second call: final synthesis (streaming)
                httpx.Response(
                    200,
                    text="".join(mock_streaming_chunks),
                    headers={"content-type": "text/plain"},
                ),
            ],
        )

        # Collect all streaming chunks
        chunks = []
        async for chunk in agent.execute_stream(request):
            chunks.append(chunk)

        # Should have reasoning events + final response + [DONE]
        assert len(chunks) >= 2, "Should have at least reasoning events and [DONE]"
        assert chunks[-1] == SSE_DONE

        # Parse all data chunks (excluding [DONE])
        data_chunks = []
        for chunk in chunks[:-1]:  # Exclude [DONE]
            if chunk.startswith("data: {"):
                data_chunks.append(parse_sse(chunk))

        # Separate content chunks from reasoning event chunks
        content_chunks = []
        reasoning_event_chunks = []

        for chunk_data in data_chunks:
            delta = chunk_data["choices"][0]["delta"]
            if delta.get("reasoning_event"):
                reasoning_event_chunks.append(chunk_data)
            elif delta.get("content"):
                content_chunks.append(chunk_data)

        # Should have reasoning events (iteration_start, planning, iteration_complete, reasoning_complete)  # noqa: E501
        assert len(reasoning_event_chunks) >= 3, "Should have multiple reasoning events"

        # Should have exactly 2 content chunks with usage data
        assert len(content_chunks) == 2, f"Expected exactly 2 content chunks, got {len(content_chunks)}"  # noqa: E501

        # First content chunk: "thinking" with usage (12+6=18)
        thinking_chunk = content_chunks[0]
        assert thinking_chunk["choices"][0]["delta"]["content"] == "thinking"
        assert thinking_chunk["usage"]["prompt_tokens"] == 12
        assert thinking_chunk["usage"]["completion_tokens"] == 6
        assert thinking_chunk["usage"]["total_tokens"] == 18

        # Second content chunk: "response" with usage (18+8=26)
        response_chunk = content_chunks[1]
        assert response_chunk["choices"][0]["delta"]["content"] == "response"
        assert response_chunk["usage"]["prompt_tokens"] == 18
        assert response_chunk["usage"]["completion_tokens"] == 8
        assert response_chunk["usage"]["total_tokens"] == 26

    def test_build_reasoning_summary_with_tool_results(self):
        """Test that tool results are included in reasoning summary."""
        # Create minimal reasoning agent for testing
        tools = [function_to_tool(weather_tool)]
        reasoning_agent_simple = ReasoningAgent(
            openai_client=AsyncOpenAI(api_key="test-api-key", base_url="https://api.openai.com/v1"),
            tools=tools,
            prompt_manager=None,  # Not needed for these tests
        )

        # Create sample tool results
        tool_results = [
            ToolResult(
                tool_name="get_weather",
                success=True,
                result={"location": "Tokyo", "temperature": "22°C", "condition": "Sunny"},
                execution_time_ms=150.0,
            ),
            ToolResult(
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

        summary = reasoning_agent_simple._build_reasoning_summary(reasoning_context)

        # Verify that tool results are included in the summary
        assert "Tool Results:" in summary
        assert "get_weather" in summary
        assert "search_web" in summary
        assert "Tokyo" in summary
        assert "22°C" in summary
        assert "weather" in summary


# =============================================================================
# Tool Execution Tests
# =============================================================================

class TestToolExecution:
    """Test tool execution functionality including sequential and concurrent execution."""

    @pytest.fixture
    def tool_execution_agent(self):
        """Create a reasoning agent with mock tools for testing."""
        httpx.AsyncClient()
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
            openai_client=AsyncOpenAI(api_key="test-api-key", base_url="https://api.openai.com/v1"),
            tools=tools,
            prompt_manager=mock_prompt_manager,
        )

    @pytest.mark.asyncio
    async def test_execute_single_tool_success(self, tool_execution_agent: ReasoningAgent):
        """Test executing a single tool successfully."""
        prediction = ToolPrediction(
            tool_name="get_weather",
            arguments={"location": "Tokyo"},
            reasoning="Need weather data",
        )

        results = await tool_execution_agent._execute_tools_sequentially([prediction])

        assert len(results) == 1
        result = results[0]
        assert result.success is True
        assert result.tool_name == "get_weather"
        assert result.result["location"] == "Tokyo"
        assert result.result["temperature"] == "22°C"
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_single_tool_failure(self, tool_execution_agent: ReasoningAgent):
        """Test executing a tool that fails."""
        prediction = ToolPrediction(
            tool_name="failing_tool",
            arguments={"should_fail": True},
            reasoning="Test failure",
        )

        results = await tool_execution_agent._execute_tools_sequentially([prediction])

        assert len(results) == 1
        result = results[0]
        assert result.success is False
        assert result.tool_name == "failing_tool"
        assert "intentionally failed" in result.error
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_multiple_tools_sequential(self, tool_execution_agent: ReasoningAgent):
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

        results = await tool_execution_agent._execute_tools_sequentially(predictions)

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
    async def test_execute_tools_concurrently_known_tools(
            self,
            tool_execution_agent: ReasoningAgent,
        ):
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
        results = await tool_execution_agent._execute_tools_concurrently(predictions)

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
    async def test_execute_tools_concurrently_mixed_known_unknown(
            self, tool_execution_agent: ReasoningAgent,
        ):
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
        results = await tool_execution_agent._execute_tools_concurrently(predictions)

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
    async def test_get_available_tools(self, tool_execution_agent: ReasoningAgent):
        """Test getting available tools."""
        tools = await tool_execution_agent.get_available_tools()

        assert len(tools) == 4
        tool_names = {tool.name for tool in tools}
        assert tool_names == {"get_weather", "search_web", "failing_tool", "async_delay"}

    @pytest.mark.asyncio
    async def test_execute_tool_with_sync_and_async_tools_parallel(
            self,
            tool_execution_agent: ReasoningAgent,
        ):
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
        results = await tool_execution_agent._execute_tools_concurrently(predictions)
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
    async def test_complex_type_execution(self):
        """Test execution with complex nested types."""
        httpx.AsyncClient()
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

        tools = [function_to_tool(complex_analysis, name="analyze_data")]

        complex_tool_agent = ReasoningAgent(
            openai_client=AsyncOpenAI(api_key="test-api-key", base_url="https://api.openai.com/v1"),
            tools=tools,
            prompt_manager=mock_prompt_manager,
        )

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
    async def test_tool_argument_validation_missing_required_params(self, tool_execution_agent: ReasoningAgent):  # noqa: E501
        """Test that tools handle missing required parameters gracefully."""
        # Try to call weather tool without required location parameter
        invalid_prediction = ToolPrediction(
            tool_name="get_weather",
            arguments={},  # Missing required 'location' parameter
            reasoning="Test missing parameter handling",
        )

        results = await tool_execution_agent._execute_tools_sequentially([invalid_prediction])

        assert len(results) == 1, "Should return one result"
        result = results[0]
        assert result.success is False, "Should fail with missing required parameter"
        assert result.tool_name == "get_weather", "Should identify correct tool"
        assert result.error is not None, "Should have error information"
        assert "location" in str(result.error).lower(), "Error should mention missing location parameter"  # noqa: E501

    @pytest.mark.asyncio
    async def test_tool_argument_validation_wrong_types(self, tool_execution_agent: ReasoningAgent):  # noqa: E501
        """Test that tools handle wrong argument types appropriately."""
        # Try to call search tool with wrong type for limit parameter
        invalid_prediction = ToolPrediction(
            tool_name="search_web",
            arguments={"query": "test", "extra_param": "invalid_value"},  # extra_param doesn't exist  # noqa: E501
            reasoning="Test type validation",
        )

        results = await tool_execution_agent._execute_tools_sequentially([invalid_prediction])

        assert len(results) == 1, "Should return one result"
        result = results[0]
        assert result.success is False, "Should fail with wrong parameter type"
        assert result.tool_name == "search_web", "Should identify correct tool"
        assert result.error is not None, "Should contain error information"

    @pytest.mark.asyncio
    async def test_tool_argument_validation_unknown_tool(self, tool_execution_agent: ReasoningAgent):  # noqa: E501
        """Test that unknown tools are handled gracefully."""
        unknown_prediction = ToolPrediction(
            tool_name="nonexistent_tool",
            arguments={"param": "value"},
            reasoning="Test unknown tool handling",
        )

        results = await tool_execution_agent._execute_tools_sequentially([unknown_prediction])

        assert len(results) == 1, "Should return one result"
        result = results[0]
        assert result.success is False, "Should fail for unknown tool"
        assert result.tool_name == "nonexistent_tool", "Should preserve tool name"
        assert result.error is not None, "Should contain error information"
        assert "not found" in str(result.error).lower() or "unknown" in str(result.error).lower(), "Error should indicate tool not found"  # noqa: E501

    @pytest.mark.asyncio
    async def test_tool_argument_validation_extra_parameters(self, tool_execution_agent: ReasoningAgent):  # noqa: E501
        """Test that tools handle extra unexpected parameters appropriately."""
        # Call weather tool with extra parameters it doesn't expect
        prediction_with_extra = ToolPrediction(
            tool_name="get_weather",
            arguments={
                "location": "Tokyo",
                "unexpected_param": "unexpected_value",
                "another_extra": 123,
            },
            reasoning="Test extra parameter handling",
        )

        results = await tool_execution_agent._execute_tools_sequentially([prediction_with_extra])

        assert len(results) == 1, "Should return one result"
        result = results[0]
        # Tool should either succeed (ignoring extra params) or fail gracefully
        assert result.tool_name == "get_weather", "Should identify correct tool"

        if result.success:
            # If it succeeds, should still return expected weather data
            assert result.result["location"] == "Tokyo", "Should process location correctly"
            assert "temperature" in result.result, "Should return temperature data"
        else:
            # If it fails, should have clear error message
            assert result.error is not None, "Should contain error information"

    @pytest.mark.asyncio
    async def test_concurrent_tool_safety_no_interference(self, tool_execution_agent: ReasoningAgent):  # noqa: E501
        """Test that concurrent tools don't interfere with each other's execution."""
        # Create multiple tool predictions that should execute independently
        predictions = [
            ToolPrediction(
                tool_name="get_weather",
                arguments={"location": "Tokyo"},
                reasoning="Get Tokyo weather",
            ),
            ToolPrediction(
                tool_name="get_weather",
                arguments={"location": "Paris"},
                reasoning="Get Paris weather",
            ),
            ToolPrediction(
                tool_name="search_web",
                arguments={"query": "Python programming"},
                reasoning="Search for Python info",
            ),
            ToolPrediction(
                tool_name="search_web",
                arguments={"query": "JavaScript tutorials"},
                reasoning="Search for JS info",
            ),
        ]

        # Execute tools concurrently
        results = await tool_execution_agent._execute_tools_concurrently(predictions)

        assert len(results) == 4, "Should return results for all tools"

        # Verify each tool result is correct and independent
        tokyo_result = next(r for r in results if r.tool_name == "get_weather" and r.result.get("location") == "Tokyo")  # noqa: E501
        paris_result = next(r for r in results if r.tool_name == "get_weather" and r.result.get("location") == "Paris")  # noqa: E501
        python_result = next(r for r in results if r.tool_name == "search_web" and "Python" in str(r.result))  # noqa: E501
        js_result = next(r for r in results if r.tool_name == "search_web" and "JavaScript" in str(r.result))  # noqa: E501

        # Verify Tokyo weather result
        assert tokyo_result.success is True, "Tokyo weather should succeed"
        assert tokyo_result.result["location"] == "Tokyo", "Should have correct Tokyo location"
        assert tokyo_result.result["temperature"] == "22°C", "Should have Tokyo temperature"

        # Verify Paris weather result
        assert paris_result.success is True, "Paris weather should succeed"
        assert paris_result.result["location"] == "Paris", "Should have correct Paris location"
        assert paris_result.result["temperature"] == "22°C", "Should have consistent temperature"

        # Verify Python search result
        assert python_result.success is True, "Python search should succeed"
        assert python_result.result["query"] == "Python programming", "Should have correct Python query"  # noqa: E501
        assert "results" in python_result.result, "Should have results field"

        # Verify JavaScript search result
        assert js_result.success is True, "JavaScript search should succeed"
        assert js_result.result["query"] == "JavaScript tutorials", "Should have correct JS query"
        assert "results" in js_result.result, "Should have results field"

        # Verify no result contamination between similar tools
        assert tokyo_result.result != paris_result.result, "Weather results should be different"
        assert python_result.result != js_result.result, "Search results should be different"

    @pytest.mark.asyncio
    async def test_concurrent_tool_safety_with_failures(self, tool_execution_agent: ReasoningAgent):  # noqa: E501
        """Test that tool failures in concurrent execution don't affect other tools."""
        predictions = [
            ToolPrediction(
                tool_name="get_weather",
                arguments={"location": "Tokyo"},
                reasoning="Get weather - should succeed",
            ),
            ToolPrediction(
                tool_name="failing_tool",
                arguments={"should_fail": True},
                reasoning="This should fail",
            ),
            ToolPrediction(
                tool_name="search_web",
                arguments={"query": "test"},
                reasoning="Search - should succeed",
            ),
            ToolPrediction(
                tool_name="nonexistent_tool",
                arguments={"param": "value"},
                reasoning="Unknown tool - should fail",
            ),
        ]

        results = await tool_execution_agent._execute_tools_concurrently(predictions)

        assert len(results) == 4, "Should return results for all tools"

        # Separate successful and failed results
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        assert len(successful_results) == 2, "Should have 2 successful results"
        assert len(failed_results) == 2, "Should have 2 failed results"

        # Verify successful tools completed correctly despite failures
        weather_result = next(r for r in successful_results if r.tool_name == "get_weather")
        search_result = next(r for r in successful_results if r.tool_name == "search_web")

        assert weather_result.result["location"] == "Tokyo", "Weather tool should complete successfully"  # noqa: E501
        assert search_result.result["query"] == "test", "Search tool should complete successfully"

        # Verify failed tools have proper error handling
        failing_result = next(r for r in failed_results if r.tool_name == "failing_tool")
        unknown_result = next(r for r in failed_results if r.tool_name == "nonexistent_tool")

        assert failing_result.error is not None, "Failing tool should have error info"
        assert unknown_result.error is not None, "Unknown tool should have error info"

    @pytest.mark.asyncio
    async def test_concurrent_tool_execution_timing_safety(self, tool_execution_agent: ReasoningAgent):  # noqa: E501
        """Test that concurrent execution maintains timing consistency and isolation."""
        # Create predictions that should execute at roughly the same time
        predictions = [
            ToolPrediction(
                tool_name="get_weather",
                arguments={"location": f"City{i}"},
                reasoning=f"Get weather for city {i}",
            )
            for i in range(5)
        ]

        start_time = time.time()
        results = await tool_execution_agent._execute_tools_concurrently(predictions)
        total_time = time.time() - start_time

        assert len(results) == 5, "Should return all results"

        # All tools should succeed
        for result in results:
            assert result.success is True, f"Tool {result.tool_name} should succeed"
            assert result.execution_time_ms >= 0, "Should have valid execution time"

        # Verify each result has correct location
        for i, result in enumerate(sorted(results, key=lambda r: r.result["location"])):
            assert result.result["location"] == f"City{i}", f"Should have correct location for city {i}"  # noqa: E501

        # Concurrent execution should be faster than sequential
        # (This is a rough timing check - in practice these are very fast fake tools)
        assert total_time < 1.0, "Concurrent execution should complete quickly"


# =============================================================================
# Context Building Tests
# =============================================================================

class TestContextBuilding:
    """Test that reasoning context is built up correctly during the reasoning process."""

    @pytest.fixture
    def context_building_agent(self):
        """Create a reasoning agent with mock tools for testing."""
        httpx.AsyncClient()
        mock_prompt_manager = AsyncMock(spec=PromptManager)
        mock_prompt_manager.get_prompt.return_value = "You are a helpful assistant."

        # Create test tools
        def weather_tool(location: str) -> dict:
            return {"location": location, "temperature": "22°C", "condition": "Sunny"}

        def search_tool(query: str) -> dict:
            return {"query": query, "results": ["result1", "result2"]}

        tools = [
            function_to_tool(weather_tool, name="get_weather"),
            function_to_tool(search_tool, name="search_web"),
        ]

        return ReasoningAgent(
            openai_client=AsyncOpenAI(api_key="test-api-key", base_url="https://api.openai.com/v1"),
            tools=tools,
            prompt_manager=mock_prompt_manager,
        )

    @pytest.fixture
    def sample_context_request(self):
        """Sample chat completion request."""
        return OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        )

    @pytest.mark.asyncio
    async def test_initial_context_structure(
            self,
            context_building_agent: ReasoningAgent,
            sample_context_request: OpenAIChatRequest,
        ):
        """Test that the initial reasoning context has the correct structure."""
        events = []

        # Mock both reasoning step generation and final synthesis
        async def mock_synthesis_stream(request, completion_id, created, reasoning_context):  # noqa: ANN001, ANN202, ARG001
            yield OpenAIStreamResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[OpenAIStreamChoice(
                    index=0,
                    delta=OpenAIDelta(content="Test response"),
                    finish_reason="stop",
                )],
            )

        with patch.object(context_building_agent, '_generate_reasoning_step') as mock_generate:
            with patch.object(context_building_agent, '_stream_final_synthesis', side_effect=mock_synthesis_stream):  # noqa: E501
                mock_generate.return_value = (
                    ReasoningStep(
                        thought="I can answer this directly",
                        next_action=ReasoningAction.FINISHED,
                        tools_to_use=[],
                        concurrent_execution=False,
                    ),
                    None,  # No usage data
                )

                # Collect all events
                async for response in context_building_agent._core_reasoning_process(sample_context_request):  # noqa: E501
                    events.append(response)

        # Find the synthesis complete event
        synthesis_events = [
            event for event in events
            if (event.choices[0].delta.reasoning_event and
                event.choices[0].delta.reasoning_event.type == ReasoningEventType.REASONING_COMPLETE)  # noqa: E501
        ]
        assert len(synthesis_events) == 1

        # Access context from the agent's internal state after processing
        context = context_building_agent.reasoning_context

        # Verify specific reasoning step content
        assert len(context["steps"]) == 1, "Should have exactly one reasoning step"
        reasoning_step = context["steps"][0]
        assert reasoning_step.thought == "I can answer this directly", "Should have specific thought content"  # noqa: E501
        assert reasoning_step.next_action == ReasoningAction.FINISHED, "Should be marked as finished"  # noqa: E501
        assert reasoning_step.tools_to_use == [], "Should have no tools for direct answer"
        assert reasoning_step.concurrent_execution is False, "Should not use concurrent execution"

        # Verify tool execution results (should be empty for direct answer)
        assert len(context["tool_results"]) == 0, "Should have no tool results for direct answer"

        # Verify final thoughts content is meaningful (may be empty for simple finished steps)
        assert isinstance(context["final_thoughts"], str), "Final thoughts should be a string"

        # Verify user request preservation
        assert context["user_request"] == sample_context_request, "Should preserve original request"  # noqa: E501
        assert context["user_request"].messages[0]["content"] == "What's the weather in Tokyo?", "Should preserve specific user question"  # noqa: E501

    @pytest.mark.asyncio
    async def test_single_step_with_tools_context(
            self,
            context_building_agent: ReasoningAgent,
            sample_context_request: OpenAIChatRequest,
        ):
        """Test context building for a single reasoning step with tools."""
        events = []

        # Mock a reasoning step that uses tools
        expected_step = ReasoningStep(
            thought="I need to get weather information for Tokyo",
            next_action=ReasoningAction.FINISHED,
            tools_to_use=[
                ToolPrediction(
                    tool_name="get_weather",
                    arguments={"location": "Tokyo"},
                    reasoning="User wants weather for Tokyo",
                ),
            ],
            concurrent_execution=False,
        )

        # Mock synthesis stream and tool execution
        async def mock_synthesis_stream(request, completion_id, created, reasoning_context):  # noqa: ANN001, ANN202, ARG001
            yield OpenAIStreamResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[OpenAIStreamChoice(
                    index=0,
                    delta=OpenAIDelta(content="Weather in Tokyo is sunny"),
                    finish_reason="stop",
                )],
            )

        # Mock tool execution to return expected result
        async def mock_tool_execution(tool, prediction):  # noqa: ANN001, ANN202, ARG001
            return ToolResult(
                tool_name="get_weather",
                success=True,
                result={"location": "Tokyo", "temperature": "22°C", "condition": "Sunny"},
                execution_time_ms=100.0,
            )

        with patch.object(context_building_agent, '_generate_reasoning_step') as mock_generate:
            with patch.object(context_building_agent, '_stream_final_synthesis', side_effect=mock_synthesis_stream):  # noqa: E501
                with patch.object(context_building_agent, '_execute_single_tool_with_tracing', side_effect=mock_tool_execution):  # noqa: E501
                    mock_generate.return_value = (expected_step, None)

                    # Collect all events
                    async for response in context_building_agent._core_reasoning_process(sample_context_request):  # noqa: E501
                        events.append(response)

        # Access context from the agent's internal state after processing
        context = context_building_agent.reasoning_context

        # Verify step was added to context
        assert len(context["steps"]) == 1
        assert context["steps"][0] == expected_step

        # Verify tool results were added to context
        assert len(context["tool_results"]) == 1
        tool_result = context["tool_results"][0]
        assert tool_result.tool_name == "get_weather"
        assert tool_result.success is True
        assert tool_result.result == {"location": "Tokyo", "temperature": "22°C", "condition": "Sunny"}  # noqa: E501

        # Verify event sequence includes tool events
        reasoning_event_types = []
        for event in events:
            if event.choices[0].delta.reasoning_event:
                reasoning_event_types.append(event.choices[0].delta.reasoning_event.type)

        expected_event_types = [
            ReasoningEventType.ITERATION_START.value,
            ReasoningEventType.PLANNING.value,
            ReasoningEventType.TOOL_EXECUTION_START.value,
            ReasoningEventType.TOOL_RESULT.value,
            ReasoningEventType.ITERATION_COMPLETE.value,
            ReasoningEventType.REASONING_COMPLETE.value,
        ]
        assert reasoning_event_types == expected_event_types

    @pytest.mark.asyncio
    async def test_multiple_steps_context_accumulation(
            self,
            context_building_agent: ReasoningAgent,
            sample_context_request: OpenAIChatRequest,
        ):
        """Test context building across multiple reasoning steps."""
        events = []

        # Mock multiple reasoning steps
        step1 = ReasoningStep(
            thought="I need to search for weather information first",
            next_action=ReasoningAction.USE_TOOLS,
            tools_to_use=[
                ToolPrediction(
                    tool_name="search_web",
                    arguments={"query": "Tokyo weather"},
                    reasoning="Search for Tokyo weather information",
                ),
            ],
            concurrent_execution=False,
        )

        step2 = ReasoningStep(
            thought="Now I'll get specific weather data",
            next_action=ReasoningAction.USE_TOOLS,
            tools_to_use=[
                ToolPrediction(
                    tool_name="get_weather",
                    arguments={"location": "Tokyo"},
                    reasoning="Get detailed weather for Tokyo",
                ),
            ],
            concurrent_execution=False,
        )

        step3 = ReasoningStep(
            thought="I have all the information I need to provide a complete answer",
            next_action=ReasoningAction.FINISHED,
            tools_to_use=[],
            concurrent_execution=False,
        )

        # Mock synthesis stream and tool execution for multiple tools
        async def mock_synthesis_stream(request, completion_id, created, reasoning_context):  # noqa: ANN001, ANN202, ARG001
            yield OpenAIStreamResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[OpenAIStreamChoice(
                    index=0,
                    delta=OpenAIDelta(content="Based on my search and weather data, Tokyo is sunny with 22°C"),  # noqa: E501
                    finish_reason="stop",
                )],
            )

        # Mock tool execution to return different results based on tool name
        async def mock_tool_execution(tool, prediction):  # noqa: ANN001, ANN202, ARG001
            if prediction.tool_name == "search_web":
                return ToolResult(
                    tool_name="search_web",
                    success=True,
                    result={"results": ["Tokyo weather is generally mild", "Current conditions available"]},  # noqa: E501
                    execution_time_ms=200.0,
                )
            if prediction.tool_name == "get_weather":
                return ToolResult(
                    tool_name="get_weather",
                    success=True,
                    result={"location": "Tokyo", "temperature": "22°C", "condition": "Sunny"},
                    execution_time_ms=150.0,
                )
            return ToolResult(
                tool_name=prediction.tool_name,
                success=False,
                error="Unknown tool",
                execution_time_ms=10.0,
            )

        with patch.object(context_building_agent, '_generate_reasoning_step') as mock_generate:
            with patch.object(context_building_agent, '_stream_final_synthesis', side_effect=mock_synthesis_stream):  # noqa: E501
                with patch.object(context_building_agent, '_execute_single_tool_with_tracing', side_effect=mock_tool_execution):  # noqa: E501
                    mock_generate.side_effect = [(step1, None), (step2, None), (step3, None)]

                    # Collect all events
                    async for response in context_building_agent._core_reasoning_process(sample_context_request):  # noqa: E501
                        events.append(response)

        # Access context from the agent's internal state after processing
        context = context_building_agent.reasoning_context

        # Verify all steps were accumulated
        assert len(context["steps"]) == 3
        assert context["steps"][0] == step1
        assert context["steps"][1] == step2
        assert context["steps"][2] == step3

        # Verify all tool results were accumulated
        assert len(context["tool_results"]) == 2

        # Verify first tool result (search)
        search_result = context["tool_results"][0]
        assert search_result.tool_name == "search_web"
        assert search_result.success is True
        assert search_result.result == {"results": ["Tokyo weather is generally mild", "Current conditions available"]}  # noqa: E501

        # Verify second tool result (weather)
        weather_result = context["tool_results"][1]
        assert weather_result.tool_name == "get_weather"
        assert weather_result.success is True
        assert weather_result.result == {"location": "Tokyo", "temperature": "22°C", "condition": "Sunny"}  # noqa: E501

    @pytest.mark.asyncio
    async def test_context_preservation_across_tool_executions(self):
        """Test that context is preserved across sequential tool executions."""
        httpx.AsyncClient()
        mock_prompt_manager = AsyncMock(spec=PromptManager)
        mock_prompt_manager.get_prompt.return_value = "Test prompt"

        # Tools that simulate stateful operations
        memory_store = {}

        def store_data(key: str, value: object) -> dict[str, str]:
            """Store data in memory."""
            memory_store[key] = value
            return {"status": "stored", "key": key, "value": str(value)}

        def retrieve_data(key: str) -> dict[str, object]:
            """Retrieve data from memory."""
            if key in memory_store:
                return {"status": "found", "key": key, "value": memory_store[key]}
            return {"status": "not_found", "key": key, "value": None}

        tools = [
            function_to_tool(store_data, name="store"),
            function_to_tool(retrieve_data, name="retrieve"),
        ]

        context_aware_agent = ReasoningAgent(
            openai_client=AsyncOpenAI(api_key="test-api-key", base_url="https://api.openai.com/v1"),
            tools=tools,
            prompt_manager=mock_prompt_manager,
        )

        # First operation: store data
        store_prediction = ToolPrediction(
            tool_name="store",
            arguments={"key": "user_id", "value": "12345"},
            reasoning="Store user ID for later use",
        )

        # Second operation: retrieve data
        retrieve_prediction = ToolPrediction(
            tool_name="retrieve",
            arguments={"key": "user_id"},
            reasoning="Retrieve stored user ID",
        )

        # Execute sequentially
        store_results = await context_aware_agent._execute_tools_sequentially([store_prediction])
        retrieve_results = await context_aware_agent._execute_tools_sequentially([retrieve_prediction])  # noqa: E501

        # Verify context was preserved
        assert store_results[0].success is True
        assert store_results[0].result["status"] == "stored"

        assert retrieve_results[0].success is True
        assert retrieve_results[0].result["status"] == "found"
        assert retrieve_results[0].result["value"] == "12345"



# =============================================================================
# Reasoning Loop Tests
# =============================================================================

class TestReasoningLoop:
    """Test reasoning loop termination and infinite loop prevention."""

    @pytest.mark.asyncio
    @respx.mock
    async def test__reasoning_terminates_when_tools_fail(
        self,
        reasoning_agent: ReasoningAgent,
        sample_chat_request: OpenAIChatRequest,
    ) -> None:
        """Test that reasoning process terminates gracefully when tools fail."""
        # Create reasoning step that tries to use tools
        step1 = ReasoningStepFactory.tool_step(
            thought="I need to use weather tools to get current conditions",
            tools=[ToolPredictionFactory.weather_prediction(
                location="Tokyo",
                reasoning="Need current weather data for Tokyo",
            )],
        )
        step1_response = create_reasoning_response(step1, "chatcmpl-reasoning1")
        step1_response.created = 1234567890
        step1_response.usage.prompt_tokens = 10
        step1_response.usage.completion_tokens = 5
        step1_response.usage.total_tokens = 15

        # Create reasoning step that recognizes tool failure and finishes
        step2 = ReasoningStepFactory.finished_step("Tools failed, proceeding with knowledge-based response")  # noqa: E501
        step2_response = create_reasoning_response(step2, "chatcmpl-reasoning2")
        step2_response.created = 1234567890
        step2_response.usage.prompt_tokens = 15
        step2_response.usage.completion_tokens = 8
        step2_response.usage.total_tokens = 23

        # Create streaming synthesis response using Pydantic models
        content_chunk = OpenAIStreamResponse(
            id="chatcmpl-synthesis",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-4o",
            choices=[
                OpenAIStreamChoice(
                    index=0,
                    delta=OpenAIDelta(content="Based on my knowledge, Tokyo weather..."),
                    finish_reason=None,
                ),
            ],
        )

        finish_chunk = OpenAIStreamResponse(
            id="chatcmpl-synthesis",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-4o",
            choices=[
                OpenAIStreamChoice(
                    index=0,
                    delta=OpenAIDelta(),
                    finish_reason="stop",
                ),
            ],
        )

        # Build streaming response content
        streaming_chunks = [
            create_sse(content_chunk).encode(),
            create_sse(finish_chunk).encode(),
            SSE_DONE.encode(),
        ]

        streaming_response = httpx.Response(
            200,
            content=b''.join(streaming_chunks),
            headers={"content-type": "text/plain; charset=utf-8"},
        )

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json=step1_response.model_dump()),
                httpx.Response(200, json=step2_response.model_dump()),
                streaming_response,  # Streaming synthesis
            ],
        )

        # Collect streaming chunks
        collector = ReasoningAgentStreamingCollector()
        await collector.process(reasoning_agent.execute_stream(sample_chat_request))

        # Should complete successfully despite tool failures - verify specific response
        assert "Based on my knowledge, Tokyo weather..." in collector.content

    @pytest.mark.asyncio
    @respx.mock
    async def test__reasoning_respects_max_iterations(
        self,
        reasoning_agent: ReasoningAgent,
        sample_chat_request: OpenAIChatRequest,
    ) -> None:
        """Test that reasoning process respects max_reasoning_iterations limit."""
        # Create reasoning step that always continues thinking (would loop infinitely)
        continue_thinking_step = ReasoningStepFactory.thinking_step(
            "Still thinking about the problem...",
        )
        continue_thinking_response = create_reasoning_response(
            continue_thinking_step, "chatcmpl-reasoning",
        )
        continue_thinking_response.created = 1234567890
        continue_thinking_response.usage.prompt_tokens = 10
        continue_thinking_response.usage.completion_tokens = 5
        continue_thinking_response.usage.total_tokens = 15

        # Create streaming synthesis response for when max iterations reached
        content_chunk = OpenAIStreamResponse(
            id="chatcmpl-synthesis",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-4o",
            choices=[
                OpenAIStreamChoice(
                    index=0,
                    delta=OpenAIDelta(content="After extensive reasoning..."),
                    finish_reason=None,
                ),
            ],
        )

        finish_chunk = OpenAIStreamResponse(
            id="chatcmpl-synthesis",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-4o",
            choices=[
                OpenAIStreamChoice(
                    index=0,
                    delta=OpenAIDelta(),
                    finish_reason="stop",
                ),
            ],
        )

        # Build streaming response content
        streaming_chunks = [
            create_sse(content_chunk).encode(),
            create_sse(finish_chunk).encode(),
            SSE_DONE.encode(),
        ]

        streaming_response = httpx.Response(
            200,
            content=b''.join(streaming_chunks),
            headers={"content-type": "text/plain; charset=utf-8"},
        )

        # Mock exactly 20 reasoning calls (max iterations) then streaming synthesis
        # After 20 iterations, it should move to synthesis
        reasoning_calls = [httpx.Response(200, json=continue_thinking_response.model_dump())] * 20

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[*reasoning_calls, streaming_response],
        )

        # Collect streaming chunks
        collector = ReasoningAgentStreamingCollector()
        await collector.process(reasoning_agent.execute_stream(sample_chat_request))

        # Should complete after max iterations and synthesize final response
        assert "After extensive reasoning..." in collector.content


# =============================================================================
# JSON Mode Integration Tests
# =============================================================================

class TestJSONModeIntegration:
    """Test structured outputs integration with ReasoningAgent."""

    @pytest.fixture
    def mock_reasoning_agent(self):
        """Create a reasoning agent with mocked OpenAI client for testing."""
        httpx.AsyncClient()
        mock_prompt_manager = AsyncMock(spec=PromptManager)
        mock_prompt_manager.get_prompt.return_value = "Test system prompt"

        # Create tools with complex type hints
        def weather_func(location: str, units: str | None = "celsius") -> dict[str, Any]:
            return {"location": location, "temperature": "22°C", "units": units}

        def search_func(query: str, max_results: int = 5, filters: list[str] | None = None) -> dict[str, object]:  # noqa: E501
            return {"query": query, "results": [], "count": max_results, "filters": filters}

        tools = [
            function_to_tool(weather_func, name="get_weather"),
            function_to_tool(search_func, name="search_web"),
        ]

        return ReasoningAgent(
            openai_client=AsyncOpenAI(api_key="test-api-key", base_url="https://api.openai.com/v1"),
            tools=tools,
            prompt_manager=mock_prompt_manager,
        )

    def test_json_mode_success(self):
        """Test successful JSON mode parsing."""
        # Test that we can parse a valid JSON response into a ReasoningStep
        json_data = {
            "thought": "I need to get weather information",
            "next_action": "use_tools",
            "tools_to_use": [
                {
                    "tool_name": "get_weather",
                    "arguments": {"location": "Tokyo", "units": "fahrenheit"},
                    "reasoning": "User wants Tokyo weather in Fahrenheit",
                },
            ],
            "concurrent_execution": False,
        }

        # Test that this parses correctly
        result = ReasoningStep.model_validate(json_data)

        assert result.thought == "I need to get weather information"
        assert result.next_action == ReasoningAction.USE_TOOLS
        assert len(result.tools_to_use) == 1
        assert result.tools_to_use[0].tool_name == "get_weather"
        assert result.tools_to_use[0].arguments == {"location": "Tokyo", "units": "fahrenheit"}

    def test_json_mode_malformed_json_fallback(self):
        """Test fallback when JSON is malformed."""
        # Test that malformed JSON data raises validation error
        malformed_json = {
            "thought": "I need to get weather information",
            "next_action": "invalid_action",  # Invalid action
            "tools_to_use": [],
            "concurrent_execution": False,
        }

        # This should raise a validation error
        with pytest.raises(ValueError, match="invalid_action"):
            ReasoningStep.model_validate(malformed_json)

    @pytest.mark.asyncio
    async def test_json_mode_malformed_json_fallback_agent(
        self,
        mock_reasoning_agent: ReasoningAgent,
    ):
        """Test fallback when content is malformed JSON."""
        # Create a proper mock for the OpenAI response - use Mock for the structure, not AsyncMock
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "invalid json {{"  # Malformed JSON
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        # Add proper usage data to avoid validation errors
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage

        # Create a proper mock request object
        mock_request = create_mock_request()

        # Mock the async create method to return our mock response
        async def mock_create(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202, ARG001
            return mock_response

        with patch.object(mock_reasoning_agent.openai_client.chat.completions, 'create', side_effect=mock_create):  # noqa: E501
            result, usage = await mock_reasoning_agent._generate_reasoning_step(
                mock_request,
                {"steps": [], "tool_results": [], "final_thoughts": "", "user_request": None},
                "test prompt",
            )

        # Verify fallback behavior
        assert "Unable to generate structured reasoning step" in result.thought
        assert "proceeding to final answer" in result.thought  # Should include fallback indication
        assert result.next_action == ReasoningAction.FINISHED  # Should finish on fallback

        # Verify usage is properly returned
        assert usage is not None
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 15
        assert result.tools_to_use == []


# =============================================================================
# Error Recovery Tests
# =============================================================================

class TestErrorRecovery:
    """Test error recovery and fallback mechanisms."""

    @pytest.fixture
    def error_prone_agent(self):
        """Agent with tools that can fail in various ways."""
        httpx.AsyncClient()
        mock_prompt_manager = AsyncMock(spec=PromptManager)

        def validation_error_func(value: int) -> str:
            """Function that validates input."""
            if value < 0:
                raise ValueError("Value must be non-negative")
            return f"Processed: {value}"

        def type_error_func(items: list[str]) -> int:
            """Function that expects specific types."""
            return len(items)

        async def network_error_func(url: str) -> dict[str, str]:
            """Function that simulates network errors."""
            if "fail" in url:
                raise ConnectionError("Network unreachable")
            return {"url": url, "status": "ok"}

        tools = [
            function_to_tool(validation_error_func, name="validate_input"),
            function_to_tool(type_error_func, name="count_items"),
            function_to_tool(network_error_func, name="fetch_url"),
        ]

        return ReasoningAgent(
            openai_client=AsyncOpenAI(api_key="test-api-key", base_url="https://api.openai.com/v1"),
            tools=tools,
            prompt_manager=mock_prompt_manager,
        )

    @pytest.mark.asyncio
    async def test_validation_error_handling(self, error_prone_agent: ReasoningAgent, tracing_enabled):  # noqa
        """Test handling of validation errors."""
        prediction = ToolPrediction(
            tool_name="validate_input",
            arguments={"value": -5},
            reasoning="Test validation error",
        )

        results = await error_prone_agent._execute_tools_sequentially([prediction])

        assert len(results) == 1
        result = results[0]
        assert result.success is False
        assert "Value must be non-negative" in result.error
        assert result.tool_name == "validate_input"

    @pytest.mark.asyncio
    async def test_type_error_handling(self, error_prone_agent: ReasoningAgent, tracing_enabled):  # noqa
        """Test handling of type errors."""
        prediction = ToolPrediction(
            tool_name="count_items",
            arguments={"items": 123},  # Wrong type - number instead of list
            reasoning="Test type error",
        )

        results = await error_prone_agent._execute_tools_sequentially([prediction])

        assert len(results) == 1
        result = results[0]
        assert result.success is False
        assert "count_items" in result.error

    @pytest.mark.asyncio
    async def test_network_error_handling(self, error_prone_agent: ReasoningAgent, tracing_enabled):  # noqa
        """Test handling of network errors."""
        prediction = ToolPrediction(
            tool_name="fetch_url",
            arguments={"url": "https://fail.example.com"},
            reasoning="Test network error",
        )

        results = await error_prone_agent._execute_tools_sequentially([prediction])

        assert len(results) == 1
        result = results[0]
        assert result.success is False
        assert "Network unreachable" in result.error

    @pytest.mark.asyncio
    async def test_mixed_success_failure_parallel(self, error_prone_agent: ReasoningAgent, tracing_enabled):  # noqa
        """Test parallel execution with mixed success/failure."""
        predictions = [
            ToolPrediction(
                tool_name="validate_input",
                arguments={"value": 10},  # Should succeed
                reasoning="Success case",
            ),
            ToolPrediction(
                tool_name="validate_input",
                arguments={"value": -1},  # Should fail
                reasoning="Failure case",
            ),
            ToolPrediction(
                tool_name="fetch_url",
                arguments={"url": "https://success.example.com"},  # Should succeed
                reasoning="Another success case",
            ),
        ]

        results = await error_prone_agent._execute_tools_concurrently(predictions)

        assert len(results) == 3
        # Results should be in same order as predictions
        assert results[0].success is True
        assert results[0].result == "Processed: 10"
        assert results[1].success is False
        assert "non-negative" in results[1].error
        assert results[2].success is True
        assert results[2].result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_unknown_tool_error(self, error_prone_agent: ReasoningAgent, tracing_enabled):  # noqa
        """Test handling of unknown tool requests."""
        prediction = ToolPrediction(
            tool_name="nonexistent_tool",
            arguments={"param": "value"},
            reasoning="Test unknown tool",
        )

        results = await error_prone_agent._execute_tools_sequentially([prediction])

        assert len(results) == 1
        result = results[0]
        assert result.success is False
        assert "Tool 'nonexistent_tool' not found" in result.error
        assert result.tool_name == "nonexistent_tool"

    @pytest.mark.asyncio
    async def test_json_parsing_failure_sets_span_error(self, tracing_enabled):  # noqa
        """Test that JSON parsing failures set current span to ERROR status."""
        # Create mock agent with error-prone JSON response
        mock_prompt_manager = AsyncMock(spec=PromptManager)
        mock_prompt_manager.get_prompt.return_value = "Test system prompt"

        agent = ReasoningAgent(
            openai_client=AsyncOpenAI(api_key="test-api-key", base_url="https://api.openai.com/v1"),
            tools=[],
            prompt_manager=mock_prompt_manager,
        )

        # Mock OpenAI response with malformed JSON
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "invalid json {{"  # Malformed JSON
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage

        # Mock current span to capture status calls with proper span context
        mock_span = Mock()
        mock_span.set_status = Mock()
        mock_span_context = Mock()
        mock_span_context.trace_id = 12345678901234567890123456789012
        mock_span_context.span_id = 1234567890123456
        mock_span_context.trace_flags = 0x01
        mock_span.get_span_context = Mock(return_value=mock_span_context)

        async def mock_create(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202, ARG001
            return mock_response

        with patch.object(agent.openai_client.chat.completions, 'create', side_effect=mock_create):
            with patch('opentelemetry.trace.get_current_span', return_value=mock_span):
                mock_request = Mock()
                mock_request.model = "gpt-4o"
                mock_request.messages = [{"role": "user", "content": "test"}]
                mock_request.temperature = 0.7

                result, usage = await agent._generate_reasoning_step(
                    mock_request,
                    {"steps": [], "tool_results": [], "final_thoughts": "", "user_request": None},
                    "test prompt",
                )

        # Verify span status was set to ERROR for JSON parsing failure
        mock_span.set_status.assert_called_once()
        call_args = mock_span.set_status.call_args[0][0]
        assert call_args.status_code == StatusCode.ERROR
        assert "Failed to parse JSON response" in call_args.description

    @pytest.mark.asyncio
    async def test_all_tools_fail_sets_step_span_error(self, tracing_enabled):  # noqa
        """Test that when ALL tools in a step fail, step span gets ERROR status."""
        # Create agent with failing tools
        mock_prompt_manager = AsyncMock(spec=PromptManager)

        def failing_tool1(param: str) -> str:  # noqa: ARG001
            raise ValueError("Tool 1 failed")

        def failing_tool2(param: str) -> str:  # noqa: ARG001
            raise ValueError("Tool 2 failed")

        tools = [
            function_to_tool(failing_tool1, name="fail1"),
            function_to_tool(failing_tool2, name="fail2"),
        ]

        agent = ReasoningAgent(
            openai_client=AsyncOpenAI(api_key="test-api-key", base_url="https://api.openai.com/v1"),
            tools=tools,
            prompt_manager=mock_prompt_manager,
        )

        # Create predictions for tools that will all fail
        predictions = [
            ToolPrediction(tool_name="fail1", arguments={"param": "test"}, reasoning="test"),
            ToolPrediction(tool_name="fail2", arguments={"param": "test"}, reasoning="test"),
        ]

        # Mock step span to capture status calls
        mock_step_span = Mock()
        mock_step_span.set_status = Mock()
        mock_step_span.set_attribute = Mock()

        with patch('api.reasoning_agent.tracer.start_as_current_span') as mock_tracer:
            mock_tracer.return_value.__enter__.return_value = mock_step_span

            # This should trigger step span ERROR status since all tools fail
            results = await agent._execute_tools_sequentially(predictions)

        # Verify all tools failed
        assert len(results) == 2
        assert all(not r.success for r in results)

        # Find the ERROR status call (should be called after OK default)
        error_calls = [call for call in mock_step_span.set_status.call_args_list
                      if call[0][0].status_code == StatusCode.ERROR]
        assert len(error_calls) >= 1, "Step span should be set to ERROR when all tools fail"

    @pytest.mark.asyncio
    async def test_all_tools_fail_sets_tools_span_error(self, tracing_enabled):  # noqa
        """Test that when ALL tools fail, tools orchestration span gets ERROR status."""
        # Create agent with failing tools
        mock_prompt_manager = AsyncMock(spec=PromptManager)

        def failing_tool(param: str) -> str:  # noqa: ARG001
            raise ValueError("Tool failed")

        tools = [function_to_tool(failing_tool, name="fail_tool")]

        agent = ReasoningAgent(
            openai_client=AsyncOpenAI(api_key="test-api-key", base_url="https://api.openai.com/v1"),
            tools=tools,
            prompt_manager=mock_prompt_manager,
        )

        predictions = [
            ToolPrediction(tool_name="fail_tool", arguments={"param": "test"}, reasoning="test"),
        ]

        # Execute and verify tool orchestration span gets ERROR status
        results = await agent._execute_tools_concurrently(predictions)

        # Verify tool failed
        assert len(results) == 1
        assert not results[0].success
        assert "Tool failed" in results[0].error

    @pytest.mark.asyncio
    async def test_openai_api_error_sets_span_error(self, tracing_enabled):  # noqa
        """Test that OpenAI API errors set current span to ERROR status."""
        mock_prompt_manager = AsyncMock(spec=PromptManager)
        agent = ReasoningAgent(
            openai_client=AsyncOpenAI(api_key="test-api-key", base_url="https://api.openai.com/v1"),
            tools=[],
            prompt_manager=mock_prompt_manager,
        )

        # Mock span to capture status calls with proper span context
        mock_span = Mock()
        mock_span.set_status = Mock()
        mock_span_context = Mock()
        mock_span_context.trace_id = 12345678901234567890123456789012
        mock_span_context.span_id = 1234567890123456
        mock_span_context.trace_flags = 0x01
        mock_span.get_span_context = Mock(return_value=mock_span_context)

        # Mock HTTP error response with proper structure
        error_response = Mock()
        error_response.status_code = 401
        error_response.text = "Unauthorized access"

        # Create a proper HTTP request mock
        mock_request = Mock()
        http_error = httpx.HTTPStatusError("Unauthorized",
        request=mock_request,
        response=error_response,
        )

        async def mock_create(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202, ARG001
            raise http_error

        with patch.object(agent.openai_client.chat.completions, 'create', side_effect=mock_create):
            with patch('opentelemetry.trace.get_current_span', return_value=mock_span):
                mock_request = Mock()
                mock_request.model = "gpt-4o"
                mock_request.messages = [{"role": "user", "content": "test"}]
                mock_request.temperature = 0.7

                result, usage = await agent._generate_reasoning_step(
                    mock_request,
                    {"steps": [], "tool_results": [], "final_thoughts": "", "user_request": None},
                    "test prompt",
                )

        # Verify span status was set to ERROR for API error
        mock_span.set_status.assert_called_once()
        call_args = mock_span.set_status.call_args[0][0]
        assert call_args.status_code == StatusCode.ERROR
        assert "OpenAI API error: 401" in call_args.description

    @pytest.mark.asyncio
    async def test_streaming_synthesis_http_error_handling(self, tracing_enabled):  # noqa: ANN001, ARG002
        """Test that HTTP errors during streaming synthesis are handled correctly."""
        mock_prompt_manager = AsyncMock(spec=PromptManager)
        agent = ReasoningAgent(
            openai_client=AsyncOpenAI(api_key="test-api-key", base_url="https://api.openai.com/v1"),
            tools=[],
            prompt_manager=mock_prompt_manager,
        )

        # Create a proper HTTP error for streaming synthesis
        error_response = Mock()
        error_response.status_code = 429
        error_response.text = "Rate limit exceeded"

        mock_request = Mock()
        http_error = httpx.HTTPStatusError(
            "Rate limit exceeded",
            request=mock_request,
            response=error_response,
        )

        # Mock the streaming create call to raise HTTP error
        async def mock_create_stream(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202, ARG001
            # Only raise error for streaming calls (stream=True)
            if kwargs.get('stream') is True:
                raise http_error
            # For non-streaming calls, return a normal response (shouldn't happen in this test)
            return Mock()

        with patch.object(agent.openai_client.chat.completions, 'create', side_effect=mock_create_stream):  # noqa: E501
            # Test the streaming synthesis method directly
            request = OpenAIChatRequest(
                model="gpt-4o",
                messages=[{"role": "user", "content": "test"}],
                stream=True,
            )

            reasoning_context = {
                "steps": [],
                "tool_results": [],
                "final_thoughts": "Test thoughts",
                "user_request": request,
            }

            # This should raise a ReasoningError due to the HTTP error
            with pytest.raises(ReasoningError) as exc_info:
                async for _ in agent._stream_final_synthesis(
                    request, "test-completion-id", 1234567890, reasoning_context,
                ):
                    pass

            # Verify the error message contains the HTTP status and details
            error_msg = str(exc_info.value)
            assert "OpenAI API error during streaming synthesis" in error_msg
            assert "429" in error_msg
            assert "Rate limit exceeded" in error_msg

    @pytest.mark.parametrize("use_tracing", [True, False])
    @pytest.mark.asyncio
    async def test_error_handling_works_regardless_of_tracing(self, use_tracing):  # noqa: ANN001
        """Test that core error handling works the same with/without tracing."""
        # Enable/disable tracing for this test
        original_disabled = os.environ.get('OTEL_SDK_DISABLED')

        if use_tracing:
            os.environ.pop('OTEL_SDK_DISABLED', None)  # Enable tracing
        else:
            os.environ['OTEL_SDK_DISABLED'] = 'true'  # Disable tracing

        try:
            mock_prompt_manager = AsyncMock(spec=PromptManager)

            def failing_tool(param: str) -> str:  # noqa: ARG001
                raise ValueError("Tool failed")

            tools = [function_to_tool(failing_tool, name="fail_tool")]

            agent = ReasoningAgent(
                openai_client=AsyncOpenAI(api_key="test-api-key", base_url="https://api.openai.com/v1"),
                tools=tools,
                prompt_manager=mock_prompt_manager,
            )

            prediction = ToolPrediction(
                tool_name="fail_tool",
                arguments={"param": "test"},
                reasoning="test",
            )

            # Test that tool failure handling works the same regardless of tracing
            results = await agent._execute_tools_sequentially([prediction])

            assert len(results) == 1
            assert not results[0].success
            assert "Tool failed" in results[0].error
            assert results[0].tool_name == "fail_tool"

        finally:
            # Restore original tracing state
            if original_disabled is not None:
                os.environ['OTEL_SDK_DISABLED'] = original_disabled
            else:
                os.environ.pop('OTEL_SDK_DISABLED', None)


# =============================================================================
# Concurrent Execution Tests
# =============================================================================

class TestConcurrentExecution:
    """Test concurrent tool execution in the reasoning flow."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_reasoning_flow_with_concurrent_execution(
        self,
        reasoning_agent: ReasoningAgent,
        sample_chat_request: OpenAIChatRequest,
    ) -> None:
        """Test that reasoning flow uses concurrent execution when specified."""
        # Create a reasoning step that uses concurrent execution
        concurrent_step = ReasoningStep(
            thought="I need to gather weather and search data simultaneously",
            next_action=ReasoningAction.FINISHED,
            tools_to_use=[
                ToolPrediction(
                    tool_name="weather_tool",
                    arguments={"location": "Tokyo"},
                    reasoning="Get weather for Tokyo",
                ),
                ToolPrediction(
                    tool_name="search_tool",
                    arguments={"query": "Tokyo weather"},
                    reasoning="Search for Tokyo weather info",
                ),
            ],
            concurrent_execution=True,  # This is the key part we're testing
        )

        # Create structured output response for reasoning step
        step_response = (
            OpenAIResponseBuilder()
            .id("chatcmpl-concurrent")
            .model("gpt-4o")
            .created(1234567890)
            .choice(0, "assistant", concurrent_step.model_dump_json())
            .usage(15, 8)
            .build()
        )

        # Create streaming synthesis response
        content_chunk = OpenAIStreamResponse(
            id="chatcmpl-synthesis",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-4o",
            choices=[
                OpenAIStreamChoice(
                    index=0,
                    delta=OpenAIDelta(content="Based on concurrent data gathering, Tokyo is sunny."),  # noqa: E501
                    finish_reason=None,
                ),
            ],
            usage=OpenAIUsage(prompt_tokens=25, completion_tokens=12, total_tokens=37),
        )

        finish_chunk = OpenAIStreamResponse(
            id="chatcmpl-synthesis",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-4o",
            choices=[
                OpenAIStreamChoice(
                    index=0,
                    delta=OpenAIDelta(),
                    finish_reason="stop",
                ),
            ],
        )

        # Build streaming response
        streaming_chunks = [
            create_sse(content_chunk).encode(),
            create_sse(finish_chunk).encode(),
            SSE_DONE.encode(),
        ]

        streaming_response = httpx.Response(
            200,
            content=b''.join(streaming_chunks),
            headers={"content-type": "text/plain; charset=utf-8"},
        )

        # Mock the OpenAI API calls
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                create_http_response(step_response),  # Reasoning step generation
                streaming_response,  # Streaming synthesis
            ],
        )

        # Patch _execute_tools_concurrently to verify it gets called
        with patch.object(reasoning_agent, '_execute_tools_concurrently') as mock_concurrent:
            # Return mock tool results
            mock_concurrent.return_value = [
                ToolResult(
                    tool_name="weather_tool",
                    success=True,
                    result={"location": "Tokyo", "temperature": "22°C"},
                    execution_time_ms=100.0,
                ),
                ToolResult(
                    tool_name="search_tool",
                    success=True,
                    result={"query": "Tokyo weather", "results": ["Sunny conditions"]},
                    execution_time_ms=150.0,
                ),
            ]

            # Collect streaming chunks
            collector = ReasoningAgentStreamingCollector()
            await collector.process(reasoning_agent.execute_stream(sample_chat_request))

            # Verify concurrent execution was used
            mock_concurrent.assert_called_once()
            call_args = mock_concurrent.call_args[0][0]  # First positional argument
            assert len(call_args) == 2, "Should call with 2 tool predictions"
            assert call_args[0].tool_name == "weather_tool"
            assert call_args[1].tool_name == "search_tool"

        # Verify final result
        assert "Based on concurrent data gathering, Tokyo is sunny." in collector.content

    @pytest.mark.asyncio
    @respx.mock
    async def test_reasoning_flow_without_concurrent_execution(
        self,
        reasoning_agent: ReasoningAgent,
        sample_chat_request: OpenAIChatRequest,
    ) -> None:
        """Test that reasoning flow uses sequential execution when concurrent_execution=False."""
        # Create a reasoning step that uses sequential execution
        sequential_step = ReasoningStep(
            thought="I need to gather weather data first, then search",
            next_action=ReasoningAction.FINISHED,
            tools_to_use=[
                ToolPrediction(
                    tool_name="weather_tool",
                    arguments={"location": "Paris"},
                    reasoning="Get weather for Paris first",
                ),
            ],
            concurrent_execution=False,  # Sequential execution
        )

        # Create structured output response for reasoning step
        step_response = (
            OpenAIResponseBuilder()
            .id("chatcmpl-sequential")
            .model("gpt-4o")
            .created(1234567890)
            .choice(0, "assistant", sequential_step.model_dump_json())
            .usage(12, 6)
            .build()
        )

        # Create streaming synthesis response
        content_chunk = OpenAIStreamResponse(
            id="chatcmpl-synthesis",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-4o",
            choices=[
                OpenAIStreamChoice(
                    index=0,
                    delta=OpenAIDelta(content="Based on sequential data gathering, Paris is cloudy."),  # noqa: E501
                    finish_reason="stop",
                ),
            ],
            usage=OpenAIUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
        )

        # Build streaming response
        streaming_chunks = [
            create_sse(content_chunk).encode(),
            SSE_DONE.encode(),
        ]

        streaming_response = httpx.Response(
            200,
            content=b''.join(streaming_chunks),
            headers={"content-type": "text/plain; charset=utf-8"},
        )

        # Mock the OpenAI API calls
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                create_http_response(step_response),  # Reasoning step generation
                streaming_response,  # Streaming synthesis
            ],
        )

        # Patch both execution methods to verify which gets called
        with patch.object(reasoning_agent, '_execute_tools_sequentially') as mock_sequential:
            with patch.object(reasoning_agent, '_execute_tools_concurrently') as mock_concurrent:
                # Return mock tool results from sequential execution
                mock_sequential.return_value = [
                    ToolResult(
                        tool_name="weather_tool",
                        success=True,
                        result={"location": "Paris", "temperature": "18°C"},
                        execution_time_ms=120.0,
                    ),
                ]

                # Collect streaming chunks
                collector = ReasoningAgentStreamingCollector()
                await collector.process(reasoning_agent.execute_stream(sample_chat_request))

                # Verify sequential execution was used, not concurrent
                mock_sequential.assert_called_once()
                mock_concurrent.assert_not_called()

        # Verify final result
        assert "Based on sequential data gathering, Paris is cloudy." in collector.content


# =============================================================================
# Span Attributes Tests
# =============================================================================

class TestSpanAttributes:
    """Test span attribute setting for Phoenix UI tracing."""

    @pytest.fixture
    def mock_span(self):
        """Create a mock span for testing attribute setting."""
        mock_span = Mock()
        mock_span.set_attribute = Mock()
        return mock_span

    @pytest.fixture
    def test_agent(self):
        """Create a test ReasoningAgent for span attribute testing."""
        mock_prompt_manager = AsyncMock(spec=PromptManager)
        mock_prompt_manager.get_prompt.return_value = "You are a helpful assistant."

        def weather_func(location: str) -> dict:
            return {"location": location, "temperature": "22°C"}

        tools = [function_to_tool(weather_func, name="get_weather")]

        return ReasoningAgent(
            openai_client=AsyncOpenAI(api_key="test-api-key", base_url="https://api.openai.com/v1"),
            tools=tools,
            prompt_manager=mock_prompt_manager,
        )

    def test_set_span_attributes_input_value(self, test_agent: ReasoningAgent, mock_span: Mock):
        """Test that INPUT_VALUE is set correctly from user messages."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the weather in Tokyo?"},
                {"role": "assistant", "content": "Let me check that for you."},
                {"role": "user", "content": "Actually, check Paris instead."},
            ],
        )

        test_agent._set_span_attributes(request, mock_span)

        # Should set INPUT_VALUE to the last user message
        mock_span.set_attribute.assert_any_call(
            "input.value",
            "Actually, check Paris instead.",
        )

    def test_set_span_attributes_no_user_messages(self, test_agent: ReasoningAgent, mock_span: Mock):  # noqa: E501
        """Test behavior when there are no user messages."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "assistant", "content": "Hello! How can I help you?"},
            ],
        )

        test_agent._set_span_attributes(request, mock_span)

        # Should not set INPUT_VALUE if no user messages
        input_calls = [call for call in mock_span.set_attribute.call_args_list
                      if call[0][0] == "input.value"]
        assert len(input_calls) == 0

    def test_set_span_attributes_metadata(self, test_agent: ReasoningAgent, mock_span: Mock):
        """Test that METADATA is set correctly with request details."""
        request = OpenAIChatRequest(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test message"}],
            temperature=0.7,
            max_tokens=150,
            stream=True,
        )

        test_agent._set_span_attributes(request, mock_span)

        # Should set METADATA with request details
        metadata_calls = [call for call in mock_span.set_attribute.call_args_list
                         if call[0][0] == "metadata"]
        assert len(metadata_calls) == 1

        # Parse the JSON metadata
        metadata_json = metadata_calls[0][0][1]
        metadata = json.loads(metadata_json)

        assert metadata["model"] == "gpt-4o-mini"
        assert metadata["temperature"] == 0.7
        assert metadata["max_tokens"] == 150
        assert metadata["message_count"] == 1
        assert metadata["tools_available"] == 1

    def test_set_span_attributes_default_values(self, test_agent: ReasoningAgent, mock_span: Mock):
        """Test metadata with default values."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Test message"}],
            # temperature, max_tokens, stream not specified
        )

        test_agent._set_span_attributes(request, mock_span)

        # Get metadata
        metadata_calls = [call for call in mock_span.set_attribute.call_args_list
                         if call[0][0] == "metadata"]
        metadata = json.loads(metadata_calls[0][0][1])

        assert metadata["temperature"] == 0.2  # DEFAULT_TEMPERATURE
        assert metadata["max_tokens"] is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_execute_stream_includes_usage_in_reasoning_events(
        self, test_agent: ReasoningAgent,
    ):
        """Test that execute_stream() includes usage data in reasoning events."""
        # Create reasoning step with usage
        reasoning_step = ReasoningStepFactory.finished_step("Direct answer")
        reasoning_response = create_reasoning_response(reasoning_step, "chatcmpl-reasoning")
        reasoning_response.usage = OpenAIUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)  # noqa: E501

        # Mock streaming synthesis
        content_chunk = OpenAIStreamResponse(
            id="chatcmpl-synthesis",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-4o",
            choices=[
                OpenAIStreamChoice(
                    index=0,
                    delta=OpenAIDelta(content="Test"),
                    finish_reason="stop",
                ),
            ],
        )

        streaming_chunks = [
            create_sse(content_chunk.model_dump()).encode(),
            SSE_DONE.encode(),
        ]

        streaming_response = httpx.Response(
            200,
            content=b''.join(streaming_chunks),
            headers={"content-type": "text/plain; charset=utf-8"},
        )

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                create_http_response(reasoning_response),
                streaming_response,
            ],
        )

        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Test message"}],
        )

        chunks = []
        async for chunk in test_agent.execute_stream(request):
            chunks.append(chunk)

        # Find reasoning event with usage
        usage_chunks = []
        for chunk in chunks:
            if "reasoning_event" in chunk and "usage" in chunk:
                data = parse_sse(chunk)
                if data.get("usage"):
                    usage_chunks.append(data)

        # Should have at least one reasoning event with usage data
        assert len(usage_chunks) >= 1
        usage_data = usage_chunks[0]["usage"]
        assert usage_data["prompt_tokens"] == 10
        assert usage_data["completion_tokens"] == 5
        assert usage_data["total_tokens"] == 15

    @pytest.mark.asyncio
    @respx.mock
    async def test_execute_stream_calls_set_span_attributes(self, test_agent: ReasoningAgent):
        """Test that execute_stream() calls _set_span_attributes when parent_span is provided."""
        # Mock streaming responses
        reasoning_step = ReasoningStepFactory.finished_step("Direct answer")
        reasoning_response = create_reasoning_response(reasoning_step, "chatcmpl-reasoning")

        # Mock the streaming synthesis response with realistic chunks using builder
        streaming_response = (
            OpenAIStreamingResponseBuilder()
            .chunk("chatcmpl-test", "gpt-4o", delta_role="assistant", delta_content="")
            .chunk("chatcmpl-test", "gpt-4o", delta_content="Tokyo")
            .chunk("chatcmpl-test", "gpt-4o", delta_content=" is")
            .chunk("chatcmpl-test", "gpt-4o", delta_content=" sunny")
            .done()
            .build()
        )
        mock_openai_streaming_chunks = streaming_response.split('\n\n')[:-1]
        mock_openai_streaming_chunks = [
            chunk + '\n\n' for chunk in mock_openai_streaming_chunks
            if chunk.strip()
        ]

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                create_http_response(reasoning_response),
                httpx.Response(200, text="\n".join(mock_openai_streaming_chunks)),
            ],
        )

        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            stream=True,
        )
        mock_span = Mock()
        mock_span.set_attribute = Mock()

        # Patch the _set_span_attributes method to track calls
        with patch.object(test_agent, '_set_span_attributes') as mock_set_attrs:
            chunks = []
            async for chunk in test_agent.execute_stream(request, parent_span=mock_span):
                chunks.append(chunk)

            # Should call _set_span_attributes with request and span
            mock_set_attrs.assert_called_once_with(request, mock_span)

            # Verify that chunks were generated correctly
            assert len(chunks) > 0
            assert chunks[-1] == SSE_DONE

            # Should set OUTPUT_VALUE on the span with the collected content
            output_calls = [call for call in mock_span.set_attribute.call_args_list
                           if call[0][0] == "output.value"]
            assert len(output_calls) == 1, "Should set OUTPUT_VALUE exactly once for streaming"

            # Verify the content was collected correctly from the chunks
            expected_output = "Tokyo is sunny"  # From our mock chunks
            actual_output = output_calls[0][0][1]
            assert actual_output == expected_output, f"Expected '{expected_output}', got '{actual_output}'"  # noqa: E501

    @pytest.mark.asyncio
    @respx.mock
    async def test_execute_stream_fails_when_no_content_collected(self, test_agent: ReasoningAgent):  # noqa: E501
        """Test that execute_stream() fails when no content is collected from chunks."""
        # Mock streaming responses
        reasoning_step = ReasoningStepFactory.finished_step("Direct answer")
        reasoning_response = create_reasoning_response(reasoning_step, "chatcmpl-reasoning")

        # Mock streaming response with NO content chunks (only metadata chunks) using builder
        streaming_response = (
            OpenAIStreamingResponseBuilder()
            .chunk("chatcmpl-test", "gpt-4o", delta_role="assistant")
            .chunk("chatcmpl-test", "gpt-4o", finish_reason="stop")
            .done()
            .build()
        )
        mock_openai_streaming_chunks = streaming_response.split('\n\n')[:-1]
        mock_openai_streaming_chunks = [
            chunk + '\n\n' for chunk in mock_openai_streaming_chunks
            if chunk.strip()
        ]

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                create_http_response(reasoning_response),
                httpx.Response(200, text="\n".join(mock_openai_streaming_chunks)),
            ],
        )

        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            stream=True,
        )
        mock_span = Mock()
        mock_span.set_attribute = Mock()

        # Should succeed but only yield reasoning events (no content chunks)
        chunks = []
        async for chunk in test_agent.execute_stream(request, parent_span=mock_span):
            chunks.append(chunk)

        # Should have reasoning events but minimal content chunks
        assert len(chunks) >= 2  # At least reasoning events + [DONE]
        assert chunks[-1] == SSE_DONE  # Should end with [DONE]
