"""
Comprehensive tests for the ReasoningAgent class.

Tests the ReasoningAgent proxy functionality, dependency injection, tool execution,
context building, error handling, and OpenAI API compatibility.

This file consolidates all reasoning agent unit tests following phase 4 of the
test refactoring plan to reduce duplication and improve maintainability.
"""

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, Mock, patch
import pytest
import httpx
import respx
from api.reasoning_agent import ReasoningAgent
from api.openai_protocol import (
    OpenAIChatRequest,
    OpenAIChatResponse,
    ErrorResponse,
    OpenAIResponseBuilder,
)
from api.prompt_manager import PromptManager
from api.reasoning_models import ReasoningAction, ReasoningStep, ToolPrediction
from api.tools import ToolResult, function_to_tool
from tests.conftest import OPENAI_TEST_MODEL, get_weather
from tests.fixtures.models import ReasoningStepFactory, ToolPredictionFactory
from tests.fixtures.responses import (
    create_error_http_response,
    create_reasoning_response,
    create_simple_response,
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
    async def test__execute__success(
        self,
        reasoning_agent: ReasoningAgent,
        sample_chat_request: OpenAIChatRequest,
        mock_openai_response: OpenAIChatResponse,
    ) -> None:
        """Test successful non-streaming chat completion."""
        # Mock OpenAI API response
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=create_http_response(mock_openai_response),
        )

        result = await reasoning_agent.execute(sample_chat_request)

        assert isinstance(result, OpenAIChatResponse)
        assert result.id == "chatcmpl-test123"
        assert result.model == "gpt-4o"
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "This is a test response from OpenAI."

    @pytest.mark.asyncio
    @respx.mock
    async def test__execute__performs_reasoning_process(
        self,
        reasoning_agent: ReasoningAgent,
        sample_chat_request: OpenAIChatRequest,
    ) -> None:
        """Test that reasoning agent performs full reasoning process."""
        # Mock the structured output call (for reasoning step generation)
        finished_step = ReasoningStepFactory.finished_step("I need to respond to the weather question")  # noqa: E501
        reasoning_response = (
            OpenAIResponseBuilder()
            .id("chatcmpl-reasoning")
            .model("gpt-4o")
            .created(1234567890)
            .choice(0, "assistant", "I need to respond to the weather question")
            .usage(10, 5)
            .build()
        )

        # Add the parsed field for structured output compatibility
        reasoning_response.choices[0].message.__dict__["parsed"] = finished_step.model_dump()

        # Mock the final synthesis call
        synthesis_response = create_simple_response(
            content="The weather in Paris is sunny.",
            completion_id="chatcmpl-synthesis",
        )
        synthesis_response.created = 1234567890
        synthesis_response.usage.prompt_tokens = 15
        synthesis_response.usage.completion_tokens = 8
        synthesis_response.usage.total_tokens = 23

        # Create proper structured output response for reasoning step
        finished_step_json = finished_step.model_dump_json()
        structured_reasoning_response = (
            OpenAIResponseBuilder()
            .id("chatcmpl-reasoning")
            .model("gpt-4o")
            .created(1234567890)
            .choice(0, "assistant", finished_step_json)  # Valid JSON content
            .usage(10, 5)
            .build()
        )

        # Set up mock responses in order they will be called
        reasoning_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                create_http_response(structured_reasoning_response),  # First call - structured reasoning  # noqa: E501
                create_http_response(synthesis_response),  # Second call - final synthesis
            ],
        )

        result = await reasoning_agent.execute(sample_chat_request)

        # Verify both reasoning and synthesis calls were made in correct order
        assert reasoning_route.called, "OpenAI API should be called"
        assert reasoning_route.call_count == 2, "Should call OpenAI API twice: reasoning + synthesis"  # noqa: E501

        # Verify reasoning request used structured output for step generation
        reasoning_request = reasoning_route.calls[0].request
        reasoning_body = json.loads(reasoning_request.content.decode())
        assert "response_format" in reasoning_body, "Reasoning should use structured output"
        assert reasoning_body["response_format"]["type"] == "json_object", "Should use JSON object format"  # noqa: E501

        # Verify synthesis request contains reasoning context
        synthesis_request = reasoning_route.calls[1].request
        synthesis_body = json.loads(synthesis_request.content.decode())
        synthesis_messages = synthesis_body["messages"]

        # Should have system prompt, user message, and reasoning context
        assert len(synthesis_messages) >= 3, "Should have system prompt, user message and reasoning context"  # noqa: E501

        # Find the user message (not system prompt)
        user_message = next((msg for msg in synthesis_messages if msg.get("role") == "user"), None)
        assert user_message is not None, "Should have user message"
        assert "What's the weather in Paris?" in user_message["content"], "Should preserve user question"  # noqa: E501

        # Verify reasoning context is included in synthesis
        reasoning_context_found = any("reasoning_context" in str(msg) or "reasoning" in str(msg).lower()  # noqa: E501
                                    for msg in synthesis_messages)
        assert reasoning_context_found, "Should include reasoning context in synthesis"

        # Verify final result structure and content
        assert isinstance(result, OpenAIChatResponse), "Should return proper OpenAI response structure"  # noqa: E501
        assert result.choices[0].message.content == "The weather in Paris is sunny.", "Should return expected final answer"  # noqa: E501
        assert result.id == "chatcmpl-synthesis", "Should use synthesis response ID"
        assert result.model == "gpt-4o", "Should preserve model from request"

    @pytest.mark.asyncio
    @respx.mock
    async def test__execute__handles_openai_error(
        self,
        reasoning_agent: ReasoningAgent,
        sample_chat_request: OpenAIChatRequest,
        mock_openai_error_response: ErrorResponse,
    ) -> None:
        """Test that OpenAI API errors are properly raised."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=create_error_http_response(mock_openai_error_response, 401),
        )

        with pytest.raises(Exception) as exc_info:  # Can be OpenAI or HTTPx error  # noqa: PT011
            await reasoning_agent.execute(sample_chat_request)

        # Check that it's an authentication-related error
        assert "401" in str(exc_info.value) or "Invalid API key" in str(exc_info.value)

    @pytest.mark.asyncio
    @respx.mock
    async def test__execute_stream__includes_reasoning_events(
        self,
        reasoning_agent: ReasoningAgent,
        sample_streaming_request: OpenAIChatRequest,
        mock_openai_streaming_chunks: list[str],
    ) -> None:
        """Test that streaming includes reasoning events with metadata."""
        # Mock streaming response for final synthesis
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                text="\n".join(mock_openai_streaming_chunks),
                headers={"content-type": "text/plain"},
            ),
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
        for reasoning_chunk in reasoning_chunks:
            chunk_data = reasoning_chunk.replace("data: ", "").strip()
            assert chunk_data, "Reasoning chunk should have data"

            # Parse and verify reasoning event structure
            reasoning_data = json.loads(chunk_data)
            assert "choices" in reasoning_data, "Reasoning event should have choices structure"
            assert len(reasoning_data["choices"]) == 1, "Should have exactly one choice"

            choice = reasoning_data["choices"][0]
            assert "delta" in choice, "Choice should have delta for streaming"
            assert "reasoning_event" in choice["delta"], "Should contain reasoning_event metadata"

            # Verify reasoning event metadata
            reasoning_event = choice["delta"]["reasoning_event"]
            assert "type" in reasoning_event, "Reasoning event should have type"
            assert reasoning_event["type"] in ["thinking", "tool_execution", "synthesis", "reasoning_step"], "Should be valid reasoning type"  # noqa: E501

        # Should have final response chunks
        assert len(response_chunks) >= 1, "Should have at least one response chunk"

        # Verify response chunks contain actual content
        response_content_found = False
        for response_chunk in response_chunks:
            chunk_data = response_chunk.replace("data: ", "").strip()
            if chunk_data:
                response_data = json.loads(chunk_data)
                if response_data.get("choices"):
                    choice = response_data["choices"][0]
                    if "delta" in choice and "content" in choice["delta"] and choice["delta"]["content"]:  # noqa: E501
                        response_content_found = True
                        break

        assert response_content_found, "Should have response chunks with actual content"

        # Verify final [DONE] chunk
        assert len(done_chunks) == 1, "Should have exactly one [DONE] chunk"
        assert done_chunks[0] == "data: [DONE]\n\n", "Should end with proper [DONE] format"

    @pytest.mark.asyncio
    @respx.mock
    async def test__execute_stream__forwards_openai_chunks(
        self,
        reasoning_agent: ReasoningAgent,
        sample_streaming_request: OpenAIChatRequest,
        mock_openai_streaming_chunks: list[str],
    ) -> None:
        """Test that OpenAI chunks are properly forwarded with modified IDs."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                text="\n".join(mock_openai_streaming_chunks),
                headers={"content-type": "text/plain"},
            ),
        )

        chunks = []
        async for chunk in reasoning_agent.execute_stream(sample_streaming_request):
            if chunk.startswith("data: {") and "chatcmpl-" in chunk:
                chunks.append(chunk)

        # Should have OpenAI-style chunks with modified completion IDs - verify structure
        assert len(chunks) >= 1  # Must have at least one valid OpenAI chunk
        for chunk in chunks:
            if "chatcmpl-" in chunk:
                # Extract JSON from chunk
                chunk_data = json.loads(chunk[6:])  # Remove "data: " prefix
                # ID should be modified to our format, not the original "chatcmpl-test123"
                assert chunk_data["id"].startswith("chatcmpl-")
                assert chunk_data["id"] != "chatcmpl-test123"

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

        with pytest.raises(httpx.HTTPStatusError):
            async for _ in reasoning_agent.execute_stream(sample_streaming_request):
                pass

    @pytest.mark.asyncio
    async def test__execute__no_tools(self) -> None:
        """Test ReasoningAgent execute without tools."""
        async with httpx.AsyncClient() as client:
            # Create mock prompt manager
            mock_prompt_manager = AsyncMock(spec=PromptManager)
            mock_prompt_manager.get_prompt.return_value = "Test system prompt"

            agent = ReasoningAgent(
                base_url="https://api.openai.com/v1",
                api_key="test-key",
                http_client=client,
                tools=[],  # no tools available
                prompt_manager=mock_prompt_manager,
            )

            request = OpenAIChatRequest(
                model=OPENAI_TEST_MODEL,
                messages=[{"role": "user", "content": "What's the weather?"}],
            )

            # Mock reasoning step (no tools to use)
            reasoning_step = ReasoningStepFactory.finished_step(
                "No tools available, using knowledge",
            )
            reasoning_response = create_reasoning_response(reasoning_step, "chatcmpl-reasoning")
            reasoning_response.created = 1234567890
            reasoning_response.model = OPENAI_TEST_MODEL
            reasoning_response.usage.prompt_tokens = 10
            reasoning_response.usage.completion_tokens = 5
            reasoning_response.usage.total_tokens = 15

            # Mock synthesis response
            synthesis_response = create_simple_response(
                "I don't have access to weather tools.",
                "chatcmpl-synthesis",
            )
            synthesis_response.created = 1234567890
            synthesis_response.model = OPENAI_TEST_MODEL
            synthesis_response.usage.prompt_tokens = 15
            synthesis_response.usage.completion_tokens = 8
            synthesis_response.usage.total_tokens = 23

            with respx.mock:
                respx.post("https://api.openai.com/v1/chat/completions").mock(
                    side_effect=[
                        httpx.Response(200, json=reasoning_response.model_dump()),
                        httpx.Response(200, json=synthesis_response.model_dump()),
                    ],
                )

                result = await agent.execute(request)

                # Should complete successfully without tools - verify specific response content
                assert result is not None
                assert isinstance(result, OpenAIChatResponse)
                assert result.choices[0].message.content == "I don't have access to weather tools."

    @pytest.mark.asyncio
    async def test__execute_stream__no_tools_available(self) -> None:
        """Test ReasoningAgent streaming when no tools are available."""
        async with httpx.AsyncClient() as client:
            # Create empty tools list for "no tools" test
            tools = []

            # Create mock prompt manager
            mock_prompt_manager = AsyncMock(spec=PromptManager)
            mock_prompt_manager.get_prompt.return_value = "Test system prompt"

            agent = ReasoningAgent(
                base_url="https://api.openai.com/v1",
                api_key="test-key",
                http_client=client,
                tools=tools,
                prompt_manager=mock_prompt_manager,
            )

            request = OpenAIChatRequest(
                model=OPENAI_TEST_MODEL,
                messages=[{"role": "user", "content": "What's the weather?"}],
                stream=True,
            )

            # Mock streaming responses (reasoning + synthesis)
            reasoning_data = {"id": "test", "choices": [{"delta": {"content": "thinking"}}]}
            reasoning_chunks = [
                f"data: {json.dumps(reasoning_data)}\n\n",
            ]

            synthesis_data = {"id": "test", "choices": [{"delta": {"content": "response"}}]}
            synthesis_chunks = [
                f"data: {json.dumps(synthesis_data)}\n\n",
                "data: [DONE]\n\n",
            ]

            all_chunks = reasoning_chunks + synthesis_chunks

            with respx.mock:
                # Mock reasoning step generation
                reasoning_step = ReasoningStepFactory.finished_step("No tools available")
                reasoning_response = create_reasoning_response(
                    reasoning_step, "chatcmpl-reasoning",
                )
                reasoning_response.created = 1234567890
                reasoning_response.model = OPENAI_TEST_MODEL

                respx.post("https://api.openai.com/v1/chat/completions").mock(
                    return_value=httpx.Response(200, json=reasoning_response.model_dump()),
                )

                # Mock streaming synthesis
                respx.post("https://api.openai.com/v1/chat/completions").mock(
                    return_value=httpx.Response(
                        200,
                        text="\n".join(all_chunks),
                        headers={"content-type": "text/plain"},
                    ),
                )

                chunks = []
                async for chunk in agent.execute_stream(request):
                    chunks.append(chunk)

                # Should have reasoning events + final response + [DONE]
                assert len(chunks) >= 2  # At least some events + [DONE]
                assert chunks[-1] == "data: [DONE]\n\n"

    def test_build_reasoning_summary_with_tool_results(self):
        """Test that tool results are included in reasoning summary."""
        # Create minimal reasoning agent for testing
        http_client = httpx.AsyncClient()
        tools = [function_to_tool(get_weather)]
        reasoning_agent_simple = ReasoningAgent(
            base_url="http://test",
            api_key="test-key",
            http_client=http_client,
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

        tools = [function_to_tool(complex_analysis, name="analyze_data")]

        complex_tool_agent = ReasoningAgent(
            base_url="http://test",
            api_key="test-key",
            http_client=http_client,
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
        http_client = httpx.AsyncClient()
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
            base_url="http://test",
            api_key="test-key",
            http_client=http_client,
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

        # Mock the reasoning step generation to return a simple finished step
        with patch.object(context_building_agent, '_generate_reasoning_step') as mock_generate:
            mock_generate.return_value = ReasoningStep(
                thought="I can answer this directly",
                next_action=ReasoningAction.FINISHED,
                tools_to_use=[],
                concurrent_execution=False,
            )

            # Collect all events
            async for event_type, event_data in context_building_agent._core_reasoning_process(sample_context_request):  # noqa: E501
                events.append((event_type, event_data))

        # Find the finish event
        finish_events = [event for event in events if event[0] == "finish"]
        assert len(finish_events) == 1

        context = finish_events[0][1]["context"]

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

        with patch.object(context_building_agent, '_generate_reasoning_step') as mock_generate:
            mock_generate.return_value = expected_step

            # Collect all events
            async for event_type, event_data in context_building_agent._core_reasoning_process(sample_context_request):  # noqa: E501
                events.append((event_type, event_data))

        # Find the finish event and extract context
        finish_events = [event for event in events if event[0] == "finish"]
        context = finish_events[0][1]["context"]

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
        event_types = [event[0] for event in events]
        assert event_types == ["start_step", "step_plan", "start_tools", "complete_tools", "complete_step", "finish"]  # noqa: E501

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

        with patch.object(context_building_agent, '_generate_reasoning_step') as mock_generate:
            mock_generate.side_effect = [step1, step2, step3]

            # Collect all events
            async for event_type, event_data in context_building_agent._core_reasoning_process(sample_context_request):  # noqa: E501
                events.append((event_type, event_data))

        # Find the finish event and extract context
        finish_events = [event for event in events if event[0] == "finish"]
        context = finish_events[0][1]["context"]

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
        assert search_result.result == {"query": "Tokyo weather", "results": ["result1", "result2"]}  # noqa: E501

        # Verify second tool result (weather)
        weather_result = context["tool_results"][1]
        assert weather_result.tool_name == "get_weather"
        assert weather_result.success is True
        assert weather_result.result == {"location": "Tokyo", "temperature": "22°C", "condition": "Sunny"}  # noqa: E501

    @pytest.mark.asyncio
    async def test_context_preservation_across_tool_executions(self):
        """Test that context is preserved across sequential tool executions."""
        http_client = httpx.AsyncClient()
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
            base_url="http://test",
            api_key="test-key",
            http_client=http_client,
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

        # Create final synthesis response
        synthesis_response = create_simple_response(
            content="Based on my knowledge, Tokyo weather...",
            completion_id="chatcmpl-synthesis",
        )
        synthesis_response.created = 1234567890
        synthesis_response.usage.prompt_tokens = 20
        synthesis_response.usage.completion_tokens = 10
        synthesis_response.usage.total_tokens = 30

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json=step1_response.model_dump()),
                httpx.Response(200, json=step2_response.model_dump()),
                httpx.Response(200, json=synthesis_response.model_dump()),
            ],
        )

        result = await reasoning_agent.execute(sample_chat_request)

        # Should complete successfully despite tool failures - verify specific response
        assert result is not None
        assert isinstance(result, OpenAIChatResponse)
        assert result.choices[0].message.content == "Based on my knowledge, Tokyo weather..."

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

        # Create synthesis response for when max iterations reached
        synthesis_response = create_simple_response(
            "After extensive reasoning...", "chatcmpl-synthesis",
        )
        synthesis_response.created = 1234567890
        synthesis_response.usage.prompt_tokens = 50
        synthesis_response.usage.completion_tokens = 15
        synthesis_response.usage.total_tokens = 65

        # Mock exactly 20 reasoning calls (max iterations) then synthesis calls
        # After 20 iterations, it should move to synthesis
        reasoning_calls = [continue_thinking_response] * 20  # Exactly max iterations
        synthesis_calls = [synthesis_response] * 10  # Multiple synthesis calls for safety
        all_calls = reasoning_calls + synthesis_calls

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[httpx.Response(200, json=resp.model_dump()) for resp in all_calls],
        )

        result = await reasoning_agent.execute(sample_chat_request)

        # Should complete after max iterations and synthesize final response
        assert result is not None
        assert isinstance(result, OpenAIChatResponse)
        assert result.choices[0].message.content == "After extensive reasoning..."


# =============================================================================
# JSON Mode Integration Tests
# =============================================================================

class TestJSONModeIntegration:
    """Test structured outputs integration with ReasoningAgent."""

    @pytest.fixture
    def mock_reasoning_agent(self):
        """Create a reasoning agent with mocked OpenAI client for testing."""
        http_client = httpx.AsyncClient()
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
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            http_client=http_client,
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

        # Create a proper mock request object
        mock_request = create_mock_request()

        # Mock the async create method to return our mock response
        async def mock_create(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202, ARG001
            return mock_response

        with patch.object(mock_reasoning_agent.openai_client.chat.completions, 'create', side_effect=mock_create):  # noqa: E501
            result = await mock_reasoning_agent._generate_reasoning_step(
                mock_request,
                {"steps": [], "tool_results": [], "final_thoughts": "", "user_request": None},
                "test prompt",
            )

        # Verify fallback behavior
        assert "Unable to generate structured reasoning step" in result.thought
        assert "Error:" in result.thought  # Should include error details
        assert "invalid json {{" in result.thought  # Should include raw response
        assert result.next_action == ReasoningAction.CONTINUE_THINKING
        assert result.tools_to_use == []


# =============================================================================
# Error Recovery Tests
# =============================================================================

class TestErrorRecovery:
    """Test error recovery and fallback mechanisms."""

    @pytest.fixture
    def error_prone_agent(self):
        """Agent with tools that can fail in various ways."""
        http_client = httpx.AsyncClient()
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
            base_url="http://test",
            api_key="test-key",
            http_client=http_client,
            tools=tools,
            prompt_manager=mock_prompt_manager,
        )

    @pytest.mark.asyncio
    async def test_validation_error_handling(self, error_prone_agent: ReasoningAgent):
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
    async def test_type_error_handling(self, error_prone_agent: ReasoningAgent):
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
    async def test_network_error_handling(self, error_prone_agent: ReasoningAgent):
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
    async def test_mixed_success_failure_parallel(self, error_prone_agent: ReasoningAgent):
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
    async def test_unknown_tool_error(self, error_prone_agent: ReasoningAgent):
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
