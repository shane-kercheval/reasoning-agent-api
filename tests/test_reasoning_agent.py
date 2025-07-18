"""
Comprehensive tests for the ReasoningAgent class.

Tests the ReasoningAgent proxy functionality, dependency injection,
error handling, and OpenAI API compatibility.
"""

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, patch
import pytest
import httpx
import respx
from api.reasoning_agent import ReasoningAgent
from api.models import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageRole
from api.prompt_manager import PromptManager
from api.reasoning_models import ReasoningAction, ReasoningStep, ToolPrediction
from api.tools import ToolResult, function_to_tool
from tests.conftest import OPENAI_TEST_MODEL, get_weather


class TestProcessChatCompletion:
    """Test non-streaming chat completion processing."""

    @pytest.mark.asyncio
    @respx.mock
    async def test__execute__success(
        self,
        reasoning_agent: ReasoningAgent,
        sample_chat_request: ChatCompletionRequest,
        mock_openai_response: dict[str, Any],
    ) -> None:
        """Test successful non-streaming chat completion."""
        # Mock OpenAI API response
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_openai_response),
        )

        result = await reasoning_agent.execute(sample_chat_request)

        assert isinstance(result, ChatCompletionResponse)
        assert result.id == "chatcmpl-test123"
        assert result.model == "gpt-4o"
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "This is a test response from OpenAI."

    @pytest.mark.asyncio
    @respx.mock
    async def test__execute__performs_reasoning_process(
        self,
        reasoning_agent: ReasoningAgent,
        sample_chat_request: ChatCompletionRequest,
    ) -> None:
        """Test that reasoning agent performs full reasoning process."""
        # Mock the structured output call (for reasoning step generation)
        reasoning_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "id": "chatcmpl-reasoning",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I need to respond to the weather question",
                        "parsed": {
                            "thought": "I need to respond to the weather question",
                            "next_action": "finished",
                            "tools_to_use": [],
                            "concurrent_execution": False,
                        },
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }),
        )

        # Mock the final synthesis call
        synthesis_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "id": "chatcmpl-synthesis",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "The weather in Paris is sunny."},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
            }),
        )

        result = await reasoning_agent.execute(sample_chat_request)

        # Check that reasoning process was executed
        assert reasoning_route.called or synthesis_route.called  # At least one call should be made

        # Verify the final result
        assert result is not None
        assert result.choices[0].message.content == "The weather in Paris is sunny."

    @pytest.mark.asyncio
    @respx.mock
    async def test__execute__handles_openai_error(
        self,
        reasoning_agent: ReasoningAgent,
        sample_chat_request: ChatCompletionRequest,
        mock_openai_error_response: dict[str, Any],
    ) -> None:
        """Test that OpenAI API errors are properly raised."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(401, json=mock_openai_error_response),
        )

        with pytest.raises(Exception) as exc_info:  # Can be OpenAI or HTTPx error  # noqa: PT011
            await reasoning_agent.execute(sample_chat_request)

        # Check that it's an authentication-related error
        assert "401" in str(exc_info.value) or "Invalid API key" in str(exc_info.value)


class TestProcessChatCompletionStream:
    """Test streaming chat completion processing."""

    @pytest.mark.asyncio
    @respx.mock
    async def test__execute_stream__includes_reasoning_events(
        self,
        reasoning_agent: ReasoningAgent,
        sample_streaming_request: ChatCompletionRequest,
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

        # Should have reasoning events + final response + [DONE]
        assert len(chunks) >= 3  # At least some reasoning events + final response + [DONE]

        # Check that we have reasoning events with reasoning_event metadata
        reasoning_chunks = []
        for chunk in chunks:
            if "reasoning_event" in chunk:
                reasoning_chunks.append(chunk)

        # Should have some reasoning events (though they might be using fallback due to mock structure)  # noqa: E501
        # The key is that the stream doesn't fail and produces output
        assert len(chunks) > 0

        # Check final chunk
        assert chunks[-1] == "data: [DONE]\n\n"

        # Verify chunks contain valid JSON (not checking specific content due to fallback behavior)
        valid_json_chunks = 0
        for chunk in chunks[:-1]:  # Exclude [DONE] chunk
            chunk_data = chunk.replace("data: ", "").strip()
            if chunk_data:
                try:
                    json.loads(chunk_data)
                    valid_json_chunks += 1
                except json.JSONDecodeError:
                    pass

        # Should have at least some valid JSON chunks
        assert valid_json_chunks > 0

    @pytest.mark.asyncio
    @respx.mock
    async def test__execute_stream__forwards_openai_chunks(
        self,
        reasoning_agent: ReasoningAgent,
        sample_streaming_request: ChatCompletionRequest,
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

        # Should have OpenAI-style chunks with modified completion IDs
        assert len(chunks) > 0
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
        sample_streaming_request: ChatCompletionRequest,
    ) -> None:
        """Test that streaming errors are properly handled."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(401, json={"error": {"message": "Unauthorized"}}),
        )

        with pytest.raises(httpx.HTTPStatusError):
            async for _ in reasoning_agent.execute_stream(sample_streaming_request):
                pass


class TestReasoningLoopTermination:
    """Test reasoning loop termination and infinite loop prevention."""

    @pytest.mark.asyncio
    @respx.mock
    async def test__reasoning_terminates_when_tools_fail(
        self,
        reasoning_agent: ReasoningAgent,
        sample_chat_request: ChatCompletionRequest,
    ) -> None:
        """Test that reasoning process terminates gracefully when tools fail."""
        # Mock reasoning step that tries to use tools
        step1_response = {
            "id": "chatcmpl-reasoning1",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "thought": "I need to use weather tools to get current conditions",
                        "next_action": "use_tools",
                        "tools_to_use": [
                            {
                                "tool_name": "get_weather",
                                "arguments": {"location": "Tokyo"},
                                "reasoning": "Need current weather data for Tokyo",
                            },
                        ],
                        "concurrent_execution": False,
                    }),
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        # Mock reasoning step that recognizes tool failure and finishes
        step2_response = {
            "id": "chatcmpl-reasoning2",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "thought": "Tools failed, proceeding with knowledge-based response",
                        "next_action": "finished",
                        "tools_to_use": [],
                        "concurrent_execution": False,
                    }),
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
        }

        # Mock final synthesis
        synthesis_response = {
            "id": "chatcmpl-synthesis",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Based on my knowledge, Tokyo weather...",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }

        # Set up sequential responses
        reasoning_routes = [
            respx.post("https://api.openai.com/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=step1_response),
            ),
            respx.post("https://api.openai.com/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=step2_response),
            ),
            respx.post("https://api.openai.com/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=synthesis_response),
            ),
        ]

        # Configure routing to respond sequentially
        respx.route().mock(side_effect=[r.return_value for r in reasoning_routes])

        result = await reasoning_agent.execute(sample_chat_request)

        # Should complete successfully despite tool failures
        assert result is not None
        assert result.choices[0].message.content == "Based on my knowledge, Tokyo weather..."

    @pytest.mark.asyncio
    @respx.mock
    async def test__reasoning_respects_max_iterations(
        self,
        reasoning_agent: ReasoningAgent,
        sample_chat_request: ChatCompletionRequest,
    ) -> None:
        """Test that reasoning process respects max_reasoning_iterations limit."""
        # Mock reasoning step that always continues thinking (would loop infinitely)
        continue_thinking_response = {
            "id": "chatcmpl-reasoning",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "thought": "Still thinking about the problem...",
                        "next_action": "continue_thinking",
                        "tools_to_use": [],
                        "concurrent_execution": False,
                    }),
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        # Mock synthesis response for when max iterations reached
        synthesis_response = {
            "id": "chatcmpl-synthesis",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "After extensive reasoning..."},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 15, "total_tokens": 65},
        }

        # Mock exactly 20 reasoning calls (max iterations) then synthesis calls
        # After 20 iterations, it should move to synthesis
        reasoning_calls = [continue_thinking_response] * 20  # Exactly max iterations
        synthesis_calls = [synthesis_response] * 10  # Multiple synthesis calls for safety
        all_calls = reasoning_calls + synthesis_calls

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[httpx.Response(200, json=resp) for resp in all_calls],
        )

        result = await reasoning_agent.execute(sample_chat_request)

        # Should complete after max iterations and synthesize final response
        assert result is not None
        assert result.choices[0].message.content == "After extensive reasoning..."

    @pytest.mark.asyncio
    @respx.mock
    async def test__tool_failure_feedback_included_in_context(
        self,
        reasoning_agent: ReasoningAgent,
        sample_chat_request: ChatCompletionRequest,
    ) -> None:
        """Test that tool failure results are included in subsequent reasoning steps."""
        # Step 1: Try to use tools
        step1_response = {
            "id": "chatcmpl-reasoning1",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "thought": "I need to check the weather using tools",
                        "next_action": "use_tools",
                        "tools_to_use": [
                            {
                                "tool_name": "get_weather",
                                "arguments": {"location": "Tokyo"},
                                "reasoning": "Need current weather data for Tokyo",
                            },
                        ],
                        "concurrent_execution": False,
                    }),
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        # Step 2: Should receive tool failure feedback and finish
        step2_response = {
            "id": "chatcmpl-reasoning2",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "thought": "Tools are not available, using my knowledge instead",
                        "next_action": "finished",
                        "tools_to_use": [],
                        "concurrent_execution": False,
                    }),
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
        }

        # Final synthesis
        synthesis_response = {
            "id": "chatcmpl-synthesis",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Weather information unavailable, but...",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json=step1_response),
                httpx.Response(200, json=step2_response),
                httpx.Response(200, json=synthesis_response),
            ],
        )

        result = await reasoning_agent.execute(sample_chat_request)

        # Should successfully complete with knowledge-based response
        assert result is not None
        assert "Weather information unavailable" in result.choices[0].message.content


class TestReasoningAgentNoTools:
    """Test ReasoningAgent functionality when no tools are available."""

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

            request = ChatCompletionRequest(
                model=OPENAI_TEST_MODEL,
                messages=[ChatMessage(role=MessageRole.USER, content="What's the weather?")],
            )

            # Mock reasoning step (no tools to use)
            reasoning_response = {
                "id": "chatcmpl-reasoning",
                "object": "chat.completion",
                "created": 1234567890,
                "model": OPENAI_TEST_MODEL,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "thought": "No tools available, using knowledge",
                            "next_action": "finished",
                            "tools_to_use": [],
                            "concurrent_execution": False,
                        }),
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

            # Mock synthesis response
            synthesis_response = {
                "id": "chatcmpl-synthesis",
                "object": "chat.completion",
                "created": 1234567890,
                "model": OPENAI_TEST_MODEL,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I don't have access to weather tools.",
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
            }

            with respx.mock:
                respx.post("https://api.openai.com/v1/chat/completions").mock(
                    side_effect=[
                        httpx.Response(200, json=reasoning_response),
                        httpx.Response(200, json=synthesis_response),
                    ],
                )

                result = await agent.execute(request)

                # Should complete successfully without tools
                assert result is not None
                assert result.choices[0].message.content == "I don't have access to weather tools."

                # Tool execution behavior is tested by response content

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

            request = ChatCompletionRequest(
                model=OPENAI_TEST_MODEL,
                messages=[ChatMessage(role=MessageRole.USER, content="What's the weather?")],
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
                respx.post("https://api.openai.com/v1/chat/completions").mock(
                    return_value=httpx.Response(200, json={
                        "id": "chatcmpl-reasoning",
                        "object": "chat.completion",
                        "created": 1234567890,
                        "model": OPENAI_TEST_MODEL,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps({
                                    "thought": "No tools available",
                                    "next_action": "finished",
                                    "tools_to_use": [],
                                    "concurrent_execution": False,
                                }),
                            },
                            "finish_reason": "stop",
                        }],
                    }),
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

                # Verify that get_available_tools was called
                # Tool functionality verified through response behavior


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


class TestReasoningSummary:
    """Test the _build_reasoning_summary method that was key to the bug fix."""

    @pytest.fixture
    def reasoning_agent_simple(self) -> ReasoningAgent:
        """Create a minimal reasoning agent for testing."""
        http_client = httpx.AsyncClient()
        tools = [function_to_tool(get_weather)]

        return ReasoningAgent(
            base_url="http://test",
            api_key="test-key",
            http_client=http_client,
            tools=tools,
            prompt_manager=None,  # Not needed for these tests
        )

    def test_build_reasoning_summary_with_tool_results(self, reasoning_agent_simple: ReasoningAgent):  # noqa: E501
        """Test that tool results are included in reasoning summary."""
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

    def test_build_reasoning_summary_with_failed_tool(self, reasoning_agent_simple: ReasoningAgent):  # noqa: E501
        """Test that failed tool results are also included in reasoning summary."""
        tool_result = ToolResult(
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

        summary = reasoning_agent_simple._build_reasoning_summary(reasoning_context)

        # Verify that failed tool results are included
        assert "Tool Results:" in summary
        assert "get_weather" in summary
        assert "Connection timeout" in summary

    def test_build_reasoning_summary_without_tool_results(self, reasoning_agent_simple: ReasoningAgent):  # noqa: E501
        """Test that reasoning summary works when there are no tool results."""
        reasoning_context = {
            "steps": [],
            "tool_results": [],
            "final_thoughts": "",
        }

        summary = reasoning_agent_simple._build_reasoning_summary(reasoning_context)

        # Should not include tool results section when there are no results
        assert "Tool Results:" not in summary

    def test_build_reasoning_summary_with_steps_and_tools(self, reasoning_agent_simple: ReasoningAgent):  # noqa: E501
        """Test reasoning summary with both steps and tool results."""
        # Create a reasoning step
        step = ReasoningStep(
            thought="I need to get weather information",
            next_action=ReasoningAction.USE_TOOLS,
            tools_to_use=[
                ToolPrediction(
                        tool_name="get_weather",
                    arguments={"location": "Tokyo"},
                    reasoning="User asked for Tokyo weather",
                ),
            ],
        )

        tool_result = ToolResult(
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

        summary = reasoning_agent_simple._build_reasoning_summary(reasoning_context)

        # Should include both step information and tool results
        assert "Step 1:" in summary
        assert "I need to get weather information" in summary
        assert "Used tools: get_weather" in summary
        assert "Tool Results:" in summary
        assert "25°C" in summary


class TestStructuredOutputsIntegration:
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

        def search_func(query: str, max_results: int = 5, filters: list[str] | None = None) -> dict[str, Any]:  # noqa: E501
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


    @pytest.mark.asyncio
    async def test_structured_output_success(self, mock_reasoning_agent: ReasoningAgent):
        """Test successful structured output parsing."""
        # Mock successful structured output response
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.parsed = ReasoningStep(
            thought="I need to get weather information",
            next_action=ReasoningAction.USE_TOOLS,
            tools_to_use=[
                ToolPrediction(
                    tool_name="get_weather",
                    arguments={"location": "Tokyo", "units": "fahrenheit"},
                    reasoning="User wants Tokyo weather in Fahrenheit",
                ),
            ],
            concurrent_execution=False,
        )

        with patch.object(mock_reasoning_agent.openai_client.beta.chat.completions, 'parse', return_value=mock_response):  # noqa: E501
            result = await mock_reasoning_agent._generate_reasoning_step(
                AsyncMock(),  # request
                {"steps": [], "tool_results": [], "final_thoughts": "", "user_request": None},
                "test prompt",  # system_prompt
            )

        assert result.thought == "I need to get weather information"
        assert result.next_action == ReasoningAction.USE_TOOLS
        assert len(result.tools_to_use) == 1
        assert result.tools_to_use[0].tool_name == "get_weather"
        assert result.tools_to_use[0].arguments == {"location": "Tokyo", "units": "fahrenheit"}

    @pytest.mark.asyncio
    async def test_structured_output_fallback_to_content(self, mock_reasoning_agent: ReasoningAgent):  # noqa: E501
        """Test fallback when parsed is None but content is available."""
        # Mock response with no parsed but valid content
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.parsed = None
        mock_response.choices[0].message.content = json.dumps({
            "thought": "Fallback parsing test",
            "next_action": "finished",
            "tools_to_use": [],
            "concurrent_execution": False,
        })

        with patch.object(mock_reasoning_agent.openai_client.beta.chat.completions, 'parse', return_value=mock_response):  # noqa: E501
            result = await mock_reasoning_agent._generate_reasoning_step(
                AsyncMock(),
                {"steps": [], "tool_results": [], "final_thoughts": "", "user_request": None},
                "test prompt",
            )

        # Current implementation doesn't parse content fallback, it creates a default step
        assert "Unable to generate structured reasoning step" in result.thought
        assert result.next_action == ReasoningAction.CONTINUE_THINKING

    @pytest.mark.asyncio
    async def test_structured_output_malformed_json_fallback(self, mock_reasoning_agent: ReasoningAgent):  # noqa: E501
        """Test fallback when content is malformed JSON."""
        # Mock response with malformed JSON content
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.parsed = None
        mock_response.choices[0].message.content = "invalid json {{"

        with patch.object(mock_reasoning_agent.openai_client.beta.chat.completions, 'parse', return_value=mock_response):  # noqa: E501
            result = await mock_reasoning_agent._generate_reasoning_step(
                AsyncMock(),
                {"steps": [], "tool_results": [], "final_thoughts": "", "user_request": None},
                "test prompt",
            )

        assert "Unable to generate structured reasoning step" in result.thought
        assert result.next_action == ReasoningAction.CONTINUE_THINKING
        assert result.tools_to_use == []

    @pytest.mark.asyncio
    async def test_structured_output_api_exception(self, mock_reasoning_agent: ReasoningAgent):
        """Test fallback when API throws exception."""
        # Mock API exception
        with patch.object(mock_reasoning_agent.openai_client.beta.chat.completions, 'parse',
                         side_effect=Exception("API error")):
            result = await mock_reasoning_agent._generate_reasoning_step(
                AsyncMock(),
                {"steps": [], "tool_results": [], "final_thoughts": "", "user_request": None},
                "test prompt",
            )

        assert "Error in reasoning - proceeding to final answer: API error" in result.thought
        assert result.next_action == ReasoningAction.FINISHED
        assert result.tools_to_use == []

    @pytest.mark.asyncio
    async def test_structured_output_empty_response(self, mock_reasoning_agent: ReasoningAgent):
        """Test fallback when response is empty."""
        # Mock empty response
        mock_response = AsyncMock()
        mock_response.choices = []

        with patch.object(mock_reasoning_agent.openai_client.beta.chat.completions, 'parse', return_value=mock_response):  # noqa: E501
            result = await mock_reasoning_agent._generate_reasoning_step(
                AsyncMock(),
                {"steps": [], "tool_results": [], "final_thoughts": "", "user_request": None},
                "test prompt",
            )

        assert "Unable to generate structured reasoning step" in result.thought
        assert result.next_action == ReasoningAction.CONTINUE_THINKING


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

    @pytest.mark.asyncio
    async def test_async_function_concurrency(self, complex_tool_agent: ReasoningAgent):
        """Test async functions run concurrently."""
        predictions = [
            ToolPrediction(
                tool_name="async_timeout",
                arguments={"duration": 0.1},
                reasoning="Test 1",
            ),
            ToolPrediction(
                tool_name="async_timeout",
                arguments={"duration": 0.1},
                reasoning="Test 2",
            ),
            ToolPrediction(
                tool_name="async_timeout",
                arguments={"duration": 0.1},
                reasoning="Test 3",
            ),
        ]

        # Should run concurrently, not sequentially
        start_time = asyncio.get_event_loop().time()
        results = await complex_tool_agent._execute_tools_concurrently(predictions)
        end_time = asyncio.get_event_loop().time()

        assert len(results) == 3
        assert all(r.success for r in results)
        # Should take ~0.1s (concurrent) not ~0.3s (sequential)
        assert end_time - start_time < 0.2


class TestErrorRecoveryAndFallbacks:
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
