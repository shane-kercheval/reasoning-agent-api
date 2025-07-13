"""
End-to-end integration tests for the reasoning agent with MCP tools.

These tests verify the complete flow from chat completion request through
reasoning, tool execution, and response generation using in-memory MCP servers.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

from api.reasoning_agent import ReasoningAgent
from api.mcp import MCPClient, MCPManager, MCPServerConfig
from api.models import ChatCompletionRequest, ChatMessage
from api.prompt_manager import PromptManager
from api.reasoning_models import ReasoningAction, ReasoningStep, ToolPrediction
from tests.mcp_servers.server_a import get_server_instance as get_server_a


class TestReasoningAgentIntegration:
    """Test full reasoning agent with in-memory MCP servers."""

    @pytest_asyncio.fixture
    async def reasoning_agent(self):
        """Create a reasoning agent with in-memory MCP server."""
        # Create MCP manager with in-memory server
        config = MCPServerConfig(name="test_server", url="", enabled=True)
        mcp_manager = MCPManager([config])

        # Set up in-memory server instead of HTTP connection
        client = MCPClient(config)
        client.set_server_instance(get_server_a())
        mcp_manager._clients["test_server"] = client

        # Create mock HTTP client for OpenAI API calls
        http_client = AsyncMock()

        # Create prompt manager
        prompt_manager = PromptManager()

        # Create reasoning agent
        agent = ReasoningAgent(
            http_client=http_client,
            mcp_manager=mcp_manager,
            prompt_manager=prompt_manager,
        )

        return agent, http_client

    @pytest.mark.asyncio
    async def test_end_to_end_reasoning_with_tools(self, reasoning_agent: tuple[ReasoningAgent, AsyncMock]):  # noqa: E501
        """Test complete reasoning flow with tool execution."""
        agent, mock_http_client = reasoning_agent

        # Mock OpenAI responses for reasoning steps
        mock_reasoning_response = {
            "choices": [{
                "message": {
                    "content": '{"thought": "I need to get weather data", "next_action": "use_tools", "tools_to_use": [{"server_name": "test_server", "tool_name": "weather_api", "arguments": {"location": "Tokyo"}, "reasoning": "Get weather for Tokyo"}], "parallel_execution": false}',  # noqa: E501
                },
            }],
        }

        mock_final_response = {
            "choices": [{
                "message": {
                    "content": "Based on the weather data I retrieved, Tokyo is currently experiencing sunny conditions with a temperature of 25°C.",  # noqa: E501
                },
            }],
        }

        # Configure mock to return reasoning step first, then final response
        mock_http_client.post.side_effect = [
            AsyncMock(json=AsyncMock(return_value=mock_reasoning_response)),
            AsyncMock(json=AsyncMock(return_value=mock_final_response)),
        ]

        # Create test request
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[
                ChatMessage(role="user", content="What's the weather like in Tokyo?"),
            ],
            max_tokens=100,
            temperature=0.7,
        )

        # Execute reasoning
        response = await agent.create_chat_completion(request)

        # Verify the response
        assert response is not None
        assert len(response.choices) == 1
        assert "Tokyo" in response.choices[0].message.content
        assert "sunny" in response.choices[0].message.content or "weather" in response.choices[0].message.content  # noqa: E501

        # Verify OpenAI API was called (for reasoning step and final synthesis)
        assert mock_http_client.post.call_count >= 2

    @pytest.mark.asyncio
    async def test_streaming_reasoning_with_tools(self, reasoning_agent: tuple[ReasoningAgent, AsyncMock]):  # noqa: E501
        """Test streaming reasoning flow with tool execution."""
        agent, mock_http_client = reasoning_agent

        # Mock OpenAI streaming responses
        async def mock_streaming_response():  # noqa: ANN202
            """Mock streaming response generator."""
            chunks = [
                '{"choices": [{"delta": {"content": "I need to check the weather"}}]}',
                '{"choices": [{"delta": {"content": " in Tokyo using the weather tool."}}]}',
                '{"choices": [{"finish_reason": "stop"}]}',
            ]
            for chunk in chunks:
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        # Mock reasoning step response
        mock_reasoning_response = {
            "choices": [{
                "message": {
                    "content": '{"thought": "I need weather data for Tokyo", "next_action": "use_tools", "tools_to_use": [{"server_name": "test_server", "tool_name": "weather_api", "arguments": {"location": "Tokyo"}, "reasoning": "Get current weather"}], "parallel_execution": false}',  # noqa: E501
                },
            }],
        }

        mock_http_client.post.return_value = AsyncMock(json=AsyncMock(return_value=mock_reasoning_response))  # noqa: E501

        # Mock streaming response
        with patch.object(agent, '_stream_openai_response', return_value=mock_streaming_response()):  # noqa: E501
            # Create test request
            request = ChatCompletionRequest(
                model="gpt-4o",
                messages=[
                    ChatMessage(role="user", content="What's the weather like in Tokyo?"),
                ],
                max_tokens=100,
                temperature=0.7,
                stream=True,
            )

            # Execute streaming reasoning
            chunks = []
            async for chunk in agent.create_chat_completion_stream(request):
                chunks.append(chunk)

            # Verify we got streaming chunks
            assert len(chunks) > 0

            # Verify reasoning events are included
            reasoning_events = [chunk for chunk in chunks if '"type":"reasoning_step"' in chunk]
            tool_events = [chunk for chunk in chunks if '"type":"tool_execution"' in chunk]

            assert len(reasoning_events) > 0, "Should have reasoning step events"
            assert len(tool_events) > 0, "Should have tool execution events"

    @pytest.mark.asyncio
    async def test_tool_execution_error_handling(self, reasoning_agent: tuple[ReasoningAgent, AsyncMock]):  # noqa: E501
        """Test error handling when tool execution fails."""
        agent, mock_http_client = reasoning_agent

        # Mock reasoning step that requests a failing tool
        mock_reasoning_response = {
            "choices": [{
                "message": {
                    "content": '{"thought": "I need to test error handling", "next_action": "use_tools", "tools_to_use": [{"server_name": "test_server", "tool_name": "failing_tool", "arguments": {"should_fail": true}, "reasoning": "Test error handling"}], "parallel_execution": false}',  # noqa: E501
                },
            }],
        }

        mock_final_response = {
            "choices": [{
                "message": {
                    "content": "I encountered an error while trying to execute the tool, but I can still provide a helpful response.",  # noqa: E501
                },
            }],
        }

        mock_http_client.post.side_effect = [
            AsyncMock(json=AsyncMock(return_value=mock_reasoning_response)),
            AsyncMock(json=AsyncMock(return_value=mock_final_response)),
        ]

        # Create test request
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[
                ChatMessage(role="user", content="Test error handling"),
            ],
            max_tokens=100,
        )

        # Execute reasoning - should not raise exception despite tool failure
        response = await agent.create_chat_completion(request)

        # Verify we still get a response even with tool failure
        assert response is not None
        assert len(response.choices) == 1
        assert response.choices[0].message.content is not None

    @pytest.mark.asyncio
    async def test_parallel_tool_execution(self, reasoning_agent: tuple[ReasoningAgent, AsyncMock]):  # noqa: E501
        """Test parallel execution of multiple tools."""
        agent, mock_http_client = reasoning_agent

        # Mock reasoning step with parallel tools
        mock_reasoning_response = {
            "choices": [{
                "message": {
                    "content": '{"thought": "I need both weather and web search", "next_action": "use_tools", "tools_to_use": [{"server_name": "test_server", "tool_name": "weather_api", "arguments": {"location": "Tokyo"}, "reasoning": "Get weather"}, {"server_name": "test_server", "tool_name": "web_search", "arguments": {"query": "Tokyo news"}, "reasoning": "Get news"}], "parallel_execution": true}',  # noqa: E501
                },
            }],
        }

        mock_final_response = {
            "choices": [{
                "message": {
                    "content": "I got both weather and news data for Tokyo in parallel.",
                },
            }],
        }

        mock_http_client.post.side_effect = [
            AsyncMock(json=AsyncMock(return_value=mock_reasoning_response)),
            AsyncMock(json=AsyncMock(return_value=mock_final_response)),
        ]

        # Create test request
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[
                ChatMessage(role="user", content="Get weather and news for Tokyo"),
            ],
            max_tokens=200,
        )

        # Execute reasoning
        response = await agent.create_chat_completion(request)

        # Verify response
        assert response is not None
        assert "Tokyo" in response.choices[0].message.content

    @pytest.mark.asyncio
    async def test_reasoning_step_validation(self, reasoning_agent: tuple[ReasoningAgent, AsyncMock]):  # noqa: E501
        """Test that invalid reasoning steps are handled gracefully."""
        agent, mock_http_client = reasoning_agent

        # Mock invalid reasoning response (malformed JSON)
        mock_invalid_response = {
            "choices": [{
                "message": {
                    "content": "This is not valid JSON for a reasoning step",
                },
            }],
        }

        mock_fallback_response = {
            "choices": [{
                "message": {
                    "content": "I can still provide a helpful response even without structured reasoning.",  # noqa: E501
                },
            }],
        }

        mock_http_client.post.side_effect = [
            AsyncMock(json=AsyncMock(return_value=mock_invalid_response)),
            AsyncMock(json=AsyncMock(return_value=mock_fallback_response)),
        ]

        # Create test request
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[
                ChatMessage(role="user", content="Test invalid reasoning step"),
            ],
            max_tokens=100,
        )

        # Execute reasoning - should handle invalid JSON gracefully
        response = await agent.create_chat_completion(request)

        # Verify we still get a response
        assert response is not None
        assert len(response.choices) == 1
        assert response.choices[0].message.content is not None


class TestToolPredictionConversion:
    """Test ToolPrediction to ToolRequest conversion."""

    def test_tool_prediction_to_mcp_request_conversion(self):
        """Test that ToolPrediction converts correctly to MCP ToolRequest."""
        # Create a ToolPrediction
        prediction = ToolPrediction(
            server_name="test_server",
            tool_name="weather_api",
            arguments={"location": "Tokyo", "units": "celsius"},
            reasoning="Need weather data for the user's query",
        )

        # Convert to MCP ToolRequest
        mcp_request = prediction.to_mcp_request()

        # Verify conversion
        assert mcp_request.server_name == "test_server"
        assert mcp_request.tool_name == "weather_api"
        assert mcp_request.arguments == {"location": "Tokyo", "units": "celsius"}

        # Verify reasoning field is not included in MCP request (as expected)
        assert not hasattr(mcp_request, "reasoning")

    def test_reasoning_step_with_tool_predictions(self):
        """Test ReasoningStep with ToolPrediction objects."""
        # Create a reasoning step with tool predictions
        step = ReasoningStep(
            thought="I need to get weather and search data",
            next_action=ReasoningAction.USE_TOOLS,
            tools_to_use=[
                ToolPrediction(
                    server_name="weather_server",
                    tool_name="get_weather",
                    arguments={"city": "Tokyo"},
                    reasoning="Get current weather",
                ),
                ToolPrediction(
                    server_name="search_server",
                    tool_name="web_search",
                    arguments={"query": "Tokyo events"},
                    reasoning="Find current events",
                ),
            ],
            parallel_execution=True,
        )

        # Verify the step was created correctly
        assert step.thought == "I need to get weather and search data"
        assert step.next_action == ReasoningAction.USE_TOOLS
        assert len(step.tools_to_use) == 2
        assert step.parallel_execution is True

        # Verify tool predictions
        weather_tool = step.tools_to_use[0]
        assert weather_tool.tool_name == "get_weather"
        assert weather_tool.arguments["city"] == "Tokyo"
        assert weather_tool.reasoning == "Get current weather"

        search_tool = step.tools_to_use[1]
        assert search_tool.tool_name == "web_search"
        assert search_tool.arguments["query"] == "Tokyo events"
        assert search_tool.reasoning == "Find current events"
