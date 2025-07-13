"""
End-to-end integration tests for the reasoning agent with MCP tools.

These tests verify the complete flow from chat completion request through
reasoning, tool execution, and response generation using in-memory MCP servers.
"""

import os
import pytest
import pytest_asyncio
import httpx
from dotenv import load_dotenv

from api.reasoning_agent import ReasoningAgent
from api.mcp import MCPClient, MCPManager, MCPServerConfig
from api.models import ChatCompletionRequest, ChatMessage
from api.prompt_manager import PromptManager
from api.reasoning_models import ReasoningAction, ReasoningStep, ToolPrediction
from tests.mcp_servers.server_a import get_server_instance as get_server_a


@pytest.mark.integration
class TestReasoningAgentIntegration:
    """Test full reasoning agent with in-memory MCP servers."""

    @pytest_asyncio.fixture
    async def reasoning_agent(self):
        """Create a reasoning agent with in-memory MCP server."""
        # Load environment variables from .env file
        load_dotenv()
        
        # Skip if no OpenAI API key (integration test requirement)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY environment variable required for integration tests")

        # Create MCP manager with in-memory server
        config = MCPServerConfig(name="test_server", url="", enabled=True)
        mcp_manager = MCPManager([config])

        # Set up in-memory server instead of HTTP connection
        client = MCPClient(config)
        client.set_server_instance(get_server_a())
        mcp_manager._clients["test_server"] = client

        # Create real HTTP client for OpenAI API calls with longer timeout
        http_client = httpx.AsyncClient(timeout=60.0)

        # Create prompt manager and initialize it
        prompt_manager = PromptManager()
        await prompt_manager.initialize()

        # Create reasoning agent with real OpenAI integration
        agent = ReasoningAgent(
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            http_client=http_client,
            mcp_manager=mcp_manager,
            prompt_manager=prompt_manager,
        )

        yield agent

        # Cleanup - use try/except to handle potential event loop issues
        try:  # noqa: SIM105
            await http_client.aclose()
        except RuntimeError:
            # Event loop might be closed already in some test scenarios
            pass

    @pytest.mark.asyncio
    async def test_end_to_end_reasoning_with_tools(self, reasoning_agent: ReasoningAgent):
        """Test complete reasoning flow with tool execution."""
        agent = reasoning_agent

        # Create test request that should trigger tool usage
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[
                ChatMessage(role="user", content="I need you to use the weather_api tool from the test_server to get the current weather for Tokyo, Japan. Please use the exact tool name 'weather_api' and pass 'Tokyo' as the location parameter."),  # noqa: E501
            ],
            max_tokens=1000,
            temperature=0.1,
        )

        # Execute reasoning with real OpenAI API
        response = await agent.process_chat_completion(request)

        # Verify the response
        assert response is not None
        assert len(response.choices) == 1
        assert response.choices[0].message.content is not None
        # The response should contain information from the tool execution
        assert len(response.choices[0].message.content) > 0

    @pytest.mark.asyncio
    async def test_streaming_reasoning_with_tools(self, reasoning_agent: ReasoningAgent):
        """Test streaming reasoning flow with tool execution."""
        agent = reasoning_agent

        # Create test request for streaming
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[
                ChatMessage(role="user", content="Please use the weather_api tool from test_server to check the weather in Tokyo. Use streaming response so I can see your reasoning process step by step."),  # noqa: E501
            ],
            max_tokens=1000,
            temperature=0.1,
            stream=True,
        )

        # Execute streaming reasoning with real OpenAI API
        response_stream = agent.process_chat_completion_stream(request)

        # Collect all chunks
        chunks = []
        async for chunk in response_stream:
            chunks.append(chunk)

        # Verify we got streaming chunks
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_tool_execution_error_handling(self, reasoning_agent: ReasoningAgent):
        """Test error handling during tool execution."""
        agent = reasoning_agent

        # Create request that tries to use a non-existent tool
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[
                ChatMessage(role="user", content="Use the nonexistent_tool to do something"),
            ],
            max_tokens=1000,
            temperature=0.1,
        )

        # Should handle the error gracefully
        response = await agent.process_chat_completion(request)
        assert response is not None
        assert len(response.choices) == 1

    @pytest.mark.asyncio
    async def test_simple_conversation_no_tools(self, reasoning_agent: ReasoningAgent):
        """Test basic conversation that doesn't require tools."""
        agent = reasoning_agent

        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[
                ChatMessage(role="user", content="Hello, how are you?"),
            ],
            max_tokens=1000,
            temperature=0.1,
        )

        response = await agent.process_chat_completion(request)
        assert response is not None
        assert len(response.choices) == 1
        assert response.choices[0].message.content is not None


class TestToolPredictionConversion:
    """Test tool prediction to MCP request conversion."""

    def test_tool_prediction_to_mcp_request_conversion(self):
        """Test conversion from ToolPrediction to MCP ToolRequest."""
        prediction = ToolPrediction(
            server_name="test_server",
            tool_name="weather_api",
            arguments={"location": "Tokyo"},
            reasoning="Get weather data for Tokyo",
        )

        mcp_request = prediction.to_mcp_request()

        assert mcp_request.server_name == "test_server"
        assert mcp_request.tool_name == "weather_api"
        assert mcp_request.arguments == {"location": "Tokyo"}

    def test_reasoning_step_with_tool_predictions(self):
        """Test ReasoningStep creation with tool predictions."""
        tools = [
            ToolPrediction(
                server_name="test_server",
                tool_name="weather_api",
                arguments={"location": "Tokyo"},
                reasoning="Get weather for Tokyo",
            ),
        ]

        step = ReasoningStep(
            thought="I need to get weather information",
            next_action=ReasoningAction.USE_TOOLS,
            tools_to_use=tools,
            parallel_execution=False,
        )

        assert step.thought == "I need to get weather information"
        assert step.next_action == ReasoningAction.USE_TOOLS
        assert len(step.tools_to_use) == 1
        assert step.tools_to_use[0].tool_name == "weather_api"
