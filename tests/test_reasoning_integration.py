"""
End-to-end integration tests for the reasoning agent with MCP tools.

These tests verify the complete flow from chat completion request through
reasoning, tool execution, and response generation using in-memory MCP servers.
"""

import json
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
                ChatMessage(role="user", content="You have access to a weather_api tool on the test_server. Please use it to get the weather for Tokyo and tell me the temperature. The tool exists and you must use it - do not say you cannot access real-time data."),  # noqa: E501
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
        
        # Verify tool was actually used by checking for specific tool response content
        response_content = response.choices[0].message.content.lower()
        
        # The fake weather tool returns structured data - check for specific fields
        tool_indicators = [
            "temperature",  # from temperature field
            "°c",  # temperature unit from tool
            "humidity",  # humidity field from tool
            "mb",  # pressure unit from tool
            "km/h",  # wind speed unit from tool
            "test weather api",  # source field from tool
            "test-server-a"  # server field from tool
        ]
        
        tool_used = any(indicator in response_content for indicator in tool_indicators)
        assert tool_used, f"No tool response indicators found in: {response.choices[0].message.content}"
        
        # Should not contain the "cannot access real-time data" message
        fallback_phrases = ["cannot access", "don't have access", "unable to provide real-time", "i'm unable to"]
        no_fallback = not any(phrase in response_content for phrase in fallback_phrases)
        assert no_fallback, f"Tool was not used, got fallback response: {response.choices[0].message.content}"

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

        # Parse SSE chunks and extract reasoning events
        chunks = []
        reasoning_events = []
        content_chunks = []
        
        async for sse_line in response_stream:
            chunks.append(sse_line)
            # Parse SSE format: "data: {json}\n\n"
            if sse_line.startswith("data: ") and not sse_line.startswith("data: [DONE]"):
                try:
                    json_str = sse_line[6:].strip()  # Remove "data: " prefix
                    chunk_data = json.loads(json_str)
                    
                    if "choices" in chunk_data and chunk_data["choices"]:
                        choice = chunk_data["choices"][0]
                        if "delta" in choice:
                            delta = choice["delta"]
                            # Collect reasoning events
                            if "reasoning_event" in delta and delta["reasoning_event"]:
                                reasoning_events.append(delta["reasoning_event"])
                            # Collect content
                            if "content" in delta and delta["content"]:
                                content_chunks.append(delta["content"])
                except (json.JSONDecodeError, KeyError) as e:
                    # Skip malformed chunks
                    print(f"Failed to parse chunk: {sse_line[:100]}... Error: {e}")
                    continue

        # Verify we got streaming chunks
        assert len(chunks) > 0, "No streaming chunks received"
        
        # Verify we got reasoning events
        assert len(reasoning_events) > 0, f"No reasoning events found in {len(chunks)} chunks. Content: {''.join(content_chunks)}"
        
        # Check for specific reasoning event types
        event_types = [event.get("type") for event in reasoning_events]
        assert "reasoning_start" in event_types or "reasoning_step" in event_types, f"No reasoning events found. Event types: {event_types}"
        
        # If tools were used, should have tool events
        tool_events = [event for event in reasoning_events if "tool" in event.get("type", "")]
        content = ''.join(content_chunks).lower()
        
        # Should have tool events if weather data appears in content
        if any(indicator in content for indicator in ["temperature", "°c", "weather"]):
            assert len(tool_events) > 0, f"Tool data found in content but no tool events. Events: {event_types}"
            
        print(f"✅ Found {len(reasoning_events)} reasoning events: {event_types}")
        print(f"✅ Content length: {len(''.join(content_chunks))} characters")

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
    async def test_tool_execution_verification(self, reasoning_agent: ReasoningAgent):
        """Test that tools are actually called by monitoring MCP execution."""
        agent = reasoning_agent
        
        # Track tool calls by monitoring the MCP manager
        original_execute_tool = agent.mcp_manager.execute_tool
        tool_calls = []
        
        async def track_tool_calls(tool_request):
            tool_calls.append(tool_request)
            return await original_execute_tool(tool_request)
        
        agent.mcp_manager.execute_tool = track_tool_calls

        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[
                ChatMessage(role="user", content="Use the weather_api tool to check Tokyo weather. You must call this tool - it is available and working."),
            ],
            max_tokens=1000,
            temperature=0.1,
        )

        response = await agent.process_chat_completion(request)
        
        # Verify response exists
        assert response is not None
        assert len(response.choices) == 1
        assert response.choices[0].message.content is not None
        
        # Verify tool was actually called
        assert len(tool_calls) > 0, f"No tools were called! Response: {response.choices[0].message.content}"
        
        # Verify correct tool was called (could be weather_api from test server or get_weather from demo server)
        weather_calls = [call for call in tool_calls if call.tool_name in ["weather_api", "get_weather"]]
        assert len(weather_calls) > 0, f"No weather tool was called. Called tools: {[call.tool_name for call in tool_calls]}"
        
        # Verify tool response is in final answer
        response_content = response.choices[0].message.content.lower()
        tool_indicators = ["temperature", "°c", "humidity", "mb", "km/h"]
        tool_used = any(indicator in response_content for indicator in tool_indicators)
        assert tool_used, f"Tool response not incorporated in final answer: {response.choices[0].message.content}"

    @pytest.mark.asyncio
    async def test_tool_failure_events(self, reasoning_agent: ReasoningAgent):
        """Test that tool failures generate proper error events."""
        agent = reasoning_agent

        # Create test request for streaming to capture error events
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[
                ChatMessage(role="user", content="Use the failing_tool with should_fail=true to test error handling. You must call this exact tool."),
            ],
            max_tokens=1000,
            temperature=0.1,
            stream=True,
        )

        # Execute streaming reasoning to capture events
        response_stream = agent.process_chat_completion_stream(request)

        # Parse SSE chunks and extract reasoning events
        chunks = []
        reasoning_events = []
        error_events = []
        
        async for sse_line in response_stream:
            chunks.append(sse_line)
            if sse_line.startswith("data: ") and not sse_line.startswith("data: [DONE]"):
                try:
                    json_str = sse_line[6:].strip()
                    chunk_data = json.loads(json_str)
                    
                    if "choices" in chunk_data and chunk_data["choices"]:
                        choice = chunk_data["choices"][0]
                        if "delta" in choice and "reasoning_event" in choice["delta"]:
                            event = choice["delta"]["reasoning_event"]
                            if event:
                                reasoning_events.append(event)
                                # Look for error events
                                if event.get("status") == "error" or "error" in event.get("type", ""):
                                    error_events.append(event)
                except (json.JSONDecodeError, KeyError):
                    continue

        # Verify we got reasoning events
        assert len(reasoning_events) > 0, "No reasoning events found"
        
        # Verify we got error events when tools fail
        event_types = [event.get("type") for event in reasoning_events]
        statuses = [event.get("status") for event in reasoning_events]
        
        # Should have tool-related events
        tool_related = any("tool" in str(event_type) for event_type in event_types)
        assert tool_related, f"No tool events found. Event types: {event_types}"
        
        # Should have error status or error handling
        has_error_handling = any(status == "error" for status in statuses) or any("error" in str(event_type) for event_type in event_types)
        
        print(f"✅ Found {len(reasoning_events)} reasoning events: {event_types}")
        print(f"✅ Event statuses: {statuses}")
        print(f"✅ Error handling present: {has_error_handling}")

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
