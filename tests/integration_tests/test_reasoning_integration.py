"""
End-to-end integration tests for the reasoning agent.

This module tests:
A) ReasoningAgent end-to-end with fake tools + real OpenAI API calls
B) ReasoningAgent end-to-end with real in-memory MCP servers + real OpenAI API calls

These tests verify the complete flow from chat completion request through
reasoning, tool execution, and response generation.

IMPORTANT - pytest-asyncio Event Loop Scope:
==============================================
Async fixtures that create AsyncOpenAI clients or other resources requiring cleanup
MUST use @pytest_asyncio.fixture(loop_scope="function") decorator.

WHY: By default, pytest-asyncio uses different event loops for fixtures (session-scoped)
and tests (function-scoped). This causes "Event loop is closed" errors during fixture
cleanup because the test's event loop closes before the fixture's cleanup runs.

SOLUTION: Add loop_scope="function" to ensure fixture uses same event loop as the test.

INCORRECT (will leak resources or raise errors):
    @pytest_asyncio.fixture
    async def my_client(self):
        openai_client = AsyncOpenAI(...)  # Created in session loop
        yield openai_client
        await openai_client.close()  # FAILS: function loop already closed

CORRECT (proper cleanup):
    @pytest_asyncio.fixture(loop_scope="function")
    async def my_client(self):
        async with AsyncOpenAI(...) as openai_client:  # Uses function loop
            yield openai_client
            # Automatic cleanup happens in same event loop

Always use async context managers (async with) when available for automatic cleanup.
"""

import json
import os
from pathlib import Path
import time
import pytest
import pytest_asyncio
import httpx
from unittest.mock import AsyncMock, patch
from dotenv import load_dotenv

from api.reasoning_agent import ReasoningAgent
from api.openai_protocol import OpenAIChatRequest
from api.prompt_manager import PromptManager
from api.reasoning_models import ReasoningAction, ReasoningEventType
from api.tools import Tool, ToolResult, function_to_tool
from api.mcp import create_mcp_client, to_tools
from tests.conftest import OPENAI_TEST_MODEL, ReasoningAgentStreamingCollector
from fastmcp import FastMCP, Client
from fastapi.testclient import TestClient
from api.main import app
from api.dependencies import ServiceContainer, get_prompt_manager, get_reasoning_agent

load_dotenv()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable required for integration tests",
)
class TestReasoningAgentEndToEndWithFakeTools:
    """Test ReasoningAgent end-to-end with fake tools + real OpenAI API."""

    @pytest_asyncio.fixture
    async def fake_tools(self):
        """Create fake tools for testing."""
        def get_weather(location: str) -> dict:
            """Get current weather for a location."""
            return {
                "location": location,
                "temperature": "22°C",
                "condition": "Sunny",
                "humidity": "60%",
            }

        def search_web(query: str, limit: int = 5) -> list:
            """Search the web for information."""
            return [
                {"title": f"Result {i} for {query}", "url": f"https://example.com/{i}"}
                for i in range(1, limit + 1)
            ]

        def analyze_sentiment(text: str) -> dict:
            """Analyze sentiment of text."""
            return {
                "text": text,
                "sentiment": "positive" if "good" in text.lower() else "neutral",
                "confidence": 0.85,
            }

        return [
            function_to_tool(get_weather, description="Get weather for any location"),
            function_to_tool(search_web, description="Search web with results"),
            function_to_tool(analyze_sentiment, description="Analyze text sentiment"),
        ]

    @pytest_asyncio.fixture(loop_scope="function")
    async def reasoning_agent_with_fake_tools(self, fake_tools: list[Tool]):
        """Create ReasoningAgent with fake tools via LiteLLM proxy."""
        # Create and initialize prompt manager
        prompt_manager = PromptManager()
        await prompt_manager.initialize()

        agent = ReasoningAgent(
            tools=fake_tools,
            prompt_manager=prompt_manager,
        )

        yield agent

    @pytest.mark.asyncio
    async def test_end_to_end_with_fake_weather_tool(self, reasoning_agent_with_fake_tools:ReasoningAgent):  # noqa: E501
        """Test complete reasoning flow with fake weather tool + real OpenAI."""
        agent = reasoning_agent_with_fake_tools

        request = OpenAIChatRequest(
            model=OPENAI_TEST_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather like in Tokyo? Use the get_weather tool.",
                },
            ],
            max_tokens=500,
            temperature=0.1,
            stream=True,
        )

        # Collect streaming response
        collector = ReasoningAgentStreamingCollector()
        await collector.process(agent.execute_stream(request))

        # Verify streaming response received
        assert len(collector.all_chunks) > 0
        content = collector.content.lower()

        # Should contain actual tool results from our fake weather tool
        assert "tokyo" in content
        assert '22°c' in content

        # Should not contain failure messages
        assert not any(failure in content for failure in ["failed", "error", "unavailable"])

    @pytest.mark.asyncio
    async def test_end_to_end_with_fake_search_tool(self, reasoning_agent_with_fake_tools: ReasoningAgent):  # noqa: E501
        """Test complete reasoning flow with fake search tool + real OpenAI."""
        agent = reasoning_agent_with_fake_tools

        request = OpenAIChatRequest(
            model=OPENAI_TEST_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Search for information about Python programming. "
                        "Use the search_web tool."
                    ),
                },
            ],
            max_tokens=500,
            temperature=0.1,
            stream=True,  # Use streaming to capture tool events
        )

        # Collect streaming response and events
        collector = ReasoningAgentStreamingCollector()
        await collector.process(agent.execute_stream(request))

        # Verify streaming response received
        assert len(collector.all_chunks) > 0
        assert len(collector.reasoning_events) > 0

        # Tool execution events are already categorized by StreamingCollector
        tool_start_events = collector.tool_start_events
        tool_complete_events = collector.tool_complete_events

        # Verify tool was actually called
        assert len(tool_start_events) > 0, "Tool was not started"
        assert len(tool_complete_events) > 0, "Tool execution did not complete"

        # Verify the correct tool was called with correct arguments
        start_event = tool_start_events[0]
        assert "metadata" in start_event
        assert "tool_predictions" in start_event["metadata"]

        tool_predictions = start_event["metadata"]["tool_predictions"]
        assert len(tool_predictions) > 0

        # Check that search_web tool was called with query about Python
        prediction = tool_predictions[0]
        assert prediction["tool_name"] == "search_web"
        args = prediction["arguments"]
        assert "query" in args
        assert "python" in args["query"].lower()

        # Verify tool results contain our fake search results
        complete_event = tool_complete_events[0]
        assert "metadata" in complete_event
        assert "tool_results" in complete_event["metadata"]

        tool_results = complete_event["metadata"]["tool_results"]
        assert len(tool_results) > 0

        # Check that we got the expected fake search results
        result = tool_results[0]
        tool_result_data = result["result"]

        # Our fake search tool returns a list of results with title and url
        assert isinstance(tool_result_data, list)
        assert len(tool_result_data) > 0

        # Verify the structure matches our fake tool output
        first_result = tool_result_data[0]
        assert "title" in first_result
        assert "url" in first_result
        assert "example.com" in first_result["url"]
        assert "Result 1" in first_result["title"]

        # Final content should reference the search results
        final_content = collector.content.lower()
        assert "python" in final_content

    @pytest.mark.asyncio
    async def test_streaming_with_fake_tools(self, reasoning_agent_with_fake_tools: ReasoningAgent):  # noqa: E501
        """Test streaming reasoning with fake tools + real OpenAI."""
        agent = reasoning_agent_with_fake_tools

        request = OpenAIChatRequest(
            model=OPENAI_TEST_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Analyze the sentiment of 'This is a good day!' "
                        "using the analyze_sentiment tool."
                    ),
                },
            ],
            max_tokens=500,
            temperature=0.1,
            stream=True,
        )

        # Collect streaming response
        collector = ReasoningAgentStreamingCollector()
        await collector.process(agent.execute_stream(request))

        # Verify streaming response
        assert len(collector.all_chunks) > 0
        assert len(collector.reasoning_events) > 0

        # Should have tool-related events
        tool_events = [e for e in collector.reasoning_events if "tool" in e.get("type", "")]
        assert len(tool_events) > 0

        # Final content should contain sentiment analysis results
        final_content = collector.content.lower()
        assert any(indicator in final_content for indicator in ["sentiment", "positive", "good"])


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable required for integration tests",
)
class TestReasoningAgentEndToEndWithInMemoryMCP:
    """Test ReasoningAgent end-to-end with in-memory MCP servers + real OpenAI API."""

    @pytest_asyncio.fixture(loop_scope="function")
    async def reasoning_agent_with_real_mcp_loading(self, in_memory_mcp_server, tmp_path: Path):  # noqa: ANN001
        """Fixture: Create ReasoningAgent with tools loaded from real MCP server."""
        # Create MCP config file
        config = {
            "mcpServers": {
                "test_server": {
                    "command": "test",
                    "args": [],
                    "env": {},
                },
            },
        }

        config_file = tmp_path / "test_mcp_config.json"
        config_file.write_text(json.dumps(config))

        # Create MCP client with in-memory server
        # For testing, we'll patch the Client creation to use our in-memory server

        def create_test_client(config):  # noqa: ANN001, ANN202, ARG001
            # Return a client connected to our in-memory server
            return Client(in_memory_mcp_server)

        with patch('api.mcp.Client', side_effect=create_test_client):
            # Create MCP client as the API would
            mcp_client = create_mcp_client(config_file)

            # Load tools from MCP server (don't keep context open)
            async with mcp_client as client:
                tools = await to_tools(client)

            # Create ReasoningAgent with MCP-loaded tools
            prompt_manager = PromptManager()
            await prompt_manager.initialize()

            agent = ReasoningAgent(
                tools=tools,
                prompt_manager=prompt_manager,
            )

            yield agent

    @pytest_asyncio.fixture
    async def in_memory_mcp_server(self):
        """Create an in-memory MCP server with test tools."""
        # Create a FastMCP server
        server = FastMCP("test_server")

        # Register tools on the server
        @server.tool
        def weather_api(location: str) -> dict:
            """Get weather information for a location."""
            return {
                "location": location,
                "temperature": "25°C",
                "condition": "Partly cloudy",
                "humidity": "65%",
                "wind_speed": "10 km/h",
                "source": "mcp_server",
            }

        @server.tool
        def calculator(operation: str, a: float, b: float) -> dict:
            """Perform calculations."""
            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                result = a / b if b != 0 else "Error: Division by zero"
            else:
                result = "Error: Unknown operation"

            return {
                "operation": operation,
                "a": a,
                "b": b,
                "result": result,
                "source": "mcp_server",
            }

        @server.tool
        def search_database(query: str, limit: int = 5) -> dict:
            """Search a mock database."""
            results = [
                {"id": i, "title": f"Result {i}: {query}", "score": 0.9 - i * 0.1}
                for i in range(1, min(limit + 1, 6))
            ]
            return {
                "query": query,
                "total_results": len(results),
                "results": results,
                "source": "mcp_server",
            }

        return server

    @pytest_asyncio.fixture
    async def mcp_config_file(self, in_memory_mcp_server, tmp_path):  # noqa: ANN001, ARG002
        """Create a temporary MCP config file for testing."""
        # Note: For in-memory testing, we'll use a special config that
        # the test will override with the in-memory server
        config = {
            "mcpServers": {
                "test_server": {
                    "command": "test",  # Placeholder - we'll use in-memory server
                    "args": [],
                    "env": {},
                },
            },
        }

        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config))
        return config_file

    @pytest_asyncio.fixture
    async def mcp_tools_from_server(self, in_memory_mcp_server):  # noqa: ANN001
        """Get tools from the in-memory MCP server."""
        # Create a client connected to the in-memory server
        async with Client(in_memory_mcp_server) as client:
            # Convert MCP tools to our Tool objects
            return await to_tools(client)

    @pytest_asyncio.fixture(loop_scope="function")
    async def reasoning_agent_with_mcp_tools(self, mcp_tools_from_server: list[Tool]):
        """Create ReasoningAgent with tools loaded from MCP server via LiteLLM proxy."""
        prompt_manager = PromptManager()
        await prompt_manager.initialize()

        agent = ReasoningAgent(
            tools=mcp_tools_from_server,  # Tools loaded from actual MCP server
            prompt_manager=prompt_manager,
        )

        yield agent

    @pytest.mark.asyncio
    async def test_end_to_end_with_mcp_weather_tool(self, reasoning_agent_with_mcp_tools):  # noqa: ANN001
        """Test complete reasoning flow with MCP weather tool + real OpenAI."""
        agent = reasoning_agent_with_mcp_tools

        # Debug: Check if tools are loaded
        # Test calling the tool directly
        weather_tool = agent.tools["weather_api"]
        direct_result = await weather_tool(location="TestLocation")
        assert direct_result.result["location"] == "TestLocation"
        assert direct_result.result["temperature"] == "25°C"
        assert direct_result.result["condition"] == "Partly cloudy"
        assert direct_result.result["humidity"] == "65%"

        used_tools = False
        executed_tools = False
        # Add debug to see reasoning steps
        original_generate_step = agent._generate_reasoning_step
        async def debug_generate_step(request, context, system_prompt):  # noqa: ANN001, ANN202
            step_result = await original_generate_step(request, context, system_prompt)
            nonlocal used_tools
            assert step_result is not None
            # _generate_reasoning_step returns a tuple (ReasoningStep, OpenAIUsage)
            step, usage = step_result
            assert step.thought is not None
            if step.next_action == ReasoningAction.USE_TOOLS:
                used_tools = True
                assert 'weather_api' in [t.tool_name for t in step.tools_to_use]
            return step_result  # Return the original tuple
        agent._generate_reasoning_step = debug_generate_step

        original_execute_tools = agent._execute_tools_sequentially
        async def debug_execute_tools(tool_predictions):  # noqa: ANN001, ANN202
            results = await original_execute_tools(tool_predictions)
            nonlocal executed_tools
            executed_tools = True
            assert len(results) == 1
            assert results[0].tool_name == "weather_api"
            assert results[0].success is True
            assert 'Paris' in results[0].result["location"]
            assert results[0].result["temperature"] == "25°C"
            return results
        agent._execute_tools_sequentially = debug_execute_tools

        request = OpenAIChatRequest(
            model=OPENAI_TEST_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Get the weather for Paris using the weather_api tool.",
                },
            ],
            max_tokens=500,
            temperature=0.1,
            stream=True,
        )

        # Collect streaming response
        collector = ReasoningAgentStreamingCollector()
        await collector.process(agent.execute_stream(request))

        assert used_tools, "No reasoning step generated"
        assert executed_tools, "No tools executed"
        content = collector.content.lower()
        # Should contain MCP tool results
        assert "paris" in content
        assert any(indicator in content for indicator in ["25°c", "partly cloudy"])
        # Should not contain failure messages
        assert not any(failure in content for failure in ["failed", "error", "unavailable"])

    @pytest.mark.asyncio
    async def test_tool_execution_verification_with_mcp(self, reasoning_agent_with_mcp_tools):  # noqa: ANN001
        """Verify that MCP tools are actually executed during reasoning."""
        agent = reasoning_agent_with_mcp_tools

        request = OpenAIChatRequest(
            model=OPENAI_TEST_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Use the weather_api to get weather for London.",
                },
            ],
            max_tokens=500,
            temperature=0.1,
            stream=True,
        )

        # Collect streaming response
        collector = ReasoningAgentStreamingCollector()
        await collector.process(agent.execute_stream(request))

        # Verify response contains tool results
        content = collector.content.lower()
        assert "london" in content
        assert any(indicator in content for indicator in ["temperature", "weather", "condition"])

        # Verify tool was called by checking for specific values that only come from tool execution
        # The response contains specific values that match our MCP tool's return values
        assert any(indicator in content for indicator in ["25°c", "partly cloudy", "humidity", "wind_speed"])  # noqa: E501

        # The presence of all these specific values proves the tool was executed
        assert "25°c" in content
        assert "partly cloudy" in content
        assert "65%" in content
        assert "10 km/h" in content

    @pytest.mark.asyncio
    async def test_full_api_flow_with_mcp_loading(self, reasoning_agent_with_real_mcp_loading):  # noqa: ANN001
        """Test the complete API flow: MCP server -> load tools -> execute request."""
        agent = reasoning_agent_with_real_mcp_loading

        # Verify tools were loaded from MCP server
        assert len(agent.tools) > 0
        assert "weather_api" in agent.tools
        assert "calculator" in agent.tools
        assert "search_database" in agent.tools

        # Test with weather tool
        request = OpenAIChatRequest(
            model=OPENAI_TEST_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in Berlin? Use the weather_api tool.",
                },
            ],
            max_tokens=500,
            temperature=0.1,
            stream=True,
        )

        # Collect streaming response
        collector = ReasoningAgentStreamingCollector()
        await collector.process(agent.execute_stream(request))

        content = collector.content.lower()

        # Should contain MCP tool results - verify specific values that only come from our MCP tool
        assert "berlin" in content
        assert any(indicator in content for indicator in ["25°c", "partly cloudy", "weather"])
        # Verify the exact values that come from our MCP tool (proves it was executed)
        assert "25°c" in content
        assert "partly cloudy" in content
        assert "65%" in content
        assert "10 km/h" in content

    @pytest.mark.asyncio
    async def test_streaming_with_mcp_loaded_tools(self, reasoning_agent_with_real_mcp_loading):  # noqa: ANN001
        """Test streaming with tools loaded from MCP server."""
        agent = reasoning_agent_with_real_mcp_loading

        request = OpenAIChatRequest(
            model=OPENAI_TEST_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Search the database for 'python tutorials' using search_database.",
                },
            ],
            max_tokens=500,
            temperature=0.1,
            stream=True,
        )

        # Collect streaming response
        collector = ReasoningAgentStreamingCollector()
        await collector.process(agent.execute_stream(request))

        # Verify streaming worked with MCP tools
        tool_events = [e for e in collector.reasoning_events if "tool" in e.get("type", "")]
        assert len(tool_events) > 0

        final_content = collector.content.lower()
        assert "python tutorials" in final_content
        assert "result" in final_content
        assert any(indicator in final_content for indicator in ["search", "database", "mcp"])


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable required for integration tests",
)
class TestAPIWithMCPServerIntegration:
    """Test the complete API integration with MCP servers."""

    @pytest_asyncio.fixture
    async def mcp_server_for_api(self):
        """Create an MCP server for API integration testing."""
        server = FastMCP("api_test_server")

        @server.tool
        def get_system_info() -> dict:
            """Get system information."""
            return {
                "platform": "test_platform",
                "version": "1.0.0",
                "status": "healthy",
                "timestamp": "2024-01-01T00:00:00Z",
            }

        @server.tool
        def process_text(text: str, mode: str = "uppercase") -> dict:
            """Process text in various ways."""
            if mode == "uppercase":
                result = text.upper()
            elif mode == "lowercase":
                result = text.lower()
            elif mode == "reverse":
                result = text[::-1]
            else:
                result = text

            return {
                "original": text,
                "processed": result,
                "mode": mode,
                "length": len(result),
            }

        return server

    @pytest.mark.asyncio
    @pytest.mark.skipif(os.getenv("SKIP_CI_TESTS") == "true", reason="Skipped in CI")
    async def test_api_loads_tools_from_mcp_server(self, mcp_server_for_api, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN001, E501
        """Test that the API correctly loads tools from an MCP server."""
        # Create MCP config
        config = {
            "mcpServers": {
                "api_test_server": {
                    "command": "test",
                    "args": [],
                    "env": {},
                },
            },
        }

        config_file = tmp_path / "api_mcp_config.json"
        config_file.write_text(json.dumps(config))

        # Set the MCP_CONFIG_PATH environment variable
        monkeypatch.setenv("MCP_CONFIG_PATH", str(config_file))

        # Patch Client creation to use our in-memory server
        def create_test_client(config):  # noqa: ANN001, ANN202, ARG001
            return Client(mcp_server_for_api)

        with patch('api.mcp.Client', create_test_client):
            # Create fresh settings instance that will read the updated environment
            from api.config import Settings  # noqa: PLC0415
            test_settings = Settings(_env_file=None)  # Force reading from environment only

            # Patch the settings in dependencies module
            with patch('api.dependencies.settings', test_settings):
                # Create a test service container
                test_container = ServiceContainer()
                await test_container.initialize()

                # Override the service container in the app
                from api import dependencies  # noqa: PLC0415
                original_container = dependencies.service_container
                dependencies.service_container = test_container

                try:
                    with TestClient(app) as client:
                        # Test the /tools endpoint
                        tools_response = client.get("/tools")
                        assert tools_response.status_code == 200

                        tools_data = tools_response.json()
                        assert "tools" in tools_data
                        assert len(tools_data["tools"]) == 2  # get_system_info and process_text

                        assert "get_system_info" in tools_data["tools"]
                        assert "process_text" in tools_data["tools"]

                        # Test chat completion with MCP tool (streaming-only)
                        chat_response = client.post(
                            "/v1/chat/completions",
                            json={
                                "model": OPENAI_TEST_MODEL,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": "Get the system information using the get_system_info tool.",  # noqa: E501
                                    },
                                ],
                                "max_tokens": 500,
                                "temperature": 0.1,
                                "stream": True,
                            },
                            headers={"X-Routing-Mode": "reasoning"},  # Route to reasoning path
                        )

                        assert chat_response.status_code == 200

                        # Collect streaming response
                        collector = ReasoningAgentStreamingCollector()
                        for line in chat_response.iter_lines():
                            collector.process_line(line)

                        content = collector.content.lower()

                        # Should contain system info from MCP tool
                        assert any(indicator in content for indicator in ["test_platform", "healthy", "1.0.0"])  # noqa: E501

                finally:
                    # Restore original container
                    dependencies.service_container = original_container
                    try:
                        await test_container.cleanup()
                    except RuntimeError as e:
                        if "Event loop is closed" in str(e):
                            # Ignore event loop closed errors during test cleanup
                            pass
                        else:
                            raise

    @pytest.mark.asyncio
    @pytest.mark.skipif(os.getenv("SKIP_CI_TESTS") == "true", reason="Skipped in CI")
    async def test_api_streaming_with_mcp_tools(
        self,
        mcp_server_for_api: FastMCP,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test streaming responses with MCP tools."""
        # Create MCP config
        config = {
            "mcpServers": {
                "api_test_server": {
                    "command": "test",
                    "args": [],
                    "env": {},
                },
            },
        }

        config_file = tmp_path / "api_mcp_config.json"
        config_file.write_text(json.dumps(config))
        monkeypatch.setenv("MCP_CONFIG_PATH", str(config_file))

        def create_test_client(config):  # noqa: ANN001, ANN202, ARG001
            return Client(mcp_server_for_api)

        with patch('api.mcp.Client', create_test_client):
            # Use async context manager to ensure proper cleanup
            async with ServiceContainer() as test_container:
                from api import dependencies  # noqa: PLC0415
                original_container = dependencies.service_container
                dependencies.service_container = test_container

                try:
                    with TestClient(app) as client:
                        # Test streaming with MCP tool
                        streaming_response = client.post(
                            "/v1/chat/completions",
                            json={
                                "model": OPENAI_TEST_MODEL,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": "Process the text 'hello world' in uppercase using the process_text tool.",  # noqa: E501
                                    },
                                ],
                                "max_tokens": 500,
                                "temperature": 0.1,
                                "stream": True,
                            },
                        )

                        assert streaming_response.status_code == 200

                        # Collect streaming content
                        collector = ReasoningAgentStreamingCollector()
                        for line in streaming_response.iter_lines():
                            collector.process_line(line)

                        final_content = collector.content.lower()

                        # Should contain processed text result
                        assert "hello world" in final_content
                        assert "uppercase" in final_content
                        # The tool should have returned "HELLO WORLD"
                        assert any(indicator in final_content for indicator in ["hello world".upper().lower(), "processed"])  # noqa: E501

                finally:
                    dependencies.service_container = original_container
                    # ServiceContainer cleanup handled by async context manager


@pytest.mark.integration
class TestToolErrorHandling:
    """Test error handling in tool execution."""

    @pytest_asyncio.fixture
    async def error_prone_tools(self):
        """Create tools that can fail for testing error handling."""
        def failing_tool(should_fail: bool = False) -> dict:
            """Tool that fails when asked to."""
            if should_fail:
                raise ValueError("Tool deliberately failed for testing")
            return {"status": "success", "message": "Tool executed successfully"}

        def slow_tool(delay: float = 0.1) -> dict:
            """Tool that simulates slow execution."""
            time.sleep(delay)
            return {"status": "completed", "execution_time": delay}

        return [
            function_to_tool(failing_tool, description="Tool that can fail on command"),
            function_to_tool(slow_tool, description="Tool with configurable delay"),
        ]

    @pytest_asyncio.fixture
    async def reasoning_agent_with_error_tools(self, error_prone_tools):  # noqa: ANN001
        """Create ReasoningAgent with error-prone tools."""
        mock_prompt_manager = AsyncMock()
        mock_prompt_manager.get_prompt.return_value = "You are a helpful assistant."

        return ReasoningAgent(
            tools=error_prone_tools,
            prompt_manager=mock_prompt_manager,
        )

    @pytest.mark.asyncio
    async def test_tool_failure_handling(self, reasoning_agent_with_error_tools):  # noqa: ANN001
        """Test that tool failures are handled gracefully."""
        agent = reasoning_agent_with_error_tools

        # Test direct tool execution with failure
        failing_tool = agent.tools["failing_tool"]
        result = await failing_tool(should_fail=True)

        # Tool should return failure result, not raise exception
        assert result.success is False
        assert "failed" in result.error.lower()
        assert result.tool_name == "failing_tool"

    @pytest.mark.asyncio
    async def test_tool_success_result_structure(self, reasoning_agent_with_error_tools):  # noqa: ANN001
        """Test that successful tool execution returns proper structure."""
        agent = reasoning_agent_with_error_tools

        # Test successful tool execution
        slow_tool = agent.tools["slow_tool"]
        result = await slow_tool(delay=0.01)

        assert result.success is True
        assert result.error is None
        assert result.result["status"] == "completed"
        assert result.execution_time_ms > 0


@pytest.mark.integration
class TestStreamingToolResultsBugFix:
    """
    Tests for the streaming tool results bug fix.

    This module specifically tests that tool results are properly included
    in both streaming and non-streaming reasoning responses using the new
    Tool abstraction.

    Background: There was a bug where streaming responses would report tool
    failures even when tools executed successfully, because the final response
    generation wasn't receiving tool results context.
    """

    @pytest.fixture
    def mock_reasoning_agent(self):
        """Create a mock reasoning agent for testing."""
        httpx.AsyncClient()

        # Create fake tools for testing
        def test_weather(location: str) -> dict:
            """Get weather for testing."""
            return {
                "location": location,
                "temperature": "22°C",
                "condition": "Sunny",
                "humidity": "60%",
            }

        tools = [function_to_tool(test_weather, description="Get weather information")]
        mock_prompt_manager = AsyncMock()

        return ReasoningAgent(
            tools=tools,
            prompt_manager=mock_prompt_manager,
        )

    @pytest.fixture
    def sample_request(self):
        """Sample chat completion request."""
        return OpenAIChatRequest(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "What's the weather in Tokyo?"},
            ],
        )

    @pytest.fixture
    def sample_tool_result(self):
        """Sample successful tool result using new ToolResult model."""
        return ToolResult(
            tool_name="test_weather",
            success=True,
            result={
                "location": "Tokyo",
                "temperature": "22°C",
                "condition": "Sunny",
            },
            execution_time_ms=150.0,
        )

    @pytest.mark.asyncio
    async def test_stream_final_response_includes_tool_results(
        self,
        mock_reasoning_agent: ReasoningAgent,
        sample_request: OpenAIChatRequest,
        sample_tool_result: ToolResult,
    ):
        """
        Test that _stream_final_response includes tool results by verifying message
        structure.
        """
        # Setup mocks
        mock_reasoning_agent.prompt_manager.get_prompt.return_value = "You are a helpful assistant."  # noqa: E501

        # Create reasoning context with tool results (the bug was that this wasn't passed to
        # streaming)
        reasoning_context = {
            "steps": [],
            "tool_results": [sample_tool_result],
            "final_thoughts": "",
            "user_request": sample_request,
        }

        # Test the internal message building logic that was part of the bug fix
        synthesis_prompt = await mock_reasoning_agent.prompt_manager.get_prompt("final_answer")

        # Build synthesis messages (this is what the bug fix added to streaming)
        messages = [
            {"role": "system", "content": synthesis_prompt},
            {
                "role": "user",
                "content": f"Original request: {sample_request.messages[-1]['content']}",
            },
        ]

        # Add reasoning summary if available (this was missing in streaming before fix)
        if reasoning_context:
            reasoning_summary = mock_reasoning_agent._build_reasoning_summary(reasoning_context)
            messages.append({
                "role": "assistant",
                "content": f"My reasoning process:\n{reasoning_summary}",
            })

        # Verify the fix: streaming now includes tool results like non-streaming
        assert len(messages) == 3  # system + user + assistant (reasoning summary)
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

        # Verify that tool results are included in the reasoning summary
        reasoning_message = messages[2]["content"]
        assert "Tool Results:" in reasoning_message
        assert "test_weather" in reasoning_message
        assert "Tokyo" in reasoning_message
        assert "22°C" in reasoning_message

    @pytest.mark.asyncio
    async def test_build_reasoning_summary_with_multiple_tool_results(
        self, mock_reasoning_agent: ReasoningAgent,
    ):
        """Test that _build_reasoning_summary handles multiple tool results."""
        tool_results = [
            ToolResult(
                tool_name="test_weather",
                success=True,
                result={"location": "Tokyo", "temp": "22°C"},
                execution_time_ms=100.0,
            ),
            ToolResult(
                tool_name="test_search",
                success=True,
                result={"query": "weather", "results": ["result1"]},
                execution_time_ms=200.0,
            ),
        ]

        reasoning_context = {
            "steps": [],
            "tool_results": tool_results,
            "final_thoughts": "",
        }

        summary = mock_reasoning_agent._build_reasoning_summary(reasoning_context)

        # Verify both tool results are included
        assert "test_weather" in summary
        assert "test_search" in summary
        assert "Tokyo" in summary
        assert "weather" in summary

    @pytest.mark.asyncio
    async def test_tool_execution_with_new_abstraction(self, mock_reasoning_agent: ReasoningAgent):
        """Test that tools work correctly with new Tool abstraction."""
        # Get the test tool
        test_weather_tool = mock_reasoning_agent.tools["test_weather"]

        # Execute the tool
        result = await test_weather_tool(location="Paris")

        # Verify the result structure
        assert result.success is True
        assert result.tool_name == "test_weather"
        assert result.result["location"] == "Paris"
        assert result.result["temperature"] == "22°C"
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_tool_error_handling_with_new_abstraction(
        self,
        mock_reasoning_agent: ReasoningAgent,  # noqa: ARG002
    ):
        """Test that tool errors are handled properly with new abstraction."""
        # Create a failing tool
        def failing_tool(should_fail: bool = True) -> dict:
            if should_fail:
                raise ValueError("Tool failed for testing")
            return {"status": "success"}

        fail_tool = function_to_tool(failing_tool, description="Tool that can fail")

        # Execute the failing tool
        result = await fail_tool(should_fail=True)

        # Verify error is handled properly
        assert result.success is False
        assert "failed" in result.error.lower()
        assert result.tool_name == "failing_tool"

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.skipif(os.getenv("SKIP_CI_TESTS") == "true", reason="Skipped in CI")
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    async def test_streaming_and_non_streaming_both_include_tool_data(self):
        """
        Integration test: Streaming responses should include actual tool data
        when tools execute successfully using the new Tool abstraction.

        Note: API is streaming-only now, non-streaming was removed.
        """
        # Create reasoning agent with fake tools for testing
        async def create_test_reasoning_agent() -> ReasoningAgent:
            def weather_api(location: str) -> dict:
                """Get weather for testing."""
                return {
                    "location": location,
                    "temperature": "24°C",
                    "condition": "Clear",
                    "humidity": "65%",
                }

            tools = [function_to_tool(weather_api, description="Get weather information")]

            # Create and initialize prompt manager
            prompt_manager = PromptManager()
            await prompt_manager.initialize()

            return ReasoningAgent(
                tools=tools,
                prompt_manager=prompt_manager,
            )

        # Create the test agent
        test_agent = await create_test_reasoning_agent()

        # Override dependencies to use our test setup
        app.dependency_overrides[get_reasoning_agent] = lambda: test_agent
        app.dependency_overrides[get_prompt_manager] = lambda: test_agent.prompt_manager

        try:
            with TestClient(app) as client:
                # Test streaming request (streaming-only architecture)
                streaming_response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": "What's the weather in Tokyo? Use the weather_api tool."}],  # noqa: E501
                        "stream": True,
                    },
                    headers={"X-Routing-Mode": "reasoning"},  # Route to reasoning path
                )

                assert streaming_response.status_code == 200

                # Parse streaming response
                streaming_content = ""
                for line in streaming_response.iter_lines():
                    if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                        try:
                            chunk_data = json.loads(line[6:])
                            if chunk_data.get("choices") and chunk_data["choices"][0].get("delta", {}).get("content"):  # noqa: E501
                                streaming_content += chunk_data["choices"][0]["delta"]["content"]
                        except json.JSONDecodeError:
                            continue

                # Should contain actual weather data (temperature, condition, etc.)
                weather_indicators = [
                    "temperature", "°c", "°f", "clear", "cloudy", "condition", "tokyo",
                ]
                failure_indicators = ["did not execute", "failed", "unavailable", "error"]

                # Check streaming
                has_weather_data_streaming = any(
                    indicator.lower() in streaming_content.lower()
                    for indicator in weather_indicators
                )
                has_failure_streaming = any(
                    indicator.lower() in streaming_content.lower()
                    for indicator in failure_indicators
                )

                # Should have weather data and no failure messages
                assert has_weather_data_streaming, (
                    f"Streaming missing weather data: {streaming_content[:200]}"
                )
                assert not has_failure_streaming, (
                    f"Streaming has failure message: {streaming_content[:200]}"
                )

        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()


    @pytest.mark.integration
    @pytest.mark.skipif(os.getenv("SKIP_CI_TESTS") == "true", reason="Skipped in CI")
    @pytest.mark.asyncio
    async def test_tool_arguments_and_results_in_streaming_events(self):
        """
        Integration test: Verify that tool arguments and results are properly
        included in streaming reasoning events.
        """
        # Skip if no OpenAI key (this is an integration test)
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        # Create reasoning agent with in-memory MCP server for testing
        async def create_test_reasoning_agent() -> ReasoningAgent:
            # Create MCP manager with in-memory server
            # Create in-memory MCP server using FastMCP

            server = FastMCP("test_server")

            @server.tool
            def weather_api(location: str) -> dict:
                """Get weather information for a location."""
                return {"location": location, "temperature": "25°C", "condition": "Partly cloudy"}

            # Convert to tools
            client = Client(server)
            tools = await to_tools(client)

            # Create and initialize prompt manager
            prompt_manager = PromptManager()
            await prompt_manager.initialize()

            return ReasoningAgent(
                tools=tools,
                prompt_manager=prompt_manager,
            )

        # Create the test agent
        test_agent = await create_test_reasoning_agent()

        # Override dependencies to use our test setup
        app.dependency_overrides[get_reasoning_agent] = lambda: test_agent
        # No MCP manager override needed with new tool architecture
        app.dependency_overrides[get_prompt_manager] = lambda: test_agent.prompt_manager

        try:
            with TestClient(app) as client:
                # Test streaming request to capture tool events
                streaming_response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": "Get the weather for Tokyo using the weather_api tool."}],  # noqa: E501
                        "stream": True,
                    },
                    headers={"X-Routing-Mode": "reasoning"},  # Route to reasoning path
                )

                assert streaming_response.status_code == 200

                # Parse streaming response and collect tool events
                tool_start_events = []
                tool_complete_events = []

                for line in streaming_response.iter_lines():
                    if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                        try:
                            chunk_data = json.loads(line[6:])
                            if chunk_data.get("choices") and chunk_data["choices"][0].get("delta", {}).get("reasoning_event"):  # noqa: E501
                                event = chunk_data["choices"][0]["delta"]["reasoning_event"]

                                if event.get("type") == ReasoningEventType.TOOL_EXECUTION_START.value:  # noqa: E501
                                    tool_start_events.append(event)
                                elif event.get("type") == ReasoningEventType.TOOL_RESULT.value:
                                    tool_complete_events.append(event)
                        except json.JSONDecodeError:
                            continue

                # Verify tool start events contain arguments
                assert len(tool_start_events) > 0, "No tool start events found"
                start_event = tool_start_events[0]

                assert "metadata" in start_event, "Tool start event missing metadata"
                assert "tool_predictions" in start_event["metadata"], "Tool start event missing tool_predictions in metadata"  # noqa: E501

                tool_predictions = start_event["metadata"]["tool_predictions"]
                assert len(tool_predictions) > 0, "No tool predictions in start event"

                # Check that tool predictions have arguments
                prediction = tool_predictions[0]
                assert "arguments" in prediction or hasattr(prediction, "arguments"), "Tool prediction missing arguments"  # noqa: E501

                # Verify tool complete events contain results
                assert len(tool_complete_events) > 0, "No tool complete events found"
                complete_event = tool_complete_events[0]

                assert "metadata" in complete_event, "Tool complete event missing metadata"
                assert "tool_results" in complete_event["metadata"], "Tool complete event missing tool_results in metadata"  # noqa: E501

                tool_results = complete_event["metadata"]["tool_results"]
                assert len(tool_results) > 0, "No tool results in complete event"

                # Check that tool results have actual result data
                result = tool_results[0]
                assert hasattr(result, "result") or "result" in result, "Tool result missing result data"  # noqa: E501
                assert hasattr(result, "tool_name") or "tool_name" in result, "Tool result missing tool_name"  # noqa: E501

        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
