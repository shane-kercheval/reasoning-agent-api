"""
Integration tests for MCP (Model Context Protocol) servers with reasoning agent.

These tests use real MCP servers to validate end-to-end tool execution
and integration with the reasoning agent workflow.
"""

import asyncio
import json
import os
import socket
import subprocess
import time

import pytest
import pytest_asyncio
import httpx
import respx

from api.models import ChatCompletionRequest, ChatMessage, MessageRole
from api.reasoning_agent import ReasoningAgent
from api.mcp_manager import MCPServerManager, MCPServerConfig
from api.prompt_manager import PromptManager
from api.reasoning_models import ToolResult, ToolRequest


class TestMCPServerLifecycle:
    """Test MCP server startup, connection, and cleanup."""

    @pytest.mark.asyncio
    async def test_mcp_server_startup_and_discovery(self):
        """Test that we can start an MCP server and discover its tools."""
        # Start real MCP server
        server_process = await self._start_mcp_server()
        try:
            # Wait for server to be ready
            await self._wait_for_server_ready("http://localhost:8001")

            # Create MCP manager with real server config
            config = MCPServerConfig(
                name="test_server",
                url="http://localhost:8001/mcp/",
                enabled=True,
            )
            manager = MCPServerManager([config])
            await manager.initialize()

            # Test tool discovery
            tools = await manager.get_available_tools()

            # Verify expected tools are discovered
            tool_names = [tool.tool_name for tool in tools]
            assert "web_search" in tool_names
            assert "weather_api" in tool_names
            assert "filesystem" in tool_names
            assert "search_news" in tool_names

            # Verify tool metadata
            web_search_tool = next(tool for tool in tools if tool.tool_name == "web_search")
            assert web_search_tool.description is not None
            assert web_search_tool.input_schema is not None
            assert web_search_tool.server_name == "test_server"

        finally:
            server_process.terminate()
            await asyncio.sleep(0.5)  # Give server time to shutdown

    async def _start_mcp_server(self) -> subprocess.Popen:
        """Start the MCP server process."""
        # Start MCP server
        return subprocess.Popen([  # noqa: ASYNC220
            "uv", "run", "python", "mcp_server/server.py",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    async def _wait_for_server_ready(self, url: str, timeout: int = 10) -> None:  # noqa: ASYNC109
        """Wait for server to be ready to accept connections."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if server process is listening on the port
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', 8001))
                sock.close()
                if result == 0:  # Port is open
                    return
            except Exception:
                pass
            await asyncio.sleep(0.1)
        raise TimeoutError(f"Server at {url} did not become ready within {timeout} seconds")


class TestRealToolExecution:
    """Test actual tool execution through real MCP servers."""

    @pytest_asyncio.fixture
    async def mcp_server_and_manager(self):
        """Fixture that starts MCP server and creates manager."""
        # Start real MCP server
        server_process = subprocess.Popen([  # noqa: ASYNC220
            "uv", "run", "python", "mcp_server/server.py",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        try:
            # Wait for server to be ready
            await self._wait_for_server_ready("http://localhost:8001")

            # Create MCP manager
            config = MCPServerConfig(
                name="test_server",
                url="http://localhost:8001/mcp/",
                enabled=True,
            )
            manager = MCPServerManager([config])
            await manager.initialize()

            yield manager

        finally:
            await manager.cleanup()
            server_process.terminate()
            await asyncio.sleep(0.5)

    async def _wait_for_server_ready(self, url: str, timeout: int = 10) -> None:  # noqa: ASYNC109
        """Wait for server to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if server process is listening on the port
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', 8001))
                sock.close()
                if result == 0:  # Port is open
                    return
            except Exception:
                pass
            await asyncio.sleep(0.1)
        raise TimeoutError(f"Server at {url} did not become ready within {timeout} seconds")

    @pytest.mark.asyncio
    async def test_weather_tool_execution(self, mcp_server_and_manager: MCPServerManager):
        """Test executing weather tool and getting real results."""
        manager = mcp_server_and_manager


        # Execute weather tool
        tool_request = ToolRequest(
            server_name="test_server",
            tool_name="weather_api",
            arguments={"location": "Tokyo"},
            reasoning="Need weather data for Tokyo",
        )

        result = await manager.execute_tool(tool_request)

        # Verify successful execution
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.server_name == "test_server"
        assert result.tool_name == "weather_api"
        assert result.error is None

        # Verify result structure
        weather_data = result.result
        assert isinstance(weather_data, dict)
        assert "location" in weather_data
        assert "current" in weather_data
        assert weather_data["location"] == "Tokyo"
        assert "temperature" in weather_data["current"]
        assert "condition" in weather_data["current"]

    @pytest.mark.asyncio
    async def test_web_search_tool_execution(self, mcp_server_and_manager):  # noqa: ANN001
        """Test executing web search tool and getting real results."""
        manager = mcp_server_and_manager


        # Execute web search tool
        tool_request = ToolRequest(
            server_name="test_server",
            tool_name="web_search",
            arguments={"query": "artificial intelligence"},
            reasoning="Need to search for AI information",
        )

        result = await manager.execute_tool(tool_request)

        # Verify successful execution
        assert result.success is True
        assert result.tool_name == "web_search"

        # Verify result structure
        search_data = result.result
        assert isinstance(search_data, dict)
        assert "query" in search_data
        assert "results" in search_data
        assert search_data["query"] == "artificial intelligence"
        assert isinstance(search_data["results"], list)
        assert len(search_data["results"]) > 0

        # Verify search result structure
        first_result = search_data["results"][0]
        assert "title" in first_result
        assert "url" in first_result
        assert "snippet" in first_result

    @pytest.mark.asyncio
    async def test_parallel_tool_execution(self, mcp_server_and_manager):  # noqa: ANN001
        """Test executing multiple tools in parallel."""
        manager = mcp_server_and_manager


        # Create multiple tool requests
        tool_requests = [
            ToolRequest(
                server_name="test_server",
                tool_name="weather_api",
                arguments={"location": "London"},
                reasoning="Need London weather",
            ),
            ToolRequest(
                server_name="test_server",
                tool_name="web_search",
                arguments={"query": "London travel"},
                reasoning="Need London travel info",
            ),
            ToolRequest(
                server_name="test_server",
                tool_name="search_news",
                arguments={"query": "London events"},
                reasoning="Need London events news",
            ),
        ]

        # Execute tools in parallel
        results = await manager.execute_tools_parallel(tool_requests)

        # Verify all tools executed successfully
        assert len(results) == 3
        for result in results:
            assert isinstance(result, ToolResult)
            assert result.success is True
            assert result.error is None

        # Verify each tool returned expected data
        weather_result = next(r for r in results if r.tool_name == "weather_api")
        search_result = next(r for r in results if r.tool_name == "web_search")
        news_result = next(r for r in results if r.tool_name == "search_news")

        assert "location" in weather_result.result
        assert "results" in search_result.result
        assert "articles" in news_result.result

    @pytest.mark.asyncio
    async def test_tool_execution_failure_handling(self, mcp_server_and_manager):  # noqa: ANN001
        """Test handling of tool execution failures."""
        manager = mcp_server_and_manager


        # Try to execute non-existent tool
        tool_request = ToolRequest(
            server_name="test_server",
            tool_name="nonexistent_tool",
            arguments={},
            reasoning="Testing failure scenario",
        )

        result = await manager.execute_tool(tool_request)

        # Verify failure is handled gracefully
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower() or "unknown" in result.error.lower()
        assert result.tool_name == "nonexistent_tool"


@pytest.mark.integration
class TestReasoningAgentWithRealMCP:
    """Integration tests for reasoning agent with real MCP server tools."""

    @pytest_asyncio.fixture
    async def reasoning_agent_with_real_mcp(self):
        """Fixture providing reasoning agent with real MCP server."""
        # Start MCP server
        server_process = subprocess.Popen([  # noqa: ASYNC220
            "uv", "run", "python", "mcp_server/server.py",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        try:
            # Wait for server to be ready
            await self._wait_for_server_ready("http://localhost:8001")

            # Create real MCP manager
            config = MCPServerConfig(
                name="reasoning_agent_tools",
                url="http://localhost:8001/mcp/",
                enabled=True,
            )
            mcp_manager = MCPServerManager([config])
            await mcp_manager.initialize()

            # Create real prompt manager
            prompt_manager = PromptManager()
            await prompt_manager.initialize()

            # Create reasoning agent with real components
            async with httpx.AsyncClient() as client:
                agent = ReasoningAgent(
                    base_url="https://api.openai.com/v1",
                    api_key=os.getenv("OPENAI_API_KEY", "test-key"),
                    http_client=client,
                    mcp_manager=mcp_manager,
                    prompt_manager=prompt_manager,
                )

                yield agent

        finally:
            await mcp_manager.cleanup()
            server_process.terminate()
            await asyncio.sleep(0.5)

    async def _wait_for_server_ready(self, url: str, timeout: int = 10) -> None:  # noqa: ASYNC109
        """Wait for server to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if server process is listening on the port
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', 8001))
                sock.close()
                if result == 0:  # Port is open
                    return
            except Exception:
                pass
            await asyncio.sleep(0.1)
        raise TimeoutError(f"Server at {url} did not become ready within {timeout} seconds")

    @pytest.mark.asyncio
    @respx.mock
    async def test_reasoning_with_real_weather_tool(self, reasoning_agent_with_real_mcp):  # noqa: ANN001
        """Test reasoning agent using real weather tool in workflow."""
        agent = reasoning_agent_with_real_mcp

        # Mock OpenAI responses for reasoning steps
        # Step 1: Agent decides to use weather tool
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
                        "thought": "I need to get current weather information for Tokyo to answer the user's question",  # noqa: E501
                        "next_action": "use_tools",
                        "tools_to_use": [
                            {
                                "server_name": "reasoning_agent_tools",
                                "tool_name": "weather_api",
                                "arguments": {"location": "Tokyo"},
                                "reasoning": "User asked about Tokyo weather, need current conditions",  # noqa: E501
                            },
                        ],
                        "parallel_execution": False,
                    }),
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        # Step 2: Agent processes weather result and finishes
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
                        "thought": "I have the weather data for Tokyo, now I can provide a comprehensive answer",  # noqa: E501
                        "next_action": "finished",
                        "tools_to_use": [],
                        "parallel_execution": False,
                    }),
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 25, "completion_tokens": 8, "total_tokens": 33},
        }

        # Final synthesis response
        synthesis_response = {
            "id": "chatcmpl-synthesis",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Based on the current weather data for Tokyo, it's currently sunny with a temperature of 22°C. The humidity is moderate at 65%, and there's a light breeze from the northwest at 12 km/h. Tomorrow's forecast shows partly cloudy conditions with highs around 25°C and lows around 15°C. It's a great day to be outside in Tokyo!",  # noqa: E501
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60},
        }

        # Set up sequential OpenAI responses
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json=step1_response),
                httpx.Response(200, json=step2_response),
                httpx.Response(200, json=synthesis_response),
            ],
        )

        # Execute reasoning with real tools
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=MessageRole.USER, content="What's the weather like in Tokyo right now?")],  # noqa: E501
        )

        result = await agent.process_chat_completion(request)

        # Verify successful completion
        assert result is not None
        assert result.choices[0].message.content is not None

        # Verify the response includes weather information
        response_content = result.choices[0].message.content
        assert "Tokyo" in response_content
        assert any(word in response_content.lower() for word in ["temperature", "weather", "sunny", "cloudy"])  # noqa: E501

    @pytest.mark.asyncio
    @respx.mock
    async def test_reasoning_with_parallel_tools(self, reasoning_agent_with_real_mcp):  # noqa: ANN001
        """Test reasoning agent using multiple tools in parallel."""
        agent = reasoning_agent_with_real_mcp

        # Step 1: Agent decides to use multiple tools in parallel
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
                        "thought": "To plan a Tokyo trip, I need weather info and search for attractions",  # noqa: E501
                        "next_action": "use_tools",
                        "tools_to_use": [
                            {
                                "server_name": "reasoning_agent_tools",
                                "tool_name": "weather_api",
                                "arguments": {"location": "Tokyo"},
                                "reasoning": "Need current weather for trip planning",
                            },
                            {
                                "server_name": "reasoning_agent_tools",
                                "tool_name": "web_search",
                                "arguments": {"query": "Tokyo tourist attractions"},
                                "reasoning": "Need attraction recommendations",
                            },
                        ],
                        "parallel_execution": True,
                    }),
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
        }

        # Step 2: Agent processes results and finishes
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
                        "thought": "I have both weather and attraction data, can now provide comprehensive trip advice",  # noqa: E501
                        "next_action": "finished",
                        "tools_to_use": [],
                        "parallel_execution": False,
                    }),
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
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
                    "content": "Great timing for a Tokyo trip! The weather is currently pleasant with temperatures around 22°C and sunny conditions. For attractions, I recommend visiting Tokyo Skytree for panoramic views, Senso-ji Temple for cultural experiences, and Shibuya Crossing for the iconic urban experience. The current weather conditions are perfect for walking around and exploring these outdoor attractions.",  # noqa: E501
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
        }

        # Mock OpenAI responses
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json=step1_response),
                httpx.Response(200, json=step2_response),
                httpx.Response(200, json=synthesis_response),
            ],
        )

        # Execute reasoning
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=MessageRole.USER, content="I'm planning a trip to Tokyo. What's the weather like and what attractions should I visit?")],  # noqa: E501
        )

        result = await agent.process_chat_completion(request)

        # Verify successful completion with both tool results
        assert result is not None
        response_content = result.choices[0].message.content
        assert "Tokyo" in response_content
        assert any(word in response_content.lower() for word in ["weather", "temperature"])
        assert any(word in response_content.lower() for word in ["attraction", "visit", "temple", "skytree"])  # noqa: E501

    @pytest.mark.asyncio
    @respx.mock
    async def test_reasoning_continues_when_tools_fail(self, reasoning_agent_with_real_mcp):  # noqa: ANN001
        """Test that reasoning continues gracefully when tools fail."""
        agent = reasoning_agent_with_real_mcp

        # Step 1: Agent tries to use non-existent tool
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
                        "thought": "I'll try to use a specialized tool for this task",
                        "next_action": "use_tools",
                        "tools_to_use": [
                            {
                                "server_name": "reasoning_agent_tools",
                                "tool_name": "nonexistent_tool",
                                "arguments": {"query": "test"},
                                "reasoning": "Testing tool that doesn't exist",
                            },
                        ],
                        "parallel_execution": False,
                    }),
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        # Step 2: Agent recognizes tool failure and adapts
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
                        "thought": "The tool failed, but I can still provide helpful information using my knowledge",  # noqa: E501
                        "next_action": "finished",
                        "tools_to_use": [],
                        "parallel_execution": False,
                    }),
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
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
                    "content": "I apologize, but I encountered an issue with the specialized tool I tried to use. However, I can still help you with general information. While I couldn't access the specific tool, I can provide guidance based on my knowledge base.",  # noqa: E501
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 35, "completion_tokens": 15, "total_tokens": 50},
        }

        # Mock OpenAI responses
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json=step1_response),
                httpx.Response(200, json=step2_response),
                httpx.Response(200, json=synthesis_response),
            ],
        )

        # Execute reasoning
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=MessageRole.USER, content="Can you help me with a specialized task?")],  # noqa: E501
        )

        result = await agent.process_chat_completion(request)

        # Verify graceful failure handling
        assert result is not None
        response_content = result.choices[0].message.content
        assert "issue" in response_content.lower() or "apologize" in response_content.lower()
        assert "knowledge" in response_content.lower() or "help" in response_content.lower()


# TODO: Add streaming integration tests
# The streaming tests are complex due to multiple OpenAI calls and respx interactions
# These should be added as a future enhancement
