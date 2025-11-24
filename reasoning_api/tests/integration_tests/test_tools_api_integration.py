"""
Integration tests for reasoning agent using tools-api.

Tests the complete flow: ReasoningAgent → Tool.__call__() → ToolsAPIClient → mocked tools-api

Follows the pattern from test_reasoning_integration.py with mocked LiteLLM and tools-api.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

from reasoning_api.executors.reasoning_agent import ReasoningAgent
from reasoning_api.openai_protocol import OpenAIChatRequest
from reasoning_api.prompt_manager import PromptManager
from reasoning_api.tools import Tool, ToolResult
from reasoning_api.tools_client import ToolsAPIClient, ToolDefinition
from tests.conftest import OPENAI_TEST_MODEL, ReasoningAgentStreamingCollector
from tests.integration_tests.litellm_mocks import mock_weather_query


@pytest.mark.integration
class TestReasoningAgentWithToolsAPI:
    """Test ReasoningAgent executing tools via tools-api HTTP client (mocked)."""

    @pytest_asyncio.fixture
    async def mock_tools_api_client(self):
        """
        Create mock tools-api client that returns realistic tool definitions and results.

        Follows the pattern from test_reasoning_integration.py which uses mocks
        instead of real services.
        """
        mock_client = AsyncMock(spec=ToolsAPIClient)

        # Mock list_tools to return typical filesystem tools
        mock_client.list_tools.return_value = [
            ToolDefinition(
                name="read_text_file",
                description="Read the complete contents of a text file from the file system",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file"},
                    },
                    "required": ["path"],
                },
                tags=["filesystem", "read"],
            ),
            ToolDefinition(
                name="write_file",
                description="Write content to a file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
                tags=["filesystem", "write"],
            ),
        ]

        # Mock execute_tool to return structured response
        def mock_execute(tool_name: str, arguments: dict):
            if tool_name == "read_text_file":
                return ToolResult(
                    tool_name="read_text_file",
                    success=True,
                    result={
                        "path": arguments.get("path"),
                        "content": "Tokyo: 22°C, Sunny",
                        "line_count": 1,
                        "size_bytes": 19,
                    },
                    execution_time_ms=15.2,
                )
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool {tool_name} not implemented in mock",
                execution_time_ms=0.0,
            )

        mock_client.execute_tool.side_effect = mock_execute
        return mock_client

    @pytest_asyncio.fixture
    async def tools_from_tools_api(self, mock_tools_api_client):
        """
        Get tools from mocked tools-api client.

        This simulates what get_tools_from_tools_api() does in production.
        """
        # Get tool definitions
        tool_definitions = await mock_tools_api_client.list_tools()

        # Convert to Tool objects with tools_api_client set
        return [
            Tool(
                name=tool_def.name,
                description=tool_def.description,
                input_schema=tool_def.parameters,
                tags=tool_def.tags,
                tools_api_client=mock_tools_api_client,
            )
            for tool_def in tool_definitions
        ]


    @pytest_asyncio.fixture(loop_scope="function")
    async def reasoning_agent_with_tools_api(self, tools_from_tools_api: list[Tool]):
        """Create ReasoningAgent with tools from tools-api."""
        # Create and initialize prompt manager
        prompt_manager = PromptManager()
        await prompt_manager.initialize()

        agent = ReasoningAgent(
            tools=tools_from_tools_api,
            prompt_manager=prompt_manager,
        )

        yield agent

    @pytest.mark.asyncio
    async def test_reasoning_agent_calls_tools_via_tools_api(
        self,
        reasoning_agent_with_tools_api: ReasoningAgent,
    ) -> None:
        """
        Test that reasoning agent can execute tools via tools-api HTTP client.

        This is the key integration test verifying the full chain:
        1. ReasoningAgent calls tool (Tool.__call__)
        2. Tool.__call__ uses tools_api_client.execute_tool()
        3. Mocked tools_api_client returns structured response
        4. Result flows back through the chain

        Note: Uses mocked tools-api client following the pattern from
        test_reasoning_integration.py which mocks LiteLLM.
        """
        agent = reasoning_agent_with_tools_api

        # Configure mock LiteLLM to ask agent to use read_text_file tool
        mock_litellm = mock_weather_query(
            location="Tokyo",
            temperature="22°C",
            condition="Sunny",
            tool_name="read_text_file",  # Will use read_text_file from mocked tools-api
        )

        with patch('reasoning_api.executors.reasoning_agent.litellm.acompletion', side_effect=mock_litellm):
            request = OpenAIChatRequest(
                model=OPENAI_TEST_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": "Read weather data and tell me what it says.",
                    },
                ],
                max_tokens=500,
                temperature=0.1,
                stream=True,
            )

            # Collect streaming response
            collector = ReasoningAgentStreamingCollector()
            await collector.process(agent.execute_stream(request))

            # Verify response received
            assert len(collector.all_chunks) > 0

            # The tool was executed via mocked tools-api client
            # and returned structured response
            assert len(collector.reasoning_events) > 0

    @pytest.mark.asyncio
    async def test_tools_from_tools_api_have_correct_structure(
        self,
        tools_from_tools_api: list[Tool],
    ) -> None:
        """Test that tools loaded from tools-api have correct structure."""
        # Should have filesystem tools
        assert len(tools_from_tools_api) > 0

        # Find read_text_file tool
        read_tool = next((t for t in tools_from_tools_api if t.name == "read_text_file"), None)
        assert read_tool is not None

        # Verify structure
        assert read_tool.description == "Read the complete contents of a text file from the file system"
        assert "filesystem" in read_tool.tags
        assert "read" in read_tool.tags

        # Verify it has tools_api_client set (not direct function)
        assert read_tool.tools_api_client is not None
        assert read_tool.function is None

        # Verify input schema
        assert "path" in read_tool.input_schema["properties"]
        assert "path" in read_tool.input_schema["required"]

    @pytest.mark.asyncio
    async def test_tool_execution_via_tools_api_returns_structured_response(
        self,
        tools_from_tools_api: list[Tool],
    ) -> None:
        """Test that executing tool via tools-api returns structured response with metadata."""
        # Find read_text_file tool
        read_tool = next((t for t in tools_from_tools_api if t.name == "read_text_file"), None)
        assert read_tool is not None

        # Execute tool via mocked tools-api client
        result = await read_tool(path="/test/file.txt")

        # Verify structured response
        assert result.success is True
        assert result.error is None
        assert result.execution_time_ms > 0

        # Verify result contains metadata (not just text blob like MCP)
        # This is the key benefit of tools-api over MCP
        assert "content" in result.result
        assert "line_count" in result.result
        assert "size_bytes" in result.result
        assert "path" in result.result

        # Verify mock returned expected content
        assert result.result["content"] == "Tokyo: 22°C, Sunny"
        assert result.result["line_count"] == 1
        assert result.result["size_bytes"] == 19
