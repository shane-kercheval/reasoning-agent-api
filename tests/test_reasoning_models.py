"""Tests for reasoning agent Pydantic models."""
import os
import json
import pytest
from pydantic import ValidationError
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()
from api.reasoning_models import (  # noqa: E402
    MCPServerConfig,
    MCPServersConfig,
    ReasoningAction,
    ReasoningEvent,
    ReasoningEventStatus,
    ReasoningEventType,
    ReasoningStep,
    ToolInfo,
    ToolRequest,
    ToolResult,
)


class TestReasoningStep:
    """Test ReasoningStep model validation and behavior."""

    def test_valid_reasoning_step_minimal(self):
        """Test creating a minimal valid reasoning step."""
        step = ReasoningStep(
            thought="I need to analyze the question",
            next_action=ReasoningAction.CONTINUE_THINKING,
        )
        assert step.thought == "I need to analyze the question"
        assert step.next_action == ReasoningAction.CONTINUE_THINKING
        assert step.tools_to_use == []
        assert step.parallel_execution is False

    def test_valid_reasoning_step_with_tools(self):
        """Test creating a reasoning step with tool requests."""
        step = ReasoningStep(
            thought="I need to search for information",
            next_action=ReasoningAction.USE_TOOLS,
            tools_to_use=[
                ToolRequest(
                    server_name="web_search",
                    tool_name="search",
                    arguments={"query": "test query"},
                    reasoning="Need to find information",
                ),
            ],
            parallel_execution=False,
        )
        assert len(step.tools_to_use) == 1
        assert step.tools_to_use[0].server_name == "web_search"

    def test_parallel_execution_with_multiple_tools(self):
        """Test parallel execution flag with multiple tools."""
        step = ReasoningStep(
            thought="I need to search multiple sources",
            next_action=ReasoningAction.USE_TOOLS,
            tools_to_use=[
                ToolRequest(
                    server_name="web_search",
                    tool_name="search",
                    arguments={"query": "Tokyo population"},
                    reasoning="Get Tokyo data",
                ),
                ToolRequest(
                    server_name="web_search",
                    tool_name="search",
                    arguments={"query": "NYC population"},
                    reasoning="Get NYC data",
                ),
            ],
            parallel_execution=True,
        )
        assert step.parallel_execution is True
        assert len(step.tools_to_use) == 2

    def test_invalid_missing_required_fields(self):
        """Test that missing required fields raise validation errors."""
        with pytest.raises(ValidationError) as exc_info:
            ReasoningStep(thought="Missing next_action")

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("next_action",) for error in errors)


class TestToolRequest:
    """Test ToolRequest model validation."""

    def test_valid_tool_request(self):
        """Test creating a valid tool request."""
        request = ToolRequest(
            server_name="calculator",
            tool_name="add",
            arguments={"a": 5, "b": 3},
            reasoning="Need to add two numbers",
        )
        assert request.server_name == "calculator"
        assert request.tool_name == "add"
        assert request.arguments == {"a": 5, "b": 3}

    def test_empty_arguments(self):
        """Test tool request with empty arguments."""
        request = ToolRequest(
            server_name="time_server",
            tool_name="get_current_time",
            arguments={},
            reasoning="Need current time",
        )
        assert request.arguments == {}

    def test_missing_required_fields(self):
        """Test that missing required fields raise errors."""
        with pytest.raises(ValidationError):
            ToolRequest(
                server_name="test",
                tool_name="test",
                # Missing arguments and reasoning
            )


class TestReasoningEvent:
    """Test ReasoningEvent model for streaming metadata."""

    def test_reasoning_step_event(self):
        """Test creating a reasoning step event."""
        event = ReasoningEvent(
            type=ReasoningEventType.REASONING_STEP,
            step_id="1",
            status=ReasoningEventStatus.IN_PROGRESS,
        )
        assert event.type == ReasoningEventType.REASONING_STEP
        assert event.tool_name is None
        assert event.metadata == {}

    def test_tool_result_event(self):
        """Test creating a tool result event with metadata."""
        event = ReasoningEvent(
            type=ReasoningEventType.TOOL_RESULT,
            step_id="2a",
            tool_name="web_search",
            status=ReasoningEventStatus.COMPLETED,
            metadata={
                "population": "37400000",
                "source": "official_data",
            },
        )
        assert event.tool_name == "web_search"
        assert event.metadata["population"] == "37400000"

    def test_parallel_tools_event(self):
        """Test creating a parallel tools execution event."""
        event = ReasoningEvent(
            type=ReasoningEventType.PARALLEL_TOOLS,
            step_id="2",
            tools=["search:tokyo", "search:nyc"],
            status=ReasoningEventStatus.STARTED,
        )
        assert event.tools == ["search:tokyo", "search:nyc"]
        assert event.tool_name is None

    def test_error_event(self):
        """Test creating an error event."""
        event = ReasoningEvent(
            type=ReasoningEventType.ERROR,
            step_id="3",
            status=ReasoningEventStatus.FAILED,
            error="Connection timeout to MCP server",
        )
        assert event.status == ReasoningEventStatus.FAILED
        assert event.error == "Connection timeout to MCP server"


class TestToolResult:
    """Test ToolResult model."""

    def test_successful_result(self):
        """Test successful tool execution result."""
        result = ToolResult(
            server_name="calculator",
            tool_name="multiply",
            success=True,
            result={"answer": 42},
            execution_time_ms=23.5,
        )
        assert result.success is True
        assert result.result["answer"] == 42
        assert result.error is None

    def test_failed_result(self):
        """Test failed tool execution result."""
        result = ToolResult(
            server_name="web_search",
            tool_name="search",
            success=False,
            error="API rate limit exceeded",
            execution_time_ms=150.0,
        )
        assert result.success is False
        assert result.result is None
        assert "rate limit" in result.error


class TestMCPServerConfig:
    """Test MCP server configuration models."""

    def test_http_server_config(self):
        """Test HTTP server configuration."""
        config = MCPServerConfig(
            name="web_search",
            url="https://mcp-web-search.example.com",
            auth_env_var="WEB_SEARCH_API_KEY",
        )
        assert config.name == "web_search"
        assert config.url == "https://mcp-web-search.example.com"
        assert config.auth_env_var == "WEB_SEARCH_API_KEY"
        assert config.enabled is True

    def test_local_server_config(self):
        """Test local server configuration."""
        config = MCPServerConfig(
            name="local_tools",
            url="http://localhost:8001",
        )
        assert config.name == "local_tools"
        assert config.url == "http://localhost:8001"
        assert config.auth_env_var is None
        assert config.enabled is True

    def test_disabled_server(self):
        """Test disabled server configuration."""
        config = MCPServerConfig(
            name="test_server",
            url="https://example.com",
            enabled=False,
        )
        assert config.enabled is False

    def test_invalid_missing_url(self):
        """Test that missing URL raises error."""
        with pytest.raises(ValidationError):
            MCPServerConfig(name="test")


class TestMCPServersConfig:
    """Test root MCP servers configuration."""

    def test_multiple_servers(self):
        """Test configuration with multiple servers."""
        config = MCPServersConfig(
            servers=[
                MCPServerConfig(
                    name="server1",
                    url="http://localhost:8001",
                ),
                MCPServerConfig(
                    name="server2",
                    url="https://example.com",
                    enabled=False,
                ),
            ],
        )
        assert len(config.servers) == 2
        assert len(config.get_enabled_servers()) == 1
        assert config.get_enabled_servers()[0].name == "server1"

    def test_empty_servers_list(self):
        """Test configuration with no servers."""
        config = MCPServersConfig(servers=[])
        assert len(config.servers) == 0
        assert len(config.get_enabled_servers()) == 0


class TestToolInfo:
    """Test ToolInfo model."""

    def test_valid_tool_info(self):
        """Test creating valid tool information."""
        info = ToolInfo(
            server_name="calculator",
            tool_name="add",
            description="Adds two numbers together",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        )
        assert info.server_name == "calculator"
        assert info.tool_name == "add"
        assert "properties" in info.input_schema


class TestOpenAICompatibility:
    """Test OpenAI structured outputs compatibility."""

    def test_reasoning_step_schema_openai_compatible(self):
        """Test that ReasoningStep generates OpenAI-compatible JSON schema."""
        schema = ReasoningStep.model_json_schema()

        # Check root level structure
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

        # Check that all object properties have additionalProperties: false
        def check_object_properties(obj_schema: dict) -> None:
            if obj_schema.get("type") == "object":
                # Objects should have additionalProperties: false for OpenAI
                assert "additionalProperties" in obj_schema
                assert obj_schema["additionalProperties"] is False

                # Check nested properties
                for prop_schema in obj_schema.get("properties", {}).values():
                    check_object_properties(prop_schema)
            elif obj_schema.get("type") == "array":
                # Check array items
                items = obj_schema.get("items")
                if items:
                    check_object_properties(items)

        check_object_properties(schema)

        # Verify specific fields exist
        properties = schema["properties"]
        assert "thought" in properties
        assert "next_action" in properties
        assert "tools_to_use" in properties
        assert "parallel_execution" in properties

        # Test that the schema can be serialized (important for OpenAI API)
        schema_json = json.dumps(schema)
        assert len(schema_json) > 0

    def test_tool_request_schema_openai_compatible(self):
        """Test that ToolRequest generates OpenAI-compatible JSON schema."""
        schema = ToolRequest.model_json_schema()

        # Check that arguments field is properly constrained
        properties = schema["properties"]
        assert "arguments" in properties

        # Arguments should be an object with additionalProperties: false
        args_schema = properties["arguments"]
        assert args_schema["type"] == "object"
        assert args_schema.get("additionalProperties") is False

        # Test creating a valid instance
        tool_req = ToolRequest(
            server_name="test_server",
            tool_name="test_tool",
            arguments={"query": "test", "limit": "10"},  # All string values
            reasoning="Testing the tool",
        )
        assert tool_req.arguments["query"] == "test"
        assert tool_req.arguments["limit"] == "10"

    def test_reasoning_step_with_tools_serialization(self):
        """Test full ReasoningStep with tools can be serialized properly."""
        step = ReasoningStep(
            thought="I need to search for information",
            next_action=ReasoningAction.USE_TOOLS,
            tools_to_use=[
                ToolRequest(
                    server_name="web_search",
                    tool_name="search",
                    arguments={"query": "test query", "limit": "5"},
                    reasoning="Need to find information",
                ),
            ],
            parallel_execution=False,
        )

        # Should serialize without errors
        step_dict = step.model_dump()
        step_json = step.model_dump_json()

        assert "thought" in step_dict
        assert len(step_dict["tools_to_use"]) == 1
        assert len(step_json) > 0


class TestOpenAISDKIntegration:
    """Test actual OpenAI SDK integration with our reasoning models."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires valid OpenAI API key")
    async def test_reasoning_step_with_openai_json_mode(self):
        """Test that ReasoningStep works with OpenAI's JSON mode."""
        # Skip if no API key
        api_key = os.getenv("OPENAI_API_KEY")
        client = AsyncOpenAI(api_key=api_key)
        try:
            # Test JSON mode approach (like our reasoning agent uses)
            schema = ReasoningStep.model_json_schema()
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a reasoning assistant. Respond with valid JSON matching this schema: {schema}",
                    },
                    {
                        "role": "user",
                        "content": "I need to find the population of Tokyo. What should I do?",
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=300,
            )

            # Verify we got a valid response
            assert response.choices is not None
            assert len(response.choices) > 0

            # Parse the JSON response
            content = response.choices[0].message.content
            assert content is not None

            json_data = json.loads(content)
            reasoning_step = ReasoningStep.model_validate(json_data)

            # Verify the reasoning step has required fields
            assert isinstance(reasoning_step.thought, str)
            assert len(reasoning_step.thought) > 0
            assert isinstance(reasoning_step.next_action, ReasoningAction)
            assert isinstance(reasoning_step.tools_to_use, list)
            assert isinstance(reasoning_step.parallel_execution, bool)

        except Exception as e:
            pytest.fail(f"OpenAI JSON mode failed with our ReasoningStep model: {e}")
        finally:
            await client.close()

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires valid OpenAI API key")
    async def test_tool_request_in_reasoning_step_json_mode(self):
        """Test that ToolRequest within ReasoningStep works with OpenAI JSON mode."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.startswith("test-") or api_key == "your-openai-api-key-here":
            pytest.skip("No valid OpenAI API key available for integration test")

        client = AsyncOpenAI(api_key=api_key)

        try:
            # Test with JSON mode and tool usage
            schema = ReasoningStep.model_json_schema()
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a reasoning assistant. Available tools: "
                            "web_search, weather_api, calculator. "
                            f"Respond with valid JSON matching this schema: {schema}. "
                            "When the user asks for specific information, include tool requests."
                        ),
                    },
                    {
                        "role": "user",
                        "content": "I need to search for the current population of Tokyo. What should I do?",
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=400,
            )

            # Parse the response
            content = response.choices[0].message.content
            assert content is not None

            json_data = json.loads(content)
            reasoning_step = ReasoningStep.model_validate(json_data)

            # Basic validation
            assert isinstance(reasoning_step.thought, str)
            assert len(reasoning_step.thought) > 0
            assert isinstance(reasoning_step.next_action, ReasoningAction)
            assert isinstance(reasoning_step.tools_to_use, list)

            # If tools were requested, verify their format
            for tool_req in reasoning_step.tools_to_use:
                assert isinstance(tool_req, ToolRequest)
                assert isinstance(tool_req.server_name, str)
                assert len(tool_req.server_name) > 0
                assert isinstance(tool_req.tool_name, str)
                assert len(tool_req.tool_name) > 0
                assert isinstance(tool_req.arguments, dict)
                assert isinstance(tool_req.reasoning, str)
                assert len(tool_req.reasoning) > 0

        except Exception as e:
            pytest.fail(f"OpenAI JSON mode failed with ToolRequest: {e}")
        finally:
            await client.close()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_openai_error_handling(self):
        """Test that we handle OpenAI API errors gracefully."""
        pytest.importorskip("openai")


        # Test with invalid API key to ensure error handling works
        client = AsyncOpenAI(api_key="invalid-key")

        try:
            with pytest.raises(Exception):  # Should raise an authentication error  # noqa: PT011
                await client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "test"}],
                    response_format=ReasoningStep,
                )
        finally:
            await client.close()
