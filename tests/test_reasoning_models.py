"""Tests for reasoning agent Pydantic models."""
import pytest
from pydantic import ValidationError

from api.reasoning_models import (
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
        """Test tool request with no arguments."""
        request = ToolRequest(
            server_name="time_server",
            tool_name="get_current_time",
            reasoning="Need current time",
        )
        assert request.arguments == {}

    def test_missing_required_fields(self):
        """Test that missing required fields raise errors."""
        with pytest.raises(ValidationError):
            ToolRequest(
                server_name="test",
                tool_name="test",
                # Missing reasoning
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
