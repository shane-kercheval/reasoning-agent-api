"""
Pydantic model factories for test data generation.

This module provides factory functions for creating validated Pydantic model
instances used in tests, replacing raw dictionaries with type-safe objects.
"""

import time
from typing import Any

from api.reasoning_models import (
    ReasoningAction,
    ReasoningStep,
    ToolPrediction,
    ReasoningEvent,
    ReasoningEventType,
    ReasoningEventStatus,
    MCPServerConfig,
    MCPServersConfig,
    ToolInfo,
)
from api.openai_protocol import (
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIDelta,
    OpenAIRequestBuilder,
    OpenAIResponseBuilder,
    OpenAIStreamingResponseBuilder,
    create_sse,
)


class ToolPredictionFactory:
    """Factory for creating ToolPrediction instances with common configurations."""

    @staticmethod
    def weather_prediction(location: str, reasoning: str | None = None) -> ToolPrediction:
        """
        Create a weather tool prediction.

        Args:
            location: Location to get weather for.
            reasoning: Optional custom reasoning, defaults to generic reasoning.

        Returns:
            ToolPrediction configured for weather lookup.
        """
        return ToolPrediction(
            tool_name="weather_tool",
            arguments={"location": location},
            reasoning=reasoning or f"Need current weather information for {location}",
        )

    @staticmethod
    def search_prediction(query: str, reasoning: str | None = None) -> ToolPrediction:
        """
        Create a search tool prediction.

        Args:
            query: Search query string.
            reasoning: Optional custom reasoning, defaults to generic reasoning.

        Returns:
            ToolPrediction configured for web search.
        """
        return ToolPrediction(
            tool_name="search_tool",
            arguments={"query": query},
            reasoning=reasoning or f"Need to search for information about: {query}",
        )

    @staticmethod
    def failing_prediction(
            should_fail: bool = True,
            reasoning: str | None = None,
        ) -> ToolPrediction:
        """
        Create a failing tool prediction for error testing.

        Args:
            should_fail: Whether the tool should fail.
            reasoning: Optional custom reasoning.

        Returns:
            ToolPrediction configured for failure testing.
        """
        return ToolPrediction(
            tool_name="failing_tool",
            arguments={"should_fail": should_fail},
            reasoning=reasoning or "Testing error handling with a tool that may fail",
        )

    @staticmethod
    def calculator_prediction(
            operation: str,
            a: float,
            b: float,
            reasoning: str | None = None,
        ) -> ToolPrediction:
        """
        Create a calculator tool prediction.

        Args:
            operation: Mathematical operation to perform.
            a: First operand.
            b: Second operand.
            reasoning: Optional custom reasoning.

        Returns:
            ToolPrediction configured for calculation.
        """
        return ToolPrediction(
            tool_name="calculator_tool",
            arguments={"operation": operation, "a": a, "b": b},
            reasoning=reasoning or f"Need to {operation} {a} and {b}",
        )

    @staticmethod
    def custom_prediction(
            tool_name: str,
            arguments: dict[str, Any],
            reasoning: str,
        ) -> ToolPrediction:
        """
        Create a custom tool prediction.

        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.
            reasoning: Reasoning for using this tool.

        Returns:
            ToolPrediction with custom configuration.
        """
        return ToolPrediction(
            tool_name=tool_name,
            arguments=arguments,
            reasoning=reasoning,
        )


class ReasoningStepFactory:
    """Factory for creating ReasoningStep instances with common patterns."""

    @staticmethod
    def thinking_step(thought: str) -> ReasoningStep:
        """
        Create a thinking step that continues reasoning.

        Args:
            thought: The reasoning thought content.

        Returns:
            ReasoningStep configured for continued thinking.
        """
        return ReasoningStep(
            thought=thought,
            next_action=ReasoningAction.CONTINUE_THINKING,
            tools_to_use=[],
            concurrent_execution=False,
        )

    @staticmethod
    def tool_step(
            thought: str,
            tools: list[ToolPrediction],
            concurrent: bool = False,
        ) -> ReasoningStep:
        """
        Create a tool execution step.

        Args:
            thought: The reasoning thought content.
            tools: List of tools to execute.
            concurrent: Whether tools should run concurrently.

        Returns:
            ReasoningStep configured for tool execution.
        """
        return ReasoningStep(
            thought=thought,
            next_action=ReasoningAction.USE_TOOLS,
            tools_to_use=tools,
            concurrent_execution=concurrent,
        )

    @staticmethod
    def finished_step(thought: str) -> ReasoningStep:
        """
        Create a finishing step.

        Args:
            thought: The final reasoning thought.

        Returns:
            ReasoningStep configured to finish reasoning.
        """
        return ReasoningStep(
            thought=thought,
            next_action=ReasoningAction.FINISHED,
            tools_to_use=[],
            concurrent_execution=False,
        )

    @staticmethod
    def weather_lookup_step(location: str) -> ReasoningStep:
        """
        Create a step that looks up weather for a location.

        Args:
            location: Location to get weather for.

        Returns:
            ReasoningStep configured for weather lookup.
        """
        return ReasoningStepFactory.tool_step(
            thought=f"I need to get current weather information for {location}",
            tools=[ToolPredictionFactory.weather_prediction(location)],
        )

    @staticmethod
    def search_and_weather_step(query: str, location: str) -> ReasoningStep:
        """
        Create a step that searches and gets weather concurrently.

        Args:
            query: Search query.
            location: Location for weather.

        Returns:
            ReasoningStep configured for concurrent search and weather.
        """
        return ReasoningStepFactory.tool_step(
            thought=f"I need to search for '{query}' and get weather for {location}",
            tools=[
                ToolPredictionFactory.search_prediction(query),
                ToolPredictionFactory.weather_prediction(location),
            ],
            concurrent=True,
        )


class ReasoningEventFactory:
    """Factory for creating ReasoningEvent instances for streaming tests."""

    @staticmethod
    def reasoning_started(step_id: str) -> ReasoningEvent:
        """
        Create a reasoning step started event.

        Args:
            step_id: Unique step identifier.

        Returns:
            ReasoningEvent for reasoning step start.
        """
        return ReasoningEvent(
            type=ReasoningEventType.REASONING_STEP,
            step_id=step_id,
            status=ReasoningEventStatus.STARTED,
            metadata={},
        )

    @staticmethod
    def tool_execution_started(step_id: str, tools: list[str]) -> ReasoningEvent:
        """
        Create a tool execution started event.

        Args:
            step_id: Unique step identifier.
            tools: List of tool names being executed.

        Returns:
            ReasoningEvent for tool execution start.
        """
        return ReasoningEvent(
            type=ReasoningEventType.TOOL_EXECUTION,
            step_id=step_id,
            status=ReasoningEventStatus.STARTED,
            metadata={"tools": tools},
        )

    @staticmethod
    def tool_result_completed(step_id: str, tool_name: str, success: bool) -> ReasoningEvent:
        """
        Create a tool result completed event.

        Args:
            step_id: Unique step identifier.
            tool_name: Name of the completed tool.
            success: Whether the tool succeeded.

        Returns:
            ReasoningEvent for tool completion.
        """
        return ReasoningEvent(
            type=ReasoningEventType.TOOL_RESULT,
            step_id=step_id,
            status=ReasoningEventStatus.COMPLETED,
            metadata={"tool_name": tool_name, "success": success},
        )

    @staticmethod
    def synthesis_completed(step_id: str) -> ReasoningEvent:
        """
        Create a synthesis completed event.

        Args:
            step_id: Unique step identifier.

        Returns:
            ReasoningEvent for synthesis completion.
        """
        return ReasoningEvent(
            type=ReasoningEventType.SYNTHESIS,
            step_id=step_id,
            status=ReasoningEventStatus.COMPLETED,
            metadata={},
        )

    @staticmethod
    def error_event(step_id: str, error_message: str) -> ReasoningEvent:
        """
        Create an error event.

        Args:
            step_id: Unique step identifier.
            error_message: Error description.

        Returns:
            ReasoningEvent for error condition.
        """
        return ReasoningEvent(
            type=ReasoningEventType.ERROR,
            step_id=step_id,
            status=ReasoningEventStatus.FAILED,
            metadata={},
            error=error_message,
        )


class OpenAIRequestFactory:
    """Factory for creating OpenAI request objects."""

    @staticmethod
    def simple_chat_request(user_message: str, model: str = "gpt-4o") -> OpenAIChatRequest:
        """
        Create a simple chat request.

        Args:
            user_message: Message from the user.
            model: OpenAI model to use.

        Returns:
            OpenAIChatRequest configured for simple chat.
        """
        return (OpenAIRequestBuilder()
                .model(model)
                .message("user", user_message)
                .build())

    @staticmethod
    def streaming_request(user_message: str, model: str = "gpt-4o") -> OpenAIChatRequest:
        """
        Create a streaming chat request.

        Args:
            user_message: Message from the user.
            model: OpenAI model to use.

        Returns:
            OpenAIChatRequest configured for streaming.
        """
        return (
            OpenAIRequestBuilder()
            .model(model)
            .message("user", user_message)
            .streaming()
            .build()
        )

    @staticmethod
    def json_mode_request(user_message: str, model: str = "gpt-4o") -> OpenAIChatRequest:
        """
        Create a JSON mode request.

        Args:
            user_message: Message from the user.
            model: OpenAI model to use.

        Returns:
            OpenAIChatRequest configured for JSON mode.
        """
        return (
            OpenAIRequestBuilder()
            .model(model)
            .message("user", user_message)
            .json_mode()
            .build()
        )

    @staticmethod
    def conversation_request(
            messages: list[dict[str, str]],
            model: str = "gpt-4o",
        ) -> OpenAIChatRequest:
        """
        Create a multi-turn conversation request.

        Args:
            messages: List of conversation messages.
            model: OpenAI model to use.

        Returns:
            OpenAIChatRequest configured for conversation.
        """
        return (
            OpenAIRequestBuilder()
            .model(model)
            .messages(messages)
            .build()
        )


class OpenAIResponseFactory:
    """Factory for creating OpenAI response objects."""

    @staticmethod
    def simple_response(
            content: str,
            completion_id: str = "chatcmpl-test123",
        ) -> OpenAIChatResponse:
        """
        Create a simple chat response.

        Args:
            content: Response content.
            completion_id: Unique completion ID.

        Returns:
            OpenAIChatResponse with the specified content.
        """
        return (
            OpenAIResponseBuilder()
            .id(completion_id)
            .model("gpt-4o")
            .created(int(time.time()))
            .choice(0, "assistant", content)
            .usage(10, 20)
            .build()
        )

    @staticmethod
    def reasoning_step_response(
            step: ReasoningStep,
            completion_id: str = "chatcmpl-reasoning",
        ) -> OpenAIChatResponse:
        """
        Create a response containing a reasoning step.

        Args:
            step: ReasoningStep to include in response.
            completion_id: Unique completion ID.

        Returns:
            OpenAIChatResponse containing the reasoning step JSON.
        """
        return (
            OpenAIResponseBuilder()
            .id(completion_id)
            .model("gpt-4o")
            .created(int(time.time()))
            .choice(0, "assistant", step.model_dump_json())
            .usage(50, 100)
            .build()
        )

    @staticmethod
    def tool_execution_response(
            tool_results: list[dict[str, Any]],
            completion_id: str = "chatcmpl-tools",
        ) -> OpenAIChatResponse:
        """
        Create a response with tool execution results.

        Args:
            tool_results: List of tool execution results.
            completion_id: Unique completion ID.

        Returns:
            OpenAIChatResponse containing tool results.
        """
        results_text = "Tool execution results:\n" + "\n".join([
            f"- {result.get('tool_name', 'Unknown')}: {result.get('result', 'No result')}"
            for result in tool_results
        ])

        return (
            OpenAIResponseBuilder()
            .id(completion_id)
            .model("gpt-4o")
            .created(int(time.time()))
            .choice(0, "assistant", results_text)
            .usage(100, 150)
            .build()
        )

    @staticmethod
    def error_response(
            error_message: str,
            completion_id: str = "chatcmpl-error",
        ) -> OpenAIChatResponse:
        """
        Create an error response.

        Args:
            error_message: Error message to include.
            completion_id: Unique completion ID.

        Returns:
            OpenAIChatResponse containing error information.
        """
        return (
            OpenAIResponseBuilder()
            .id(completion_id)
            .model("gpt-4o")
            .created(int(time.time()))
            .choice(0, "assistant", f"Error: {error_message}")
            .usage(5, 10)
            .build()
        )


class StreamingResponseFactory:
    """Factory for creating streaming response sequences."""

    @staticmethod
    def simple_streaming_response(content: str, completion_id: str = "chatcmpl-stream") -> str:
        """
        Create a simple streaming response.

        Args:
            content: Content to stream.
            completion_id: Unique completion ID.

        Returns:
            Complete SSE stream as string.
        """
        builder = OpenAIStreamingResponseBuilder()

        # Initial role chunk
        builder.chunk(completion_id, "gpt-4o", delta_role="assistant")

        # Stream content word by word
        words = content.split()
        for i, word in enumerate(words):
            chunk_content = word if i == 0 else f" {word}"
            builder.chunk(completion_id, "gpt-4o", delta_content=chunk_content)

        # Final chunk with finish reason
        builder.chunk(completion_id, "gpt-4o", finish_reason="stop")
        builder.done()

        return builder.build()

    @staticmethod
    def reasoning_streaming_response(steps: list[ReasoningStep], final_content: str) -> str:
        """
        Create a streaming response with reasoning steps.

        Args:
            steps: List of reasoning steps to stream.
            final_content: Final response content.

        Returns:
            Complete SSE stream with reasoning events.
        """
        builder = OpenAIStreamingResponseBuilder()
        completion_id = "chatcmpl-reasoning-stream"

        # Initial role chunk
        builder.chunk(completion_id, "gpt-4o", delta_role="assistant")

        # Stream reasoning steps
        for i, step in enumerate(steps):
            # Reasoning event for this step
            reasoning_event = ReasoningEventFactory.reasoning_started(f"step_{i+1}")

            # Create delta with reasoning event
            delta_with_event = OpenAIDelta(reasoning_event=reasoning_event)

            # Manual chunk creation to include reasoning_event
            chunk_data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "delta": delta_with_event.model_dump(exclude_none=True),
                    "finish_reason": None,
                }],
            }

            builder._chunks.append(create_sse(chunk_data))

        # Stream final content
        words = final_content.split()
        for i, word in enumerate(words):
            chunk_content = word if i == 0 else f" {word}"
            builder.chunk(completion_id, "gpt-4o", delta_content=chunk_content)

        # Final chunk
        builder.chunk(completion_id, "gpt-4o", finish_reason="stop")
        builder.done()

        return builder.build()


class MCPConfigFactory:
    """Factory for creating MCP configuration objects."""

    @staticmethod
    def local_server_config(name: str, port: int = 8001) -> MCPServerConfig:
        """
        Create a local MCP server configuration.

        Args:
            name: Server name.
            port: Server port.

        Returns:
            MCPServerConfig for local server.
        """
        return MCPServerConfig(
            name=name,
            url=f"http://localhost:{port}",
            enabled=True,
        )

    @staticmethod
    def remote_server_config(
            name: str,
            url: str, auth_env_var: str | None = None,
        ) -> MCPServerConfig:
        """
        Create a remote MCP server configuration.

        Args:
            name: Server name.
            url: Server URL.
            auth_env_var: Environment variable for auth token.

        Returns:
            MCPServerConfig for remote server.
        """
        return MCPServerConfig(
            name=name,
            url=url,
            auth_env_var=auth_env_var,
            enabled=True,
        )

    @staticmethod
    def multi_server_config(server_configs: list[MCPServerConfig]) -> MCPServersConfig:
        """
        Create a configuration with multiple MCP servers.

        Args:
            server_configs: List of server configurations.

        Returns:
            MCPServersConfig containing all servers.
        """
        return MCPServersConfig(servers=server_configs)


class ToolInfoFactory:
    """Factory for creating ToolInfo objects."""

    @staticmethod
    def weather_tool_info() -> ToolInfo:
        """
        Create tool info for weather tool.

        Returns:
            ToolInfo for weather tool.
        """
        return ToolInfo(
            tool_name="weather_tool",
            description="Get current weather information for a location",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Name of the location to get weather for",
                    },
                },
                "required": ["location"],
            },
        )

    @staticmethod
    def search_tool_info() -> ToolInfo:
        """
        Create tool info for search tool.

        Returns:
            ToolInfo for search tool.
        """
        return ToolInfo(
            tool_name="search_tool",
            description="Search the web for information",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        )
