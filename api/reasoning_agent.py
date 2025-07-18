r"""
Advanced reasoning agent that orchestrates OpenAI structured outputs with MCP tool execution.

This agent implements a sophisticated reasoning process that:
1. Uses OpenAI structured outputs to generate reasoning steps
2. Orchestrates concurrent tool execution when needed
3. Streams reasoning progress with enhanced metadata via Server-Sent Events (SSE)
4. Maintains full OpenAI API compatibility

STREAMING EXPLANATION:
The agent supports both non-streaming and streaming modes. Streaming uses the Server-Sent
Events (SSE) format - a web standard for real-time server-to-client communication. SSE
events have the format:
    data: {json_payload}\n\n

This allows clients to receive reasoning steps and final responses incrementally, providing
a responsive user experience. The stream terminates with "data: [DONE]\n\n".

The agent acts as the main orchestrator coordinating between prompts, reasoning,
tool execution, and response synthesis.

---

┌─────────────────────────────────────────────────────────────────────────────┐
│                            REASONING AGENT FLOW                             │
└─────────────────────────────────────────────────────────────────────────────┘

                        User Request
                            │
                ┌────────────┴────────────┐
                │                         │
        NON-STREAMING                STREAMING
          execute                   execute_stream
                │                         │
                └────────────┬────────────┘
                            │
                    ┌────────────────────┐
                    │ _core_reasoning_   │ ← SINGLE SOURCE OF TRUTH
                    │    _process()      │   (Event Generator)
                    └────────────────────┘
                            │
                    ┌────────────────────┐
                    │   EVENT STREAM     │
                    │                    │
                    │ Step 1:            │
                    │ ├─ start_step      │
                    │ ├─ step_plan       │
                    │ ├─ start_tools     │  (if tools needed)
                    │ ├─ complete_tools  │  (if tools needed)
                    │ └─ complete_step   │
                    │                    │
                    │ Step 2:            │
                    │ ├─ start_step      │
                    │ ├─ step_plan       │
                    │ └─ complete_step   │  (no tools this time)
                    │                    │
                    │ Final:             │
                    │ └─ finish          │
                    └────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
        NON-STREAMING                STREAMING
                │                         │
                │                         │
        ┌───────▼────────┐        ┌───────▼────────┐
        │ _execute_      │        │ _stream_       │
        │ _reasoning_    │        │ _reasoning_    │
        │ _process()     │        │ _process()     │
        │                │        │                │
        │ • Silently     │        │ • Convert      │
        │   consume      │        │   events to    │
        │   all events   │        │   SSE chunks   │
        │ • Return       │        │ • Yield each   │
        │   final        │        │   chunk        │
        │   context      │        │ • Store final  │
        │                │        │   context      │
        └───────┬────────┘        └───────┬────────┘
                │                         │
                │                         │
        ┌───────▼────────┐        ┌───────▼────────┐
        │ _synthesize_   │        │ _stream_       │
        │ _final_        │        │ _final_        │
        │ _response()    │        │ _response()    │
        │                │        │                │
        │ • Build msgs   │        │ • Build msgs   │
        │ • Include      │        │ • Include      │
        │   reasoning    │        │   reasoning    │
        │   summary      │        │   summary      │
        │ • Call OpenAI  │        │ • Stream from  │
        │ • Return       │        │   OpenAI       │
        │   complete     │        │ • Yield SSE    │
        │   response     │        │   chunks       │
        └───────┬────────┘        └───────┬────────┘
                │                         │
                │                         │
        ┌───────▼────────┐        ┌───────▼────────┐
        │   RESPONSE     │        │   RESPONSE     │
        │                │        │                │
        │ ChatCompletion │        │ SSE Stream:    │
        │ Response with  │        │ data: {chunk1} │
        │ reasoning      │        │ data: {chunk2} │
        │ summary        │        │ data: {chunk3} │
        │                │        │ data: [DONE]   │
        └────────────────┘        └────────────────┘

Example Event Flow for a Request with Tools

Request: "What's the weather in Tokyo?"

┌─────────────────────────────────────────────────────────────────────────────┐
│                         _core_reasoning_process()                            │
│                             YIELDS EVENTS                                    │
└─────────────────────────────────────────────────────────────────────────────┘

EVENT 1: ("start_step", {"iteration": 1})
    │
    ├─ NON-STREAMING: ✓ (consume silently)
    └─ STREAMING: → SSE chunk with ReasoningEvent(REASONING_STEP, IN_PROGRESS)

EVENT 2: ("step_plan", {"iteration": 1, "reasoning_step": ReasoningStep})
    │
    ├─ NON-STREAMING: ✓ (consume silently)
    └─ STREAMING: → SSE chunk with thought + tools_planned

EVENT 3: ("start_tools", {"iteration": 1, "tools": [get_weather]})
    │
    ├─ NON-STREAMING: ✓ (consume silently)
    └─ STREAMING: → SSE chunk with ReasoningEvent(TOOL_EXECUTION, IN_PROGRESS)

EVENT 4: ("complete_tools", {"iteration": 1, "tool_results": [WeatherResult]})
    │
    ├─ NON-STREAMING: ✓ (consume silently)
    └─ STREAMING: → SSE chunk with ReasoningEvent(TOOL_EXECUTION, COMPLETED)

EVENT 5: ("complete_step", {"iteration": 1, "reasoning_step": Step, "had_tools": true})
    │
    ├─ NON-STREAMING: ✓ (consume silently)
    └─ STREAMING: → SSE chunk with ReasoningEvent(REASONING_STEP, COMPLETED)

EVENT 6: ("finish", {"context": {steps: [...], tool_results: [...]}})
    │
    ├─ NON-STREAMING: ✓ Return this context
    └─ STREAMING: → Store context + SSE chunk with ReasoningEvent(SYNTHESIS, COMPLETED)
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from openai import AsyncOpenAI

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    StreamChoice,
    Delta,
    Choice,
    ChatMessage,
    Usage,
)
from .tools import Tool, ToolResult
from .prompt_manager import PromptManager
from .reasoning_models import (
    ReasoningStep,
    ReasoningAction,
    ToolPrediction,
    ReasoningEvent,
    ReasoningEventType,
    ReasoningEventStatus,
)


class ReasoningError(Exception):
    """Raised when reasoning process fails."""

    def __init__(
            self,
            message: str,
            step: int | None = None,
            details: dict[str, Any] | None = None,
        ):
        super().__init__(message)
        self.step = step
        self.details = details or {}


class ReasoningAgent:
    """
    Advanced reasoning agent that orchestrates OpenAI structured outputs with MCP tools.

    This agent implements the complete reasoning workflow:
    1. Loads reasoning prompts from markdown files
    2. Uses OpenAI structured outputs to generate reasoning steps
    3. Orchestrates concurrent tool execution when needed
    4. Streams progress with reasoning_event metadata
    5. Synthesizes final responses
    """

    def __init__(
        self,
        base_url: str,
        http_client: httpx.AsyncClient,
        tools: list[Tool],
        prompt_manager: PromptManager,
        api_key: str | None = None,
        max_reasoning_iterations: int = 20,
    ):
        """
        Initialize the reasoning agent.

        Args:
            base_url: Base URL for the OpenAI-compatible API
            http_client: HTTP client for making requests
            tools: List of available tools for execution
            prompt_manager: Prompt manager for loading reasoning prompts
            api_key: OpenAI API key for authentication
            max_reasoning_iterations: Maximum number of reasoning iterations to perform
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.http_client = http_client
        self.tools = {tool.name: tool for tool in tools}
        self.prompt_manager = prompt_manager

        # Initialize OpenAI client for structured outputs
        self.openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )

        # Reasoning context
        self.max_reasoning_iterations = max_reasoning_iterations

        # Ensure auth header is set on http_client
        if api_key and 'Authorization' not in self.http_client.headers:
            self.http_client.headers['Authorization'] = f"Bearer {api_key}"

    async def execute(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """
        Process a non-streaming chat completion request with full reasoning.

        Args:
            request: The chat completion request

        Returns:
            OpenAI-compatible chat completion response with reasoning applied

        Raises:
            ReasoningError: If reasoning process fails
            ValidationError: If request validation fails
            httpx.HTTPStatusError: If the OpenAI API returns an error
        """
        # Run the reasoning process
        reasoning_context = await self._execute_reasoning_process(request)
        # Generate final response using synthesis prompt
        return await self._synthesize_final_response(request, reasoning_context)

    async def _core_reasoning_process(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[tuple[str, Any]]:
        """
        Core reasoning process that yields events as (event_type, event_data) tuples.

        This is the single source of truth for the reasoning loop.
        Both streaming and non-streaming paths consume these events.

        Yields:
            Tuples of (event_type, event_data) where:
            - event_type: "start_step", "complete_step", "start_tools", "complete_tools", "finish"
            - event_data: Dict with relevant data for each event type
        """
        reasoning_context = {
            "steps": [],
            "tool_results": [],
            "final_thoughts": "",
            "user_request": request,
        }

        # Get reasoning system prompt
        system_prompt = await self.prompt_manager.get_prompt("reasoning_system")

        for iteration in range(self.max_reasoning_iterations):
            # Yield start of reasoning step
            yield ("start_step", {"iteration": iteration + 1})

            # Generate next reasoning step using structured outputs
            reasoning_step = await self._generate_reasoning_step(
                request,
                reasoning_context,
                system_prompt,
            )
            reasoning_context["steps"].append(reasoning_step)

            # Yield reasoning step plan (what we plan to do)
            yield ("step_plan", {
                "iteration": iteration + 1,
                "reasoning_step": reasoning_step,
            })

            # Execute tools if needed
            if reasoning_step.tools_to_use:
                # Yield start of tool execution
                yield ("start_tools", {
                    "iteration": iteration + 1,
                    "tools": reasoning_step.tools_to_use,
                    "concurrent_execution": reasoning_step.concurrent_execution,
                })

                if reasoning_step.concurrent_execution:
                    tool_results = await self._execute_tools_concurrently(
                        reasoning_step.tools_to_use,
                    )
                else:
                    tool_results = await self._execute_tools_sequentially(
                        reasoning_step.tools_to_use,
                    )
                reasoning_context["tool_results"].extend(tool_results)

                # Yield completed tool execution
                yield ("complete_tools", {
                    "iteration": iteration + 1,
                    "tool_results": tool_results,
                })

            # Now we can yield step completion (after tools are done)
            yield ("complete_step", {
                "iteration": iteration + 1,
                "reasoning_step": reasoning_step,
                "had_tools": bool(reasoning_step.tools_to_use),
            })

            # Check if we should continue reasoning
            if reasoning_step.next_action == ReasoningAction.FINISHED:
                break

        # Yield final context
        yield ("finish", {"context": reasoning_context})

    async def _execute_reasoning_process(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """Execute the full reasoning process by consuming core events silently."""
        reasoning_context = None

        # Consume events from core reasoning process
        async for event_type, event_data in self._core_reasoning_process(request):
            # ignore all events except the 'finish' event to get final context
            if event_type == "finish":
                reasoning_context = event_data["context"]

        return reasoning_context

    async def _generate_reasoning_step(
        self,
        request: ChatCompletionRequest,
        context: dict[str, Any],
        system_prompt: str,
    ) -> ReasoningStep:
        """Generate a single reasoning step using OpenAI structured outputs."""
        # Build conversation history for reasoning
        # TODO: hard coding the last 6 messages for now, should be dynamic
        last_6_messages = '\n'.join([msg.content for msg in request.messages[-6:]])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Original request: {last_6_messages}"},
        ]

        # Add context from previous steps
        if context["steps"]:
            context_summary = "\n".join([
                f"Step {i+1}: {step.thought}"
                for i, step in enumerate(context["steps"])
            ])
            messages.append({
                "role": "assistant",
                "content": f"Previous reasoning:\n\n```\n{context_summary}\n```",
            })

        # Add tool results if available
        if context["tool_results"]:
            tool_summary = "\n".join([
                f"Tool {result.tool_name}: " +
                (f"SUCCESS - {result.result}" if result.success else f"FAILED - {result.error}")
                for result in context["tool_results"]
            ])
            messages.append({
                "role": "assistant",
                "content": f"Tool execution results:\n\n````\n{tool_summary}\n```",
            })

        # Get available tools
        if self.tools:
            tool_descriptions = "\n".join([
                f"- {tool.name}: {tool.description}"
                for tool in self.tools.values()
            ])
            messages.append({
                "role": "assistant",
                "content": f"Available tools:\n\n```\n{tool_descriptions}\n```",
            })
        else:
            messages.append({
                "role": "assistant",
                "content": "No tools are currently available.",
            })

        # Use structured outputs - OpenAI will return properly typed ReasoningStep
        try:
            response = await self.openai_client.beta.chat.completions.parse(
                model=request.model,
                messages=messages,
                response_format=ReasoningStep,
                temperature=0.1,
            )

            if response.choices and response.choices[0].message.parsed:
                # successfully parsed structured output
                return response.choices[0].message.parsed

            # Fallback - create a simple reasoning step indicating failure
            return ReasoningStep(
                thought="Unable to generate structured reasoning step",
                next_action=ReasoningAction.CONTINUE_THINKING,
                tools_to_use=[],
                concurrent_execution=False,
            )
        except Exception as e:
            # Fallback - create a simple reasoning step
            return ReasoningStep(
                thought=f"Error in reasoning - proceeding to final answer: {e!s}",
                next_action=ReasoningAction.FINISHED,
                tools_to_use=[],
                concurrent_execution=False,
            )

    async def _execute_tools_concurrently(
            self,
            tool_predictions: list[ToolPrediction],
        ) -> list[ToolResult]:
        """Execute tool predictions concurrently using asyncio.gather."""
        tasks = []
        for pred in tool_predictions:
            if pred.tool_name not in self.tools:
                # Create a failed result for unknown tools
                task = asyncio.create_task(self._create_failed_result(
                    pred.tool_name, f"Tool '{pred.tool_name}' not found",
                ))
            else:
                tool = self.tools[pred.tool_name]
                task = asyncio.create_task(tool(**pred.arguments))
            tasks.append(task)
        return await asyncio.gather(*tasks)

    async def _execute_tools_sequentially(
            self,
            tool_predictions: list[ToolPrediction],
        ) -> list[ToolResult]:
        """Execute tool predictions sequentially using generic Tool interface."""
        results = []
        for pred in tool_predictions:
            if pred.tool_name not in self.tools:
                result = await self._create_failed_result(
                    pred.tool_name, f"Tool '{pred.tool_name}' not found",
                )
            else:
                tool = self.tools[pred.tool_name]
                result = await tool(**pred.arguments)
            results.append(result)
        return results

    async def _create_failed_result(self, tool_name: str, error_msg: str) -> ToolResult:
        """Create a failed ToolResult for error cases."""
        return ToolResult(
            tool_name=tool_name,
            success=False,
            error=error_msg,
            execution_time_ms=0.0,
        )

    async def get_available_tools(self) -> list[Tool]:
        """Get list of all available tools."""
        return list(self.tools.values())

    async def _synthesize_final_response(
        self,
        request: ChatCompletionRequest,
        reasoning_context: dict[str, Any],
    ) -> ChatCompletionResponse:
        """Synthesize final response using reasoning context."""
        # Get synthesis prompt
        synthesis_prompt = await self.prompt_manager.get_prompt("final_answer")
        # Build synthesis messages
        # TODO: hard coding the last 6 messages for now, should be dynamic
        last_6_messages = '\n'.join([msg.content for msg in request.messages[-6:]])
        messages = [
            {"role": "system", "content": synthesis_prompt},
            {"role": "user", "content": f"Original request: {last_6_messages}"},
        ]
        # Add reasoning summary
        reasoning_summary = self._build_reasoning_summary(reasoning_context)
        messages.append({
            "role": "assistant",
            "content": f"My reasoning process:\n{reasoning_summary}",
        })
        # Generate final response
        response = await self.openai_client.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens,
        )
        # Convert OpenAI response to our Pydantic models

        choices = []
        for choice in response.choices:
            choices.append(Choice(
                index=choice.index,
                message=ChatMessage(
                    role=choice.message.role,
                    content=choice.message.content,
                ),
                finish_reason=choice.finish_reason,
            ))

        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return ChatCompletionResponse(
            id=response.id,
            created=response.created,
            model=response.model,
            choices=choices,
            usage=usage,
        )

    def _build_reasoning_summary(self, context: dict[str, Any]) -> str:
        """Build a summary of the reasoning process for final synthesis."""
        summary_parts = []

        for i, step in enumerate(context["steps"]):
            summary_parts.append(f"Step {i+1}: {step.thought}")
            if step.tools_to_use:
                tool_names = [tool.tool_name for tool in step.tools_to_use]
                summary_parts.append(f"  Used tools: {', '.join(tool_names)}")

        if context["tool_results"]:
            summary_parts.append("\nTool Results:")
            for result in context["tool_results"]:
                if result.success:
                    summary_parts.append(f"- {result.tool_name}: {result.result}")
                else:
                    summary_parts.append(f"- {result.tool_name}: FAILED - {result.error}")

        return "\n".join(summary_parts)

    async def execute_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[str]:
        r"""
        Process a streaming chat completion request with reasoning steps.

        This method implements Server-Sent Events (SSE) streaming, which is the standard
        format used by OpenAI's streaming API. SSE is a web standard that allows a server
        to push data to a client in real-time over a single HTTP connection.

        SSE Format Requirements (mandatory per SSE specification):
        - Each event MUST be prefixed with "data: " followed by the JSON payload
        - Each event MUST end with two newlines ("\n\n") to signal event boundary
        - The final event is always "data: [DONE]\n\n" to signal stream completion
        - These are not optional formatting choices - they are required by the SSE standard

        Example SSE stream:
        ```
        data: {"id": "chatcmpl-123", "choices": [{"delta": {"reasoning_event": {...}}}]}\n\n
        data: {"id": "chatcmpl-123", "choices": [{"delta": {"content": "Hello"}}]}\n\n
        data: {"id": "chatcmpl-123", "choices": [{"delta": {"content": " world"}}]}\n\n
        data: [DONE]\n\n
        ```

        This format allows clients (like web browsers or HTTP clients) to parse the stream
        incrementally and process each reasoning step and content chunk as it arrives.

        Args:
            request: The chat completion request

        Yields:
            Server-sent event formatted strings compatible with OpenAI streaming API.
            Each yielded string is a complete SSE event ready to send to the client.

        Raises:
            ReasoningError: If reasoning process fails
            ValidationError: If request validation fails
            httpx.HTTPStatusError: If the OpenAI API returns an error
        """
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        created = int(time.time())

        # Stream the reasoning process with reasoning_event metadata
        # Each reasoning_chunk is JSON data that must be wrapped in mandatory SSE format
        async for reasoning_chunk in self._stream_reasoning_process(
            request, completion_id, created,
        ):
            # Apply required SSE format: "data: {json}\n\n" (mandated by SSE spec)
            yield f"data: {reasoning_chunk}\n\n"

        # Stream the final synthesized response from OpenAI
        # Each final_chunk is JSON data that must be wrapped in mandatory SSE format
        async for final_chunk in self._stream_final_response(
            request, completion_id, created, self._current_reasoning_context,
        ):
            # Apply required SSE format: "data: {json}\n\n" (mandated by SSE spec)
            yield f"data: {final_chunk}\n\n"

        # Signal end of stream with standard SSE termination event (required by spec)
        yield "data: [DONE]\n\n"

    async def _stream_reasoning_process(
        self,
        request: ChatCompletionRequest,
        completion_id: str,
        created: int,
    ) -> AsyncGenerator[str]:
        """Stream the reasoning process by consuming core events and emitting SSE."""
        # Consume events from core reasoning process and emit as SSE
        async for event_type, event_data in self._core_reasoning_process(request):
            if event_type == "start_step":
                start_event = ReasoningEvent(
                    type=ReasoningEventType.REASONING_STEP,
                    step_id=str(event_data["iteration"]),
                    status=ReasoningEventStatus.IN_PROGRESS,
                    metadata={
                        "tools": [],
                        "thought": f"Starting reasoning step {event_data['iteration']}...",
                    },
                )
                yield self._format_reasoning_event(
                    start_event,
                    completion_id,
                    created,
                    request.model,
                )

            elif event_type == "step_plan":
                reasoning_step = event_data["reasoning_step"]
                plan_event = ReasoningEvent(
                    type=ReasoningEventType.REASONING_STEP,
                    step_id=str(event_data["iteration"]) + "-plan",
                    status=ReasoningEventStatus.IN_PROGRESS,
                    metadata={
                        "tools": [tool.tool_name for tool in reasoning_step.tools_to_use],
                        "thought": reasoning_step.thought,
                        "tools_planned": [tool.tool_name for tool in reasoning_step.tools_to_use],
                    },
                )
                yield self._format_reasoning_event(
                    plan_event,
                    completion_id,
                    created,
                    request.model,
                )

            elif event_type == "start_tools":
                tool_start_event = ReasoningEvent(
                    type=ReasoningEventType.TOOL_EXECUTION,
                    step_id=f"{event_data['iteration']}-tools",
                    status=ReasoningEventStatus.IN_PROGRESS,
                    metadata={
                        "tools": [tool.tool_name for tool in event_data["tools"]],
                        "tool_predictions": event_data["tools"],
                    },
                )
                yield self._format_reasoning_event(
                    tool_start_event,
                    completion_id,
                    created,
                    request.model,
                    )

            elif event_type == "complete_tools":
                tool_complete_event = ReasoningEvent(
                    type=ReasoningEventType.TOOL_EXECUTION,
                    step_id=f"{event_data['iteration']}-tools",
                    status=ReasoningEventStatus.COMPLETED,
                    metadata={
                        "tools": [result.tool_name for result in event_data["tool_results"]],
                        "tool_results": event_data["tool_results"],
                    },
                )
                yield self._format_reasoning_event(
                    tool_complete_event,
                    completion_id,
                    created,
                    request.model,
                )

            elif event_type == "complete_step":
                reasoning_step = event_data["reasoning_step"]
                step_event = ReasoningEvent(
                    type=ReasoningEventType.REASONING_STEP,
                    step_id=str(event_data["iteration"]),
                    status=ReasoningEventStatus.COMPLETED,
                    metadata={
                        "tools": [tool.tool_name for tool in reasoning_step.tools_to_use],
                        "thought": reasoning_step.thought,
                        "had_tools": event_data["had_tools"],
                    },
                )
                yield self._format_reasoning_event(
                    step_event,
                    completion_id,
                    created,
                    request.model,
                )

            elif event_type == "finish":
                reasoning_context = event_data["context"]
                # Store context for final response
                self._current_reasoning_context = reasoning_context

                # Emit reasoning completion
                complete_event = ReasoningEvent(
                    type=ReasoningEventType.SYNTHESIS,
                    step_id="final",
                    status=ReasoningEventStatus.COMPLETED,
                    metadata={
                        "tools": [],
                        "total_steps": len(reasoning_context["steps"]),
                    },
                )
                yield self._format_reasoning_event(
                    complete_event,
                    completion_id,
                    created,
                    request.model,
                )

    def _format_reasoning_event(
        self,
        event: ReasoningEvent,
        completion_id: str,
        created: int,
        model: str,
    ) -> str:
        """Format a reasoning event as a JSON SSE chunk."""
        chunk = ChatCompletionStreamResponse(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                StreamChoice(
                    index=0,
                    delta=Delta(reasoning_event=event),
                    finish_reason=None,
                ),
            ],
        )
        return chunk.model_dump_json()

    async def _stream_final_response(
        self,
        request: ChatCompletionRequest,
        completion_id: str,
        created: int,
        reasoning_context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str]:
        r"""
        Stream the final synthesized response from OpenAI.

        This method calls OpenAI's streaming API and processes the response to:
        1. Parse each SSE chunk from OpenAI (which arrives as "data: {json}\n\n")
        2. Extract the JSON payload from OpenAI's SSE format
        3. Update the completion_id and created timestamp to match our stream
        4. Yield the modified JSON strings (without SSE wrapping)

        Note: OpenAI sends data in SSE format, but we extract just the JSON payload here.
        The calling method (execute_stream) will re-wrap these JSON strings in the
        required SSE format ("data: {json}\n\n") before sending to the client.

        This separation allows for easier testing and cleaner responsibility separation.

        Yields:
            JSON strings representing OpenAI chat completion chunks. The caller
            must wrap these in the mandatory SSE format before sending to clients.
        """
        # Get synthesis prompt and build response
        synthesis_prompt = await self.prompt_manager.get_prompt("final_answer")

        # Build synthesis messages (include reasoning context like non-streaming)
        # TODO: hard coding the last 6 messages for now, should be dynamic
        last_6_messages = '\n'.join([msg.content for msg in request.messages[-6:]])
        messages = [
            {"role": "system", "content": synthesis_prompt},
            {"role": "user", "content": f"Original request: {last_6_messages}"},
        ]

        # Add reasoning summary if available
        if reasoning_context:
            reasoning_summary = self._build_reasoning_summary(reasoning_context)
            messages.append({
                "role": "assistant",
                "content": f"My reasoning process:\n{reasoning_summary}",
            })

        # Stream synthesis response
        payload = {
            "model": request.model,
            "messages": messages,
            "stream": True,
            "temperature": request.temperature or 0.2,
        }

        async with self.http_client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()

            # Process OpenAI's SSE stream line by line
            async for line in response.aiter_lines():
                # OpenAI sends SSE format: "data: {json}" or "data: [DONE]"
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix to get JSON payload

                    # Check for stream termination signal
                    if data == "[DONE]":
                        break

                    try:
                        # Parse OpenAI's JSON chunk and modify it for our stream
                        chunk_data = json.loads(data)

                        # Replace OpenAI's completion_id with ours to maintain consistency
                        # across reasoning events and final response chunks
                        chunk_data["id"] = completion_id
                        chunk_data["created"] = created

                        # Yield the modified JSON (caller will wrap in SSE format)
                        yield json.dumps(chunk_data)
                    except json.JSONDecodeError:
                        # Skip malformed JSON chunks (defensive programming)
                        continue
