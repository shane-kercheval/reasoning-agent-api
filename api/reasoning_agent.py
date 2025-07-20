r"""
Advanced reasoning agent that orchestrates OpenAI JSON mode with tool execution.

This agent implements a sophisticated reasoning process that:
1. Uses OpenAI JSON mode to generate reasoning steps (not structured outputs)
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
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from openai import AsyncOpenAI
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes
from .openai_protocol import (
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIStreamResponse,
    OpenAIStreamChoice,
    OpenAIDelta,
    OpenAIChoice,
    OpenAIMessage,
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

logger = logging.getLogger(__name__)

# Get tracer for reasoning agent instrumentation
tracer = trace.get_tracer(__name__)

DEFAULT_TEMPERATURE = 0.2

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
    Advanced reasoning agent that orchestrates OpenAI JSON mode with tool execution.

    This agent implements the complete reasoning workflow:
    1. Loads reasoning prompts from markdown files
    2. Uses OpenAI JSON mode to generate reasoning steps (not structured outputs)
    3. Orchestrates concurrent tool execution when needed
    4. Streams progress with reasoning_event metadata
    5. Synthesizes final responses
    """

    def __init__(
        self,
        base_url: str,
        tools: list[Tool],
        prompt_manager: PromptManager,
        api_key: str | None = None,
        max_reasoning_iterations: int = 20,
    ):
        """
        Initialize the reasoning agent.

        Args:
            base_url: Base URL for the OpenAI-compatible API
            tools: List of available tools for execution
            prompt_manager: Prompt manager for loading reasoning prompts
            api_key: OpenAI API key for authentication
            max_reasoning_iterations: Maximum number of reasoning iterations to perform
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.tools = {tool.name: tool for tool in tools}
        self.prompt_manager = prompt_manager
        self.reasoning_context = {
                'steps': [],
                'tool_results': [],
                'final_thoughts': '',
                'user_request': None,
            }
        # Initialize OpenAI client - handles authentication and HTTP internally
        self.openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        # Reasoning context
        self.max_reasoning_iterations = max_reasoning_iterations

    async def execute(
        self,
        request: OpenAIChatRequest,
        parent_span: trace.Span | None = None,
    ) -> OpenAIChatResponse:
        """
        Process a non-streaming chat completion request with full reasoning.

        This method uses execute_stream internally and collects only the final
        response content chunks, ignoring reasoning events. This ensures a single
        source of truth for response generation logic.

        Args:
            request: The chat completion request
            parent_span: Optional parent span for tracing

        Returns:
            OpenAI-compatible chat completion response with reasoning applied

        Raises:
            ReasoningError: If reasoning process fails
            ValidationError: If request validation fails
            httpx.HTTPStatusError: If the OpenAI API returns an error
        """
        # Use execute_stream internally and collect only final response chunks
        collected_content = []
        collected_choices = []
        final_response_data = None

        # Call execute_stream with stream=False to get unified tracing
        async for sse_chunk in self.execute_stream(request, parent_span, is_streaming=False):
            # Skip SSE formatting and [DONE] marker
            if sse_chunk == "data: [DONE]\n\n":
                break
            if not sse_chunk.startswith("data: "):
                continue

            # Extract JSON from SSE format
            json_data = sse_chunk[6:-2]  # Remove "data: " prefix and "\n\n" suffix

            try:
                stream_response = OpenAIStreamResponse.model_validate_json(json_data)

                # Only process chunks that contain actual content (not reasoning events)
                if (stream_response.choices and
                    len(stream_response.choices) > 0 and
                    stream_response.choices[0].delta.content is not None):

                    # Store the response metadata from the first content chunk
                    if final_response_data is None:
                        final_response_data = {
                            "id": stream_response.id,
                            "created": stream_response.created,
                            "model": stream_response.model,
                        }

                    # Collect content
                    content = stream_response.choices[0].delta.content
                    collected_content.append(content)

                    # Store the choice structure (we'll use the last one for finish_reason)
                    collected_choices = stream_response.choices

            except Exception as e:
                # Skip malformed chunks - this shouldn't happen in normal operation
                logger.warning(f"Failed to parse streaming chunk: {e}")
                continue

        if not final_response_data:
            raise ReasoningError("No final response content received from streaming")

        # Build the final response using collected data
        complete_content = "".join(collected_content)

        # Create the final choice with complete content
        final_choice = OpenAIChoice(
            index=0,
            message=OpenAIMessage(
                role="assistant",
                content=complete_content,
            ),
            finish_reason=collected_choices[0].finish_reason if collected_choices else "stop",
        )

        response = OpenAIChatResponse(
            id=final_response_data["id"],
            created=final_response_data["created"],
            model=final_response_data["model"],
            choices=[final_choice],
            usage=None,  # Usage is not available in streaming mode
        )

        # Set output attribute on parent span if provided
        if parent_span and response.choices:
            output_content = response.choices[0].message.content or ""
            parent_span.set_attribute(SpanAttributes.OUTPUT_VALUE, output_content)

        return response

    async def execute_stream(
        self,
        request: OpenAIChatRequest,
        parent_span: trace.Span | None = None,
        is_streaming: bool = True,
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
            parent_span: Optional parent span for tracing
            is_streaming: Whether the response is streaming to end user (only used for tracing)

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

        # Set input and metadata attributes on parent span if provided
        if parent_span:
            self._set_span_attributes(request, parent_span)

        with tracer.start_as_current_span(
            "reasoning_agent.execute",
            attributes={
                "reasoning.model": request.model,
                "reasoning.message_count": len(request.messages),
                "reasoning.stream": is_streaming,
                "reasoning.max_tokens": request.max_tokens or 0,
                "reasoning.temperature": request.temperature or DEFAULT_TEMPERATURE,
            },
        ) as span:
            chunk_count = 0

            # Stream the reasoning process with reasoning_event metadata
            # Each reasoning_chunk is JSON data that must be wrapped in mandatory SSE format
            async for reasoning_chunk in self._stream_reasoning_process(
                request, completion_id, created,
            ):
                chunk_count += 1
                # Apply required SSE format: "data: {json}\n\n" (mandated by SSE spec)
                yield f"data: {reasoning_chunk}\n\n"

            if not self.reasoning_context:
                raise ReasoningError("Reasoning process failed - no context generated")

            # Stream the final synthesized response from OpenAI
            # Each final_chunk is JSON data that must be wrapped in mandatory SSE format
            # Collect content for OUTPUT_VALUE attribute
            collected_content = []
            async for final_chunk in self._stream_final_response(
                request, completion_id, created, self.reasoning_context,
            ):
                chunk_count += 1

                # Extract content from chunk for OUTPUT_VALUE using Pydantic model
                stream_response = OpenAIStreamResponse.model_validate_json(final_chunk)
                if stream_response.choices and len(stream_response.choices) > 0:
                    choice = stream_response.choices[0]
                    if choice.delta.content:
                        collected_content.append(choice.delta.content)

                # Apply required SSE format: "data: {json}\n\n" (mandated by SSE spec)
                yield f"data: {final_chunk}\n\n"

            # Add streaming metrics to span
            span.set_attribute("reasoning.chunks_sent", chunk_count)
            span.set_attribute(
                "reasoning.steps_count",
                len(self.reasoning_context.get("steps", [])),
            )
            span.set_attribute(
                "reasoning.tool_calls_count",
                len(self.reasoning_context.get("tool_results", [])),
            )

            # Set output attribute on parent span if provided (for streaming)
            # Use the actual streamed response content
            if parent_span:
                if collected_content:
                    # Join all the content chunks to get the complete response
                    complete_response = "".join(collected_content)
                    parent_span.set_attribute(SpanAttributes.OUTPUT_VALUE, complete_response)
                else:
                    # This should not happen in normal operation - it indicates a problem
                    error_msg = "No content collected from streaming response for OUTPUT_VALUE"
                    logger.error(error_msg)
                    raise ReasoningError(error_msg)

            # Signal end of stream with standard SSE termination event (required by spec)
            yield "data: [DONE]\n\n"

    async def get_available_tools(self) -> list[Tool]:
        """Get list of all available tools."""
        return list(self.tools.values())

    async def _core_reasoning_process(
        self,
        request: OpenAIChatRequest,
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
        self.reasoning_context['user_request'] = request
        with tracer.start_as_current_span("reasoning_process") as span:
            # Get reasoning system prompt
            system_prompt = await self.prompt_manager.get_prompt("reasoning_system")

            for iteration in range(self.max_reasoning_iterations):
                # Create a span for each reasoning step/iteration
                with tracer.start_as_current_span(
                    f"reasoning_step_{iteration + 1}",
                    attributes={
                        "reasoning.step_number": iteration + 1,
                        "reasoning.step_id": f"step_{iteration + 1}",
                    },
                ) as step_span:
                    # Yield start of reasoning step
                    yield ("start_step", {"iteration": iteration + 1})

                    # Generate next reasoning step using structured outputs
                    reasoning_step = await self._generate_reasoning_step(
                        request,
                        self.reasoning_context,
                        system_prompt,
                    )
                    self.reasoning_context["steps"].append(reasoning_step)

                    # Add step details to span
                    # Truncate long thoughts
                    step_span.set_attribute("reasoning.step_thought", reasoning_step.thought[:500])
                    step_span.set_attribute("reasoning.step_action", reasoning_step.next_action.value)  # noqa: E501
                    step_span.set_attribute("reasoning.tools_planned", len(reasoning_step.tools_to_use))  # noqa: E501
                    if reasoning_step.tools_to_use:
                        step_span.set_attribute(
                            "reasoning.tool_names",
                            [tool.tool_name for tool in reasoning_step.tools_to_use],
                        )

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
                        self.reasoning_context["tool_results"].extend(tool_results)

                        # Add tool results to step span
                        step_span.set_attribute("reasoning.tools_executed", len(tool_results))
                        successful_tools = sum(1 for r in tool_results if r.success)
                        step_span.set_attribute("reasoning.tools_successful", successful_tools)
                        step_span.set_attribute(
                            "reasoning.tools_failed",
                            len(tool_results) - successful_tools,
                        )

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

                    # Mark step as finished
                    if reasoning_step.next_action == ReasoningAction.FINISHED:
                        step_span.set_attribute("reasoning.step_final", True)

                    # Check if we should continue reasoning
                    if reasoning_step.next_action == ReasoningAction.FINISHED:
                        break

            # Add final metrics to reasoning span
            span.set_attribute("reasoning.iterations_completed", iteration + 1)
            span.set_attribute(
                "reasoning.steps_total",
                len(self.reasoning_context["steps"]),
            )
            span.set_attribute(
                "reasoning.tools_total",
                len(self.reasoning_context["tool_results"]),
            )
            # Yield final context
            yield ("finish", {"context": self.reasoning_context})

    @staticmethod
    def get_content_from_message(msg: dict[str, Any] | OpenAIMessage) -> str | None:
        """Get content from a message, handling both dict and OpenAIMessage."""
        if hasattr(msg, 'content'):
            return msg.content
        if isinstance(msg, dict):
            return msg.get('content')
        return getattr(msg, 'content', None)

    async def _generate_reasoning_step(
        self,
        request: OpenAIChatRequest,
        context: dict[str, Any],
        system_prompt: str,
    ) -> ReasoningStep:
        """Generate a single reasoning step using OpenAI JSON mode."""
        # Build conversation history for reasoning
        # TODO: hard coding the last 6 messages for now, should be dynamic
        # Handle both dict messages and Pydantic OpenAIMessage objects
        last_6_messages = '\n'.join([
            self.get_content_from_message(msg) for msg in request.messages[-6:]
            if self.get_content_from_message(msg) is not None
        ])
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

        # Add JSON schema instructions to the system prompt
        json_schema = ReasoningStep.model_json_schema()
        schema_instructions = f"""

You must respond with valid JSON that matches this exact schema:

```json
{json.dumps(json_schema, indent=2)}
```

Your response must be valid JSON only, no other text.
"""
        # Update the last message to include schema instructions
        messages[-1]["content"] += schema_instructions

        # Request reasoning step using JSON mode
        try:
            response = await self.openai_client.chat.completions.create(
                model=request.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=request.temperature or DEFAULT_TEMPERATURE,
            )
            error_message = None
            if response.choices and response.choices[0].message.content:
                try:
                    json_response = json.loads(response.choices[0].message.content)
                    return ReasoningStep.model_validate(json_response)
                except (json.JSONDecodeError, ValueError) as parse_error:
                    logger.warning(f"Failed to parse JSON response: {parse_error}")
                    logger.warning(f"Raw response: {response.choices[0].message.content}")
                    error_message = str(parse_error)
                    error_message += f" (raw response: {response.choices[0].message.content})"

            # Fallback - create a simple reasoning step for malformed responses
            logger.warning(f"Unexpected response format: {response}")
            thought = "Unable to generate structured reasoning step"
            if error_message:
                thought += f" - Error: {error_message}"
            return ReasoningStep(
                thought=thought,
                next_action=ReasoningAction.CONTINUE_THINKING,
                tools_to_use=[],
                concurrent_execution=False,
            )
        except httpx.HTTPStatusError as http_error:
            # OpenAI API errors (auth, rate limits, invalid model, etc.)
            # Log the error but keep the original fallback behavior for reasoning step generation
            logger.error(f"OpenAI API error during reasoning: {http_error}")
            return ReasoningStep(
                thought=f"OpenAI API error: {http_error.response.status_code} - proceeding to final answer",  # noqa: E501
                next_action=ReasoningAction.FINISHED,
                tools_to_use=[],
                concurrent_execution=False,
            )
        except Exception as e:
            # Fallback - create a simple reasoning step for unexpected errors
            logger.warning(f"Unexpected error during reasoning: {e}")
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
        with tracer.start_as_current_span(
            "tools.execute_concurrent",
            attributes={
                "tool.execution_mode": "concurrent",
                "tool.count": len(tool_predictions),
                "tool.names": [pred.tool_name for pred in tool_predictions],
            },
        ) as tools_span:
            start_time = time.time()

            tasks = []
            for pred in tool_predictions:
                if pred.tool_name not in self.tools:
                    # Create a failed result for unknown tools
                    task = asyncio.create_task(self._create_failed_result(
                        pred.tool_name, f"Tool '{pred.tool_name}' not found",
                    ))
                else:
                    tool = self.tools[pred.tool_name]
                    # Wrap individual tool execution in its own span
                    task = asyncio.create_task(self._execute_single_tool_with_tracing(tool, pred))
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            # Add execution metrics
            duration_ms = (time.time() - start_time) * 1000
            tools_span.set_attribute("tool.duration_ms", duration_ms)
            success_count = sum(1 for r in results if r.success)
            failure_count = sum(1 for r in results if not r.success)
            tools_span.set_attribute("tool.success_count", success_count)
            tools_span.set_attribute("tool.failure_count", failure_count)

            return results

    async def _execute_tools_sequentially(
            self,
            tool_predictions: list[ToolPrediction],
        ) -> list[ToolResult]:
        """Execute tool predictions sequentially using generic Tool interface."""
        with tracer.start_as_current_span(
            "tools.execute_sequential",
            attributes={
                "tool.execution_mode": "sequential",
                "tool.count": len(tool_predictions),
                "tool.names": [pred.tool_name for pred in tool_predictions],
            },
        ) as tools_span:
            start_time = time.time()
            results = []

            for pred in tool_predictions:
                if pred.tool_name not in self.tools:
                    result = await self._create_failed_result(
                        pred.tool_name, f"Tool '{pred.tool_name}' not found",
                    )
                else:
                    tool = self.tools[pred.tool_name]
                    result = await self._execute_single_tool_with_tracing(tool, pred)
                results.append(result)

            # Add execution metrics
            duration_ms = (time.time() - start_time) * 1000
            tools_span.set_attribute("tool.duration_ms", duration_ms)
            success_count = sum(1 for r in results if r.success)
            failure_count = sum(1 for r in results if not r.success)
            tools_span.set_attribute("tool.success_count", success_count)
            tools_span.set_attribute("tool.failure_count", failure_count)

            return results

    async def _execute_single_tool_with_tracing(
        self,
        tool: Tool,
        prediction: ToolPrediction,
    ) -> ToolResult:
        """Execute a single tool with tracing instrumentation."""
        with tracer.start_as_current_span(
            f"tool.{prediction.tool_name}",
            attributes={
                "tool.name": prediction.tool_name,
                "tool.input": str(prediction.arguments),
            },
        ) as tool_span:
            start_time = time.time()

            try:
                result = await tool(**prediction.arguments)

                # Add result attributes
                tool_span.set_attribute("tool.success", result.success)
                tool_span.set_attribute("tool.duration_ms", result.execution_time_ms)

                if result.success:
                    # Truncate long outputs
                    tool_span.set_attribute(
                        "tool.output", str(result.result)[:1000],
                    )
                    tool_span.set_status(trace.Status(trace.StatusCode.OK))
                else:
                    tool_span.set_attribute("tool.error", result.error or "Unknown error")
                    tool_span.set_status(trace.Status(trace.StatusCode.ERROR, result.error or "Tool execution failed"))  # noqa: E501

                return result

            except Exception as e:
                # Handle exceptions in tool execution
                tool_span.record_exception(e)
                tool_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))

                # Create a failed result
                duration_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    tool_name=prediction.tool_name,
                    success=False,
                    error=str(e),
                    execution_time_ms=duration_ms,
                )

    async def _create_failed_result(self, tool_name: str, error_msg: str) -> ToolResult:
        """Create a failed ToolResult for error cases."""
        return ToolResult(
            tool_name=tool_name,
            success=False,
            error=error_msg,
            execution_time_ms=0.0,
        )


    @staticmethod
    def _build_reasoning_summary(context: dict[str, Any]) -> str:
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

    async def _stream_reasoning_process(
        self,
        request: OpenAIChatRequest,
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
                # Emit reasoning completion
                complete_event = ReasoningEvent(
                    type=ReasoningEventType.SYNTHESIS,
                    step_id="final",
                    status=ReasoningEventStatus.COMPLETED,
                    metadata={
                        "tools": [],
                        "total_steps": len(self.reasoning_context["steps"]),
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
        chunk = OpenAIStreamResponse(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                OpenAIStreamChoice(
                    index=0,
                    delta=OpenAIDelta(reasoning_event=event),
                    finish_reason=None,
                ),
            ],
        )
        return chunk.model_dump_json()

    async def _stream_final_response(
        self,
        request: OpenAIChatRequest,
        completion_id: str,
        created: int,
        reasoning_context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str]:
        r"""
        Stream the final synthesized response from OpenAI.

        This is the single source of truth for final response generation, used by both
        streaming and non-streaming execution paths. This method:

        1. Builds synthesis messages using reasoning context
        2. Calls OpenAI's streaming API
        3. Processes each SSE chunk from OpenAI
        4. Updates completion_id and created timestamp to match our stream
        5. Yields modified JSON strings (without SSE wrapping)

        The calling method (execute_stream) wraps these JSON strings in the required
        SSE format ("data: {json}\n\n"), while execute() collects and assembles
        the content chunks into a complete non-streaming response.

        This unified approach eliminates code duplication and ensures both execution
        paths use identical response generation logic.

        Yields:
            JSON strings representing OpenAI chat completion chunks. The caller
            must wrap these in the mandatory SSE format before sending to clients.
        """
        # Get synthesis prompt and build response
        synthesis_prompt = await self.prompt_manager.get_prompt("final_answer")

        # Build synthesis messages (include reasoning context like non-streaming)
        # TODO: hard coding the last 6 messages for now, should be dynamic
        # Handle both dict messages and Pydantic OpenAIMessage objects
        last_6_messages = '\n'.join([
            self.get_content_from_message(msg) for msg in request.messages[-6:]
            if self.get_content_from_message(msg) is not None
        ])
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

        # Stream synthesis response using properly authenticated OpenAI client
        try:
            stream = await self.openai_client.chat.completions.create(
                model=request.model,
                messages=messages,
                stream=True,
                temperature=request.temperature or DEFAULT_TEMPERATURE,
            )
        except httpx.HTTPStatusError as http_error:
            # OpenAI API errors during streaming should be propagated immediately
            # These never come through the SSE stream - they interrupt it before starting
            logger.error(f"OpenAI API error during streaming synthesis: {http_error}")
            raise ReasoningError(
                f"OpenAI API error during streaming synthesis: {http_error.response.status_code} {http_error.response.text}",  # noqa: E501
                details={"http_status": http_error.response.status_code, "response": http_error.response.text},  # noqa: E501
            ) from http_error
        except Exception as e:
            # Unexpected errors (network, timeout, etc.)
            logger.error(f"Unexpected error during streaming synthesis: {e}")
            raise ReasoningError(f"Unexpected error during streaming synthesis: {e}") from e

        # Process OpenAI's streaming response
        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                # Replace OpenAI's completion_id with ours to maintain consistency
                # across reasoning events and final response chunks
                chunk.id = completion_id
                chunk.created = created

                # Yield the modified JSON (caller will wrap in SSE format)
                yield chunk.model_dump_json()

    def _set_span_attributes(self, request: OpenAIChatRequest, span: trace.Span) -> None:
        """
        Set input and metadata attributes on the provided span.

        This method sets the attributes that are displayed in Phoenix UI columns:
        - INPUT_VALUE: The user's request content
        - METADATA: Additional context about the request
        """
        # Set input value from the last user message
        user_messages = [
            msg for msg in request.messages
            if getattr(msg, 'role', msg.get('role')) == 'user'
        ]
        if user_messages:
            last_user_message = user_messages[-1]
            content = self.get_content_from_message(last_user_message)
            if content:
                span.set_attribute(SpanAttributes.INPUT_VALUE, content)

        # Set metadata with request details
        metadata = {
            "model": request.model,
            "temperature": request.temperature or DEFAULT_TEMPERATURE,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
            "message_count": len(request.messages),
            "tools_available": len(self.tools),
        }
        span.set_attribute(SpanAttributes.METADATA, json.dumps(metadata))
