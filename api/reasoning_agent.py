"""
Reasoning agent.

This agent implements a reasoning process that:
1. Uses OpenAI JSON mode to generate reasoning steps
2. Orchestrates concurrent tool execution when needed
3. Streams reasoning progress with enhanced metadata via Server-Sent Events (SSE)
4. Maintains full OpenAI API compatibility
5. Propagates OpenTelemetry trace context to LiteLLM for distributed tracing

Distributed Tracing:
All LLM API calls (reasoning step generation and final synthesis) propagate trace context
to LiteLLM using OpenTelemetry's W3C TraceContext standard. The pattern used is:
  carrier: dict[str, str] = {}
  propagate.inject(carrier)  # Injects traceparent headers into carrier dict
  openai_client.create(..., extra_headers=carrier)  # Passes trace headers to LiteLLM
This ensures all LLM calls are correlated with the parent request span in observability tools.

┌─────────────────────────────────────────────────────────────────────┐
│                  REASONING AGENT FLOW (STREAMING-ONLY)              │
└─────────────────────────────────────────────────────────────────────┘

                        User Request
                            │
                            ▼
                  ┌────────────────────┐
                  │   execute_stream   │
                  └────────────────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ _core_reasoning_   │ ← SINGLE SOURCE OF TRUTH
                  │    _process()      │   (Yields OpenAIStreamResponse)
                  └────────────────────┘
                            │
                ┌─────────────────────────┐
                │  OPENAI RESPONSES       │
                │                         │
                │ Reasoning Events:       │
                │ ├─ ITERATION_START      │
                │ ├─ PLANNING             │ ← Contains OpenAIUsage
                │ ├─ TOOL_EXECUTION_START │
                │ ├─ TOOL_RESULT          │
                │ ├─ ITERATION_COMPLETE   │
                │ └─ REASONING_COMPLETE   │
                │                         │
                │ Final Synthesis:        │
                │ ├─ delta.content        │ ← Multiple streaming
                │ ├─ ...                  │   chunks, last chunk
                │ └─ finish_reason        │   contains OpenAIUsage
                └─────────────────────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Wrap in SSE Format │
                  │                    │
                  │ • create_sse()     │
                  │   each chunk       │
                  │ • yield directly   │
                  └────────────────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │   SSE RESPONSE     │
                  │                    │
                  │ data: {chunk1}     │
                  │ data: {chunk2}     │
                  │ data: [DONE]       │
                  └────────────────────┘

EVENT TYPES:
- ITERATION_START: Beginning of a reasoning step
- PLANNING: Generated reasoning plan with thought and tool decisions (contains OpenAIUsage)
- TOOL_EXECUTION_START: Starting tool execution
- TOOL_RESULT: Tool execution completed with results
- ITERATION_COMPLETE: Reasoning step finished
- REASONING_COMPLETE: All reasoning iterations completed, starting final synthesis
- ERROR: Error occurred during reasoning or tool execution

USAGE TRACKING (OpenAIUsage objects):
- PLANNING events contain usage data from reasoning step generation
- Final synthesis content chunks contain usage data from the synthesis API call
- Usage data is streamed in real-time with each event
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
from opentelemetry import trace, propagate
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues

from .openai_protocol import (
    SSE_DONE,
    OpenAIUsage,
    create_sse,
    OpenAIChatRequest,
    OpenAIStreamResponse,
    OpenAIStreamChoice,
    OpenAIDelta,
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
    Advanced reasoning agent with streaming-only architecture.

    This agent implements the complete reasoning workflow:
    1. Loads reasoning prompts from markdown files
    2. Uses OpenAI JSON mode to generate reasoning steps
    3. Orchestrates concurrent tool execution when needed
    4. Streams progress with reasoning_event metadata
    5. Synthesizes final responses

    Key architectural principle: _core_reasoning_process is the single source
    of truth, yielding OpenAIStreamResponse objects for reasoning events and
    final synthesis content. All responses are streamed.
    """

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        tools: list[Tool],
        prompt_manager: PromptManager,
        max_reasoning_iterations: int = 20,
    ):
        """
        Initialize the reasoning agent.

        Args:
            openai_client: Shared AsyncOpenAI client for LiteLLM proxy
            tools: List of available tools for execution
            prompt_manager: Prompt manager for loading reasoning prompts
            max_reasoning_iterations: Maximum number of reasoning iterations to perform
        """
        self.openai_client = openai_client
        self.tools = {tool.name: tool for tool in tools}
        self.prompt_manager = prompt_manager
        self.reasoning_context = {
            "steps": [],
            "tool_results": [],
            "final_thoughts": "",
            "user_request": None,
        }

        # Reasoning context
        self.max_reasoning_iterations = max_reasoning_iterations

    async def execute_stream(
        self,
        request: OpenAIChatRequest,
        parent_span: trace.Span | None = None,
    ) -> AsyncGenerator[str]:
        """
        Process a chat completion request with full reasoning (streaming-only).

        This is the only execution path for the reasoning agent. Wraps
        OpenAIStreamResponse objects from _core_reasoning_process in SSE format.

        Args:
            request: The chat completion request
            parent_span: Optional parent span for tracing

        Yields:
            Server-sent event formatted strings compatible with OpenAI streaming API.
            Each yielded string is a complete SSE event ready to send to the client.

        Raises:
            ReasoningError: If reasoning process fails
            ValidationError: If request validation fails
            httpx.HTTPStatusError: If the OpenAI API returns an error
        """
        chunk_count = 0
        collected_content = []

        # Wrap _core_reasoning_process output in SSE format
        async for response in self._core_reasoning_process(
            request, parent_span=parent_span,
        ):
            chunk_count += 1

            # Extract content for output tracing
            choice = response.choices[0]
            if choice.delta.content:
                collected_content.append(choice.delta.content)

            # Wrap in SSE format and yield
            yield create_sse(response)

        # Signal end of stream with standard SSE termination event (required by spec)
        yield SSE_DONE

    async def get_available_tools(self) -> list[Tool]:
        """Get list of all available tools."""
        return list(self.tools.values())

    async def _core_reasoning_process(  # noqa: PLR0915
        self,
        request: OpenAIChatRequest,
        parent_span: trace.Span | None = None,
    ) -> AsyncGenerator[OpenAIStreamResponse]:
        """
        Complete reasoning process yielding OpenAI-compatible stream responses.

        This is the single source of truth for ALL reasoning logic including:
        - Reasoning step generation and streaming
        - Tool execution and result streaming
        - Final synthesis streaming
        - Tracing and context management

        Yields OpenAIStreamResponse objects with either:
        - delta.reasoning_event populated (reasoning steps)
        - delta.content populated (final synthesis)
        """
        self.reasoning_context["user_request"] = request
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        created = int(time.time())

        # Set input and metadata attributes on parent span if provided
        if parent_span:
            self._set_span_attributes(request, parent_span)

        # Track collected content for output span attribute
        collected_output_content = []

        with tracer.start_as_current_span(
            "reasoning_agent.execute_stream",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
                "reasoning.model": request.model,
                "reasoning.message_count": len(request.messages),
                "reasoning.max_tokens": request.max_tokens or 0,
                "reasoning.temperature": request.temperature or DEFAULT_TEMPERATURE,
            },
        ) as execute_span:
            execute_span.set_status(trace.Status(trace.StatusCode.OK))
            # Get reasoning system prompt
            system_prompt = await self.prompt_manager.get_prompt("reasoning_system")

            for iteration in range(self.max_reasoning_iterations):
                # Create a span for each reasoning step/iteration
                with tracer.start_as_current_span(
                    f"reasoning_step_{iteration + 1}",
                    attributes={
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,  # noqa: E501
                        "reasoning.step_number": iteration + 1,
                        "reasoning.step_id": f"step_{iteration + 1}",
                    },
                ) as step_span:
                    step_span.set_status(trace.Status(trace.StatusCode.OK))
                    # Yield start of reasoning step
                    yield self._create_reasoning_response(
                        ReasoningEvent(
                            type=ReasoningEventType.ITERATION_START,
                            step_iteration=iteration + 1,
                            metadata={
                                "tools": [],
                            },
                        ),
                        completion_id,
                        created,
                        request.model,
                    )

                    # Generate next reasoning step using structured outputs
                    reasoning_step, step_usage = await self._generate_reasoning_step(
                        request,
                        self.reasoning_context,
                        system_prompt,
                    )
                    self.reasoning_context["steps"].append(reasoning_step)
                    # Yield reasoning step plan (what we plan to do) - includes usage
                    yield self._create_reasoning_response(
                        ReasoningEvent(
                            type=ReasoningEventType.PLANNING,
                            step_iteration=iteration + 1,
                            metadata={
                                "tools": [tool.tool_name for tool in reasoning_step.tools_to_use],
                                "thought": reasoning_step.thought,
                                "tools_planned": [tool.tool_name for tool in reasoning_step.tools_to_use],  # noqa: E501
                            },
                        ),
                        completion_id,
                        created,
                        request.model,
                        step_usage,
                    )

                    # Add step details to span
                    step_span.set_attribute("reasoning.step_thought", reasoning_step.thought[:500])
                    step_span.set_attribute(
                        "reasoning.step_action",
                        reasoning_step.next_action.value,
                    )
                    step_span.set_attribute(
                        "reasoning.tools_planned",
                        len(reasoning_step.tools_to_use),
                    )
                    if reasoning_step.tools_to_use:
                        step_span.set_attribute(
                            "reasoning.tool_names",
                            [tool.tool_name for tool in reasoning_step.tools_to_use],
                        )

                    # Execute tools if needed
                    if reasoning_step.tools_to_use:
                        # Yield start of tool execution
                        yield self._create_reasoning_response(
                            ReasoningEvent(
                                type=ReasoningEventType.TOOL_EXECUTION_START,
                                step_iteration=iteration + 1,
                                metadata={
                                    "tools": [tool.tool_name for tool in reasoning_step.tools_to_use],  # noqa: E501
                                    "tool_predictions": reasoning_step.tools_to_use,
                                    "concurrent_execution": reasoning_step.concurrent_execution,
                                },
                            ),
                            completion_id,
                            created,
                            request.model,
                        )

                        if reasoning_step.concurrent_execution:
                            tool_results = await self._execute_tools_concurrently(
                                reasoning_step.tools_to_use,
                            )
                        else:
                            tool_results = await self._execute_tools_sequentially(
                                reasoning_step.tools_to_use,
                            )
                        self.reasoning_context["tool_results"].extend(tool_results)
                        # Yield completed tool execution
                        yield self._create_reasoning_response(
                            ReasoningEvent(
                                type=ReasoningEventType.TOOL_RESULT,
                                step_iteration=iteration + 1,
                                metadata={
                                    "tools": [result.tool_name for result in tool_results],
                                    "tool_results": tool_results,
                                },
                            ),
                            completion_id,
                            created,
                            request.model,
                        )

                        # Add tool results to step span
                        step_span.set_attribute("reasoning.tools_executed", len(tool_results))
                        successful_tools = sum(1 for r in tool_results if r.success)
                        step_span.set_attribute("reasoning.tools_successful", successful_tools)
                        step_span.set_attribute(
                            "reasoning.tools_failed",
                            len(tool_results) - successful_tools,
                        )

                        if not tool_results:
                            step_span.set_status(trace.Status(
                                trace.StatusCode.ERROR,
                                f"No tool results despite {len(reasoning_step.tools_to_use)} tools planned",  # noqa: E501
                            ))
                        elif all(not r.success for r in tool_results):
                            step_span.set_status(trace.Status(
                                trace.StatusCode.ERROR,
                                f"All {len(tool_results)} tools failed in this step",
                            ))

                    # Yield step completion (after tools are done)
                    yield self._create_reasoning_response(
                        ReasoningEvent(
                            type=ReasoningEventType.ITERATION_COMPLETE,
                            step_iteration=iteration + 1,
                            metadata={
                                "tools": [tool.tool_name for tool in reasoning_step.tools_to_use],
                                "had_tools": bool(reasoning_step.tools_to_use),
                            },
                        ),
                        completion_id,
                        created,
                        request.model,
                    )


                    # Mark step as finished
                    if reasoning_step.next_action == ReasoningAction.FINISHED:
                        step_span.set_attribute("reasoning.step_final", True)
                        break

            # Add final metrics to reasoning span
            execute_span.set_attribute("reasoning.iterations_completed", iteration + 1)
            execute_span.set_attribute(
                "reasoning.steps_total",
                len(self.reasoning_context["steps"]),
            )
            execute_span.set_attribute(
                "reasoning.tools_total",
                len(self.reasoning_context["tool_results"]),
            )

            # Yield reasoning completion
            yield self._create_reasoning_response(
                ReasoningEvent(
                    type=ReasoningEventType.REASONING_COMPLETE,
                    step_iteration=0,  # Not tied to a specific iteration
                    metadata={
                        "tools": [],
                        "total_steps": len(self.reasoning_context["steps"]),
                    },
                ),
                completion_id,
                created,
                request.model,
            )

            # Now yield final synthesis stream - integrated into the same generator!
            async for synthesis_response in self._stream_final_synthesis(
                request, completion_id, created, self.reasoning_context,
            ):
                # Collect content for output span attribute
                choice = synthesis_response.choices[0]
                if choice.delta.content:
                    collected_output_content.append(choice.delta.content)

                yield synthesis_response

            # Set output attribute on parent span (where input/metadata are set)
            complete_output = "".join(collected_output_content)
            if parent_span:
                parent_span.set_attribute(SpanAttributes.OUTPUT_VALUE, complete_output)

    def _create_reasoning_response(
        self,
        event: ReasoningEvent,
        completion_id: str,
        created: int,
        model: str,
        usage: OpenAIUsage | None = None,
    ) -> OpenAIStreamResponse:
        """Create an OpenAIStreamResponse for a reasoning event."""
        return OpenAIStreamResponse(
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
            usage=usage,
        )

    async def _stream_final_synthesis(
        self,
        request: OpenAIChatRequest,
        completion_id: str,
        created: int,
        reasoning_context: dict[str, Any],
    ) -> AsyncGenerator[OpenAIStreamResponse]:
        """
        Stream the final synthesized response from OpenAI as OpenAIStreamResponse objects.

        This method handles the final synthesis step of the reasoning process,
        integrated directly into _core_reasoning_process for unified flow.
        """
        # Get synthesis prompt and build response
        synthesis_prompt = await self.prompt_manager.get_prompt("final_answer")

        # Build synthesis messages
        last_6_messages = "\n".join([
            self.get_content_from_message(msg) for msg in request.messages[-6:]
            if self.get_content_from_message(msg) is not None
        ])
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

        # Inject trace context into headers for LiteLLM propagation
        carrier: dict[str, str] = {}
        propagate.inject(carrier)

        # Stream synthesis response using properly authenticated OpenAI client
        try:
            stream = await self.openai_client.chat.completions.create(
                model=request.model,
                messages=messages,
                stream=True,
                temperature=request.temperature or DEFAULT_TEMPERATURE,
                stream_options={"include_usage": True},  # Request usage data in stream
                extra_headers=carrier,  # Propagate trace context to LiteLLM
            )
        except httpx.HTTPStatusError as http_error:
            logger.error(f"OpenAI API error during streaming synthesis: {http_error}")
            raise ReasoningError(
                f"OpenAI API error during streaming synthesis: {http_error.response.status_code} {http_error.response.text}",  # noqa: E501
                details={
                    "http_status": http_error.response.status_code,
                    "response": http_error.response.text,
                },
            ) from http_error
        except Exception as e:
            logger.error(f"Unexpected error during streaming synthesis: {e}")
            raise ReasoningError(f"Unexpected error during streaming synthesis: {e}") from e

        # Process OpenAI's streaming response and convert to our format
        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                # Convert OpenAI chunk to our OpenAIStreamResponse format
                choice = chunk.choices[0]

                # Convert usage if present
                usage = None
                if chunk.usage:
                    usage = OpenAIUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    )

                # Create our response with consistent ID and timestamp
                yield OpenAIStreamResponse(
                    id=completion_id,
                    created=created,
                    model=chunk.model,
                    choices=[
                        OpenAIStreamChoice(
                            index=0,
                            delta=OpenAIDelta(
                                role=choice.delta.role,
                                content=choice.delta.content,
                                tool_calls=choice.delta.tool_calls,
                            ),
                            finish_reason=choice.finish_reason,
                        ),
                    ],
                    usage=usage,
                )

    @staticmethod
    def get_content_from_message(msg: dict[str, Any] | OpenAIMessage) -> str | None:
        """Get content from a message, handling both dict and OpenAIMessage."""
        if hasattr(msg, "content"):
            return msg.content
        if isinstance(msg, dict):
            return msg.get("content")
        return getattr(msg, "content", None)

    async def _generate_reasoning_step(
        self,
        request: OpenAIChatRequest,
        context: dict[str, Any],
        system_prompt: str,
    ) -> tuple[ReasoningStep, OpenAIUsage | None]:
        """Generate a single reasoning step using OpenAI JSON mode."""
        # Build conversation history for reasoning
        last_6_messages = "\n".join([
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
                "content": f"Tool execution results:\n\n```\n{tool_summary}\n```",
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

        # Add JSON schema instructions
        json_schema = ReasoningStep.model_json_schema()
        schema_instructions = f"""

You must respond with valid JSON that matches this exact schema:

```json
{json.dumps(json_schema, indent=2)}
```

Your response must be valid JSON only, no other text.
"""
        messages[-1]["content"] += schema_instructions

        # Inject trace context into headers for LiteLLM propagation
        carrier: dict[str, str] = {}
        propagate.inject(carrier)

        # Request reasoning step using JSON mode
        try:
            response = await self.openai_client.chat.completions.create(
                model=request.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=request.temperature or DEFAULT_TEMPERATURE,
                extra_headers=carrier,  # Propagate trace context to LiteLLM
            )

            if response.choices and response.choices[0].message.content:
                try:
                    json_response = json.loads(response.choices[0].message.content)
                    # Convert OpenAI usage to our format
                    usage = None
                    if response.usage:
                        usage = OpenAIUsage(
                            prompt_tokens=response.usage.prompt_tokens,
                            completion_tokens=response.usage.completion_tokens,
                            total_tokens=response.usage.total_tokens,
                        )
                    return ReasoningStep.model_validate(json_response), usage
                except (json.JSONDecodeError, ValueError) as parse_error:
                    logger.warning(f"Failed to parse JSON response: {parse_error}")
                    logger.warning(f"Raw response: {response.choices[0].message.content}")
                    current_span = trace.get_current_span()
                    if current_span:
                        current_span.set_status(trace.Status(
                            trace.StatusCode.ERROR,
                            f"Failed to parse JSON response: {parse_error}",
                        ))

            # Fallback - create a simple reasoning step for malformed responses
            logger.warning(f"Unexpected response format: {response}")
            return ReasoningStep(
                thought="Unable to generate structured reasoning step - proceeding to final answer",  # noqa: E501
                next_action=ReasoningAction.FINISHED,
                tools_to_use=[],
                concurrent_execution=False,
            ), OpenAIUsage(
                prompt_tokens=response.usage.prompt_tokens if response and response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response and response.usage else 0,  # noqa: E501
                total_tokens=response.usage.total_tokens if response and response.usage else 0,
            )

        except httpx.HTTPStatusError as http_error:
            logger.error(f"OpenAI API error during reasoning: {http_error}")

            current_span = trace.get_current_span()
            if current_span:
                current_span.set_status(trace.Status(
                    trace.StatusCode.ERROR,
                    f"OpenAI API error: {http_error.response.status_code}",
                ))

            return ReasoningStep(
                thought=f"OpenAI API error: {http_error.response.status_code} - proceeding to final answer",  # noqa: E501
                next_action=ReasoningAction.FINISHED,
                tools_to_use=[],
                concurrent_execution=False,
            ), None
        except Exception as e:
            logger.warning(f"Unexpected error during reasoning: {e}")
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_status(trace.Status(
                    trace.StatusCode.ERROR,
                    f"Unexpected error during reasoning: {e}",
                ))

            return ReasoningStep(
                thought=f"Error in reasoning - proceeding to final answer: {e!s}",
                next_action=ReasoningAction.FINISHED,
                tools_to_use=[],
                concurrent_execution=False,
            ), None

    async def _execute_tools_concurrently(
        self,
        tool_predictions: list[ToolPrediction],
    ) -> list[ToolResult]:
        """Execute tool predictions concurrently using asyncio.gather."""
        with tracer.start_as_current_span(
            "tools.execute_concurrent",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                "tool.execution_mode": "concurrent",
                "tool.count": len(tool_predictions),
                "tool.names": [pred.tool_name for pred in tool_predictions],
            },
        ) as tools_span:
            tools_span.set_status(trace.Status(trace.StatusCode.OK))
            start_time = time.time()

            tasks = []
            for pred in tool_predictions:
                if pred.tool_name not in self.tools:
                    task = asyncio.create_task(self._create_failed_result(
                        pred.tool_name, f"Tool '{pred.tool_name}' not found",
                    ))
                else:
                    tool = self.tools[pred.tool_name]
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

            if failure_count == len(results):
                tools_span.set_status(trace.Status(
                    trace.StatusCode.ERROR,
                    f"All {len(results)} tools failed",
                ))

            return results

    async def _execute_tools_sequentially(
        self,
        tool_predictions: list[ToolPrediction],
    ) -> list[ToolResult]:
        """Execute tool predictions sequentially."""
        with tracer.start_as_current_span(
            "tools.execute_sequential",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                "tool.execution_mode": "sequential",
                "tool.count": len(tool_predictions),
                "tool.names": [pred.tool_name for pred in tool_predictions],
            },
        ) as tools_span:
            tools_span.set_status(trace.Status(trace.StatusCode.OK))
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

            if failure_count == len(results):
                tools_span.set_status(trace.Status(
                    trace.StatusCode.ERROR,
                    f"All {len(results)} tools failed",
                ))

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
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
                SpanAttributes.TOOL_NAME: prediction.tool_name,
                SpanAttributes.TOOL_PARAMETERS: json.dumps(prediction.arguments),
            },
        ) as tool_span:
            # On Error, returns ToolResult with success=False
            result = await tool(**prediction.arguments)
            # Add result attributes
            tool_span.set_attribute("tool.success", result.success)
            tool_span.set_attribute("tool.duration_ms", result.execution_time_ms)
            if result.success:
                tool_span.set_status(trace.Status(trace.StatusCode.OK))
                tool_span.set_attribute(
                    SpanAttributes.OUTPUT_VALUE, str(result.result)[:1000],
                )
            else:
                tool_span.set_status(trace.Status(trace.StatusCode.ERROR, result.error or "Tool execution failed"))  # noqa: E501
                tool_span.set_attribute("tool.error", result.error or "Unknown error")

            return result


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
            if getattr(msg, "role", msg.get("role")) == "user"
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
            "message_count": len(request.messages),
            "tools_available": len(self.tools),
        }
        span.set_attribute(SpanAttributes.METADATA, json.dumps(metadata))
