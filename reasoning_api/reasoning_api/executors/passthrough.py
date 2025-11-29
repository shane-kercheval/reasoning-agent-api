"""
Passthrough executor for direct OpenAI API calls.

This module provides fast, low-latency responses for straightforward queries
that don't require reasoning or multi-agent orchestration. It maintains full
OpenAI API compatibility while preserving tracing, error handling, and authentication.

Passthrough path characteristics:
- Direct litellm calls (no reasoning layer, no orchestration)
- Streaming-only architecture (clients can collect all chunks if needed)
- OpenTelemetry tracing integration
- Proper error forwarding from OpenAI API
- Client disconnection handling for streaming
- Default execution path (matches OpenAI experience)
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Callable

import litellm
from opentelemetry import trace, propagate
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues

from reasoning_api.config import settings
from reasoning_api.openai_protocol import (
    OpenAIChatRequest,
    OpenAIStreamResponse,
    OpenAIStreamChoice,
    OpenAIDelta,
    convert_litellm_to_stream_response,
)
from reasoning_api.reasoning_models import ReasoningEvent, ReasoningEventType
from reasoning_api.executors.base import BaseExecutor
from reasoning_api.conversation_utils import build_metadata_from_response
from reasoning_api.context_manager import ContextManager, Context

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class PassthroughExecutor(BaseExecutor):
    """Executes passthrough streaming with content buffering."""

    def __init__(
        self,
        context_manager: ContextManager | None = None,
        parent_span: trace.Span | None = None,
        check_disconnected: Callable[[], bool] | None = None,
    ) -> None:
        """
        Initialize passthrough executor.

        Args:
            context_manager: ContextManager for managing LLM context windows.
                If not provided, creates one with default FULL utilization.
            parent_span: Optional parent span for tracing
            check_disconnected: Optional callback to check client disconnection
        """
        super().__init__(parent_span, check_disconnected)
        self.context_manager = context_manager or ContextManager()
        # Set routing path once at initialization
        self.accumulate_metadata({"routing_path": "passthrough"})

        # Reasoning content buffering state
        self._reasoning_buffer: list[str] = []
        self._reasoning_active = False
        self._reasoning_event_sent = False

    async def _execute_stream(
        self,
        request: OpenAIChatRequest,
    ) -> AsyncGenerator[OpenAIStreamResponse]:
        """
        Execute a streaming chat completion via direct LLM API call.

        Converts LiteLLM chunks to OpenAIStreamResponse objects. Base class handles
        SSE conversion, buffering, disconnection checking, and span management.

        Args:
            request: The OpenAI chat completion request

        Yields:
            OpenAIStreamResponse objects (converted from LiteLLM chunks)

        Raises:
            asyncio.CancelledError: If client disconnects
            litellm.APIError: If LLM API returns an error
        """
        with tracer.start_as_current_span(
            "passthrough.execute_stream",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                "llm.model": request.model,
                "llm.message_count": len(request.messages),
                "llm.max_tokens": request.max_tokens or 0,
                "llm.temperature": request.temperature or 1.0,
                "routing.path": "passthrough",
            },
        ) as span:
            span.set_status(trace.Status(trace.StatusCode.OK))

            try:
                # Apply context management BEFORE litellm call
                context = Context(conversation_history=request.messages)
                filtered_messages, context_metadata = self.context_manager(
                    model_name=request.model,
                    context=context,
                )

                # Inject trace context into headers for LiteLLM propagation
                carrier: dict[str, str] = {}
                propagate.inject(carrier)

                # Make streaming API call with trace propagation via litellm
                # Use filtered messages from context manager
                stream = await litellm.acompletion(
                    model=request.model,
                    messages=filtered_messages,  # Context-managed messages
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    n=request.n,
                    stop=request.stop,
                    presence_penalty=request.presence_penalty,
                    frequency_penalty=request.frequency_penalty,
                    logit_bias=request.logit_bias,
                    user=request.user,
                    reasoning_effort=request.reasoning_effort,
                    stream=True,
                    stream_options={"include_usage": True},  # Request usage data
                    extra_headers=carrier,  # Propagate trace context to LiteLLM
                    api_key=settings.llm_api_key,
                    base_url=settings.llm_base_url,
                )

                # Stream chunks - base class handles SSE conversion and buffering
                async for chunk in stream:
                    # Track usage if provided (use getattr for safe access)
                    chunk_usage = getattr(chunk, 'usage', None)
                    if chunk_usage:
                        span.set_attribute("llm.token_count.prompt", chunk_usage.prompt_tokens)
                        span.set_attribute(
                            "llm.token_count.completion",
                            chunk_usage.completion_tokens,
                        )
                        span.set_attribute("llm.token_count.total", chunk_usage.total_tokens)

                        # Accumulate metadata for storage (usage, cost, model, context)
                        metadata = build_metadata_from_response(chunk)
                        metadata["context_utilization"] = context_metadata
                        self.accumulate_metadata(metadata)

                    response_chunk = convert_litellm_to_stream_response(chunk)

                    # Add context metadata to usage chunk for client visibility
                    if chunk_usage and response_chunk.usage:
                        response_chunk.usage.context_utilization = context_metadata
                        logger.debug(
                            f"[Passthrough] Added context_utilization to usage chunk: "
                            f"{context_metadata}",
                        )
                    # Handle reasoning_content buffering for models that support it
                    if response_chunk.choices:
                        delta = response_chunk.choices[0].delta
                        # Check for reasoning_content in delta (provider-specific field)
                        # Use getattr for safe access since not all models have this field
                        has_reasoning = False
                        reasoning_content = None
                        if hasattr(delta, 'reasoning_content'):
                            reasoning_content = getattr(delta, 'reasoning_content', None)
                            has_reasoning = (
                                reasoning_content is not None
                                and reasoning_content != ""
                            )

                        has_content = delta.content is not None and delta.content != ""

                        # Buffer reasoning content
                        if has_reasoning:
                            self._reasoning_buffer.append(reasoning_content)
                            self._reasoning_active = True
                            continue  # Don't emit this chunk, buffer it

                        # Transition: reasoning → content
                        # Emit EXTERNAL_REASONING event when reasoning completes
                        should_emit = (
                            self._reasoning_active
                            and has_content
                            and not self._reasoning_event_sent
                        )
                        if should_emit:
                            # Create reasoning event with buffered content
                            reasoning_event = ReasoningEvent(
                                type=ReasoningEventType.EXTERNAL_REASONING,
                                step_iteration=1,
                                metadata={
                                    "thought": "".join(self._reasoning_buffer),
                                },
                            )
                            yield OpenAIStreamResponse(
                                id=response_chunk.id,
                                object="chat.completion.chunk",
                                created=response_chunk.created,
                                model=response_chunk.model,
                                choices=[
                                    OpenAIStreamChoice(
                                        index=0,
                                        delta=OpenAIDelta(reasoning_event=reasoning_event),
                                        finish_reason=None,
                                    ),
                                ],
                            )
                            self._reasoning_event_sent = True
                            self._reasoning_active = False

                    yield response_chunk

            except asyncio.CancelledError:
                # Cancellation is expected behavior - mark span and re-raise
                # Re-raising allows main.py to set http.cancelled consistently
                span.set_attribute("stream.cancelled", True)
                span.set_attribute("cancellation.reason", "Client disconnected")
                raise

    def _set_span_attributes(
        self,
        request: OpenAIChatRequest,
        span: trace.Span,
    ) -> None:
        """
        Set input and metadata attributes on the provided span.

        Args:
            request: The OpenAI chat completion request
            span: The span to set attributes on
        """
        # Extract last user message for input attribute
        user_messages = [
            msg for msg in request.messages
            if msg.get("role") == "user"
        ]

        if user_messages:
            last_user_content = user_messages[-1].get("content", "")
            if last_user_content:
                span.set_attribute(SpanAttributes.INPUT_VALUE, last_user_content)

        # Set metadata
        metadata = {
            "model": request.model,
            "temperature": request.temperature or 1.0,
            "max_tokens": request.max_tokens,
            "stream": True,  # Passthrough always streams
            "message_count": len(request.messages),
            "routing_path": "passthrough",
        }
        span.set_attribute(SpanAttributes.METADATA, json.dumps(metadata))
