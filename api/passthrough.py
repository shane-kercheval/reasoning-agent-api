"""
Passthrough path implementation for direct OpenAI API calls.

This module provides fast, low-latency responses for straightforward queries
that don't require reasoning or multi-agent orchestration. It maintains full
OpenAI API compatibility while preserving tracing, error handling, and authentication.

Passthrough path characteristics:
- Direct AsyncOpenAI client calls (no reasoning layer, no orchestration)
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

from openai import AsyncOpenAI
from opentelemetry import trace, propagate
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues

from .openai_protocol import (
    OpenAIChatRequest,
    create_sse,
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


async def execute_passthrough_stream(
    request: OpenAIChatRequest,
    openai_client: AsyncOpenAI,
    parent_span: trace.Span | None = None,
    check_disconnected: Callable[[], bool] | None = None,
) -> AsyncGenerator[str]:
    """
    Execute a streaming chat completion via direct OpenAI API call.

    This is the only execution path for passthrough - always streams.
    Clients can collect all chunks if they want non-streaming behavior.

    Args:
        request: The OpenAI chat completion request
        openai_client: Shared AsyncOpenAI client for LiteLLM proxy
        parent_span: Optional parent span for tracing
        check_disconnected: Async callable that returns True if client disconnected

    Yields:
        SSE-formatted strings containing OpenAI streaming chunks

    Raises:
        asyncio.CancelledError: If client disconnects
        httpx.HTTPStatusError: If OpenAI API returns an error
    """
    # Set input attributes on parent span if provided
    if parent_span:
        _set_span_attributes(request, parent_span)

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

        # Track collected output for tracing
        collected_content = []

        try:
            # Inject trace context into headers for LiteLLM propagation
            carrier: dict[str, str] = {}
            propagate.inject(carrier)

            # Make streaming API call with trace propagation
            stream = await openai_client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                n=request.n,
                stop=request.stop,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                logit_bias=request.logit_bias,
                user=request.user,
                stream=True,
                stream_options={"include_usage": True},  # Request usage data
                extra_headers=carrier,  # Propagate trace context to LiteLLM
            )

            # Stream chunks and convert to SSE format
            async for chunk in stream:
                # Check for client disconnection if callable provided
                if check_disconnected and await check_disconnected():
                    logger.info("Client disconnected during passthrough streaming")
                    raise asyncio.CancelledError("Client disconnected")

                # Collect content for output tracing
                if chunk.choices and chunk.choices[0].delta.content:
                    collected_content.append(chunk.choices[0].delta.content)

                # Track usage if provided
                if chunk.usage:
                    span.set_attribute("llm.token_count.prompt", chunk.usage.prompt_tokens)
                    span.set_attribute("llm.token_count.completion", chunk.usage.completion_tokens)
                    span.set_attribute("llm.token_count.total", chunk.usage.total_tokens)

                # Convert chunk to SSE format
                # OpenAI SDK returns pydantic models, convert to dict for SSE
                chunk_dict = chunk.model_dump()
                yield create_sse(chunk_dict)

            # Set output attribute on parent span
            complete_output = "".join(collected_content)
            if parent_span and complete_output:
                parent_span.set_attribute(SpanAttributes.OUTPUT_VALUE, complete_output)

        except asyncio.CancelledError:
            # Cancellation is expected behavior
            span.set_attribute("stream.cancelled", True)
            span.set_attribute("cancellation.reason", "Client disconnected")
            # Don't re-raise - let stream end gracefully
            return


def _set_span_attributes(
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
