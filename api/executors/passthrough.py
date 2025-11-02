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
from collections.abc import AsyncGenerator

import litellm
from opentelemetry import trace, propagate
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues

from api.config import settings
from api.openai_protocol import (
    OpenAIChatRequest,
    OpenAIStreamResponse,
    convert_litellm_to_stream_response,
)
from api.executors.base import BaseExecutor

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class PassthroughExecutor(BaseExecutor):
    """Executes passthrough streaming with content buffering."""

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
                # Inject trace context into headers for LiteLLM propagation
                carrier: dict[str, str] = {}
                propagate.inject(carrier)

                # Make streaming API call with trace propagation via litellm
                stream = await litellm.acompletion(
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

                    # Convert LiteLLM chunk to OpenAIStreamResponse
                    # Base handles SSE conversion, buffering, disconnection checking
                    yield convert_litellm_to_stream_response(chunk)

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
