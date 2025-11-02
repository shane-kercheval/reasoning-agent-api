"""
Base executor class for all execution paths.

Provides common interface for streaming with disconnection support,
content buffering, SSE conversion, and span management. Subclasses
yield OpenAIStreamResponse objects, base handles SSE conversion and buffering.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable

from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes

from api.openai_protocol import OpenAIChatRequest, OpenAIStreamResponse, SSE_DONE, create_sse


class BaseExecutor(ABC):
    """
    Base class for all execution paths (passthrough, reasoning, orchestration).

    Enforces single-use pattern and centralizes:
    - Client disconnection checking (single location)
    - Content buffering (no SSE roundtrip)
    - SSE conversion
    - Parent span attribute management

    Subclasses implement _execute_stream to yield structured responses.
    """

    def __init__(
        self,
        parent_span: trace.Span | None = None,
        check_disconnected: Callable[[], bool] | None = None,
    ) -> None:
        """
        Initialize executor with request-specific parameters.

        Args:
            parent_span: Optional parent span for setting input/output attributes
            check_disconnected: Optional callback to check client disconnection
        """
        self._content_buffer: list[str] = []
        self._parent_span = parent_span
        self._check_disconnected_callback = check_disconnected
        self._executed = False

    async def execute_stream(
        self,
        request: OpenAIChatRequest,
    ) -> AsyncGenerator[str]:
        """
        Execute streaming request with content buffering (public API).

        This wrapper method:
        1. Enforces single-use pattern
        2. Sets input attributes on parent span
        3. Calls subclass's _execute_stream
        4. Checks for client disconnection (single location!)
        5. Buffers content directly from structured responses (no SSE roundtrip!)
        6. Converts responses to SSE format
        7. Yields SSE formatted chunks
        8. Yields [DONE] marker
        9. Sets output attributes on parent span

        Args:
            request: OpenAI chat request

        Yields:
            SSE formatted chunks

        Raises:
            RuntimeError: If executor is used more than once
            asyncio.CancelledError: If client disconnects during streaming
        """
        if self._executed:
            raise RuntimeError(
                "Executor can only be used once. Create a new instance for each request.",
            )
        self._executed = True

        # Set input attributes on parent span if provided
        if self._parent_span:
            self._set_span_attributes(request, self._parent_span)

        # Stream structured responses from subclass
        async for response in self._execute_stream(request):
            # Check for client disconnection (SINGLE LOCATION - eliminates duplicates!)
            if self._check_disconnected_callback and await self._check_disconnected_callback():
                    raise asyncio.CancelledError("Client disconnected")

            # Buffer content directly from structured response (NO SSE ROUNDTRIP!)
            # Reasoning events never have content, so this naturally excludes them
            # delta.content can be None in finish/usage chunks - defensive checks needed
            if response.choices and len(response.choices) > 0:
                delta = response.choices[0].delta
                if delta and delta.content:
                    self._content_buffer.append(delta.content)

            # Convert to SSE format and yield
            yield create_sse(response)

        # Yield [DONE] marker
        yield SSE_DONE

        # Set output attribute on parent span after streaming completes
        if self._parent_span:
            complete_output = self.get_buffered_content()
            if complete_output:
                self._parent_span.set_attribute(SpanAttributes.OUTPUT_VALUE, complete_output)

    @abstractmethod
    async def _execute_stream(
        self,
        request: OpenAIChatRequest,
    ) -> AsyncGenerator[OpenAIStreamResponse]:
        """
        Execute streaming request and yield OpenAIStreamResponse objects.

        Subclasses must convert LiteLLM chunks using convert_litellm_to_stream_response()
        from api.openai_protocol. Base class handles SSE conversion, buffering, and
        disconnection checking.

        Args:
            request: OpenAI chat request

        Yields:
            OpenAIStreamResponse objects (NOT SSE formatted strings)
        """
        pass

    @abstractmethod
    def _set_span_attributes(
        self,
        request: OpenAIChatRequest,
        span: trace.Span,
    ) -> None:
        """
        Set input and metadata attributes on the provided OTEL span.

        Executor-specific implementation - different executors may set different attributes.

        Args:
            request: The OpenAI chat completion request
            span: The OTEL span to set attributes on
        """
        pass

    def get_buffered_content(self) -> str:
        """Get complete buffered assistant content after streaming."""
        return "".join(self._content_buffer)
