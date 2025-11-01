"""
Base executor class for all execution paths.

Provides common interface for streaming with disconnection support,
content buffering, and shared SSE content extraction.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable

from opentelemetry import trace

from api.openai_protocol import OpenAIChatRequest


def extract_content_from_sse_chunk(chunk: str) -> str | None:
    r"""
    Extract content delta from SSE chunk.

    Shared by all executors for content buffering.
    Returns None if chunk is malformed or contains no content.

    Args:
        chunk: SSE formatted chunk string (e.g., "data: {...}\\n\\n")

    Returns:
        Content string if found, None otherwise

    Examples:
        >>> chunk = 'data: {"choices": [{"delta": {"content": "Hello"}}]}\\n\\n'
        >>> extract_content_from_sse_chunk(chunk)
        'Hello'

        >>> extract_content_from_sse_chunk('data: [DONE]\\n\\n')
        None
    """
    if not (isinstance(chunk, str) and "data: " in chunk):
        return None

    try:
        data_line = chunk.strip().replace("data: ", "")
        if data_line and data_line != "[DONE]":
            chunk_data = json.loads(data_line)
            if choices := chunk_data.get("choices"):
                delta = choices[0].get("delta", {})
                return delta.get("content")
    except (json.JSONDecodeError, KeyError, IndexError):
        pass
    return None


class BaseExecutor(ABC):
    """
    Base class for all execution paths (passthrough, reasoning, orchestration).

    Provides:
    - Common interface for streaming with disconnection support
    - Content buffering during streaming
    - Shared SSE content extraction
    """

    def __init__(self) -> None:
        """Initialize base executor with empty content buffer."""
        self._content_buffer: list[str] = []

    @abstractmethod
    async def execute_stream(
        self,
        request: OpenAIChatRequest,
        parent_span: trace.Span | None = None,
        check_disconnected: Callable[[], bool] | None = None,
    ) -> AsyncGenerator[str]:
        """
        Execute streaming request with content buffering.

        Args:
            request: OpenAI chat request
            parent_span: Optional parent span for tracing
            check_disconnected: Optional callback to check client disconnection

        Yields:
            SSE formatted chunks

        Raises:
            asyncio.CancelledError: If client disconnects during streaming
        """
        pass

    def get_buffered_content(self) -> str:
        """Get complete buffered assistant content after streaming."""
        return "".join(self._content_buffer)

    def _reset_buffer(self) -> None:
        """Reset content buffer for new request."""
        self._content_buffer = []

    def _buffer_chunk(self, chunk: str) -> None:
        """Buffer content from SSE chunk."""
        if content := extract_content_from_sse_chunk(chunk):
            self._content_buffer.append(content)

    async def _check_disconnection(
        self,
        check_disconnected: Callable[[], bool] | None,
    ) -> None:
        """
        Check if client disconnected and raise CancelledError if so.

        Should be called before each yield in execute_stream.

        Args:
            check_disconnected: Callback that returns True if client disconnected

        Raises:
            asyncio.CancelledError: If client has disconnected
        """
        if check_disconnected and await check_disconnected():
            raise asyncio.CancelledError("Client disconnected")
