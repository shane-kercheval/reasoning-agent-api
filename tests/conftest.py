"""
Test configuration and fixtures for the reasoning agent API.

Provides common test fixtures and mock configurations for testing
the FastAPI application and ReasoningAgent with proper HTTP mocking.

This module now imports centralized fixtures from the fixtures/ package
while maintaining backward compatibility with existing tests.
"""

import os
from pathlib import Path

# Remove OTEL environment variable that triggers automatic background batch processors
# This must happen before any imports that might trigger OpenTelemetry initialization
# This fixes timeout issues that occur when OpenTelemetry tries to export spans and phoenix is
# unavailable
#
# CONTEXT: This prevents OpenTelemetry from automatically setting up background span processors
# that would attempt to export spans even when we're not explicitly testing tracing functionality.
# Without this, tests can hang for 30+ seconds waiting for OTLP exports to timeout.
os.environ.pop('OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE', None)

# NUCLEAR OPTION: Completely disable OpenTelemetry auto-instrumentation for tests
# This prevents automatic instrumentation of OpenAI clients which can cause persistent
# global state that contaminates tests across test sessions
os.environ['OTEL_SDK_DISABLED'] = 'true'

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
import json
import logging
import pytest
import pytest_asyncio
import httpx
from unittest.mock import AsyncMock
from tests.utils.phoenix_helpers import (
    phoenix_environment,
    mock_settings,
    reset_global_tracer_provider,  # Used by tracing_enabled fixture for cleanup
)
from api.openai_protocol import (
    OpenAIChatRequest,
)
from api.reasoning_agent import ReasoningAgent
from api.reasoning_models import ReasoningEventType
from api.prompt_manager import PromptManager
from api.tools import function_to_tool
from api.tracing import setup_tracing

# Import all centralized fixtures to make them available globally
from tests.fixtures.tools import *  # noqa: F403
from tests.fixtures.agents import *  # noqa: F403
from tests.fixtures.models import *  # noqa: F403
from tests.fixtures.requests import *  # noqa: F403
from tests.fixtures.responses import *  # noqa: F403


OPENAI_TEST_MODEL = "gpt-4o-mini"

logger = logging.getLogger(__name__)


@dataclass
class ReasoningAgentStreamingCollector:
    """
    Collect and categorize data from SSE streaming responses from ReasoningAgent.

    This helper processes SSE (Server-Sent Events) streaming chunks and categorizes
    them for easy testing assertions. It handles both content and reasoning events,
    with fail-fast error handling suitable for test environments.

    Recommended Usage:
        collector = StreamingCollector()
        await collector.process(reasoning_agent.execute_stream(request))
        assert "tokyo" in collector.content.lower()
        assert len(collector.tool_start_events) > 0

    Alternative (line-by-line control):
        collector = StreamingCollector()
        async for line in stream:
            collector.process_line(line)
            # Custom per-line processing here if needed
    """

    all_chunks: list[str] = field(default_factory=list)
    content_parts: list[str] = field(default_factory=list)
    reasoning_events: list[dict] = field(default_factory=list)
    tool_start_events: list[dict] = field(default_factory=list)
    tool_complete_events: list[dict] = field(default_factory=list)

    def process_line(self, line: str) -> None:
        """
        Process a single SSE line and categorize its data.

        This method uses fail-fast error handling - if JSON parsing fails,
        the test will fail immediately. This is intentional for test code
        to catch malformed responses early.

        Args:
            line: SSE formatted line to process
        """
        # Skip empty lines (SSE keep-alives)
        if not line.strip():
            return

        # Skip non-data lines
        if not line.startswith("data: "):
            return

        # Handle end marker
        if line.startswith("data: [DONE]"):
            self.all_chunks.append(line)
            return

        # Extract JSON payload
        json_str = line[6:].strip()
        if not json_str:  # Empty data line
            return

        # Parse JSON - no try/except, let it raise if malformed (fail-fast for tests)
        chunk_data = json.loads(json_str)
        self.all_chunks.append(line)

        # Use .get() for optional fields to avoid KeyError
        if not chunk_data.get("choices"):
            return

        choice = chunk_data["choices"][0]
        delta = choice.get("delta", {})

        # Collect content
        if delta.get("content"):
            self.content_parts.append(delta["content"])

        # Collect reasoning events
        if delta.get("reasoning_event"):
            event = delta["reasoning_event"]
            self.reasoning_events.append(event)

            # Categorize tool events
            event_type = event.get("type")
            if event_type == ReasoningEventType.TOOL_EXECUTION_START.value:
                self.tool_start_events.append(event)
            elif event_type == ReasoningEventType.TOOL_RESULT.value:
                self.tool_complete_events.append(event)

    async def process(self, stream: AsyncGenerator[str]) -> None:
        """
        Process all lines from an async stream.

        This is the recommended way to use StreamingCollector - pass the entire
        async generator and let it handle iteration internally.

        Args:
            stream: Async generator yielding SSE lines

        Example:
            collector = StreamingCollector()
            await collector.process(reasoning_agent.execute_stream(request))
            assert "tokyo" in collector.content.lower()
        """
        async for line in stream:
            self.process_line(line)

    @property
    def content(self) -> str:
        """Get full content as a single string."""
        return "".join(self.content_parts)

# Fast timeout environment variables to prevent tests from hanging
# PERFORMANCE CRITICAL: Uses 1-second timeouts instead of OpenTelemetry's
# default 30-second timeouts to prevent tests from hanging when Phoenix
# or OTLP endpoints are unavailable during testing
FAST_TRACING_TIMEOUTS = {
    'OTEL_EXPORTER_OTLP_TIMEOUT': '1',  # 1-second timeout vs 30-second default
    'OTEL_BSP_EXPORT_TIMEOUT': '1000',  # 1 second in milliseconds
    'OTEL_BSP_SCHEDULE_DELAY': '100',   # 100ms delay vs longer defaults
}

# Fast timeouts + SQLite mode for tracing tests
FAST_TRACING_TIMEOUTS_WITH_SQLITE = {
    **FAST_TRACING_TIMEOUTS,
    'PHOENIX_COLLECTOR_ENDPOINT': '',  # Empty string forces SQLite mode
}


@pytest.fixture
def sample_chat_request() -> OpenAIChatRequest:
    """Sample OpenAI-compatible chat request."""
    return OpenAIChatRequest(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "What's the weather in Paris?"},
        ],
        temperature=0.7,
        max_tokens=150,
    )


@pytest.fixture
def sample_streaming_request() -> OpenAIChatRequest:
    """Sample streaming chat request."""
    return OpenAIChatRequest(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Search for recent news about AI"},
        ],
        stream=True,
        temperature=0.7,
    )


@pytest_asyncio.fixture
async def http_client() -> AsyncGenerator[httpx.AsyncClient]:
    """HTTP client for testing with proper cleanup."""
    async with httpx.AsyncClient() as client:
        yield client


@pytest_asyncio.fixture
async def reasoning_agent() -> AsyncGenerator[ReasoningAgent]:
    """
    ReasoningAgent instance for testing with mock tools.

    Note: OpenTelemetry is disabled by default for tests (OTEL_SDK_DISABLED=true)
    so this fixture no longer needs tracing cleanup.
    After migration to litellm, ReasoningAgent no longer needs an injected client.
    """
    async with httpx.AsyncClient():
        # Import centralized tools
        # Create mock tools
        tools = [
            function_to_tool(weather_tool),  # noqa: F405
            function_to_tool(search_tool),  # noqa: F405
        ]

        # Create mock prompt manager
        mock_prompt_manager = AsyncMock(spec=PromptManager)
        mock_prompt_manager.get_prompt.return_value = "Test system prompt"

        # ReasoningAgent now uses litellm.acompletion() - no client needed
        yield ReasoningAgent(
            tools=tools,
            prompt_manager=mock_prompt_manager,
        )


# =============================================================================
# PHOENIX TESTING FIXTURES
# =============================================================================

@pytest.fixture
def phoenix_sqlite_test():
    """
    Create temporary SQLite Phoenix instance for unit tests.

    Each test gets a fresh Phoenix instance using SQLite backend
    in an isolated temporary directory with fast timeouts to prevent
    test slowdowns.

    PERFORMANCE OPTIMIZATION: Uses 1-second timeouts instead of default 30-second
    timeouts to prevent tests from hanging when Phoenix/OTLP endpoints are unavailable.

    Yields:
        str: Path to temporary directory where Phoenix stores SQLite data.
    """
    with phoenix_environment(**FAST_TRACING_TIMEOUTS) as temp_dir:
        yield temp_dir


@pytest.fixture
def tracing_enabled():
    """
    Temporarily enable tracing for tests.

    This fixture re-enables OpenTelemetry (which is disabled by default for tests)
    and configures fast timeouts with SQLite mode for optimal test performance.

    PERFORMANCE OPTIMIZATION:
    - Uses 1-second timeouts instead of 30-second defaults to prevent test hangs
    - Forces SQLite mode to avoid external dependencies during testing
    - Automatically restores disabled state after test completion
    """
    # Store original OTEL_SDK_DISABLED state and temporarily enable OpenTelemetry for this test
    original_otel_disabled = os.environ.get('OTEL_SDK_DISABLED')
    os.environ.pop('OTEL_SDK_DISABLED', None)  # Remove to enable OpenTelemetry

    # Set fast timeouts and force SQLite mode for tests
    original_env = {}

    # Store and set environment variables
    for key, value in FAST_TRACING_TIMEOUTS_WITH_SQLITE.items():
        if key in os.environ:
            original_env[key] = os.environ[key]
        os.environ[key] = value

    try:
        with mock_settings(enable_tracing=True, phoenix_collector_endpoint=''):
            # Setup tracing once here with SQLite mode
            setup_tracing(enabled=True, project_name="test-reasoning", endpoint='')
            yield
    finally:
        # Restore environment variables
        for key, original_value in original_env.items():
            os.environ[key] = original_value
        for key in FAST_TRACING_TIMEOUTS_WITH_SQLITE:
            if key not in original_env:
                os.environ.pop(key, None)

        # Clean up tracer provider
        reset_global_tracer_provider()

        # Restore original OTEL_SDK_DISABLED state to disable OpenTelemetry again
        if original_otel_disabled is not None:
            os.environ['OTEL_SDK_DISABLED'] = original_otel_disabled
        else:
            os.environ['OTEL_SDK_DISABLED'] = 'true'


@pytest.fixture
def empty_mcp_config(tmp_path: Path) -> str:
    """
    Create empty MCP configuration to prevent connection failures during tests.

    Returns:
        Path to empty MCP config file.
    """
    empty_config = tmp_path / "empty_mcp_config.json"
    empty_config.write_text('{}')  # Empty JSON object
    return str(empty_config)
