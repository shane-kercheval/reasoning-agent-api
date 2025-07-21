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
    OpenAIChatResponse,
    ErrorResponse,
)
from api.reasoning_agent import ReasoningAgent
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


@pytest.fixture
def mock_openai_response() -> OpenAIChatResponse:
    """
    Mock OpenAI API response - DEPRECATED: Use simple_openai_response from
    fixtures.responses.
    """
    return create_simple_response("This is a test response from OpenAI.", "chatcmpl-test123")  # noqa: F405


@pytest.fixture
def mock_openai_streaming_chunks() -> list[str]:
    """
    Mock OpenAI streaming response chunks - DEPRECATED: Use streaming_chunks from
    fixtures.responses.
    """
    return create_streaming_response("This is a test", "chatcmpl-test123").split('\n\n')[:-1]  # noqa: F405


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

        yield ReasoningAgent(
            base_url="https://api.openai.com/v1",
            api_key="test-api-key",
            tools=tools,
            prompt_manager=mock_prompt_manager,
        )



@pytest.fixture
def mock_openai_error_response() -> ErrorResponse:
    """Mock OpenAI API error response - DEPRECATED: Use error_response from fixtures.responses."""
    return ErrorResponse(
        error=ErrorDetail(  # noqa: F405
            message="Invalid API key provided",
            type="invalid_request_error",
            code="invalid_api_key",
        ),
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
