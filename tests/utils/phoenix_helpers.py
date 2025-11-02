"""
Phoenix test utilities for testing tracing integration.

Provides helper functions for testing Phoenix SQLite instances,
trace validation, and environment management.
"""

import os
import glob
import tempfile
import sqlite3
from typing import Any
from contextlib import contextmanager
from unittest.mock import patch
from opentelemetry import trace
from opentelemetry.util._once import Once
from opentelemetry.trace import NoOpTracerProvider
from api.config import settings
from api.openai_protocol import OpenAIResponseBuilder


def reset_global_tracer_provider():
    """
    Reset OpenTelemetry's global tracer provider to prevent cross-test contamination.

    PROBLEM SOLVED: OpenTelemetry's global tracer provider persists across tests.
    Once a test sets up tracing (like our integration tests), all subsequent tests
    inherit that tracer provider and attempt to export spans, causing:
    - Tests to hang for 30+ seconds waiting for OTLP export timeouts
    - False failures due to network connection attempts to invalid endpoints
    - 10-20x slower test execution for non-tracing tests

    SOLUTION: Reset the global state and install a NoOp provider that doesn't
    attempt any span exports, ensuring tests run fast and independently.

    This function should be called after any test that might set up tracing.
    """
    # Reset the global tracer provider by clearing OpenTelemetry's internal state
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = Once()

    # Install a NoOp provider that doesn't export spans, preventing timeouts
    trace.set_tracer_provider(NoOpTracerProvider())


def get_trace_count(working_dir: str) -> int:
    """
    Get number of traces in SQLite Phoenix database.

    Args:
        working_dir: Phoenix working directory containing SQLite files.

    Returns:
        Number of traces found (simplified validation).
    """
    # Look for Phoenix SQLite database files
    db_files = glob.glob(f"{working_dir}/**/*.db", recursive=True)

    if not db_files:
        return 0

    # Simple validation - if database exists and has data, assume traces were created
    # This is intentionally light validation as per milestone requirements
    total_traces = 0
    for db_file in db_files:
        try:
            with sqlite3.connect(db_file) as conn:
                cursor = conn.cursor()
                # Check if spans table exists and has data
                cursor.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='spans'",
                )
                if cursor.fetchone()[0] > 0:
                    cursor.execute("SELECT COUNT(*) FROM spans")
                    total_traces += cursor.fetchone()[0]
        except sqlite3.Error:
            # If we can't read the database, assume no traces
            continue

    return total_traces


def has_phoenix_database(working_dir: str) -> bool:
    """
    Check if Phoenix has created a database in the working directory.

    Args:
        working_dir: Phoenix working directory to check.

    Returns:
        True if Phoenix database files exist.
    """
    db_files = glob.glob(f"{working_dir}/**/*.db", recursive=True)
    return len(db_files) > 0


@contextmanager
def mock_settings(**overrides: Any):
    """
    Context manager to temporarily override settings for testing.

    Args:
        **overrides: Settings attributes to override.

    Example:
        with mock_settings(enable_tracing=True):
            # Test with tracing enabled
            pass
    """
    original_values = {}

    # Store original values
    for key, value in overrides.items():
        if hasattr(settings, key):
            original_values[key] = getattr(settings, key)
            setattr(settings, key, value)

    try:
        yield
    finally:
        # Restore original values
        for key, value in original_values.items():
            setattr(settings, key, value)


@contextmanager
def setup_authentication():
    """
    Context manager to set up authentication for testing.

    Configures the API to accept 'test-token' as a valid bearer token.
    """
    original_require_auth = settings.require_auth
    original_api_tokens = settings.api_tokens

    # Configure authentication for tests
    settings.require_auth = True
    settings.api_tokens = "test-token"

    try:
        yield
    finally:
        # Restore original values
        settings.require_auth = original_require_auth
        settings.api_tokens = original_api_tokens


@contextmanager
def disable_authentication():
    """
    Context manager to disable authentication for testing.

    Temporarily disables authentication requirements.
    """
    original_require_auth = settings.require_auth

    # Disable authentication for tests
    settings.require_auth = False

    try:
        yield
    finally:
        # Restore original value
        settings.require_auth = original_require_auth


@contextmanager
def phoenix_environment(working_dir: str | None = None, **env_vars: str):
    """
    Context manager for Phoenix test environment setup.

    Sets fast timeouts by default to prevent test slowdowns, and allows
    additional environment variable customization.

    Args:
        working_dir: Custom working directory for Phoenix.
        **env_vars: Additional environment variables to set.

    Example:
        with phoenix_environment() as temp_dir:
            # Phoenix will use temp_dir for SQLite with fast timeouts
            setup_tracing(enabled=True)
    """
    original_env = {}

    # Create temporary directory if not provided
    if working_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory()
        working_dir = temp_dir_obj.name
    else:
        temp_dir_obj = None

    # Default fast timeout settings for tests
    # CRITICAL OPTIMIZATION: Use 1-second timeouts instead of OpenTelemetry's
    # default 30-second timeouts to prevent tests from hanging when Phoenix
    # or OTLP endpoints are unavailable during testing
    default_test_env = {
        'OTEL_EXPORTER_OTLP_TIMEOUT': '1',  # 1 second timeout vs 30-second default
        'OTEL_BSP_EXPORT_TIMEOUT': '1000',  # 1 second in milliseconds
        'OTEL_BSP_SCHEDULE_DELAY': '100',   # 100ms delay vs longer defaults
    }

    # Merge with user-provided env vars (user vars take precedence)
    all_env_vars = {**default_test_env, **env_vars}

    try:
        # Clean up any existing tracer provider to avoid conflicts
        # This ensures we start with a clean state for each test
        reset_global_tracer_provider()

        # Set Phoenix working directory
        if 'PHOENIX_WORKING_DIR' in os.environ:
            original_env['PHOENIX_WORKING_DIR'] = os.environ['PHOENIX_WORKING_DIR']
        os.environ['PHOENIX_WORKING_DIR'] = working_dir

        # Set environment variables
        for key, value in all_env_vars.items():
            if key in os.environ:
                original_env[key] = os.environ[key]
            os.environ[key] = value

        yield working_dir

    finally:
        # Clean up tracer provider to prevent contamination of subsequent tests
        # This is critical to prevent cross-test performance issues
        reset_global_tracer_provider()

        # Restore original environment
        for key, original_value in original_env.items():
            os.environ[key] = original_value

        # Remove new environment variables
        for key in all_env_vars:
            if key not in original_env:
                os.environ.pop(key, None)

        if 'PHOENIX_WORKING_DIR' not in original_env:
            os.environ.pop('PHOENIX_WORKING_DIR', None)

        # Clean up temporary directory
        if temp_dir_obj:
            temp_dir_obj.cleanup()


def mock_openai_chat_response() -> dict[str, Any]:
    """
    Standard OpenAI chat completion response for testing.

    Returns:
        Mock response matching OpenAI API format.
    """
    return (
        OpenAIResponseBuilder()
        .id("chatcmpl-test123")
        .model("gpt-4o-mini")
        .created(1234567890)
        .choice(0, "assistant", "This is a test response from the mocked OpenAI API.", "stop")
        .usage(10, 12)
        .build()
        .model_dump()
    )


def mock_openai_chat_response_with_tools() -> dict[str, Any]:
    """
    OpenAI chat completion response with tool calls for testing.

    Returns:
        Mock response with tool calls.
    """
    tool_calls = [
        {
            "id": "call_test123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Paris"}',
            },
        },
    ]

    return (
        OpenAIResponseBuilder()
        .id("chatcmpl-test456")
        .model("gpt-4o-mini")
        .created(1234567890)
        .choice_with_tool_calls(0, "assistant", "I'll check the weather for you.", tool_calls, "tool_calls")  # noqa: E501
        .usage(15, 8)
        .build()
        .model_dump()
    )


@contextmanager
def mock_phoenix_unavailable():
    """
    Context manager to simulate Phoenix service being unavailable.

    This patches the Phoenix register function to raise connection errors.
    """
    with patch('api.tracing.register') as mock_register:
        mock_register.side_effect = ConnectionError("Phoenix service unavailable")
        yield mock_register
