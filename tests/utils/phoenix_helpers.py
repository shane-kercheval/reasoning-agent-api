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
from api.config import settings


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
def mock_settings(**overrides: Any):  # noqa: ANN401
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
def test_authentication():
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

    Args:
        working_dir: Custom working directory for Phoenix.
        **env_vars: Additional environment variables to set.

    Example:
        with phoenix_environment() as temp_dir:
            # Phoenix will use temp_dir for SQLite
            setup_tracing(enabled=True)
    """
    original_env = {}

    # Create temporary directory if not provided
    if working_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory()
        working_dir = temp_dir_obj.name
    else:
        temp_dir_obj = None

    try:
        # Clean up any existing tracer provider to avoid conflicts
        reset_global_tracer_provider()

        # Set Phoenix working directory
        if 'PHOENIX_WORKING_DIR' in os.environ:
            original_env['PHOENIX_WORKING_DIR'] = os.environ['PHOENIX_WORKING_DIR']
        os.environ['PHOENIX_WORKING_DIR'] = working_dir

        # Set additional environment variables
        for key, value in env_vars.items():
            if key in os.environ:
                original_env[key] = os.environ[key]
            os.environ[key] = value

        yield working_dir

    finally:
        # Clean up tracer provider
        reset_global_tracer_provider()

        # Restore original environment
        for key, original_value in original_env.items():
            os.environ[key] = original_value

        # Remove new environment variables
        for key in env_vars:
            if key not in original_env:
                os.environ.pop(key, None)

        if 'PHOENIX_WORKING_DIR' not in original_env:
            os.environ.pop('PHOENIX_WORKING_DIR', None)

        # Clean up temporary directory
        if temp_dir_obj:
            temp_dir_obj.cleanup()


def reset_global_tracer_provider():
    """
    Reset the global tracer provider to avoid conflicts between tests.

    This prevents the "Overriding of current TracerProvider is not allowed" warning.
    """
    try:
        # Reset the global tracer provider
        trace._TRACER_PROVIDER = None
        trace._TRACER_PROVIDER_SET_ONCE = Once()
    except Exception:
        # If we can't reset it, that's ok - the warning isn't critical
        pass


def mock_openai_chat_response() -> dict[str, Any]:
    """
    Standard OpenAI chat completion response for testing.

    Returns:
        Mock response matching OpenAI API format.
    """
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from the mocked OpenAI API.",
                },
                "finish_reason": "stop",
            },
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 12,
            "total_tokens": 22,
        },
    }


def mock_openai_chat_response_with_tools() -> dict[str, Any]:
    """
    OpenAI chat completion response with tool calls for testing.

    Returns:
        Mock response with tool calls.
    """
    return {
        "id": "chatcmpl-test456",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I'll check the weather for you.",
                    "tool_calls": [
                        {
                            "id": "call_test123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Paris"}',
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            },
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 8,
            "total_tokens": 23,
        },
    }


@contextmanager
def mock_phoenix_unavailable():
    """
    Context manager to simulate Phoenix service being unavailable.

    This patches the Phoenix register function to raise connection errors.
    """
    with patch('api.tracing.register') as mock_register:
        mock_register.side_effect = ConnectionError("Phoenix service unavailable")
        yield mock_register
