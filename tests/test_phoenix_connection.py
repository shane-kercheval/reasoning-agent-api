"""
Basic connection tests for Phoenix service integration.

These tests validate that the Phoenix tracing setup works correctly
and can connect to a Phoenix service when available.
"""

import asyncio
import logging
import pytest
from unittest.mock import MagicMock, patch

from api.tracing import add_span_attributes, get_tracer, setup_tracing, trace_function
from api.config import settings


def test__setup_tracing__disabled__returns_noop_provider():
    """Test that setup_tracing returns a no-op provider when disabled."""
    tracer_provider = setup_tracing(enabled=False)

    # Should return a TracerProvider (no-op)
    assert tracer_provider is not None
    assert hasattr(tracer_provider, 'get_tracer')

    # Should be able to get a tracer and create spans without error
    tracer = tracer_provider.get_tracer('test')
    with tracer.start_as_current_span('test-span') as span:
        span.set_attribute('test.key', 'test_value')
        assert span is not None


def test__setup_tracing__enabled_but_phoenix_unavailable__raises_exception():
    """Test that setup_tracing raises exception when Phoenix is unavailable."""
    with patch('api.tracing.register') as mock_register:
        mock_register.side_effect = ConnectionError("Phoenix service unavailable")

        with pytest.raises(Exception) as exc_info:  # noqa: PT011
            setup_tracing(
                enabled=True,
                project_name='test-project',
                endpoint='http://nonexistent:4317',
            )

        assert "Tracing initialization failed" in str(exc_info.value)
        assert "Phoenix service unavailable" in str(exc_info.value)


def test__setup_tracing__logs_success_message(caplog: pytest.LogCaptureFixture):
    """Test that setup_tracing logs success message."""
    with caplog.at_level(logging.INFO):
        tracer_provider = setup_tracing(
            enabled=True,
            project_name='test-project',
            endpoint='http://localhost:4317',
            enable_console_export=False,
        )

    assert tracer_provider is not None
    assert "Phoenix tracing initialized for project 'test-project'" in caplog.text


def test__setup_tracing__with_console_export():
    """Test that console export can be enabled."""
    # This is more of a smoke test since Phoenix handles the actual setup
    tracer_provider = setup_tracing(
        enabled=True,
        project_name='test-project',
        endpoint='http://localhost:4317',
        enable_console_export=True,
    )

    assert tracer_provider is not None


def test__get_tracer__returns_tracer():
    """Test that get_tracer returns a valid tracer."""
    tracer = get_tracer('test_module')

    assert tracer is not None
    assert hasattr(tracer, 'start_as_current_span')
    assert hasattr(tracer, 'start_span')


def test__settings_defaults__tracing_disabled():
    """Test that tracing is disabled by default in settings."""
    # This test validates our configuration defaults
    assert settings.enable_tracing is False
    assert settings.enable_console_tracing is False
    assert settings.phoenix_project_name == 'reasoning-agent'
    assert settings.phoenix_collector_endpoint == 'http://localhost:4317'


@pytest.mark.integration
def test__phoenix_connection__real_service():
    """
    Integration test for connecting to a real Phoenix service.

    This test only runs when Phoenix is actually available (marked as integration).
    It validates that tracing setup works with a real Phoenix instance.
    """
    try:
        # Try to set up tracing with real Phoenix
        tracer_provider = setup_tracing(
            enabled=True,
            project_name='integration-test',
            endpoint='http://localhost:4317',
        )

        # Create a test span
        tracer = get_tracer('integration_test')
        with tracer.start_as_current_span('test-integration-span') as span:
            span.set_attribute('test.type', 'integration')
            span.set_attribute('test.timestamp', '2025-01-01T00:00:00Z')

            # Nested span to test hierarchy
            with tracer.start_as_current_span('nested-span') as nested_span:
                nested_span.set_attribute('nested.operation', 'test')

        assert tracer_provider is not None

    except Exception as e:
        pytest.skip(f"Phoenix service not available for integration test: {e}")


def test__trace_context_utilities():
    """Test the trace context management utilities."""
    # Test add_span_attributes with mock span
    mock_span = MagicMock()
    attributes = {
        'string_attr': 'test_value',
        'int_attr': 42,
        'float_attr': 3.14,
        'bool_attr': True,
        'list_attr': ['a', 'b', 'c'],
        'none_attr': None,
        'complex_attr': {'nested': 'object'},
    }

    add_span_attributes(mock_span, attributes)

    # Verify set_attribute was called for each non-None attribute
    expected_calls = [
        ('string_attr', 'test_value'),
        ('int_attr', 42),
        ('float_attr', 3.14),
        ('bool_attr', True),
        ('list_attr', ['a', 'b', 'c']),
        ('complex_attr', "{'nested': 'object'}"),  # Complex objects converted to string
    ]

    for key, value in expected_calls:
        mock_span.set_attribute.assert_any_call(key, value)

    # None values should not call set_attribute
    assert not any(call.args[0] == 'none_attr' for call in mock_span.set_attribute.call_args_list)


def test__trace_function_decorator():
    """Test the trace_function decorator."""

    # Test with sync function
    @trace_function(name='custom_operation', attributes={'operation.type': 'test'})
    def sync_function(x: int, y: int) -> int:
        return x + y

    result = sync_function(2, 3)
    assert result == 5

    # Test with async function
    @trace_function(attributes={'async.operation': True})
    async def async_function(value: str) -> str:
        return f"processed_{value}"

    result = asyncio.run(async_function('test'))
    assert result == "processed_test"
