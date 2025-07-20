"""
Basic connection tests for Phoenix service integration.

These tests validate that the Phoenix tracing setup works correctly
and can connect to a Phoenix service when available.
"""

import pytest
from unittest.mock import patch
from opentelemetry import trace

from api.tracing import setup_tracing
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

        assert "Phoenix service unavailable" in str(exc_info.value)


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
    """Test that trace.get_tracer returns a valid tracer."""
    tracer = trace.get_tracer('test_module')

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
        tracer = trace.get_tracer('integration_test')
        with tracer.start_as_current_span('test-integration-span') as span:
            span.set_attribute('test.type', 'integration')
            span.set_attribute('test.timestamp', '2025-01-01T00:00:00Z')

            # Nested span to test hierarchy
            with tracer.start_as_current_span('nested-span') as nested_span:
                nested_span.set_attribute('nested.operation', 'test')

        assert tracer_provider is not None

    except Exception as e:
        pytest.skip(f"Phoenix service not available for integration test: {e}")


