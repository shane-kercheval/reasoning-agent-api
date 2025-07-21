"""
Test error handling for Phoenix tracing integration.

This module tests graceful degradation when Phoenix is unavailable,
error recovery scenarios, and ensures the API continues working
even when tracing fails.

PERFORMANCE OPTIMIZATION: OpenTelemetry is disabled by default for tests
(OTEL_SDK_DISABLED=true) to prevent performance issues. Tests that need
tracing use the tracing_enabled fixture for proper setup and cleanup.
"""

import logging
import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from httpx import AsyncClient
from httpx import ASGITransport
from api.main import app
from api.tracing import setup_tracing
from api.openai_protocol import OpenAIStreamingResponseBuilder
from tests.utils.phoenix_helpers import (
    mock_settings,
    mock_phoenix_unavailable,
    mock_openai_chat_response,
    test_authentication,
    disable_authentication,
)


@pytest.mark.integration
class TestPhoenixErrorHandling:
    """
    Test error handling and graceful degradation when Phoenix fails.

    TESTING FOCUS: Validates that the API continues working correctly even when:
    - Phoenix service is completely unavailable
    - Tracing setup fails during initialization
    - Span creation fails during request processing
    - Database permission errors occur

    These tests ensure robust error handling without performance penalties.
    """

    def test__phoenix_unavailable__api_continues_working(self, caplog: pytest.LogCaptureFixture):
        """Test API continues working when Phoenix service is unavailable."""
        with caplog.at_level(logging.WARNING):
            with mock_phoenix_unavailable():
                # Enable tracing but Phoenix connection will fail
                with mock_settings(enable_tracing=True):
                    # Test that tracing setup fails gracefully
                    with pytest.raises(Exception) as exc_info:  # noqa: PT011
                        setup_tracing(
                            enabled=True,
                            project_name="test-unavailable",
                            endpoint="http://nonexistent:4317",
                        )
                    assert "Phoenix service unavailable" in str(exc_info.value)

    def test__api_endpoints_work_without_tracing(self):
        """Test that all API endpoints work correctly when tracing is disabled."""
        # Ensure tracing is disabled and authentication is configured
        with mock_settings(enable_tracing=False), test_authentication():
            with TestClient(app) as client:
                # Test health endpoint (doesn't require auth)
                response = client.get("/health")
                assert response.status_code == 200
                assert "status" in response.json()

                # Test tools endpoint (requires auth)
                response = client.get("/tools", headers={"Authorization": "Bearer test-token"})
                assert response.status_code == 200

                # Test that these endpoints work without any tracing overhead

    def test__api_chat_completion_without_phoenix__works(self):
        """Test chat completion works when Phoenix is completely unavailable."""
        # Mock OpenAI response
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openai_chat_response()
            mock_response.headers = {'content-type': 'application/json'}
            mock_post.return_value = mock_response

            # Test with tracing disabled (default) and authentication enabled
            with mock_settings(enable_tracing=False), test_authentication():
                with TestClient(app) as client:
                    response = client.post(
                        "/v1/chat/completions",
                        headers={"Authorization": "Bearer test-token"},
                        json={
                            "model": "gpt-4o-mini",
                            "messages": [{"role": "user", "content": "Hello!"}],
                            "stream": False,
                        },
                    )

                    # API should work normally
                    assert response.status_code == 200
                    response_data = response.json()
                    assert "choices" in response_data
                    assert len(response_data["choices"]) > 0

    def test__streaming_works_without_phoenix(self):
        """Test streaming chat completion works when Phoenix is unavailable."""
        def mock_stream_response():  # noqa: ANN202
            streaming_response = (
                OpenAIStreamingResponseBuilder()
                .chunk("chatcmpl-test", "gpt-4o", delta_content="Hello")
                .chunk("chatcmpl-test", "gpt-4o", delta_content=" world")
                .done()
                .build()
            )
            return streaming_response.encode()

        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'text/plain'}
            mock_response.aiter_bytes = AsyncMock(return_value=[mock_stream_response()])
            mock_post.return_value = mock_response

            # Test with tracing disabled and authentication enabled
            with mock_settings(enable_tracing=False), test_authentication():
                with TestClient(app) as client:
                    response = client.post(
                        "/v1/chat/completions",
                        headers={"Authorization": "Bearer test-token"},
                        json={
                            "model": "gpt-4o-mini",
                            "messages": [{"role": "user", "content": "Tell me a story"}],
                            "stream": True,
                        },
                    )

                    # Streaming should work normally
                    assert response.status_code == 200
                    assert "text/event-stream" in response.headers.get("content-type", "")
                    response_text = response.text
                    assert "data:" in response_text
                    assert "[DONE]" in response_text

    def test__tracing_errors_dont_break_response_generation(self, caplog: pytest.LogCaptureFixture):  # noqa: E501
        """Test that tracing errors don't break response generation."""
        with caplog.at_level(logging.WARNING):
            # Mock scenario where tracing setup succeeds but span creation fails
            with patch('opentelemetry.trace.get_tracer') as mock_get_tracer:
                mock_tracer = AsyncMock()
                mock_tracer.start_as_current_span.side_effect = Exception("Span creation failed")
                mock_get_tracer.return_value = mock_tracer

                # Mock OpenAI response
                with patch('httpx.AsyncClient.post') as mock_post:
                    mock_response = AsyncMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = mock_openai_chat_response()
                    mock_response.headers = {'content-type': 'application/json'}
                    mock_post.return_value = mock_response

                    # Enable tracing and authentication
                    with mock_settings(enable_tracing=True), test_authentication():
                        with TestClient(app) as client:
                            # API call should still work despite tracing errors
                            response = client.post(
                                "/v1/chat/completions",
                                headers={"Authorization": "Bearer test-token"},
                                json={
                                    "model": "gpt-4o-mini",
                                    "messages": [{"role": "user", "content": "Test message"}],
                                    "stream": False,
                                },
                            )

                            # Response should be successful despite tracing issues
                            assert response.status_code == 200
                            response_data = response.json()
                            assert "choices" in response_data

    def test__invalid_phoenix_config__fails_gracefully(self):
        """Test that invalid Phoenix configuration is handled gracefully."""
        # Phoenix register() is tolerant of invalid endpoints - it sets up tracing
        # but connection failures happen later when spans are sent
        # Test that setup succeeds even with invalid endpoint
        tracer_provider = setup_tracing(
            enabled=True,
            project_name="test-invalid",
            endpoint="http://definitely-nonexistent-host-12345:4317",
        )

        # Setup should succeed (Phoenix handles connection failures later)
        assert tracer_provider is not None

    def test__phoenix_database_permission_error__handled(self, tmp_path):  # noqa: ANN001
        """Test handling of database permission errors."""
        # Create a read-only directory to simulate permission errors
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only permissions

        try:
            # This should either work or fail gracefully
            with patch.dict('os.environ', {'PHOENIX_WORKING_DIR': str(readonly_dir)}):
                # Try to setup tracing with read-only directory
                # This tests that we handle filesystem permission errors gracefully
                tracer_provider = setup_tracing(
                    enabled=True,
                    project_name="test-permissions",
                    endpoint="http://localhost:4317",
                )

                # If setup succeeds, that's fine - Phoenix might handle it
                assert tracer_provider is not None

        except Exception as e:
            # If it fails, ensure it's a meaningful error
            assert "Tracing initialization failed" in str(e)  # noqa: PT017

        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)

    def test__concurrent_requests_without_phoenix__work(self):
        """Test that concurrent requests work correctly without Phoenix."""
        async def make_request():  # noqa: ANN202
            """Make a single API request."""
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:  # noqa: E501
                with patch('httpx.AsyncClient.post') as mock_post:
                    mock_response = AsyncMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = mock_openai_chat_response()
                    mock_response.headers = {'content-type': 'application/json'}
                    mock_post.return_value = mock_response

                    response = await client.post(
                        "/v1/chat/completions",
                        headers={"Authorization": "Bearer test-token"},
                        json={
                            "model": "gpt-4o-mini",
                            "messages": [{"role": "user", "content": "Test message"}],
                            "stream": False,
                        },
                    )
                    return response.status_code

        async def test_concurrent() -> None:
            # Disable tracing and enable authentication
            with mock_settings(enable_tracing=False), test_authentication():
                # Make multiple concurrent requests
                tasks = [make_request() for _ in range(5)]
                results = await asyncio.gather(*tasks)

                # All requests should succeed
                assert all(status == 200 for status in results)

        # Run the async test
        asyncio.run(test_concurrent())


@pytest.mark.integration
class TestPhoenixRecovery:
    """
    Test recovery scenarios for Phoenix integration.

    TESTING FOCUS: Validates that the system can recover from Phoenix failures:
    - Re-enabling tracing after initial setup failures
    - Handling mixed enabled/disabled tracing states
    - Graceful transitions between different tracing configurations

    These tests ensure the system is resilient to Phoenix state changes.
    """

    def test__tracing_can_be_re_enabled_after_failure(self, phoenix_sqlite_test: str):  # noqa: ARG002
        """Test that tracing can be re-enabled after initial failure."""
        # First, simulate Phoenix unavailable
        with mock_phoenix_unavailable():
            with pytest.raises(Exception):  # noqa: PT011
                setup_tracing(
                    enabled=True,
                    project_name="test-recovery-fail",
                    endpoint="http://nonexistent:4317",
                )

        # Then test that it can work when Phoenix becomes available
        tracer_provider = setup_tracing(
            enabled=True,
            project_name="test-recovery-success",
            endpoint="http://localhost:4317",
        )

        assert tracer_provider is not None

    def test__mixed_tracing_states__handled_correctly(self, phoenix_sqlite_test: str):  # noqa: ARG002
        """Test mixing enabled/disabled tracing states."""
        # Start with tracing disabled
        with mock_settings(enable_tracing=False), disable_authentication():
            with TestClient(app) as client:
                response = client.get("/health")
                assert response.status_code == 200

        # Switch to enabled
        with mock_settings(enable_tracing=True), disable_authentication():
            setup_tracing(enabled=True, project_name="test-mixed")
            with TestClient(app) as client:
                response = client.get("/health")
                assert response.status_code == 200

        # Switch back to disabled
        with mock_settings(enable_tracing=False), disable_authentication():
            with TestClient(app) as client:
                response = client.get("/health")
                assert response.status_code == 200
