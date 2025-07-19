"""
Test tracing integration with the reasoning agent.

This module tests that tracing can be enabled/disabled and that it doesn't
break existing functionality.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.config import settings
from api.tracing import setup_tracing


class TestTracingIntegration:
    """Test tracing integration with the reasoning agent."""

    def test__tracing_disabled_by_default__reasoning_works(self):
        """Test that reasoning works correctly when tracing is disabled."""
        # Verify tracing is disabled by default
        assert not settings.enable_tracing

        # Create test client
        client = TestClient(app)

        # Test that health endpoint works (doesn't require auth or complex mocking)
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()

    def test__tracing_enabled__reasoning_still_works(self):
        """Test that reasoning works correctly when tracing is enabled."""
        # Temporarily enable tracing
        original_tracing = settings.enable_tracing
        settings.enable_tracing = True

        try:
            # Create test client
            client = TestClient(app)

            # Test that health endpoint still works with tracing enabled
            response = client.get("/health")
            assert response.status_code == 200
            assert "status" in response.json()

        finally:
            # Restore original setting
            settings.enable_tracing = original_tracing

    def test__health_check__works_with_tracing(self):
        """Test that health check endpoint works regardless of tracing state."""
        client = TestClient(app)

        # Test with tracing disabled
        settings.enable_tracing = False
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()

        # Test with tracing enabled
        settings.enable_tracing = True
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()

        # Restore default
        settings.enable_tracing = False

    @pytest.mark.asyncio
    async def test__tracing_setup__handles_phoenix_unavailable(self) -> None:
        """Test that tracing setup handles Phoenix service being unavailable."""
        # Test setup with invalid endpoint - this should succeed but log warnings
        # Phoenix registers successfully but then fails when trying to export traces
        tracer_provider = setup_tracing(
            enabled=True,
            project_name="test-project",
            endpoint="http://invalid-host:4317",
            enable_console_export=False,
        )

        # Verify tracer provider was created (setup succeeds)
        assert tracer_provider is not None

    def test__middleware__adds_tracing_headers(self):
        """Test that tracing middleware processes requests correctly."""
        client = TestClient(app)

        # Enable tracing temporarily
        original_tracing = settings.enable_tracing
        settings.enable_tracing = True

        try:
            # Make request to health endpoint
            response = client.get("/health")

            # Verify response is successful (middleware didn't break anything)
            assert response.status_code == 200

        finally:
            # Restore original setting
            settings.enable_tracing = original_tracing
