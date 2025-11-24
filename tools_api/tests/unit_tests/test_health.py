"""Tests for health check endpoint."""

import pytest


@pytest.mark.asyncio
async def test_health_check(client) -> None:
    """Test health check returns correct response."""
    response = await client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "tools-api"
    assert data["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_tools_list_endpoint_exists(client) -> None:
    """Test tools list endpoint exists (empty for now)."""
    response = await client.get("/tools/")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_prompts_list_endpoint_exists(client) -> None:
    """Test prompts list endpoint exists (empty for now)."""
    response = await client.get("/prompts/")
    assert response.status_code == 200
    assert response.json() == []
