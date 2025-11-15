"""
Integration tests for MCP tools and prompts endpoints.

These tests verify the full end-to-end behavior of MCP endpoints,
including authentication, MCP client integration, and response formats.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.config import settings


def get_auth_header() -> dict[str, str]:
    """
    Get authentication header for tests.

    Returns empty dict if auth is disabled or no tokens configured.
    """
    if not settings.require_auth or not settings.allowed_tokens:
        return {}
    return {"Authorization": f"Bearer {settings.allowed_tokens[0]}"}


@pytest.mark.integration
class TestMCPEndpointsIntegration:
    """Integration tests for MCP tools/prompts endpoints."""

    def test__mcp_tools_endpoint__returns_tools_list(self) -> None:
        """Test /v1/mcp/tools with real MCP client."""
        with TestClient(app) as client:
            response = client.get(
                "/v1/mcp/tools",
                headers=get_auth_header(),
            )

            assert response.status_code == 200
            data = response.json()
            assert "tools" in data
            assert isinstance(data["tools"], list)

            # Verify structure if tools exist
            if data["tools"]:
                tool = data["tools"][0]
                assert "name" in tool
                assert "description" in tool
                assert "input_schema" in tool

    def test__mcp_prompts_list_endpoint__returns_prompts_list(self) -> None:
        """Test /v1/mcp/prompts with real MCP client."""
        with TestClient(app) as client:
            response = client.get(
                "/v1/mcp/prompts",
                headers=get_auth_header(),
            )

            assert response.status_code == 200
            data = response.json()
            assert "prompts" in data
            assert isinstance(data["prompts"], list)

            # Verify structure if prompts exist
            if data["prompts"]:
                prompt = data["prompts"][0]
                assert "name" in prompt
                assert "description" in prompt
                assert "arguments" in prompt

    def test__mcp_prompts_get_endpoint__returns_specific_prompt(self) -> None:
        """Test /v1/mcp/prompts/{name} with real MCP client."""
        with TestClient(app) as client:
            # First get list to find a valid prompt name
            list_response = client.get(
                "/v1/mcp/prompts",
                headers=get_auth_header(),
            )

            prompts = list_response.json()["prompts"]
            if prompts:
                # Test getting first prompt
                prompt_name = prompts[0]["name"]
                response = client.get(
                    f"/v1/mcp/prompts/{prompt_name}",
                    headers=get_auth_header(),
                )

                assert response.status_code == 200
                data = response.json()
                assert data["name"] == prompt_name
                assert "description" in data
                assert "arguments" in data

    def test__mcp_prompts_get_endpoint__404_for_nonexistent(self) -> None:
        """Test /v1/mcp/prompts/{name} returns 404 for unknown prompt."""
        with TestClient(app) as client:
            response = client.get(
                "/v1/mcp/prompts/nonexistent_prompt_12345",
                headers=get_auth_header(),
            )

            assert response.status_code == 404
            error_response = response.json()
            assert "detail" in error_response
            assert "error" in error_response["detail"]
            assert "not found" in error_response["detail"]["error"]["message"].lower()

    def test__mcp_tools_endpoint__requires_authentication(self) -> None:
        """Test /v1/mcp/tools requires bearer token when auth is enabled."""
        # Skip if auth is disabled
        if not settings.require_auth:
            pytest.skip("Authentication is disabled in test environment")

        with TestClient(app) as client:
            # Request without auth header should fail
            response = client.get("/v1/mcp/tools")

            assert response.status_code == 401

    def test__mcp_prompts_endpoint__requires_authentication(self) -> None:
        """Test /v1/mcp/prompts requires bearer token when auth is enabled."""
        # Skip if auth is disabled
        if not settings.require_auth:
            pytest.skip("Authentication is disabled in test environment")

        with TestClient(app) as client:
            # Request without auth header should fail
            response = client.get("/v1/mcp/prompts")

            assert response.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__])
