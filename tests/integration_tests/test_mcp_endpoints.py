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

    def test__mcp_prompts_execute_endpoint__executes_prompt_with_arguments(self) -> None:
        """Test POST /v1/mcp/prompts/{name} executes prompt with arguments."""
        with TestClient(app) as client:
            # First get list to find a valid prompt name
            list_response = client.get(
                "/v1/mcp/prompts",
                headers=get_auth_header(),
            )

            prompts = list_response.json()["prompts"]
            if prompts:
                # Find a prompt with arguments to test
                prompt_with_args = None
                for prompt in prompts:
                    if prompt["arguments"]:
                        prompt_with_args = prompt
                        break

                if prompt_with_args:
                    prompt_name = prompt_with_args["name"]

                    # Build arguments based on the prompt's argument specifications
                    test_args = {}
                    for arg in prompt_with_args["arguments"]:
                        # Provide test values for each argument
                        test_args[arg["name"]] = "test_value"

                    # Execute the prompt
                    response = client.post(
                        f"/v1/mcp/prompts/{prompt_name}",
                        json=test_args,
                        headers=get_auth_header(),
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert "description" in data
                    assert "messages" in data
                    assert isinstance(data["messages"], list)

                    # Verify messages have correct structure
                    if data["messages"]:
                        message = data["messages"][0]
                        assert "role" in message
                        assert "content" in message

    def test__mcp_prompts_execute_endpoint__404_for_nonexistent(self) -> None:
        """Test POST /v1/mcp/prompts/{name} returns 404 for unknown prompt."""
        with TestClient(app) as client:
            response = client.post(
                "/v1/mcp/prompts/nonexistent_prompt_12345",
                json={},
                headers=get_auth_header(),
            )

            assert response.status_code == 404
            error_response = response.json()
            assert "detail" in error_response
            assert "error" in error_response["detail"]
            assert "not found" in error_response["detail"]["error"]["message"].lower()

    def test__mcp_prompts_execute_endpoint__400_for_missing_required_args(self) -> None:
        """Test POST /v1/mcp/prompts/{name} returns 400 for missing required arguments."""
        with TestClient(app) as client:
            # First get list to find a prompt with required arguments
            list_response = client.get(
                "/v1/mcp/prompts",
                headers=get_auth_header(),
            )

            prompts = list_response.json()["prompts"]
            if prompts:
                # Find a prompt with required arguments
                prompt_with_required = None
                for prompt in prompts:
                    if any(arg.get("required", False) for arg in prompt["arguments"]):
                        prompt_with_required = prompt
                        break

                if prompt_with_required:
                    prompt_name = prompt_with_required["name"]

                    # Execute without providing required arguments
                    response = client.post(
                        f"/v1/mcp/prompts/{prompt_name}",
                        json={},  # Empty arguments
                        headers=get_auth_header(),
                    )

                    assert response.status_code == 400
                    error_response = response.json()
                    assert "detail" in error_response
                    assert "error" in error_response["detail"]
                    assert "required" in error_response["detail"]["error"]["message"].lower()


if __name__ == "__main__":
    pytest.main([__file__])
