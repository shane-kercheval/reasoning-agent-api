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

    def test__mcp_tools_execute_endpoint__executes_tool_with_arguments(self) -> None:
        """Test POST /v1/mcp/tools/{name} executes tool with arguments."""
        with TestClient(app) as client:
            # First get list to find a valid tool
            list_response = client.get(
                "/v1/mcp/tools",
                headers=get_auth_header(),
            )

            tools = list_response.json()["tools"]
            if tools:
                # Find a tool to test
                # Try to find a simple tool without too many required parameters
                test_tool = tools[0]
                tool_name = test_tool["name"]

                # Build arguments based on the tool's input schema
                test_args = {}
                schema = test_tool.get("input_schema", {})
                properties = schema.get("properties", {})
                required = schema.get("required", [])

                # Provide test values for required arguments
                for param_name in required:
                    param_info = properties.get(param_name, {})
                    param_type = param_info.get("type", "")

                    # Provide appropriate test value based on type
                    if "str" in param_type.lower():
                        test_args[param_name] = "test_value"
                    elif "int" in param_type.lower():
                        test_args[param_name] = 42
                    elif "bool" in param_type.lower():
                        test_args[param_name] = True
                    elif "float" in param_type.lower():
                        test_args[param_name] = 3.14
                    else:
                        # Default to string for unknown types
                        test_args[param_name] = "test"

                # Execute the tool
                response = client.post(
                    f"/v1/mcp/tools/{tool_name}",
                    json=test_args,
                    headers=get_auth_header(),
                )

                # Tool execution might succeed or fail depending on MCP server
                # We just verify the response structure is correct
                assert response.status_code in [200, 500]

                data = response.json()

                if response.status_code == 200:
                    # Success case
                    assert "tool_name" in data
                    assert "success" in data
                    assert "result" in data
                    assert "execution_time_ms" in data
                    assert data["tool_name"] == tool_name
                    assert data["success"] is True
                    assert isinstance(data["execution_time_ms"], (int, float))
                else:
                    # Failure case (tool execution failed)
                    assert "detail" in data
                    assert "error" in data["detail"]

    def test__mcp_tools_execute_endpoint__404_for_nonexistent(self) -> None:
        """Test POST /v1/mcp/tools/{name} returns 404 for unknown tool."""
        with TestClient(app) as client:
            response = client.post(
                "/v1/mcp/tools/nonexistent_tool_12345",
                json={},
                headers=get_auth_header(),
            )

            assert response.status_code == 404
            error_response = response.json()
            assert "detail" in error_response
            assert "error" in error_response["detail"]
            assert "not found" in error_response["detail"]["error"]["message"].lower()
            assert error_response["detail"]["error"]["type"] == "not_found_error"

    def test__mcp_tools_execute_endpoint__400_for_missing_required_args(self) -> None:
        """Test POST /v1/mcp/tools/{name} returns 400 for missing required arguments."""
        with TestClient(app) as client:
            # First get list to find a tool with required arguments
            list_response = client.get(
                "/v1/mcp/tools",
                headers=get_auth_header(),
            )

            tools = list_response.json()["tools"]
            if tools:
                # Find a tool with required parameters
                tool_with_required = None
                for tool in tools:
                    schema = tool.get("input_schema", {})
                    required = schema.get("required", [])
                    if required:
                        tool_with_required = tool
                        break

                if tool_with_required:
                    tool_name = tool_with_required["name"]

                    # Execute without providing required arguments
                    response = client.post(
                        f"/v1/mcp/tools/{tool_name}",
                        json={},  # Empty arguments
                        headers=get_auth_header(),
                    )

                    assert response.status_code == 400
                    error_response = response.json()
                    assert "detail" in error_response
                    assert "error" in error_response["detail"]
                    assert "required" in error_response["detail"]["error"]["message"].lower()
                    assert error_response["detail"]["error"]["type"] == "invalid_request_error"


if __name__ == "__main__":
    pytest.main([__file__])
