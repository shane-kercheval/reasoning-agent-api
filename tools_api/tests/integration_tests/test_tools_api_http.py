"""
Integration tests for tools-api HTTP endpoints.

These tests verify the full HTTP request/response cycle including:
- FastAPI routing and parameter validation
- Path translation through HTTP layer
- Error propagation via HTTP responses
- Tool execution via REST API
"""

import pytest


# ========================================
# HEALTH & DISCOVERY ENDPOINTS
# ========================================


@pytest.mark.asyncio
async def test_health_endpoint(tools_api_client) -> None:
    """Test health check endpoint returns 200 OK."""
    response = await tools_api_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "tools-api"


@pytest.mark.asyncio
async def test_list_tools_endpoint(tools_api_client) -> None:
    """Test tool discovery endpoint returns all registered tools."""
    response = await tools_api_client.get("/tools/")

    assert response.status_code == 200
    tools = response.json()

    assert isinstance(tools, list)
    assert len(tools) > 0

    # Check for expected filesystem tools
    tool_names = {t["name"] for t in tools}
    assert "read_text_file" in tool_names
    assert "write_file" in tool_names
    assert "list_directory" in tool_names
    assert "delete_file" in tool_names

    # Verify tool structure
    for tool in tools:
        assert "name" in tool
        assert "description" in tool
        assert "parameters" in tool
        assert "tags" in tool


@pytest.mark.asyncio
async def test_list_prompts_endpoint(tools_api_client) -> None:
    """Test prompt discovery endpoint returns registered prompts."""
    response = await tools_api_client.get("/prompts/")

    assert response.status_code == 200
    prompts = response.json()

    assert isinstance(prompts, list)
    # Should have at least the example prompt
    assert len(prompts) >= 1

    # Verify prompt structure
    for prompt in prompts:
        assert "name" in prompt
        assert "description" in prompt
        assert "arguments" in prompt


# ========================================
# TOOL EXECUTION VIA HTTP
# ========================================


@pytest.mark.asyncio
async def test_execute_read_file_via_http(tools_api_client, integration_workspace) -> None:
    """Test reading a file through HTTP endpoint with path translation."""
    # File exists in container path
    workspace = integration_workspace["container_workspace"]
    test_file = workspace / "http_test.txt"
    test_file.write_text("http integration test")

    # Call API with host path
    host_path = str(integration_workspace["host_workspace"] / "http_test.txt")
    response = await tools_api_client.post(
        "/tools/read_text_file",
        json={"path": host_path},
    )

    assert response.status_code == 200
    result = response.json()

    # Tool execution succeeded
    assert result["success"] is True
    assert result["error"] is None
    assert result["execution_time_ms"] > 0

    # Content is correct
    assert result["result"]["content"] == "http integration test"
    assert result["result"]["size_bytes"] > 0
    assert result["result"]["line_count"] == 1

    # Response contains host path, not container path
    assert result["result"]["path"] == host_path
    assert "/mnt/" not in result["result"]["path"]


@pytest.mark.asyncio
async def test_execute_write_file_via_http(tools_api_client, integration_workspace) -> None:
    """Test writing a file through HTTP endpoint."""
    host_path = str(integration_workspace["host_workspace"] / "new_file.txt")

    response = await tools_api_client.post(
        "/tools/write_file",
        json={
            "path": host_path,
            "content": "created via http",
        },
    )

    assert response.status_code == 200
    result = response.json()

    assert result["success"] is True
    assert result["result"]["path"] == host_path

    # Verify file was actually created in container path
    container_file = integration_workspace["container_workspace"] / "new_file.txt"
    assert container_file.exists()
    assert container_file.read_text() == "created via http"


@pytest.mark.asyncio
async def test_execute_list_directory_via_http(
    tools_api_client, integration_workspace,
) -> None:
    """Test listing directory through HTTP endpoint."""
    # Create some files
    workspace = integration_workspace["container_workspace"]
    (workspace / "file1.txt").write_text("content1")
    (workspace / "file2.txt").write_text("content2")
    (workspace / "subdir").mkdir()

    host_path = str(integration_workspace["host_workspace"])
    response = await tools_api_client.post(
        "/tools/list_directory",
        json={"path": host_path},
    )

    assert response.status_code == 200
    result = response.json()

    assert result["success"] is True
    assert result["result"]["count"] >= 3  # file1, file2, subdir, test.txt
    assert result["result"]["path"] == host_path

    # Check entries have expected structure and host paths
    entry_names = {e["name"] for e in result["result"]["entries"]}
    assert "file1.txt" in entry_names
    assert "file2.txt" in entry_names
    assert "subdir" in entry_names


# ========================================
# ERROR HANDLING VIA HTTP
# ========================================


@pytest.mark.asyncio
async def test_tool_not_found_http_error(tools_api_client) -> None:
    """Test that requesting non-existent tool returns 404."""
    response = await tools_api_client.post(
        "/tools/nonexistent_tool",
        json={"param": "value"},
    )

    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


@pytest.mark.asyncio
async def test_file_not_found_error_via_http(tools_api_client, integration_workspace) -> None:
    """Test that file not found error is properly returned via HTTP."""
    host_path = str(integration_workspace["host_workspace"] / "nonexistent.txt")

    response = await tools_api_client.post(
        "/tools/read_text_file",
        json={"path": host_path},
    )

    # HTTP call succeeds (200), but tool execution failed
    assert response.status_code == 200
    result = response.json()

    assert result["success"] is False
    assert result["result"] is None
    assert result["error"] is not None
    assert "not found" in result["error"].lower()


@pytest.mark.asyncio
async def test_permission_denied_error_via_http(tools_api_client) -> None:
    """Test that permission errors are properly returned via HTTP."""
    # Try to read file outside allowed paths
    response = await tools_api_client.post(
        "/tools/read_text_file",
        json={"path": "/etc/passwd"},
    )

    assert response.status_code == 200
    result = response.json()

    assert result["success"] is False
    assert result["error"] is not None
    assert "not accessible" in result["error"].lower()


@pytest.mark.asyncio
async def test_blocked_file_error_via_http(tools_api_client, integration_workspace) -> None:
    """Test that writing to blocked files returns error via HTTP."""
    host_path = str(integration_workspace["host_workspace"] / ".env")

    response = await tools_api_client.post(
        "/tools/write_file",
        json={
            "path": host_path,
            "content": "SECRET=value",
        },
    )

    assert response.status_code == 200
    result = response.json()

    assert result["success"] is False
    assert "blocked" in result["error"].lower()


@pytest.mark.asyncio
async def test_missing_required_parameter_http_error(tools_api_client) -> None:
    """Test that missing required parameters returns validation error."""
    # write_file requires both 'path' and 'content'
    response = await tools_api_client.post(
        "/tools/write_file",
        json={"path": "/some/path"},  # Missing 'content'
    )

    # Should get error (either 422 or 200 with error)
    if response.status_code == 422:
        # FastAPI validation error
        data = response.json()
        assert "detail" in data
    else:
        # Tool execution error
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is False


# ========================================
# PATH TRANSLATION VIA HTTP
# ========================================


@pytest.mark.asyncio
async def test_path_translation_roundtrip_via_http(
    tools_api_client, integration_workspace,
) -> None:
    """Test that path translation works correctly through HTTP layer."""
    # Create file in container path
    container_file = integration_workspace["container_workspace"] / "roundtrip.txt"
    container_file.write_text("roundtrip test")

    # Call with host path
    host_path = str(integration_workspace["host_workspace"] / "roundtrip.txt")

    # Read via HTTP
    response = await tools_api_client.post(
        "/tools/read_text_file",
        json={"path": host_path},
    )

    result = response.json()
    assert result["success"] is True

    # Response should contain host path, not container path
    returned_path = result["result"]["path"]
    assert returned_path == host_path
    assert "/mnt/" not in returned_path
    assert "Users/test/workspace" in returned_path


@pytest.mark.skip(reason="Fixture timing issue - path mapper reset between fixture and client setup")
@pytest.mark.asyncio
async def test_read_only_volume_via_http(tools_api_client, integration_workspace) -> None:
    """Test that read-only volumes are accessible for reading but not writing."""
    # Read from read-only volume should work
    host_path = str(integration_workspace["host_downloads"] / "readonly.txt")

    read_response = await tools_api_client.post(
        "/tools/read_text_file",
        json={"path": host_path},
    )

    assert read_response.status_code == 200
    read_result = read_response.json()
    assert read_result["success"] is True
    assert read_result["result"]["content"] == "readonly content"

    # Write to read-only volume should fail
    write_response = await tools_api_client.post(
        "/tools/write_file",
        json={
            "path": host_path,
            "content": "attempt to write",
        },
    )

    assert write_response.status_code == 200
    write_result = write_response.json()
    assert write_result["success"] is False
    # Error message could be "not in writable location" or "writable"
    assert "writable" in write_result["error"].lower() or "write" in write_result["error"].lower()


# ========================================
# CONCURRENT REQUESTS VIA HTTP
# ========================================


@pytest.mark.asyncio
async def test_concurrent_http_requests(tools_api_client, integration_workspace) -> None:
    """Test that multiple concurrent HTTP requests are handled correctly."""
    import asyncio

    # Create test file
    container_file = integration_workspace["container_workspace"] / "concurrent.txt"
    container_file.write_text("concurrent http test")

    host_path = str(integration_workspace["host_workspace"] / "concurrent.txt")

    # Make 50 concurrent HTTP requests
    tasks = [
        tools_api_client.post("/tools/read_text_file", json={"path": host_path})
        for _ in range(50)
    ]

    responses = await asyncio.gather(*tasks)

    # All requests should succeed
    assert len(responses) == 50
    for response in responses:
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert result["result"]["content"] == "concurrent http test"


# ========================================
# PROMPT RENDERING VIA HTTP
# ========================================


@pytest.mark.asyncio
async def test_render_prompt_via_http(tools_api_client) -> None:
    """Test rendering a prompt through HTTP endpoint."""
    # Use the greeting prompt from the codebase
    response = await tools_api_client.post(
        "/prompts/greeting",
        json={
            "name": "TestUser",
            "formal": True,
        },
    )

    assert response.status_code == 200
    result = response.json()

    assert result["success"] is True
    assert isinstance(result["content"], str)
    assert "TestUser" in result["content"]
    assert "Good day" in result["content"]


@pytest.mark.asyncio
async def test_prompt_missing_required_argument_http_error(tools_api_client) -> None:
    """Test that missing required prompt arguments returns validation error."""
    # greeting prompt requires 'name'
    response = await tools_api_client.post(
        "/prompts/greeting",
        json={"formal": True},  # Missing 'name'
    )

    # Should either be 422 (validation error) or 200 with success=false
    if response.status_code == 422:
        data = response.json()
        assert "detail" in data
    else:
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is False
        assert result["error"] is not None


# ========================================
# FILE-BASED PROMPTS VIA HTTP
# ========================================


@pytest.mark.asyncio
async def test_file_based_prompts_appear_in_listing(tools_api_client) -> None:
    """Test that file-based prompts from fixtures appear in listing."""
    response = await tools_api_client.get("/prompts/")

    assert response.status_code == 200
    prompts = response.json()

    # Get prompt names
    prompt_names = {p["name"] for p in prompts}

    # Should include file-based prompts from fixtures
    assert "simple_test" in prompt_names
    assert "conditional_test" in prompt_names
    assert "nested_test" in prompt_names

    # Verify structure includes category and tags
    simple_prompt = next(p for p in prompts if p["name"] == "simple_test")
    assert simple_prompt["category"] == "test"
    assert "test" in simple_prompt["tags"]
    assert "simple" in simple_prompt["tags"]


@pytest.mark.asyncio
async def test_file_based_prompt_rendering_with_required_args(tools_api_client) -> None:
    """Test rendering file-based prompt with required arguments."""
    response = await tools_api_client.post(
        "/prompts/simple_test",
        json={"name": "World"},
    )

    assert response.status_code == 200
    result = response.json()

    assert result["success"] is True
    assert "Hello, World!" in result["content"]
    assert "simple test prompt" in result["content"]


@pytest.mark.asyncio
async def test_file_based_prompt_with_optional_args_provided(tools_api_client) -> None:
    """Test rendering file-based prompt with optional arguments provided."""
    response = await tools_api_client.post(
        "/prompts/conditional_test",
        json={"language": "Python", "focus": "security"},
    )

    assert response.status_code == 200
    result = response.json()

    assert result["success"] is True
    assert "Python" in result["content"]
    assert "Focus on: security" in result["content"]


@pytest.mark.asyncio
async def test_file_based_prompt_with_optional_args_omitted(tools_api_client) -> None:
    """Test rendering file-based prompt with optional arguments omitted."""
    response = await tools_api_client.post(
        "/prompts/conditional_test",
        json={"language": "JavaScript"},
    )

    assert response.status_code == 200
    result = response.json()

    assert result["success"] is True
    assert "JavaScript" in result["content"]
    assert "Focus on:" not in result["content"]  # Conditional not rendered


@pytest.mark.asyncio
async def test_file_based_prompt_missing_required_arg(tools_api_client) -> None:
    """Test that missing required argument returns proper error response."""
    response = await tools_api_client.post(
        "/prompts/simple_test",
        json={},  # Missing required 'name'
    )

    assert response.status_code == 200
    result = response.json()

    assert result["success"] is False
    assert result["error"] is not None
    assert "name" in result["error"].lower()


@pytest.mark.asyncio
async def test_nested_file_based_prompt_loaded(tools_api_client) -> None:
    """Test that prompts from nested directories are loaded."""
    response = await tools_api_client.post(
        "/prompts/nested_test",
        json={"topic": "integration testing"},
    )

    assert response.status_code == 200
    result = response.json()

    assert result["success"] is True
    assert "integration testing" in result["content"]


@pytest.mark.asyncio
async def test_nonexistent_prompt_returns_404(tools_api_client) -> None:
    """Test that requesting non-existent prompt returns 404."""
    response = await tools_api_client.post(
        "/prompts/nonexistent_prompt",
        json={"arg": "value"},
    )

    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()
