"""Tests for tool and prompt API endpoints."""

from typing import Any
import pytest
import pytest_asyncio

from tools_api.services.registry import ToolRegistry, PromptRegistry
from tools_api.services.tools.example_tool import EchoTool
from tools_api.services.prompts.example_prompt import GreetingPrompt


@pytest_asyncio.fixture(autouse=True)
async def setup_test_tools_prompts():
    """Register test tools and prompts before each test."""
    ToolRegistry.clear()
    PromptRegistry.clear()

    ToolRegistry.register(EchoTool())
    PromptRegistry.register(GreetingPrompt())

    yield

    ToolRegistry.clear()
    PromptRegistry.clear()


@pytest.mark.asyncio
async def test_list_tools_endpoint(client: Any) -> None:
    """Test listing tools via API endpoint."""
    response = await client.get("/tools/")
    assert response.status_code == 200

    tools = response.json()
    assert len(tools) == 1
    assert tools[0]["name"] == "echo"
    assert tools[0]["description"] == "Echo the input message back with metadata"
    assert tools[0]["tags"] == ["example", "test"]
    assert "properties" in tools[0]["parameters"]
    # output_schema is included and derived from result_model
    assert "output_schema" in tools[0]
    assert tools[0]["output_schema"]["type"] == "object"
    assert "echo" in tools[0]["output_schema"]["properties"]
    assert "length" in tools[0]["output_schema"]["properties"]
    assert "reversed" in tools[0]["output_schema"]["properties"]


@pytest.mark.asyncio
async def test_execute_tool_success(client: Any) -> None:
    """Test executing tool via API endpoint."""
    response = await client.post(
        "/tools/echo",
        json={"message": "hello world"},
    )
    assert response.status_code == 200

    result = response.json()
    assert result["success"] is True
    assert result["result"]["echo"] == "hello world"
    assert result["result"]["length"] == 11
    assert result["error"] is None
    assert result["execution_time_ms"] > 0


@pytest.mark.asyncio
async def test_execute_tool_not_found(client: Any) -> None:
    """Test executing non-existent tool returns 404."""
    response = await client.post(
        "/tools/nonexistent",
        json={},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_execute_tool_missing_params(client: Any) -> None:
    """Test executing tool with missing parameters."""
    response = await client.post(
        "/tools/echo",
        json={},  # Missing required 'message' parameter
    )
    assert response.status_code == 200  # Tool handles error, doesn't raise

    result = response.json()
    assert result["success"] is False
    assert result["error"] is not None


@pytest.mark.asyncio
async def test_list_prompts_endpoint(client: Any) -> None:
    """Test listing prompts via API endpoint."""
    response = await client.get("/prompts/")
    assert response.status_code == 200

    prompts = response.json()
    assert len(prompts) == 1
    assert prompts[0]["name"] == "greeting"
    assert prompts[0]["description"] == "Generate a greeting message"
    assert prompts[0]["tags"] == ["example", "test"]
    assert len(prompts[0]["arguments"]) == 2


@pytest.mark.asyncio
async def test_render_prompt_success(client: Any) -> None:
    """Test rendering prompt via API endpoint."""
    response = await client.post(
        "/prompts/greeting",
        json={"name": "Alice", "formal": False},
    )
    assert response.status_code == 200

    result = response.json()
    assert result["success"] is True
    assert "Alice" in result["content"]
    assert "Hey" in result["content"]
    assert result["error"] is None


@pytest.mark.asyncio
async def test_render_prompt_formal(client: Any) -> None:
    """Test rendering formal prompt."""
    response = await client.post(
        "/prompts/greeting",
        json={"name": "Bob", "formal": True},
    )
    assert response.status_code == 200

    result = response.json()
    assert result["success"] is True
    assert "Bob" in result["content"]
    assert "Good day" in result["content"]


@pytest.mark.asyncio
async def test_render_prompt_not_found(client: Any) -> None:
    """Test rendering non-existent prompt returns 404."""
    response = await client.post(
        "/prompts/nonexistent",
        json={},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_render_prompt_missing_params(client: Any) -> None:
    """Test rendering prompt with missing parameters."""
    response = await client.post(
        "/prompts/greeting",
        json={},  # Missing required 'name' parameter
    )
    assert response.status_code == 200  # Prompt handles error

    result = response.json()
    assert result["success"] is False
    assert result["error"] is not None
