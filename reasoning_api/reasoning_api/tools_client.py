"""HTTP client for tools-api service."""

import logging
from typing import Any

import httpx

from reasoning_api.tools import ToolResult

logger = logging.getLogger(__name__)


class ToolDefinition:
    """Tool metadata from tools-api."""

    def __init__(self, name: str, description: str, parameters: dict[str, Any], tags: list[str]):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.tags = tags


class PromptDefinition:
    """Prompt metadata from tools-api."""

    def __init__(
        self,
        name: str,
        description: str,
        arguments: list[dict[str, Any]],
        tags: list[str],
    ):
        self.name = name
        self.description = description
        self.arguments = arguments
        self.tags = tags


class ToolsAPIClient:
    """
    HTTP client for tools-api service.

    Provides methods to:
    - Discover available tools and prompts
    - Execute tools with structured responses
    - Render prompts with template arguments
    """

    def __init__(self, base_url: str, timeout: float = 60.0):
        """
        Initialize tools-api client.

        Args:
            base_url: Base URL for tools-api (e.g., http://tools-api:8001)
            timeout: HTTP timeout in seconds for tool execution
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout)
        logger.info(f"Initialized ToolsAPIClient with base_url={base_url}")

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def health_check(self) -> dict[str, Any]:
        """Check if tools-api is healthy."""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    async def list_tools(self) -> list[ToolDefinition]:
        """
        List all available tools.

        Returns:
            List of ToolDefinition objects with metadata
        """
        response = await self.client.get(f"{self.base_url}/tools/")
        response.raise_for_status()
        tools_data = response.json()

        return [
            ToolDefinition(
                name=tool["name"],
                description=tool["description"],
                parameters=tool["parameters"],
                tags=tool.get("tags", []),
            )
            for tool in tools_data
        ]

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """
        Execute a tool with the provided arguments.

        Args:
            name: Tool name (e.g., "read_text_file")
            arguments: Tool arguments as dict

        Returns:
            ToolResult with success flag, result data, and metadata

        Raises:
            httpx.HTTPStatusError: If tool not found or execution fails
        """
        response = await self.client.post(
            f"{self.base_url}/tools/{name}",
            json=arguments,
        )
        response.raise_for_status()
        result_data = response.json()

        # Convert tools-api ToolResult to reasoning-api ToolResult
        return ToolResult(
            tool_name=name,
            success=result_data["success"],
            result=result_data.get("result"),
            error=result_data.get("error"),
            execution_time_ms=result_data["execution_time_ms"],
        )

    async def list_prompts(self) -> list[PromptDefinition]:
        """
        List all available prompts.

        Returns:
            List of PromptDefinition objects with metadata
        """
        response = await self.client.get(f"{self.base_url}/prompts/")
        response.raise_for_status()
        prompts_data = response.json()

        return [
            PromptDefinition(
                name=prompt["name"],
                description=prompt["description"],
                arguments=prompt["arguments"],
                tags=prompt.get("tags", []),
            )
            for prompt in prompts_data
        ]

    async def render_prompt(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Render a prompt with the provided arguments.

        Args:
            name: Prompt name (e.g., "code_review")
            arguments: Template arguments as dict

        Returns:
            Dict with success, messages, error, metadata

        Raises:
            httpx.HTTPStatusError: If prompt not found or rendering fails
        """
        response = await self.client.post(
            f"{self.base_url}/prompts/{name}",
            json=arguments,
        )
        response.raise_for_status()
        return response.json()
