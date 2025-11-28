"""Base classes for tools and prompts."""

import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from tools_api.models import ToolResult, PromptResult


class BaseTool(ABC):
    """
    Base class for all tools.

    Tools execute operations and return structured results with metadata.
    Each tool must implement name, description, parameters schema, and _execute method.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (used in API endpoint)."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        pass

    @property
    @abstractmethod
    def result_model(self) -> type[BaseModel]:
        """Pydantic model defining the output structure."""
        pass

    @property
    def output_schema(self) -> dict[str, Any]:
        """JSON Schema for tool output, derived from result_model."""
        return self.result_model.model_json_schema()

    @property
    def tags(self) -> list[str]:
        """Optional tags for categorization."""
        return []

    @property
    def category(self) -> str | None:
        """Category for organization. Override in subclasses."""
        return None

    @abstractmethod
    async def _execute(self, **kwargs) -> BaseModel:
        """
        Execute tool and return result as Pydantic model.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            Instance of result_model Pydantic class

        Raises:
            Exception: Any errors during execution
        """
        pass

    async def __call__(self, **kwargs) -> ToolResult:
        """
        Wrapper that handles timing and error handling.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult with success status, result data, and execution metrics
        """
        start = time.time()
        try:
            result = await self._execute(**kwargs)
            execution_time_ms = (time.time() - start) * 1000
            return ToolResult(
                success=True,
                result=result,
                execution_time_ms=execution_time_ms,
            )
        except Exception as e:
            execution_time_ms = (time.time() - start) * 1000
            return ToolResult(
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=execution_time_ms,
            )


class BasePrompt(ABC):
    """
    Base class for all prompts.

    Prompts render templates with arguments and return content strings.
    Each prompt must implement name, description, arguments schema, and render method.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Prompt name (used in API endpoint)."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Prompt description."""
        pass

    @property
    @abstractmethod
    def arguments(self) -> list[dict[str, Any]]:
        """
        Prompt arguments schema.

        Returns:
            List of argument definitions:
            [{"name": "...", "required": bool, "description": "..."}]
        """
        pass

    @property
    def tags(self) -> list[str]:
        """Optional tags for categorization."""
        return []

    @property
    def category(self) -> str | None:
        """Category for organization. Override in subclasses."""
        return None

    @abstractmethod
    async def render(self, **kwargs) -> str:
        """
        Render prompt with arguments and return content string.

        Args:
            **kwargs: Prompt-specific arguments

        Returns:
            Rendered prompt content string

        Raises:
            Exception: Any errors during rendering
        """
        pass

    async def __call__(self, **kwargs) -> PromptResult:
        """
        Wrapper that handles error handling.

        Args:
            **kwargs: Prompt-specific arguments

        Returns:
            PromptResult with success status and rendered content
        """
        try:
            content = await self.render(**kwargs)
            return PromptResult(success=True, content=content)
        except Exception as e:
            return PromptResult(
                success=False,
                content="",
                error=str(e),
            )
