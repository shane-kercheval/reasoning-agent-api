"""Base classes for tools and prompts."""

import time
from abc import ABC, abstractmethod
from typing import Any

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
    def tags(self) -> list[str]:
        """Optional tags for categorization."""
        return []

    @abstractmethod
    async def _execute(self, **kwargs) -> Any:
        """
        Execute tool and return structured result.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            Tool-specific data structure (dict, list, str, etc.)

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

    Prompts render templates with arguments and return OpenAI-compatible messages.
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

    @abstractmethod
    async def render(self, **kwargs) -> list[dict[str, str]]:
        """
        Render prompt with arguments and return messages.

        Args:
            **kwargs: Prompt-specific arguments

        Returns:
            OpenAI-compatible messages:
            [{"role": "user", "content": "..."}]

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
            PromptResult with success status and rendered messages
        """
        try:
            messages = await self.render(**kwargs)
            return PromptResult(success=True, messages=messages)
        except Exception as e:
            return PromptResult(
                success=False,
                messages=[],
                error=str(e),
            )
