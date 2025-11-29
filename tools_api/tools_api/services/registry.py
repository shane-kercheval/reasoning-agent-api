"""Registry for tools and prompts."""


from typing import ClassVar
from tools_api.services.base import BaseTool, BasePrompt

class ToolRegistry:
    """Central registry for all tools."""

    _tools: ClassVar[dict[str, BaseTool]] = {}

    @classmethod
    def register(cls, tool: BaseTool) -> None:
        """
        Register a tool instance.

        Args:
            tool: Tool instance to register

        Raises:
            ValueError: If tool with same name already registered
        """
        if tool.name in cls._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        cls._tools[tool.name] = tool

    @classmethod
    def get(cls, name: str) -> BaseTool | None:
        """
        Get tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return cls._tools.get(name)

    @classmethod
    def list(cls) -> list[BaseTool]:
        """
        List all registered tools.

        Returns:
            List of all registered tool instances
        """
        return list(cls._tools.values())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tools (useful for testing)."""
        cls._tools.clear()

class PromptRegistry:
    """Central registry for all prompts."""

    _prompts: ClassVar[dict[str, BasePrompt]] = {}

    @classmethod
    def register(cls, prompt: BasePrompt) -> None:
        """
        Register a prompt instance.

        Args:
            prompt: Prompt instance to register

        Raises:
            ValueError: If prompt with same name already registered
        """
        if prompt.name in cls._prompts:
            raise ValueError(f"Prompt '{prompt.name}' is already registered")
        cls._prompts[prompt.name] = prompt

    @classmethod
    def get(cls, name: str) -> BasePrompt | None:
        """
        Get prompt by name.

        Args:
            name: Prompt name

        Returns:
            Prompt instance or None if not found
        """
        return cls._prompts.get(name)

    @classmethod
    def list(cls) -> list[BasePrompt]:
        """
        List all registered prompts.

        Returns:
            List of all registered prompt instances
        """
        return list(cls._prompts.values())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered prompts (useful for testing)."""
        cls._prompts.clear()
