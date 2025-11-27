"""
Example code-based prompt for testing and demonstration.

This shows how to create a prompt class that extends BasePrompt for complex
use cases requiring custom logic. For simple template-based prompts, use
markdown files with YAML frontmatter instead (see examples/prompts/).

NOTE: This prompt is registered by default for backwards compatibility and testing.
For production use, consider using file-based prompts loaded from PROMPTS_DIRECTORY.
"""

from typing import Any

from tools_api.services.base import BasePrompt


class GreetingPrompt(BasePrompt):
    """
    Example prompt that generates a greeting.

    Useful for testing the prompt rendering pipeline. This demonstrates
    how to create code-based prompts for cases requiring dynamic logic
    that cannot be expressed in Jinja2 templates.
    """

    @property
    def name(self) -> str:
        """Prompt name."""
        return "greeting"

    @property
    def description(self) -> str:
        """Prompt description."""
        return "Generate a greeting message"

    @property
    def arguments(self) -> list[dict[str, Any]]:
        """Prompt arguments."""
        return [
            {
                "name": "name",
                "required": True,
                "description": "Person's name to greet",
            },
            {
                "name": "formal",
                "required": False,
                "description": "Whether to use formal greeting",
            },
        ]

    @property
    def tags(self) -> list[str]:
        """Prompt semantic tags."""
        return ["example", "test"]

    @property
    def category(self) -> str | None:
        """Prompt category."""
        return "example"

    async def render(self, name: str, formal: bool = False, **kwargs) -> str:
        """
        Render greeting message.

        Args:
            name: Person's name
            formal: Whether to use formal greeting
            **kwargs: Additional arguments (ignored)

        Returns:
            Rendered greeting content string
        """
        if formal:
            return f"Good day, {name}. How may I assist you today?"
        else:
            return f"Hey {name}! What can I help you with?"
