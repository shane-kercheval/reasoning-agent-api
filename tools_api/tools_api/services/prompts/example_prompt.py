"""Example prompt for testing and demonstration."""

from typing import Any

from tools_api.services.base import BasePrompt


class GreetingPrompt(BasePrompt):
    """
    Example prompt that generates a greeting.

    Useful for testing the prompt rendering pipeline.
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

    async def render(self, name: str, formal: bool = False, **kwargs) -> list[dict[str, str]]:  # noqa
        """
        Render greeting message.

        Args:
            name: Person's name
            formal: Whether to use formal greeting
            **kwargs: Additional arguments (ignored)

        Returns:
            OpenAI-compatible messages
        """
        if formal:
            greeting = f"Good day, {name}. How may I assist you today?"
        else:
            greeting = f"Hey {name}! What can I help you with?"

        return [
            {"role": "user", "content": greeting},
        ]
