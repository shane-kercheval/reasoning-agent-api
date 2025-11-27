"""Jinja2-based prompt template implementation."""

from pathlib import Path
from typing import Any

from jinja2 import ChainableUndefined, Environment

from tools_api.services.base import BasePrompt


# Shared Jinja2 environment configured for lenient undefined handling
_jinja_env = Environment(undefined=ChainableUndefined)


class PromptTemplate(BasePrompt):
    """Prompt implementation using Jinja2 templates."""

    def __init__(
        self,
        name: str,
        description: str,
        template: str,
        arguments: list[dict[str, Any]] | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        source_path: Path | None = None,
    ) -> None:
        """
        Initialize a Jinja2-based prompt template.

        Args:
            name: Prompt name (used in API endpoint)
            description: Human-readable description
            template: Jinja2 template string
            arguments: List of argument definitions with name, required, description
            category: Optional category for organization
            tags: Optional semantic tags for categorization
            source_path: Optional path to source file for error messages
        """
        self._name = name
        self._description = description
        self._template_str = template
        self._template = _jinja_env.from_string(template)
        self._arguments = arguments or []
        self._category = category
        self._tags = tags or []
        self._source_path = source_path

    @property
    def name(self) -> str:
        """Prompt name (used in API endpoint)."""
        return self._name

    @property
    def description(self) -> str:
        """Prompt description."""
        return self._description

    @property
    def arguments(self) -> list[dict[str, Any]]:
        """Prompt arguments schema."""
        return self._arguments

    @property
    def category(self) -> str | None:
        """Category for organization."""
        return self._category

    @property
    def tags(self) -> list[str]:
        """Optional tags for categorization."""
        return self._tags

    @property
    def source_path(self) -> Path | None:
        """Path to source file (for error messages)."""
        return self._source_path

    async def render(self, **kwargs: Any) -> str:
        """
        Render template with provided arguments.

        Args:
            **kwargs: Template variables

        Returns:
            Rendered template string

        Raises:
            ValueError: If a required argument is missing
        """
        # Validate required arguments are present
        for arg in self._arguments:
            if arg.get("required", False) and arg["name"] not in kwargs:
                location = f" ({self._source_path})" if self._source_path else ""
                raise ValueError(
                    f"Missing required argument '{arg['name']}' for prompt '{self._name}'{location}",
                )

        # Render with Jinja2 (undefined variables become empty string via ChainableUndefined)
        return self._template.render(**kwargs)
