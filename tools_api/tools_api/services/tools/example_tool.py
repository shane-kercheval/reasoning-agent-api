"""Example tool for testing and demonstration."""

from typing import Any

from pydantic import BaseModel, Field

from tools_api.services.base import BaseTool


class EchoResult(BaseModel):
    """Result from the echo tool."""

    echo: str = Field(description="The echoed message")
    length: int = Field(description="Length of the message in characters")
    reversed: str = Field(description="The message reversed")


class EchoTool(BaseTool):
    """
    Example tool that echoes input back.

    Useful for testing the tool execution pipeline.
    """

    @property
    def name(self) -> str:
        """Tool name."""
        return "echo"

    @property
    def description(self) -> str:
        """Tool description."""
        return "Echo the input message back with metadata"

    @property
    def parameters(self) -> dict[str, Any]:
        """Tool parameters JSON Schema."""
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to echo",
                },
            },
            "required": ["message"],
        }

    @property
    def result_model(self) -> type[BaseModel]:
        """Pydantic model for the tool result."""
        return EchoResult

    @property
    def tags(self) -> list[str]:
        """Tool semantic tags."""
        return ["example", "test"]

    @property
    def category(self) -> str | None:
        """Tool category."""
        return "example"

    async def _execute(self, message: str) -> EchoResult:
        """
        Echo the message back with metadata.

        Args:
            message: Message to echo

        Returns:
            EchoResult with echo, length, and reversed message
        """
        return EchoResult(
            echo=message,
            length=len(message),
            reversed=message[::-1],
        )
