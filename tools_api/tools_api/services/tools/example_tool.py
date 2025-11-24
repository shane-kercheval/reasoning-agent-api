"""Example tool for testing and demonstration."""

from typing import Any

from tools_api.services.base import BaseTool


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
    def tags(self) -> list[str]:
        """Tool semantic tags."""
        return ["example", "test"]

    async def _execute(self, message: str) -> dict[str, Any]:
        """
        Echo the message back with metadata.

        Args:
            message: Message to echo

        Returns:
            Dict with echo and length
        """
        return {
            "echo": message,
            "length": len(message),
            "reversed": message[::-1],
        }
