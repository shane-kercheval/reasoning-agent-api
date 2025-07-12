"""
Reasoning agent stub for the API.

This is a minimal stub implementation that will be extended later with
specific reasoning logic. Currently just passes requests through unchanged.
"""

import asyncio
from collections.abc import AsyncGenerator

from .models import ChatCompletionRequest
from .mcp_client import MCPClient


class ReasoningAgent:
    """Stub reasoning agent that passes requests through unchanged."""

    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client

    async def process_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Process request - currently just returns unchanged."""
        # TODO: Add reasoning logic here
        return request

    async def process_streaming_request(
            self,
            request: ChatCompletionRequest,
        ) -> AsyncGenerator[dict]:
        """Process streaming request with simple progress updates."""
        # Simple progress indicator
        yield {
            "type": "reasoning_step",
            "content": "🔍 Processing request...",
        }

        await asyncio.sleep(0.1)  # Brief pause for demo

        yield {
            "type": "reasoning_step",
            "content": "✅ Ready to generate response",
        }

        yield {
            "type": "enhanced_request",
            "request": request,
        }
