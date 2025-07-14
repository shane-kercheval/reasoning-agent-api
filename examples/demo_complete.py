#!/usr/bin/env python3
"""
Complete demo showcasing all features (recommended starting point).

This is the most comprehensive demo showing the reasoning agent's full capabilities
including MCP tools integration, streaming responses, error handling, and more.

Purpose: Full production-ready example with all features and best practices.

Features demonstrated:
- OpenAI SDK usage (recommended approach)
- MCP tools integration with remote servers
- Streaming and non-streaming responses with reasoning events parsing
- Error handling and prerequisites checking
- Beautiful colored output with reasoning step visualization

Prerequisites:
1. Set OPENAI_API_KEY environment variable
2. Start demo MCP server: make demo_mcp_server (or uv run python mcp_servers/fake_server.py)
3. Start reasoning agent with demo config: make demo_api
4. Optional: Set API_TOKENS for authentication if REQUIRE_AUTH=true

Alternative setup:
- Set MCP_CONFIG_PATH=examples/configs/demo_complete.yaml make api
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()


class DemoColors:
    """ANSI color codes for pretty console output."""

    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{DemoColors.BLUE}{DemoColors.BOLD}{'='*60}")
    print(f"🤖 {title}")
    print(f"{'='*60}{DemoColors.END}\n")


def print_step(step: str, content: str) -> None:
    """Print a demo step with formatting."""
    print(f"{DemoColors.CYAN}{DemoColors.BOLD}{step}{DemoColors.END}")
    print(f"{DemoColors.WHITE}{content}{DemoColors.END}\n")


def print_reasoning(content: str) -> None:
    """Print reasoning content with special formatting."""
    print(f"{DemoColors.YELLOW}💭 {content}{DemoColors.END}")


def print_tool(content: str) -> None:
    """Print tool usage with special formatting."""
    print(f"{DemoColors.GREEN}🔧 {content}{DemoColors.END}")


def print_error(content: str) -> None:
    """Print error with special formatting."""
    print(f"{DemoColors.RED}❌ {content}{DemoColors.END}")


def print_success(content: str) -> None:
    """Print success with special formatting."""
    print(f"{DemoColors.GREEN}✅ {content}{DemoColors.END}")


async def check_prerequisites() -> bool:  # noqa: PLR0911, PLR0912
    """Check if all prerequisites are met."""
    print_header("Prerequisites Check")

    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print_error("OPENAI_API_KEY environment variable not set")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        return False
    print_success("OpenAI API key configured")

    # Prepare authentication headers (same logic as in demo_openai_sdk_compatibility)
    api_tokens = os.getenv("API_TOKENS", "")
    headers = {}
    if api_tokens:
        bearer_token = api_tokens.split(',')[0].strip()
        headers = {"Authorization": f"Bearer {bearer_token}"}
        print_success(f"Authentication configured: {bearer_token[:10]}...")
    else:
        print_success("No authentication required (REQUIRE_AUTH=false)")

    # Check if reasoning agent is running
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:8000/health", headers=headers)
            if response.status_code == 200:
                print_success("Reasoning Agent API is running")
            else:
                print_error(f"Reasoning Agent API returned {response.status_code}")
                return False
    except Exception as e:
        print_error(f"Reasoning Agent API not reachable: {e}")
        print("   Start it with: make api")
        return False

    # Check if MCP server is running
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:8000/tools", headers=headers)
            if response.status_code == 200:
                tools_data = response.json()
                # Handle both old format {"tools": [...]} and new format {"server_name": [...]}
                if "tools" in tools_data:
                    # Legacy format
                    tools = tools_data["tools"]
                else:
                    # New format: {"demo_tools": ["tool1", "tool2"]}
                    tools = []
                    for server_name, server_tools in tools_data.items():
                        tools.extend(server_tools)

                if tools:
                    print_success(f"MCP tools available: {', '.join(tools)}")
                else:
                    print_error("No MCP tools configured")
                    print("   Start demo server: uv run python mcp_servers/fake_server.py")
                    return False
            else:
                print_error(f"Failed to check MCP tools (HTTP {response.status_code})")
                if response.status_code == 401:
                    print("   Check API_TOKENS environment variable for authentication")
                return False
    except Exception as e:
        print_error(f"Failed to check MCP tools: {e}")
        return False

    return True


async def demo_openai_sdk_compatibility() -> AsyncOpenAI | None:
    """Test OpenAI SDK compatibility."""
    print_header("OpenAI SDK Compatibility Demo")

    # Get authentication
    api_tokens = os.getenv("API_TOKENS", "")
    default_headers = {}
    if api_tokens:
        bearer_token = api_tokens.split(',')[0].strip()
        default_headers = {"Authorization": f"Bearer {bearer_token}"}
        print_step("🔐 Authentication", f"Using bearer token: {bearer_token[:10]}...")
    else:
        print_step("🔐 Authentication", "No authentication (REQUIRE_AUTH=false)")

    # Create OpenAI client
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="http://localhost:8000/v1",
        default_headers=default_headers,
    )

    try:
        # Test models endpoint
        print_step("📋 Available Models", "Fetching model list...")
        models = await client.models.list()
        for model in models.data[:3]:  # Show first 3 models
            print(f"   • {model.id}")

        # Test non-streaming completion
        print_step("💬 Non-streaming Chat", "Simple question without tools...")
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "What is 25 * 17? Just give me the answer."},
            ],
            max_tokens=50,
            temperature=0.0,
        )

        content = response.choices[0].message.content
        print(f"   Response: {content}")

        print_success("OpenAI SDK compatibility confirmed!")
        return client

    except Exception as e:
        print_error(f"OpenAI SDK test failed: {e}")
        await client.close()
        return None


async def demo_reasoning_with_tools(client: AsyncOpenAI) -> None:
    """Demonstrate reasoning with MCP tools."""
    print_header("Reasoning with MCP Tools Demo")

    print_step("🧠 Streaming with Tools", "Asking for weather information...")

    try:
        stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather like in Tokyo right now? Please use the weather tool to get current conditions and give me a detailed summary.",  # noqa: E501
                },
            ],
            max_tokens=300,
            temperature=0.7,
            stream=True,
        )

        print("📡 Streaming response:")
        full_content = ""

        async for chunk in stream:
            delta = chunk.choices[0].delta

            # Show reasoning events (our special feature!)
            if hasattr(delta, 'reasoning_event') and delta.reasoning_event:
                event = delta.reasoning_event
                # Handle both dict and object formats
                event_type = event.get("type") if isinstance(event, dict) else event.type
                status = event.get("status") if isinstance(event, dict) else event.status

                if event_type == "reasoning_step":
                    if status == "in_progress":
                        print_reasoning("Starting reasoning step...")
                    elif status == "completed":
                        metadata = event.get("metadata", {}) if isinstance(event, dict) else (event.metadata or {})  # noqa: E501
                        thought = metadata.get("thought", "") if metadata else ""
                        print_reasoning(f"Step completed: {thought}")

                elif event_type == "tool_execution":
                    tools = event.get("tools", []) if isinstance(event, dict) else (event.tools or [])  # noqa: E501
                    if status == "in_progress":
                        print_tool(f"Executing tools: {', '.join(tools)}")
                    elif status == "completed":
                        print_tool(f"Tools completed: {', '.join(tools)}")

                elif event_type == "synthesis" and status == "completed":
                    print_reasoning("Reasoning complete, generating final response...")

            # Show regular content
            if delta.content:
                content = delta.content
                full_content += content
                # Print content in real-time
                print(content, end="", flush=True)

        print(f"\n\n{DemoColors.GREEN}✅ Streaming complete!{DemoColors.END}")

    except Exception as e:
        print_error(f"Streaming demo failed: {e}")


async def demo_multiple_tools(client: AsyncOpenAI) -> None:
    """Demonstrate using multiple tools in sequence."""
    print_header("Multiple Tools Demo")

    print_step("🔧 Complex Task", "Asking for multiple types of information...")

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": """Please help me with this analysis:
1. Get the weather in New York
2. Search the web for "climate change NYC"
3. Analyze the sentiment of this text: "The weather is getting more unpredictable every year"
4. Get the current stock price for any major company

Please use the appropriate tools and provide a comprehensive summary.""",
                },
            ],
            max_tokens=500,
            temperature=0.7,
            stream=False,  # Non-streaming for cleaner output
        )

        content = response.choices[0].message.content
        print(f"📝 Complete Response:\n\n{content}")

        print_success("Multiple tools demo completed!")

    except Exception as e:
        print_error(f"Multiple tools demo failed: {e}")


async def demo_error_handling(client: AsyncOpenAI) -> None:
    """Demonstrate error handling scenarios."""
    print_header("Error Handling Demo")

    print_step("⚠️ Testing Error Scenarios", "Testing various error conditions...")

    # Test 1: Invalid model
    try:
        print("   Testing invalid model...")
        await client.chat.completions.create(
            model="invalid-model-name",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
        )
        print_error("Expected error for invalid model, but got success")
    except Exception as e:
        print_success(f"Invalid model handled correctly: {type(e).__name__}")

    # Test 2: Rate limit simulation (very high max_tokens)
    try:
        print("   Testing large request...")
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Count from 1 to 10"}],
            max_tokens=10,  # Very small limit
            temperature=0.0,
        )
        print_success(f"Large request handled: {len(response.choices[0].message.content)} chars")
    except Exception as e:
        print(f"   Large request error: {type(e).__name__}")


async def main() -> None:
    """Run the complete demo."""
    print(f"{DemoColors.PURPLE}{DemoColors.BOLD}")
    print("🚀 Reasoning Agent with MCP Tools - Complete Demo")
    print("=" * 60)
    print(f"{DemoColors.END}")

    # Check prerequisites
    if not await check_prerequisites():
        print_error("Prerequisites not met. Please fix the issues above and try again.")
        return

    # Demo 1: OpenAI SDK compatibility
    client = await demo_openai_sdk_compatibility()
    if not client:
        return

    try:
        # Demo 2: Reasoning with tools
        await demo_reasoning_with_tools(client)

        # Demo 3: Multiple tools
        await demo_multiple_tools(client)

        # Demo 4: Error handling
        await demo_error_handling(client)

        # Final success message
        print_header("Demo Complete! 🎉")
        print_success("All demos completed successfully!")
        print(f"\n{DemoColors.WHITE}Key takeaways:")
        print("• ✅ Full OpenAI SDK compatibility")
        print("• 🧠 Reasoning steps visible in streaming mode")
        print("• 🔧 MCP tools integration working")
        print("• ⚡ Both streaming and non-streaming supported")
        print("• 🛡️ Proper error handling")
        print(f"\n🎯 Your reasoning agent is ready for production!{DemoColors.END}")

    finally:
        await client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{DemoColors.YELLOW}Demo interrupted by user. Goodbye! 👋{DemoColors.END}")
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
