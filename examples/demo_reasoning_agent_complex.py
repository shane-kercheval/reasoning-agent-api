#!/usr/bin/env python3
"""
Demo script showcasing the Reasoning Agent with MCP server integration.

This script demonstrates:
1. Setting up MCP servers for different capabilities
2. Running reasoning tasks with tool orchestration
3. Streaming reasoning events in real-time
4. Parallel tool execution across multiple servers

Prerequisites:
- Set OPENAI_API_KEY environment variable
- Ensure FastAPI server is running (uv run python -m api.main)
- Optional: Set up real MCP servers or use mock servers

Usage:
    python demo_reasoning_agent.py
"""

import asyncio
import json
import os
import sys
from typing import Any

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv
load_dotenv()

from api.models import ChatCompletionRequest, ChatMessage, MessageRole  # noqa: E402

console = Console()

class ReasoningAgentDemo:
    """Demo class for showcasing reasoning agent capabilities."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        # Get auth token from environment or use default
        auth_token = os.getenv("API_TOKEN", "token1")
        self.client = httpx.AsyncClient(
            timeout=60.0,
            headers={"Authorization": f"Bearer {auth_token}"},
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        await self.client.aclose()

    async def check_server_health(self) -> bool:
        """Check if the reasoning agent server is running."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def get_available_tools(self) -> list[str]:
        """Get list of available tools from MCP servers."""
        try:
            response = await self.client.get(f"{self.base_url}/tools")
            if response.status_code == 200:
                data = response.json()
                # Handle both formats: {"tools": [...]} or [...]
                if isinstance(data, dict) and "tools" in data:
                    return data["tools"]
                if isinstance(data, list):
                    return data
                return []
            return []
        except Exception:
            return []

    async def run_reasoning_task(self, query: str, stream: bool = True) -> None:
        """Run a reasoning task and display the results."""
        console.print(f"\n[bold blue]🧠 Running Reasoning Task:[/bold blue] {query}")
        console.print("─" * 80)

        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=MessageRole.USER, content=query)],
            stream=stream,
            temperature=0.7,
        )

        if stream:
            await self._handle_streaming_response(request)
        else:
            await self._handle_non_streaming_response(request)

    async def _handle_streaming_response(self, request: ChatCompletionRequest) -> None:
        """Handle streaming response with real-time reasoning events."""
        reasoning_steps = []
        tool_executions = []
        final_content = []

        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=request.model_dump(exclude_unset=True),
                headers={"Content-Type": "application/json"},
            ) as response:

                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            await self._process_streaming_chunk(
                                chunk, reasoning_steps, tool_executions, final_content,
                            )
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            console.print(f"[red]❌ Error during streaming: {e}[/red]")
            return

        # Display final summary
        await self._display_reasoning_summary(reasoning_steps, tool_executions, final_content)

    async def _process_streaming_chunk(
        self,
        chunk: dict[str, Any],
        reasoning_steps: list[dict[str, Any]],
        tool_executions: list[dict[str, Any]],
        final_content: list[str],
    ) -> None:
        """Process individual streaming chunks and display real-time updates."""
        choices = chunk.get("choices", [])
        if not choices:
            return

        choice = choices[0]
        delta = choice.get("delta", {})

        # Handle reasoning events
        reasoning_event = delta.get("reasoning_event")
        if reasoning_event:
            await self._display_reasoning_event(reasoning_event, reasoning_steps, tool_executions)

        # Handle regular content
        content = delta.get("content")
        if content:
            final_content.append(content)
            # Display content in real-time
            console.print(content, end="")

    async def _display_reasoning_event(
        self,
        event: dict[str, Any],
        reasoning_steps: list[dict[str, Any]],
        tool_executions: list[dict[str, Any]],
    ) -> None:
        """Display reasoning events with appropriate formatting."""
        event_type = event.get("type", "unknown")
        status = event.get("status", "unknown")
        step_id = event.get("step_id", "?")

        if event_type == "reasoning_step":
            if status == "in_progress":
                console.print(f"\n[yellow]🤔 Step {step_id}: Starting reasoning...[/yellow]")
            elif status == "completed":
                thought = event.get("metadata", {}).get("thought", "Thinking...")
                tools_planned = event.get("metadata", {}).get("tools_planned", [])

                reasoning_steps.append({
                    "step_id": step_id,
                    "thought": thought,
                    "tools_planned": tools_planned,
                })

                console.print(f"[green]✅ Step {step_id}:[/green] {thought}")
                if tools_planned:
                    console.print(f"[cyan]🔧 Tools planned: {', '.join(tools_planned)}[/cyan]")

        elif event_type == "tool_execution":
            tools = event.get("tools", [])
            if status == "in_progress":
                console.print(f"[blue]⚡ Executing tools: {', '.join(tools)}[/blue]")
            elif status == "completed":
                tool_executions.append({
                    "step_id": step_id,
                    "tools": tools,
                    "status": "completed",
                })
                console.print(f"[green]✅ Tools completed: {', '.join(tools)}[/green]")

        elif event_type == "synthesis" and status == "completed":
            total_steps = event.get("metadata", {}).get("total_steps", 0)
            console.print(f"\n[magenta]🎯 Reasoning complete! ({total_steps} steps)[/magenta]")
            console.print("[bold]📝 Final Response:[/bold]")

    async def _handle_non_streaming_response(self, request: ChatCompletionRequest) -> None:
        """Handle non-streaming response."""
        request.stream = False

        with console.status("[bold green]Processing reasoning task..."):
            try:
                response = await self.client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=request.model_dump(exclude_unset=True),
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()

                result = response.json()
                choices = result.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "No response")
                    console.print(f"\n[bold]📝 Response:[/bold]\n{content}")

                    # Display usage stats if available
                    usage = result.get("usage")
                    if usage:
                        console.print(f"\n[dim]📊 Usage: {usage}[/dim]")

            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")

    async def _display_reasoning_summary(
        self,
        reasoning_steps: list[dict[str, Any]],
        tool_executions: list[dict[str, Any]],
        final_content: list[str],
    ) -> None:
        """Display a summary of the reasoning process."""
        console.print("\n\n" + "═" * 80)
        console.print("[bold]📋 Reasoning Summary[/bold]")
        console.print("═" * 80)

        # Reasoning steps table
        if reasoning_steps:
            steps_table = Table(title="🧠 Reasoning Steps")
            steps_table.add_column("Step", style="cyan")
            steps_table.add_column("Thought", style="white")
            steps_table.add_column("Tools Planned", style="green")

            for step in reasoning_steps:
                tools_str = ", ".join(step["tools_planned"]) if step["tools_planned"] else "None"
                steps_table.add_row(
                    step["step_id"],
                    step["thought"][:80] + "..." if len(step["thought"]) > 80 else step["thought"],
                    tools_str,
                )

            console.print(steps_table)

        # Tool executions table
        if tool_executions:
            tools_table = Table(title="🔧 Tool Executions")
            tools_table.add_column("Step", style="cyan")
            tools_table.add_column("Tools", style="yellow")
            tools_table.add_column("Status", style="green")

            for execution in tool_executions:
                tools_str = ", ".join(execution["tools"])
                tools_table.add_row(
                    execution["step_id"],
                    tools_str,
                    execution["status"],
                )

            console.print(tools_table)

        # Final response length
        total_chars = sum(len(content) for content in final_content)
        console.print(f"\n[dim]📏 Final response: {total_chars} characters[/dim]")

async def main() -> None:  # noqa: PLR0912, PLR0915
    """Main demo function."""
    console.print(Panel.fit(
        "[bold blue]🤖 Reasoning Agent Demo[/bold blue]\n\n"
        "This demo showcases the sophisticated reasoning agent with MCP integration.\n"
        "The agent will demonstrate multi-step reasoning with tool orchestration.",
        title="Welcome",
        border_style="blue",
    ))

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]❌ Error: OPENAI_API_KEY environment variable not set[/red]")
        console.print("Please set your OpenAI API key: export OPENAI_API_KEY=your_key_here")
        return

    async with ReasoningAgentDemo() as demo:
        # Check server health
        console.print("\n[bold]🔍 Checking server status...[/bold]")

        if not await demo.check_server_health():
            console.print("[red]❌ Server not running![/red]")
            console.print("Please start the server: [cyan]uv run python -m api.main[/cyan]")
            return

        console.print("[green]✅ Server is running![/green]")

        # Get available tools
        console.print("\n[bold]🔧 Checking available MCP tools...[/bold]")
        tools = await demo.get_available_tools()

        if tools:
            tools_table = Table(title="Available MCP Tools")
            tools_table.add_column("Tool", style="yellow")
            tools_table.add_column("Status", style="green")

            for tool in tools:
                tools_table.add_row(tool, "Available")

            console.print(tools_table)
        else:
            console.print("[yellow]⚠️  No MCP tools configured. Demo will show reasoning without tools.[/yellow]")  # noqa: E501
            console.print("To see tool integration, configure MCP servers in config/mcp_servers.yaml")  # noqa: E501

        # Demo scenarios
        demo_scenarios = [
            {
                "title": "🌍 Multi-Step Research Task",
                "query": "Compare the populations of Tokyo and New York City. Tell me which is larger and by how much.",  # noqa: E501
                "description": "Demonstrates parallel tool usage for data gathering and synthesis",
            },
            {
                "title": "🔄 Sequential Reasoning Task",
                "query": "Explain the process of photosynthesis and then describe how it relates to climate change.",  # noqa: E501
                "description": "Shows step-by-step reasoning without tools",
            },
            {
                "title": "🎯 Problem-Solving Task",
                "query": "I need to plan a 3-day trip to Paris. Help me find the top attractions, estimate costs, and suggest an itinerary.",  # noqa: E501
                "description": "Complex planning task that may use multiple tools in sequence",
            },
        ]

        console.print("\n[bold]🎬 Demo Scenarios[/bold]")
        for i, scenario in enumerate(demo_scenarios, 1):
            console.print(f"{i}. [bold]{scenario['title']}[/bold]")
            console.print(f"   {scenario['description']}")

        # Interactive demo
        console.print("\n[bold]🚀 Starting Interactive Demo[/bold]")

        for i, scenario in enumerate(demo_scenarios, 1):
            console.print(f"\n{'='*80}")
            console.print(f"[bold cyan]Demo {i}: {scenario['title']}[/bold cyan]")
            console.print(f"[dim]{scenario['description']}[/dim]")
            console.print('='*80)

            # Ask user if they want to continue
            response = console.input("\n[bold]▶️  Run this demo? (y/n/s for streaming/q to quit): [/bold]")  # noqa: E501

            if response.lower() in ['q', 'quit']:
                break
            if response.lower() in ['n', 'no']:
                continue

            stream_mode = response.lower() in ['s', 'streaming', 'y', 'yes', '']

            try:
                await demo.run_reasoning_task(scenario["query"], stream=stream_mode)
            except KeyboardInterrupt:
                console.print("\n[yellow]⏸️  Demo interrupted by user[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]❌ Error during demo: {e}[/red]")

            # Pause between demos
            if i < len(demo_scenarios):
                console.input("\n[dim]Press Enter to continue to next demo...[/dim]")

        # Custom query option
        console.print(f"\n{'='*80}")
        console.print("[bold cyan]🎯 Custom Query Demo[/bold cyan]")
        console.print("[dim]Try your own reasoning task![/dim]")
        console.print('='*80)

        custom_query = console.input("\n[bold]💭 Enter your custom query (or Enter to skip): [/bold]")  # noqa: E501
        if custom_query.strip():
            stream_choice = console.input("[bold]Stream response? (y/n): [/bold]")
            stream_mode = stream_choice.lower() in ['y', 'yes', '']

            try:
                await demo.run_reasoning_task(custom_query, stream=stream_mode)
            except Exception as e:
                console.print(f"\n[red]❌ Error: {e}[/red]")

    console.print(Panel.fit(
        "[bold green]🎉 Demo Complete![/bold green]\n\n"
        "You've seen the reasoning agent in action with:\n"
        "• Multi-step reasoning with structured outputs\n"
        "• Real-time streaming with reasoning events\n"
        "• MCP tool integration and orchestration\n"
        "• Graceful fallback behavior\n\n"
        "The agent maintains full OpenAI API compatibility while adding\n"
        "sophisticated reasoning capabilities!",
        title="Thank You!",
        border_style="green",
    ))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Demo interrupted. Goodbye![/yellow]")
    except Exception as e:
        console.print(f"\n[red]💥 Unexpected error: {e}[/red]")
