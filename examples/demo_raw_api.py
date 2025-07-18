#!/usr/bin/env python3
"""
Low-level demo using raw HTTP API (without OpenAI SDK).

This demo shows how the reasoning agent API works under the hood by making
direct HTTP requests. It's useful for understanding the API format, building
custom clients, or debugging.

Purpose: Educational - see the raw API requests/responses and reasoning events.

Prerequisites:
- Set OPENAI_API_KEY environment variable
- Start demo MCP server: make demo_mcp_server
- Start reasoning agent with demo config:
    - `MCP_CONFIG_PATH=examples/configs/demo_raw_api.json make api`

Note: For production use, see demo_complete.py which uses the OpenAI SDK.
"""

import asyncio
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from api.openai_protocol import OpenAIChatRequest

from dotenv import load_dotenv
load_dotenv()


async def main():  # noqa
    """Simple demo of the reasoning agent."""
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: Set OPENAI_API_KEY environment variable")
        return

    # Create HTTP client with auth
    auth_token = os.getenv("API_TOKEN", "token1")
    client = httpx.AsyncClient(
        timeout=60.0,
        headers={"Authorization": f"Bearer {auth_token}"},
    )

    try:
        # Test 1: Simple non-streaming request
        print("🧠 Test 1: Simple reasoning (non-streaming)")
        print("-" * 50)

        request = OpenAIChatRequest(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 15 * 24? Show your work."}],
            stream=False,
        )

        response = await client.post(
            "http://localhost:8000/v1/chat/completions",
            json=request.model_dump(exclude_unset=True),
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ Response:", result["choices"][0]["message"]["content"])
        else:
            print("❌ Error:", response.status_code, response.text)

        print("\n" + "=" * 60 + "\n")

        # Test 2: Streaming request with reasoning events
        print("🧠 Test 2: Streaming with reasoning events")
        print("-" * 50)

        request = OpenAIChatRequest(
            model="gpt-4o",  # Better support for structured outputs
            messages=[{"role": "user", "content": "I need to plan a weekend trip to Tokyo. First research the current weather, then find the population, then search for top tourist attractions. Show me your step-by-step reasoning process."}],  # noqa: E501
            stream=True,
        )

        async with client.stream(
            "POST",
            "http://localhost:8000/v1/chat/completions",
            json=request.model_dump(exclude_unset=True),
        ) as response:

            if response.status_code != 200:
                print("❌ Error:", response.status_code, await response.aread())
                return

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})

                            # Show reasoning events
                            reasoning_event = delta.get("reasoning_event")
                            if reasoning_event:
                                event_type = reasoning_event.get("type")
                                status = reasoning_event.get("status")
                                step_id = reasoning_event.get("step_id", "?")

                                if event_type == "reasoning_step" and status == "completed":
                                    thought = reasoning_event.get("metadata", {}).get("thought", "")  # noqa: E501
                                    print(f"💭 Step {step_id}: {thought}")

                                elif event_type == "tool_execution":
                                    tools = reasoning_event.get("tools", [])
                                    if status == "in_progress":
                                        print(f"🔧 Using tools: {', '.join(tools)}")
                                    elif status == "completed":
                                        print(f"✅ Tools completed: {', '.join(tools)}")

                            # Show regular content
                            content = delta.get("content")
                            if content:
                                print(content, end="", flush=True)

                    except json.JSONDecodeError:
                        continue

        print("\n\n" + "=" * 60 + "\n")

        # Test 3: Check available tools
        print("🔧 Test 3: Available tools")
        print("-" * 50)

        tools_response = await client.get("http://localhost:8000/tools")
        if tools_response.status_code == 200:
            tools_data = tools_response.json()
            tools = tools_data.get("tools", []) if isinstance(tools_data, dict) else tools_data
            if tools:
                print("Available tools:", ", ".join(tools))
            else:
                print("No tools configured")
        else:
            print("❌ Tools endpoint error:", tools_response.status_code)

        print("\n" + "=" * 60 + "\n")

        # Test 4: Actually use tools (if available)
        print("🔧 Test 4: Using tools with reasoning")
        print("-" * 50)

        if tools:
            print("Requesting a task that should use the available tools...")

            request = OpenAIChatRequest(
                model="gpt-4o",
                messages=[{"role": "user", "content": "What's the weather like in Tokyo right now? Use the weather tool to get current conditions."}],  # noqa: E501
                stream=True,
            )

            async with client.stream(
                "POST",
                "http://localhost:8000/v1/chat/completions",
                json=request.model_dump(exclude_unset=True),
            ) as response:

                if response.status_code != 200:
                    print("❌ Error:", response.status_code, await response.aread())
                    return

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})

                                # Show reasoning events
                                reasoning_event = delta.get("reasoning_event")
                                if reasoning_event:
                                    event_type = reasoning_event.get("type")
                                    status = reasoning_event.get("status")
                                    step_id = reasoning_event.get("step_id", "?")

                                    if event_type == "reasoning_step" and status == "completed":
                                        thought = reasoning_event.get("metadata", {}).get("thought", "")  # noqa: E501
                                        print(f"💭 Step {step_id}: {thought}")

                                    elif event_type == "tool_execution":
                                        tools_used = reasoning_event.get("tools", [])
                                        if status == "in_progress":
                                            print(f"🔧 Executing tools: {', '.join(tools_used)}")
                                        elif status == "completed":
                                            print(f"✅ Tools completed: {', '.join(tools_used)}")

                                # Show regular content
                                content = delta.get("content")
                                if content:
                                    print(content, end="", flush=True)

                        except json.JSONDecodeError:
                            continue
        else:
            print("No tools available - skipping tool usage test")

    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
