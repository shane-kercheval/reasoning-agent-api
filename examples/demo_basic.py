#!/usr/bin/env python3
"""
Basic demo showing OpenAI SDK compatibility with reasoning events.

This is the simplest demo - it shows how to use the reasoning agent as a drop-in
replacement for OpenAI's API using the official OpenAI Python SDK, while also
demonstrating how to parse and display the enhanced reasoning events.

Purpose: Quick test to verify the API is working and showcases reasoning capabilities.

Prerequisites:
- Set OPENAI_API_KEY environment variable
- Start reasoning agent: make api (or MCP_CONFIG_PATH=examples/configs/demo_basic.json make api)
- Optional: Set API_TOKENS for authentication if REQUIRE_AUTH=true
"""

import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()

async def demo_openai_sdk_usage() -> None:  # noqa: PLR0912, PLR0915
    """Demonstrate using our API with the OpenAI SDK."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        return

    # Get bearer token for our API authentication
    api_tokens = os.getenv("API_TOKENS", "")
    default_headers = {}
    if api_tokens:
        # Use the first token from the list
        bearer_token = api_tokens.split(',')[0].strip()
        default_headers = {"Authorization": f"Bearer {bearer_token}"}

    # Create OpenAI client pointing to our local API
    client = AsyncOpenAI(
        api_key=api_key,  # This is sent to your API, which forwards to OpenAI
        base_url="http://localhost:8000/v1",  # Point to our local server
        default_headers=default_headers,  # Auth for your API
    )

    try:
        print("🤖 Testing OpenAI SDK compatibility with our API\n")

        # Test 1: Non-streaming chat completion
        print("1️⃣ Non-streaming chat completion:")
        response = await client.chat.completions.create(
            model="gpt-4o-mini", # Model doesn't matter - we forward to OpenAI
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the weather like on Mars?"},
            ],
            max_tokens=100,
            temperature=0.7,
        )

        print(f"   Response ID: {response.id}")
        print(f"   Model: {response.model}")
        print(f"   Content: {response.choices[0].message.content}\n")

        # Test 2: Streaming chat completion (shows our reasoning enhancement)
        print("2️⃣ Streaming chat completion (with reasoning steps):")
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Solve this step by step: If a train travels 60 mph for 2.5 hours, how far does it go?"},  # noqa: E501
            ],
            max_tokens=200,
            temperature=0.0,
            stream=True,
        )

        print("   Stream chunks:")
        async for chunk in stream:
            delta = chunk.choices[0].delta

            # Show reasoning events (our special feature!)
            if hasattr(delta, 'reasoning_event') and delta.reasoning_event:
                event = delta.reasoning_event
                # Handle both dict and object formats
                event_type = event.get("type") if isinstance(event, dict) else event.type
                status = event.get("status") if isinstance(event, dict) else event.status

                if event_type == "reasoning_step" and status == "completed":
                    metadata = event.get("metadata", {}) if isinstance(event, dict) else (event.metadata or {})  # noqa: E501
                    thought = metadata.get("thought", "") if metadata else ""
                    print(f"   🧠 Reasoning: {thought}")
                elif event_type == "tool_execution":
                    tools = event.get("tools", []) if isinstance(event, dict) else (event.tools or [])  # noqa: E501
                    metadata = event.get("metadata", {}) if isinstance(event, dict) else (getattr(event, "metadata", {}) or {})  # noqa: E501

                    if status == "in_progress":
                        # Show tool arguments when starting
                        tool_predictions = metadata.get("tool_predictions", [])
                        if tool_predictions:
                            print(f"   🔧 Using tools: {', '.join(tools)}")
                            for prediction in tool_predictions:
                                tool_name = getattr(prediction, "tool_name", prediction.get("tool_name", "unknown"))  # noqa: E501
                                arguments = getattr(prediction, "arguments", prediction.get("arguments", {}))  # noqa: E501
                                if arguments and len(arguments) > 0:
                                    # Show first argument as example
                                    first_key = next(iter(arguments.keys()))
                                    first_value = arguments[first_key]
                                    if isinstance(first_value, str) and len(first_value) > 20:
                                        first_value = first_value[:17] + "..."
                                    print(f"      └─ {tool_name}({first_key}={first_value})")
                                else:
                                    print(f"      └─ {tool_name}()")
                        else:
                            print(f"   🔧 Using tools: {', '.join(tools)}")

                    elif status == "completed":
                        # Show tool results when completed
                        tool_results = metadata.get("tool_results", [])

                        if tool_results:
                            print(f"   ✅ Tools completed: {', '.join(tools)}")
                            for result in tool_results:
                                tool_name = getattr(result, "tool_name", result.get("tool_name", "unknown"))  # noqa: E501
                                if hasattr(result, 'result') and result.result:
                                    result_data = result.result
                                    if isinstance(result_data, dict) and result_data:
                                        # Show first key-value pair as a sample
                                        first_key = next(iter(result_data.keys()))
                                        first_value = result_data[first_key]
                                        if isinstance(first_value, str) and len(first_value) > 30:
                                            first_value = first_value[:27] + "..."
                                        print(f"      └─ {tool_name}: {first_key}={first_value}")
                                    else:
                                        result_str = str(result_data)[:40]
                                        if len(str(result_data)) > 40:
                                            result_str += "..."
                                        print(f"      └─ {tool_name}: {result_str}")
                                else:
                                    print(f"      └─ {tool_name}: completed")
                        else:
                            print(f"   ✅ Tools completed: {', '.join(tools)}")

            # Show regular content
            if delta.content:
                # Print each chunk as it arrives
                content = delta.content
                print(f"   → {content!r}")

        print()

        # Test 3: Models list
        print("3️⃣ Available models:")
        models = await client.models.list()
        for model in models.data:
            print(f"   • {model.id} (owned by {model.owned_by})")

        print("\n✅ All tests completed successfully!")
        print("   Your API is fully compatible with the OpenAI SDK! 🎉")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure:")
        print("   1. Server is running: make api")
        print("   2. OPENAI_API_KEY is set in .env")
        print("   3. API_TOKENS is set in .env (e.g., API_TOKENS=token1,token2,token3)")
        print("   4. REQUIRE_AUTH=true is set in .env")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(demo_openai_sdk_usage())
