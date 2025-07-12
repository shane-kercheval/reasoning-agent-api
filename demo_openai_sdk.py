#!/usr/bin/env python3
"""
Demo script showing how to use our API with the official OpenAI SDK.

This demonstrates that our API is a drop-in replacement for OpenAI's API.
Run the server first: uv run uvicorn api.main:app --reload --port 8000
"""

import asyncio
import os
from openai import AsyncOpenAI


async def demo_openai_sdk_usage():
    """Demonstrate using our API with the OpenAI SDK."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        return

    # Create OpenAI client pointing to our local API
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="http://localhost:8000/v1"  # Point to our local server
    )

    try:
        print("🤖 Testing OpenAI SDK compatibility with our API\n")

        # Test 1: Non-streaming chat completion
        print("1️⃣ Non-streaming chat completion:")
        response = await client.chat.completions.create(
            model="gpt-4o-mini", # Model doesn't matter - we forward to OpenAI
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the weather like on Mars?"}
            ],
            max_tokens=100,
            temperature=0.7
        )

        print(f"   Response ID: {response.id}")
        print(f"   Model: {response.model}")
        print(f"   Content: {response.choices[0].message.content}\n")

        # Test 2: Streaming chat completion (shows our reasoning enhancement)
        print("2️⃣ Streaming chat completion (with reasoning steps):")
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Count from 1 to 5, one number per line."}
            ],
            max_tokens=50,
            temperature=0.0,
            stream=True
        )

        print("   Stream chunks:")
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                # Print each chunk as it arrives
                content = chunk.choices[0].delta.content
                print(f"   → {repr(content)}")

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
        print("   Make sure the server is running: uv run uvicorn api.main:app --reload --port 8000")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(demo_openai_sdk_usage())