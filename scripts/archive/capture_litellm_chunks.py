"""
Capture actual LiteLLM streaming response chunks for validation.

This script sends a real request to LiteLLM and captures the actual chunk structure
so we can validate our types and mocks match reality.
"""

import asyncio
import traceback
import json
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import from api
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import litellm  # noqa: E402


async def capture_litellm_chunks() -> int:  # noqa: PLR0915
    """Capture and display actual LiteLLM streaming chunks."""
    print("=" * 80)
    print("CAPTURING LITELLM STREAMING CHUNKS")
    print("=" * 80)
    print()

    # Get API key from environment
    api_key = os.getenv("LITELLM_API_KEY")
    base_url = os.getenv("LITELLM_BASE_URL", "http://localhost:4000")

    if not api_key:
        print("❌ ERROR: LITELLM_API_KEY not set in .env file")
        print("\nPlease set LITELLM_API_KEY in your .env file.")
        print("You may need to create a virtual key first using the LiteLLM UI.")
        return 1

    print(f"Using LiteLLM at: {base_url}")
    print(f"API Key: {api_key[:10]}... (masked)")
    print()

    # Simple test request
    messages = [
        {"role": "user", "content": "Count from 1 to 3. Be brief."},
    ]

    print("Sending request to LiteLLM...")
    print(f"Messages: {json.dumps(messages, indent=2)}")
    print()

    try:
        # Call LiteLLM with streaming
        stream = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            api_key=api_key,
            base_url=base_url,
        )

        print("=" * 80)
        print("RAW CHUNK OBJECTS (type and attributes)")
        print("=" * 80)
        print()

        chunks_captured = []
        chunk_num = 0

        async for chunk in stream:
            chunk_num += 1
            print(f"\n{'=' * 40} CHUNK {chunk_num} {'=' * 40}")

            # Show type
            print(f"Type: {type(chunk)}")
            print(f"Type name: {type(chunk).__name__}")
            print(f"Module: {type(chunk).__module__}")

            # Show all attributes
            print("\nAll attributes:")
            for attr in dir(chunk):
                if not attr.startswith('_'):
                    try:
                        value = getattr(chunk, attr)
                        if not callable(value):
                            print(f"  {attr}: {value!r}")
                    except Exception as e:
                        print(f"  {attr}: <error getting value: {e}>")

            # Show model_dump() if available
            if hasattr(chunk, 'model_dump'):
                print("\nmodel_dump():")
                dumped = chunk.model_dump()
                print(json.dumps(dumped, indent=2, default=str))
                chunks_captured.append(dumped)

            # Show dict conversion
            if hasattr(chunk, '__dict__'):
                print("\n__dict__:")
                print(json.dumps(chunk.__dict__, indent=2, default=str))

        print("\n" + "=" * 80)
        print(f"SUMMARY: Captured {chunk_num} chunks")
        print("=" * 80)

        # Save to file for reference
        output_file = Path(__file__).parent / "litellm_chunks_captured.json"
        with output_file.open("w") as f:
            json.dump({
                "total_chunks": chunk_num,
                "chunks": chunks_captured,
            }, f, indent=2, default=str)

        print(f"\nSaved chunk data to: {output_file}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"Error type: {type(e)}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(capture_litellm_chunks()))
