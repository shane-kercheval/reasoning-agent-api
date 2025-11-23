"""
Test if token_counter(messages=...) ≈ token_counter(text=str(messages))

This would enable simple @lru_cache using stringified messages as keys.
"""

import json
from litellm import token_counter

MODEL = "gpt-4o"


def test_string_vs_messages() -> None:
    """Compare token counting: messages vs stringified messages."""
    print("=" * 80)
    print("TOKEN COUNTING: messages vs str(messages)")
    print("=" * 80)
    print()

    test_cases = [
        # Simple single message
        {
            "name": "Single user message",
            "messages": [{"role": "user", "content": "What is the weather?"}],
        },
        # Multiple messages
        {
            "name": "Conversation (3 messages)",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the weather?"},
                {"role": "assistant", "content": "I don't have access to real-time weather data."},
            ],
        },
        # Longer content
        {
            "name": "Long message (~1K tokens)",
            "messages": [
                {
                    "role": "user",
                    "content": "The quick brown fox jumps over the lazy dog. " * 200,
                },
            ],
        },
        # Many short messages
        {
            "name": "10 short messages",
            "messages": [
                {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
                for i in range(10)
            ],
        },
    ]

    print(f"{'Test Case':<30} {'messages=':<12} {'text=str()':<12} {'text=json()':<12} {'Difference':<15}")
    print("-" * 80)

    for test in test_cases:
        messages = test["messages"]

        # Method 1: Using messages parameter (official way)
        tokens_messages = token_counter(model=MODEL, messages=messages)

        # Method 2: Using str(messages)
        tokens_str = token_counter(model=MODEL, text=str(messages))

        # Method 3: Using json.dumps(messages) (more canonical)
        tokens_json = token_counter(model=MODEL, text=json.dumps(messages))

        diff_str = tokens_str - tokens_messages
        diff_json = tokens_json - tokens_messages

        print(
            f"{test['name']:<30} "
            f"{tokens_messages:>10,}  "
            f"{tokens_str:>10,}  "
            f"{tokens_json:>10,}  "
            f"str:{diff_str:+5} json:{diff_json:+5}",
        )

    print()
    print("=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    print()

    # Detailed test with exact output
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
    ]

    tokens_messages = token_counter(model=MODEL, messages=messages)
    tokens_str = token_counter(model=MODEL, text=str(messages))
    tokens_json = token_counter(model=MODEL, text=json.dumps(messages))

    print("Messages:")
    print(json.dumps(messages, indent=2))
    print()
    print("Results:")
    print(f"  token_counter(messages={len(messages)} msgs) = {tokens_messages} tokens")
    print(f"  token_counter(text=str(messages))        = {tokens_str} tokens")
    print(f"  token_counter(text=json.dumps(messages)) = {tokens_json} tokens")
    print()
    print("String representations:")
    print(f"  str(messages):        {str(messages)[:100]}...")
    print(f"  json.dumps(messages): {json.dumps(messages)[:100]}...")
    print()

    if tokens_messages == tokens_str:
        print("✓ str(messages) gives EXACT same count!")
    elif abs(tokens_messages - tokens_str) <= 5:
        print("⚠ str(messages) is CLOSE but not exact (±5 tokens)")
    else:
        print("✗ str(messages) gives DIFFERENT count - don't use for caching!")

    if tokens_messages == tokens_json:
        print("✓ json.dumps(messages) gives EXACT same count!")
    elif abs(tokens_messages - tokens_json) <= 5:
        print("⚠ json.dumps(messages) is CLOSE but not exact (±5 tokens)")
    else:
        print("✗ json.dumps(messages) gives DIFFERENT count - don't use for caching!")

    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()

    if tokens_messages == tokens_str or tokens_messages == tokens_json:
        print("YES! We can use @lru_cache with stringified messages:")
        print()
        print("from functools import lru_cache")
        print()
        print("@lru_cache(maxsize=1024)")
        print("def count_tokens_cached(model: str, messages_json: str) -> int:")
        print('    """Cache token counts using JSON string as key."""')
        print("    messages = json.loads(messages_json)")
        print("    return token_counter(model=model, messages=messages)")
        print()
        print("# Usage:")
        print("tokens = count_tokens_cached(model, json.dumps(messages))")
    else:
        print("NO - str() and json.dumps() give different token counts.")
        print("Token counting depends on proper message structure formatting.")
        print("We'd need a custom hash-based caching solution.")


if __name__ == "__main__":
    test_string_vs_messages()
