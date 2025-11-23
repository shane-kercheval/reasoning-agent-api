"""
Test token counting performance at different scales.

This script measures the performance of litellm.token_counter() to inform
caching strategy for ContextManager implementation.
"""

import time
import statistics
from litellm import token_counter

# Test with a common model
MODEL = "gpt-4o"

def generate_text_of_approx_tokens(target_tokens: int) -> str:
    """
    Generate text that approximately results in target token count.

    Using rough estimate: 1 token ≈ 4 characters for English text.
    """
    # Rough heuristic: 1 token ≈ 4 characters
    chars_needed = target_tokens * 4

    # Generate repetitive but varied text to approximate token count
    # Use words to make it more realistic than just "a" * n
    base_text = "The quick brown fox jumps over the lazy dog. "
    repetitions = chars_needed // len(base_text)

    return base_text * repetitions


def time_token_counting(text: str, num_runs: int = 10) -> dict:
    """
    Time token counting for a given text.

    Returns:
        dict with timing statistics
    """
    message = {"role": "user", "content": text}

    # Warm-up run
    actual_tokens = token_counter(model=MODEL, messages=[message])

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        token_counter(model=MODEL, messages=[message])
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds

    return {
        "actual_tokens": actual_tokens,
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def test_multiple_messages(num_messages: int, tokens_per_message: int) -> dict:
    """
    Test token counting performance with multiple messages.

    Simulates a conversation history with many messages.
    """
    messages = []
    for i in range(num_messages):
        text = generate_text_of_approx_tokens(tokens_per_message)
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": text})

    # Time counting all messages at once
    start = time.perf_counter()
    total_tokens = token_counter(model=MODEL, messages=messages)
    elapsed_all = (time.perf_counter() - start) * 1000

    # Time counting messages individually
    start = time.perf_counter()
    elapsed_individual = (time.perf_counter() - start) * 1000

    return {
        "num_messages": num_messages,
        "tokens_per_message": tokens_per_message,
        "total_tokens": total_tokens,
        "all_at_once_ms": elapsed_all,
        "individual_ms": elapsed_individual,
        "individual_per_msg_ms": elapsed_individual / num_messages,
    }


def main() -> None:
    """Run token counting performance tests."""
    print("=" * 80)
    print("TOKEN COUNTING PERFORMANCE TEST")
    print(f"Model: {MODEL}")
    print("=" * 80)
    print()

    # Test 1: Single message at different token counts
    print("TEST 1: Single Message at Different Scales")
    print("-" * 80)
    print(f"{'Target Tokens':<15} {'Actual':<10} {'Mean (ms)':<12} {'Median (ms)':<12} {'StdDev':<10}")
    print("-" * 80)

    token_targets = [10, 100, 1_000, 10_000, 100_000]

    for target in token_targets:
        text = generate_text_of_approx_tokens(target)
        result = time_token_counting(text, num_runs=10)

        print(
            f"{target:>13,}  "
            f"{result['actual_tokens']:>8,}  "
            f"{result['mean_ms']:>10.3f}  "
            f"{result['median_ms']:>10.3f}  "
            f"{result['stdev_ms']:>8.3f}",
        )

    print()
    print()

    # Test 2: Multiple messages (simulating conversation history)
    print("TEST 2: Multiple Messages (Conversation History Simulation)")
    print("-" * 80)
    print(f"{'Messages':<12} {'Tokens/Msg':<12} {'Total Tokens':<15} "
          f"{'All-at-Once':<15} {'Individual':<15} {'Per-Msg':<12}")
    print("-" * 80)

    test_cases = [
        (10, 100),      # 10 messages, ~100 tokens each
        (50, 100),      # 50 messages, ~100 tokens each
        (100, 100),     # 100 messages, ~100 tokens each
        (10, 1_000),    # 10 messages, ~1K tokens each
        (50, 1_000),    # 50 messages, ~1K tokens each
        (100, 1_000),   # 100 messages, ~1K tokens each
    ]

    for num_messages, tokens_per_msg in test_cases:
        result = test_multiple_messages(num_messages, tokens_per_msg)

        print(
            f"{result['num_messages']:>10,}  "
            f"{result['tokens_per_message']:>10,}  "
            f"{result['total_tokens']:>13,}  "
            f"{result['all_at_once_ms']:>11.2f} ms  "
            f"{result['individual_ms']:>11.2f} ms  "
            f"{result['individual_per_msg_ms']:>8.2f} ms",
        )

    print()
    print()

    # Test 3: Repeated counting of same message (simulates caching benefit)
    print("TEST 3: Repeated Counting of Same Message (Cache Potential)")
    print("-" * 80)

    text = generate_text_of_approx_tokens(1_000)
    message = {"role": "user", "content": text}

    num_repeats = 100

    start = time.perf_counter()
    for _ in range(num_repeats):
        token_counter(model=MODEL, messages=[message])
    elapsed_total = (time.perf_counter() - start) * 1000

    print(f"Counting same 1K token message {num_repeats} times:")
    print(f"  Total time: {elapsed_total:.2f} ms")
    print(f"  Per-call: {elapsed_total / num_repeats:.3f} ms")
    print()
    print("If caching were perfect (0ms per cached call):")
    print(f"  First call: ~{elapsed_total / num_repeats:.3f} ms")
    print("  Subsequent 99 calls: 0 ms")
    print(f"  Total with cache: ~{elapsed_total / num_repeats:.3f} ms")
    print(f"  Speedup: {num_repeats:.0f}x")

    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("Based on the results above:")
    print("- If per-message time > 1ms and messages are often repeated: USE CACHING")
    print("- If per-message time < 0.1ms: caching overhead may not be worth it")
    print("- Consider caching based on (model, message_content) tuple")
    print("- Monitor cache hit rate in production to validate benefit")
    print()


if __name__ == "__main__":
    main()
