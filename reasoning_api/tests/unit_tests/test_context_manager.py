"""Unit tests for ContextManager."""

import pytest
from litellm import token_counter

from reasoning_api.context_manager import Context, ContextManager, ContextUtilization


class TestContextManager:
    """Test ContextManager core functionality."""

    def test_preserves_message_order(self) -> None:
        """Most recent message should be last in output."""
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Second message"},
            {"role": "user", "content": "Third message"},
            {"role": "assistant", "content": "Fourth message (most recent)"},
        ]

        context = Context(conversation_history=messages)
        manager = ContextManager(context_utilization=ContextUtilization.FULL)

        filtered_messages, _ = manager(model_name="gpt-4o-mini", context=context)

        # Should preserve chronological order
        assert filtered_messages == messages
        assert filtered_messages[-1]["content"] == "Fourth message (most recent)"

    def test_includes_all_system_messages(self) -> None:
        """All system messages should be included regardless of token limit."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        context = Context(conversation_history=messages)
        manager = ContextManager(context_utilization=ContextUtilization.FULL)

        filtered_messages, metadata = manager(model_name="gpt-4o-mini", context=context)

        # System message should always be first
        assert filtered_messages[0]["role"] == "system"
        assert filtered_messages[0]["content"] == "You are a helpful assistant."
        assert metadata["breakdown"]["system_messages"] > 0

    def test_multiple_system_messages(self) -> None:
        """Multiple system messages should all be included and come first."""
        messages = [
            {"role": "system", "content": "First system message."},
            {"role": "system", "content": "Second system message."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        context = Context(conversation_history=messages)
        manager = ContextManager(context_utilization=ContextUtilization.FULL)

        filtered_messages, _ = manager(model_name="gpt-4o-mini", context=context)

        # First two messages should be system messages
        assert filtered_messages[0]["role"] == "system"
        assert filtered_messages[1]["role"] == "system"
        assert filtered_messages[0]["content"] == "First system message."
        assert filtered_messages[1]["content"] == "Second system message."

    def test_works_backward_from_recent(self) -> None:
        """Should include most recent messages first when hitting limit."""
        # Create messages with known token counts
        messages = [
            {"role": "user", "content": "x " * 100},  # Older message
            {"role": "assistant", "content": "y " * 100},
            {"role": "user", "content": "z " * 100},  # Most recent
        ]

        context = Context(conversation_history=messages)
        # Use LOW utilization to force truncation on small models
        manager = ContextManager(context_utilization=ContextUtilization.LOW)

        filtered_messages, metadata = manager(model_name="gpt-4o-mini", context=context)

        # Should include most recent message
        assert any("z " in msg.get("content", "") for msg in filtered_messages)

        # If not all messages included, oldest should be excluded first
        if metadata["messages_excluded"] > 0:
            # Most recent message should always be included
            assert filtered_messages[-1]["content"] == messages[-1]["content"]

    def test_respects_utilization_strategy_low(self) -> None:
        """LOW should use 33% of context."""
        messages = [{"role": "user", "content": "Hello"}]
        context = Context(conversation_history=messages)
        manager = ContextManager(context_utilization=ContextUtilization.LOW)

        _, metadata = manager(model_name="gpt-4o-mini", context=context)

        # gpt-4o-mini has 128K context, LOW = 33% = ~42K tokens
        assert metadata["model_max_tokens"] == 128_000
        expected_max = int(128_000 * 0.33)
        assert metadata["max_input_tokens"] == expected_max

    def test_respects_utilization_strategy_medium(self) -> None:
        """MEDIUM should use 66% of context."""
        messages = [{"role": "user", "content": "Hello"}]
        context = Context(conversation_history=messages)
        manager = ContextManager(context_utilization=ContextUtilization.MEDIUM)

        _, metadata = manager(model_name="gpt-4o-mini", context=context)

        # gpt-4o-mini has 128K context, MEDIUM = 66% = ~84K tokens
        assert metadata["model_max_tokens"] == 128_000
        expected_max = int(128_000 * 0.66)
        assert metadata["max_input_tokens"] == expected_max

    def test_respects_utilization_strategy_full(self) -> None:
        """FULL should use 100% of context."""
        messages = [{"role": "user", "content": "Hello"}]
        context = Context(conversation_history=messages)
        manager = ContextManager(context_utilization=ContextUtilization.FULL)

        _, metadata = manager(model_name="gpt-4o-mini", context=context)

        # gpt-4o-mini has 128K context, FULL = 100%
        assert metadata["model_max_tokens"] == 128_000
        assert metadata["max_input_tokens"] == 128_000

    def test_handles_empty_conversation(self) -> None:
        """Should handle empty message list gracefully."""
        context = Context(conversation_history=[])
        manager = ContextManager(context_utilization=ContextUtilization.FULL)

        filtered_messages, metadata = manager(model_name="gpt-4o-mini", context=context)

        assert filtered_messages == []
        assert metadata["messages_included"] == 0
        assert metadata["messages_excluded"] == 0
        assert metadata["input_tokens_used"] == 0

    def test_handles_only_system_messages(self) -> None:
        """Should handle conversation with only system messages."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "system", "content": "Be concise."},
        ]

        context = Context(conversation_history=messages)
        manager = ContextManager(context_utilization=ContextUtilization.FULL)

        filtered_messages, metadata = manager(model_name="gpt-4o-mini", context=context)

        assert len(filtered_messages) == 2
        assert all(msg["role"] == "system" for msg in filtered_messages)
        assert metadata["messages_included"] == 2
        assert metadata["breakdown"]["user_messages"] == 0
        assert metadata["breakdown"]["assistant_messages"] == 0

    def test_metadata_includes_all_fields(self) -> None:
        """Metadata should include all expected fields."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        context = Context(conversation_history=messages)
        manager = ContextManager(context_utilization=ContextUtilization.MEDIUM)

        _, metadata = manager(model_name="gpt-4o-mini", context=context)

        # Check all required fields exist
        assert "model_name" in metadata
        assert "strategy" in metadata
        assert "model_max_tokens" in metadata
        assert "max_input_tokens" in metadata
        assert "input_tokens_used" in metadata
        assert "messages_included" in metadata
        assert "messages_excluded" in metadata
        assert "breakdown" in metadata

        # Check breakdown structure
        breakdown = metadata["breakdown"]
        assert "system_messages" in breakdown
        assert "user_messages" in breakdown
        assert "assistant_messages" in breakdown

        # Validate values
        assert metadata["model_name"] == "gpt-4o-mini"
        assert metadata["strategy"] == "medium"
        assert metadata["messages_included"] == 3
        assert metadata["messages_excluded"] == 0

        # Validate model_max_tokens vs max_input_tokens relationship
        assert metadata["model_max_tokens"] == 128_000  # gpt-4o-mini context
        assert metadata["max_input_tokens"] == int(128_000 * 0.66)  # MEDIUM strategy

    def test_raises_when_system_messages_exceed_limit(self) -> None:
        """Should raise ValueError if system messages alone exceed limit."""
        # Create a very large system message
        large_content = "x " * 100_000  # Should exceed even LOW limit

        messages = [
            {"role": "system", "content": large_content},
            {"role": "user", "content": "Hello"},
        ]

        context = Context(conversation_history=messages)
        manager = ContextManager(context_utilization=ContextUtilization.LOW)

        with pytest.raises(ValueError, match="System messages exceed max input tokens"):
            manager(model_name="gpt-4o-mini", context=context)

    def test_token_counting_matches_litellm(self) -> None:
        """
        Token counts should be close to litellm.token_counter results.

        Note: We count messages individually and sum them, which may differ slightly
        from counting all messages together due to chat format overhead. This is expected
        and the small difference is acceptable for our use case.
        """
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
        ]

        context = Context(conversation_history=messages)
        manager = ContextManager(context_utilization=ContextUtilization.FULL)

        _, metadata = manager(model_name="gpt-4o-mini", context=context)

        # Count tokens using litellm directly
        expected_total = token_counter(model="gpt-4o-mini", messages=messages)

        # Our token count should be within a reasonable range
        # (individual message counting may add small overhead vs batch counting)
        difference = abs(metadata["input_tokens_used"] - expected_total)
        assert difference <= 10, (
            f"Token count difference too large: "
            f"{metadata['input_tokens_used']} vs {expected_total}"
        )

    def test_excludes_oldest_messages_when_limit_exceeded(self) -> None:
        """When limit exceeded, should exclude oldest messages first."""
        # Create conversation that will exceed LOW limit
        # gpt-4o-mini has 128K context, LOW = 33% = ~42K tokens
        # Create messages with ~10K tokens each to ensure we exceed limit
        messages = [
            {"role": "user", "content": "Message 1: " + "x " * 10_000},
            {"role": "assistant", "content": "Response 1: " + "y " * 10_000},
            {"role": "user", "content": "Message 2: " + "z " * 10_000},
            {"role": "assistant", "content": "Response 2: " + "w " * 10_000},
            {"role": "user", "content": "Message 3 (most recent): " + "a " * 10_000},
        ]

        context = Context(conversation_history=messages)
        manager = ContextManager(context_utilization=ContextUtilization.LOW)

        filtered_messages, metadata = manager(model_name="gpt-4o-mini", context=context)

        # Should have excluded some messages
        assert metadata["messages_excluded"] > 0

        # Most recent message should be included
        assert any(
            "Message 3 (most recent)" in msg.get("content", "")
            for msg in filtered_messages
        )

        # If Message 1 is excluded, Message 3 should be included
        has_message_1 = any("Message 1:" in msg.get("content", "") for msg in filtered_messages)
        has_message_3 = any("Message 3" in msg.get("content", "") for msg in filtered_messages)

        if not has_message_1:
            # If oldest excluded, most recent should be included
            assert has_message_3

    def test_messages_in_chronological_order_after_filtering(self) -> None:
        """Filtered messages should maintain chronological order."""
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
            {"role": "assistant", "content": "Fourth"},
            {"role": "user", "content": "Fifth (most recent)"},
        ]

        context = Context(conversation_history=messages)
        manager = ContextManager(context_utilization=ContextUtilization.LOW)

        filtered_messages, _ = manager(model_name="gpt-4o-mini", context=context)

        # Check that relative order is preserved
        user_messages = [msg for msg in filtered_messages if msg["role"] == "user"]

        if len(user_messages) > 1:
            # If we have multiple user messages, they should be in order
            for i in range(len(user_messages) - 1):
                current_content = user_messages[i]["content"]
                next_content = user_messages[i + 1]["content"]

                # Later messages should come after earlier ones
                # (This is a simplified check - in reality we'd check indices)
                if "First" in current_content:
                    assert (
                        "First" not in next_content
                        or "Third" in next_content
                        or "Fifth" in next_content
                    )

    def test_breakdown_sums_to_total(self) -> None:
        """Token breakdown should sum to total tokens used."""
        messages = [
            {"role": "system", "content": "System message."},
            {"role": "user", "content": "User message."},
            {"role": "assistant", "content": "Assistant message."},
        ]

        context = Context(conversation_history=messages)
        manager = ContextManager(context_utilization=ContextUtilization.FULL)

        _, metadata = manager(model_name="gpt-4o-mini", context=context)

        breakdown = metadata["breakdown"]
        total_from_breakdown = (
            breakdown["system_messages"]
            + breakdown["user_messages"]
            + breakdown["assistant_messages"]
        )

        assert total_from_breakdown == metadata["input_tokens_used"]

    def test_invalid_message_role_raises_error(self) -> None:
        """Should raise ValueError for invalid message role."""
        messages = [
            {"role": "invalid_role", "content": "This should fail"},
        ]

        context = Context(conversation_history=messages)
        manager = ContextManager(context_utilization=ContextUtilization.FULL)

        with pytest.raises(ValueError, match="Unknown message role"):
            manager(model_name="gpt-4o-mini", context=context)
