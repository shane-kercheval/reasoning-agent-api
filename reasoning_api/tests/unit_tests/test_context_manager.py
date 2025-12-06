"""Unit tests for ContextManager."""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from litellm import token_counter

from reasoning_api.context_manager import Context, ContextManager, ContextUtilization, ContextGoal
from reasoning_api.reasoning_models import ReasoningStepRecord, ReasoningAction, ToolPrediction
from reasoning_api.tools import ToolResult

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


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


class TestBuildReasoningContext:
    """Test reasoning context building via unified __call__ interface."""

    def test_empty_step_records(self) -> None:
        """Should handle empty step_records list."""
        manager = ContextManager(context_utilization=ContextUtilization.FULL)
        messages = [{"role": "user", "content": "What is the weather?"}]

        context = Context(
            conversation_history=messages,
            step_records=[],
            goal=ContextGoal.PLANNING,
            system_prompt_override="You are a reasoning agent.",
        )
        filtered_messages, metadata = manager("gpt-4o-mini", context)

        # Should have system prompt and user message
        assert len(filtered_messages) == 2
        assert filtered_messages[0]["role"] == "system"
        assert filtered_messages[0]["content"] == "You are a reasoning agent."
        assert filtered_messages[1]["role"] == "user"
        assert filtered_messages[1]["content"] == "What is the weather?"
        assert metadata["goal"] == "planning"

    def test_with_step_records_no_tools(self) -> None:
        """Should include reasoning steps without tool results."""
        manager = ContextManager(context_utilization=ContextUtilization.FULL)
        messages = [{"role": "user", "content": "What is 2+2?"}]

        step_records = [
            ReasoningStepRecord(
                step_index=0,
                thought="I need to calculate 2+2",
                next_action=ReasoningAction.CONTINUE_THINKING,
                tool_predictions=[],
                tool_results=[],
            ),
        ]

        context = Context(
            conversation_history=messages,
            step_records=step_records,
            goal=ContextGoal.PLANNING,
            system_prompt_override="You are a reasoning agent.",
        )
        filtered_messages, metadata = manager("gpt-4o-mini", context)

        # Should have system, user, and reasoning context
        assert len(filtered_messages) == 3
        assert filtered_messages[0]["role"] == "system"
        assert filtered_messages[1]["role"] == "user"
        assert filtered_messages[2]["role"] == "assistant"
        # New format groups step info together
        reasoning_content = filtered_messages[2]["content"]
        assert "### Step 1" in reasoning_content
        assert "**Thought:** I need to calculate 2+2" in reasoning_content
        assert metadata["goal"] == "planning"

    def test_with_step_records_and_tool_results(self) -> None:
        """Should include both reasoning steps and tool results."""
        manager = ContextManager(context_utilization=ContextUtilization.FULL)
        messages = [{"role": "user", "content": "What is the weather in Tokyo?"}]

        step_records = [
            ReasoningStepRecord(
                step_index=0,
                thought="I need to check the weather",
                next_action=ReasoningAction.USE_TOOLS,
                tool_predictions=[
                    ToolPrediction(
                        tool_name="get_weather",
                        arguments={"location": "Tokyo"},
                        reasoning="Get weather data",
                    ),
                ],
                tool_results=[
                    ToolResult(
                        tool_name="get_weather",
                        success=True,
                        result={"temperature": "22°C", "condition": "Sunny"},
                        execution_time_ms=150.0,
                    ),
                ],
            ),
        ]

        context = Context(
            conversation_history=messages,
            step_records=step_records,
            goal=ContextGoal.SYNTHESIS,
            system_prompt_override="You are a reasoning agent.",
        )
        filtered_messages, metadata = manager("gpt-4o-mini", context)

        # Should have system, user, and combined reasoning context (single message)
        assert len(filtered_messages) == 3
        assert filtered_messages[0]["role"] == "system"
        assert filtered_messages[1]["role"] == "user"
        # New format groups everything per step in a single assistant message
        reasoning_content = filtered_messages[2]["content"]
        assert "### Step 1" in reasoning_content
        assert "**Thought:** I need to check the weather" in reasoning_content
        assert "**get_weather**" in reasoning_content
        assert "✅ SUCCESS" in reasoning_content
        assert '"temperature": "22°C"' in reasoning_content
        assert metadata["goal"] == "synthesis"

    def test_with_failed_tool_results(self) -> None:
        """Should include failed tool results with error messages."""
        manager = ContextManager(context_utilization=ContextUtilization.FULL)
        messages = [{"role": "user", "content": "Search for something"}]

        step_records = [
            ReasoningStepRecord(
                step_index=0,
                thought="I need to search",
                next_action=ReasoningAction.USE_TOOLS,
                tool_predictions=[
                    ToolPrediction(
                        tool_name="search",
                        arguments={"query": "test"},
                        reasoning="Search for info",
                    ),
                ],
                tool_results=[
                    ToolResult(
                        tool_name="search",
                        success=False,
                        error="Connection timeout",
                        execution_time_ms=5000.0,
                    ),
                ],
            ),
        ]

        context = Context(
            conversation_history=messages,
            step_records=step_records,
            goal=ContextGoal.PLANNING,
        )
        filtered_messages, _ = manager("gpt-4o-mini", context)

        # Should include the error message paired with its tool
        reasoning_content = filtered_messages[-1]["content"]
        assert "- Result: ❌ FAILED - Connection timeout" in reasoning_content
        assert "- **search**" in reasoning_content

    def test_multiple_steps_accumulation(self) -> None:
        """Should accumulate context from multiple reasoning steps."""
        manager = ContextManager(context_utilization=ContextUtilization.FULL)
        messages = [{"role": "user", "content": "Complex question"}]

        step_records = [
            ReasoningStepRecord(
                step_index=0,
                thought="First step thinking",
                next_action=ReasoningAction.USE_TOOLS,
                tool_predictions=[
                    ToolPrediction(
                        tool_name="tool1",
                        arguments={"arg": "value1"},
                        reasoning="Reason 1",
                    ),
                ],
                tool_results=[
                    ToolResult(
                        tool_name="tool1",
                        success=True,
                        result="Result 1",
                        execution_time_ms=100.0,
                    ),
                ],
            ),
            ReasoningStepRecord(
                step_index=1,
                thought="Second step thinking",
                next_action=ReasoningAction.USE_TOOLS,
                tool_predictions=[
                    ToolPrediction(
                        tool_name="tool2",
                        arguments={"arg": "value2"},
                        reasoning="Reason 2",
                    ),
                ],
                tool_results=[
                    ToolResult(
                        tool_name="tool2",
                        success=True,
                        result="Result 2",
                        execution_time_ms=100.0,
                    ),
                ],
            ),
        ]

        context = Context(
            conversation_history=messages,
            step_records=step_records,
            goal=ContextGoal.SYNTHESIS,
        )
        filtered_messages, _ = manager("gpt-4o-mini", context)

        # Should have user and combined reasoning (single assistant message)
        reasoning_content = filtered_messages[1]["content"]
        # New format groups everything per step
        assert "### Step 1" in reasoning_content
        assert "**Thought:** First step thinking" in reasoning_content
        assert "### Step 2" in reasoning_content
        assert "**Thought:** Second step thinking" in reasoning_content
        assert "**tool1**" in reasoning_content
        assert "**tool2**" in reasoning_content
        assert "✅ SUCCESS" in reasoning_content

    def test_planning_goal_metadata(self) -> None:
        """Should include goal in metadata for PLANNING."""
        manager = ContextManager(context_utilization=ContextUtilization.FULL)
        messages = [{"role": "user", "content": "Test"}]

        context = Context(
            conversation_history=messages,
            step_records=[],
            goal=ContextGoal.PLANNING,
        )
        _, metadata = manager("gpt-4o-mini", context)

        assert metadata["goal"] == "planning"

    def test_synthesis_goal_metadata(self) -> None:
        """Should include goal in metadata for SYNTHESIS."""
        manager = ContextManager(context_utilization=ContextUtilization.FULL)
        messages = [{"role": "user", "content": "Test"}]

        context = Context(
            conversation_history=messages,
            step_records=[],
            goal=ContextGoal.SYNTHESIS,
        )
        _, metadata = manager("gpt-4o-mini", context)

        assert metadata["goal"] == "synthesis"

    def test_no_system_prompt_override_preserves_existing(self) -> None:
        """Should preserve existing system messages when no override provided."""
        manager = ContextManager(context_utilization=ContextUtilization.FULL)
        messages = [
            {"role": "system", "content": "Existing system prompt"},
            {"role": "user", "content": "Test"},
        ]

        context = Context(
            conversation_history=messages,
            step_records=[],
            goal=ContextGoal.PLANNING,
            system_prompt_override=None,  # No override - preserve existing
        )
        filtered_messages, _ = manager("gpt-4o-mini", context)

        # Should preserve the existing system message
        assert len(filtered_messages) == 2
        assert filtered_messages[0]["role"] == "system"
        assert filtered_messages[0]["content"] == "Existing system prompt"

    def test_system_prompt_override_replaces_existing(self) -> None:
        """Should replace existing system messages with override."""
        manager = ContextManager(context_utilization=ContextUtilization.FULL)
        messages = [
            {"role": "system", "content": "Old system prompt"},
            {"role": "user", "content": "Test"},
        ]

        context = Context(
            conversation_history=messages,
            step_records=[],
            goal=ContextGoal.PLANNING,
            system_prompt_override="New system prompt",
        )
        filtered_messages, _ = manager("gpt-4o-mini", context)

        # Should have new system prompt, not old one
        assert filtered_messages[0]["role"] == "system"
        assert filtered_messages[0]["content"] == "New system prompt"
        assert "Old system prompt" not in str(filtered_messages)

    def test_applies_token_limit(self) -> None:
        """Should apply token limit management."""
        manager = ContextManager(context_utilization=ContextUtilization.LOW)
        messages = [{"role": "user", "content": "Test"}]

        context = Context(
            conversation_history=messages,
            step_records=[],
            goal=ContextGoal.PLANNING,
        )
        _, metadata = manager("gpt-4o-mini", context)

        # Should have LOW utilization applied
        assert metadata["strategy"] == "low"
        assert metadata["max_input_tokens"] == int(128_000 * 0.33)


class TestGoalAwareFiltering:
    """Test goal-aware context filtering with preview()."""

    def test_planning_uses_preview_for_large_results(self) -> None:
        """PLANNING goal should use preview for large tool results."""
        manager = ContextManager(context_utilization=ContextUtilization.FULL)
        messages = [{"role": "user", "content": "Test"}]

        # Create a large tool result (over PREVIEW_THRESHOLD of 2000 chars)
        large_data = {"key_" + str(i): "value_" * 50 for i in range(50)}

        step_records = [
            ReasoningStepRecord(
                step_index=0,
                thought="Testing large result",
                next_action=ReasoningAction.USE_TOOLS,
                tool_predictions=[
                    ToolPrediction(
                        tool_name="large_tool",
                        arguments={},
                        reasoning="Get large data",
                    ),
                ],
                tool_results=[
                    ToolResult(
                        tool_name="large_tool",
                        success=True,
                        result=large_data,
                        execution_time_ms=100.0,
                    ),
                ],
            ),
        ]

        context = Context(
            conversation_history=messages,
            step_records=step_records,
            goal=ContextGoal.PLANNING,
        )
        filtered_messages, _ = manager("gpt-4o-mini", context)

        # Should indicate it's a preview
        reasoning_content = filtered_messages[-1]["content"]
        assert "SUCCESS (preview)" in reasoning_content
        # Should NOT have all 50 keys (preview limits to 5 items with PREVIEW_MAX_ITEMS)
        assert "key_49" not in reasoning_content

    def test_synthesis_uses_full_content(self) -> None:
        """SYNTHESIS goal should use full tool results."""
        manager = ContextManager(context_utilization=ContextUtilization.FULL)
        messages = [{"role": "user", "content": "Test"}]

        # Create a large tool result
        large_data = {"key_" + str(i): "value_" + str(i) for i in range(50)}

        step_records = [
            ReasoningStepRecord(
                step_index=0,
                thought="Testing large result",
                next_action=ReasoningAction.USE_TOOLS,
                tool_predictions=[
                    ToolPrediction(
                        tool_name="large_tool",
                        arguments={},
                        reasoning="Get large data",
                    ),
                ],
                tool_results=[
                    ToolResult(
                        tool_name="large_tool",
                        success=True,
                        result=large_data,
                        execution_time_ms=100.0,
                    ),
                ],
            ),
        ]

        context = Context(
            conversation_history=messages,
            step_records=step_records,
            goal=ContextGoal.SYNTHESIS,
        )
        filtered_messages, _ = manager("gpt-4o-mini", context)

        # Should NOT indicate preview
        reasoning_content = filtered_messages[-1]["content"]
        assert "(preview)" not in reasoning_content
        # Should have all 50 keys
        assert "key_49" in reasoning_content

    def test_errors_are_always_full(self) -> None:
        """Error messages should always be shown in full regardless of goal."""
        manager = ContextManager(context_utilization=ContextUtilization.FULL)
        messages = [{"role": "user", "content": "Test"}]

        # Create a long error message
        long_error = "Error: " + "x" * 2000

        step_records = [
            ReasoningStepRecord(
                step_index=0,
                thought="Testing error",
                next_action=ReasoningAction.USE_TOOLS,
                tool_predictions=[
                    ToolPrediction(
                        tool_name="failing_tool",
                        arguments={},
                        reasoning="This will fail",
                    ),
                ],
                tool_results=[
                    ToolResult(
                        tool_name="failing_tool",
                        success=False,
                        error=long_error,
                        execution_time_ms=100.0,
                    ),
                ],
            ),
        ]

        # Test with PLANNING goal
        context = Context(
            conversation_history=messages,
            step_records=step_records,
            goal=ContextGoal.PLANNING,
        )
        filtered_messages, _ = manager("gpt-4o-mini", context)

        reasoning_content = filtered_messages[-1]["content"]
        # Full error should be present (not truncated)
        assert long_error in reasoning_content
        assert "FAILED" in reasoning_content

    def test_small_results_not_previewed(self) -> None:
        """Small results should not be previewed even for PLANNING goal."""
        manager = ContextManager(context_utilization=ContextUtilization.FULL)
        messages = [{"role": "user", "content": "Test"}]

        # Create a small tool result (under PREVIEW_THRESHOLD of 2000 chars)
        small_data = {"temperature": "22°C", "condition": "Sunny"}

        step_records = [
            ReasoningStepRecord(
                step_index=0,
                thought="Testing small result",
                next_action=ReasoningAction.USE_TOOLS,
                tool_predictions=[
                    ToolPrediction(
                        tool_name="weather",
                        arguments={"location": "Tokyo"},
                        reasoning="Get weather",
                    ),
                ],
                tool_results=[
                    ToolResult(
                        tool_name="weather",
                        success=True,
                        result=small_data,
                        execution_time_ms=100.0,
                    ),
                ],
            ),
        ]

        context = Context(
            conversation_history=messages,
            step_records=step_records,
            goal=ContextGoal.PLANNING,
        )
        filtered_messages, _ = manager("gpt-4o-mini", context)

        # Should NOT indicate preview for small results
        reasoning_content = filtered_messages[-1]["content"]
        assert "(preview)" not in reasoning_content
        assert "22°C" in reasoning_content

    def test_planning_context_smaller_than_synthesis(self) -> None:
        """PLANNING context should be smaller than SYNTHESIS for large results."""
        manager = ContextManager(context_utilization=ContextUtilization.FULL)
        messages = [{"role": "user", "content": "Test"}]

        # Create a large tool result
        large_data = {"key_" + str(i): "value_" * 100 for i in range(100)}

        step_records = [
            ReasoningStepRecord(
                step_index=0,
                thought="Testing large result",
                next_action=ReasoningAction.USE_TOOLS,
                tool_predictions=[
                    ToolPrediction(
                        tool_name="large_tool",
                        arguments={},
                        reasoning="Get large data",
                    ),
                ],
                tool_results=[
                    ToolResult(
                        tool_name="large_tool",
                        success=True,
                        result=large_data,
                        execution_time_ms=100.0,
                    ),
                ],
            ),
        ]

        planning_context = Context(
            conversation_history=messages,
            step_records=step_records,
            goal=ContextGoal.PLANNING,
        )
        planning_messages, _ = manager("gpt-4o-mini", planning_context)

        synthesis_context = Context(
            conversation_history=messages,
            step_records=step_records,
            goal=ContextGoal.SYNTHESIS,
        )
        synthesis_messages, _ = manager("gpt-4o-mini", synthesis_context)

        # Planning context should be smaller
        planning_size = len(str(planning_messages))
        synthesis_size = len(str(synthesis_messages))
        assert planning_size < synthesis_size

    @patch("reasoning_api.context_manager.MIN_SIZE_TO_PREVIEW", 2000)
    def test_string_result_truncation_for_planning(self) -> None:
        """Large string results should be truncated for PLANNING goal."""
        manager = ContextManager(context_utilization=ContextUtilization.FULL)
        messages = [{"role": "user", "content": "Test"}]

        # Create a large string result (over patched MIN_SIZE_TO_PREVIEW of 2000 chars)
        large_string = "x" * 3000

        step_records = [
            ReasoningStepRecord(
                step_index=0,
                thought="Testing large string",
                next_action=ReasoningAction.USE_TOOLS,
                tool_predictions=[
                    ToolPrediction(
                        tool_name="string_tool",
                        arguments={},
                        reasoning="Get large string",
                    ),
                ],
                tool_results=[
                    ToolResult(
                        tool_name="string_tool",
                        success=True,
                        result=large_string,
                        execution_time_ms=100.0,
                    ),
                ],
            ),
        ]

        context = Context(
            conversation_history=messages,
            step_records=step_records,
            goal=ContextGoal.PLANNING,
        )
        filtered_messages, _ = manager("gpt-4o-mini", context)

        reasoning_content = filtered_messages[-1]["content"]
        # Should indicate truncation
        assert "(truncated)" in reasoning_content
        assert "3000 chars total" in reasoning_content


class TestContextManagerSnapshot:
    """Snapshot test to track context manager output format over time."""

    @patch("reasoning_api.context_manager.MIN_SIZE_TO_PREVIEW", 1000)
    def test_realistic_reasoning_context_snapshot(self) -> None:
        """
        Build realistic reasoning context and dump to YAML for easy viewing and change tracking.

        This test creates a multi-step reasoning scenario with various tool results
        to capture the full output format. Check the artifact file to see the actual
        context being built and diff changes over time.

        Note: MIN_SIZE_TO_PREVIEW is patched to 2000 to trigger preview behavior
        with the test data (production default is higher).
        """
        manager = ContextManager(context_utilization=ContextUtilization.FULL)

        # Realistic conversation history
        conversation = [
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": "What are the latest features in Python 3.14?"},
        ]

        # Realistic multi-step reasoning with various tool results
        # Intentionally includes:
        # - >10 web results to test PREVIEW_MAX_ITEMS truncation
        # - Large web_scraper text (>2000 chars) to test PREVIEW_THRESHOLD
        step_records = [
            # Step 1: Web search with many results (tests PREVIEW_MAX_ITEMS)
            ReasoningStepRecord(
                step_index=0,
                thought="I need to search for the latest features in Python 3.14 to provide accurate and up-to-date information.",
                next_action=ReasoningAction.USE_TOOLS,
                tool_predictions=[
                    ToolPrediction(
                        tool_name="web_search",
                        arguments={"q": "Python 3.14 new features 2025"},
                        reasoning="Search for recent articles about Python 3.14 features",
                    ),
                ],
                tool_results=[
                    ToolResult(
                        tool_name="web_search",
                        success=True,
                        result={
                            "query": {"original": "Python 3.14 new features 2025"},
                            "web_results": [
                                {
                                    "title": "What's new in Python 3.14 - Official Docs",
                                    "url": "https://docs.python.org/3/whatsnew/3.14.html",
                                    "description": "The CPython runtime supports running multiple copies of Python in the same process simultaneously...",
                                },
                                {
                                    "title": "Python 3.14: Cool New Features – Real Python",
                                    "url": "https://realpython.com/python314-new-features/",
                                    "description": "Tab completion now extends to import statements...",
                                },
                                {
                                    "title": "The best new features in Python 3.14 | InfoWorld",
                                    "url": "https://www.infoworld.com/article/python-3-14.html",
                                    "description": "Official support for free-threaded Python, experimental JIT...",
                                },
                                {
                                    "title": "Python 3.14 Release Notes | Python.org",
                                    "url": "https://www.python.org/downloads/release/python-3140/",
                                    "description": "Python 3.14.0 is the newest major release of the Python programming language...",
                                },
                                {
                                    "title": "Free-threaded Python: What You Need to Know",
                                    "url": "https://medium.com/python-free-threading",
                                    "description": "A deep dive into Python 3.14's free-threading support and how to use it...",
                                },
                                {
                                    "title": "Python 3.14 JIT Compiler Explained",
                                    "url": "https://dev.to/python-jit-314",
                                    "description": "Understanding the new experimental JIT compiler in Python 3.14...",
                                },
                                {
                                    "title": "Migrating to Python 3.14: A Complete Guide",
                                    "url": "https://blog.example.com/python-314-migration",
                                    "description": "Step-by-step guide to upgrading your projects to Python 3.14...",
                                },
                                {
                                    "title": "Python 3.14 Performance Benchmarks",
                                    "url": "https://benchmarks.python.org/314",
                                    "description": "Comprehensive performance comparisons between Python 3.13 and 3.14...",
                                },
                                {
                                    "title": "New Typing Features in Python 3.14",
                                    "url": "https://typing.python.org/314",
                                    "description": "TypeGuard improvements, new type syntax, and more typing enhancements...",
                                },
                                {
                                    "title": "Python 3.14 REPL Improvements",
                                    "url": "https://repl.python.org/314",
                                    "description": "The interactive shell gets tab completion for imports and better error messages...",
                                },
                                {
                                    "title": "Python 3.14 on Windows: New Installation Options",
                                    "url": "https://windows.python.org/314",
                                    "description": "Improved Windows installer with new configuration options...",
                                },
                                {
                                    "title": "Python 3.14 Security Updates",
                                    "url": "https://security.python.org/314",
                                    "description": "Security improvements and vulnerability fixes in Python 3.14...",
                                },
                            ],
                            "total_results": 12,
                        },
                        execution_time_ms=245.3,
                    ),
                ],
            ),
            # Step 2: Scrape multiple URLs (tests multiple tools formatting)
            # One succeeds with large text (>1000 chars to test preview), one fails
            ReasoningStepRecord(
                step_index=1,
                thought="I found several relevant articles. Let me scrape the official Python documentation and Real Python for comprehensive information.",
                next_action=ReasoningAction.USE_TOOLS,
                tool_predictions=[
                    ToolPrediction(
                        tool_name="web_scraper",
                        arguments={"url": "https://docs.python.org/3/whatsnew/3.14.html"},
                        reasoning="Get detailed information from official Python docs",
                    ),
                    ToolPrediction(
                        tool_name="web_scraper",
                        arguments={"url": "https://realpython.com/python314-new-features/"},
                        reasoning="Get practical examples from Real Python tutorial",
                    ),
                ],
                tool_results=[
                    ToolResult(
                        tool_name="web_scraper",
                        success=True,
                        result={
                            "url": "https://docs.python.org/3/whatsnew/3.14.html",
                            "final_url": "https://docs.python.org/3/whatsnew/3.14.html",
                            "status": 200,
                            "title": "What's New In Python 3.14",
                            "text": """What's New In Python 3.14

This article explains the new features in Python 3.14, compared to 3.13. For full details, see the changelog.

Summary – Release Highlights
============================

Python 3.14 includes several significant improvements:

- Free-threaded CPython: The GIL can now be disabled at runtime
- JIT compiler: Experimental just-in-time compilation for performance
- Improved REPL: Tab completion for imports and better error messages
- New typing features: TypeGuard improvements and new type syntax
- Performance improvements: 10-15% faster on average benchmarks

New Features
============

Free-threading Support (PEP 703)
--------------------------------
The CPython runtime now officially supports free-threaded mode where the Global Interpreter Lock (GIL) can be disabled. This enables true parallel execution of Python threads on multi-core systems.

To enable free-threading, use: python3.14 -X gil=0 script.py

Or set the environment variable: PYTHON_GIL=0

Benefits:
- True multi-threaded parallelism for CPU-bound tasks
- Better utilization of multi-core processors
- Improved performance for concurrent workloads

Caveats:
- Some C extensions may not be thread-safe
- Memory usage may increase slightly
- Not all workloads will see improvements

JIT Compiler (PEP 744)
----------------------
An experimental JIT (Just-In-Time) compiler has been added that can significantly improve performance for compute-intensive workloads.

To enable the JIT compiler: python3.14 -X jit script.py

The JIT compiler works by:
1. Identifying hot code paths during execution
2. Compiling them to optimized machine code
3. Replacing the interpreted bytecode with native code

Early benchmarks show 10-30% improvements for numerical code.

Improved Interactive Shell
--------------------------
The REPL now supports tab completion for import statements and provides better error messages with syntax highlighting.

New features include:
- Tab completion for module names after 'import' and 'from'
- Syntax highlighting for error messages
- Multi-line editing improvements
- Better history navigation

Typing Enhancements
-------------------
Several improvements to the typing module:
- TypeGuard now supports narrowing in negative branches
- New TypeForm for runtime type introspection
- Improved TypeVar bounds checking

Deprecations and Removals
=========================
- asyncio.get_event_loop() now raises DeprecationWarning
- Various legacy codecs removed
- distutils fully removed (use setuptools instead)""",
                            "word_count": 380,
                            "links_count": 67,
                            "images_count": 0,
                            "tables_count": 2,
                        },
                        execution_time_ms=1523.7,
                    ),
                    ToolResult(
                        tool_name="web_scraper",
                        success=False,
                        error="Connection timeout after 30 seconds",
                        execution_time_ms=30000.0,
                    ),
                ],
            ),
            # Step 3: Continue reasoning after mixed results
            ReasoningStepRecord(
                step_index=2,
                thought="The Real Python scrape timed out, but I have comprehensive information from the official docs. Let me also check InfoWorld for additional perspective.",
                next_action=ReasoningAction.USE_TOOLS,
                tool_predictions=[
                    ToolPrediction(
                        tool_name="web_scraper",
                        arguments={"url": "https://www.infoworld.com/article/python-3-14.html"},
                        reasoning="Get additional coverage from InfoWorld",
                    ),
                ],
                tool_results=[
                    ToolResult(
                        tool_name="web_scraper",
                        success=True,
                        result={
                            "url": "https://www.infoworld.com/article/python-3-14.html",
                            "status": 200,
                            "title": "Python 3.14: The best new features",
                            "text": "Python 3.14 brings exciting improvements including free-threading and JIT compilation.",
                            "word_count": 12,
                        },
                        execution_time_ms=892.1,
                    ),
                ],
            ),
            # Step 4: Continue with synthesis decision
            ReasoningStepRecord(
                step_index=3,
                thought="Despite the timeout on Real Python, I have sufficient information from the official docs to provide a comprehensive answer about Python 3.14 features.",
                next_action=ReasoningAction.FINISHED,
                tool_predictions=[],
                tool_results=[],
            ),
        ]

        # Test PLANNING context (what the agent sees when deciding next step)
        planning_context = Context(
            conversation_history=conversation,
            step_records=step_records,
            goal=ContextGoal.PLANNING,
            system_prompt_override="You are a reasoning agent. Analyze the situation and decide the next step.",
        )
        planning_messages, planning_metadata = manager("gpt-4o-mini", planning_context)

        # Test SYNTHESIS context (what the agent sees when generating final answer)
        synthesis_context = Context(
            conversation_history=conversation,
            step_records=step_records,
            goal=ContextGoal.SYNTHESIS,
            system_prompt_override="You are a helpful assistant. Synthesize a clear answer from the reasoning steps.",
        )
        synthesis_messages, synthesis_metadata = manager("gpt-4o-mini", synthesis_context)

        # Minimal assertions - just verify we got non-empty results
        assert len(planning_messages) > 0
        assert len(synthesis_messages) > 0
        assert planning_metadata
        assert synthesis_metadata

        # Build output structure for YAML
        output = {
            "description": "Context Manager output snapshot for tracking format changes",
            "planning_context": {
                "metadata": planning_metadata,
                "messages": planning_messages,
            },
            "synthesis_context": {
                "metadata": synthesis_metadata,
                "messages": synthesis_messages,
            },
        }

        # Dump to YAML artifact with literal block style for multiline strings
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        artifact_path = ARTIFACTS_DIR / "context_manager_output.yaml"

        # Custom representer for readable multiline strings
        def str_representer(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
            if "\n" in data:
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            return dumper.represent_scalar("tag:yaml.org,2002:str", data)

        yaml.add_representer(str, str_representer)

        with artifact_path.open("w") as f:
            yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
