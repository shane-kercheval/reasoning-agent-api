"""
Tests for the generic prompt abstraction.

This module tests the Prompt class and related utilities to ensure they work
correctly with both sync and async functions, handle errors properly, and
provide a consistent interface for MCP prompts.
"""

import asyncio
import time
import pytest

from api.prompts import (
    Prompt,
    PromptResult,
    format_prompt_for_display,
    format_prompts_for_display,
)


class TestPrompt:
    """Test the Prompt class functionality."""

    def test_prompt_creation(self) -> None:
        """Test basic Prompt object creation."""
        def simple_func(topic: str) -> PromptResult:
            return PromptResult(
                prompt_name="test_prompt",
                success=True,
                messages=[{"role": "user", "content": f"Tell me about {topic}"}],
                execution_time_ms=0,
            )

        prompt = Prompt(
            name="ask_question",
            description="Generate a question",
            arguments=[
                {"name": "topic", "required": True, "description": "Topic to ask about"},
            ],
            function=simple_func,
        )

        assert prompt.name == "ask_question"
        assert prompt.description == "Generate a question"
        assert len(prompt.arguments) == 1
        assert prompt.arguments[0]["name"] == "topic"

    @pytest.mark.asyncio
    async def test_sync_function_execution(self) -> None:
        """Test Prompt execution with synchronous function."""
        def generate_prompt(topic: str) -> PromptResult:
            return PromptResult(
                prompt_name="test_prompt",
                success=True,
                messages=[
                    {"role": "user", "content": f"What is {topic}?"},
                ],
                execution_time_ms=0,
            )

        prompt = Prompt(
            name="test_prompt",
            description="Generate a question",
            arguments=[
                {"name": "topic", "required": True, "description": "The topic"},
            ],
            function=generate_prompt,
        )

        result = await prompt(topic="Python")

        assert isinstance(result, PromptResult)
        assert result.success is True
        assert result.messages is not None
        assert len(result.messages) == 1
        assert result.messages[0]["content"] == "What is Python?"
        assert result.error is None
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_async_function_execution(self) -> None:
        """Test Prompt execution with asynchronous function."""
        async def async_generate_prompt(topic: str, detail_level: str) -> PromptResult:
            await asyncio.sleep(0.01)  # Simulate async work
            content = f"Explain {topic} at {detail_level} level"
            return PromptResult(
                prompt_name="async_prompt",
                success=True,
                messages=[{"role": "user", "content": content}],
                execution_time_ms=0,
            )

        prompt = Prompt(
            name="async_prompt",
            description="Generate detailed question",
            arguments=[
                {"name": "topic", "required": True, "description": "Topic to ask about"},
                {"name": "detail_level", "required": True, "description": "Detail level"},
            ],
            function=async_generate_prompt,
        )

        result = await prompt(topic="Rust", detail_level="beginner")

        assert result.success is True
        assert result.messages is not None
        assert "Explain Rust at beginner level" in result.messages[0]["content"]
        assert result.execution_time_ms >= 10  # Should take at least 10ms due to sleep

    @pytest.mark.asyncio
    async def test_function_exception_handling(self) -> None:
        """Test Prompt handles exceptions properly."""
        def failing_function(should_fail: bool) -> PromptResult:
            if should_fail:
                raise ValueError("This function intentionally failed")
            return PromptResult(
                prompt_name="test_prompt",
                success=True,
                messages=[{"role": "user", "content": "Success"}],
                execution_time_ms=0,
            )

        prompt = Prompt(
            name="failing_prompt",
            description="A prompt that can fail",
            arguments=[
                {"name": "should_fail", "required": True, "description": "Whether to fail"},
            ],
            function=failing_function,
        )

        # Test failure case
        result = await prompt(should_fail=True)
        assert result.success is False
        assert "intentionally failed" in result.error
        assert result.messages is None
        assert result.execution_time_ms > 0

        # Test success case
        result = await prompt(should_fail=False)
        assert result.success is True
        assert result.messages is not None
        assert result.error is None

    @pytest.mark.asyncio
    async def test_input_validation_required_arguments(self) -> None:
        """Test that missing required arguments are caught."""
        def simple_func(required_arg: str) -> PromptResult:
            return PromptResult(
                prompt_name="test_prompt",
                success=True,
                messages=[{"role": "user", "content": required_arg}],
                execution_time_ms=0,
            )

        prompt = Prompt(
            name="test_prompt",
            description="Test required validation",
            arguments=[
                {"name": "required_arg", "required": True, "description": "A required argument"},
            ],
            function=simple_func,
        )

        # Should fail without required argument
        result = await prompt()
        assert result.success is False
        assert "Missing required argument: required_arg" in result.error

    @pytest.mark.asyncio
    async def test_input_validation_unexpected_arguments(self) -> None:
        """Test validation of unexpected arguments."""
        def simple_func(param: str) -> PromptResult:
            return PromptResult(
                prompt_name="test_prompt",
                success=True,
                messages=[{"role": "user", "content": param}],
                execution_time_ms=0,
            )

        prompt = Prompt(
            name="strict_prompt",
            description="Prompt with strict argument validation",
            arguments=[
                {"name": "param", "required": True, "description": "Valid parameter"},
            ],
            function=simple_func,
        )

        # Should fail with unexpected argument
        result = await prompt(param="valid", unexpected="invalid")
        assert result.success is False
        assert "Unexpected arguments: unexpected" in result.error

    @pytest.mark.asyncio
    async def test_optional_arguments(self) -> None:
        """Test prompt with both required and optional arguments."""
        def flexible_func(required: str, optional: str = "default") -> PromptResult:
            return PromptResult(
                prompt_name="test_prompt",
                success=True,
                messages=[{"role": "user", "content": f"{required} - {optional}"}],
                execution_time_ms=0,
            )

        prompt = Prompt(
            name="flexible_prompt",
            description="Prompt with optional arguments",
            arguments=[
                {"name": "required", "required": True, "description": "Required argument"},
                {"name": "optional", "required": False, "description": "Optional argument"},
            ],
            function=flexible_func,
        )

        # Test with only required argument
        result = await prompt(required="test")
        assert result.success is True

        # Test with both arguments
        result = await prompt(required="test", optional="custom")
        assert result.success is True

    def test_to_dict(self) -> None:
        """Test Prompt serialization to dictionary."""
        def dummy_func() -> None:
            pass

        prompt = Prompt(
            name="test_prompt",
            description="Test prompt",
            arguments=[
                {"name": "arg1", "required": True, "description": "First argument"},
                {"name": "arg2", "required": False, "description": "Second argument"},
            ],
            function=dummy_func,
        )

        result = prompt.to_dict()

        assert result == {
            "name": "test_prompt",
            "description": "Test prompt",
            "arguments": [
                {"name": "arg1", "required": True, "description": "First argument"},
                {"name": "arg2", "required": False, "description": "Second argument"},
            ],
        }
        # Function should not be included in serialization
        assert "function" not in result

    @pytest.mark.asyncio
    async def test_multiple_messages_result(self) -> None:
        """Test prompt that returns multiple messages."""
        def multi_message_func(topic: str) -> PromptResult:
            return PromptResult(
                prompt_name="test_prompt",
                success=True,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": f"Tell me about {topic}"},
                ],
                execution_time_ms=0,
            )

        prompt = Prompt(
            name="multi_message_prompt",
            description="Prompt with multiple messages",
            arguments=[
                {"name": "topic", "required": True, "description": "Topic to discuss"},
            ],
            function=multi_message_func,
        )

        result = await prompt(topic="AI")

        assert result.success is True
        assert result.messages is not None
        assert len(result.messages) == 2
        assert result.messages[0]["role"] == "system"
        assert result.messages[1]["role"] == "user"


class TestPromptPerformance:
    """Test Prompt performance characteristics."""

    @pytest.mark.asyncio
    async def test_async_function_concurrency(self) -> None:
        """Test async prompts run concurrently."""
        async def async_delay_prompt(delay: float) -> PromptResult:
            await asyncio.sleep(delay)
            return PromptResult(
                prompt_name="test_prompt",
                success=True,
                messages=[{"role": "user", "content": f"Completed after {delay}s"}],
                execution_time_ms=0,
            )

        prompt = Prompt(
            name="delay_prompt",
            description="Prompt with delay",
            arguments=[
                {"name": "delay", "required": True, "description": "Delay in seconds"},
            ],
            function=async_delay_prompt,
        )

        # Run concurrent async tasks
        start_time = time.time()
        tasks = [prompt(delay=0.1) for _ in range(100)]  # 100ms each
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        # All should succeed
        assert all(r.success for r in results)
        # Should take roughly 100ms concurrently, not 100 * 100ms = 10 seconds
        elapsed = end_time - start_time
        assert elapsed < 0.5  # Should be less than 500ms total


class TestPromptFormatting:
    """Test prompt formatting functions for display."""

    def test_format_prompt_with_required_arguments(self) -> None:
        """Test formatting a prompt with required arguments."""
        def dummy_func() -> None:
            pass

        prompt = Prompt(
            name="ask_question",
            description="Generate a question about a topic",
            arguments=[
                {"name": "topic", "required": True, "description": "Topic to ask about"},
            ],
            function=dummy_func,
        )

        result = format_prompt_for_display(prompt)

        # Check all essential parts are present
        assert "## ask_question" in result
        assert "### Description" in result
        assert "Generate a question about a topic" in result
        assert "### Arguments" in result
        assert "#### Required" in result
        assert "`topic`" in result
        assert "Topic to ask about" in result

    def test_format_prompt_with_optional_arguments(self) -> None:
        """Test formatting a prompt with optional arguments."""
        def dummy_func() -> None:
            pass

        prompt = Prompt(
            name="search_query",
            description="Generate a search query",
            arguments=[
                {"name": "query", "required": True, "description": "Search query text"},
                {"name": "max_results", "required": False, "description": "Maximum results"},
            ],
            function=dummy_func,
        )

        result = format_prompt_for_display(prompt)

        assert "#### Required" in result
        assert "`query`" in result
        assert "Search query text" in result
        assert "#### Optional" in result
        assert "`max_results`" in result
        assert "Maximum results" in result

    def test_format_prompt_no_arguments(self) -> None:
        """Test formatting a prompt with no arguments."""
        def dummy_func() -> None:
            pass

        prompt = Prompt(
            name="get_template",
            description="Get a standard template",
            arguments=[],
            function=dummy_func,
        )

        result = format_prompt_for_display(prompt)

        assert "## get_template" in result
        assert "### Description" in result
        assert "Get a standard template" in result
        assert "No arguments." in result

    def test_format_prompt_without_descriptions(self) -> None:
        """Test formatting a prompt where arguments have no descriptions."""
        def dummy_func() -> None:
            pass

        prompt = Prompt(
            name="simple_prompt",
            description="A simple prompt",
            arguments=[
                {"name": "arg1", "required": True},
                {"name": "arg2", "required": False},
            ],
            function=dummy_func,
        )

        result = format_prompt_for_display(prompt)

        assert "#### Required" in result
        assert "`arg1`" in result
        assert "#### Optional" in result
        assert "`arg2`" in result

    def test_format_multiple_prompts(self) -> None:
        """Test formatting multiple prompts."""
        def dummy_func() -> None:
            pass

        prompt1 = Prompt(
            name="prompt_one",
            description="First prompt",
            arguments=[
                {"name": "param1", "required": True, "description": "First parameter"},
            ],
            function=dummy_func,
        )

        prompt2 = Prompt(
            name="prompt_two",
            description="Second prompt",
            arguments=[
                {"name": "param2", "required": False, "description": "Second parameter"},
            ],
            function=dummy_func,
        )

        result = format_prompts_for_display([prompt1, prompt2])

        # Both prompts should be present
        assert "## prompt_one" in result
        assert "## prompt_two" in result
        assert "First prompt" in result
        assert "Second prompt" in result
        assert "`param1`" in result
        assert "`param2`" in result
        assert "---" in result  # Separator between prompts

    def test_format_empty_prompts_list(self) -> None:
        """Test formatting with no prompts."""
        result = format_prompts_for_display([])

        assert result == "No prompts are currently available."

    def test_formatted_output_structure(self) -> None:
        """Test the overall structure of formatted output."""
        def dummy_func() -> None:
            pass

        prompt = Prompt(
            name="test_prompt",
            description="A test prompt",
            arguments=[
                {"name": "param", "required": True, "description": "A parameter"},
            ],
            function=dummy_func,
        )

        result = format_prompt_for_display(prompt)

        # Should have clear structure with markdown headers
        lines = result.split("\n")
        assert lines[0].startswith("## test_prompt")
        assert "### Description" in result
        assert "### Arguments" in result
        assert "#### Required" in result
        assert "`param`" in result


class TestPromptResult:
    """Test PromptResult model."""

    def test_prompt_result_success(self) -> None:
        """Test creating a successful PromptResult."""
        result = PromptResult(
            prompt_name="test_prompt",
            success=True,
            messages=[{"role": "user", "content": "Test message"}],
            execution_time_ms=15.5,
        )

        assert result.prompt_name == "test_prompt"
        assert result.success is True
        assert result.messages == [{"role": "user", "content": "Test message"}]
        assert result.error is None
        assert result.execution_time_ms == 15.5

    def test_prompt_result_failure(self) -> None:
        """Test creating a failed PromptResult."""
        result = PromptResult(
            prompt_name="test_prompt",
            success=False,
            error="Something went wrong",
            execution_time_ms=10.0,
        )

        assert result.prompt_name == "test_prompt"
        assert result.success is False
        assert result.messages is None
        assert result.error == "Something went wrong"
        assert result.execution_time_ms == 10.0
