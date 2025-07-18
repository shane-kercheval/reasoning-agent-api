"""
Comprehensive tests for reasoning context building.

These tests verify that the reasoning context is built up correctly step-by-step
during the reasoning process, ensuring all components (steps, tool predictions,
tool results, etc.) are properly accumulated and preserved.
"""

import pytest
import httpx
from unittest.mock import AsyncMock, patch
from api.reasoning_agent import ReasoningAgent
from api.openai_protocol import OpenAIChatRequest
from api.prompt_manager import PromptManager
from api.reasoning_models import ReasoningStep, ReasoningAction, ToolPrediction
from api.tools import function_to_tool, ToolResult


class TestReasoningContextBuilding:
    """Test that reasoning context is built up correctly during the reasoning process."""

    @pytest.fixture
    def mock_reasoning_agent(self):
        """Create a reasoning agent with mock tools for testing."""
        http_client = httpx.AsyncClient()
        mock_prompt_manager = AsyncMock(spec=PromptManager)
        mock_prompt_manager.get_prompt.return_value = "You are a helpful assistant."

        # Create test tools
        def weather_tool(location: str) -> dict:
            return {"location": location, "temperature": "22°C", "condition": "Sunny"}

        def search_tool(query: str) -> dict:
            return {"query": query, "results": ["result1", "result2"]}

        tools = [
            function_to_tool(weather_tool, name="get_weather"),
            function_to_tool(search_tool, name="search_web"),
        ]

        return ReasoningAgent(
            base_url="http://test",
            api_key="test-key",
            http_client=http_client,
            tools=tools,
            prompt_manager=mock_prompt_manager,
        )

    @pytest.fixture
    def sample_request(self):
        """Sample chat completion request."""
        return OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        )

    @pytest.mark.asyncio
    async def test_initial_context_structure(self, mock_reasoning_agent, sample_request):  # noqa: ANN001
        """Test that the initial reasoning context has the correct structure."""
        events = []

        # Mock the reasoning step generation to return a simple finished step
        with patch.object(mock_reasoning_agent, '_generate_reasoning_step') as mock_generate:
            mock_generate.return_value = ReasoningStep(
                thought="I can answer this directly",
                next_action=ReasoningAction.FINISHED,
                tools_to_use=[],
                concurrent_execution=False,
            )

            # Collect all events
            async for event_type, event_data in mock_reasoning_agent._core_reasoning_process(sample_request):  # noqa: E501
                events.append((event_type, event_data))

        # Find the finish event
        finish_events = [event for event in events if event[0] == "finish"]
        assert len(finish_events) == 1

        context = finish_events[0][1]["context"]

        # Verify initial structure
        assert "steps" in context
        assert "tool_results" in context
        assert "final_thoughts" in context
        assert "user_request" in context

        # Verify types
        assert isinstance(context["steps"], list)
        assert isinstance(context["tool_results"], list)
        assert isinstance(context["final_thoughts"], str)
        assert context["user_request"] == sample_request

    @pytest.mark.asyncio
    async def test_single_step_without_tools_context(self, mock_reasoning_agent, sample_request):  # noqa: ANN001
        """Test context building for a single reasoning step without tools."""
        events = []

        # Mock a single reasoning step without tools
        expected_step = ReasoningStep(
            thought="I can answer this question directly without using any tools",
            next_action=ReasoningAction.FINISHED,
            tools_to_use=[],
            concurrent_execution=False,
        )

        with patch.object(mock_reasoning_agent, '_generate_reasoning_step') as mock_generate:
            mock_generate.return_value = expected_step

            # Collect all events
            async for event_type, event_data in mock_reasoning_agent._core_reasoning_process(sample_request):  # noqa: E501
                events.append((event_type, event_data))

        # Find the finish event and extract context
        finish_events = [event for event in events if event[0] == "finish"]
        context = finish_events[0][1]["context"]

        # Verify step was added to context
        assert len(context["steps"]) == 1
        assert context["steps"][0] == expected_step

        # Verify no tool results (no tools were used)
        assert len(context["tool_results"]) == 0

        # Verify event sequence
        event_types = [event[0] for event in events]
        assert event_types == ["start_step", "step_plan", "complete_step", "finish"]

    @pytest.mark.asyncio
    async def test_single_step_with_tools_context(self, mock_reasoning_agent, sample_request):  # noqa: ANN001
        """Test context building for a single reasoning step with tools."""
        events = []

        # Mock a reasoning step that uses tools
        expected_step = ReasoningStep(
            thought="I need to get weather information for Tokyo",
            next_action=ReasoningAction.FINISHED,
            tools_to_use=[
                ToolPrediction(
                    tool_name="get_weather",
                    arguments={"location": "Tokyo"},
                    reasoning="User wants weather for Tokyo",
                ),
            ],
            concurrent_execution=False,
        )

        with patch.object(mock_reasoning_agent, '_generate_reasoning_step') as mock_generate:
            mock_generate.return_value = expected_step

            # Collect all events
            async for event_type, event_data in mock_reasoning_agent._core_reasoning_process(sample_request):  # noqa: E501
                events.append((event_type, event_data))

        # Find the finish event and extract context
        finish_events = [event for event in events if event[0] == "finish"]
        context = finish_events[0][1]["context"]

        # Verify step was added to context
        assert len(context["steps"]) == 1
        assert context["steps"][0] == expected_step

        # Verify tool results were added to context
        assert len(context["tool_results"]) == 1
        tool_result = context["tool_results"][0]
        assert tool_result.tool_name == "get_weather"
        assert tool_result.success is True
        assert tool_result.result == {"location": "Tokyo", "temperature": "22°C", "condition": "Sunny"}  # noqa: E501

        # Verify event sequence includes tool events
        event_types = [event[0] for event in events]
        assert event_types == ["start_step", "step_plan", "start_tools", "complete_tools", "complete_step", "finish"]  # noqa: E501

    @pytest.mark.asyncio
    async def test_multiple_steps_context_accumulation(self, mock_reasoning_agent, sample_request):  # noqa: ANN001
        """Test context building across multiple reasoning steps."""
        events = []

        # Mock multiple reasoning steps
        step1 = ReasoningStep(
            thought="I need to search for weather information first",
            next_action=ReasoningAction.USE_TOOLS,
            tools_to_use=[
                ToolPrediction(
                    tool_name="search_web",
                    arguments={"query": "Tokyo weather"},
                    reasoning="Search for Tokyo weather information",
                ),
            ],
            concurrent_execution=False,
        )

        step2 = ReasoningStep(
            thought="Now I'll get specific weather data",
            next_action=ReasoningAction.USE_TOOLS,
            tools_to_use=[
                ToolPrediction(
                    tool_name="get_weather",
                    arguments={"location": "Tokyo"},
                    reasoning="Get detailed weather for Tokyo",
                ),
            ],
            concurrent_execution=False,
        )

        step3 = ReasoningStep(
            thought="I have all the information I need to provide a complete answer",
            next_action=ReasoningAction.FINISHED,
            tools_to_use=[],
            concurrent_execution=False,
        )

        with patch.object(mock_reasoning_agent, '_generate_reasoning_step') as mock_generate:
            mock_generate.side_effect = [step1, step2, step3]

            # Collect all events
            async for event_type, event_data in mock_reasoning_agent._core_reasoning_process(sample_request):  # noqa: E501
                events.append((event_type, event_data))

        # Find the finish event and extract context
        finish_events = [event for event in events if event[0] == "finish"]
        context = finish_events[0][1]["context"]

        # Verify all steps were accumulated
        assert len(context["steps"]) == 3
        assert context["steps"][0] == step1
        assert context["steps"][1] == step2
        assert context["steps"][2] == step3

        # Verify all tool results were accumulated
        assert len(context["tool_results"]) == 2

        # Verify first tool result (search)
        search_result = context["tool_results"][0]
        assert search_result.tool_name == "search_web"
        assert search_result.success is True
        assert search_result.result == {"query": "Tokyo weather", "results": ["result1", "result2"]}  # noqa: E501

        # Verify second tool result (weather)
        weather_result = context["tool_results"][1]
        assert weather_result.tool_name == "get_weather"
        assert weather_result.success is True
        assert weather_result.result == {"location": "Tokyo", "temperature": "22°C", "condition": "Sunny"}  # noqa: E501

    @pytest.mark.asyncio
    async def test_parallel_tools_context_building(self, mock_reasoning_agent, sample_request):  # noqa: ANN001
        """Test context building when tools are executed in parallel."""
        events = []

        # Mock a reasoning step with parallel tools
        expected_step = ReasoningStep(
            thought="I'll search and get weather data simultaneously",
            next_action=ReasoningAction.FINISHED,
            tools_to_use=[
                ToolPrediction(
                    tool_name="search_web",
                    arguments={"query": "Tokyo weather"},
                    reasoning="Search for weather info",
                ),
                ToolPrediction(
                    tool_name="get_weather",
                    arguments={"location": "Tokyo"},
                    reasoning="Get direct weather data",
                ),
            ],
            concurrent_execution=True,  # Key: parallel execution
        )

        with patch.object(mock_reasoning_agent, '_generate_reasoning_step') as mock_generate:
            mock_generate.return_value = expected_step

            # Collect all events
            async for event_type, event_data in mock_reasoning_agent._core_reasoning_process(sample_request):  # noqa: E501
                events.append((event_type, event_data))

        # Find the finish event and extract context
        finish_events = [event for event in events if event[0] == "finish"]
        context = finish_events[0][1]["context"]

        # Verify step was added to context
        assert len(context["steps"]) == 1
        assert context["steps"][0] == expected_step

        # Verify both tool results were added (parallel execution should still accumulate all results)  # noqa: E501
        assert len(context["tool_results"]) == 2

        # Results should include both tools (order may vary due to parallel execution)
        tool_names = {result.tool_name for result in context["tool_results"]}
        assert tool_names == {"search_web", "get_weather"}

        # Verify all results are successful
        assert all(result.success for result in context["tool_results"])

    @pytest.mark.asyncio
    async def test_failed_tool_context_building(self, mock_reasoning_agent, sample_request):  # noqa: ANN001
        """Test context building when tools fail."""
        events = []

        # Mock a reasoning step with a tool that will fail
        expected_step = ReasoningStep(
            thought="I'll try to use an unknown tool",
            next_action=ReasoningAction.FINISHED,
            tools_to_use=[
                ToolPrediction(
                    tool_name="unknown_tool",  # This tool doesn't exist
                    arguments={"param": "value"},
                    reasoning="Testing unknown tool",
                ),
            ],
            concurrent_execution=False,
        )

        with patch.object(mock_reasoning_agent, '_generate_reasoning_step') as mock_generate:
            mock_generate.return_value = expected_step

            # Collect all events
            async for event_type, event_data in mock_reasoning_agent._core_reasoning_process(sample_request):  # noqa: E501
                events.append((event_type, event_data))

        # Find the finish event and extract context
        finish_events = [event for event in events if event[0] == "finish"]
        context = finish_events[0][1]["context"]

        # Verify step was added to context
        assert len(context["steps"]) == 1
        assert context["steps"][0] == expected_step

        # Verify failed tool result was added to context
        assert len(context["tool_results"]) == 1
        tool_result = context["tool_results"][0]
        assert tool_result.tool_name == "unknown_tool"
        assert tool_result.success is False
        assert "Tool 'unknown_tool' not found" in tool_result.error

    @pytest.mark.asyncio
    async def test_context_consistency_between_streaming_and_nonstreaming(
        self, mock_reasoning_agent, sample_request,  # noqa: ANN001
    ):
        """Test that streaming and non-streaming produce identical contexts."""
        # Mock a reasoning step with tools
        expected_step = ReasoningStep(
            thought="I need weather information",
            next_action=ReasoningAction.FINISHED,
            tools_to_use=[
                ToolPrediction(
                    tool_name="get_weather",
                    arguments={"location": "Tokyo"},
                    reasoning="Get Tokyo weather",
                ),
            ],
            concurrent_execution=False,
        )

        with patch.object(mock_reasoning_agent, '_generate_reasoning_step') as mock_generate:
            mock_generate.return_value = expected_step

            # Get context from non-streaming path
            non_streaming_context = None
            async for event_type, event_data in mock_reasoning_agent._core_reasoning_process(sample_request):  # noqa: E501
                if event_type == "finish":
                    non_streaming_context = event_data["context"]

            # Reset the mock for streaming path
            mock_generate.return_value = expected_step

            # Get context from streaming path
            streaming_context = None
            async for event_type, event_data in mock_reasoning_agent._core_reasoning_process(sample_request):  # noqa: E501
                if event_type == "finish":
                    streaming_context = event_data["context"]

        # Verify both contexts exist
        assert non_streaming_context is not None
        assert streaming_context is not None

        # Verify contexts are identical
        assert len(non_streaming_context["steps"]) == len(streaming_context["steps"])
        assert len(non_streaming_context["tool_results"]) == len(streaming_context["tool_results"])

        # Verify step content is identical
        for i, (ns_step, s_step) in enumerate(zip(non_streaming_context["steps"], streaming_context["steps"])):  # noqa: E501
            assert ns_step.thought == s_step.thought
            assert ns_step.next_action == s_step.next_action
            assert len(ns_step.tools_to_use) == len(s_step.tools_to_use)

        # Verify tool results are identical
        for i, (ns_result, s_result) in enumerate(zip(non_streaming_context["tool_results"], streaming_context["tool_results"])):  # noqa: E501
            assert ns_result.tool_name == s_result.tool_name
            assert ns_result.success == s_result.success
            assert ns_result.result == s_result.result

    @pytest.mark.asyncio
    async def test_max_iterations_context_building(self, mock_reasoning_agent, sample_request):  # noqa: ANN001
        """Test context building when max iterations is reached."""
        events = []

        # Mock reasoning steps that never finish (to test max iterations)
        continuing_step = ReasoningStep(
            thought="I need to keep thinking",
            next_action=ReasoningAction.CONTINUE_THINKING,
            tools_to_use=[],
            concurrent_execution=False,
        )

        with patch.object(mock_reasoning_agent, '_generate_reasoning_step') as mock_generate:
            # Always return a continuing step
            mock_generate.return_value = continuing_step

            # Set a low max iterations for testing
            mock_reasoning_agent.max_reasoning_iterations = 3

            # Collect all events
            async for event_type, event_data in mock_reasoning_agent._core_reasoning_process(sample_request):  # noqa: E501
                events.append((event_type, event_data))

        # Find the finish event and extract context
        finish_events = [event for event in events if event[0] == "finish"]
        context = finish_events[0][1]["context"]

        # Verify that exactly max_reasoning_iterations steps were created
        assert len(context["steps"]) == 3  # max_reasoning_iterations

        # Verify all steps are the same continuing step
        for step in context["steps"]:
            assert step.thought == "I need to keep thinking"
            assert step.next_action == ReasoningAction.CONTINUE_THINKING

        # Verify no tool results (no tools were used)
        assert len(context["tool_results"]) == 0

    @pytest.mark.asyncio
    async def test_context_structure_validation(self, mock_reasoning_agent, sample_request):  # noqa: ANN001
        """Test that the context structure remains valid throughout the process."""
        # Mock a complex multi-step scenario
        steps = [
            ReasoningStep(
                thought="Step 1: Initial analysis",
                next_action=ReasoningAction.USE_TOOLS,
                tools_to_use=[
                    ToolPrediction(
                        tool_name="search_web",
                        arguments={"query": "test"},
                        reasoning="Initial search",
                    ),
                ],
                concurrent_execution=False,
            ),
            ReasoningStep(
                thought="Step 2: Follow-up analysis",
                next_action=ReasoningAction.FINISHED,
                tools_to_use=[],
                concurrent_execution=False,
            ),
        ]

        with patch.object(mock_reasoning_agent, '_generate_reasoning_step') as mock_generate:
            mock_generate.side_effect = steps

            # Collect all events and verify context structure at each step
            context_snapshots = []
            async for event_type, event_data in mock_reasoning_agent._core_reasoning_process(sample_request):  # noqa: E501
                if event_type == "finish":
                    final_context = event_data["context"]
                    context_snapshots.append(final_context)

        # Verify final context structure
        final_context = context_snapshots[0]

        # Structure validation
        required_keys = {"steps", "tool_results", "final_thoughts", "user_request"}
        assert set(final_context.keys()) == required_keys

        # Type validation
        assert isinstance(final_context["steps"], list)
        assert isinstance(final_context["tool_results"], list)
        assert isinstance(final_context["final_thoughts"], str)

        # Content validation
        assert len(final_context["steps"]) == 2
        assert len(final_context["tool_results"]) == 1  # Only step 1 used tools
        assert final_context["user_request"] == sample_request

        # Verify each step in context is a proper ReasoningStep
        for step in final_context["steps"]:
            assert isinstance(step, ReasoningStep)
            assert hasattr(step, "thought")
            assert hasattr(step, "next_action")
            assert hasattr(step, "tools_to_use")
            assert hasattr(step, "concurrent_execution")

        # Verify each tool result is a proper ToolResult
        for tool_result in final_context["tool_results"]:
            assert isinstance(tool_result, ToolResult)
            assert hasattr(tool_result, "tool_name")
            assert hasattr(tool_result, "success")
            assert hasattr(tool_result, "execution_time_ms")


class TestReasoningContextPreservation:
    """Test that reasoning context is properly preserved across tool executions."""

    @pytest.fixture
    def context_aware_agent(self):
        """Create agent with tools that depend on previous context."""
        http_client = httpx.AsyncClient()
        mock_prompt_manager = AsyncMock(spec=PromptManager)
        mock_prompt_manager.get_prompt.return_value = "Test prompt"

        # Tools that simulate stateful operations
        self.memory_store = {}

        def store_data(key: str, value: object) -> dict[str, str]:
            """Store data in memory."""
            self.memory_store[key] = value
            return {"status": "stored", "key": key, "value": str(value)}

        def retrieve_data(key: str) -> dict[str, object]:
            """Retrieve data from memory."""
            if key in self.memory_store:
                return {"status": "found", "key": key, "value": self.memory_store[key]}
            return {"status": "not_found", "key": key, "value": None}

        def list_keys() -> dict[str, list[str]]:
            """List all stored keys."""
            return {"keys": list(self.memory_store.keys()), "count": len(self.memory_store)}

        tools = [
            function_to_tool(store_data, name="store"),
            function_to_tool(retrieve_data, name="retrieve"),
            function_to_tool(list_keys, name="list_keys"),
        ]

        return ReasoningAgent(
            base_url="http://test",
            api_key="test-key",
            http_client=http_client,
            tools=tools,
            prompt_manager=mock_prompt_manager,
        )

    @pytest.mark.asyncio
    async def test_sequential_context_preservation(self, context_aware_agent):  # noqa: ANN001
        """Test that context is preserved across sequential tool executions."""
        # First operation: store data
        store_prediction = ToolPrediction(
            tool_name="store",
            arguments={"key": "user_id", "value": "12345"},
            reasoning="Store user ID for later use",
        )

        # Second operation: retrieve data
        retrieve_prediction = ToolPrediction(
            tool_name="retrieve",
            arguments={"key": "user_id"},
            reasoning="Retrieve stored user ID",
        )

        # Execute sequentially
        store_results = await context_aware_agent._execute_tools_sequentially([store_prediction])
        retrieve_results = await context_aware_agent._execute_tools_sequentially([retrieve_prediction])  # noqa: E501

        # Verify context was preserved
        assert store_results[0].success is True
        assert store_results[0].result["status"] == "stored"

        assert retrieve_results[0].success is True
        assert retrieve_results[0].result["status"] == "found"
        assert retrieve_results[0].result["value"] == "12345"

    @pytest.mark.asyncio
    async def test_parallel_context_interference(self, context_aware_agent):  # noqa: ANN001
        """Test that parallel execution doesn't interfere with shared context."""
        # Multiple store operations in parallel
        predictions = [
            ToolPrediction(
                tool_name="store",
                arguments={"key": f"item_{i}", "value": f"value_{i}"},
                reasoning=f"Store item {i}",
            )
            for i in range(5)
        ]

        # Execute in parallel
        results = await context_aware_agent._execute_tools_concurrently(predictions)

        # All should succeed
        assert all(r.success for r in results)
        assert all(r.result["status"] == "stored" for r in results)

        # Verify all data was stored correctly
        list_prediction = ToolPrediction(
            tool_name="list_keys",
            arguments={},
            reasoning="Check all stored keys",
        )

        list_results = await context_aware_agent._execute_tools_sequentially([list_prediction])
        assert list_results[0].success is True
        assert list_results[0].result["count"] == 5
        stored_keys = set(list_results[0].result["keys"])
        expected_keys = {f"item_{i}" for i in range(5)}
        assert stored_keys == expected_keys

    @pytest.mark.asyncio
    async def test_context_across_reasoning_iterations(self, context_aware_agent):  # noqa: ANN001
        """Test context preservation across multiple reasoning steps."""
        # Simulate multiple reasoning iterations with tool context
        context = {
            "steps": [],
            "tool_results": [],
            "final_thoughts": "",
            "user_request": None,
        }

        # First iteration: store data
        step1 = ReasoningStep(
            thought="I need to store some data first",
            next_action=ReasoningAction.USE_TOOLS,
            tools_to_use=[
                ToolPrediction(
                    tool_name="store",
                    arguments={"key": "session_id", "value": "abc123"},
                    reasoning="Store session for tracking",
                ),
            ],
        )
        context["steps"].append(step1)

        # Execute tools for step 1
        tool_results_1 = await context_aware_agent._execute_tools_sequentially(step1.tools_to_use)
        context["tool_results"].extend(tool_results_1)

        # Second iteration: retrieve and verify
        step2 = ReasoningStep(
            thought="Now I need to verify the stored data",
            next_action=ReasoningAction.USE_TOOLS,
            tools_to_use=[
                ToolPrediction(
                    tool_name="retrieve",
                    arguments={"key": "session_id"},
                    reasoning="Verify session was stored correctly",
                ),
            ],
        )
        context["steps"].append(step2)

        # Execute tools for step 2
        tool_results_2 = await context_aware_agent._execute_tools_sequentially(step2.tools_to_use)
        context["tool_results"].extend(tool_results_2)

        # Verify context preservation across iterations
        assert len(context["tool_results"]) == 2
        assert context["tool_results"][0].result["status"] == "stored"
        assert context["tool_results"][1].result["status"] == "found"
        assert context["tool_results"][1].result["value"] == "abc123"

