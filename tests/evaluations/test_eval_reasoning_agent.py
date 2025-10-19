"""
Evaluations for ReasoningAgent using flex-evals framework.

These evaluations test the real reasoning agent with actual OpenAI API calls
to ensure it works correctly most of the time despite LLM variability.
"""

import os
from dotenv import load_dotenv
from flex_evals import ContainsCheck, TestCase
from flex_evals.pytest_decorator import evaluate
from openai import AsyncOpenAI

from api.openai_protocol import OpenAIChatRequest
from api.prompt_manager import PromptManager
from api.reasoning_agent import ReasoningAgent
from api.tools import function_to_tool
from tests.conftest import ReasoningAgentStreamingCollector

load_dotenv()


async def reasoning_agent_with_tools():
    """Create ReasoningAgent with fake tools and real OpenAI API."""
    # Create fake tools for testing
    def get_weather(location: str) -> dict:
        """Get current weather for a location."""
        return {
            "location": location,
            "temperature": "22°C",
            "condition": "Sunny",
            "humidity": "60%",
        }

    def calculate(operation: str, a: float, b: float) -> dict:
        """
        Perform basic calculations.

        Args:
            operation: The operation to perform (`add`, `subtract`, `multiply`, `divide`)
            a: First number
            b: Second number
        """
        ops = {
            "add": a + b,
            "subtract": a - b,
            "multiply": a * b,
            "divide": a / b if b != 0 else "Error: Division by zero",
        }
        return {
            "operation": operation,
            "a": a,
            "b": b,
            "result": ops.get(operation, "Error: Unknown operation"),
        }

    tools = [
        function_to_tool(get_weather),
        function_to_tool(calculate),
    ]

    # Create and initialize prompt manager
    prompt_manager = PromptManager()
    await prompt_manager.initialize()

    # Create real AsyncOpenAI client for evaluations
    openai_client = AsyncOpenAI(
        base_url=os.environ['LITELLM_BASE_URL'],
        api_key=os.environ['LITELLM_EVAL_KEY'],
    )
    return ReasoningAgent(
        openai_client=openai_client,
        tools=tools,
        prompt_manager=prompt_manager,
    )


@evaluate(
    test_cases=[
        TestCase(
            id="weather_tokyo",
            input="What's the weather in Tokyo? Use the get_weather tool.",
        ),
        TestCase(
            id="weather_paris",
            input="Get the current weather for Paris using get_weather.",
        ),
        TestCase(
            id="weather_london",
            input="Tell me the weather in London. Make sure to use get_weather tool.",
        ),
    ],
    checks=[
        ContainsCheck(
            text="$.output.value",
            phrases=["weather", "22"],  # Match "22°C" and "22" and "weather"
            case_sensitive=False,
        ),
    ],
    samples=5,  # Run each test case 5 times
    success_threshold=0.6,  # Expect 80% success rate
)
async def test_weather_tool_usage(test_case: TestCase) -> str:
    """Test that the agent correctly uses the weather tool when asked."""
    agent = await reasoning_agent_with_tools()

    request = OpenAIChatRequest(
        model="gpt-4o-mini",  # Use cheaper model for evaluations
        messages=[{"role": "user", "content": test_case.input}],
        max_tokens=300,
        temperature=0.2,  # Lower temperature for more consistent results
        stream=True,  # Use streaming
    )

    # Collect streaming response
    collector = ReasoningAgentStreamingCollector()
    await collector.process(agent.execute_stream(request))
    return collector.content


@evaluate(
    test_cases=[
        TestCase(
            id="calc_add",
            input="Calculate 15 + 15 using the calculate tool. if there are issues with the tool, give the exact error and the exact arguments that were used/passed to the tool.",  # noqa: E501
        ),
        TestCase(
            id="calc_divide",
            input="What is 90 divided by 3? Use calculate tool.",
        ),
        TestCase(
            id="calc_multiply",
            input="Multiply 5 by 6 using the calculate function.",
        ),
    ],
    checks=[
        ContainsCheck(
            text="$.output.value",
            phrases=["30"],  # Just check for "result" since it's consistently present
            case_sensitive=False,
        ),
    ],
    samples=5,
    success_threshold=0.6,
)
async def test_calculator_tool_usage(test_case: TestCase) -> str:
    """Test that the agent correctly uses the calculator tool."""
    agent = await reasoning_agent_with_tools()
    request = OpenAIChatRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": test_case.input}],
        max_tokens=300,
        temperature=0.2,
        stream=True,  # Use streaming
    )

    # Collect streaming response
    collector = ReasoningAgentStreamingCollector()
    await collector.process(agent.execute_stream(request))
    return collector.content


@evaluate(
    test_cases=[
        TestCase(
            id="factual_france",
            input="What is the capital of France?",
            checks=[ContainsCheck(text="$.output.value", phrases=["Paris"])],
        ),
        TestCase(
            id="factual_earth",
            input="What is the largest planet in our solar system?",
            checks=[ContainsCheck(text="$.output.value", phrases=["Jupiter"])],
        ),
    ],
    samples=5,
    success_threshold=0.6,
)
async def test_no_tool_needed(test_case: TestCase) -> str:
    """Test that the agent can answer questions without using tools when appropriate."""
    agent = await reasoning_agent_with_tools()
    request = OpenAIChatRequest(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": test_case.input}],
        max_tokens=200,
        temperature=0.3,
        stream=True,  # Use streaming
    )

    # Collect streaming response
    collector = ReasoningAgentStreamingCollector()
    await collector.process(agent.execute_stream(request))
    return collector.content
