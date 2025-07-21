"""
Evaluations for ReasoningAgent using flex-evals framework.

These evaluations test the real reasoning agent with actual OpenAI API calls
to ensure it works correctly most of the time despite LLM variability.
"""

import json
import os

import pytest
from dotenv import load_dotenv
from flex_evals import ContainsCheck, TestCase
from flex_evals.pytest_decorator import evaluate

from api.openai_protocol import OpenAIChatRequest
from api.prompt_manager import PromptManager
from api.reasoning_agent import ReasoningAgent
from api.tools import function_to_tool

load_dotenv()


# Mark all tests as evaluations and skip if no OpenAI API key
pytestmark = [
    pytest.mark.evaluation,  # Mark as evaluation for opt-in running
    pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY required for evaluations",
    ),
]


@pytest.fixture
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
        """Perform basic calculations."""
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

    def search_knowledge(query: str) -> dict:
        """Search knowledge base."""
        # Simulate search results based on query
        results = {
            "python": ["Python is a high-level programming language", "Known for readability"],
            "weather": ["Weather is the state of the atmosphere", "Includes temperature, humidity"],  # noqa: E501
            "math": ["Mathematics is the study of numbers", "Includes arithmetic, algebra"],
        }

        matching_results = []
        for key, values in results.items():
            if key.lower() in query.lower():
                matching_results.extend(values)

        return {
            "query": query,
            "results": matching_results or ["No specific results found"],
            "count": len(matching_results),
        }

    tools = [
        function_to_tool(get_weather, description="Get weather for any location"),
        function_to_tool(calculate, description="Perform mathematical calculations"),
        function_to_tool(search_knowledge, description="Search the knowledge base"),
    ]

    # Create and initialize prompt manager
    prompt_manager = PromptManager()
    await prompt_manager.initialize()

    return ReasoningAgent(
        base_url="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
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
            phrases=["weather", "temperature", "22"],  # Match "22°C" or "22"
            case_sensitive=False,
        ),
    ],
    samples=5,  # Run each test case 5 times
    success_threshold=0.8,  # Expect 80% success rate
)
async def test_weather_tool_usage(
        test_case: TestCase,
        reasoning_agent_with_tools: ReasoningAgent,
    ) -> str:
    """Test that the agent correctly uses the weather tool when asked."""
    agent = reasoning_agent_with_tools

    request = OpenAIChatRequest(
        model="gpt-4o-mini",  # Use cheaper model for evaluations
        messages=[{"role": "user", "content": test_case.input}],
        max_tokens=300,
        temperature=0.3,  # Lower temperature for more consistent results
    )

    response = await agent.execute(request)
    return response.choices[0].message.content


@evaluate(
    test_cases=[
        TestCase(id="calc_add", input="Calculate 15 + 27 using the calculate tool."),
        TestCase(id="calc_divide", input="What is 100 divided by 4? Use calculate tool."),
        TestCase(id="calc_multiply", input="Multiply 12 by 8 using the calculate function."),
    ],
    checks=[
        ContainsCheck(
            text="$.output.value",
            phrases=["result", "calculation", "answer"],
            case_sensitive=False,
        ),
    ],
    samples=5,
    success_threshold=0.8,
)
async def test_calculator_tool_usage(
        test_case: TestCase,
        reasoning_agent_with_tools: ReasoningAgent,
    ) -> str:
    """Test that the agent correctly uses the calculator tool."""
    agent = reasoning_agent_with_tools

    request = OpenAIChatRequest(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": test_case.input}],
        max_tokens=300,
        temperature=0.3,
    )

    response = await agent.execute(request)
    return response.choices[0].message.content


@evaluate(
    test_cases=[
        TestCase(
            id="search_python",
            input="Search for information about Python programming using search_knowledge.",
        ),
        TestCase(
            id="search_weather",
            input="Use search_knowledge to find information about weather.",
        ),
        TestCase(
            id="search_math",
            input="Look up math topics with the search_knowledge tool.",
        ),
    ],
    checks=[
        ContainsCheck(
            text="$.output.value",
            phrases=["search", "results", "found"],
            case_sensitive=False,
        ),
    ],
    samples=5,
    success_threshold=0.8,
)
async def test_search_tool_usage(
        test_case: TestCase,
        reasoning_agent_with_tools: ReasoningAgent,
    ) -> str:
    """Test that the agent correctly uses the search tool."""
    agent = reasoning_agent_with_tools

    request = OpenAIChatRequest(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": test_case.input}],
        max_tokens=300,
        temperature=0.3,
    )

    response = await agent.execute(request)
    return response.choices[0].message.content


@evaluate(
    test_cases=[
        TestCase(
            id="multi_weather_calc",
            input="What's the weather in Tokyo and calculate 25 + 15? Use the appropriate tools.",
        ),
        TestCase(
            id="multi_search_weather",
            input="Search for Python info and get weather for Paris. Use both tools.",
        ),
        TestCase(
            id="multi_calc_search",
            input="Calculate 10 * 5 and then search for math information. Use the tools.",
        ),
    ],
    checks=[
        ContainsCheck(
            text="$.output.value",
            phrases=["weather", "calculate", "search", "result"],
            case_sensitive=False,
        ),
    ],
    samples=5,
    success_threshold=0.7,  # Lower threshold for multi-tool usage
)
async def test_multiple_tool_usage(
        test_case: TestCase,
        reasoning_agent_with_tools: ReasoningAgent,
    ) -> str:
    """Test that the agent can use multiple tools in one request."""
    agent = reasoning_agent_with_tools

    request = OpenAIChatRequest(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": test_case.input}],
        max_tokens=500,  # More tokens for complex responses
        temperature=0.3,
    )

    response = await agent.execute(request)
    return response.choices[0].message.content


@evaluate(
    test_cases=[
        TestCase(id="factual_france", input="What is the capital of France?"),
        TestCase(id="factual_photosynthesis", input="Explain photosynthesis briefly."),
        TestCase(id="factual_python_year", input="What year was Python created?"),
    ],
    checks=[
        ContainsCheck(
            text="$.output.value",
            phrases=["Paris", "photosynthesis", "1991"],  # Expected answers
            case_sensitive=False,
            require_any=True,  # Only need one phrase to match
        ),
    ],
    samples=5,
    success_threshold=0.9,  # Higher threshold for simple factual questions
)
async def test_no_tool_needed(
        test_case: TestCase,
        reasoning_agent_with_tools: ReasoningAgent,
    ) -> str:
    """Test that the agent can answer questions without using tools when appropriate."""
    agent = reasoning_agent_with_tools

    request = OpenAIChatRequest(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": test_case.input}],
        max_tokens=200,
        temperature=0.3,
    )

    response = await agent.execute(request)
    return response.choices[0].message.content


# Additional evaluation for streaming responses
@evaluate(
    test_cases=[
        TestCase(
            id="stream_weather",
            input="Get weather for Berlin using get_weather tool and explain it.",
        ),
        TestCase(
            id="stream_calc",
            input="Calculate 50 divided by 5 with the calculate tool.",
        ),
    ],
    checks=[
        ContainsCheck(
            text="$.output.value",
            phrases=["weather", "Berlin", "22°C", "calculate", "result", "10"],
            case_sensitive=False,
            require_any=True,
        ),
    ],
    samples=3,  # Fewer samples for streaming test
    success_threshold=0.8,
)
async def test_streaming_with_tools(
        test_case: TestCase,
        reasoning_agent_with_tools: ReasoningAgent,
    ) -> str:
    """Test that streaming responses work correctly with tool usage."""
    agent = reasoning_agent_with_tools

    request = OpenAIChatRequest(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": test_case.input}],
        max_tokens=300,
        temperature=0.3,
        stream=True,
    )

    # Collect streaming response
    content_parts = []
    async for chunk in agent.execute_stream(request):
        if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
            try:
                chunk_data = json.loads(chunk[6:])
                if chunk_data.get("choices") and chunk_data["choices"][0].get("delta", {}).get("content"):  # noqa: E501
                    content_parts.append(chunk_data["choices"][0]["delta"]["content"])
            except json.JSONDecodeError:
                continue

    return "".join(content_parts)


# Custom evaluation with specific result checking
class WeatherResultCheck:
    """Custom check for weather tool results."""

    def __init__(self, expected_location: str):
        self.expected_location = expected_location

    def check(self, output: object) -> bool:
        """Check if the output contains the expected weather information."""
        content = output.value.lower() if hasattr(output, 'value') else str(output).lower()

        # Check for location and temperature
        has_location = self.expected_location.lower() in content
        has_temperature = "22°c" in content or "22 degrees" in content
        has_weather_terms = any(term in content for term in ["sunny", "weather", "temperature"])

        return has_location and has_temperature and has_weather_terms


@evaluate(
    test_cases=[
        TestCase(
            id="custom_tokyo",
            input="What's the weather in Tokyo?",
            metadata={"location": "Tokyo"},
        ),
        TestCase(
            id="custom_nyc",
            input="Get weather for New York.",
            metadata={"location": "New York"},
        ),
    ],
    checks=[
        lambda tc, output: WeatherResultCheck(tc.metadata["location"]).check(output),
    ],
    samples=5,
    success_threshold=0.8,
)
async def test_weather_with_custom_check(
        test_case: TestCase,
        reasoning_agent_with_tools: ReasoningAgent,
    ) -> str:
    """Test weather tool with custom result validation."""
    agent = reasoning_agent_with_tools

    request = OpenAIChatRequest(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": test_case.input}],
        max_tokens=300,
        temperature=0.3,
    )

    response = await agent.execute(request)
    return response.choices[0].message.content
