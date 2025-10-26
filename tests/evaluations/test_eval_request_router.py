
"""
Evaluations for  using flex-evals framework.

These evaluations test the real reasoning agent with actual OpenAI API calls
to ensure it works correctly most of the time despite LLM variability.
"""
import pytest
from dotenv import load_dotenv
from flex_evals import IsEmptyCheck, TestCase, EqualsCheck
from flex_evals.pytest_decorator import evaluate

from api.openai_protocol import OpenAIChatRequest
from api.request_router import RoutingMode, determine_routing

load_dotenv()


async def _generate_routing_response(prompt: str) -> dict:
    request = OpenAIChatRequest(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    headers = {"x-routing-mode": "auto"}
    result = await determine_routing(
        request,
        headers=headers,
    )
    return result.model_dump()


@pytest.mark.evaluation
@evaluate(
    test_cases=[
        TestCase(input="What is 2 + 2?"),
        TestCase(input="Tell me a short joke about computers."),
        TestCase(input="What is the definition of `photosynthesis`?"),
    ],
    checks=[
        EqualsCheck(actual="$.output.value.routing_mode", expected=RoutingMode.PASSTHROUGH),
        EqualsCheck(actual="$.output.value.decision_source", expected="llm_classifier"),
        IsEmptyCheck(value="$.output.value.reason", negate=True),
    ],
    samples=5,  # Run each test case 5 times
    success_threshold=0.6,  # Expect 60% success rate
)
async def test_router__expects__passthrough(test_case: TestCase) -> str:
    """
    When using `auto` routing mode with simple prompts, expects the llm classifer to choose
    `passthrough`.
    """
    return await _generate_routing_response(test_case.input)


@pytest.mark.evaluation
@evaluate(
    test_cases=[
        TestCase(input="Plan a trip to Paris including flights, hotel, and sightseeing."),
        TestCase(input="Review my resume and suggest improvements for a software engineering job."),  # noqa: E501
        TestCase(input="Review the code base and suggest optimizations for better performance."),
    ],
    checks=[
        EqualsCheck(actual="$.output.value.routing_mode", expected=RoutingMode.ORCHESTRATION),
        EqualsCheck(actual="$.output.value.decision_source", expected="llm_classifier"),
        IsEmptyCheck(value="$.output.value.reason", negate=True),
    ],
    samples=5,  # Run each test case 5 times
    success_threshold=0.6,  # Expect 60% success rate
)

async def test_router__expects__orchestration(test_case: TestCase) -> str:
    """
    When using `auto` routing mode with more complex prompts that expected multiple steps,
    expects the llm classifer to choose `orchestration`.
    """
    return await _generate_routing_response(test_case.input)
