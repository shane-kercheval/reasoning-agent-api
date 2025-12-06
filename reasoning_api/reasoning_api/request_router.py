"""
Request routing logic for three-path execution strategy.

This module implements routing to support three distinct execution paths:
- Passthrough: Direct OpenAI API call (default, fast path)
- Reasoning: Single-loop reasoning agent (manual, baseline)

Routing Strategy:
1. Explicit passthrough rules (response_format, tools) → force passthrough
2. Header-based routing (X-Routing-Mode: passthrough|reasoning|auto)
3. Default behavior: passthrough if no header provided
4. Auto-routing: LLM classifier chooses between passthrough or reasoning only

Distributed Tracing:
The auto-routing classifier propagates OpenTelemetry trace context to LiteLLM to ensure
the classifier's LLM call appears as a child span of the parent request. The pattern is:
  carrier: dict[str, str] = {}
  propagate.inject(carrier)  # Injects traceparent headers into carrier dict
  litellm.acompletion(..., extra_headers=carrier)
The carrier dict is passed to extra_headers so LiteLLM receives the trace context and
can correlate this LLM call with the parent request in observability tools (Phoenix, Jaeger).
"""

import logging
from enum import Enum

import litellm
from opentelemetry import propagate
from pydantic import BaseModel, Field

from .openai_protocol import OpenAIChatRequest
from .config import settings

logger = logging.getLogger(__name__)


class RoutingMode(str, Enum):
    """
    Execution path for chat completion request.

    Values:
        passthrough: Direct OpenAI API call (default)
        reasoning: Single-loop reasoning agent (manual, baseline)
        auto: Use LLM classifier to choose between passthrough or reasoning
    """

    PASSTHROUGH = "passthrough"
    REASONING = "reasoning"
    AUTO = "auto"


class RoutingDecision(BaseModel):
    """
    Routing decision for a chat completion request.

    Attributes:
        routing_mode: Execution path (passthrough/reasoning)
            - passthrough: Direct OpenAI API call
            - reasoning: Sequential-loop reasoning agent (manual, baseline)
        reasoning: Human-readable explanation of the routing decision
        decision_source: Source of the decision
            (passthrough_rule, header_override, default, llm_classifier)
    """

    routing_mode: RoutingMode = Field(
        description=("Execution path: passthrough (direct OpenAI call) or reasoning (sequential-loop reasoning agent"),  # noqa: E501
    )
    reason: str = Field(
        description="Concise reason why this routing decision was made",
    )
    decision_source: str = Field(
        description=(
            "Source of the decision: passthrough_rule, header_override, "
            "default, or llm_classifier"
        ),
    )


# Separate model for LLM classifier (only passthrough or reasoning)
class ClassifierRoutingDecision(BaseModel):
    """
    LLM classifier routing decision (only passthrough or reasoning).

    - passthrough: Direct OpenAI API call
    - reasoning: Sequential-loop reasoning agent
    """

    routing_mode: RoutingMode = Field(description=("Execution path: passthrough or reasoning."))


async def determine_routing(
    request: OpenAIChatRequest,
    headers: dict[str, str] | None = None,
) -> RoutingDecision:
    """
    Determine routing path for a chat completion request.

    Three-tier routing strategy:
    1. Explicit passthrough rules (response_format, tools) → force passthrough
    2. Header-based routing (X-Routing-Mode) → explicit path selection
    3. Default behavior → passthrough if no header provided

    Args:
        request: The OpenAI chat completion request
        headers: HTTP headers from the request (optional)

    Returns:
        RoutingDecision with routing mode, reason, and source

    Raises:
        ValueError: If X-Routing-Mode header has invalid value
        Exception: If LLM classifier fails (when using auto mode)

    Examples:
        >>> request = OpenAIChatRequest(model="gpt-4o", messages=[...], tools=[...])
        >>> decision = await determine_routing(request)
        >>> decision.routing_mode
        RoutingMode.PASSTHROUGH
        >>> decision.reason
        'Request has tools parameter - user wants direct LLM control'
    """
    headers = headers or {}

    # Tier 1: Explicit passthrough rules (highest priority)
    # If request has response_format or tools, force passthrough
    if request.response_format is not None:
        return RoutingDecision(
            routing_mode=RoutingMode.PASSTHROUGH,
            reason=(
                "Request has response_format parameter - "
                "forcing passthrough for direct LLM control"
            ),
            decision_source="passthrough_rule",
        )

    if request.tools is not None:
        return RoutingDecision(
            routing_mode=RoutingMode.PASSTHROUGH,
            reason=(
                "Request has tools parameter - "
                "forcing passthrough for direct function calling"
            ),
            decision_source="passthrough_rule",
        )

    # Tier 2: Header-based routing (X-Routing-Mode)
    routing_mode_header = headers.get("x-routing-mode", "").lower()

    if routing_mode_header:
        # Validate and parse header value
        if routing_mode_header == "passthrough":
            return RoutingDecision(
                routing_mode=RoutingMode.PASSTHROUGH,
                reason=(
                    "X-Routing-Mode header set to `passthrough` - "
                    "user explicitly requested direct OpenAI"
                ),
                decision_source="header_override",
            )
        if routing_mode_header == "reasoning":
            return RoutingDecision(
                routing_mode=RoutingMode.REASONING,
                reason=(
                    "X-Routing-Mode header set to `reasoning` - "
                    "user explicitly requested reasoning agent"
                ),
                decision_source="header_override",
            )
        if routing_mode_header == "auto":
            # Use LLM classifier to choose between passthrough and reasoning
            return await _classify_with_llm(request)

        # Invalid header value - fail fast
        raise ValueError(
            f"Invalid X-Routing-Mode header value: '{routing_mode_header}'. "
            f"Valid values: passthrough, reasoning, auto",
        )

    # Tier 3: Default behavior (no header provided)
    # Default to passthrough for OpenAI-like experience
    return RoutingDecision(
        routing_mode=RoutingMode.PASSTHROUGH,
        reason=(
            "No routing header provided - "
            "defaulting to passthrough (matches OpenAI experience)"
        ),
        decision_source="default",
    )


async def _classify_with_llm(
        request: OpenAIChatRequest,
    ) -> RoutingDecision:
    """
    Use LLM with structured outputs to classify routing path.

    This function analyzes the user's query to determine if it requires:
    - Passthrough: Direct answer, simple query
    - Reasoning: Multi-step reasoning, research synthesis

    Args:
        request: The OpenAI chat completion request

    Returns:
        Dictionary with 'routing_mode' (RoutingMode) and 'reason' (str)

    Raises:
        Exception: If LLM API call fails or response parsing fails
    """
    # Extract the last user message for classification
    user_messages = [
        msg for msg in request.messages
        if msg.get("role") == "user"
    ]

    if not user_messages:
        return RoutingDecision(
            routing_mode=RoutingMode.PASSTHROUGH,
            reason="No user message found - defaulting to passthrough",
            decision_source="llm_classifier",
        )

    last_user_message = user_messages[-1].get("content", "")

    # Classification prompt for structured output
    # Long lines in multi-line string are acceptable for readability
    classification_prompt = """
You are a routing classifier that determines if a user query requires a simple and direct LLM
API call (passthrough) or a more complex reasoning process (reasoning).

Analyze the user's query and classify it as either:
- PASSTHROUGH: Direct answer, widely known knowledge likely known to LLMs, simple calculation, summarization, translation, creative writing
- REASONING: Multi-step reasoning, research synthesis, complex analysis, or tasks requiring multiple tools or data sources

Examples of PASSTHROUGH queries:
- "What's the capital of France?"
- "Translate 'hello' to Spanish"
- "What is 2+2?"
- "Summarize this text: [text]"
- "Write a poem about the ocean"
- "Explain quantum entanglement"

Examples of REASONING queries:
- "What is the current stock price of Company X?"
- "Research renewable energy trends and create an investment strategy"
- "Analyze weather patterns across multiple cities and recommend travel destinations"
- "Compare pricing for flights, hotels, and activities for a week-long trip"
- "Find relevant papers on quantum computing and synthesize key findings"

Respond with your classification and reasoning.
"""  # noqa: E501

    # Inject trace context into headers for LiteLLM propagation
    carrier: dict[str, str] = {}
    propagate.inject(carrier)

    # Use structured outputs with Pydantic schema via litellm
    response = await litellm.acompletion(
        model=settings.routing_classifier_model,
        messages=[
            {"role": "system", "content": classification_prompt},
            {"role": "user", "content": f"Query to classify: {last_user_message}"},
        ],
        temperature=settings.routing_classifier_temperature,
        response_format=ClassifierRoutingDecision,
        extra_headers=carrier,  # Propagate trace context to LiteLLM
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )

    # Parse structured response (litellm returns JSON string in content)
    if response.choices and response.choices[0].message.content:
        try:
            content = response.choices[0].message.content
            decision = ClassifierRoutingDecision.model_validate_json(content)
            return RoutingDecision(
                routing_mode=decision.routing_mode,
                reason="LLM classifier decision",
                decision_source="llm_classifier",
            )
        except Exception as e:
            logger.warning(f"Failed to parse classifier response: {e}")
            # Fallback if parsing fails
            return RoutingDecision(
                routing_mode=RoutingMode.PASSTHROUGH,
                reason="Failed to parse LLM classifier response - defaulting to passthrough",
                decision_source="llm_classifier",
            )

    # Fallback if no content
    logger.warning("LLM classifier returned no content, defaulting to passthrough")
    return RoutingDecision(
        routing_mode=RoutingMode.PASSTHROUGH,
        reason="LLM classifier returned no content - defaulting to passthrough",
        decision_source="llm_classifier",
    )
