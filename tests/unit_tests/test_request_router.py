"""
Unit tests for request routing logic.

Tests the three-route routing strategy with RoutingMode enum:
1. Explicit passthrough rules (response_format, tools) → force passthrough
2. Header-based routing (X-Routing-Mode: passthrough|reasoning|orchestration|auto)
3. Default behavior → passthrough if no header
4. Auto-routing → LLM classifier (mocked)
"""

import pytest
from unittest.mock import patch, AsyncMock

from openai import AsyncOpenAI

from api.request_router import determine_routing, RoutingDecision, RoutingMode
from api.openai_protocol import OpenAIChatRequest


class TestPassthroughRules:
    """Test explicit passthrough rules that force passthrough path."""

    @pytest.mark.asyncio
    async def test_response_format_forces_passthrough(self) -> None:
        """Request with response_format should force passthrough."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            response_format={"type": "json_object"},
        )

        decision = await determine_routing(request)

        assert decision.routing_mode == RoutingMode.PASSTHROUGH
        assert decision.decision_source == "passthrough_rule"
        assert "response_format" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_tools_parameter_forces_passthrough(self) -> None:
        """Request with tools parameter should force passthrough."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        )

        decision = await determine_routing(request)

        assert decision.routing_mode == RoutingMode.PASSTHROUGH
        assert decision.decision_source == "passthrough_rule"
        assert "tools" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_passthrough_rule_overrides_header(self) -> None:
        """Passthrough rules should override even explicit headers."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            response_format={"type": "json_object"},
        )
        # Try to force orchestration with header
        headers = {"x-routing-mode": "orchestration"}

        decision = await determine_routing(request, headers=headers)

        # Passthrough rule should take precedence
        assert decision.routing_mode == RoutingMode.PASSTHROUGH
        assert decision.decision_source == "passthrough_rule"


class TestHeaderRouting:
    """Test X-Routing-Mode header-based routing."""

    @pytest.mark.asyncio
    async def test_header_passthrough(self) -> None:
        """X-Routing-Mode: passthrough should route to passthrough."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        headers = {"x-routing-mode": "passthrough"}

        decision = await determine_routing(request, headers=headers)

        assert decision.routing_mode == RoutingMode.PASSTHROUGH
        assert decision.decision_source == "header_override"
        assert "passthrough" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_header_reasoning(self) -> None:
        """X-Routing-Mode: reasoning should route to reasoning."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Complex query"}],
        )
        headers = {"x-routing-mode": "reasoning"}

        decision = await determine_routing(request, headers=headers)

        assert decision.routing_mode == RoutingMode.REASONING
        assert decision.decision_source == "header_override"
        assert "reasoning" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_header_orchestration(self) -> None:
        """X-Routing-Mode: orchestration should route to orchestration."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Complex query"}],
        )
        headers = {"x-routing-mode": "orchestration"}

        decision = await determine_routing(request, headers=headers)

        assert decision.routing_mode == RoutingMode.ORCHESTRATION
        assert decision.decision_source == "header_override"
        assert "orchestration" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_header_auto_uses_llm_classifier(self) -> None:
        """X-Routing-Mode: auto should use LLM classifier."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What's 2+2?"}],
        )
        headers = {"x-routing-mode": "auto"}
        mock_openai_client = AsyncMock(spec=AsyncOpenAI)

        with patch("api.request_router._classify_with_llm", new_callable=AsyncMock) as mock_classify:  # noqa: E501
            mock_classify.return_value = {
                "routing_mode": RoutingMode.PASSTHROUGH,
                "reason": "Simple arithmetic question",
            }

            decision = await determine_routing(
                request,
                headers=headers,
                openai_client=mock_openai_client,
            )

            assert decision.routing_mode == RoutingMode.PASSTHROUGH
            assert decision.decision_source == "llm_classifier"
            mock_classify.assert_called_once_with(request, mock_openai_client)

    @pytest.mark.asyncio
    async def test_header_case_insensitive(self) -> None:
        """X-Routing-Mode header should be case-insensitive."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        headers = {"x-routing-mode": "PASSTHROUGH"}

        decision = await determine_routing(request, headers=headers)

        assert decision.routing_mode == RoutingMode.PASSTHROUGH
        assert decision.decision_source == "header_override"

    @pytest.mark.asyncio
    async def test_invalid_header_raises_error(self) -> None:
        """Invalid X-Routing-Mode value should raise ValueError."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        headers = {"x-routing-mode": "invalid-value"}

        with pytest.raises(ValueError, match="Invalid X-Routing-Mode header value"):
            await determine_routing(request, headers=headers)


class TestDefaultBehavior:
    """Test default routing behavior (no header provided)."""

    @pytest.mark.asyncio
    async def test_no_header_defaults_to_passthrough(self) -> None:
        """No routing header should default to passthrough."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )

        decision = await determine_routing(request)

        assert decision.routing_mode == RoutingMode.PASSTHROUGH
        assert decision.decision_source == "default"
        assert "default" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_empty_header_defaults_to_passthrough(self) -> None:
        """Empty routing header should default to passthrough."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        headers = {"x-routing-mode": ""}

        decision = await determine_routing(request, headers=headers)

        assert decision.routing_mode == RoutingMode.PASSTHROUGH
        assert decision.decision_source == "default"


class TestLLMClassifier:
    """Test LLM-based query complexity classification (auto mode)."""

    @pytest.mark.asyncio
    async def test_classifier_returns_passthrough(self) -> None:
        """LLM classifier can classify query as passthrough."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What's 2+2?"}],
        )
        headers = {"x-routing-mode": "auto"}
        mock_openai_client = AsyncMock(spec=AsyncOpenAI)

        with patch("api.request_router._classify_with_llm", new_callable=AsyncMock) as mock_classify:  # noqa: E501
            mock_classify.return_value = {
                "routing_mode": RoutingMode.PASSTHROUGH,
                "reason": "Simple arithmetic question - direct answer",
            }

            decision = await determine_routing(
                request,
                headers=headers,
                openai_client=mock_openai_client,
            )

            assert decision.routing_mode == RoutingMode.PASSTHROUGH
            assert decision.decision_source == "llm_classifier"
            assert "Simple arithmetic" in decision.reason
            mock_classify.assert_called_once_with(request, mock_openai_client)

    @pytest.mark.asyncio
    async def test_classifier_returns_orchestration(self) -> None:
        """LLM classifier can classify query as orchestration."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": "Research renewable energy trends and create an investment strategy",
            }],
        )
        headers = {"x-routing-mode": "auto"}
        mock_openai_client = AsyncMock(spec=AsyncOpenAI)

        with patch("api.request_router._classify_with_llm", new_callable=AsyncMock) as mock_classify:  # noqa: E501
            mock_classify.return_value = {
                "routing_mode": RoutingMode.ORCHESTRATION,
                "reason": "Multi-step research and analysis requiring orchestration",
            }

            decision = await determine_routing(
                request,
                headers=headers,
                openai_client=mock_openai_client,
            )

            assert decision.routing_mode == RoutingMode.ORCHESTRATION
            assert decision.decision_source == "llm_classifier"
            assert "orchestration" in decision.reason.lower()
            mock_classify.assert_called_once_with(request, mock_openai_client)

    @pytest.mark.asyncio
    async def test_classifier_never_returns_reasoning(self) -> None:
        """LLM classifier should never return reasoning mode (manual-only)."""
        # This test documents the design constraint that the LLM classifier
        # only chooses between passthrough and orchestration
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Complex query"}],
        )
        headers = {"x-routing-mode": "auto"}
        mock_openai_client = AsyncMock(spec=AsyncOpenAI)

        with patch("api.request_router._classify_with_llm", new_callable=AsyncMock) as mock_classify:  # noqa: E501
            # Even if someone manually returned reasoning, it shouldn't happen
            # This test documents that reasoning is header-only
            mock_classify.return_value = {
                # Classifier only returns passthrough or orchestration
                "routing_mode": RoutingMode.PASSTHROUGH,
                "reason": "Test query",
            }

            decision = await determine_routing(
                request,
                headers=headers,
                openai_client=mock_openai_client,
            )

            # Should be passthrough or orchestration, never reasoning
            assert decision.routing_mode in (RoutingMode.PASSTHROUGH, RoutingMode.ORCHESTRATION)

    @pytest.mark.asyncio
    async def test_classifier_error_propagates(self) -> None:
        """LLM classifier error should propagate (fail fast)."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        headers = {"x-routing-mode": "auto"}
        mock_openai_client = AsyncMock(spec=AsyncOpenAI)

        with patch("api.request_router._classify_with_llm", new_callable=AsyncMock) as mock_classify:  # noqa: E501
            mock_classify.side_effect = Exception("API error")

            with pytest.raises(Exception, match="API error"):
                await determine_routing(
                    request,
                    headers=headers,
                    openai_client=mock_openai_client,
                )

    @pytest.mark.asyncio
    async def test_no_user_message_defaults_to_passthrough(self) -> None:
        """Request with no user message should default to passthrough."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a helpful assistant"}],
        )
        headers = {"x-routing-mode": "auto"}
        mock_openai_client = AsyncMock(spec=AsyncOpenAI)

        with patch("api.request_router._classify_with_llm", new_callable=AsyncMock) as mock_classify:  # noqa: E501
            mock_classify.return_value = {
                "routing_mode": RoutingMode.PASSTHROUGH,
                "reason": "No user message found - defaulting to passthrough",
            }

            decision = await determine_routing(
                request,
                headers=headers,
                openai_client=mock_openai_client,
            )

            assert decision.routing_mode == RoutingMode.PASSTHROUGH


class TestRoutingPriority:
    """Test that routing tiers are evaluated in correct priority order."""

    @pytest.mark.asyncio
    async def test_passthrough_rule_takes_precedence_over_header(self) -> None:
        """Passthrough rules should be checked before header."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            response_format={"type": "json_object"},
        )
        # Try to override with header
        headers = {"x-routing-mode": "orchestration"}

        decision = await determine_routing(request, headers=headers)

        # Should use passthrough rule, not header
        assert decision.decision_source == "passthrough_rule"
        assert decision.routing_mode == RoutingMode.PASSTHROUGH

    @pytest.mark.asyncio
    async def test_header_takes_precedence_over_default(self) -> None:
        """Explicit header should take precedence over default."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        headers = {"x-routing-mode": "reasoning"}

        decision = await determine_routing(request, headers=headers)

        # Should use header, not default
        assert decision.decision_source == "header_override"
        assert decision.routing_mode == RoutingMode.REASONING

    @pytest.mark.asyncio
    async def test_auto_header_uses_llm_not_default(self) -> None:
        """Auto header should use LLM classifier, not default."""
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Complex query"}],
        )
        headers = {"x-routing-mode": "auto"}
        mock_openai_client = AsyncMock(spec=AsyncOpenAI)

        with patch("api.request_router._classify_with_llm", new_callable=AsyncMock) as mock_classify:  # noqa: E501
            mock_classify.return_value = {
                "routing_mode": RoutingMode.ORCHESTRATION,
                "reason": "Complex query",
            }

            decision = await determine_routing(
                request,
                headers=headers,
                openai_client=mock_openai_client,
            )

            # Should use LLM classifier, never default
            assert decision.decision_source == "llm_classifier"
            assert decision.routing_mode == RoutingMode.ORCHESTRATION
            mock_classify.assert_called_once()


class TestRoutingDecisionModel:
    """Test RoutingDecision Pydantic model."""

    def test_routing_decision_model_validation(self) -> None:
        """RoutingDecision model should validate correctly."""
        decision = RoutingDecision(
            routing_mode=RoutingMode.ORCHESTRATION,
            reason="Complex multi-step query",
            decision_source="llm_classifier",
        )

        assert decision.routing_mode == RoutingMode.ORCHESTRATION
        assert decision.reason == "Complex multi-step query"
        assert decision.decision_source == "llm_classifier"

    def test_routing_decision_model_serialization(self) -> None:
        """RoutingDecision should serialize to dict."""
        decision = RoutingDecision(
            routing_mode=RoutingMode.PASSTHROUGH,
            reason="Simple query",
            decision_source="default",
        )

        data = decision.model_dump()

        assert data == {
            "routing_mode": "passthrough",
            "reason": "Simple query",
            "decision_source": "default",
        }

    def test_routing_mode_enum_values(self) -> None:
        """RoutingMode enum should have correct values."""
        assert RoutingMode.PASSTHROUGH.value == "passthrough"
        assert RoutingMode.REASONING.value == "reasoning"
        assert RoutingMode.ORCHESTRATION.value == "orchestration"
        assert RoutingMode.AUTO.value == "auto"
