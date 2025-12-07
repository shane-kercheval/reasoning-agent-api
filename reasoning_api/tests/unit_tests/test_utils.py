"""Tests for reasoning_api.utils module."""

import pytest
from reasoning_api.utils import preview, merge_dicts, merge_models
from reasoning_api.conversation_utils import (
    UsageMetadata,
    CostMetadata,
    ResponseMetadata,
)
from reasoning_api.context_manager import (
    ContextBreakdown,
    ContextUtilizationMetadata,
)


class TestPreview:
    """Tests for the preview function."""

    def test_string_under_limit_unchanged(self) -> None:
        """Strings under max_str_len are returned unchanged."""
        result = preview("short string", max_str_len=100)
        assert result == "short string"

    def test_string_at_limit_unchanged(self) -> None:
        """Strings exactly at max_str_len are returned unchanged."""
        text = "a" * 100
        result = preview(text, max_str_len=100)
        assert result == text

    def test_string_over_limit_truncated(self) -> None:
        """Strings over max_str_len are truncated with indicator."""
        text = "a" * 500
        result = preview(text, max_str_len=100)
        assert result == "a" * 100 + "... [truncated, 500 chars total]"

    def test_list_under_limit_unchanged(self) -> None:
        """Lists under max_items are returned unchanged."""
        result = preview([1, 2, 3], max_items=5)
        assert result == [1, 2, 3]

    def test_list_at_limit_unchanged(self) -> None:
        """Lists exactly at max_items are returned unchanged."""
        result = preview([1, 2, 3], max_items=3)
        assert result == [1, 2, 3]

    def test_list_over_limit_truncated(self) -> None:
        """Lists over max_items are truncated with indicator."""
        result = preview([1, 2, 3, 4, 5], max_items=2)
        assert result == [1, 2, "[... 3 more items, 5 total]"]

    def test_empty_list(self) -> None:
        """Empty lists are returned unchanged."""
        result = preview([])
        assert result == []

    def test_empty_dict(self) -> None:
        """Empty dicts are returned unchanged."""
        result = preview({})
        assert result == {}

    def test_dict_keys_truncated(self) -> None:
        """Dicts with more keys than max_items are truncated."""
        data = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        result = preview(data, max_items=2)
        # Only first 2 keys preserved
        assert result["a"] == 1
        assert result["b"] == 2
        # Remaining keys truncated
        assert "c" not in result
        assert "d" not in result
        assert "e" not in result
        # Truncation indicator added
        assert result["[... 3 more keys, 5 total]"] == "..."

    def test_dict_at_limit_unchanged(self) -> None:
        """Dicts exactly at max_items limit are returned unchanged."""
        data = {"a": 1, "b": 2, "c": 3}
        result = preview(data, max_items=3)
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_empty_string(self) -> None:
        """Empty strings are returned unchanged."""
        result = preview("")
        assert result == ""

    def test_primitives_unchanged(self) -> None:
        """Primitives (int, float, bool, None) pass through unchanged."""
        assert preview(42) == 42
        assert preview(3.14) == 3.14
        assert preview(True) is True
        assert preview(False) is False
        assert preview(None) is None

    def test_dict_with_string_values(self) -> None:
        """Dict string values are truncated recursively."""
        data = {"short": "hello", "long": "x" * 500}
        result = preview(data, max_str_len=100)
        assert result["short"] == "hello"
        assert result["long"] == "x" * 100 + "... [truncated, 500 chars total]"

    def test_dict_with_list_values(self) -> None:
        """Dict list values are truncated recursively."""
        data = {"items": [1, 2, 3, 4, 5, 6, 7]}
        result = preview(data, max_items=3)
        assert result["items"] == [1, 2, 3, "[... 4 more items, 7 total]"]

    def test_nested_dict(self) -> None:
        """Nested dicts are processed recursively."""
        data = {"outer": {"inner": "a" * 500}}
        result = preview(data, max_str_len=100)
        assert result["outer"]["inner"] == "a" * 100 + "... [truncated, 500 chars total]"

    def test_nested_list(self) -> None:
        """Nested lists are processed recursively."""
        data = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        result = preview(data, max_items=2)
        # Outer list truncated to 2 items, inner lists also truncated
        assert result == [
            [1, 2, "[... 3 more items, 5 total]"],
            [6, 7, "[... 3 more items, 5 total]"],
        ]

    def test_list_of_dicts(self) -> None:
        """Lists of dicts have dict values truncated recursively."""
        data = [
            {"text": "a" * 500},
            {"text": "b" * 500},
            {"text": "c" * 500},
            {"text": "d" * 500},
        ]
        result = preview(data, max_str_len=100, max_items=2)
        assert len(result) == 3  # 2 items + truncation indicator
        assert result[0]["text"] == "a" * 100 + "... [truncated, 500 chars total]"
        assert result[1]["text"] == "b" * 100 + "... [truncated, 500 chars total]"
        assert result[2] == "[... 2 more items, 4 total]"

    def test_deeply_nested_structure(self) -> None:
        """Deeply nested structures are fully processed."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "text": "x" * 1000,
                        "items": list(range(20)),
                    },
                },
            },
        }
        result = preview(data, max_str_len=50, max_items=3)
        inner = result["level1"]["level2"]["level3"]
        assert inner["text"] == "x" * 50 + "... [truncated, 1000 chars total]"
        assert inner["items"] == [0, 1, 2, "[... 17 more items, 20 total]"]

    def test_mixed_types_in_list(self) -> None:
        """Lists with mixed types are handled correctly."""
        data = ["short", "x" * 500, 42, None, {"key": "value"}]
        result = preview(data, max_str_len=100, max_items=10)
        assert result[0] == "short"
        assert result[1] == "x" * 100 + "... [truncated, 500 chars total]"
        assert result[2] == 42
        assert result[3] is None
        assert result[4] == {"key": "value"}

    def test_web_scraper_like_result(self) -> None:
        """Test with a structure similar to web scraper results."""
        data = {
            "url": "https://example.com/page",
            "status": 200,
            "title": "Example Page",
            "text": "This is the main content of the page. " * 100,
            "references": [
                {"id": 1, "url": "https://link1.com", "text": "Link 1"},
                {"id": 2, "url": "https://link2.com", "text": "Link 2"},
                {"id": 3, "url": "https://link3.com", "text": "Link 3"},
                {"id": 4, "url": "https://link4.com", "text": "Link 4"},
                {"id": 5, "url": "https://link5.com", "text": "Link 5"},
            ],
            "navigation": [
                {"text": "Home", "url": "/"},
                {"text": "About", "url": "/about"},
                {"text": "Contact", "url": "/contact"},
                {"text": "Products", "url": "/products"},
            ],
        }
        # Use max_items=6 to include all top-level keys (6 keys total)
        result = preview(data, max_str_len=100, max_items=6)
        # Metadata preserved
        assert result["url"] == "https://example.com/page"
        assert result["status"] == 200
        assert result["title"] == "Example Page"

        # Text truncated
        assert "truncated" in result["text"]
        assert "chars total" in result["text"]

        # References list has 5 items (within max_items=6), all preserved
        assert len(result["references"]) == 5
        # Inner dicts have 3 keys each (within max_items=6)
        assert result["references"][0] == {"id": 1, "url": "https://link1.com", "text": "Link 1"}

        # Navigation has 4 items (under max_items=6)
        assert len(result["navigation"]) == 4

    def test_default_parameters(self) -> None:
        """Test default parameter values (300 chars, 3 items)."""
        long_text = "a" * 500
        many_items = list(range(10))

        text_result = preview(long_text)
        assert text_result == "a" * 300 + "... [truncated, 500 chars total]"

        list_result = preview(many_items)
        assert list_result == [0, 1, 2, "[... 7 more items, 10 total]"]

    def test_custom_parameters(self) -> None:
        """Test custom max_str_len and max_items."""
        # With max_items=1, dict keys are also truncated to 1 key + indicator
        result = preview(
            {"text": "a" * 100, "items": [1, 2, 3, 4, 5]},
            max_str_len=20,
            max_items=1,
        )
        # Only first key preserved, truncated string value
        assert result["text"] == "a" * 20 + "... [truncated, 100 chars total]"
        # Indicator for remaining keys
        assert "[... 1 more keys, 2 total]" in result
        # "items" key was truncated out
        assert "items" not in result

    def test_does_not_mutate_input(self) -> None:
        """Verify original data is not mutated."""
        original = {"text": "a" * 500, "items": [1, 2, 3, 4, 5]}
        original_copy = {"text": "a" * 500, "items": [1, 2, 3, 4, 5]}

        preview(original, max_str_len=100, max_items=2)
        assert original == original_copy

    def test_does_not_mutate_input_with_none_values(self) -> None:
        """Verify original data with None values is not mutated when skip_none=True."""
        original = {"title": "Hello", "description": None, "raw_html": None}
        original_copy = {"title": "Hello", "description": None, "raw_html": None}

        result = preview(original, skip_none=True)
        # Original unchanged
        assert original == original_copy
        # Result has None keys removed
        assert result == {"title": "Hello"}

    def test_skip_none_default_true(self) -> None:
        """By default, None values are skipped in dicts."""
        data = {"title": "Hello", "description": None}
        result = preview(data)
        assert result == {"title": "Hello"}
        assert "description" not in result

    def test_skip_none_false_preserves_none(self) -> None:
        """When skip_none=False, None values are preserved."""
        data = {"title": "Hello", "description": None}
        result = preview(data, skip_none=False)
        assert result == {"title": "Hello", "description": None}

    def test_skip_none_nested_dicts(self) -> None:
        """skip_none works recursively in nested dicts."""
        data = {
            "outer": {
                "inner": "value",
                "empty": None,
            },
            "also_none": None,
        }
        result = preview(data, skip_none=True)
        assert result == {"outer": {"inner": "value"}}

    def test_skip_none_in_list_of_dicts(self) -> None:
        """skip_none removes None values from dicts inside lists."""
        data = [
            {"id": 1, "name": "First", "extra": None},
            {"id": 2, "name": "Second", "extra": None},
        ]
        result = preview(data, skip_none=True)
        assert result == [
            {"id": 1, "name": "First"},
            {"id": 2, "name": "Second"},
        ]

    def test_skip_none_does_not_remove_none_from_lists(self) -> None:
        """skip_none only affects dict values, not list elements."""
        data = [1, None, 3, None, 5]
        result = preview(data, skip_none=True)
        assert result == [1, None, 3, "[... 2 more items, 5 total]"]

    def test_skip_none_with_all_none_dict(self) -> None:
        """Dict with all None values becomes empty dict."""
        data = {"a": None, "b": None, "c": None}
        result = preview(data, skip_none=True)
        assert result == {}

    def test_skip_none_web_scraper_like_result(self) -> None:
        """Test skip_none with web scraper-like result containing null fields."""
        data = {
            "url": "https://example.com",
            "status": 200,
            "title": "Example",
            "description": None,
            "raw_html": None,
            "language": None,
        }
        # With default max_items=3, only first 3 keys are kept + truncation indicator
        # First 3 keys happen to be the non-None ones (url, status, title)
        result = preview(data, skip_none=True)
        assert result["url"] == "https://example.com"
        assert result["status"] == 200
        assert result["title"] == "Example"
        # Truncation indicator added (6 original keys, 3 kept)
        assert "[... 3 more keys, 6 total]" in result


class TestMergeDicts:
    """Tests for merge_dicts function."""

    def test__merge_dicts__both_none(self) -> None:
        """Test merging when both dicts are None."""
        result = merge_dicts(None, None)
        assert result == {}

    def test__merge_dicts__existing_none(self) -> None:
        """Test merging when existing is None."""
        new = {"count": 5}
        result = merge_dicts(None, new)
        assert result == {"count": 5}
        assert result is not new  # Should be a copy

    def test__merge_dicts__new_none(self) -> None:
        """Test merging when new is None."""
        existing = {"count": 10}
        result = merge_dicts(existing, None)
        assert result == {"count": 10}
        assert result is not existing  # Should be a copy

    def test__merge_dicts__sum_integers(self) -> None:
        """Test that integer values are summed."""
        existing = {"tokens": 10, "calls": 2}
        new = {"tokens": 5, "calls": 3}
        result = merge_dicts(existing, new)
        assert result == {"tokens": 15, "calls": 5}

    def test__merge_dicts__sum_floats(self) -> None:
        """Test that float values are summed."""
        existing = {"cost": 0.01, "latency": 1.5}
        new = {"cost": 0.005, "latency": 0.8}
        result = merge_dicts(existing, new)
        assert result == {"cost": 0.015, "latency": 2.3}

    def test__merge_dicts__recursive_merge(self) -> None:
        """Test that nested dicts are recursively merged."""
        existing = {"usage": {"prompt": 10, "completion": 5}, "model": "gpt-4"}
        new = {"usage": {"prompt": 20, "completion": 8}, "model": "gpt-4"}
        result = merge_dicts(existing, new)
        assert result == {
            "usage": {"prompt": 30, "completion": 13},
            "model": "gpt-4",
        }

    def test__merge_dicts__preserve_non_numeric(self) -> None:
        """Test that non-numeric values are replaced with new value."""
        existing = {"name": "old", "count": 10}
        new = {"name": "new", "count": 5}
        result = merge_dicts(existing, new)
        assert result == {"name": "new", "count": 15}

    def test__merge_dicts__preserve_keys_from_both(self) -> None:
        """Test that keys from both dicts are preserved."""
        existing = {"a": 1, "b": "text"}
        new = {"c": 3, "d": "more"}
        result = merge_dicts(existing, new)
        assert result == {"a": 1, "b": "text", "c": 3, "d": "more"}

    def test__merge_dicts__complex_real_world_example(self) -> None:
        """Test with real-world usage/cost metadata structure."""
        existing = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "cost": {"prompt_cost": 0.00001, "completion_cost": 0.00002, "total_cost": 0.00003},
            "model": "gpt-4",
            "routing_path": "reasoning",
        }
        new = {
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
            "cost": {"prompt_cost": 0.00002, "completion_cost": 0.00003, "total_cost": 0.00005},
            "model": "gpt-4",
        }
        result = merge_dicts(existing, new)

        assert result["usage"]["prompt_tokens"] == 30
        assert result["usage"]["completion_tokens"] == 13
        assert result["usage"]["total_tokens"] == 43
        assert result["model"] == "gpt-4"
        assert result["routing_path"] == "reasoning"
        assert result["cost"]["prompt_cost"] == pytest.approx(0.00003)
        assert result["cost"]["completion_cost"] == pytest.approx(0.00005)
        assert result["cost"]["total_cost"] == pytest.approx(0.00008)


class TestMergeModels:
    """Tests for merge_models function with Pydantic models."""

    def test__merge_models__both_none(self) -> None:
        """Test merging when both models are None."""
        result = merge_models(None, None)
        assert result is None

    def test__merge_models__existing_none(self) -> None:
        """Test merging when existing is None returns new."""
        new = UsageMetadata(prompt_tokens=10, completion_tokens=5)
        result = merge_models(None, new)
        assert result is new

    def test__merge_models__new_none(self) -> None:
        """Test merging when new is None returns existing."""
        existing = UsageMetadata(prompt_tokens=10, completion_tokens=5)
        result = merge_models(existing, None)
        assert result is existing

    def test__merge_models__sum_numeric_fields(self) -> None:
        """Test that numeric fields are summed."""
        existing = UsageMetadata(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        new = UsageMetadata(prompt_tokens=20, completion_tokens=8, total_tokens=28)
        result = merge_models(existing, new)

        assert result is not None
        assert result.prompt_tokens == 30
        assert result.completion_tokens == 13
        assert result.total_tokens == 43

    def test__merge_models__sum_float_fields(self) -> None:
        """Test that float fields are summed."""
        existing = CostMetadata(prompt_cost=0.01, completion_cost=0.02, total_cost=0.03)
        new = CostMetadata(prompt_cost=0.005, completion_cost=0.008, total_cost=0.013)
        result = merge_models(existing, new)

        assert result is not None
        assert result.prompt_cost == pytest.approx(0.015)
        assert result.completion_cost == pytest.approx(0.028)
        assert result.total_cost == pytest.approx(0.043)

    def test__merge_models__string_replacement(self) -> None:
        """Test that string fields take newer value."""
        existing = ResponseMetadata(model="gpt-4", routing_path="passthrough")
        new = ResponseMetadata(model="gpt-4o", routing_path="reasoning")
        result = merge_models(existing, new)

        assert result is not None
        assert result.model == "gpt-4o"
        assert result.routing_path == "reasoning"

    def test__merge_models__nested_model_merge(self) -> None:
        """Test that nested models are recursively merged."""
        existing = ResponseMetadata(
            usage=UsageMetadata(prompt_tokens=10, completion_tokens=5),
            cost=CostMetadata(total_cost=0.01),
            model="gpt-4",
        )
        new = ResponseMetadata(
            usage=UsageMetadata(prompt_tokens=20, completion_tokens=8),
            cost=CostMetadata(total_cost=0.02),
            model="gpt-4",
        )
        result = merge_models(existing, new)

        assert result is not None
        assert result.usage is not None
        assert result.usage.prompt_tokens == 30
        assert result.usage.completion_tokens == 13
        assert result.cost is not None
        assert result.cost.total_cost == pytest.approx(0.03)

    def test__merge_models__partial_fields(self) -> None:
        """Test merging models with different optional fields set."""
        existing = ResponseMetadata(model="gpt-4", routing_path="passthrough")
        new = ResponseMetadata(usage=UsageMetadata(prompt_tokens=100))
        result = merge_models(existing, new)

        assert result is not None
        assert result.model == "gpt-4"  # From existing (new is None)
        assert result.routing_path == "passthrough"  # From existing (new is None)
        assert result.usage is not None
        assert result.usage.prompt_tokens == 100

    def test__merge_models__none_to_value(self) -> None:
        """Test that None fields in existing are replaced by new values."""
        existing = ResponseMetadata(model="gpt-4")
        new = ResponseMetadata(routing_path="reasoning")
        result = merge_models(existing, new)

        assert result is not None
        assert result.model == "gpt-4"  # Existing preserved
        assert result.routing_path == "reasoning"  # New value used

    def test__merge_models__deeply_nested(self) -> None:
        """Test merging with deeply nested models."""
        existing = ResponseMetadata(
            context_utilization=ContextUtilizationMetadata(
                model_name="gpt-3",
                strategy="balanced",
                breakdown=ContextBreakdown(system_messages=100, user_messages=500),
            ),
        )
        new = ResponseMetadata(
            context_utilization=ContextUtilizationMetadata(
                model_name="gpt-4",
                breakdown=ContextBreakdown(system_messages=50, user_messages=200),
            ),
        )
        result = merge_models(existing, new)

        assert result is not None
        assert result.context_utilization is not None
        assert result.context_utilization.model_name == "gpt-4"
        assert result.context_utilization.strategy == "balanced"
        assert result.context_utilization.breakdown is not None
        assert result.context_utilization.breakdown.system_messages == 150
        assert result.context_utilization.breakdown.user_messages == 700

    def test__merge_models__type_preservation(self) -> None:
        """Test that result type matches input type."""
        existing = UsageMetadata(prompt_tokens=10)
        new = UsageMetadata(prompt_tokens=20)
        result = merge_models(existing, new)

        assert isinstance(result, UsageMetadata)

    def test__merge_models__mixed_none_nested(self) -> None:
        """Test when one model has nested model and other has None."""
        existing = ResponseMetadata(
            usage=UsageMetadata(prompt_tokens=100),
            model="gpt-4",
        )
        new = ResponseMetadata(
            usage=None,  # Explicitly None
            model="gpt-4o",
        )
        result = merge_models(existing, new)

        assert result is not None
        assert result.model == "gpt-4o"  # New value
        # usage should be preserved from existing since new is None
        assert result.usage is not None
        assert result.usage.prompt_tokens == 100
