"""Tests for reasoning_api.utils module."""

from reasoning_api.utils import preview


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
