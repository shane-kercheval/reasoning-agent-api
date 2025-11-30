"""
Utility functions.

This module contains general-purpose utilities useful across the reasoning API,
particularly for managing data passed to/from LLMs.
"""

from typing import Any


def preview(
    data: Any,
    max_str_len: int = 300,
    max_items: int = 3,
    skip_none: bool = True,
) -> Any:
    """
    Generate a truncated preview of data for LLM consumption.

    Recursively truncates strings and arrays while preserving structure,
    useful for giving LLMs a quick overview of tool results without
    overwhelming context windows.

    Args:
        data: Any data structure (dict, list, str, primitives)
        max_str_len: Maximum string length before truncation
        max_items: Maximum array items before truncation
        skip_none: If True, omit keys with None values from dicts

    Returns:
        Truncated copy with indicators showing omitted content

    Examples:
        >>> preview({"text": "a" * 500})
        {'text': 'aaa... [truncated, 500 chars total]'}

        >>> preview({"items": [1, 2, 3, 4, 5]}, max_items=2)
        {'items': [1, 2, '[... 3 more items, 5 total]']}

        >>> preview({"title": "Hello", "description": None})
        {'title': 'Hello'}
    """
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            if skip_none and v is None:
                continue
            result[k] = preview(v, max_str_len, max_items, skip_none)
        return result

    if isinstance(data, list):
        if len(data) <= max_items:
            return [preview(item, max_str_len, max_items, skip_none) for item in data]
        truncated = [
            preview(item, max_str_len, max_items, skip_none) for item in data[:max_items]
        ]
        truncated.append(f"[... {len(data) - max_items} more items, {len(data)} total]")
        return truncated

    if isinstance(data, str):
        if len(data) <= max_str_len:
            return data
        return f"{data[:max_str_len]}... [truncated, {len(data)} chars total]"

    # Primitives (int, float, bool, None) pass through unchanged
    return data
