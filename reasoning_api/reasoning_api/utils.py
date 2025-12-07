"""
Utility functions.

This module contains general-purpose utilities useful across the reasoning API,
particularly for managing data passed to/from LLMs.
"""

from typing import Any, TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


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

        >>> preview({"k1": 1, "k2": 2, "k3": 3, "k4": 4}, max_items=2)
        {'k1': 1, 'k2': 2, '[... 2 more keys, 4 total]': '...'}
    """
    if isinstance(data, dict):
        result = {}
        items = list(data.items())
        # Truncate dict keys if too many
        if len(items) > max_items:
            for k, v in items[:max_items]:
                if skip_none and v is None:
                    continue
                result[k] = preview(v, max_str_len, max_items, skip_none)
            result[f"[... {len(items) - max_items} more keys, {len(items)} total]"] = "..."
        else:
            for k, v in items:
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


def merge_dicts(
    existing: dict[str, Any] | None,
    new: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Recursively merge two dictionaries, summing numeric values.

    Traverses nested dict structures and sums numeric fields (int, float)
    while preserving non-numeric values from the newer dict.
    Useful for accumulating usage/cost data across multiple API calls.

    Args:
        existing: Existing dict or None
        new: New dict to merge or None

    Returns:
        Merged dict with numeric values summed

    Examples:
        >>> merge_dicts(None, {"count": 5})
        {"count": 5}

        >>> existing = {"tokens": 10, "cost": 0.01}
        >>> new = {"tokens": 5, "cost": 0.005}
        >>> merge_dicts(existing, new)
        {"tokens": 15, "cost": 0.015}

        >>> existing = {"usage": {"prompt": 10, "completion": 5}, "model": "gpt-4"}
        >>> new = {"usage": {"prompt": 20, "completion": 8}, "model": "gpt-4"}
        >>> merge_dicts(existing, new)
        {"usage": {"prompt": 30, "completion": 13}, "model": "gpt-4"}
    """
    # Handle None cases
    if existing is None and new is None:
        return {}
    if existing is None:
        return new.copy() if new else {}
    if new is None:
        return existing.copy()

    # Start with a copy of existing
    merged = existing.copy()

    # Merge each key from new dict
    for key, new_value in new.items():
        if key not in merged:
            # New key - just add it
            merged[key] = new_value
        else:
            existing_value = merged[key]

            # If both are dicts, recursively merge
            if isinstance(existing_value, dict) and isinstance(new_value, dict):
                merged[key] = merge_dicts(existing_value, new_value)
            # If both are numeric (int or float), sum them
            elif isinstance(existing_value, (int, float)) and isinstance(new_value, (int, float)):
                merged[key] = existing_value + new_value
            # Otherwise, take the new value
            else:
                merged[key] = new_value

    return merged


def merge_models(existing: T | None, new: T | None) -> T | None:
    """
    Merge two Pydantic models, summing numeric fields.

    Iterates model_fields directly without dict conversion.
    Recursively handles nested BaseModel instances.
    Useful for accumulating metadata across multiple LLM API calls.

    Args:
        existing: Existing model or None
        new: New model to merge or None

    Returns:
        Merged model, or None if both inputs are None

    Examples:
        >>> from reasoning_api.conversation_utils import UsageMetadata
        >>> a = UsageMetadata(prompt_tokens=10, completion_tokens=5)
        >>> b = UsageMetadata(prompt_tokens=20, completion_tokens=8)
        >>> merged = merge_models(a, b)
        >>> merged.prompt_tokens
        30
    """
    if existing is None:
        return new
    if new is None:
        return existing

    merged = {}
    for field_name in existing.model_fields:
        existing_val = getattr(existing, field_name)
        new_val = getattr(new, field_name)

        # Recursively merge nested models
        if isinstance(existing_val, BaseModel) and isinstance(new_val, BaseModel):
            merged[field_name] = merge_models(existing_val, new_val)
        # Sum numeric types
        elif isinstance(existing_val, (int, float)) and isinstance(new_val, (int, float)):
            merged[field_name] = existing_val + new_val
        # Take newer non-None value, fallback to existing
        elif new_val is not None:
            merged[field_name] = new_val
        else:
            merged[field_name] = existing_val

    return type(existing).model_validate(merged)
