"""
Standardized test tools for reasoning agent tests.

This module provides centralized tool definitions to eliminate duplication
across test files and ensure consistency in tool behavior.
"""

import asyncio
import time
from typing import Any

from api.tools import Tool, function_to_tool


# =============================================================================
# Basic Tools - Standard implementations used across most tests
# =============================================================================

def weather_tool(location: str) -> dict[str, Any]:
    """
    Get weather information for a location.

    Standard weather tool that returns consistent format across all tests.

    Args:
        location: Name of the location to get weather for.

    Returns:
        Weather information including temperature, condition, and humidity.
    """
    return {
        "location": location,
        "temperature": "22°C",
        "condition": "Partly cloudy",
        "humidity": "65%",
        "source": "mock_weather_api",
    }


def search_tool(query: str, num_results: int = 5) -> dict[str, Any]:
    """
    Search the web for information.

    Standard search tool that returns consistent format across all tests.

    Args:
        query: Search query string.
        num_results: Number of results to return.

    Returns:
        Search results with title, URL, and snippet for each result.
    """
    return {
        "query": query,
        "results": [
            {
                "title": f"Result {i+1} for {query}",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"Mock search result {i+1} containing information about {query}",
            }
            for i in range(num_results)
        ],
        "total_results": num_results,
    }


def failing_tool(should_fail: bool = True) -> dict[str, Any]:
    """
    Tool that can be configured to fail for error testing.

    Args:
        should_fail: Whether the tool should raise an exception.

    Returns:
        Success status if not failing.

    Raises:
        ValueError: When should_fail is True.
    """
    if should_fail:
        raise ValueError("Tool intentionally failed")
    return {"success": True, "status": "Tool executed successfully"}


async def async_delay_tool(delay: float) -> str:
    """
    Async tool that introduces a delay for concurrency testing.

    Args:
        delay: Number of seconds to delay.

    Returns:
        Completion message with delay duration.
    """
    await asyncio.sleep(delay)
    return f"Completed after {delay}s"


def sleep_tool(duration: int) -> str:
    """
    Tool that sleeps for a specified duration to test timeouts.

    Args:
        duration: Number of seconds to sleep.

    Returns:
        Message indicating sleep duration.
    """
    time.sleep(duration)
    return f"Slept for {duration} seconds"


# =============================================================================
# Extended Tools - Enhanced versions with additional fields
# =============================================================================

def enhanced_weather_tool(location: str) -> dict[str, Any]:
    """
    Enhanced weather tool with additional meteorological data.

    Args:
        location: Name of the location to get weather for.

    Returns:
        Extended weather information including wind speed.
    """
    return {
        "location": location,
        "temperature": "25°C",
        "condition": "Partly cloudy",
        "humidity": "65%",
        "wind_speed": "10 km/h",
        "source": "enhanced_weather_api",
    }


def database_search_tool(query: str, limit: int = 5) -> dict[str, Any]:
    """
    Database search tool with scored results.

    Args:
        query: Search query string.
        limit: Maximum number of results to return.

    Returns:
        Search results with relevance scores.
    """
    results = [
        {
            "title": f"Database result {i+1} for {query}",
            "content": f"Content about {query} from database entry {i+1}",
            "score": round(1.0 - (i * 0.1), 2),
            "id": f"db_{i+1}",
        }
        for i in range(limit)
    ]

    return {
        "query": query,
        "total_results": len(results),
        "results": results,
        "source": "mock_database",
    }


# =============================================================================
# Stateful Tools - Tools that maintain state for complex testing
# =============================================================================

# Shared memory store for stateful tools
_memory_store: dict[str, Any] = {}


def store_data_tool(key: str, value: object) -> dict[str, str]:
    """
    Store data in memory for stateful testing.

    Args:
        key: Storage key.
        value: Value to store.

    Returns:
        Storage confirmation with key and stringified value.
    """
    _memory_store[key] = value
    return {"status": "stored", "key": key, "value": str(value)}


def retrieve_data_tool(key: str) -> dict[str, Any]:
    """
    Retrieve data from memory storage.

    Args:
        key: Storage key to retrieve.

    Returns:
        Retrieved data or not found status.
    """
    if key in _memory_store:
        return {"status": "found", "key": key, "value": _memory_store[key]}
    return {"status": "not_found", "key": key, "value": None}


def list_keys_tool() -> dict[str, Any]:
    """
    List all stored keys in memory.

    Returns:
        List of all keys and count.
    """
    return {"keys": list(_memory_store.keys()), "count": len(_memory_store)}


def clear_store_tool() -> dict[str, str]:
    """
    Clear the memory store.

    Returns:
        Confirmation of store clearing.
    """
    _memory_store.clear()
    return {"status": "cleared", "count": "0"}


# =============================================================================
# Analysis Tools - Tools for content processing testing
# =============================================================================

def sentiment_analysis_tool(text: str) -> dict[str, Any]:
    """
    Analyze sentiment of text.

    Args:
        text: Text to analyze.

    Returns:
        Sentiment analysis results with confidence.
    """
    # Simple keyword-based sentiment
    text_lower = text.lower()
    if any(word in text_lower for word in ["good", "great", "excellent", "amazing"]):
        sentiment = "positive"
        confidence = 0.85
    elif any(word in text_lower for word in ["bad", "terrible", "awful", "horrible"]):
        sentiment = "negative"
        confidence = 0.82
    else:
        sentiment = "neutral"
        confidence = 0.70

    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "length": len(text),
    }


def calculator_tool(operation: str, a: float, b: float) -> dict[str, Any]:
    """
    Perform basic mathematical operations.

    Args:
        operation: Mathematical operation (add, subtract, multiply, divide).
        a: First operand.
        b: Second operand.

    Returns:
        Calculation result.

    Raises:
        ValueError: For unsupported operations or division by zero.
    """
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else (_ for _ in ()).throw(ValueError("Division by zero")),  # noqa: E501
    }

    if operation not in operations:
        raise ValueError(f"Unsupported operation: {operation}")

    result = operations[operation](a, b)
    return {
        "operation": operation,
        "a": a,
        "b": b,
        "result": result,
        "source": "calculator_tool",
    }


# =============================================================================
# Tool Collections - Predefined sets for different test scenarios
# =============================================================================

def create_basic_tools() -> list[Tool]:
    """
    Create basic tool set for simple tests.

    Returns:
        List containing weather tool only.
    """
    return [function_to_tool(weather_tool)]


def create_full_tools() -> list[Tool]:
    """
    Create full tool set for complex workflow tests.

    Returns:
        List containing weather and search tools.
    """
    return [
        function_to_tool(weather_tool),
        function_to_tool(search_tool),
    ]


def create_error_tools() -> list[Tool]:
    """
    Create tool set for error handling tests.

    Returns:
        List containing weather and failing tools.
    """
    return [
        function_to_tool(weather_tool),
        function_to_tool(failing_tool),
    ]


def create_async_tools() -> list[Tool]:
    """
    Create tool set for concurrency tests.

    Returns:
        List containing weather and async delay tools.
    """
    return [
        function_to_tool(weather_tool),
        function_to_tool(async_delay_tool),
    ]


def create_extended_tools() -> list[Tool]:
    """
    Create extended tool set for comprehensive testing.

    Returns:
        List containing all standard tools plus analysis tools.
    """
    return [
        function_to_tool(weather_tool),
        function_to_tool(search_tool),
        function_to_tool(sentiment_analysis_tool),
        function_to_tool(calculator_tool),
    ]


def create_stateful_tools() -> list[Tool]:
    """
    Create stateful tool set for memory/persistence testing.

    Returns:
        List containing data storage and retrieval tools.
    """
    return [
        function_to_tool(store_data_tool),
        function_to_tool(retrieve_data_tool),
        function_to_tool(list_keys_tool),
        function_to_tool(clear_store_tool),
    ]


def create_timeout_tools() -> list[Tool]:
    """
    Create tool set for timeout testing.

    Returns:
        List containing weather and sleep tools.
    """
    return [
        function_to_tool(weather_tool),
        function_to_tool(sleep_tool),
    ]


# Standard tool collections as constants for easy access
BASIC_TOOLS = create_basic_tools()
FULL_TOOLS = create_full_tools()
ERROR_TOOLS = create_error_tools()
ASYNC_TOOLS = create_async_tools()
EXTENDED_TOOLS = create_extended_tools()
STATEFUL_TOOLS = create_stateful_tools()
SLEEP_TOOLS = create_timeout_tools()
