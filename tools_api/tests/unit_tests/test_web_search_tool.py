"""Tests for web search tool."""

from unittest.mock import AsyncMock, patch

import pytest

from tools_api.clients.brave_search import (
    BraveSearchResponse,
    Query,
    SearchResult,
    WebResults,
)
from tools_api.services.tools.web_search import BraveSearchTool


@pytest.fixture
def mock_search_response() -> BraveSearchResponse:
    """Create a mock search response."""
    return BraveSearchResponse(
        type="search",
        query=Query(original="test query"),
        web=WebResults(
            type="search",
            results=[
                SearchResult(
                    type="search_result",
                    title="Test Result 1",
                    url="https://example.com/1",
                    description="Test description 1",
                    age="1 day ago",
                ),
                SearchResult(
                    type="search_result",
                    title="Test Result 2",
                    url="https://example.com/2",
                    description="Test description 2",
                    age="2 days ago",
                ),
            ],
        ),
    )


@pytest.mark.asyncio
async def test_brave_search_tool_success(mock_search_response: BraveSearchResponse) -> None:
    """Test successful web search."""
    with (
        patch("tools_api.services.tools.web_search.settings") as mock_settings,
        patch("tools_api.services.tools.web_search.BraveSearchClient") as mock_client_class,
    ):
        # Setup settings mock
        mock_settings.brave_api_key = "test-api-key"

        # Setup client mock
        mock_client = AsyncMock()
        mock_client.search.return_value = mock_search_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        # Execute tool
        tool = BraveSearchTool()
        result = await tool(q="test query")

        # Verify
        assert result.success is True
        assert "web_results" in result.result
        assert len(result.result["web_results"]) == 2
        assert result.result["web_results"][0]["title"] == "Test Result 1"
        assert result.result["web_results"][0]["url"] == "https://example.com/1"


@pytest.mark.asyncio
async def test_brave_search_tool_missing_api_key() -> None:
    """Test web search with missing API key."""
    with patch("tools_api.services.tools.web_search.settings") as mock_settings:
        mock_settings.brave_api_key = ""

        tool = BraveSearchTool()
        result = await tool(q="test query")

        assert result.success is False
        assert "api key not configured" in result.error.lower()


@pytest.mark.asyncio
async def test_brave_search_tool_with_filters(mock_search_response: BraveSearchResponse) -> None:
    """Test web search with various filters."""
    with (
        patch("tools_api.services.tools.web_search.settings") as mock_settings,
        patch("tools_api.services.tools.web_search.BraveSearchClient") as mock_client_class,
    ):
        # Setup settings mock
        mock_settings.brave_api_key = "test-api-key"

        # Setup client mock
        mock_client = AsyncMock()
        mock_client.search.return_value = mock_search_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        # Execute tool with filters
        tool = BraveSearchTool()
        result = await tool(
            q="test query",
            count=10,
            country="GB",
            safesearch="strict",
            freshness="pw",
        )

        # Verify
        assert result.success is True
        # Verify the search was called with correct parameters
        mock_client.search.assert_called_once()
        call_params = mock_client.search.call_args[1]["params"]
        assert call_params.q == "test query"
        assert call_params.count == 10
        assert call_params.country == "GB"
        assert call_params.safesearch == "strict"
        assert call_params.freshness == "pw"


@pytest.mark.asyncio
async def test_brave_search_tool_empty_results() -> None:
    """Test web search with empty results."""
    empty_response = BraveSearchResponse(
        type="search",
        query=Query(original="test query"),
        web=WebResults(type="search", results=[]),
    )

    with (
        patch("tools_api.services.tools.web_search.settings") as mock_settings,
        patch("tools_api.services.tools.web_search.BraveSearchClient") as mock_client_class,
    ):
        # Setup settings mock
        mock_settings.brave_api_key = "test-api-key"

        # Setup client mock
        mock_client = AsyncMock()
        mock_client.search.return_value = empty_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        # Execute tool
        tool = BraveSearchTool()
        result = await tool(q="test query")

        # Verify
        assert result.success is True
        assert "web_results" not in result.result  # No results, so key not added
        assert "query" in result.result


@pytest.mark.asyncio
async def test_brave_search_tool_metadata() -> None:
    """Test tool metadata."""
    tool = BraveSearchTool()

    assert tool.name == "brave_search"
    assert "brave" in tool.description.lower() or "search" in tool.description.lower()
    assert "q" in tool.parameters["properties"]
    assert "count" in tool.parameters["properties"]
    assert "search" in tool.tags or "brave" in tool.tags


@pytest.mark.asyncio
async def test_brave_search_tool_api_error() -> None:
    """Test handling of API errors."""
    with (
        patch("tools_api.services.tools.web_search.settings") as mock_settings,
        patch("tools_api.services.tools.web_search.BraveSearchClient") as mock_client_class,
    ):
        # Setup settings mock
        mock_settings.brave_api_key = "test-api-key"

        # Setup mock to raise an exception
        mock_client = AsyncMock()
        mock_client.search.side_effect = Exception("API Error")
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        # Execute tool
        tool = BraveSearchTool()
        result = await tool(q="test query")

        # Verify error is handled
        assert result.success is False
        assert "api error" in result.error.lower()
