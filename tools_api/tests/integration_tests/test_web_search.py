"""
Integration tests for web search tool (Brave Search API).

These tests make real API calls to Brave Search when BRAVE_API_KEY is set.
They verify the complete flow: HTTP request → WebSearchTool → BraveSearchClient → Brave API
"""

import os

import pytest


# Skip all tests in this file if BRAVE_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.getenv("BRAVE_API_KEY"),
    reason="BRAVE_API_KEY environment variable not set",
)


@pytest.mark.asyncio
async def test_web_search_basic_query(tools_api_client) -> None:
    """Test basic web search with real API call."""
    response = await tools_api_client.post(
        "/tools/web_search",
        json={"q": "Python programming"},
    )

    assert response.status_code == 200
    result = response.json()

    # Verify success
    assert result["success"] is True
    assert "result" in result

    # Verify response structure
    assert "query" in result["result"]
    assert result["result"]["query"]["original"] == "Python programming"

    # Should have web results
    assert "web_results" in result["result"]
    web_results = result["result"]["web_results"]
    assert len(web_results) > 0

    # Verify result structure
    first_result = web_results[0]
    assert "title" in first_result
    assert "url" in first_result
    assert "description" in first_result
    assert first_result["url"].startswith("http")


@pytest.mark.asyncio
async def test_web_search_with_filters(tools_api_client) -> None:
    """Test web search with various filters and parameters."""
    response = await tools_api_client.post(
        "/tools/web_search",
        json={
            "q": "artificial intelligence",
            "count": 10,
            "country": "US",
            "search_lang": "en",
            "safesearch": "moderate",
        },
    )

    assert response.status_code == 200
    result = response.json()

    assert result["success"] is True
    assert "web_results" in result["result"]

    # Should have up to 10 results
    web_results = result["result"]["web_results"]
    assert len(web_results) <= 10
    assert len(web_results) > 0


@pytest.mark.asyncio
async def test_web_search_freshness_filter(tools_api_client) -> None:
    """Test web search with freshness filter (recent results)."""
    response = await tools_api_client.post(
        "/tools/web_search",
        json={
            "q": "latest technology news",
            "freshness": "pd",  # Past day
            "count": 5,
        },
    )

    assert response.status_code == 200
    result = response.json()

    assert result["success"] is True
    if "web_results" in result["result"]:
        # If we got results, verify structure
        web_results = result["result"]["web_results"]
        assert len(web_results) <= 5
        for item in web_results:
            assert "title" in item
            assert "url" in item


@pytest.mark.asyncio
async def test_web_search_news_results(tools_api_client) -> None:
    """Test web search that may include news results."""
    response = await tools_api_client.post(
        "/tools/web_search",
        json={
            "q": "Stock market news",
            "count": 5,
        },
    )

    assert response.status_code == 200
    result = response.json()

    assert result["success"] is True
    # News results are optional - just verify structure if present
    assert "news_results" in result["result"]
    news_results = result["result"]["news_results"]
    for item in news_results:
        assert "title" in item
        assert "url" in item
        # News items may have additional fields
        if "breaking" in item:
            assert isinstance(item["breaking"], bool)


@pytest.mark.asyncio
async def test_web_search_pagination(tools_api_client) -> None:
    """Test web search with offset pagination."""
    # First page
    response1 = await tools_api_client.post(
        "/tools/web_search",
        json={"q": "machine learning", "count": 5, "offset": 0},
    )

    # Second page
    response2 = await tools_api_client.post(
        "/tools/web_search",
        json={"q": "machine learning", "count": 5, "offset": 1},
    )

    assert response1.status_code == 200
    assert response2.status_code == 200

    result1 = response1.json()
    result2 = response2.json()

    assert result1["success"] is True
    assert result2["success"] is True

    # Both should have results
    assert "web_results" in result1["result"]
    assert "web_results" in result2["result"]

    # Results should be different (different pages)
    if len(result1["result"]["web_results"]) > 0 and len(
        result2["result"]["web_results"],
    ) > 0:
        first_page_urls = {r["url"] for r in result1["result"]["web_results"]}
        second_page_urls = {r["url"] for r in result2["result"]["web_results"]}
        # At least some URLs should be different
        assert len(first_page_urls & second_page_urls) < len(first_page_urls)
