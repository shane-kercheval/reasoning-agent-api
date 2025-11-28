"""Web search tool."""

from typing import Any

from tools_api.clients.brave_search import (
    BraveSearchClient,
    BraveSearchParams,
    BraveSearchResponse,
)
from tools_api.config import settings
from tools_api.services.base import BaseTool


class WebSearchTool(BaseTool):
    """Web search tool."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "web_search"

    @property
    def description(self) -> str:
        """Tool description."""
        return "Search the web for information. Returns web results, news, and videos."

    @property
    def parameters(self) -> dict[str, Any]:
        """Tool parameters JSON Schema."""
        return {
            "type": "object",
            "properties": {
                "q": {
                    "type": "string",
                    "description": "Search query (max 400 chars, 50 words)",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of results (1-20, default: 20)",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 20,
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset (0-9, default: 0)",
                    "default": 0,
                    "minimum": 0,
                    "maximum": 9,
                },
                "country": {
                    "type": "string",
                    "description": "2-character country code (default: US)",
                    "default": "US",
                },
                "search_lang": {
                    "type": "string",
                    "description": "Language code for results (default: en)",
                    "default": "en",
                },
                "safesearch": {
                    "type": "string",
                    "enum": ["off", "moderate", "strict"],
                    "description": "Content filtering level (default: moderate)",
                    "default": "moderate",
                },
                "freshness": {
                    "type": "string",
                    "description": "Date filter (pd=past day, pw=past week, pm=past month, py=past year)",  # noqa: E501
                },
                "result_filter": {
                    "type": "string",
                    "description": "Comma-delimited result types (discussions,faq,infobox,news,videos,web,locations)",  # noqa: E501
                },
            },
            "required": ["q"],
        }

    @property
    def tags(self) -> list[str]:
        """Tool semantic tags."""
        return ["web", "search"]

    @property
    def category(self) -> str | None:
        """Tool category."""
        return "web"

    async def _execute(
        self,
        q: str,
        count: int = 20,
        offset: int = 0,
        country: str = "US",
        search_lang: str = "en",
        safesearch: str = "moderate",
        freshness: str | None = None,
        result_filter: str | None = None,
    ) -> dict[str, Any]:
        """Execute web search using Brave Search API."""
        # Check if API key is configured
        if not settings.brave_api_key:
            raise ValueError(
                "Brave API key not configured. Set BRAVE_API_KEY environment variable.",
            )

        # Build search parameters
        params = BraveSearchParams(
            q=q,
            count=count,
            offset=offset,
            country=country,
            search_lang=search_lang,
            safesearch=safesearch,  # type: ignore
            freshness=freshness,
            result_filter=result_filter,
        )

        # Execute search
        async with BraveSearchClient(api_key=settings.brave_api_key) as client:
            response: BraveSearchResponse = await client.search(query=params)

            # Format results for easier consumption
            results: dict[str, Any] = {
                "query": response.query.model_dump(exclude_none=True),
            }

            # Add web results
            if response.web and response.web.results:
                results["web_results"] = [
                    {
                        "title": r.title,
                        "url": r.url,
                        "description": r.description,
                        "age": r.age,
                    }
                    for r in response.web.results
                ]

            # Add news results
            if response.news and response.news.results:
                results["news_results"] = [
                    {
                        "title": r.title,
                        "url": r.url,
                        "description": r.description,
                        "age": r.age,
                        "breaking": r.breaking,
                    }
                    for r in response.news.results
                ]

            # Add video results
            if response.videos and response.videos.results:
                results["video_results"] = [
                    {
                        "title": r.title,
                        "url": r.url,
                        "description": r.description,
                        "duration": r.duration,
                        "views": r.views,
                        "creator": r.creator,
                    }
                    for r in response.videos.results
                ]

            return results
