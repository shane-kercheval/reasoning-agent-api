"""Web search tool."""

from typing import Any

from pydantic import BaseModel, Field

from tools_api.clients.brave_search import (
    BraveSearchClient,
    BraveSearchParams,
    BraveSearchResponse,
)
from tools_api.config import settings
from tools_api.services.base import BaseTool


class WebResult(BaseModel):
    """A single web search result."""

    title: str = Field(description="Title of the web page")
    url: str = Field(description="URL of the web page")
    description: str | None = Field(default=None, description="Description or snippet of the web page")  # noqa: E501
    age: str | None = Field(default=None, description="Age of the result (e.g., '2 hours ago')")


class NewsResult(BaseModel):
    """A single news search result."""

    title: str = Field(description="Title of the news article")
    url: str = Field(description="URL of the news article")
    description: str | None = Field(default=None, description="Description or snippet of the article")  # noqa: E501
    age: str | None = Field(default=None, description="Age of the article")
    breaking: bool | None = Field(default=None, description="Whether this is breaking news")


class VideoResult(BaseModel):
    """A single video search result."""

    title: str = Field(description="Title of the video")
    url: str = Field(description="URL of the video")
    description: str | None = Field(default=None, description="Description of the video")
    duration: str | None = Field(default=None, description="Duration of the video")
    views: str | None = Field(default=None, description="Number of views")
    creator: str | None = Field(default=None, description="Creator or channel name")


class WebSearchResult(BaseModel):
    """Result from web search."""

    query: dict[str, Any] = Field(description="The query information including original query and alterations")  # noqa: E501
    web_results: list[WebResult] | None = Field(default=None, description="List of web search results")  # noqa: E501
    news_results: list[NewsResult] | None = Field(default=None, description="List of news results")
    video_results: list[VideoResult] | None = Field(default=None, description="List of video results")  # noqa: E501


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
                    "description": "2-character country code, e.g., US, UK, FR (default: US)",
                    "default": "US",
                },
                "search_lang": {
                    "type": "string",
                    "description": "Language code for results, e.g., en, es, fr (default: en)",
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
    def result_model(self) -> type[BaseModel]:
        """Pydantic model for the tool result."""
        return WebSearchResult

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
    ) -> WebSearchResult:
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

            # Build web results
            web_results: list[WebResult] | None = None
            if response.web and response.web.results:
                web_results = [
                    WebResult(
                        title=r.title,
                        url=r.url,
                        description=r.description,
                        age=r.age,
                    )
                    for r in response.web.results
                ]

            # Build news results
            news_results: list[NewsResult] | None = None
            if response.news and response.news.results:
                news_results = [
                    NewsResult(
                        title=r.title,
                        url=r.url,
                        description=r.description,
                        age=r.age,
                        breaking=r.breaking,
                    )
                    for r in response.news.results
                ]

            # Build video results
            video_results: list[VideoResult] | None = None
            if response.videos and response.videos.results:
                video_results = [
                    VideoResult(
                        title=r.title,
                        url=r.url,
                        description=r.description,
                        duration=r.duration,
                        views=r.views,
                        creator=r.creator,
                    )
                    for r in response.videos.results
                ]

            return WebSearchResult(
                query=response.query.model_dump(exclude_none=True),
                web_results=web_results,
                news_results=news_results,
                video_results=video_results,
            )
