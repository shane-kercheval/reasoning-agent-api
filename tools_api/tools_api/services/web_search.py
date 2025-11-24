"""
Brave Search API client for web search functionality.

This module provides a type-safe, async client for the Brave Search API,
following the codebase patterns for dependency injection and error handling.

Documentation: https://api-dashboard.search.brave.com/app/documentation/web-search
"""

import asyncio
import os
from enum import Enum
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Custom Exceptions
# ============================================================================


class BraveSearchError(Exception):
    """Base exception for all Brave Search API errors."""

    pass


class BraveSearchAuthError(BraveSearchError):
    """Authentication failed - invalid or missing API key."""

    pass


class BraveSearchRateLimitError(BraveSearchError):
    """Rate limit exceeded."""

    def __init__(
        self,
        message: str,
        limit: str | None = None,
        remaining: str | None = None,
        reset: str | None = None,
    ) -> None:
        super().__init__(message)
        self.limit = limit
        self.remaining = remaining
        self.reset = reset


class BraveSearchAPIError(BraveSearchError):
    """General API error with status code."""

    def __init__(self, message: str, status_code: int, response_body: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


# ============================================================================
# Request Models
# ============================================================================


class SafeSearch(str, Enum):
    """Content filtering options."""

    OFF = "off"
    MODERATE = "moderate"
    STRICT = "strict"


class Freshness(str, Enum):
    """Date range filters for search results."""

    PAST_DAY = "pd"
    PAST_WEEK = "pw"
    PAST_MONTH = "pm"
    PAST_YEAR = "py"


class Units(str, Enum):
    """Measurement system preference."""

    METRIC = "metric"
    IMPERIAL = "imperial"


class BraveSearchParams(BaseModel):
    """
    Parameters for Brave Search API web search requests.

    See: https://api-dashboard.search.brave.com/app/documentation/web-search/query
    """

    q: str = Field(..., description="Search query (max 400 chars, 50 words)")
    country: str = Field(
        default="US", description="2-character country code (ISO 3166-1 alpha-2)",
    )
    search_lang: str = Field(default="en", description="Language code for results")
    ui_lang: str = Field(
        default="en-US", description="UI language (format: <language>-<country>)",
    )
    count: int = Field(
        default=20, ge=1, le=20, description="Number of results (max 20)",
    )
    offset: int = Field(
        default=0, ge=0, le=9, description="Zero-based pagination offset (max 9)",
    )
    safesearch: SafeSearch = Field(
        default=SafeSearch.MODERATE, description="Content filtering level",
    )
    freshness: str | None = Field(
        default=None,
        description="Date filter (pd/pw/pm/py or YYYY-MM-DDtoYYYY-MM-DD)",
    )
    text_decorations: bool = Field(
        default=True, description="Include decoration markers",
    )
    spellcheck: bool = Field(default=True, description="Enable query spellcheck")
    result_filter: str | None = Field(
        default=None,
        description=(
            "Comma-delimited result types "
            "(discussions,faq,infobox,news,videos,web,locations)"
        ),
    )
    goggles: list[str] | None = Field(
        default=None, description="Custom re-ranking URLs or definitions",
    )
    units: Units | None = Field(
        default=None, description="Measurement system preference",
    )
    extra_snippets: bool | None = Field(
        default=None, description="Include up to 5 additional excerpts",
    )
    summary: bool | None = Field(
        default=None, description="Enable AI summary generation",
    )
    operators: bool = Field(
        default=True, description="Apply search operator functionality",
    )

    @field_validator("q")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query length constraints."""
        if len(v) > 400:
            raise ValueError("Query must not exceed 400 characters")
        if len(v.split()) > 50:
            raise ValueError("Query must not exceed 50 words")
        return v

    def to_query_params(self) -> dict[str, Any]:
        """Convert to dict for URL query parameters, excluding None values."""
        return {
            k: v.value if isinstance(v, Enum) else v
            for k, v in self.model_dump(exclude_none=True).items()
            if v is not None
        }


# ============================================================================
# Response Models
# ============================================================================


class MetaUrl(BaseModel):
    """URL metadata with protocol, domain, and display path."""

    scheme: str | None = None
    netloc: str | None = None
    hostname: str | None = None
    favicon: str | None = None
    path: str | None = None


class Thumbnail(BaseModel):
    """Image thumbnail with source URL."""

    src: str
    original: str | None = None


class Rating(BaseModel):
    """Rating information with value and review count."""

    value: float = Field(..., ge=0, le=5)
    count: int | None = None
    is_tripadvisor: bool | None = None


class SearchResult(BaseModel):
    """Individual web search result."""

    type: Literal["search_result"] = "search_result"
    title: str
    url: str
    description: str | None = None
    meta_url: MetaUrl | None = None
    thumbnail: Thumbnail | None = None
    language: str | None = None
    age: str | None = None
    family_friendly: bool | None = None
    extra_snippets: list[str] | None = None


class NewsResult(BaseModel):
    """News article result."""

    type: Literal["news_result"] = "news_result"
    title: str
    url: str
    description: str | None = None
    meta_url: MetaUrl | None = None
    thumbnail: Thumbnail | None = None
    age: str | None = None
    breaking: bool | None = None
    is_live: bool | None = None


class VideoResult(BaseModel):
    """Video content result."""

    type: Literal["video_result"] = "video_result"
    title: str
    url: str
    description: str | None = None
    meta_url: MetaUrl | None = None
    thumbnail: Thumbnail | None = None
    duration: str | None = None
    views: int | None = None
    creator: str | None = None


class Query(BaseModel):
    """Query metadata from the response."""

    original: str
    show_strict_warning: bool | None = None
    altered: str | None = None
    safesearch: bool | None = None
    is_navigational: bool | None = None
    is_news_breaking: bool | None = None
    spellcheck_off: bool | None = None
    country: str | None = None
    bad_results: bool | None = None
    should_fallback: bool | None = None
    postal_code: str | None = None
    city: str | None = None
    header_country: str | None = None
    more_results_available: bool | None = None
    state: str | None = None


class WebResults(BaseModel):
    """Container for web search results."""

    type: Literal["search"] = "search"
    results: list[SearchResult] = Field(default_factory=list)
    family_friendly: bool | None = None


class NewsResults(BaseModel):
    """Container for news results."""

    type: Literal["news"] = "news"
    results: list[NewsResult] = Field(default_factory=list)


class VideoResults(BaseModel):
    """Container for video results."""

    type: Literal["videos"] = "videos"
    results: list[VideoResult] = Field(default_factory=list)


class BraveSearchResponse(BaseModel):
    """
    Complete response from Brave Search API.

    See: https://api-dashboard.search.brave.com/app/documentation/web-search/responses
    """

    type: Literal["search"] = "search"
    query: Query
    web: WebResults | None = None
    news: NewsResults | None = None
    videos: VideoResults | None = None
    # Additional fields like discussions, faq, infobox, locations can be added as needed


class RateLimitInfo(BaseModel):
    """Rate limit information from response headers."""

    limit: str | None = None
    policy: str | None = None
    remaining: str | None = None
    reset: str | None = None


# ============================================================================
# Brave Search Client
# ============================================================================


class BraveSearchClient:
    """
    Async client for Brave Search API with automatic rate limit handling.

    Automatically retries requests that hit the per-second rate limit (1 req/sec).
    Monthly rate limits are raised immediately without retry.

    Example usage:
        client = BraveSearchClient(api_key="your-key")
        results = await client.search("Python programming")
        print(f"Found {len(results.web.results)} results")

    Environment variables:
        BRAVE_SEARCH_API: API subscription token (if not passed to constructor)
    """

    BASE_URL = "https://api.search.brave.com/res/v1"

    def __init__(
        self,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
        max_retries_per_second_limit: int = 3,
    ) -> None:
        """
        Initialize Brave Search client.

        Args:
            api_key: Brave Search API subscription token (falls back to BRAVE_SEARCH_API env var)
            http_client: Optional custom httpx client (useful for testing)
            timeout: Request timeout in seconds (default: 30.0)
            max_retries_per_second_limit: Max retries for per-second rate limit (default: 3)

        Raises:
            BraveSearchAuthError: If no API key is provided or found in environment
        """
        self.api_key = api_key or os.getenv("BRAVE_SEARCH_API")
        if not self.api_key:
            raise BraveSearchAuthError(
                "No API key provided. Set BRAVE_SEARCH_API environment variable "
                "or pass api_key parameter.",
            )

        self._client = http_client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = http_client is None
        self._last_rate_limit: RateLimitInfo | None = None
        self._max_retries = max_retries_per_second_limit

    async def __aenter__(self) -> "BraveSearchClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client if owned by this instance."""
        if self._owns_client and self._client:
            await self._client.aclose()

    @property
    def last_rate_limit(self) -> RateLimitInfo | None:
        """Get rate limit information from the last successful request."""
        return self._last_rate_limit

    def _build_headers(
        self,
        user_agent: str | None = None,
        location: dict[str, str] | None = None,
        cache_control: str | None = None,
    ) -> dict[str, str]:
        """
        Build request headers.

        Args:
            user_agent: Custom User-Agent string
            location: Location headers (lat, long, timezone, city, state, country, postal_code)
            cache_control: Cache control header (e.g., "no-cache")

        Returns:
            Dictionary of HTTP headers
        """
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }

        if user_agent:
            headers["User-Agent"] = user_agent

        if cache_control:
            headers["Cache-Control"] = cache_control

        # Add optional location headers
        if location:
            location_mapping = {
                "lat": "X-Loc-Lat",
                "long": "X-Loc-Long",
                "timezone": "X-Loc-Timezone",
                "city": "X-Loc-City",
                "state": "X-Loc-State",
                "state_name": "X-Loc-State-Name",
                "country": "X-Loc-Country",
                "postal_code": "X-Loc-Postal-Code",
            }
            for key, header_name in location_mapping.items():
                if key in location:
                    headers[header_name] = str(location[key])

        return headers

    def _parse_rate_limit_headers(self, headers: httpx.Headers) -> RateLimitInfo:
        """Extract rate limit information from response headers."""
        return RateLimitInfo(
            limit=headers.get("X-RateLimit-Limit"),
            policy=headers.get("X-RateLimit-Policy"),
            remaining=headers.get("X-RateLimit-Remaining"),
            reset=headers.get("X-RateLimit-Reset"),
        )

    async def _make_request(
        self,
        url: str,
        query_params: dict[str, Any],
        headers: dict[str, str],
    ) -> BraveSearchResponse:
        """
        Make HTTP request to Brave Search API (internal method).

        Raises:
            BraveSearchAuthError: Authentication failed
            BraveSearchRateLimitError: Rate limit exceeded (caller should handle retries)
            BraveSearchAPIError: Other API errors
        """
        try:
            response = await self._client.get(url, params=query_params, headers=headers)

            # Store rate limit info
            self._last_rate_limit = self._parse_rate_limit_headers(response.headers)

            # Handle error responses
            if response.status_code == 401:
                raise BraveSearchAuthError(
                    "Authentication failed. Check your API key.",
                )
            if response.status_code == 429:
                rate_limit = self._last_rate_limit
                # Parse reset times to provide clearer error message
                reset_times = rate_limit.reset.split(", ") if rate_limit.reset else ["0", "0"]
                per_second_reset = int(reset_times[0])
                per_month_reset = int(reset_times[1]) if len(reset_times) > 1 else 0

                # Determine which limit was hit based on reset time
                if per_second_reset <= 1:
                    msg = f"Per-second rate limit (1 req/sec). Retry in {per_second_reset}s."
                else:
                    days = per_month_reset // 86400
                    msg = f"Monthly rate limit. Resets in {per_month_reset}s ({days} days)."

                raise BraveSearchRateLimitError(
                    msg,
                    limit=rate_limit.limit,
                    remaining=rate_limit.remaining,
                    reset=rate_limit.reset,
                )
            if response.status_code != 200:
                raise BraveSearchAPIError(
                    f"API request failed with status {response.status_code}",
                    status_code=response.status_code,
                    response_body=response.text,
                )

            # Parse and return successful response
            response.raise_for_status()
            data = response.json()
            return BraveSearchResponse(**data)

        except httpx.HTTPError as e:
            if isinstance(e, (BraveSearchError,)):
                raise
            raise BraveSearchAPIError(
                f"HTTP request failed: {e!s}", status_code=0, response_body="",
            ) from e

    async def search(
        self,
        query: str | BraveSearchParams,
        user_agent: str | None = None,
        location: dict[str, str] | None = None,
        cache_control: str | None = None,
    ) -> BraveSearchResponse:
        """
        Perform a web search using Brave Search API.

        Automatically retries on per-second rate limit errors (1 req/sec).
        Monthly rate limit errors are raised immediately.

        Args:
            query: Search query string or BraveSearchParams object for advanced options
            user_agent: Optional User-Agent header for tailored results
            location: Optional location data
                (lat, long, timezone, city, state, country, postal_code)
            cache_control: Optional cache control (e.g., "no-cache")

        Returns:
            BraveSearchResponse with search results

        Raises:
            BraveSearchAuthError: Authentication failed
            BraveSearchRateLimitError: Monthly rate limit exceeded
                (per-second limit retried automatically)
            BraveSearchAPIError: Other API errors
        """
        # Convert string query to params object
        params = BraveSearchParams(q=query) if isinstance(query, str) else query

        headers = self._build_headers(
            user_agent=user_agent, location=location, cache_control=cache_control,
        )

        url = f"{self.BASE_URL}/web/search"
        query_params = params.to_query_params()

        # Retry loop for per-second rate limits
        for attempt in range(self._max_retries):
            try:
                return await self._make_request(url, query_params, headers)
            except BraveSearchRateLimitError as e:
                # Parse reset times to determine which limit was hit
                reset_times = e.reset.split(", ") if e.reset else ["0", "0"]
                per_second_reset = int(reset_times[0])

                # Only retry for per-second rate limits (reset <= 1 second)
                if per_second_reset <= 1 and attempt < self._max_retries - 1:
                    # Wait for the reset time plus a small buffer
                    wait_time = max(per_second_reset, 1) + 0.1
                    await asyncio.sleep(wait_time)
                    continue

                # Re-raise for monthly limits or if max retries exceeded
                raise

        # This should never be reached due to the loop logic, but satisfy type checker
        msg = "Search failed after maximum retries"
        raise BraveSearchAPIError(msg, status_code=0, response_body="")
