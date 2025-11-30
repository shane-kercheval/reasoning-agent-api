"""Web scraper tool for fetching and parsing web pages."""

from datetime import datetime, UTC
from typing import Any
from urllib.parse import urljoin, urlparse
import httpx
from bs4 import BeautifulSoup, NavigableString, Tag
from pydantic import BaseModel, Field
from tools_api.services.base import BaseTool


class LinkReference(BaseModel):
    """A link reference extracted from a web page."""

    id: int = Field(description="Numeric ID for the link (matches [n] markers in text)")
    url: str = Field(description="Absolute URL of the link")
    text: str = Field(description="Link text content")
    external: bool = Field(description="Whether the link points to an external domain")


class NavigationItem(BaseModel):
    """A navigation link extracted from sidebar/nav elements."""

    text: str = Field(description="Link text content")
    url: str = Field(description="URL of the navigation link")
    section: str | None = Field(default=None, description="Section header this link belongs to")


class WebScraperResult(BaseModel):
    """Result from scraping a web page."""

    url: str = Field(description="Original URL that was requested")
    final_url: str = Field(description="Final URL after redirects")
    status: int = Field(description="HTTP status code of the response")
    fetched_at: str = Field(description="ISO timestamp when the page was fetched")
    title: str | None = Field(default=None, description="Page title from <title> tag")
    description: str | None = Field(default=None, description="Page description from meta tags")
    language: str | None = Field(default=None, description="Page language from html lang attribute")  # noqa: E501
    content_type: str | None = Field(default=None, description="Content-Type header value")
    content_length: int = Field(description="Length of extracted text in characters")
    text: str = Field(description="Extracted text content with [n] markers for links")
    references: list[LinkReference] = Field(description="Array of link references mapping IDs to URLs")  # noqa: E501
    navigation: list[NavigationItem] = Field(
        default_factory=list,
        description="Navigation links extracted from sidebars/nav elements",
    )
    raw_html: str | None = Field(default=None, description="Raw HTML content (only if include_html=true)")  # noqa: E501


class WebScraperTool(BaseTool):
    """
    Fetch and parse web pages into structured text with link references.

    This tool provides Lynx-like text rendering with additional structure:
    - Numbered [n] markers for links inline with text
    - References array mapping IDs to URLs
    - Metadata extraction (title, description)

    Limitations:
    - Only parses static HTML (no JavaScript)
    - Respects content-type headers (rejects non-HTML)
    - Subject to timeout and size limits

    Example:
        result = await tool(url="https://example.com")
        print(result.result.text)  # "Content with links [1]..."
        print(result.result.references[0])  # LinkReference(id=1, url="...")
    """

    # Configuration
    DEFAULT_TIMEOUT = 30  # seconds
    MAX_CONTENT_SIZE = 5_000_000  # 5MB
    USER_AGENT = "ReasoningAgent/1.0"

    # Tags to exclude from text extraction
    EXCLUDED_TAGS = frozenset([
        "script", "style", "noscript", "nav", "footer", "header",
        "aside", "form", "button", "input", "select", "textarea",
    ])

    # Classes that indicate navigation/chrome elements (checked with substring match)
    NAVIGATION_CLASSES = frozenset([
        "nav", "navbar", "navigation", "sidebar", "menu", "toc",
        "table-of-contents", "breadcrumb", "related",
    ])

    # IDs that indicate navigation/chrome elements
    NAVIGATION_IDS = frozenset([
        "sidebar", "nav", "navbar", "navigation", "menu", "toc",
        "table-of-contents", "sidebar-content", "navigation-items",
    ])

    # Classes to filter out completely (not navigation, just noise)
    EXCLUDED_CLASSES = frozenset([
        "headerlink",  # Permalink anchors (¶ symbols)
        "anchor-link",
        "permalink",
    ])

    @property
    def name(self) -> str:
        """Tool name."""
        return "web_scraper"

    @property
    def description(self) -> str:
        """Tool description."""
        return (
            "Fetch a web page and extract readable text with numbered link references "
            "(static HTML only, no JavaScript). "
            "Returns structured JSON with text, metadata, and a references array "
            "that maps [n] markers in the text to their URLs."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """Tool parameters JSON Schema."""
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch and parse",
                },
                "include_html": {
                    "type": "boolean",
                    "description": "Include raw HTML in response (default: false)",
                    "default": False,
                },
                "timeout": {
                    "type": "integer",
                    "description": f"Request timeout in seconds (default: {self.DEFAULT_TIMEOUT})",
                    "minimum": 5,
                    "maximum": 60,
                },
            },
            "required": ["url"],
        }

    @property
    def result_model(self) -> type[BaseModel]:
        """Pydantic model for the tool result."""
        return WebScraperResult

    @property
    def tags(self) -> list[str]:
        """Tool semantic tags."""
        return ["web", "scraper", "fetch"]

    @property
    def category(self) -> str | None:
        """Tool category."""
        return "web"

    async def _execute(
        self,
        url: str,
        include_html: bool = False,
        timeout: int | None = None,  # noqa: ASYNC109
    ) -> WebScraperResult:
        """
        Fetch and parse a web page.

        Args:
            url: The URL to fetch
            include_html: Whether to include raw HTML in response
            timeout: Request timeout in seconds

        Returns:
            WebScraperResult with text, references, and metadata
        """
        effective_timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT

        # Fetch the page
        response = await self._fetch_page(url, effective_timeout)

        # Parse HTML
        soup = BeautifulSoup(response.text, "lxml")

        # Extract metadata before modifying DOM
        title, description, language = self._extract_metadata(soup)

        # Extract and remove navigation elements (modifies soup in place)
        navigation = self._extract_navigation(soup, str(response.url))

        # Find main content area
        content_root = self._find_main_content(soup)

        # Build link index from main content only
        link_index = self._build_link_index(content_root, str(response.url))

        # Extract text with [n] markers
        text = self._extract_text_with_markers(content_root, str(response.url), link_index)

        # Build references array
        references = self._build_references(content_root, link_index, str(response.url))

        return WebScraperResult(
            url=url,
            final_url=str(response.url),
            status=response.status_code,
            fetched_at=datetime.now(UTC).isoformat(),
            title=title,
            description=description,
            language=language,
            content_type=response.headers.get("content-type"),
            content_length=len(text),
            text=text,
            references=references,
            navigation=navigation,
            raw_html=response.text if include_html else None,
        )

    async def _fetch_page(self, url: str, timeout: int) -> httpx.Response:  # noqa: ASYNC109
        """
        Fetch a page with proper error handling.

        Args:
            url: URL to fetch
            timeout: Request timeout in seconds

        Returns:
            httpx.Response object

        Raises:
            ValueError: On fetch errors (timeout, connection, HTTP errors)
        """
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout),
                follow_redirects=True,
                headers={"User-Agent": self.USER_AGENT},
            ) as client:
                response = await client.get(url)

                # Check content type
                content_type = response.headers.get("content-type", "")
                if "text/html" not in content_type.lower():
                    raise ValueError(f"Non-HTML content type: {content_type}")

                # Check size (from Content-Length header if available)
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > self.MAX_CONTENT_SIZE:
                    raise ValueError(
                        f"Content too large: {content_length} bytes "
                        f"(max: {self.MAX_CONTENT_SIZE})",
                    )

                # Check rate limiting
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After", "unknown")
                    raise ValueError(f"Rate limited. Retry after: {retry_after}")

                response.raise_for_status()
                return response

        except httpx.TimeoutException:
            raise ValueError(f"Timeout fetching {url} after {timeout}s")
        except httpx.ConnectError as e:
            raise ValueError(f"Connection error for {url}: {e}")
        except httpx.HTTPStatusError as e:
            raise ValueError(f"HTTP error {e.response.status_code} for {url}")

    def _is_navigation_element(self, element: Tag) -> bool:
        """
        Check if an element is a navigation/sidebar element.

        Args:
            element: BeautifulSoup Tag to check

        Returns:
            True if the element appears to be navigation/chrome
        """
        # Check ID
        element_id = element.get("id", "")
        if element_id and element_id.lower() in self.NAVIGATION_IDS:
            return True

        # Check classes (substring match for flexibility)
        classes = element.get("class", [])
        for cls in classes:
            cls_lower = cls.lower()
            for nav_class in self.NAVIGATION_CLASSES:
                if nav_class in cls_lower:
                    return True

        # Check role attribute
        role = element.get("role", "").lower()
        return role == "navigation"

    def _extract_navigation(
        self,
        soup: BeautifulSoup,
        base_url: str,
    ) -> list[NavigationItem]:
        """
        Extract navigation links and remove nav elements from the DOM.

        This method finds navigation containers (sidebars, nav bars, etc.),
        extracts their links as structured data, then removes those elements
        from the soup so they don't appear in main content extraction.

        Uses a while loop to process one element at a time, decomposing
        immediately after extraction. This prevents duplicate extraction
        from nested navigation elements (e.g., sidebar > sidebar-content).

        Args:
            soup: Parsed HTML (modified in place)
            base_url: Base URL for resolving relative links

        Returns:
            List of NavigationItem objects with link text, URL, and section
        """
        nav_items: list[NavigationItem] = []

        # Process navigation elements one at a time, decomposing immediately
        # to prevent duplicate extraction from nested elements
        while True:
            nav_element = self._find_first_navigation_element(soup)
            if nav_element is None:
                break

            # Extract links from this navigation element
            current_section: str | None = None
            for child in nav_element.descendants:
                if not isinstance(child, Tag):
                    continue

                # Track section headers
                if child.name in ["h2", "h3", "h4", "h5", "h6"]:
                    section_text = child.get_text(strip=True)
                    if section_text:
                        current_section = section_text

                # Extract links
                if child.name == "a" and child.get("href"):
                    href = child.get("href", "")
                    is_anchor = href.startswith("#") or href.startswith("javascript:")
                    if href and not is_anchor:
                        link_text = child.get_text(strip=True)
                        if link_text:
                            nav_items.append(NavigationItem(
                                text=link_text,
                                url=urljoin(base_url, href),
                                section=current_section,
                            ))

            # Decompose immediately - removes element AND all its children
            # This prevents nested nav elements from being processed again
            nav_element.decompose()

        return nav_items

    def _find_first_navigation_element(self, soup: BeautifulSoup) -> Tag | None:
        """
        Find the first element in the document that matches navigation patterns.

        Args:
            soup: Parsed HTML

        Returns:
            First matching Tag, or None if no navigation elements found
        """
        for element in soup.find_all(True):
            if isinstance(element, Tag) and self._is_navigation_element(element):
                return element
        return None

    def _find_main_content(self, soup: BeautifulSoup) -> Tag | BeautifulSoup:
        """
        Find the main content area of the page.

        Tries semantic elements first, then falls back to body.

        Args:
            soup: Parsed HTML

        Returns:
            The main content element or soup.body/soup as fallback
        """
        # Try semantic main content elements
        for selector in ["main", "article", "[role='main']"]:
            element = soup.select_one(selector)
            if element:
                return element

        # Try common content class patterns
        content_patterns = ["content", "main-content", "post-content", "entry-content", "article"]
        for element in soup.find_all(["div", "section"]):
            if not isinstance(element, Tag):
                continue
            classes = element.get("class", [])
            element_id = element.get("id", "")

            for pattern in content_patterns:
                if any(pattern in cls.lower() for cls in classes):
                    return element
                if pattern in element_id.lower():
                    return element

        return soup.body or soup

    def _has_excluded_class(self, element: Tag) -> bool:
        """
        Check if an element has a class that should be excluded.

        Args:
            element: BeautifulSoup Tag to check

        Returns:
            True if the element has an excluded class (e.g., headerlink)
        """
        classes = element.get("class", [])
        return any(cls.lower() in self.EXCLUDED_CLASSES for cls in classes)

    def _build_link_index(
        self,
        content_root: Tag | BeautifulSoup,
        base_url: str,
    ) -> dict[str, int]:
        """
        Build a mapping of absolute URLs to numeric IDs.

        Args:
            content_root: Content element to extract links from
            base_url: Base URL for resolving relative links

        Returns:
            Dict mapping absolute URLs to their numeric IDs (1-indexed)
        """
        link_index: dict[str, int] = {}
        next_id = 1

        for a in content_root.find_all("a", href=True):
            # Skip links with excluded classes (e.g., headerlinks with ¶)
            if isinstance(a, Tag) and self._has_excluded_class(a):
                continue

            href = a.get("href", "")
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue

            abs_href = urljoin(base_url, href)
            if abs_href not in link_index:
                link_index[abs_href] = next_id
                next_id += 1

        return link_index

    def _extract_text_with_markers(
        self,
        content_root: Tag | BeautifulSoup,
        base_url: str,
        link_index: dict[str, int],
    ) -> str:
        """
        Extract text from HTML with [n] markers for links.

        Args:
            content_root: Content element to extract text from
            base_url: Base URL for resolving relative links
            link_index: Mapping of URLs to numeric IDs

        Returns:
            Text content with [n] markers for links
        """
        def walk(node: Any) -> list[str]:
            """Recursively walk DOM and extract text with markers."""
            # Handle text nodes
            if isinstance(node, NavigableString):
                text = str(node).strip()
                return [text] if text else []

            # Skip excluded tags
            if hasattr(node, "name") and node.name in self.EXCLUDED_TAGS:
                return []

            # Skip anchor tags with excluded classes (e.g., headerlinks)
            if (
                hasattr(node, "name")
                and node.name == "a"
                and isinstance(node, Tag)
                and self._has_excluded_class(node)
            ):
                return []

            # Process children
            parts: list[str] = []
            for child in node.children:
                parts.extend(walk(child))

            # Add [n] marker after link text
            if hasattr(node, "name") and node.name == "a":
                href = node.get("href", "")
                if href and not href.startswith("#") and not href.startswith("javascript:"):
                    abs_href = urljoin(base_url, href)
                    if abs_href in link_index:
                        parts.append(f"[{link_index[abs_href]}]")

            # Add spacing for block elements
            if hasattr(node, "name") and node.name in [
                "p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6",
                "li", "tr", "td", "th", "blockquote", "pre",
            ]:
                parts.append("\n")

            return parts

        parts = walk(content_root)

        # Join and clean up whitespace
        text = " ".join(parts)
        # Normalize whitespace: collapse multiple spaces/newlines
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        return "\n".join(lines)


    def _build_references(
        self,
        content_root: Tag | BeautifulSoup,
        link_index: dict[str, int],
        base_url: str,
    ) -> list[LinkReference]:
        """
        Build the references array with link details.

        Args:
            content_root: Content element to search for links
            link_index: Mapping of URLs to numeric IDs
            base_url: Base URL for domain comparison

        Returns:
            List of LinkReference objects
        """
        base_domain = urlparse(base_url).netloc
        references: list[LinkReference] = []

        # Sort by ID to maintain consistent ordering
        for href, link_id in sorted(link_index.items(), key=lambda x: x[1]):
            # Find the first <a> tag with this href (excluding headerlinks)
            a_tag = None
            for a in content_root.find_all("a", href=True):
                if isinstance(a, Tag) and self._has_excluded_class(a):
                    continue
                resolved = urljoin(base_url, a.get("href", ""))
                if resolved == href:
                    a_tag = a
                    break

            link_text = a_tag.get_text(strip=True) if a_tag else ""
            link_domain = urlparse(href).netloc

            references.append(LinkReference(
                id=link_id,
                url=href,
                text=link_text,
                external=link_domain != base_domain,
            ))

        return references

    def _extract_metadata(
        self,
        soup: BeautifulSoup,
    ) -> tuple[str | None, str | None, str | None]:
        """
        Extract page metadata.

        Args:
            soup: Parsed HTML

        Returns:
            Tuple of (title, description, language)
        """
        # Get title
        title = None
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Get description from meta tag
        description = self._get_meta_content(soup, "description")
        if not description:
            # Try OpenGraph description
            description = self._get_meta_property(soup, "og:description")

        # Get language
        language = None
        if soup.html:
            language = soup.html.get("lang")

        return title, description, language

    def _get_meta_content(self, soup: BeautifulSoup, name: str) -> str | None:
        """Get content from a meta tag by name attribute."""
        meta = soup.find("meta", attrs={"name": name})
        if meta:
            return meta.get("content")
        return None

    def _get_meta_property(self, soup: BeautifulSoup, property_name: str) -> str | None:
        """Get content from a meta tag by property attribute (OpenGraph)."""
        meta = soup.find("meta", attrs={"property": property_name})
        if meta:
            return meta.get("content")
        return None
