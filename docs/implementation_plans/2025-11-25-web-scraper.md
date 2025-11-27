# Implementation Plan: Web Scraper Tool

## Overview

**Goal**: Create a web scraping tool using BeautifulSoup that provides structured, agent-friendly output similar to Lynx's `-dump` but with richer metadata and programmatic link references.

**Context**: The current tools-api provides web search via the Brave Search API, but lacks the ability to fetch and parse actual web page content. This tool will enable the reasoning agent to:
- Fetch and parse HTML from URLs
- Extract readable text content with numbered link references (like Lynx)
- Return structured JSON with metadata, references, and optional chunking
- Support agents in navigating and extracting information from web pages

**Why BeautifulSoup**:
- Pure Python, no external dependencies like Lynx
- Full control over output format
- Can extract rich metadata (titles, descriptions, links with context)
- Works inside Docker without additional CLI tools

---

## Documentation References

Before implementing, read these resources:

1. **BeautifulSoup Documentation**: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
   - Focus on: NavigableString, find_all(), get_text(), recursive navigation

2. **httpx Documentation**: https://www.python-httpx.org/
   - Focus on: AsyncClient, redirects, timeouts, headers

3. **Existing Tool Patterns**: Review these files in the codebase:
   - `tools_api/tools_api/services/base.py` - BaseTool abstract class
   - `tools_api/tools_api/services/tools/web_search_tool.py` - BraveSearchTool as example
   - `tools_api/tools_api/services/tools/filesystem.py` - ReadTextFileTool for structured responses

---

## Milestone 1: Core Web Fetcher and Text Extractor

### Goal
Create the basic web scraper that fetches HTML and extracts clean text with numbered link references.

### Success Criteria
- [ ] `WebScraperTool` class following `BaseTool` pattern
- [ ] Fetches HTML from URL using httpx
- [ ] Extracts readable text (excludes script, style, nav, footer)
- [ ] Inserts `[n]` markers for links inline with text
- [ ] Returns structured JSON with text and references array
- [ ] Handles redirects and follows them
- [ ] Proper error handling (timeouts, 404s, connection errors)
- [ ] Unit tests with mocked HTTP responses

### Key Changes

**1. Create the tool file:**
```python
# tools_api/tools_api/services/tools/web_scraper.py
from typing import Any
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup, NavigableString

from tools_api.services.base import BaseTool


class WebScraperTool(BaseTool):
    """Fetch and parse web pages into structured text with link references."""

    @property
    def name(self) -> str:
        return "web_scraper"

    @property
    def description(self) -> str:
        return (
            "Fetch a web page and extract readable text with numbered link references. "
            "Returns structured JSON with text, metadata, and a references array "
            "that maps [n] markers in the text to their URLs."
        )

    @property
    def parameters(self) -> dict[str, Any]:
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
            },
            "required": ["url"],
        }

    @property
    def tags(self) -> list[str]:
        return ["web", "scraper", "fetch"]

    async def _execute(
        self,
        url: str,
        include_html: bool = False,
    ) -> dict[str, Any]:
        # Implementation here - see Testing Strategy for expected behavior
        ...
```

**2. Key implementation patterns:**

```python
# Building link index with numeric IDs
link_index: dict[str, int] = {}
next_id = 1

for a in soup.find_all("a", href=True):
    abs_href = urljoin(url, a["href"])
    if abs_href not in link_index:
        link_index[abs_href] = next_id
        next_id += 1

# Walking DOM and inserting [n] markers
def walk(node) -> list[str]:
    if isinstance(node, NavigableString):
        text = str(node).strip()
        return [text] if text else []

    if node.name in ["script", "style", "noscript", "nav", "footer", "header"]:
        return []

    parts = []
    for child in node.children:
        parts.extend(walk(child))

    if node.name == "a" and node.get("href"):
        abs_href = urljoin(url, node["href"])
        if abs_href in link_index:
            parts.append(f"[{link_index[abs_href]}]")

    return parts
```

**3. Expected response structure:**
```json
{
  "url": "https://example.com/page",
  "final_url": "https://example.com/page",
  "status": 200,
  "metadata": {
    "title": "Page Title",
    "description": "Meta description if available",
    "content_type": "text/html; charset=UTF-8"
  },
  "text": "Page content with links [1] inline and more text [2]...",
  "references": [
    {"id": 1, "url": "https://example.com/link1", "text": "links"},
    {"id": 2, "url": "https://example.com/link2", "text": "more text"}
  ],
  "raw_html": "<!doctype html>..."  // Only if include_html=true
}
```

**4. Register the tool:**
```python
# tools_api/tools_api/main.py
from tools_api.services.tools.web_scraper import WebScraperTool

# In register_tools()
ToolRegistry.register(WebScraperTool())
```

### Testing Strategy

**What to test:**

1. **Basic fetch and parse:**
   - Simple HTML returns expected text
   - Links are numbered correctly
   - References array matches [n] markers in text

2. **Link handling:**
   - Relative URLs resolved to absolute
   - Duplicate URLs get same ID
   - Links without href are ignored

3. **Content filtering:**
   - Script/style tags excluded
   - Nav/footer optionally excluded (discuss with user)
   - Empty text nodes ignored

4. **Error conditions:**
   - Timeout handling
   - 404/500 responses
   - Connection errors
   - Invalid URLs

5. **Edge cases:**
   - Empty page
   - Page with no links
   - Non-HTML content type
   - Very large pages (consider limits)

**Example test:**
```python
# tools_api/tests/unit_tests/test_web_scraper.py
import pytest
from unittest.mock import AsyncMock, patch

from tools_api.services.tools.web_scraper import WebScraperTool


@pytest.fixture
def simple_html() -> str:
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Test Page</title></head>
    <body>
        <p>Hello <a href="/link1">world</a>!</p>
        <p>Visit <a href="https://example.com">example</a>.</p>
    </body>
    </html>
    """


@pytest.mark.asyncio
async def test_web_scraper_basic(simple_html: str) -> None:
    """Test basic HTML parsing with link extraction."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = simple_html
        mock_response.url = "https://test.com/page"
        mock_response.headers = {"content-type": "text/html"}

        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

        tool = WebScraperTool()
        result = await tool(url="https://test.com/page")

        assert result.success is True
        assert "[1]" in result.result["text"]
        assert "[2]" in result.result["text"]
        assert len(result.result["references"]) == 2
        assert result.result["references"][0]["url"] == "https://test.com/link1"


@pytest.mark.asyncio
async def test_web_scraper_timeout() -> None:
    """Test timeout handling."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get.side_effect = httpx.TimeoutException("timeout")

        tool = WebScraperTool()
        result = await tool(url="https://test.com/slow")

        assert result.success is False
        assert "timeout" in result.error.lower()
```

### Dependencies
- httpx (already in project)
- beautifulsoup4 (needs to be added to pyproject.toml)
- lxml parser (recommended for BeautifulSoup, add to dependencies)

### Risk Factors
- **Large pages**: May need content limits to avoid memory issues
- **JavaScript-rendered content**: BeautifulSoup only sees static HTML (document this limitation)
- **Encoding issues**: Need to handle various character encodings
- **Rate limiting by sites**: May want to add delay options in future

---

## Milestone 2: Enhanced Metadata and Content Structure

### Goal
Add richer metadata extraction, heading outline, and optional content chunking for better agent usability.

### Success Criteria
- [ ] Extract `<meta>` description and other metadata
- [ ] Build heading outline (h1-h6) for navigation
- [ ] Add content length and quality metrics
- [ ] Optional chunking for large pages
- [ ] Tests for metadata extraction

### Key Changes

**1. Enhanced metadata extraction:**
```python
def extract_metadata(soup: BeautifulSoup, response: httpx.Response) -> dict[str, Any]:
    return {
        "title": soup.title.string.strip() if soup.title else None,
        "description": _get_meta_content(soup, "description"),
        "language": soup.html.get("lang") if soup.html else None,
        "content_type": response.headers.get("content-type"),
        "content_length": len(response.text),
    }

def _get_meta_content(soup: BeautifulSoup, name: str) -> str | None:
    meta = soup.find("meta", attrs={"name": name})
    if meta:
        return meta.get("content")
    # Also check og: tags
    og_meta = soup.find("meta", attrs={"property": f"og:{name}"})
    if og_meta:
        return og_meta.get("content")
    return None
```

**2. Heading outline:**
```python
def build_outline(soup: BeautifulSoup) -> list[dict[str, Any]]:
    outline = []
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        level = int(heading.name[1])
        text = heading.get_text(strip=True)
        if text:
            outline.append({"level": level, "text": text})
    return outline
```

**3. Quality metrics:**
```python
def compute_quality_metrics(text: str, references: list) -> dict[str, Any]:
    return {
        "text_length": len(text),
        "word_count": len(text.split()),
        "link_count": len(references),
        "likely_article": _is_likely_article(text),
    }

def _is_likely_article(text: str) -> bool:
    # Simple heuristic: articles typically have >500 chars
    return len(text) > 500
```

**4. Optional chunking:**
```python
def chunk_text(
    text: str,
    references: list[dict],
    chunk_size: int = 1000,
) -> list[dict[str, Any]]:
    """Split text into chunks for embedding or retrieval."""
    chunks = []
    words = text.split()

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)

        # Find which references appear in this chunk
        chunk_refs = [
            ref["id"] for ref in references
            if f"[{ref['id']}]" in chunk_text
        ]

        chunks.append({
            "id": f"chunk-{len(chunks)}",
            "text": chunk_text,
            "references": chunk_refs,
            "start_word": i,
            "end_word": min(i + chunk_size, len(words)),
        })

    return chunks
```

**5. Updated parameters:**
```python
@property
def parameters(self) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch and parse",
            },
            "include_html": {
                "type": "boolean",
                "description": "Include raw HTML in response",
                "default": False,
            },
            "include_chunks": {
                "type": "boolean",
                "description": "Split content into chunks for retrieval",
                "default": False,
            },
            "chunk_size": {
                "type": "integer",
                "description": "Words per chunk (default: 1000)",
                "default": 1000,
            },
        },
        "required": ["url"],
    }
```

**6. Updated response structure:**
```json
{
  "url": "https://example.com/article",
  "final_url": "https://example.com/article",
  "status": 200,
  "fetched_at": "2025-11-25T12:34:56Z",
  "metadata": {
    "title": "Article Title",
    "description": "Article summary from meta tag",
    "language": "en",
    "content_type": "text/html; charset=UTF-8",
    "content_length": 15234
  },
  "text": "Article content with links [1]...",
  "references": [
    {
      "id": 1,
      "url": "https://example.com/source",
      "text": "links",
      "context": "...with links to sources...",
      "external": true
    }
  ],
  "outline": [
    {"level": 1, "text": "Main Heading"},
    {"level": 2, "text": "Section One"}
  ],
  "quality": {
    "text_length": 5432,
    "word_count": 892,
    "link_count": 15,
    "likely_article": true
  },
  "chunks": [  // Only if include_chunks=true
    {
      "id": "chunk-0",
      "text": "First 1000 words...",
      "references": [1, 2],
      "start_word": 0,
      "end_word": 1000
    }
  ]
}
```

### Testing Strategy

**What to test:**

1. **Metadata extraction:**
   - Title from `<title>` tag
   - Description from meta tag
   - OpenGraph fallbacks
   - Missing metadata handled gracefully

2. **Outline building:**
   - All heading levels captured
   - Empty headings excluded
   - Correct level numbers

3. **Chunking:**
   - Chunks are correct size
   - References correctly associated with chunks
   - Edge cases (small pages, no chunks needed)

4. **Quality metrics:**
   - Accurate counts
   - Article detection heuristic

### Dependencies
- Milestone 1 complete

### Risk Factors
- **Chunking strategy**: May need refinement based on agent usage patterns
- **Metadata inconsistency**: Different sites structure metadata differently

---

## Milestone 3: Enhanced Link Context and Navigation

### Goal
Improve link references with surrounding context, same-domain detection, and better navigation support.

### Success Criteria
- [ ] Links include surrounding text context
- [ ] `external` flag for same-domain vs external links
- [ ] Link type detection (a, img, etc.)
- [ ] Context snippet around each link reference
- [ ] Tests for context extraction

### Key Changes

**1. Enhanced reference structure:**
```python
def build_references(
    soup: BeautifulSoup,
    link_index: dict[str, int],
    base_url: str,
) -> list[dict[str, Any]]:
    from urllib.parse import urlparse

    base_domain = urlparse(base_url).netloc
    references = []

    for href, link_id in sorted(link_index.items(), key=lambda x: x[1]):
        # Find the <a> tag for this href
        a_tag = soup.find("a", href=lambda h: urljoin(base_url, h or "") == href)

        link_text = a_tag.get_text(strip=True) if a_tag else ""
        context = _get_link_context(a_tag) if a_tag else ""
        link_domain = urlparse(href).netloc

        references.append({
            "id": link_id,
            "url": href,
            "text": link_text,
            "context": context,
            "external": link_domain != base_domain,
            "type": "a",
        })

    return references


def _get_link_context(a_tag, context_chars: int = 100) -> str:
    """Get surrounding text context for a link."""
    # Get parent paragraph or container
    parent = a_tag.find_parent(["p", "div", "li", "td"])
    if parent:
        full_text = parent.get_text(strip=True)
        # Find link text position and extract surrounding context
        link_text = a_tag.get_text(strip=True)
        pos = full_text.find(link_text)
        if pos != -1:
            start = max(0, pos - context_chars // 2)
            end = min(len(full_text), pos + len(link_text) + context_chars // 2)
            return full_text[start:end]
    return ""
```

**2. Support for non-anchor links (images, etc.):**
```python
def extract_all_links(soup: BeautifulSoup, base_url: str) -> dict[str, int]:
    """Extract all linkable resources, not just <a> tags."""
    link_index: dict[str, int] = {}
    next_id = 1

    # Standard anchor links
    for a in soup.find_all("a", href=True):
        abs_href = urljoin(base_url, a["href"])
        if abs_href not in link_index:
            link_index[abs_href] = next_id
            next_id += 1

    # Images (optional - discuss with user if needed)
    # for img in soup.find_all("img", src=True):
    #     abs_src = urljoin(base_url, img["src"])
    #     if abs_src not in link_index:
    #         link_index[abs_src] = next_id
    #         next_id += 1

    return link_index
```

### Testing Strategy

**What to test:**

1. **Context extraction:**
   - Context captured from surrounding text
   - Handles various parent containers (p, div, li)
   - Graceful handling when no parent

2. **External detection:**
   - Same domain links marked internal
   - Different domain links marked external
   - Handles subdomains appropriately

3. **Edge cases:**
   - Links without text
   - Very long context (should truncate)
   - Nested link structures

### Dependencies
- Milestone 2 complete

### Risk Factors
- **Context extraction complexity**: May need tuning for different page structures
- **Image links**: Discuss with user if image extraction is needed

---

## Milestone 4: Error Handling, Limits, and Production Hardening

### Goal
Add robust error handling, content size limits, and production-ready features.

### Success Criteria
- [ ] Configurable timeout and size limits
- [ ] User-agent configuration
- [ ] Graceful handling of non-HTML content
- [ ] Rate limiting awareness (respect Retry-After headers)
- [ ] Comprehensive error messages
- [ ] Integration tests with real URLs (optional, can skip in CI)

### Key Changes

**1. Configuration via settings:**
```python
# tools_api/tools_api/config.py - Add to Settings class
class Settings(BaseSettings):
    # ... existing fields ...

    # Web scraper configuration
    web_scraper_timeout: int = 30  # seconds
    web_scraper_max_size: int = 5_000_000  # 5MB
    web_scraper_user_agent: str = "ReasoningAgent/1.0 (compatible; +https://github.com/yourrepo)"
```

**2. Robust fetching:**
```python
async def fetch_page(self, url: str) -> tuple[httpx.Response, str | None]:
    """Fetch page with proper error handling."""
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(settings.web_scraper_timeout),
            follow_redirects=True,
            headers={"User-Agent": settings.web_scraper_user_agent},
        ) as client:
            response = await client.get(url)

            # Check content type
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type:
                return response, f"Non-HTML content type: {content_type}"

            # Check size
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > settings.web_scraper_max_size:
                return response, f"Content too large: {content_length} bytes"

            # Check rate limiting
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "unknown")
                return response, f"Rate limited. Retry after: {retry_after}"

            response.raise_for_status()
            return response, None

    except httpx.TimeoutException:
        raise ValueError(f"Timeout fetching {url} after {settings.web_scraper_timeout}s")
    except httpx.ConnectError as e:
        raise ValueError(f"Connection error for {url}: {e}")
    except httpx.HTTPStatusError as e:
        raise ValueError(f"HTTP error {e.response.status_code} for {url}")
```

**3. Updated parameters with limits:**
```python
@property
def parameters(self) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch and parse",
            },
            "include_html": {
                "type": "boolean",
                "description": "Include raw HTML in response",
                "default": False,
            },
            "include_chunks": {
                "type": "boolean",
                "description": "Split content into chunks",
                "default": False,
            },
            "chunk_size": {
                "type": "integer",
                "description": "Words per chunk (default: 1000)",
                "default": 1000,
                "minimum": 100,
                "maximum": 10000,
            },
            "timeout": {
                "type": "integer",
                "description": f"Request timeout in seconds (default: {settings.web_scraper_timeout})",
                "default": None,  # Uses settings default
                "minimum": 5,
                "maximum": 60,
            },
        },
        "required": ["url"],
    }
```

### Testing Strategy

**What to test:**

1. **Error handling:**
   - Timeout behavior
   - 404/500 error messages
   - Connection refused
   - Invalid URLs

2. **Limits:**
   - Large page handling (mock or skip)
   - Non-HTML content rejection
   - Rate limit detection

3. **Configuration:**
   - Settings override defaults
   - Timeout parameter works

4. **Integration tests (optional):**
   ```python
   @pytest.mark.integration
   @pytest.mark.asyncio
   async def test_web_scraper_real_url() -> None:
       """Test with a real, stable URL."""
       tool = WebScraperTool()
       result = await tool(url="https://example.com")

       assert result.success is True
       assert "Example Domain" in result.result["metadata"]["title"]
   ```

### Dependencies
- Milestones 1-3 complete
- Configuration settings in place

### Risk Factors
- **External dependencies**: Integration tests depend on external sites
- **Flaky tests**: Network-dependent tests may be flaky in CI

---

## Milestone 5: Documentation and Final Polish

### Goal
Complete documentation, add usage examples, and ensure production readiness.

### Success Criteria
- [ ] Tool documented in tools-api README
- [ ] Usage examples for agents
- [ ] Type hints complete
- [ ] All tests passing
- [ ] Code review and cleanup

### Key Changes

**1. Add to tools-api README:**
```markdown
### Web Scraper Tool

Fetches and parses web pages into structured, agent-friendly format.

**Parameters:**
- `url` (required): URL to fetch
- `include_html` (optional): Include raw HTML in response
- `include_chunks` (optional): Split content into chunks for retrieval

**Response includes:**
- `text`: Page content with numbered `[n]` link references
- `references`: Array mapping IDs to URLs with context
- `metadata`: Title, description, language
- `outline`: Heading structure (h1-h6)
- `quality`: Content metrics

**Example:**
\`\`\`python
result = await tool(url="https://example.com/article")
# result.result["text"] = "Article with links [1] to sources [2]..."
# result.result["references"][0] = {"id": 1, "url": "...", "text": "links"}
\`\`\`

**Limitations:**
- Only parses static HTML (no JavaScript rendering)
- Respects robots.txt and rate limits
- Maximum content size: 5MB
```

**2. Inline docstrings:**
```python
class WebScraperTool(BaseTool):
    """
    Fetch and parse web pages into structured text with link references.

    This tool provides Lynx-like text rendering with additional structure:
    - Numbered [n] markers for links inline with text
    - References array mapping IDs to URLs
    - Metadata extraction (title, description)
    - Heading outline for navigation
    - Optional chunking for retrieval/embedding

    Limitations:
    - Only parses static HTML (no JavaScript)
    - Respects content-type headers (rejects non-HTML)
    - Subject to timeout and size limits

    Example:
        result = await tool(url="https://example.com")
        print(result.result["text"])  # "Content with links [1]..."
        print(result.result["references"][0])  # {"id": 1, "url": "..."}
    """
```

### Testing Strategy

- Run full test suite
- Review test coverage
- Manual testing with various real-world pages

### Dependencies
- All previous milestones complete

### Risk Factors
- None significant at this stage

---

## Summary

**Total Milestones:** 5

**Estimated Complexity:** Medium
- Core parsing logic is straightforward with BeautifulSoup
- Main complexity is in robust error handling and edge cases

**Key Benefits:**
- Agent-friendly structured output
- Programmatic link references (not just text dump)
- Rich metadata for decision-making
- Optional chunking for retrieval workflows
- No external CLI dependencies (pure Python)

**Dependencies to Add:**
```toml
# pyproject.toml
dependencies = [
    # ... existing ...
    "beautifulsoup4>=4.12",
    "lxml>=5.0",  # Parser for BeautifulSoup
]
```

**Agent Instructions:**
- Complete each milestone fully before moving to next
- Ask clarifying questions before implementing
- Write meaningful tests (edge cases, error conditions)
- Stop after each milestone for review
- Focus on clean, maintainable code over clever optimizations
- Discuss design decisions (e.g., which elements to exclude, chunking strategy)
