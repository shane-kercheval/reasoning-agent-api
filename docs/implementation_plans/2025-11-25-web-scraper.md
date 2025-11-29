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

## Milestone 2: Content Length ✅ COMPLETE

### Goal
Add `content_length` field to track text size.

### What Was Implemented
- Added `content_length: int` field to `WebScraperResult` Pydantic model
- Returns character count of extracted text

---

## ~~Milestone 3: Enhanced Link Context~~ SKIPPED

Not needed - the current `external` flag and link text extraction is sufficient.

---

## Milestone 3: Error Handling and Production Hardening (formerly Milestone 4)

### Goal
Add robust error handling, content size limits, and production-ready features.

### Status: ✅ MOSTLY COMPLETE

Most error handling is already implemented in `_fetch_page`:
- Timeout handling with configurable timeout parameter
- Content-type validation (rejects non-HTML)
- Size limits via Content-Length header (5MB max)
- Rate limit detection (429 + Retry-After)
- Proper exception handling for timeouts, connection errors, HTTP errors

### Remaining (Optional)
- [ ] Move constants to `config.py` for environment-based configuration
- [ ] Integration tests with real URLs (optional, can skip in CI)

---

## Milestone 4: Documentation and Final Polish (formerly Milestone 5)

### Goal
Complete documentation and ensure production readiness.

### Success Criteria
- [ ] Type hints complete ✅
- [ ] All tests passing ✅
- [ ] Docstrings in place ✅

---

## Summary

**Total Milestones:** 4 (1 skipped)

| Milestone | Status |
|-----------|--------|
| 1. Core Web Fetcher | ✅ Complete |
| 2. Content Length | ✅ Complete |
| 3. Error Handling | ✅ Mostly Complete |
| 4. Documentation | ✅ Complete |

**What Was Built:**
- `WebScraperTool` class with Pydantic response models
- Lynx-style `[n]` link markers inline with text
- `LinkReference` model with id, url, text, external fields
- `WebScraperResult` flat model with all metadata
- Robust error handling (timeouts, HTTP errors, non-HTML, rate limits)
- 25 unit tests covering all edge cases

**Dependencies Added:**
```toml
# pyproject.toml
"beautifulsoup4>=4.13.4",
"lxml>=6.0.0",
```
