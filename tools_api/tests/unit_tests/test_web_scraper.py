"""Tests for web scraper tool."""

import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tools_api.services.tools.web_scraper import WebScraperTool


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "html"
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


@pytest.fixture
def simple_html() -> str:
    """Simple HTML with two links."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Test Page</title>
        <meta name="description" content="A test page for scraping">
    </head>
    <body>
        <p>Hello <a href="/link1">world</a>!</p>
        <p>Visit <a href="https://example.com">example</a>.</p>
    </body>
    </html>
    """


@pytest.fixture
def html_with_script_and_style() -> str:
    """HTML with script and style tags that should be excluded."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Page with Scripts</title>
        <style>body { color: red; }</style>
    </head>
    <body>
        <script>console.log("hidden");</script>
        <p>Visible content here.</p>
        <style>.hidden { display: none; }</style>
        <noscript>No JS fallback</noscript>
    </body>
    </html>
    """


@pytest.fixture
def html_with_nav_footer() -> str:
    """HTML with nav and footer that should be excluded."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Page with Nav</title></head>
    <body>
        <nav><a href="/home">Home</a> | <a href="/about">About</a></nav>
        <main>
            <p>Main content with <a href="/article">article link</a>.</p>
        </main>
        <footer><a href="/privacy">Privacy</a></footer>
    </body>
    </html>
    """


@pytest.fixture
def html_with_duplicate_links() -> str:
    """HTML with duplicate links to same URL."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Duplicate Links</title></head>
    <body>
        <p>Click <a href="/page">here</a> or <a href="/page">there</a>.</p>
        <p>Also see <a href="/other">other page</a>.</p>
    </body>
    </html>
    """


@pytest.fixture
def html_with_relative_links() -> str:
    """HTML with various relative link formats."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Relative Links</title></head>
    <body>
        <p><a href="/absolute/path">Absolute</a></p>
        <p><a href="relative/path">Relative</a></p>
        <p><a href="../parent/path">Parent</a></p>
        <p><a href="https://external.com/page">External</a></p>
    </body>
    </html>
    """


@pytest.fixture
def html_with_headings() -> str:
    """HTML with heading structure."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Article</title></head>
    <body>
        <h1>Main Title</h1>
        <p>Intro paragraph.</p>
        <h2>Section One</h2>
        <p>Content for section one.</p>
        <h3>Subsection</h3>
        <p>More content.</p>
    </body>
    </html>
    """


@pytest.fixture
def html_empty() -> str:
    """Empty HTML page."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Empty</title></head>
    <body></body>
    </html>
    """


@pytest.fixture
def html_no_links() -> str:
    """HTML with no links."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>No Links</title></head>
    <body>
        <p>This is a paragraph with no links at all.</p>
        <p>Just plain text content.</p>
    </body>
    </html>
    """


@pytest.fixture
def html_with_og_meta() -> str:
    """HTML with OpenGraph meta tags."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>OG Page</title>
        <meta property="og:description" content="OpenGraph description here">
    </head>
    <body>
        <p>Content</p>
    </body>
    </html>
    """


def create_mock_response(
    html: str,
    url: str = "https://test.com/page",
    status_code: int = 200,
    content_type: str = "text/html; charset=utf-8",
) -> MagicMock:
    """Create a mock httpx.Response."""
    mock_response = MagicMock()
    mock_response.text = html
    mock_response.url = url
    mock_response.status_code = status_code
    mock_response.headers = {"content-type": content_type}
    return mock_response


@pytest.mark.asyncio
async def test_web_scraper_basic(simple_html: str) -> None:
    """Test basic HTML parsing with link extraction."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(simple_html)
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/page")

        assert result.success is True
        assert "[1]" in result.result.text
        assert "[2]" in result.result.text
        assert len(result.result.references) == 2
        assert result.result.references[0].url == "https://test.com/link1"
        assert result.result.references[1].url == "https://example.com"


@pytest.mark.asyncio
async def test_web_scraper_metadata_extraction(simple_html: str) -> None:
    """Test metadata extraction from HTML."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(simple_html)
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/page")

        assert result.success is True
        assert result.result.title == "Test Page"
        assert result.result.description == "A test page for scraping"
        assert result.result.language == "en"
        assert "text/html" in result.result.content_type


@pytest.mark.asyncio
async def test_web_scraper_excludes_scripts(html_with_script_and_style: str) -> None:
    """Test that script and style content is excluded."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(html_with_script_and_style)
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/page")

        assert result.success is True
        text = result.result.text
        assert "Visible content here" in text
        assert "console.log" not in text
        assert "color: red" not in text
        assert "No JS fallback" not in text  # noscript excluded


@pytest.mark.asyncio
async def test_web_scraper_excludes_nav_footer(html_with_nav_footer: str) -> None:
    """Test that nav and footer content is excluded."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(html_with_nav_footer)
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/page")

        assert result.success is True
        text = result.result.text
        # Main content should be present
        assert "Main content" in text
        assert "article link" in text
        # Nav and footer should be excluded
        assert "Home" not in text
        assert "About" not in text
        assert "Privacy" not in text


@pytest.mark.asyncio
async def test_web_scraper_duplicate_links(html_with_duplicate_links: str) -> None:
    """Test that duplicate links get the same ID."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(html_with_duplicate_links)
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/page")

        assert result.success is True
        # Should have 2 unique references, not 3
        assert len(result.result.references) == 2
        # Both occurrences of /page should have same ID
        text = result.result.text
        assert text.count("[1]") == 2  # Same link appears twice with same ID


@pytest.mark.asyncio
async def test_web_scraper_relative_urls(html_with_relative_links: str) -> None:
    """Test that relative URLs are resolved correctly."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(
            html_with_relative_links,
            url="https://test.com/dir/page",
        )
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/dir/page")

        assert result.success is True
        refs = result.result.references

        # Check URLs are resolved correctly
        urls = [r.url for r in refs]
        assert "https://test.com/absolute/path" in urls
        assert "https://test.com/dir/relative/path" in urls
        assert "https://test.com/parent/path" in urls
        assert "https://external.com/page" in urls


@pytest.mark.asyncio
async def test_web_scraper_external_flag(html_with_relative_links: str) -> None:
    """Test that external links are flagged correctly."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(
            html_with_relative_links,
            url="https://test.com/dir/page",
        )
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/dir/page")

        assert result.success is True
        refs = result.result.references

        # Find internal and external refs
        external_refs = [r for r in refs if r.external]
        internal_refs = [r for r in refs if not r.external]

        assert len(external_refs) == 1
        assert external_refs[0].url == "https://external.com/page"
        assert len(internal_refs) == 3


@pytest.mark.asyncio
async def test_web_scraper_empty_page(html_empty: str) -> None:
    """Test handling of empty page."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(html_empty)
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/page")

        assert result.success is True
        assert result.result.text == ""
        assert len(result.result.references) == 0


@pytest.mark.asyncio
async def test_web_scraper_no_links(html_no_links: str) -> None:
    """Test handling of page with no links."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(html_no_links)
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/page")

        assert result.success is True
        assert "paragraph with no links" in result.result.text
        assert len(result.result.references) == 0
        # No [n] markers in text
        assert "[" not in result.result.text


@pytest.mark.asyncio
async def test_web_scraper_og_meta_fallback(html_with_og_meta: str) -> None:
    """Test OpenGraph meta fallback for description."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(html_with_og_meta)
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/page")

        assert result.success is True
        assert result.result.description == "OpenGraph description here"


@pytest.mark.asyncio
async def test_web_scraper_include_html(simple_html: str) -> None:
    """Test including raw HTML in response."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(simple_html)
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/page", include_html=True)

        assert result.success is True
        assert result.result.raw_html is not None
        assert "<!DOCTYPE html>" in result.result.raw_html


@pytest.mark.asyncio
async def test_web_scraper_no_html_by_default(simple_html: str) -> None:
    """Test that raw HTML is not included by default."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(simple_html)
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/page")

        assert result.success is True
        assert result.result.raw_html is None


@pytest.mark.asyncio
async def test_web_scraper_timeout() -> None:
    """Test timeout handling."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.get.side_effect = httpx.TimeoutException("Connection timed out")
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/slow")

        assert result.success is False
        assert "timeout" in result.error.lower()


@pytest.mark.asyncio
async def test_web_scraper_connection_error() -> None:
    """Test connection error handling."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.get.side_effect = httpx.ConnectError("Connection refused")
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/page")

        assert result.success is False
        assert "connection" in result.error.lower()


@pytest.mark.asyncio
async def test_web_scraper_http_error() -> None:
    """Test HTTP error handling (404, 500, etc.)."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.headers = {"content-type": "text/html"}

        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found",
            request=MagicMock(),
            response=mock_response,
        )
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/notfound")

        assert result.success is False
        assert "404" in result.error


@pytest.mark.asyncio
async def test_web_scraper_non_html_content() -> None:
    """Test rejection of non-HTML content types."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}

        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/api/data")

        assert result.success is False
        assert "non-html" in result.error.lower()


@pytest.mark.asyncio
async def test_web_scraper_rate_limited() -> None:
    """Test handling of rate limiting (429)."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {
            "content-type": "text/html",
            "Retry-After": "60",
        }

        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/page")

        assert result.success is False
        assert "rate limit" in result.error.lower()
        assert "60" in result.error


@pytest.mark.asyncio
async def test_web_scraper_content_too_large() -> None:
    """Test rejection of content that's too large."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {
            "content-type": "text/html",
            "content-length": "10000000",  # 10MB
        }

        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/huge")

        assert result.success is False
        assert "too large" in result.error.lower()


@pytest.mark.asyncio
async def test_web_scraper_final_url_after_redirect(simple_html: str) -> None:
    """Test that final URL is captured after redirects."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        # Simulate a redirect by having different URL in response
        mock_response = create_mock_response(
            simple_html,
            url="https://test.com/final-page",  # Different from request URL
        )
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/redirect")

        assert result.success is True
        assert result.result.url == "https://test.com/redirect"  # Original
        assert result.result.final_url == "https://test.com/final-page"  # After redirect


@pytest.mark.asyncio
async def test_web_scraper_ignores_anchor_links() -> None:
    """Test that anchor links (#section) are ignored."""
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>Page</title></head>
    <body>
        <p><a href="#section1">Jump to section</a></p>
        <p><a href="/real-page">Real link</a></p>
    </body>
    </html>
    """
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(html)
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/page")

        assert result.success is True
        # Only one reference (the real link)
        assert len(result.result.references) == 1
        assert result.result.references[0].url == "https://test.com/real-page"


@pytest.mark.asyncio
async def test_web_scraper_ignores_javascript_links() -> None:
    """Test that javascript: links are ignored."""
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>Page</title></head>
    <body>
        <p><a href="javascript:void(0)">Click me</a></p>
        <p><a href="/real-page">Real link</a></p>
    </body>
    </html>
    """
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(html)
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/page")

        assert result.success is True
        # Only one reference (the real link)
        assert len(result.result.references) == 1
        assert result.result.references[0].url == "https://test.com/real-page"


@pytest.mark.asyncio
async def test_web_scraper_response_structure(simple_html: str) -> None:
    """Test that response has all expected fields."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(simple_html)
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/page")

        assert result.success is True
        response = result.result

        # Check required fields exist via attribute access
        assert response.url is not None
        assert response.final_url is not None
        assert response.status is not None
        assert response.fetched_at is not None
        assert response.text is not None
        assert response.references is not None

        # Check metadata fields
        assert response.title is not None
        assert response.description is not None
        assert response.language is not None
        assert response.content_type is not None
        assert response.content_length is not None
        assert isinstance(response.content_length, int)

        # Check reference structure
        for ref in response.references:
            assert ref.id is not None
            assert ref.url is not None
            assert ref.text is not None
            assert ref.external is not None


@pytest.mark.asyncio
async def test_web_scraper_tool_metadata() -> None:
    """Test tool metadata properties."""
    tool = WebScraperTool()

    assert tool.name == "web_scraper"
    assert "fetch" in tool.description.lower() or "web" in tool.description.lower()
    assert "url" in tool.parameters["properties"]
    assert tool.parameters["required"] == ["url"]
    assert "web" in tool.tags or "scraper" in tool.tags


@pytest.mark.asyncio
async def test_web_scraper_custom_timeout(simple_html: str) -> None:
    """Test that custom timeout is passed to client."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(simple_html)
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/page", timeout=10)

        assert result.success is True
        # Verify AsyncClient was called with custom timeout
        call_kwargs = mock_client.call_args[1]
        assert call_kwargs["timeout"].connect == 10


@pytest.mark.asyncio
async def test_web_scraper_link_text_extraction(simple_html: str) -> None:
    """Test that link text is correctly extracted."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(simple_html)
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/page")

        assert result.success is True
        refs = result.result.references

        # Check link text was extracted
        ref1 = next(r for r in refs if r.id == 1)
        ref2 = next(r for r in refs if r.id == 2)

        assert ref1.text == "world"
        assert ref2.text == "example"


# =============================================================================
# Tests using realistic HTML fixtures
# =============================================================================


@pytest.fixture
def docs_style_html() -> str:
    """Load the Python docs-style HTML fixture."""
    return (FIXTURES_DIR / "docs_style_page.html").read_text()


@pytest.fixture
def modern_spa_html() -> str:
    """Load the modern SPA-style HTML fixture."""
    return (FIXTURES_DIR / "modern_spa_page.html").read_text()


@pytest.mark.asyncio
async def test_save_scraper_artifacts(docs_style_html: str, modern_spa_html: str) -> None:
    """
    Save scraper results to artifact files for tracking changes over time.

    Run this test and use `git diff` on the artifact files to see how
    scraper output changes when the implementation is modified.
    """
    # Fields to exclude from artifacts (volatile or always null in tests)
    exclude_fields = {'fetched_at', 'execution_time_ms', 'raw_html'}

    fixtures = [
        (docs_style_html, "https://docs.example.com/api.html", "docs_style_scraper_result.json"),
        (modern_spa_html, "https://mcp.example.io/docs/intro", "modern_spa_scraper_result.json"),
    ]

    for html, url, filename in fixtures:
        with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
            mock_response = create_mock_response(html, url=url)
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_instance

            tool = WebScraperTool()
            result = await tool(url=url)

            assert result.success is True

            # Dump to dict and remove volatile fields
            data = result.model_dump()
            for field in exclude_fields:
                data.pop(field, None)
                if data.get('result'):
                    data['result'].pop(field, None)

            artifact_path = ARTIFACTS_DIR / filename
            artifact_path.write_text(json.dumps(data, indent=2))


@pytest.mark.asyncio
async def test_docs_style_navigation_extracted(docs_style_html: str) -> None:
    """Navigation links from sidebar should be extracted separately."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(
            docs_style_html,
            url="https://docs.python.org/3/library/api.html",
        )
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://docs.python.org/3/library/api.html")

        assert result.success is True

        # Navigation should contain sidebar links
        nav_texts = [n.text for n in result.result.navigation]
        assert "Flask Guide" in nav_texts
        assert "Django REST Framework" in nav_texts

        # Top navigation links should also be extracted
        nav_urls = [n.url for n in result.result.navigation]
        assert any("search" in url for url in nav_urls)


@pytest.mark.asyncio
async def test_docs_style_headerlinks_filtered(docs_style_html: str) -> None:
    """Headerlink anchors (¶ symbols) should not appear in text or references."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(
            docs_style_html,
            url="https://docs.python.org/3/library/api.html",
        )
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://docs.python.org/3/library/api.html")

        assert result.success is True

        # No ¶ symbols in text
        assert "¶" not in result.result.text

        # No headerlink URLs in references (they point to #section anchors anyway)
        ref_urls = [r.url for r in result.result.references]
        assert not any("#introduction" in url for url in ref_urls)
        assert not any("#getting-started" in url for url in ref_urls)


@pytest.mark.asyncio
async def test_docs_style_main_content_links_indexed(docs_style_html: str) -> None:
    """Only links in main content should be indexed with sequential IDs."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(
            docs_style_html,
            url="https://docs.python.org/3/library/api.html",
        )
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://docs.python.org/3/library/api.html")

        assert result.success is True

        # Main content links should be indexed
        ref_urls = [r.url for r in result.result.references]
        assert "https://fastapi.tiangolo.com" in ref_urls
        assert "https://pip.pypa.io" in ref_urls
        assert "https://redis.io" in ref_urls

        # References should have sequential IDs starting from 1
        ref_ids = [r.id for r in result.result.references]
        assert ref_ids == list(range(1, len(ref_ids) + 1))

        # Markers in text should match references
        for ref in result.result.references:
            assert f"[{ref.id}]" in result.result.text


@pytest.mark.asyncio
async def test_docs_style_nav_links_not_in_main_content(docs_style_html: str) -> None:
    """Navigation links should not appear in main content text."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(
            docs_style_html,
            url="https://docs.python.org/3/library/api.html",
        )
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://docs.python.org/3/library/api.html")

        assert result.success is True

        # Navigation-only text should not appear in main content
        text = result.result.text
        assert "Flask Guide" not in text
        assert "Django REST Framework" not in text
        assert "Table of Contents" not in text


@pytest.mark.asyncio
async def test_docs_style_footer_excluded(docs_style_html: str) -> None:
    """Footer content should be excluded from main text."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(
            docs_style_html,
            url="https://docs.python.org/3/library/api.html",
        )
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://docs.python.org/3/library/api.html")

        assert result.success is True
        assert "Python Software Foundation" not in result.result.text
        assert "Legal" not in result.result.text


@pytest.mark.asyncio
async def test_docs_style_script_excluded(docs_style_html: str) -> None:
    """Script content should never appear in output."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(
            docs_style_html,
            url="https://docs.python.org/3/library/api.html",
        )
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://docs.python.org/3/library/api.html")

        assert result.success is True
        assert "console.log" not in result.result.text
        assert "JavaScript content" not in result.result.text


@pytest.mark.asyncio
async def test_modern_spa_navigation_with_sections(modern_spa_html: str) -> None:
    """Navigation items should include section headers when available."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(
            modern_spa_html,
            url="https://modelcontextprotocol.io/docs/intro",
        )
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://modelcontextprotocol.io/docs/intro")

        assert result.success is True

        # Check navigation has section information
        nav_with_sections = [n for n in result.result.navigation if n.section]
        assert len(nav_with_sections) > 0

        # Check specific sections exist
        sections = {n.section for n in result.result.navigation if n.section}
        assert "Get started" in sections or "Core Concepts" in sections or "Build" in sections


@pytest.mark.asyncio
async def test_modern_spa_sidebar_extracted(modern_spa_html: str) -> None:
    """Sidebar navigation should be extracted from id='sidebar'."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(
            modern_spa_html,
            url="https://modelcontextprotocol.io/docs/intro",
        )
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://modelcontextprotocol.io/docs/intro")

        assert result.success is True

        # Sidebar links should be in navigation
        nav_texts = [n.text for n in result.result.navigation]
        assert "What is MCP?" in nav_texts
        assert "Architecture" in nav_texts
        assert "Build a Server" in nav_texts


@pytest.mark.asyncio
async def test_modern_spa_main_content_only(modern_spa_html: str) -> None:
    """Main content text should only contain article content, not nav."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(
            modern_spa_html,
            url="https://modelcontextprotocol.io/docs/intro",
        )
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://modelcontextprotocol.io/docs/intro")

        assert result.success is True

        text = result.result.text

        # Main content should be present
        assert "Building MCP Servers" in text
        assert "Prerequisites" in text
        assert "pip install mcp-sdk" in text

        # Sidebar-only text should not be in main content
        assert "What is MCP?" not in text  # sidebar link text
        assert "Quickstart" not in text  # sidebar link text


@pytest.mark.asyncio
async def test_modern_spa_sequential_reference_ids(modern_spa_html: str) -> None:
    """Reference IDs should be sequential with no gaps."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(
            modern_spa_html,
            url="https://modelcontextprotocol.io/docs/intro",
        )
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://modelcontextprotocol.io/docs/intro")

        assert result.success is True

        # IDs should be 1, 2, 3, ... with no gaps
        ref_ids = sorted([r.id for r in result.result.references])
        expected_ids = list(range(1, len(ref_ids) + 1))
        assert ref_ids == expected_ids


@pytest.mark.asyncio
async def test_modern_spa_external_links_flagged(modern_spa_html: str) -> None:
    """External links should be correctly flagged."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(
            modern_spa_html,
            url="https://modelcontextprotocol.io/docs/intro",
        )
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://modelcontextprotocol.io/docs/intro")

        assert result.success is True

        # Find external links
        external_refs = [r for r in result.result.references if r.external]
        internal_refs = [r for r in result.result.references if not r.external]

        # Should have both internal and external
        assert len(external_refs) > 0
        assert len(internal_refs) > 0

        # Check specific external links
        external_urls = [r.url for r in external_refs]
        assert any("json-schema.org" in url for url in external_urls)
        assert any("github.com" in url for url in external_urls)


@pytest.mark.asyncio
async def test_modern_spa_header_excluded(modern_spa_html: str) -> None:
    """Header/top nav should be excluded from main content."""
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(
            modern_spa_html,
            url="https://modelcontextprotocol.io/docs/intro",
        )
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://modelcontextprotocol.io/docs/intro")

        assert result.success is True

        # Header navigation text should not be in main content
        text = result.result.text
        assert "Toggle Theme" not in text
        assert "MCP Docs" not in text  # site logo/title from header


@pytest.mark.asyncio
async def test_all_text_markers_have_references() -> None:
    """Every [n] marker in text should have a corresponding reference."""
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>Test</title></head>
    <body>
        <main>
            <p>See <a href="/page1">link one</a> and <a href="/page2">link two</a>.</p>
            <p>Also <a href="/page1">link one again</a>.</p>
        </main>
    </body>
    </html>
    """
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(html, url="https://test.com/")
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/")

        assert result.success is True

        # Extract all [n] markers from text
        markers = re.findall(r'\[(\d+)\]', result.result.text)
        marker_ids = {int(m) for m in markers}

        # All markers should have corresponding references
        ref_ids = {r.id for r in result.result.references}
        assert marker_ids == ref_ids


@pytest.mark.asyncio
async def test_navigation_result_field_present() -> None:
    """The navigation field should always be present in results."""
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>Simple</title></head>
    <body><p>No navigation here.</p></body>
    </html>
    """
    with patch("tools_api.services.tools.web_scraper.httpx.AsyncClient") as mock_client:
        mock_response = create_mock_response(html, url="https://test.com/")
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        tool = WebScraperTool()
        result = await tool(url="https://test.com/")

        assert result.success is True
        assert hasattr(result.result, 'navigation')
        assert isinstance(result.result.navigation, list)
        assert len(result.result.navigation) == 0
