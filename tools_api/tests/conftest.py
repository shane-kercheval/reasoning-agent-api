"""Pytest configuration and fixtures for tools-api tests."""

from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from tools_api import config
from tools_api.main import app


@pytest.fixture(autouse=True)
def setup_test_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override settings to use local paths for testing."""
    # Create test base directory structure mimicking /mnt/read_write and /mnt/read_only
    test_base = Path(__file__).parent.parent.parent / "workspace" / "test_tools_api"
    test_base.mkdir(parents=True, exist_ok=True)

    # Create read-write and read-only base directories
    test_read_write = test_base / "read_write"
    test_read_only = test_base / "read_only"

    # Create subdirectories mimicking host paths for path mapper discovery
    # Mirrored structure: /mnt/read_write/home/user/workspace -> /home/user/workspace
    test_workspace = test_read_write / "home" / "user" / "workspace"
    test_repos = test_read_write / "home" / "user" / "repos"
    test_downloads = test_read_only / "home" / "user" / "Downloads"
    test_playbooks = test_read_only / "data" / "playbooks"

    # Create all directories
    for path in [test_workspace, test_repos, test_downloads, test_playbooks]:
        path.mkdir(parents=True, exist_ok=True)

    # Override settings to use test base paths
    monkeypatch.setattr(config.settings, "read_write_base", test_read_write)
    monkeypatch.setattr(config.settings, "read_only_base", test_read_only)

    # Re-initialize path mapper with test paths
    from tools_api.path_mapper import PathMapper

    config.settings.path_mapper = PathMapper(test_read_write, test_read_only)
    config.settings.path_mapper.discover_mounts()


@pytest_asyncio.fixture
async def client():
    """
    Async HTTP client for testing tools-api.

    Uses ASGITransport for in-process testing (no real server).
    """
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac
