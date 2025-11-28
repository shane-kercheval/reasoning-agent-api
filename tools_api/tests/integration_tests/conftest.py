"""Fixtures for tools-api integration tests."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from httpx import ASGITransport, AsyncClient

# Load .env file from project root for integration tests
# This ensures BRAVE_API_KEY and other secrets are available
# Try multiple paths to handle different working directories (CLI vs VS Code debugging)
possible_env_paths = [
    Path(__file__).parent.parent.parent.parent / ".env",  # From test file location
    Path.cwd() / ".env",  # From current working directory
    Path.cwd().parent / ".env",  # One level up from cwd
]

for env_path in possible_env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break


# Path to test fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "prompts"


@pytest_asyncio.fixture(loop_scope="function")
async def tools_api_client():
    """
    Test client for tools-api using ASGITransport (in-process).

    This runs the FastAPI app in the same process as the test, no Docker needed.
    Uses ASGITransport to make HTTP requests without network overhead.

    NOTE: We need to manually trigger the lifespan context manager because
    ASGITransport doesn't automatically run lifespan events.
    """
    from tools_api.main import app
    from tools_api import config
    from tools_api.services.registry import PromptRegistry

    # Clear registry before test to avoid duplicates from previous tests
    PromptRegistry.clear()

    # Set prompts_directory on settings for this test
    original_prompts_dir = config.settings.prompts_directory
    config.settings.prompts_directory = FIXTURES_DIR

    try:
        # Manually trigger lifespan startup
        async with app.router.lifespan_context(app), AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            yield ac
    finally:
        # Restore original setting and clear registry
        config.settings.prompts_directory = original_prompts_dir
        PromptRegistry.clear()


@pytest.fixture
def integration_workspace():
    """
    Create temporary directory structure that mimics Docker volume mounts.

    This fixture simulates the Docker volume structure:
    - /mnt/read_write/Users/test/workspace (container path)
    - /Users/test/workspace (host path)

    The PathMapper is initialized with these paths, allowing tests to verify
    path translation works correctly through the HTTP layer.

    Returns:
        dict with:
            - host_workspace: Path - Simulated host path for read-write access
            - host_downloads: Path - Simulated host path for read-only access
            - container_workspace: Path - Simulated container path (read-write)
            - container_downloads: Path - Simulated container path (read-only)
    """
    # Create temp base directory
    temp_dir = Path(tempfile.mkdtemp(prefix="tools_api_integration_"))

    # Mimic Docker mount structure:
    # Container paths: /mnt/read_write/* and /mnt/read_only/*
    # Host paths: /Users/test/*
    read_write_base = temp_dir / "mnt" / "read_write"
    read_only_base = temp_dir / "mnt" / "read_only"

    # Simulated host paths (what user references)
    host_workspace = Path("/Users/test/workspace")
    host_downloads = Path("/Users/test/downloads")

    # Simulated container paths (mirrored structure)
    container_workspace = read_write_base / "Users" / "test" / "workspace"
    container_downloads = read_only_base / "Users" / "test" / "downloads"

    # Create directories
    container_workspace.mkdir(parents=True)
    container_downloads.mkdir(parents=True)

    # Create some test files
    (container_workspace / "test.txt").write_text("test content")
    (container_downloads / "readonly.txt").write_text("readonly content")

    # Reinitialize PathMapper with test paths
    from tools_api.config import settings

    settings.read_write_base = read_write_base
    settings.read_only_base = read_only_base
    settings.path_mapper.read_write_base = read_write_base
    settings.path_mapper.read_only_base = read_only_base
    settings.path_mapper.discover_mounts()

    yield {
        "host_workspace": host_workspace,
        "host_downloads": host_downloads,
        "container_workspace": container_workspace,
        "container_downloads": container_downloads,
        "temp_dir": temp_dir,
    }

    # Cleanup
    shutil.rmtree(temp_dir)

    # Re-discover original mounts after test
    settings.path_mapper.discover_mounts()
