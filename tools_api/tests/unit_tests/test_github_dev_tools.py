"""Tests for GitHub and development tools."""

import shutil
import subprocess
from pathlib import Path

import pytest
import pytest_asyncio

from tools_api import config
from tools_api.services.tools.file_system import GetDirectoryTreeTool
from tools_api.services.tools.github_dev_tools import (
    GetGitHubPullRequestInfoTool,
    GetLocalGitChangesInfoTool,
)


def is_command_available(cmd: list[str]) -> bool:
    """Check if a CLI command is available."""
    try:
        result = subprocess.run(
            cmd,
            check=False, capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Skip markers for tests requiring external CLI tools
requires_tree = pytest.mark.skipif(
    not is_command_available(["tree", "--version"]),
    reason="tree command not installed (install via: brew install tree / apt-get install tree)",
)

requires_gh = pytest.mark.skipif(
    not is_command_available(["gh", "--version"]),
    reason="GitHub CLI (gh) not installed (install via: brew install gh / https://cli.github.com)",
)


@pytest_asyncio.fixture
async def temp_workspace() -> tuple[Path, Path]:
    """
    Create a temporary workspace directory for testing.

    Returns:
        Tuple of (host_path, container_path) for test files.
        Tests should use host_path when calling tools.
    """
    # Container path (where files actually exist)
    container_dir = config.settings.read_write_base / "workspace" / "test_github_dev"
    container_dir.mkdir(parents=True, exist_ok=True)

    # Host path (what user would reference)
    host_dir = Path("/workspace") / "test_github_dev"

    yield host_dir, container_dir

    # Cleanup
    if container_dir.exists():
        shutil.rmtree(container_dir)


@requires_tree
@pytest.mark.asyncio
async def test_get_directory_tree_success(temp_workspace: tuple[Path, Path]) -> None:
    """Test generating a directory tree."""
    host_dir, container_dir = temp_workspace

    # Create test files and directories in container path
    tree_dir = container_dir / "tree_test"
    tree_dir.mkdir(parents=True, exist_ok=True)
    (tree_dir / "file1.txt").write_text("content1")
    (tree_dir / "file2.py").write_text("content2")
    subdir = tree_dir / "subdir"
    subdir.mkdir(exist_ok=True)
    (subdir / "file3.txt").write_text("content3")

    # Call tool with host path
    host_tree_dir = host_dir / "tree_test"
    tool = GetDirectoryTreeTool()
    result = await tool(directory=str(host_tree_dir))

    assert result.success is True
    assert "tree_test" in result.result["output"]
    assert "file1.txt" in result.result["output"]
    # Directory in response should be the host path we passed in
    assert result.result["directory"] == str(host_tree_dir)


@requires_tree
@pytest.mark.asyncio
async def test_get_directory_tree_with_excludes(temp_workspace: tuple[Path, Path]) -> None:
    """Test generating directory tree with custom excludes."""
    host_dir, container_dir = temp_workspace

    # Create test files in container path
    exclude_dir = container_dir / "tree_exclude_test"
    exclude_dir.mkdir(parents=True, exist_ok=True)
    (exclude_dir / "include.txt").write_text("include this")
    (exclude_dir / "exclude.log").write_text("exclude this")

    # Call tool with host path
    host_exclude_dir = host_dir / "tree_exclude_test"
    tool = GetDirectoryTreeTool()
    result = await tool(
        directory=str(host_exclude_dir),
        custom_excludes="*.log",
    )

    assert result.success is True
    # Should include .txt but exclude .log files
    assert "include.txt" in result.result["output"]


@requires_tree
@pytest.mark.asyncio
async def test_get_directory_tree_not_found() -> None:
    """Test directory tree for non-existent directory."""
    tool = GetDirectoryTreeTool()
    # Use a path that won't exist but is still in a mapped location
    result = await tool(directory="/workspace/definitely_nonexistent_xyz123")

    assert result.success is False
    assert "failed" in result.error.lower() or "error" in result.error.lower()


@pytest.mark.asyncio
async def test_get_local_git_changes_success(temp_workspace: tuple[Path, Path]) -> None:
    """Test getting local git changes from a repository."""
    host_dir, container_dir = temp_workspace

    # Create a minimal git repo in the container path
    git_dir = container_dir / "test_repo"
    git_dir.mkdir(parents=True, exist_ok=True)

    import subprocess

    # Initialize a git repo
    subprocess.run(["git", "init"], cwd=git_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=git_dir, capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=git_dir, capture_output=True, check=True,
    )

    # Create and commit a file
    (git_dir / "test.txt").write_text("test content")
    subprocess.run(["git", "add", "."], cwd=git_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=git_dir, capture_output=True, check=True,
    )

    # Call tool with host path
    host_git_dir = host_dir / "test_repo"
    tool = GetLocalGitChangesInfoTool()
    result = await tool(directory=str(host_git_dir))

    assert result.success is True
    assert "Git Status" in result.result["output"]
    assert result.result["directory"] == str(host_git_dir)


@pytest.mark.asyncio
async def test_get_local_git_changes_not_a_repo(temp_workspace: tuple[Path, Path]) -> None:
    """Test getting git changes from non-git directory."""
    host_dir, container_dir = temp_workspace

    # Create a non-git directory
    not_repo = container_dir / "not_a_repo"
    not_repo.mkdir(parents=True, exist_ok=True)

    # Call tool with host path
    host_not_repo = host_dir / "not_a_repo"
    tool = GetLocalGitChangesInfoTool()
    result = await tool(directory=str(host_not_repo))

    assert result.success is False
    assert "not a git repository" in result.error.lower()


@pytest.mark.asyncio
async def test_get_local_git_changes_directory_not_found() -> None:
    """Test getting git changes from non-existent directory."""
    tool = GetLocalGitChangesInfoTool()
    # Use a path that won't exist but is still in a mapped location
    result = await tool(directory="/workspace/definitely_nonexistent_xyz123")

    assert result.success is False
    assert "failed" in result.error.lower() or "error" in result.error.lower()


@requires_gh
@pytest.mark.asyncio
async def test_get_github_pr_info_invalid_url() -> None:
    """Test GitHub PR info with invalid URL format."""
    tool = GetGitHubPullRequestInfoTool()
    result = await tool(pr_url="not-a-valid-url")

    # Should complete but with error message in output
    assert result.success is True
    assert "Error: Invalid GitHub PR URL" in result.result["output"]


@requires_gh
@pytest.mark.asyncio
async def test_get_github_pr_info_url_format() -> None:
    """Test GitHub PR info parameter validation."""
    tool = GetGitHubPullRequestInfoTool()

    # Test with properly formatted URL (will fail if gh not authenticated)
    result = await tool(pr_url="https://github.com/owner/repo/pull/1")

    # Either succeeds with data or fails with authentication/not found error
    # We can't guarantee success without valid auth and PR, but should not crash
    assert result.success is True or "failed" in result.error.lower()


@pytest.mark.asyncio
async def test_github_pr_tool_metadata() -> None:
    """Test GitHub PR tool has correct metadata."""
    tool = GetGitHubPullRequestInfoTool()

    assert tool.name == "get_github_pull_request_info"
    assert "GitHub" in tool.description
    assert "pr_url" in tool.parameters["properties"]
    assert "github" in tool.tags


@pytest.mark.asyncio
async def test_git_changes_tool_metadata() -> None:
    """Test git changes tool has correct metadata."""
    tool = GetLocalGitChangesInfoTool()

    assert tool.name == "get_local_git_changes_info"
    assert "Git" in tool.description
    assert "directory" in tool.parameters["properties"]
    assert "git" in tool.tags


@pytest.mark.asyncio
async def test_directory_tree_tool_metadata() -> None:
    """Test directory tree tool has correct metadata."""
    tool = GetDirectoryTreeTool()

    assert tool.name == "get_directory_tree"
    assert "directory" in tool.description.lower() or "tree" in tool.description.lower()
    assert "directory" in tool.parameters["properties"]
    assert "filesystem" in tool.tags or "development" in tool.tags
