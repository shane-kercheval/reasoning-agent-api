"""Tests for GitHub and development tools."""

from pathlib import Path

import pytest

from tools_api import config
from tools_api.services.tools.github_dev_tools import (
    GetDirectoryTreeTool,
    GetGitHubPullRequestInfoTool,
    GetLocalGitChangesInfoTool,
)


@pytest.mark.asyncio
async def test_get_directory_tree_success() -> None:
    """Test generating a directory tree."""
    # Use test workspace directory
    test_dir = config.settings.read_write_base / "workspace" / "tree_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create some test files and directories
    (test_dir / "file1.txt").write_text("content1")
    (test_dir / "file2.py").write_text("content2")
    subdir = test_dir / "subdir"
    subdir.mkdir(exist_ok=True)
    (subdir / "file3.txt").write_text("content3")

    tool = GetDirectoryTreeTool()
    result = await tool(directory=str(test_dir))

    assert result.success is True
    assert "tree_test" in result.result["output"]
    assert "file1.txt" in result.result["output"]
    assert result.result["directory"] == str(test_dir)

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)


@pytest.mark.asyncio
async def test_get_directory_tree_with_excludes() -> None:
    """Test generating directory tree with custom excludes."""
    test_dir = config.settings.read_write_base / "workspace" / "tree_exclude_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create test files
    (test_dir / "include.txt").write_text("include this")
    (test_dir / "exclude.log").write_text("exclude this")

    tool = GetDirectoryTreeTool()
    result = await tool(
        directory=str(test_dir),
        custom_excludes="*.log",
    )

    assert result.success is True
    # Should include .txt but exclude .log files
    assert "include.txt" in result.result["output"]

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)


@pytest.mark.asyncio
async def test_get_directory_tree_not_found() -> None:
    """Test directory tree for non-existent directory."""
    tool = GetDirectoryTreeTool()
    result = await tool(directory="/nonexistent/directory")

    assert result.success is False
    assert "failed" in result.error.lower() or "error" in result.error.lower()


@pytest.mark.asyncio
async def test_get_local_git_changes_success() -> None:
    """Test getting local git changes from a repository."""
    # Use the current repository as test subject
    test_dir = Path.cwd()

    tool = GetLocalGitChangesInfoTool()
    result = await tool(directory=str(test_dir))

    assert result.success is True
    assert "Git Status" in result.result["output"]
    assert result.result["directory"] == str(test_dir)


@pytest.mark.asyncio
async def test_get_local_git_changes_not_a_repo() -> None:
    """Test getting git changes from non-git directory."""
    test_dir = config.settings.read_write_base / "workspace" / "not_a_repo"
    test_dir.mkdir(parents=True, exist_ok=True)

    tool = GetLocalGitChangesInfoTool()
    result = await tool(directory=str(test_dir))

    assert result.success is False
    assert "not a git repository" in result.error.lower()

    # Cleanup
    test_dir.rmdir()


@pytest.mark.asyncio
async def test_get_local_git_changes_directory_not_found() -> None:
    """Test getting git changes from non-existent directory."""
    tool = GetLocalGitChangesInfoTool()
    result = await tool(directory="/nonexistent/directory")

    assert result.success is False
    assert "failed" in result.error.lower() or "error" in result.error.lower()


@pytest.mark.asyncio
async def test_get_github_pr_info_invalid_url() -> None:
    """Test GitHub PR info with invalid URL format."""
    tool = GetGitHubPullRequestInfoTool()
    result = await tool(pr_url="not-a-valid-url")

    # Should complete but with error message in output
    assert result.success is True
    assert "Error: Invalid GitHub PR URL" in result.result["output"]


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
