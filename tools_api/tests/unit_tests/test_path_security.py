"""Tests for path security validation."""

from pathlib import Path

import pytest

from tools_api.config import settings


def test_blocked_patterns() -> None:
    """Test that protected patterns are blocked."""
    read_write_path = settings.read_write_base

    # .git directory should be blocked
    is_writable, error = settings.is_writable(read_write_path / "project" / ".git" / "config")
    assert not is_writable
    assert "blocked" in error.lower()

    # node_modules should be blocked
    is_writable, error = settings.is_writable(read_write_path / "project" / "node_modules" / "package.json")
    assert not is_writable
    assert "blocked" in error.lower()

    # .env files should be blocked
    is_writable, error = settings.is_writable(read_write_path / "project" / ".env")
    assert not is_writable
    assert "blocked" in error.lower()

    # .venv should be blocked
    is_writable, error = settings.is_writable(read_write_path / "project" / ".venv" / "lib" / "python3.13" / "site-packages" / "foo.py")
    assert not is_writable
    assert "blocked" in error.lower()

    # __pycache__ should be blocked
    is_writable, error = settings.is_writable(read_write_path / "project" / "__pycache__" / "module.pyc")
    assert not is_writable
    assert "blocked" in error.lower()


def test_normal_files_in_read_write_writable() -> None:
    """Test that normal files in read-write volumes are considered writable."""
    test_path = settings.read_write_base / "project" / "src" / "main.py"
    is_writable, error = settings.is_writable(test_path)
    # Should be writable if parent exists (may fail if parent doesn't exist in test)
    # Either writable OR error mentions parent
    assert is_writable or (error and "parent" in error.lower())


def test_read_only_not_writable() -> None:
    """Test that read-only directory is not writable."""
    test_path = settings.read_only_base / "downloads" / "file.txt"
    is_readable, _ = settings.is_readable(test_path)
    assert is_readable

    is_writable, error = settings.is_writable(test_path)
    assert not is_writable
    assert "not in writable location" in error.lower()


def test_playbooks_not_writable() -> None:
    """Test that playbooks directory (read-only) is not writable."""
    test_path = settings.read_only_base / "playbooks" / "meta-prompts.yaml"
    is_writable, error = settings.is_writable(test_path)
    assert not is_writable
    assert "not in writable location" in error.lower()


def test_workspace_writable() -> None:
    """Test that workspace directory (read-write) is writable."""
    test_path = settings.read_write_base / "workspace" / "temp.txt"
    is_writable, error = settings.is_writable(test_path)
    # Should be writable if parent exists
    assert is_writable or (error and "parent" in error.lower())


def test_outside_allowed_paths_not_readable() -> None:
    """Test that paths outside allowed directories are rejected."""
    is_readable, error = settings.is_readable(Path("/etc/passwd"))
    assert not is_readable
    assert "not in readable locations" in error.lower()


def test_path_traversal_blocked() -> None:
    """Test that path traversal attempts are blocked."""
    # Try to escape via path traversal
    is_readable, error = settings.is_readable(Path("/mnt/read_write/project/../../etc/passwd"))
    assert not is_readable
    assert "not in readable locations" in error.lower()


def test_complex_path_traversal_blocked() -> None:
    """
    Test that complex path traversal patterns are blocked.

    Specifically tests patterns like /workspace/real/../../../.env that combine
    legitimate-looking paths with traversal attempts to access sensitive files.
    These paths resolve (via Path.resolve()) to locations OUTSIDE the allowed
    directories, and the path mapper correctly rejects them.
    """
    # Test various complex traversal patterns via the path mapper
    # Each path must have MORE '..' components than directory components to escape
    traversal_patterns = [
        # Basic pattern that looks like a real path but escapes
        # /workspace/real/../../.. -> / (escapes /workspace)
        "/workspace/real/../../../etc/passwd",
        # Multiple traversals - 4 dirs, 5 up = escapes
        "/workspace/a/b/c/../../../../etc/shadow",
        # Traversal to access .env in parent - 1 dir, 3 up = escapes
        "/workspace/project/../../../.env",
        # Mixed: valid prefix + traversal + blocked file - 1 dir, 2 up = escapes
        "/workspace/src/../../.env.local",
        # Deep traversal - 5 dirs, 6 up = escapes
        "/workspace/a/b/c/d/e/../../../../../../etc/passwd",
    ]

    for pattern in traversal_patterns:
        # Test that path mapper rejects these paths
        with pytest.raises(PermissionError):
            settings.path_mapper.to_container_path(pattern)


@pytest.mark.asyncio
async def test_path_traversal_via_read_tool() -> None:
    """Test path traversal attempts through ReadTextFileTool are blocked."""
    from tools_api.services.tools.filesystem import ReadTextFileTool

    tool = ReadTextFileTool()

    # These paths look like they might be in /workspace but actually escape
    traversal_attempts = [
        "/workspace/../../../etc/passwd",
        "/workspace/project/../../.env",
        "/workspace/./../../etc/shadow",
    ]

    for path in traversal_attempts:
        result = await tool(path=path)
        assert result.success is False
        assert "not accessible" in result.error.lower() or "permission" in result.error.lower()


@pytest.mark.asyncio
async def test_path_traversal_via_write_tool() -> None:
    """Test path traversal attempts through WriteFileTool are blocked."""
    from tools_api.services.tools.filesystem import WriteFileTool

    tool = WriteFileTool()

    # Attempts to write to sensitive locations via traversal
    traversal_attempts = [
        "/workspace/../../../etc/cron.d/malicious",
        "/workspace/project/../../.env",
        "/workspace/./../../tmp/evil.sh",
    ]

    for path in traversal_attempts:
        result = await tool(path=path, content="malicious content")
        assert result.success is False
        assert "not accessible" in result.error.lower() or "permission" in result.error.lower()
