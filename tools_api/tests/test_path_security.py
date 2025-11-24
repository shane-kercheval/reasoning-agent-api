"""Tests for path security validation."""

from pathlib import Path


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
