"""Tests for filesystem tools."""

from pathlib import Path

import pytest
import pytest_asyncio

from tools_api import config
from tools_api.services.tools.filesystem import (
    CreateDirectoryTool,
    DeleteDirectoryTool,
    DeleteFileTool,
    EditFileTool,
    GetFileInfoTool,
    ListAllowedDirectoriesTool,
    ListDirectoryTool,
    ListDirectoryWithSizesTool,
    MoveFileTool,
    ReadTextFileTool,
    SearchFilesTool,
    WriteFileTool,
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
    container_dir = config.settings.read_write_base / "workspace" / "test_filesystem"
    container_dir.mkdir(parents=True, exist_ok=True)

    # Host path (what user would reference)
    # Mirrored structure: /mnt/read_write/workspace -> /workspace
    host_dir = Path("/workspace") / "test_filesystem"

    yield host_dir, container_dir

    # Cleanup
    import shutil
    if container_dir.exists():
        shutil.rmtree(container_dir)


@pytest.mark.asyncio
async def test_read_text_file_success(temp_workspace: tuple[Path, Path]) -> None:
    """Test reading a text file."""
    host_dir, container_dir = temp_workspace

    # Create a test file (in container location)
    test_file_container = container_dir / "test.txt"
    test_content = "Hello, World!\nLine 2\nLine 3"
    test_file_container.write_text(test_content)

    # Call tool with host path
    test_file_host = host_dir / "test.txt"
    tool = ReadTextFileTool()
    result = await tool(path=str(test_file_host))

    assert result.success is True
    assert result.result["content"] == test_content
    assert result.result["line_count"] == 3
    assert result.result["char_count"] == len(test_content)
    assert result.result["size_bytes"] > 0
    # Response should contain host path
    assert result.result["path"] == str(test_file_host)


@pytest.mark.asyncio
async def test_read_text_file_not_found(temp_workspace: tuple[Path, Path]) -> None:
    """Test reading non-existent file returns error."""
    host_dir, _container_dir = temp_workspace

    tool = ReadTextFileTool()
    result = await tool(path=str(host_dir / "nonexistent.txt"))

    assert result.success is False
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_read_text_file_permission_denied() -> None:
    """Test reading file outside allowed paths returns error."""
    tool = ReadTextFileTool()
    result = await tool(path="/etc/passwd")

    assert result.success is False
    assert "not accessible" in result.error.lower()


@pytest.mark.asyncio
async def test_write_file_success(temp_workspace: tuple[Path, Path]) -> None:
    """Test writing a file."""
    host_dir, container_dir = temp_workspace

    test_file_host = host_dir / "new_file.txt"
    test_file_container = container_dir / "new_file.txt"
    test_content = "New content"

    tool = WriteFileTool()
    result = await tool(path=str(test_file_host), content=test_content)

    assert result.success is True
    assert result.result["success"] is True
    assert test_file_container.exists()
    assert test_file_container.read_text() == test_content
    assert result.result["path"] == str(test_file_host)


@pytest.mark.asyncio
async def test_write_file_permission_denied() -> None:
    """Test writing file to blocked location returns error."""
    tool = WriteFileTool()
    result = await tool(path="/etc/test.txt", content="test")

    assert result.success is False
    assert "not accessible" in result.error.lower()


@pytest.mark.asyncio
async def test_write_file_blocked_pattern(temp_workspace: tuple[Path, Path]) -> None:
    """Test writing to blocked pattern location returns error."""
    host_dir, _container_dir = temp_workspace

    # Try to write to .env file (blocked pattern)
    tool = WriteFileTool()
    result = await tool(path=str(host_dir / ".env"), content="test")

    assert result.success is False
    assert "blocked" in result.error.lower()


@pytest.mark.asyncio
async def test_edit_file_success(temp_workspace: tuple[Path, Path]) -> None:
    """Test editing a file."""
    host_dir, container_dir = temp_workspace

    test_file_container = container_dir / "edit_test.txt"
    test_file_container.write_text("Hello World")

    test_file_host = host_dir / "edit_test.txt"
    tool = EditFileTool()
    result = await tool(
        path=str(test_file_host),
        old_text="World",
        new_text="Python",
    )

    assert result.success is True
    assert result.result["replacements"] == 1
    assert test_file_container.read_text() == "Hello Python"
    assert result.result["path"] == str(test_file_host)


@pytest.mark.asyncio
async def test_edit_file_text_not_found(temp_workspace: tuple[Path, Path]) -> None:
    """Test editing file when text not found returns error."""
    host_dir, container_dir = temp_workspace

    test_file_container = container_dir / "edit_test.txt"
    test_file_container.write_text("Hello World")

    test_file_host = host_dir / "edit_test.txt"
    tool = EditFileTool()
    result = await tool(
        path=str(test_file_host),
        old_text="NotFound",
        new_text="Python",
    )

    assert result.success is False
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_edit_file_multiple_occurrences(temp_workspace: tuple[Path, Path]) -> None:
    """Test that edit replaces ALL occurrences, not just the first."""
    host_dir, container_dir = temp_workspace

    test_file_container = container_dir / "multi_edit.txt"
    test_file_container.write_text("foo bar foo baz foo")

    test_file_host = host_dir / "multi_edit.txt"
    tool = EditFileTool()
    result = await tool(
        path=str(test_file_host),
        old_text="foo",
        new_text="XXX",
    )

    assert result.success is True
    assert result.result["replacements"] == 3
    assert test_file_container.read_text() == "XXX bar XXX baz XXX"


@pytest.mark.asyncio
async def test_edit_file_replace_with_empty_string(temp_workspace: tuple[Path, Path]) -> None:
    """Test replacing text with empty string (deletion)."""
    host_dir, container_dir = temp_workspace

    test_file_container = container_dir / "delete_edit.txt"
    test_file_container.write_text("Hello DEBUG World DEBUG End")

    test_file_host = host_dir / "delete_edit.txt"
    tool = EditFileTool()
    result = await tool(
        path=str(test_file_host),
        old_text="DEBUG ",
        new_text="",
    )

    assert result.success is True
    assert result.result["replacements"] == 2
    assert test_file_container.read_text() == "Hello World End"


@pytest.mark.asyncio
async def test_edit_file_special_characters(temp_workspace: tuple[Path, Path]) -> None:
    """Test editing with special characters (regex metacharacters, unicode)."""
    host_dir, container_dir = temp_workspace

    test_file_container = container_dir / "special_edit.txt"
    # Content with regex metacharacters and unicode
    test_file_container.write_text("Price: $100.00 (USD) → €85.00")

    test_file_host = host_dir / "special_edit.txt"
    tool = EditFileTool()

    # Replace with regex-like characters (should be literal, not regex)
    result = await tool(
        path=str(test_file_host),
        old_text="$100.00",
        new_text="$200.00",
    )

    assert result.success is True
    assert test_file_container.read_text() == "Price: $200.00 (USD) → €85.00"


@pytest.mark.asyncio
async def test_edit_file_whitespace_only(temp_workspace: tuple[Path, Path]) -> None:
    """Test editing whitespace (tabs, spaces, newlines)."""
    host_dir, container_dir = temp_workspace

    test_file_container = container_dir / "whitespace_edit.txt"
    test_file_container.write_text("line1\t\tline2\n\nline3")

    test_file_host = host_dir / "whitespace_edit.txt"
    tool = EditFileTool()

    # Replace double tab with single space
    result = await tool(
        path=str(test_file_host),
        old_text="\t\t",
        new_text=" ",
    )

    assert result.success is True
    assert test_file_container.read_text() == "line1 line2\n\nline3"


@pytest.mark.asyncio
async def test_edit_file_makes_file_empty(temp_workspace: tuple[Path, Path]) -> None:
    """Test editing that results in empty file."""
    host_dir, container_dir = temp_workspace

    test_file_container = container_dir / "to_empty.txt"
    test_file_container.write_text("delete me")

    test_file_host = host_dir / "to_empty.txt"
    tool = EditFileTool()
    result = await tool(
        path=str(test_file_host),
        old_text="delete me",
        new_text="",
    )

    assert result.success is True
    assert test_file_container.read_text() == ""
    assert result.result["size_bytes"] == 0


@pytest.mark.asyncio
async def test_create_directory_success(temp_workspace: tuple[Path, Path]) -> None:
    """Test creating a directory."""
    host_dir, container_dir = temp_workspace

    new_dir_host = host_dir / "new_directory"
    new_dir_container = container_dir / "new_directory"

    tool = CreateDirectoryTool()
    result = await tool(path=str(new_dir_host))

    assert result.success is True
    assert new_dir_container.exists()
    assert new_dir_container.is_dir()
    assert result.result["path"] == str(new_dir_host)


@pytest.mark.asyncio
async def test_create_directory_nested_success(temp_workspace: tuple[Path, Path]) -> None:
    """Test creating nested directories."""
    host_dir, container_dir = temp_workspace

    new_dir_host = host_dir / "level1" / "level2" / "level3"
    new_dir_container = container_dir / "level1" / "level2" / "level3"

    tool = CreateDirectoryTool()
    result = await tool(path=str(new_dir_host))

    assert result.success is True
    assert new_dir_container.exists()
    assert new_dir_container.is_dir()
    assert result.result["path"] == str(new_dir_host)


@pytest.mark.asyncio
async def test_list_directory_success(temp_workspace: tuple[Path, Path]) -> None:
    """Test listing directory contents."""
    host_dir, container_dir = temp_workspace

    # Create some files and directories
    (container_dir / "file1.txt").write_text("content1")
    (container_dir / "file2.txt").write_text("content2")
    (container_dir / "subdir").mkdir()

    tool = ListDirectoryTool()
    result = await tool(path=str(host_dir))

    assert result.success is True
    assert result.result["count"] == 3
    assert len(result.result["entries"]) == 3

    # Check entries have expected fields and host paths
    entry_names = [e["name"] for e in result.result["entries"]]
    assert "file1.txt" in entry_names
    assert "file2.txt" in entry_names
    assert "subdir" in entry_names
    assert result.result["path"] == str(host_dir)


@pytest.mark.asyncio
async def test_list_directory_not_found(temp_workspace: tuple[Path, Path]) -> None:
    """Test listing non-existent directory returns error."""
    host_dir, _container_dir = temp_workspace

    tool = ListDirectoryTool()
    result = await tool(path=str(host_dir / "nonexistent"))

    assert result.success is False
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_list_directory_with_sizes_success(temp_workspace: tuple[Path, Path]) -> None:
    """Test listing directory with sizes."""
    host_dir, container_dir = temp_workspace

    # Create files with known content
    (container_dir / "file1.txt").write_text("12345")
    (container_dir / "file2.txt").write_text("1234567890")

    tool = ListDirectoryWithSizesTool()
    result = await tool(path=str(host_dir))

    assert result.success is True
    assert result.result["count"] == 2
    assert result.result["total_size_bytes"] == 15  # 5 + 10 bytes

    # Check entries have size information and host paths
    for entry in result.result["entries"]:
        assert "size_bytes" in entry
        assert "modified_time" in entry
    assert result.result["path"] == str(host_dir)


@pytest.mark.asyncio
async def test_search_files_success(temp_workspace: tuple[Path, Path]) -> None:
    """Test searching for files."""
    host_dir, container_dir = temp_workspace

    # Create test files
    (container_dir / "test1.py").write_text("python")
    (container_dir / "test2.txt").write_text("text")
    (container_dir / "test3.py").write_text("python")
    subdir = container_dir / "subdir"
    subdir.mkdir()
    (subdir / "test4.py").write_text("python")

    tool = SearchFilesTool()
    result = await tool(path=str(host_dir), pattern="*.py")

    assert result.success is True
    assert result.result["count"] == 3
    assert all(m["name"].endswith(".py") for m in result.result["matches"])
    assert result.result["search_path"] == str(host_dir)


@pytest.mark.asyncio
async def test_search_files_with_max_results(temp_workspace: tuple[Path, Path]) -> None:
    """Test search with max results limit."""
    host_dir, container_dir = temp_workspace

    # Create many files
    for i in range(10):
        (container_dir / f"test{i}.txt").write_text(f"content{i}")

    tool = SearchFilesTool()
    result = await tool(path=str(host_dir), pattern="*.txt", max_results=5)

    assert result.success is True
    assert result.result["count"] == 5
    assert result.result["truncated"] is True
    assert result.result["search_path"] == str(host_dir)


@pytest.mark.asyncio
async def test_search_files_no_matches(temp_workspace: tuple[Path, Path]) -> None:
    """Test search with pattern that matches no files."""
    host_dir, container_dir = temp_workspace

    # Create some files that won't match
    (container_dir / "test.txt").write_text("content")
    (container_dir / "test.py").write_text("code")

    tool = SearchFilesTool()
    result = await tool(path=str(host_dir), pattern="*.nonexistent")

    assert result.success is True
    assert result.result["count"] == 0
    assert result.result["matches"] == []
    assert result.result["truncated"] is False


@pytest.mark.asyncio
async def test_search_files_invalid_glob_patterns(temp_workspace: tuple[Path, Path]) -> None:
    """Test search with malformed glob patterns.

    Python's pathlib.rglob() is lenient with malformed patterns - it returns
    empty results rather than raising errors. This test documents that behavior.
    """
    host_dir, container_dir = temp_workspace

    # Create a file
    (container_dir / "test.txt").write_text("content")

    tool = SearchFilesTool()

    # These patterns are technically malformed (unclosed brackets)
    # but rglob handles them gracefully by returning no matches
    malformed_patterns = [
        "[",           # Unclosed bracket
        "[abc",        # Unclosed bracket
        "test[",       # Unclosed bracket in middle
        "*[*",         # Mixed invalid
    ]

    for pattern in malformed_patterns:
        result = await tool(path=str(host_dir), pattern=pattern)
        # Should succeed but return no matches (graceful handling)
        assert result.success is True, f"Failed for pattern: {pattern}"
        assert result.result["count"] == 0, f"Unexpected match for pattern: {pattern}"


@pytest.mark.asyncio
async def test_search_files_empty_pattern(temp_workspace: tuple[Path, Path]) -> None:
    """Test search with empty pattern.

    Python's rglob("") returns only the directory itself (not files).
    Since SearchFilesTool filters for files only, empty pattern returns 0 matches.
    This documents the current (expected) behavior.
    """
    host_dir, container_dir = temp_workspace

    # Create files
    (container_dir / "test.txt").write_text("content")
    (container_dir / "test.py").write_text("code")

    tool = SearchFilesTool()
    result = await tool(path=str(host_dir), pattern="")

    # Empty pattern with rglob returns directory, not files
    # SearchFilesTool filters for is_file(), so count is 0
    assert result.success is True
    assert result.result["count"] == 0


@pytest.mark.asyncio
async def test_get_file_info_success(temp_workspace: tuple[Path, Path]) -> None:
    """Test getting file info."""
    host_dir, container_dir = temp_workspace

    test_file_container = container_dir / "info_test.txt"
    test_file_container.write_text("test content")

    test_file_host = host_dir / "info_test.txt"
    tool = GetFileInfoTool()
    result = await tool(path=str(test_file_host))

    assert result.success is True
    assert result.result["name"] == "info_test.txt"
    assert result.result["is_file"] is True
    assert result.result["is_dir"] is False
    assert result.result["size_bytes"] > 0
    assert "permissions" in result.result
    assert result.result["path"] == str(test_file_host)


@pytest.mark.asyncio
async def test_get_file_info_directory(temp_workspace: tuple[Path, Path]) -> None:
    """Test getting directory info."""
    host_dir, container_dir = temp_workspace

    test_dir_container = container_dir / "info_dir"
    test_dir_container.mkdir()

    test_dir_host = host_dir / "info_dir"
    tool = GetFileInfoTool()
    result = await tool(path=str(test_dir_host))

    assert result.success is True
    assert result.result["is_file"] is False
    assert result.result["is_dir"] is True
    assert result.result["path"] == str(test_dir_host)


@pytest.mark.asyncio
async def test_list_allowed_directories_success() -> None:
    """Test listing allowed directories."""
    tool = ListAllowedDirectoriesTool()
    result = await tool()

    assert result.success is True
    assert "directories" in result.result
    assert "read_write_base" in result.result
    assert "read_only_base" in result.result
    assert "total_count" in result.result
    assert "blocked_patterns" in result.result

    # Should have at least the test directories (repos, workspace, downloads, playbooks)
    assert result.result["total_count"] >= 2
    assert len(result.result["directories"]) >= 2

    # Check directory structure
    for directory in result.result["directories"]:
        assert "path" in directory
        assert "access" in directory
        assert directory["access"] in ["read-write", "read-only"]
        assert "exists" in directory


@pytest.mark.asyncio
async def test_move_file_success(temp_workspace: tuple[Path, Path]) -> None:
    """Test moving a file."""
    host_dir, container_dir = temp_workspace

    source_container = container_dir / "source.txt"
    source_container.write_text("move me")

    source_host = host_dir / "source.txt"
    dest_host = host_dir / "dest.txt"
    dest_container = container_dir / "dest.txt"

    tool = MoveFileTool()
    result = await tool(source=str(source_host), destination=str(dest_host))

    assert result.success is True
    assert not source_container.exists()
    assert dest_container.exists()
    assert dest_container.read_text() == "move me"
    assert result.result["source"] == str(source_host)
    assert result.result["destination"] == str(dest_host)


@pytest.mark.asyncio
async def test_move_file_source_not_found(temp_workspace: tuple[Path, Path]) -> None:
    """Test moving non-existent file returns error."""
    host_dir, _container_dir = temp_workspace

    source_host = host_dir / "nonexistent.txt"
    dest_host = host_dir / "dest.txt"

    tool = MoveFileTool()
    result = await tool(source=str(source_host), destination=str(dest_host))

    assert result.success is False
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_move_file_destination_exists(temp_workspace: tuple[Path, Path]) -> None:
    """Test moving to existing destination returns error."""
    host_dir, container_dir = temp_workspace

    source_container = container_dir / "source.txt"
    source_container.write_text("move me")
    dest_container = container_dir / "dest.txt"
    dest_container.write_text("already exists")

    source_host = host_dir / "source.txt"
    dest_host = host_dir / "dest.txt"

    tool = MoveFileTool()
    result = await tool(source=str(source_host), destination=str(dest_host))

    assert result.success is False
    assert "exists" in result.error.lower()


@pytest.mark.asyncio
async def test_delete_file_success(temp_workspace: tuple[Path, Path]) -> None:
    """Test deleting a file."""
    host_dir, container_dir = temp_workspace

    test_file_container = container_dir / "delete_me.txt"
    test_file_container.write_text("delete this")

    test_file_host = host_dir / "delete_me.txt"
    tool = DeleteFileTool()
    result = await tool(path=str(test_file_host))

    assert result.success is True
    assert not test_file_container.exists()
    assert result.result["path"] == str(test_file_host)


@pytest.mark.asyncio
async def test_delete_file_not_found(temp_workspace: tuple[Path, Path]) -> None:
    """Test deleting non-existent file returns error."""
    host_dir, _container_dir = temp_workspace

    tool = DeleteFileTool()
    result = await tool(path=str(host_dir / "nonexistent.txt"))

    assert result.success is False
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_delete_file_is_directory(temp_workspace: tuple[Path, Path]) -> None:
    """Test deleting directory with delete_file returns error."""
    host_dir, container_dir = temp_workspace

    test_dir_container = container_dir / "a_directory"
    test_dir_container.mkdir()

    test_dir_host = host_dir / "a_directory"
    tool = DeleteFileTool()
    result = await tool(path=str(test_dir_host))

    assert result.success is False
    assert "not a file" in result.error.lower()


@pytest.mark.asyncio
async def test_delete_directory_success(temp_workspace: tuple[Path, Path]) -> None:
    """Test deleting a directory."""
    host_dir, container_dir = temp_workspace

    test_dir_container = container_dir / "delete_dir"
    test_dir_container.mkdir()
    (test_dir_container / "file.txt").write_text("content")
    (test_dir_container / "subdir").mkdir()

    test_dir_host = host_dir / "delete_dir"
    tool = DeleteDirectoryTool()
    result = await tool(path=str(test_dir_host))

    assert result.success is True
    assert not test_dir_container.exists()
    assert result.result["path"] == str(test_dir_host)


@pytest.mark.asyncio
async def test_delete_directory_not_found(temp_workspace: tuple[Path, Path]) -> None:
    """Test deleting non-existent directory returns error."""
    host_dir, _container_dir = temp_workspace

    tool = DeleteDirectoryTool()
    result = await tool(path=str(host_dir / "nonexistent_dir"))

    assert result.success is False
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_delete_directory_is_file(temp_workspace: tuple[Path, Path]) -> None:
    """Test deleting file with delete_directory returns error."""
    host_dir, container_dir = temp_workspace

    test_file_container = container_dir / "a_file.txt"
    test_file_container.write_text("content")

    test_file_host = host_dir / "a_file.txt"
    tool = DeleteDirectoryTool()
    result = await tool(path=str(test_file_host))

    assert result.success is False
    assert "not a directory" in result.error.lower()


# ========================================
# HIGH-VALUE EDGE CASE TESTS
# ========================================


@pytest.mark.asyncio
async def test_concurrent_reads(temp_workspace: tuple[Path, Path]) -> None:
    """Test 100 concurrent file reads don't cause issues."""
    import asyncio

    host_dir, container_dir = temp_workspace

    # Create test file
    test_file_container = container_dir / "concurrent_test.txt"
    test_file_container.write_text("concurrent read test content")

    test_file_host = host_dir / "concurrent_test.txt"
    tool = ReadTextFileTool()

    # Execute 100 concurrent reads
    tasks = [tool(path=str(test_file_host)) for _ in range(100)]
    results = await asyncio.gather(*tasks)

    # All should succeed
    assert all(r.success for r in results)
    assert all(r.result["content"] == "concurrent read test content" for r in results)
    assert len(results) == 100


@pytest.mark.asyncio
async def test_read_file_size_limit(temp_workspace: tuple[Path, Path]) -> None:
    """Test that files larger than 50MB are rejected."""
    host_dir, container_dir = temp_workspace

    # Create a file larger than 50MB (simulate with size check)
    test_file_container = container_dir / "large_file.txt"
    # Write small file but we'll mock the stat to simulate large file
    test_file_container.write_text("small content")

    # Create a large file (just over 50MB)
    large_content = "x" * (51 * 1024 * 1024)  # 51MB
    large_file_container = container_dir / "actual_large_file.txt"
    large_file_container.write_text(large_content)

    test_file_host = host_dir / "actual_large_file.txt"
    tool = ReadTextFileTool()
    result = await tool(path=str(test_file_host))

    assert result.success is False
    assert "too large" in result.error.lower()
    assert "50" in result.error  # Should mention 50MB limit


@pytest.mark.asyncio
async def test_empty_file_read(temp_workspace: tuple[Path, Path]) -> None:
    """Test reading an empty file."""
    host_dir, container_dir = temp_workspace

    test_file_container = container_dir / "empty.txt"
    test_file_container.write_text("")

    test_file_host = host_dir / "empty.txt"
    tool = ReadTextFileTool()
    result = await tool(path=str(test_file_host))

    assert result.success is True
    assert result.result["content"] == ""
    assert result.result["size_bytes"] == 0
    assert result.result["line_count"] == 0


@pytest.mark.asyncio
async def test_write_blocked_sensitive_files(temp_workspace: tuple[Path, Path]) -> None:
    """Test that sensitive files are blocked from writing."""
    host_dir, _container_dir = temp_workspace

    tool = WriteFileTool()

    # Test various sensitive file patterns
    sensitive_files = [
        ".env",
        ".env.local",
        ".env.production",
        "secrets.yaml",
        "credentials.json",
        "private.pem",
        "server.key",
        "cert.crt",
        "keystore.p12",
        "database.db",
        "app.sqlite",
    ]

    for filename in sensitive_files:
        test_file_host = host_dir / filename
        result = await tool(path=str(test_file_host), content="sensitive")

        assert result.success is False, f"Failed to block {filename}"
        assert "blocked" in result.error.lower(), f"Wrong error for {filename}"


@pytest.mark.asyncio
async def test_write_blocked_sensitive_directories(temp_workspace: tuple[Path, Path]) -> None:
    """Test that files in sensitive directories are blocked."""
    host_dir, container_dir = temp_workspace

    # Create sensitive directories
    for dirname in [".aws", ".ssh", ".gnupg"]:
        (container_dir / dirname).mkdir(exist_ok=True)

    tool = WriteFileTool()

    sensitive_paths = [
        ".aws/credentials",
        ".ssh/id_rsa",
        ".gnupg/private-keys-v1.d/test.key",
    ]

    for path_str in sensitive_paths:
        test_file_host = host_dir / path_str
        result = await tool(path=str(test_file_host), content="secret")

        assert result.success is False, f"Failed to block {path_str}"
        assert "blocked" in result.error.lower(), f"Wrong error for {path_str}"


@pytest.mark.asyncio
async def test_concurrent_writes_to_same_file(temp_workspace: tuple[Path, Path]) -> None:
    """Test concurrent writes to the same file.

    NOTE: There is currently NO locking mechanism for concurrent writes.
    This test documents the current behavior: the last write wins.
    Multiple concurrent writes may interleave, resulting in corrupted content.
    For production use with concurrent writers, external coordination is required.
    """
    import asyncio

    host_dir, container_dir = temp_workspace
    test_file_host = host_dir / "concurrent_write_test.txt"

    tool = WriteFileTool()

    # Create tasks that write different content
    async def write_content(content: str) -> None:
        await tool(path=str(test_file_host), content=content)

    # Launch multiple concurrent writes
    tasks = [
        write_content(f"Content from writer {i}\n" * 100)
        for i in range(10)
    ]

    await asyncio.gather(*tasks)

    # File should exist with SOME content (last writer wins in race)
    test_file_container = container_dir / "concurrent_write_test.txt"
    assert test_file_container.exists()
    content = test_file_container.read_text()
    # Content should be from one of the writers (not interleaved/corrupted)
    # With no locking, this may occasionally fail with interleaved content
    assert content.startswith("Content from writer ")


@pytest.mark.asyncio
async def test_concurrent_writes_to_different_files(temp_workspace: tuple[Path, Path]) -> None:
    """Test concurrent writes to different files - should always succeed."""
    import asyncio

    host_dir, container_dir = temp_workspace

    tool = WriteFileTool()

    async def write_file(filename: str, content: str) -> None:
        result = await tool(path=str(host_dir / filename), content=content)
        assert result.success is True

    # Write to 50 different files concurrently
    tasks = [
        write_file(f"file_{i}.txt", f"Content {i}")
        for i in range(50)
    ]

    await asyncio.gather(*tasks)

    # All files should exist with correct content
    for i in range(50):
        file_path = container_dir / f"file_{i}.txt"
        assert file_path.exists()
        assert file_path.read_text() == f"Content {i}"
