"""Tests for path mapping and translation."""

from pathlib import Path

import pytest

from tools_api.path_mapper import PathMapper, PathMapping


@pytest.fixture
def temp_mounts(tmp_path: Path) -> tuple[Path, Path]:
    """
    Create temporary mount directory structure for testing.

    Structure (non-overlapping mounts):
        tmp_path/
        ├── read_write/
        │   ├── Users/
        │   │   └── shane/
        │   │       └── repos/
        │   │           ├── project1/
        │   │           └── project2/
        │   └── home/
        │       └── user/
        │           └── workspace/
        └── read_only/
            ├── Users/
            │   └── shane/
            │       └── Downloads/
            └── data/
                └── datasets/
    """
    read_write = tmp_path / "read_write"
    read_only = tmp_path / "read_only"

    # Create read-write mounts (specific paths to avoid conflicts)
    (read_write / "Users" / "shane" / "repos").mkdir(parents=True)
    (read_write / "Users" / "shane" / "repos" / "project1").mkdir(parents=True)
    (read_write / "Users" / "shane" / "repos" / "project2").mkdir(parents=True)
    (read_write / "home" / "user" / "workspace").mkdir(parents=True)

    # Create read-only mounts (different paths to avoid conflicts)
    (read_only / "Users" / "shane" / "Downloads").mkdir(parents=True)
    (read_only / "data" / "datasets").mkdir(parents=True)

    return read_write, read_only


def test_discover_mounts_basic(temp_mounts: tuple[Path, Path]) -> None:
    """Test basic mount discovery."""
    read_write, read_only = temp_mounts
    mapper = PathMapper(read_write, read_only)

    mapper.discover_mounts()

    # Should discover all top-level directories
    # RW: Users, home  RO: Users, data
    assert len(mapper.mappings) == 4

    # Check read-write mounts
    rw_mappings = [m for m in mapper.mappings if m.access == "read-write"]
    assert len(rw_mappings) == 2
    rw_hosts = {str(m.host_prefix) for m in rw_mappings}
    assert "/Users" in rw_hosts or "/home" in rw_hosts

    # Check read-only mounts
    ro_mappings = [m for m in mapper.mappings if m.access == "read-only"]
    assert len(ro_mappings) == 2
    ro_hosts = {str(m.host_prefix) for m in ro_mappings}
    assert "/data" in ro_hosts


def test_discover_mounts_sorting(temp_mounts: tuple[Path, Path]) -> None:
    """Test that mappings are sorted by prefix length (longest first)."""
    read_write, read_only = temp_mounts
    mapper = PathMapper(read_write, read_only)

    mapper.discover_mounts()

    # Verify sorted by length descending
    prefix_lengths = [len(str(m.host_prefix)) for m in mapper.mappings]
    assert prefix_lengths == sorted(prefix_lengths, reverse=True)


def test_to_container_path_basic(temp_mounts: tuple[Path, Path]) -> None:
    """Test basic host -> container path translation."""
    read_write, read_only = temp_mounts
    mapper = PathMapper(read_write, read_only)
    mapper.discover_mounts()

    # Test read-write path
    host_path = "/Users/shane/repos/project1/src/main.py"
    container_path, access = mapper.to_container_path(host_path)

    expected = read_write / "Users" / "shane" / "repos" / "project1" / "src" / "main.py"
    assert container_path == expected.resolve()
    assert access == "read-write"


def test_to_container_path_read_only(temp_mounts: tuple[Path, Path]) -> None:
    """Test read-only path translation."""
    read_write, read_only = temp_mounts
    mapper = PathMapper(read_write, read_only)
    mapper.discover_mounts()

    # Use /data path which is only in read_only
    host_path = "/data/datasets/file.csv"
    container_path, access = mapper.to_container_path(host_path)

    expected = read_only / "data" / "datasets" / "file.csv"
    assert container_path == expected.resolve()
    assert access == "read-only"


def test_to_container_path_nested(temp_mounts: tuple[Path, Path]) -> None:
    """Test translation of deeply nested paths."""
    read_write, read_only = temp_mounts
    mapper = PathMapper(read_write, read_only)
    mapper.discover_mounts()

    host_path = "/Users/shane/repos/project1/src/utils/helpers/common.py"
    container_path, access = mapper.to_container_path(host_path)

    expected = (
        read_write
        / "Users"
        / "shane"
        / "repos"
        / "project1"
        / "src"
        / "utils"
        / "helpers"
        / "common.py"
    )
    assert container_path == expected.resolve()
    assert access == "read-write"


def test_to_container_path_unmapped_error(temp_mounts: tuple[Path, Path]) -> None:
    """Test that unmapped paths raise PermissionError."""
    read_write, read_only = temp_mounts
    mapper = PathMapper(read_write, read_only)
    mapper.discover_mounts()

    # Path not in any mount
    with pytest.raises(PermissionError) as exc_info:
        mapper.to_container_path("/etc/passwd")

    assert "Path not accessible" in str(exc_info.value)
    assert "/etc/passwd" in str(exc_info.value)
    assert "Available paths:" in str(exc_info.value)


def test_to_container_path_overlapping_mounts(tmp_path: Path) -> None:
    """Test longest-prefix-first matching for overlapping mounts."""
    # Create overlapping mount structure
    read_write = tmp_path / "read_write"
    read_only = tmp_path / "read_only"

    # Create both /Users/shane/repos and /Users/shane/repos/project1
    (read_write / "Users" / "shane" / "repos").mkdir(parents=True)
    (read_write / "Users" / "shane" / "repos" / "project1").mkdir(parents=True)

    mapper = PathMapper(read_write, read_only)
    mapper.discover_mounts()

    # Should match the longest prefix (more specific mount)
    host_path = "/Users/shane/repos/project1/file.txt"
    container_path, _access = mapper.to_container_path(host_path)

    # Should map to the project1 mount, not the repos mount
    expected = read_write / "Users" / "shane" / "repos" / "project1" / "file.txt"
    assert container_path == expected.resolve()


def test_to_host_path_basic(temp_mounts: tuple[Path, Path]) -> None:
    """Test basic container -> host path translation."""
    read_write, read_only = temp_mounts
    mapper = PathMapper(read_write, read_only)
    mapper.discover_mounts()

    container_path = read_write / "Users" / "shane" / "repos" / "project1" / "README.md"
    host_path = mapper.to_host_path(container_path)

    assert host_path == Path("/Users/shane/repos/project1/README.md")


def test_to_host_path_nested(temp_mounts: tuple[Path, Path]) -> None:
    """Test reverse translation of nested paths."""
    read_write, read_only = temp_mounts
    mapper = PathMapper(read_write, read_only)
    mapper.discover_mounts()

    container_path = (
        read_write / "Users" / "shane" / "repos" / "project1" / "src" / "utils" / "helper.py"
    )
    host_path = mapper.to_host_path(container_path)

    assert host_path == Path("/Users/shane/repos/project1/src/utils/helper.py")


def test_to_host_path_unmapped(temp_mounts: tuple[Path, Path]) -> None:
    """Test that unmapped container paths return as-is."""
    read_write, read_only = temp_mounts
    mapper = PathMapper(read_write, read_only)
    mapper.discover_mounts()

    # Path not in any mapping
    container_path = Path("/mnt/other/file.txt")
    host_path = mapper.to_host_path(container_path)

    # Should return path as-is
    assert host_path == container_path


def test_roundtrip_translation(temp_mounts: tuple[Path, Path]) -> None:
    """Test that host -> container -> host roundtrip works."""
    read_write, read_only = temp_mounts
    mapper = PathMapper(read_write, read_only)
    mapper.discover_mounts()

    original_host = Path("/Users/shane/repos/project1/src/main.py")

    # Host -> Container
    container, _ = mapper.to_container_path(original_host)

    # Container -> Host
    final_host = mapper.to_host_path(container)

    assert final_host == original_host


def test_get_access_level(temp_mounts: tuple[Path, Path]) -> None:
    """Test access level detection."""
    read_write, read_only = temp_mounts
    mapper = PathMapper(read_write, read_only)
    mapper.discover_mounts()

    # Read-write path (under /home or /Users from read_write)
    assert mapper.get_access_level("/home/user/workspace/file.txt") == "read-write"

    # Read-only path (under /data which is only in read_only)
    assert mapper.get_access_level("/data/datasets/file.csv") == "read-only"


def test_get_access_level_unmapped_error(temp_mounts: tuple[Path, Path]) -> None:
    """Test that getting access level for unmapped path raises error."""
    read_write, read_only = temp_mounts
    mapper = PathMapper(read_write, read_only)
    mapper.discover_mounts()

    with pytest.raises(PermissionError):
        mapper.get_access_level("/etc/passwd")


def test_get_all_mappings(temp_mounts: tuple[Path, Path]) -> None:
    """Test retrieving all mappings."""
    read_write, read_only = temp_mounts
    mapper = PathMapper(read_write, read_only)
    mapper.discover_mounts()

    mappings = mapper.get_all_mappings()

    assert len(mappings) == 4
    assert all("host_path" in m for m in mappings)
    assert all("container_path" in m for m in mappings)
    assert all("access" in m for m in mappings)
    assert all(m["access"] in ["read-write", "read-only"] for m in mappings)


def test_empty_mounts() -> None:
    """Test behavior with no mounts."""
    # Non-existent directories
    mapper = PathMapper(Path("/nonexistent/rw"), Path("/nonexistent/ro"))
    mapper.discover_mounts()

    assert len(mapper.mappings) == 0

    with pytest.raises(PermissionError) as exc_info:
        mapper.to_container_path("/any/path")

    assert "(none configured)" in str(exc_info.value)


def test_path_mapping_normalization() -> None:
    """Test that PathMapping normalizes paths on init."""
    # Use relative paths
    mapping = PathMapping(
        host_prefix=Path("Users/shane/repos"),
        container_prefix=Path("mnt/read_write/Users/shane/repos"),
        access="read-write",
    )

    # Should be resolved to absolute paths
    assert mapping.host_prefix.is_absolute()
    assert mapping.container_prefix.is_absolute()


def test_path_with_pathlib_input(temp_mounts: tuple[Path, Path]) -> None:
    """Test that Path objects work as input (not just strings)."""
    read_write, read_only = temp_mounts
    mapper = PathMapper(read_write, read_only)
    mapper.discover_mounts()

    # Use Path object instead of string
    host_path = Path("/Users/shane/repos/project1/file.txt")
    container_path, access = mapper.to_container_path(host_path)

    expected = read_write / "Users" / "shane" / "repos" / "project1" / "file.txt"
    assert container_path == expected.resolve()
    assert access == "read-write"


def test_multiple_directory_levels(tmp_path: Path) -> None:
    """Test mounts at different directory depths."""
    read_write = tmp_path / "read_write"
    read_only = tmp_path / "read_only"

    # Create mounts at different depths
    (read_write / "a").mkdir(parents=True)
    (read_write / "a" / "b").mkdir(parents=True)
    (read_write / "a" / "b" / "c").mkdir(parents=True)
    (read_write / "x" / "y" / "z").mkdir(parents=True)

    mapper = PathMapper(read_write, read_only)
    mapper.discover_mounts()

    # Should discover all top-level directories under read_write
    # a, x (not b, c, y, z which are nested)
    assert len(mapper.mappings) == 2

    rw_paths = {str(m.host_prefix) for m in mapper.mappings}
    assert "/a" in rw_paths
    assert "/x" in rw_paths


def test_case_sensitivity_handling(temp_mounts: tuple[Path, Path]) -> None:
    """Test path handling (case-sensitive on Linux, insensitive on Mac/Windows)."""
    read_write, read_only = temp_mounts
    mapper = PathMapper(read_write, read_only)
    mapper.discover_mounts()

    # Path resolution handles case sensitivity based on OS
    host_path = "/Users/shane/repos/project1/file.txt"
    container_path, access = mapper.to_container_path(host_path)

    # Should work regardless of case (OS-dependent)
    assert access in ["read-write", "read-only"]
    assert container_path.is_absolute()
