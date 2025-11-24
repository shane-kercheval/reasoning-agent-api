"""
Path mapping for host <-> container path translation.

Supports automatic discovery of volume mounts from /mnt/read_write and /mnt/read_only,
enabling natural host path usage while maintaining container path security.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PathMapping:
    """Represents a host <-> container path mapping."""

    host_prefix: Path
    container_prefix: Path
    access: str  # "read-write" or "read-only"

    def __post_init__(self) -> None:
        """Normalize paths on initialization."""
        self.host_prefix = self.host_prefix.resolve()
        self.container_prefix = self.container_prefix.resolve()


class PathMapper:
    """
    Maps host filesystem paths to container paths and vice versa.

    Example:
        Mount: /Users/shane/repos/project:/mnt/read_write/Users/shane/repos/project:rw

        Host path:      /Users/shane/repos/project/src/main.py
        Container path: /mnt/read_write/Users/shane/repos/project/src/main.py
    """

    def __init__(self, read_write_base: Path, read_only_base: Path) -> None:
        """
        Initialize PathMapper.

        Args:
            read_write_base: Base path for read-write mounts (e.g., /mnt/read_write)
            read_only_base: Base path for read-only mounts (e.g., /mnt/read_only)
        """
        self.read_write_base = read_write_base.resolve()
        self.read_only_base = read_only_base.resolve()
        self.mappings: list[PathMapping] = []

    def discover_mounts(self) -> None:
        """
        Discover volume mounts by scanning read_write_base and read_only_base.

        Creates mappings by extracting the tail of each container path as the host path.
        Sorts mappings by prefix length (longest first) for correct overlap handling.
        """
        self.mappings = []

        # Discover read-write mounts
        if self.read_write_base.exists():
            for child in self.read_write_base.iterdir():
                if child.is_dir():
                    # Extract host path from container path
                    # /mnt/read_write/Users/shane/repos/project -> /Users/shane/repos/project
                    relative_to_base = child.relative_to(self.read_write_base)
                    host_path = Path("/") / relative_to_base

                    self.mappings.append(
                        PathMapping(
                            host_prefix=host_path,
                            container_prefix=child,
                            access="read-write",
                        ),
                    )
                    logger.info(f"Discovered RW mount: {host_path} -> {child}")

        # Discover read-only mounts
        if self.read_only_base.exists():
            for child in self.read_only_base.iterdir():
                if child.is_dir():
                    relative_to_base = child.relative_to(self.read_only_base)
                    host_path = Path("/") / relative_to_base

                    self.mappings.append(
                        PathMapping(
                            host_prefix=host_path,
                            container_prefix=child,
                            access="read-only",
                        ),
                    )
                    logger.info(f"Discovered RO mount: {host_path} -> {child}")

        # Sort by prefix length (longest first) to handle overlapping mounts correctly
        self.mappings.sort(key=lambda m: len(str(m.host_prefix)), reverse=True)

        logger.info(f"Discovered {len(self.mappings)} total mounts")

    def to_container_path(self, host_path: str | Path) -> tuple[Path, str]:
        """
        Translate host path to container path.

        Args:
            host_path: Path from host filesystem

        Returns:
            Tuple of (container_path, access_level)

        Raises:
            PermissionError: If path is not accessible via any mount

        Example:
            >>> mapper.to_container_path("/Users/shane/repos/project/file.txt")
            (Path("/mnt/read_write/Users/shane/repos/project/file.txt"), "read-write")
        """
        # Normalize to absolute path
        try:
            host_path = Path(host_path).resolve()
        except (OSError, RuntimeError) as e:
            raise PermissionError(f"Invalid path: {e}")

        # Find matching mapping (longest prefix first)
        for mapping in self.mappings:
            try:
                if host_path.is_relative_to(mapping.host_prefix):
                    # Extract relative part
                    relative = host_path.relative_to(mapping.host_prefix)
                    # Build container path
                    container_path = mapping.container_prefix / relative
                    return container_path, mapping.access
            except ValueError:
                continue

        # No mapping found
        available = "\n".join(
            f"  - {m.host_prefix} ({m.access})" for m in self.mappings
        )
        raise PermissionError(
            f"Path not accessible: {host_path}\n"
            f"Available paths:\n{available if self.mappings else '  (none configured)'}\n\n"
            f"Hint: Ensure Docker volume mount mirrors the full host path.\n"
            f"Example: /Users/you/repos/proj:/mnt/read_write/Users/you/repos/proj:rw",
        )

    def to_host_path(self, container_path: str | Path) -> Path:
        """
        Translate container path back to host path.

        Args:
            container_path: Path from container filesystem

        Returns:
            Host filesystem path

        Example:
            >>> mapper.to_host_path("/mnt/read_write/Users/shane/repos/project/file.txt")
            Path("/Users/shane/repos/project/file.txt")
        """
        try:
            container_path = Path(container_path).resolve()
        except (OSError, RuntimeError):
            # If path doesn't exist or can't be resolved, return as-is
            return Path(container_path)

        # Find matching mapping
        for mapping in self.mappings:
            try:
                if container_path.is_relative_to(mapping.container_prefix):
                    relative = container_path.relative_to(mapping.container_prefix)
                    return mapping.host_prefix / relative
            except ValueError:
                continue

        # No mapping found - return as-is (might be a container-only path)
        return container_path

    def get_access_level(self, host_path: str | Path) -> str:
        """
        Get access level for a host path.

        Args:
            host_path: Path from host filesystem

        Returns:
            Access level ("read-write" or "read-only")

        Raises:
            PermissionError: If path is not accessible
        """
        _, access = self.to_container_path(host_path)
        return access

    def get_all_mappings(self) -> list[dict[str, str]]:
        """
        Get all discovered mappings.

        Returns:
            List of mapping dictionaries with host_path, container_path, and access
        """
        return [
            {
                "host_path": str(m.host_prefix),
                "container_path": str(m.container_prefix),
                "access": m.access,
            }
            for m in self.mappings
        ]
