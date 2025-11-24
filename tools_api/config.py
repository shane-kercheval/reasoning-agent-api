"""Configuration for tools-api service using Pydantic settings."""

import logging
from fnmatch import fnmatch
from pathlib import Path

from pydantic_settings import BaseSettings

from tools_api.path_mapper import PathMapper

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Tools API configuration."""

    # Base mount points for dynamic volume mounting
    # All volumes under /mnt/read_write/* are read-write accessible
    # All volumes under /mnt/read_only/* are read-only accessible
    read_write_base: Path = Path("/mnt/read_write")
    read_only_base: Path = Path("/mnt/read_only")

    # API keys
    github_token: str = ""
    brave_api_key: str = ""

    # Protected path patterns (NEVER writable, even in RW volumes)
    write_blocked_patterns: list[str] = [
        # Version control
        "*/.git/*", "*/.git",

        # Dependencies (never edit directly)
        "*/node_modules/*", "*/.venv/*", "*/venv/*",
        "*/__pycache__/*", "*/site-packages/*",

        # Build artifacts
        "*/dist/*", "*/build/*", "*/.next/*", "*/target/*",

        # Compiled files
        "*.pyc", "*.pyo", "*.so", "*.dylib", "*.class",

        # IDE/Editor files
        "*/.idea/*", "*/.vscode/*", "*.swp",

        # Sensitive files (expanded patterns)
        "*/.env*",  # Catches .env, .env.local, .env.production, .env.development, etc.
        "*/secrets.*", "*/credentials.*",
        "*.pem", "*.key", "*.crt", "*.p12",

        # Sensitive directories
        "*/.aws/*", "*/.ssh/*", "*/.gnupg/*",

        # Database files
        "*.db", "*.sqlite", "*.sqlite3",
    ]

    # Path mapper for host <-> container translation
    path_mapper: PathMapper | None = None

    def __init__(self, **kwargs):
        """Initialize settings and discover volume mounts."""
        super().__init__(**kwargs)

        # Initialize and discover path mappings
        self.path_mapper = PathMapper(self.read_write_base, self.read_only_base)
        self.path_mapper.discover_mounts()
        logger.info(f"Discovered {len(self.path_mapper.mappings)} volume mounts")

    def is_path_blocked(self, path: Path) -> tuple[bool, str | None]:
        """
        Check if path matches blocked patterns.

        Args:
            path: Path to check

        Returns:
            Tuple of (is_blocked, reason)
        """
        path_str = str(path)
        for pattern in self.write_blocked_patterns:
            if fnmatch(path_str, pattern):
                return True, f"Path matches protected pattern: {pattern}"
        return False, None

    def is_readable(self, path: Path) -> tuple[bool, str | None]:
        """
        Check if container path is in readable location.

        NOTE: This method expects container paths (e.g., /mnt/read_write/...),
        not host paths. Tools should translate host->container first.

        Args:
            path: Container path to check

        Returns:
            Tuple of (is_readable, error_message)
        """
        try:
            path = path.resolve()
        except (OSError, RuntimeError) as e:
            return False, f"Invalid path: {e}"

        # Check if under read-write or read-only base
        # Need to resolve base paths too for comparison
        try:
            read_write_resolved = self.read_write_base.resolve()
            read_only_resolved = self.read_only_base.resolve()

            if path.is_relative_to(read_write_resolved):
                return True, None
            if path.is_relative_to(read_only_resolved):
                return True, None
        except ValueError:
            pass

        return False, (
            f"Path not in readable locations. "
            f"Allowed: {self.read_write_base}, {self.read_only_base}"
        )

    def is_writable(self, path: Path) -> tuple[bool, str | None]:
        """
        Check if container path is writable.

        NOTE: This method expects container paths (e.g., /mnt/read_write/...),
        not host paths. Tools should translate host->container first.

        Writable paths must be:
          - Under /mnt/read_write/*
          - NOT matching blocked patterns (.git, node_modules, etc.)

        Args:
            path: Container path to check

        Returns:
            Tuple of (is_writable, error_message)
        """
        try:
            path = path.resolve()
        except (OSError, RuntimeError) as e:
            return False, f"Invalid path: {e}"

        # First check if blocked by pattern
        is_blocked, block_reason = self.is_path_blocked(path)
        if is_blocked:
            return False, f"Write blocked: {block_reason}"

        # Check if in writable location (/mnt/read_write/*)
        # Need to resolve base path for comparison
        try:
            read_write_resolved = self.read_write_base.resolve()
            if path.is_relative_to(read_write_resolved):
                if not path.parent.exists():
                    return False, f"Parent directory does not exist: {path.parent}"
                return True, None
        except ValueError:
            pass

        return False, f"Path not in writable location. Allowed: {self.read_write_base}"

    model_config = {
        "env_file": ".env",
        "extra": "ignore",  # Ignore extra fields from shared .env file
    }


settings = Settings()
