"""Filesystem tools for file and directory operations."""

import asyncio
import shutil
from typing import Any

from tools_api.config import settings
from tools_api.services.base import BaseTool


class ReadTextFileTool(BaseTool):
    """Read text file with metadata."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "read_text_file"

    @property
    def description(self) -> str:
        """Tool description."""
        return "Read a text file and return its contents with metadata"

    @property
    def parameters(self) -> dict[str, Any]:
        """Tool parameters JSON Schema."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read",
                },
            },
            "required": ["path"],
        }

    @property
    def tags(self) -> list[str]:
        """Tool semantic tags."""
        return ["filesystem", "read"]

    @property
    def category(self) -> str | None:
        """Tool category."""
        return "filesystem"

    async def _execute(self, path: str) -> dict[str, Any]:
        """Read text file with metadata."""
        # Maximum file size: 50MB
        MAX_FILE_SIZE = 50 * 1024 * 1024

        # Translate host path to container path
        container_path, _ = settings.path_mapper.to_container_path(path)

        # Check readable (on container path)
        is_readable, error = settings.is_readable(container_path)
        if not is_readable:
            raise PermissionError(error)

        if not container_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not container_path.is_file():
            raise ValueError(f"Not a file: {path}")

        # Check file size before reading
        stats = container_path.stat()
        if stats.st_size > MAX_FILE_SIZE:
            size_mb = stats.st_size / (1024 * 1024)
            raise ValueError(
                f"File too large ({size_mb:.1f}MB). "
                f"Maximum file size is {MAX_FILE_SIZE / (1024 * 1024):.0f}MB. "
                f"Consider using a tool that supports partial reads or file streaming.",
            )

        # Read content
        content = container_path.read_text()

        # Translate back to host path for response
        host_path = settings.path_mapper.to_host_path(container_path)

        return {
            "content": content,
            "path": str(host_path),
            "size_bytes": stats.st_size,
            "line_count": len(content.splitlines()),
            "char_count": len(content),
            "modified_time": stats.st_mtime,
        }


class WriteFileTool(BaseTool):
    """Write content to a file."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "write_file"

    @property
    def description(self) -> str:
        """Tool description."""
        return "Write content to a file, creating it if it doesn't exist"

    @property
    def parameters(self) -> dict[str, Any]:
        """Tool parameters JSON Schema."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        }

    @property
    def tags(self) -> list[str]:
        """Tool semantic tags."""
        return ["filesystem", "write"]

    @property
    def category(self) -> str | None:
        """Tool category."""
        return "filesystem"

    async def _execute(self, path: str, content: str) -> dict[str, Any]:
        """Write content to file."""
        # Translate host path to container path
        container_path, _ = settings.path_mapper.to_container_path(path)

        # Check writable (on container path)
        is_writable, error = settings.is_writable(container_path)
        if not is_writable:
            raise PermissionError(error)

        # Write file
        container_path.write_text(content)
        stats = container_path.stat()

        # Translate back to host path for response
        host_path = settings.path_mapper.to_host_path(container_path)

        return {
            "path": str(host_path),
            "size_bytes": stats.st_size,
            "line_count": len(content.splitlines()),
            "success": True,
        }


class EditFileTool(BaseTool):
    """Edit an existing file by replacing text."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "edit_file"

    @property
    def description(self) -> str:
        """Tool description."""
        return "Edit a file by replacing old text with new text"

    @property
    def parameters(self) -> dict[str, Any]:
        """Tool parameters JSON Schema."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit",
                },
                "old_text": {
                    "type": "string",
                    "description": "Text to find and replace",
                },
                "new_text": {
                    "type": "string",
                    "description": "Text to replace with",
                },
            },
            "required": ["path", "old_text", "new_text"],
        }

    @property
    def tags(self) -> list[str]:
        """Tool semantic tags."""
        return ["filesystem", "write", "edit"]

    @property
    def category(self) -> str | None:
        """Tool category."""
        return "filesystem"

    async def _execute(self, path: str, old_text: str, new_text: str) -> dict[str, Any]:
        """Edit file by replacing text."""
        # Translate host path to container path
        container_path, _ = settings.path_mapper.to_container_path(path)

        # Check readable and writable (on container path)
        is_readable, read_error = settings.is_readable(container_path)
        if not is_readable:
            raise PermissionError(read_error)

        is_writable, write_error = settings.is_writable(container_path)
        if not is_writable:
            raise PermissionError(write_error)

        if not container_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Read, replace, write
        content = container_path.read_text()
        if old_text not in content:
            raise ValueError(f"Text not found in file: {old_text}")

        new_content = content.replace(old_text, new_text)
        container_path.write_text(new_content)
        stats = container_path.stat()

        # Translate back to host path for response
        host_path = settings.path_mapper.to_host_path(container_path)

        return {
            "path": str(host_path),
            "replacements": content.count(old_text),
            "size_bytes": stats.st_size,
            "success": True,
        }


class CreateDirectoryTool(BaseTool):
    """Create a directory."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "create_directory"

    @property
    def description(self) -> str:
        """Tool description."""
        return "Create a directory (and parent directories if needed)"

    @property
    def parameters(self) -> dict[str, Any]:
        """Tool parameters JSON Schema."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to create",
                },
            },
            "required": ["path"],
        }

    @property
    def tags(self) -> list[str]:
        """Tool semantic tags."""
        return ["filesystem", "write", "directory"]

    @property
    def category(self) -> str | None:
        """Tool category."""
        return "filesystem"

    async def _execute(self, path: str) -> dict[str, Any]:
        """Create directory."""
        # Translate host path to container path
        container_path, _ = settings.path_mapper.to_container_path(path)

        # For directory creation, check if path is in a writable location
        # But don't require parent to exist (since mkdir creates parents)
        try:
            resolved_path = container_path.resolve()
        except (OSError, RuntimeError) as e:
            raise PermissionError(f"Invalid path: {e}")

        # Check if blocked by pattern
        is_blocked, block_reason = settings.is_path_blocked(resolved_path)
        if is_blocked:
            raise PermissionError(f"Write blocked: {block_reason}")

        # Check if in writable location (under /mnt/read_write)
        try:
            if not resolved_path.is_relative_to(settings.read_write_base):
                raise PermissionError(
                    f"Path not in writable location. Allowed: {settings.read_write_base}",
                )
        except ValueError:
            raise PermissionError(
                f"Path not in writable location. Allowed: {settings.read_write_base}",
            )

        # Create directory
        container_path.mkdir(parents=True, exist_ok=True)

        # Translate back to host path for response
        host_path = settings.path_mapper.to_host_path(container_path)

        return {
            "path": str(host_path),
            "exists": container_path.exists(),
            "success": True,
        }


class ListDirectoryTool(BaseTool):
    """List directory contents."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "list_directory"

    @property
    def description(self) -> str:
        """Tool description."""
        return "List contents of a directory"

    @property
    def parameters(self) -> dict[str, Any]:
        """Tool parameters JSON Schema."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list",
                },
            },
            "required": ["path"],
        }

    @property
    def tags(self) -> list[str]:
        """Tool semantic tags."""
        return ["filesystem", "read", "directory"]

    @property
    def category(self) -> str | None:
        """Tool category."""
        return "filesystem"

    async def _execute(self, path: str) -> dict[str, Any]:
        """List directory contents."""
        # Translate host path to container path
        container_path, _ = settings.path_mapper.to_container_path(path)

        # Check readable
        is_readable, error = settings.is_readable(container_path)
        if not is_readable:
            raise PermissionError(error)

        if not container_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        if not container_path.is_dir():
            raise ValueError(f"Not a directory: {path}")

        # List contents
        entries = []
        for entry in sorted(container_path.iterdir()):
            # Translate each entry back to host path
            entry_host = settings.path_mapper.to_host_path(entry)
            entries.append({
                "name": entry.name,
                "path": str(entry_host),
                "is_file": entry.is_file(),
                "is_dir": entry.is_dir(),
            })

        host_path = settings.path_mapper.to_host_path(container_path)
        return {
            "path": str(host_path),
            "entries": entries,
            "count": len(entries),
        }


class ListDirectoryWithSizesTool(BaseTool):
    """List directory contents with file sizes."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "list_directory_with_sizes"

    @property
    def description(self) -> str:
        """Tool description."""
        return "List directory contents with file sizes and metadata"

    @property
    def parameters(self) -> dict[str, Any]:
        """Tool parameters JSON Schema."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list",
                },
            },
            "required": ["path"],
        }

    @property
    def tags(self) -> list[str]:
        """Tool semantic tags."""
        return ["filesystem", "read", "directory"]

    @property
    def category(self) -> str | None:
        """Tool category."""
        return "filesystem"

    async def _execute(self, path: str) -> dict[str, Any]:
        """List directory contents with sizes."""
        # Translate host path to container path
        container_path, _ = settings.path_mapper.to_container_path(path)

        # Check readable
        is_readable, error = settings.is_readable(container_path)
        if not is_readable:
            raise PermissionError(error)

        if not container_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        if not container_path.is_dir():
            raise ValueError(f"Not a directory: {path}")

        # List contents with metadata
        entries = []
        for entry in sorted(container_path.iterdir()):
            try:
                stats = entry.stat()
                # Translate each entry back to host path
                entry_host = settings.path_mapper.to_host_path(entry)
                entries.append({
                    "name": entry.name,
                    "path": str(entry_host),
                    "is_file": entry.is_file(),
                    "is_dir": entry.is_dir(),
                    "size_bytes": stats.st_size if entry.is_file() else 0,
                    "modified_time": stats.st_mtime,
                })
            except OSError:
                # Skip entries we can't stat
                continue

        host_path = settings.path_mapper.to_host_path(container_path)
        return {
            "path": str(host_path),
            "entries": entries,
            "count": len(entries),
            "total_size_bytes": sum(e["size_bytes"] for e in entries),
        }


class SearchFilesTool(BaseTool):
    """Search for files matching a pattern."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "search_files"

    @property
    def description(self) -> str:
        """Tool description."""
        return "Search for files matching a glob pattern in a directory tree"

    @property
    def parameters(self) -> dict[str, Any]:
        """Tool parameters JSON Schema."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Root directory to search from",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match (e.g., '*.py', '**/*.txt')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 100,
                },
            },
            "required": ["path", "pattern"],
        }

    @property
    def tags(self) -> list[str]:
        """Tool semantic tags."""
        return ["filesystem", "read", "search"]

    @property
    def category(self) -> str | None:
        """Tool category."""
        return "filesystem"

    async def _execute(self, path: str, pattern: str, max_results: int = 100) -> dict[str, Any]:
        """Search for files matching pattern."""
        # Translate host path to container path
        container_path, _ = settings.path_mapper.to_container_path(path)

        # Check readable
        is_readable, error = settings.is_readable(container_path)
        if not is_readable:
            raise PermissionError(error)

        if not container_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        if not container_path.is_dir():
            raise ValueError(f"Not a directory: {path}")

        # Search for files
        matches = []
        for entry in container_path.rglob(pattern):
            if len(matches) >= max_results:
                break
            if entry.is_file():
                # Translate each match back to host path
                entry_host = settings.path_mapper.to_host_path(entry)
                matches.append({
                    "name": entry.name,
                    "path": str(entry_host),
                    "relative_path": str(entry.relative_to(container_path)),
                    "size_bytes": entry.stat().st_size,
                })

        host_path = settings.path_mapper.to_host_path(container_path)
        return {
            "search_path": str(host_path),
            "pattern": pattern,
            "matches": matches,
            "count": len(matches),
            "truncated": len(matches) >= max_results,
        }


class GetFileInfoTool(BaseTool):
    """Get file metadata and information."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "get_file_info"

    @property
    def description(self) -> str:
        """Tool description."""
        return "Get metadata and information about a file or directory"

    @property
    def parameters(self) -> dict[str, Any]:
        """Tool parameters JSON Schema."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file or directory",
                },
            },
            "required": ["path"],
        }

    @property
    def tags(self) -> list[str]:
        """Tool semantic tags."""
        return ["filesystem", "read"]

    @property
    def category(self) -> str | None:
        """Tool category."""
        return "filesystem"

    async def _execute(self, path: str) -> dict[str, Any]:
        """Get file information."""
        # Translate host path to container path
        container_path, _ = settings.path_mapper.to_container_path(path)

        # Check readable
        is_readable, error = settings.is_readable(container_path)
        if not is_readable:
            raise PermissionError(error)

        if not container_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        stats = container_path.stat()

        # Translate back to host path for response
        host_path = settings.path_mapper.to_host_path(container_path)

        return {
            "path": str(host_path),
            "name": container_path.name,
            "is_file": container_path.is_file(),
            "is_dir": container_path.is_dir(),
            "is_symlink": container_path.is_symlink(),
            "size_bytes": stats.st_size,
            "created_time": stats.st_ctime,
            "modified_time": stats.st_mtime,
            "accessed_time": stats.st_atime,
            "permissions": oct(stats.st_mode)[-3:],
        }


class ListAllowedDirectoriesTool(BaseTool):
    """List directories the tool can access."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "list_allowed_directories"

    @property
    def description(self) -> str:
        """Tool description."""
        return "List all directories that tools can read from and write to"

    @property
    def parameters(self) -> dict[str, Any]:
        """Tool parameters JSON Schema."""
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    @property
    def tags(self) -> list[str]:
        """Tool semantic tags."""
        return ["filesystem", "read", "meta"]

    @property
    def category(self) -> str | None:
        """Tool category."""
        return "filesystem"

    async def _execute(self) -> dict[str, Any]:
        """List allowed directories by scanning mounted volumes."""
        directories = []

        # Scan /mnt/read_write/* (read-write volumes)
        if settings.read_write_base.exists():
            for child in settings.read_write_base.iterdir():
                if child.is_dir():
                    # Translate container path to host path
                    host_path = settings.path_mapper.to_host_path(child)
                    directories.append({
                        "path": str(host_path),
                        "access": "read-write",
                        "exists": True,
                    })

        # Scan /mnt/read_only/* (read-only volumes)
        if settings.read_only_base.exists():
            for child in settings.read_only_base.iterdir():
                if child.is_dir():
                    # Translate container path to host path
                    host_path = settings.path_mapper.to_host_path(child)
                    directories.append({
                        "path": str(host_path),
                        "access": "read-only",
                        "exists": True,
                    })

        return {
            "directories": directories,
            "read_write_base": str(settings.read_write_base),
            "read_only_base": str(settings.read_only_base),
            "total_count": len(directories),
            "blocked_patterns": settings.write_blocked_patterns,
        }


class MoveFileTool(BaseTool):
    """Move or rename a file or directory."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "move_file"

    @property
    def description(self) -> str:
        """Tool description."""
        return "Move or rename a file or directory"

    @property
    def parameters(self) -> dict[str, Any]:
        """Tool parameters JSON Schema."""
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Source path",
                },
                "destination": {
                    "type": "string",
                    "description": "Destination path",
                },
            },
            "required": ["source", "destination"],
        }

    @property
    def tags(self) -> list[str]:
        """Tool semantic tags."""
        return ["filesystem", "write", "move"]

    @property
    def category(self) -> str | None:
        """Tool category."""
        return "filesystem"

    async def _execute(self, source: str, destination: str) -> dict[str, Any]:
        """Move file or directory."""
        # Translate BOTH host paths to container paths
        source_container, _ = settings.path_mapper.to_container_path(source)
        dest_container, _ = settings.path_mapper.to_container_path(destination)

        # Check source is readable
        is_readable, read_error = settings.is_readable(source_container)
        if not is_readable:
            raise PermissionError(f"Source: {read_error}")

        # Check both paths are writable
        is_src_writable, src_error = settings.is_writable(source_container)
        if not is_src_writable:
            raise PermissionError(f"Source: {src_error}")

        is_dest_writable, dest_error = settings.is_writable(dest_container)
        if not is_dest_writable:
            raise PermissionError(f"Destination: {dest_error}")

        if not source_container.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        if dest_container.exists():
            raise FileExistsError(f"Destination already exists: {destination}")

        # Move
        shutil.move(str(source_container), str(dest_container))

        # Translate BOTH paths back to host paths for response
        source_host = settings.path_mapper.to_host_path(source_container)
        dest_host = settings.path_mapper.to_host_path(dest_container)

        return {
            "source": str(source_host),
            "destination": str(dest_host),
            "success": True,
        }


class DeleteFileTool(BaseTool):
    """Delete a file."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "delete_file"

    @property
    def description(self) -> str:
        """Tool description."""
        return "Delete a file"

    @property
    def parameters(self) -> dict[str, Any]:
        """Tool parameters JSON Schema."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to delete",
                },
            },
            "required": ["path"],
        }

    @property
    def tags(self) -> list[str]:
        """Tool semantic tags."""
        return ["filesystem", "write", "delete"]

    @property
    def category(self) -> str | None:
        """Tool category."""
        return "filesystem"

    async def _execute(self, path: str) -> dict[str, Any]:
        """Delete file."""
        # Translate host path to container path
        container_path, _ = settings.path_mapper.to_container_path(path)

        # Check writable
        is_writable, error = settings.is_writable(container_path)
        if not is_writable:
            raise PermissionError(error)

        if not container_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not container_path.is_file():
            raise ValueError(f"Not a file: {path}")

        # Delete
        container_path.unlink()

        # Translate back to host path for response
        host_path = settings.path_mapper.to_host_path(container_path)

        return {
            "path": str(host_path),
            "deleted": True,
            "success": True,
        }


class DeleteDirectoryTool(BaseTool):
    """Delete a directory."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "delete_directory"

    @property
    def description(self) -> str:
        """Tool description."""
        return "Delete a directory and all its contents"

    @property
    def parameters(self) -> dict[str, Any]:
        """Tool parameters JSON Schema."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to delete",
                },
            },
            "required": ["path"],
        }

    @property
    def tags(self) -> list[str]:
        """Tool semantic tags."""
        return ["filesystem", "write", "delete", "directory"]

    @property
    def category(self) -> str | None:
        """Tool category."""
        return "filesystem"

    async def _execute(self, path: str) -> dict[str, Any]:
        """Delete directory."""
        # Translate host path to container path
        container_path, _ = settings.path_mapper.to_container_path(path)

        # Check writable
        is_writable, error = settings.is_writable(container_path)
        if not is_writable:
            raise PermissionError(error)

        if not container_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        if not container_path.is_dir():
            raise ValueError(f"Not a directory: {path}")

        # Delete recursively
        shutil.rmtree(container_path)

        # Translate back to host path for response
        host_path = settings.path_mapper.to_host_path(container_path)

        return {
            "path": str(host_path),
            "deleted": True,
            "success": True,
        }


class GetDirectoryTreeTool(BaseTool):
    """Generate a directory tree with standard exclusions and gitignore support."""

    @property
    def name(self) -> str:
        return "get_directory_tree"

    @property
    def description(self) -> str:
        return "Generate a directory tree with standard exclusions and gitignore support"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory to generate tree for (quotes are handled automatically in the command)",  # noqa: E501
                },
                "custom_excludes": {
                    "type": "string",
                    "description": "Additional patterns to exclude (pipe-separated, e.g., 'build|dist|target')",  # noqa: E501
                    "default": "",
                },
                "format_args": {
                    "type": "string",
                    "description": "Additional tree command options (e.g., '-L 3 -C --dirsfirst')",
                    "default": "",
                },
            },
            "required": ["directory"],
        }

    @property
    def tags(self) -> list[str]:
        return ["filesystem", "development", "tree"]

    @property
    def category(self) -> str | None:
        """Tool category."""
        return "filesystem"

    async def _execute(
        self,
        directory: str,
        custom_excludes: str = "",
        format_args: str = "",
    ) -> dict[str, Any]:
        """Generate directory tree using tree command."""
        # Translate host path to container path
        container_path, _ = settings.path_mapper.to_container_path(directory)
        container_path_str = str(container_path)

        # Build tree command with exclusions
        base_excludes = ".git|.claude|.env|.venv|env|node_modules|__pycache__|.DS_Store|*.pyc"

        cmd_parts = [
            "tree",
            f"'{container_path_str}'",
            "-a",
            "--gitignore",
            f"-I \"{base_excludes}\"",
        ]

        if custom_excludes:
            cmd_parts.append(f'-I "{custom_excludes}"')

        if format_args:
            cmd_parts.append(format_args)

        cmd = " ".join(cmd_parts)

        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                stderr_str = stderr.decode()
                stdout_str = stdout.decode()

                # tree command not available
                if "command not found" in stderr_str.lower():
                    raise RuntimeError(
                        "tree command not found. Please install tree: "
                        "brew install tree (macOS) or apt-get install tree (Linux)",
                    )

                # Provide detailed error message
                error_msg = f"Command failed with exit code {process.returncode}"
                if stderr_str:
                    error_msg += f": {stderr_str}"
                if stdout_str and "error opening dir" in stdout_str.lower():
                    error_msg += f". Tree output: {stdout_str[:200]}"

                raise RuntimeError(error_msg)

            output = stdout.decode()
            return {
                "output": output,
                "directory": directory,
                "success": True,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to generate directory tree: {e!s}")
