"""GitHub and development tools using shell commands."""

import asyncio
from typing import Any

from tools_api.services.base import BaseTool


class GetGitHubPullRequestInfoTool(BaseTool):
    """Get comprehensive information about a GitHub Pull Request."""

    @property
    def name(self) -> str:
        return "get_github_pull_request_info"

    @property
    def description(self) -> str:
        return "Get comprehensive information about a GitHub Pull Request including overview, files changed, and cumulative diff"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pr_url": {
                    "type": "string",
                    "description": "GitHub Pull Request URL (e.g., https://github.com/owner/repo/pull/123)",
                },
            },
            "required": ["pr_url"],
        }

    @property
    def tags(self) -> list[str]:
        return ["github", "git", "development"]

    async def _execute(self, pr_url: str) -> dict[str, Any]:
        """Get GitHub PR information using gh CLI."""
        # Build the command - use bash explicitly for regex support
        cmd = (
            '/bin/bash -c \''
            f'PR_URL="{pr_url}" && '
            'if [[ "$PR_URL" =~ github\\.com/([^/]+)/([^/]+)/pull/([0-9]+) ]]; then '
            'OWNER="${BASH_REMATCH[1]}" && '
            'REPO="${BASH_REMATCH[2]}" && '
            'PR_NUMBER="${BASH_REMATCH[3]}" && '
            'echo "=== PR Overview ===" && '
            'gh pr view "$PR_URL" && '
            'printf "\\n\\n=== Files Changed (Summary) ===\\n" && '
            'gh api "repos/$OWNER/$REPO/pulls/$PR_NUMBER/files" | '
            'jq -r ".[] | (.filename + \\" (+\\" + (.additions|tostring) + \\"/-\\" + (.deletions|tostring) + \\") [\\" + .status + \\"]\\") " && '
            'printf "\\n\\n=== File Changes ===\\n" && '
            'gh pr diff "$PR_URL"; '
            'else '
            'echo "Error: Invalid GitHub PR URL format. Expected: https://github.com/owner/repo/pull/NUMBER"; '
            'fi\''
        )

        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            output = stdout.decode()

            # For invalid URL, we get the error message in stdout, not an error code
            if "Error: Invalid GitHub PR URL" in output:
                return {
                    "output": output,
                    "pr_url": pr_url,
                    "success": True,
                }

            if process.returncode != 0:
                raise RuntimeError(
                    f"Command failed with exit code {process.returncode}: {stderr.decode()}",
                )

            return {
                "output": output,
                "pr_url": pr_url,
                "success": True,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get PR info: {e!s}")


class GetLocalGitChangesInfoTool(BaseTool):
    """Get comprehensive information about local Git changes."""

    @property
    def name(self) -> str:
        return "get_local_git_changes_info"

    @property
    def description(self) -> str:
        return "Get comprehensive information about local Git changes including overview and diffs"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Path to the Git repository directory",
                },
            },
            "required": ["directory"],
        }

    @property
    def tags(self) -> list[str]:
        return ["git", "development"]

    async def _execute(self, directory: str) -> dict[str, Any]:
        """Get local Git changes using git commands."""
        cmd = (
            f'if [ ! -d "{directory}" ]; then '
            f'echo "Error: Directory does not exist: {directory}"; exit 1; '
            'fi && '
            f'cd "{directory}" && '
            'if [ ! -d ".git" ]; then echo "Error: Not a Git repository"; exit 1; fi && '
            'echo "=== Git Status ===" && '
            'git status && '
            'echo "" && '
            'echo "=== Change Summary ===" && '
            'git diff --stat HEAD 2>/dev/null || echo "No changes to summarize" && '
            'echo "" && '
            'echo "=== Staged Changes ===" && '
            'if git diff --cached --quiet; then echo "No staged changes"; else git diff --cached; fi && '
            'echo "" && '
            'echo "=== Unstaged Changes ===" && '
            'if git diff --quiet; then echo "No unstaged changes"; else git diff; fi && '
            'echo "" && '
            'echo "=== Untracked Files ===" && '
            'TRACKED_FILES=$(mktemp) && '
            'git ls-files --others --exclude-standard > "$TRACKED_FILES" && '
            'if [ ! -s "$TRACKED_FILES" ]; then '
            '  echo "No untracked files"; '
            'else '
            '  while IFS= read -r file; do '
            '    if [ -f "$file" ]; then '
            '      FILE_SIZE=$(wc -c < "$file" 2>/dev/null || echo 0) && '
            '      case "$file" in '
            '        *.jpg|*.jpeg|*.png|*.gif|*.pdf|*.zip|*.tar|*.gz|*.exe|*.bin|*.so|*.pyc|*.class|*.o) '
            '          echo "Binary file: $file (${FILE_SIZE} bytes, skipped)" ;; '
            '        *) '
            '          if [ "$FILE_SIZE" -gt 102400 ]; then '
            '            echo "Large file: $file (${FILE_SIZE} bytes, >100KB, skipped)"; '
            '          else '
            '            echo "=== New file: $file (${FILE_SIZE} bytes) ===" && '
            '            cat "$file" 2>/dev/null; '
            '          fi ;; '
            '      esac; '
            '    else '
            '      echo "Directory/Special: $file"; '
            '    fi && '
            '    echo ""; '
            '  done < "$TRACKED_FILES"; '
            'fi && '
            'rm -f "$TRACKED_FILES"'
        )

        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            output_stdout = stdout.decode()
            output_stderr = stderr.decode()

            # Check for our explicit error messages
            if process.returncode != 0:
                # Check stdout for error messages
                if "Error:" in output_stdout:
                    raise RuntimeError(output_stdout.strip())
                raise RuntimeError(
                    f"Command failed with exit code {process.returncode}: {output_stderr}",
                )

            return {
                "output": output_stdout,
                "directory": directory,
                "success": True,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get git changes: {e!s}")


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
                    "description": "Directory to generate tree for",
                },
                "custom_excludes": {
                    "type": "string",
                    "description": "Additional patterns to exclude (pipe-separated, e.g., 'build|dist|target')",
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

    async def _execute(
        self,
        directory: str,
        custom_excludes: str = "",
        format_args: str = "",
    ) -> dict[str, Any]:
        """Generate directory tree using tree command."""
        # Build tree command with exclusions
        base_excludes = ".git|.claude|.env|.venv|env|node_modules|__pycache__|.DS_Store|*.pyc"

        cmd_parts = [
            "tree",
            f"'{directory}'",
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
                # tree command not available
                if "command not found" in stderr.decode().lower():
                    raise RuntimeError(
                        "tree command not found. Please install tree: "
                        "brew install tree (macOS) or apt-get install tree (Linux)",
                    )
                raise RuntimeError(
                    f"Command failed with exit code {process.returncode}: {stderr.decode()}",
                )

            output = stdout.decode()
            return {
                "output": output,
                "directory": directory,
                "success": True,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to generate directory tree: {e!s}")
