"""GitHub and development tools using shell commands."""

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from tools_api.config import settings
from tools_api.services.base import BaseTool

# Default timeout for shell commands (seconds)
SHELL_COMMAND_TIMEOUT = 60


class GetGitHubPullRequestInfoResult(BaseModel):
    """Result from getting GitHub PR information."""

    output: str = Field(description="The PR information output including overview, files changed, and diff")  # noqa: E501
    pr_url: str = Field(description="The GitHub PR URL that was queried")


class GetLocalGitChangesInfoResult(BaseModel):
    """Result from getting local Git changes."""

    output: str = Field(description="The Git changes output including status, staged, unstaged, and untracked changes")  # noqa: E501
    directory: str = Field(description="The directory that was queried")


class GetGitHubPullRequestInfoTool(BaseTool):
    """Get comprehensive information about a GitHub Pull Request."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "get_github_pull_request_info"

    @property
    def description(self) -> str:
        """Tool description."""
        return "Get comprehensive information about a GitHub Pull Request including overview, files changed, and cumulative diff"  # noqa: E501

    @property
    def parameters(self) -> dict[str, Any]:
        """Tool parameters JSON Schema."""
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
    def result_model(self) -> type[BaseModel]:
        """Pydantic model for the tool result."""
        return GetGitHubPullRequestInfoResult

    @property
    def tags(self) -> list[str]:
        """Tool semantic tags."""
        return ["github", "git", "development"]

    @property
    def category(self) -> str | None:
        """Tool category."""
        return "github"

    async def _execute(self, pr_url: str) -> GetGitHubPullRequestInfoResult:
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
            'jq -r ".[] | (.filename + \\" (+\\" + (.additions|tostring) + \\"/-\\" + (.deletions|tostring) + \\") [\\" + .status + \\"]\\") " && '  # noqa: E501
            'printf "\\n\\n=== File Changes ===\\n" && '
            'gh pr diff "$PR_URL"; '
            'else '
            'echo "Error: Invalid GitHub PR URL format. Expected: https://github.com/owner/repo/pull/NUMBER"; '  # noqa: E501
            'fi\''
        )

        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=SHELL_COMMAND_TIMEOUT,
            )

            output = stdout.decode(errors="replace")

            # For invalid URL, we get the error message in stdout, not an error code
            if "Error: Invalid GitHub PR URL" in output:
                return GetGitHubPullRequestInfoResult(
                    output=output,
                    pr_url=pr_url,
                )

            if process.returncode != 0:
                raise RuntimeError(
                    f"Command failed with exit code {process.returncode}: {stderr.decode(errors='replace')}",  # noqa: E501
                )

            return GetGitHubPullRequestInfoResult(
                output=output,
                pr_url=pr_url,
            )
        except TimeoutError:
            process.kill()
            raise RuntimeError(f"Command timed out after {SHELL_COMMAND_TIMEOUT}s")
        except Exception as e:
            raise RuntimeError(f"Failed to get PR info: {e!s}")


class GetLocalGitChangesInfoTool(BaseTool):
    """Get comprehensive information about local Git changes."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "get_local_git_changes_info"

    @property
    def description(self) -> str:
        """Tool description."""
        return (
            "Get comprehensive Git repository status including "
            "staged, unstaged, and untracked changes with diffs"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """Tool parameters JSON Schema."""
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
    def result_model(self) -> type[BaseModel]:
        """Pydantic model for the tool result."""
        return GetLocalGitChangesInfoResult

    @property
    def tags(self) -> list[str]:
        """Tool semantic tags."""
        return ["git", "development"]

    @property
    def category(self) -> str | None:
        """Tool category."""
        return "github"

    async def _execute(self, directory: str) -> GetLocalGitChangesInfoResult:
        """Get local Git changes using git commands."""
        # Translate host path to container path
        container_path, _ = settings.path_mapper.to_container_path(directory)
        container_path_str = str(container_path)

        cmd = (
            f'if [ ! -d "{container_path_str}" ]; then '
            f'echo "Error: Directory does not exist: {directory}"; exit 1; '
            'fi && '
            f'cd "{container_path_str}" && '
            'if [ ! -d ".git" ]; then echo "Error: Not a Git repository"; exit 1; fi && '
            'echo "=== Git Status ===" && '
            'git status && '
            'echo "" && '
            'echo "=== Change Summary ===" && '
            'git diff --stat HEAD 2>/dev/null || echo "No changes to summarize" && '
            'echo "" && '
            'echo "=== Staged Changes ===" && '
            'if git diff --cached --quiet; then echo "No staged changes"; else git diff --cached; fi && '  # noqa: E501
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
            '        *.jpg|*.jpeg|*.png|*.gif|*.pdf|*.zip|*.tar|*.gz|*.exe|*.bin|*.so|*.pyc|*.class|*.o) '  # noqa: E501
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
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=SHELL_COMMAND_TIMEOUT,
            )

            output_stdout = stdout.decode(errors="replace")
            output_stderr = stderr.decode(errors="replace")

            # Check for our explicit error messages
            if process.returncode != 0:
                # Check stdout for error messages
                if "Error:" in output_stdout:
                    raise RuntimeError(output_stdout.strip())
                raise RuntimeError(
                    f"Command failed with exit code {process.returncode}: {output_stderr}",
                )

            return GetLocalGitChangesInfoResult(
                output=output_stdout,
                directory=directory,
            )
        except TimeoutError:
            process.kill()
            raise RuntimeError(f"Command timed out after {SHELL_COMMAND_TIMEOUT}s")
        except Exception as e:
            raise RuntimeError(f"Failed to get git changes: {e!s}")
