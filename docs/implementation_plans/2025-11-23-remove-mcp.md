# Implementation Plan: MCP to Custom Tools API Migration

## Overview

**Goal**: Replace MCP architecture with a custom REST API for tools and prompts, prioritizing clean design and structured responses.

**Context**: 

### Current Architecture: MCP with Bridge

The current system uses Model Context Protocol (MCP) to integrate tools and prompts into the reasoning agent. This architecture evolved from a need to quickly integrate existing MCP servers (filesystem, brave-search, github tools) into a Dockerized API.

**The Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│                    Host Machine                          │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │ reasoning-api (Docker Container)                 │   │
│  │ - Uses api/mcp.py for MCP client integration    │   │
│  │ - Loads config/mcp_servers.json                  │   │
│  │ - Converts MCP tools to Tool objects             │   │
│  └───────────────────┬──────────────────────────────┘   │
│                      │ HTTP (host.docker.internal:9000)  │
│                      ▼                                   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ mcp_bridge (Python Process on Host)              │   │
│  │ - Runs via `make mcp_bridge`                     │   │
│  │ - Loads config/mcp_bridge_config.json            │   │
│  │ - Spawns stdio MCP servers as subprocesses       │   │
│  │ - Exposes stdio servers as HTTP endpoints        │   │
│  └──────────────┬───────────────────────────────────┘   │
│                 │ stdio (pipes)                          │
│                 ▼                                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │ MCP Servers (stdio processes)                    │   │
│  │ - filesystem (npx @modelcontextprotocol/...)     │   │
│  │ - github-custom (uvx mcp-this --preset github)   │   │
│  │ - brave-search (npx ...)                         │   │
│  │ - meta/thinking/dev prompts (uvx mcp-this)       │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Configuration Complexity

The system requires **three separate configuration files**, each serving a different purpose:

**1. `config/mcp_servers.json` (API → Bridge connection):**
```json
{
  "mcpServers": {
    "local_bridge": {
      "url": "http://localhost:9000/mcp/",
      "enabled": true
    }
  }
}
```
This tells the reasoning-api (in Docker) where to find the bridge server (on host).

**2. `config/mcp_bridge_config.json` (Bridge → stdio servers):**
```json
{
  "mcpServers": {
    "github-custom": {
      "command": "uvx",
      "args": ["mcp-this", "--preset", "github"],
      "enabled": true
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/you/repos"],
      "enabled": true
    }
    // ... 7 more servers
  }
}
```
This tells the bridge which stdio MCP servers to spawn and how to run them.

**3. `config/mcp_overrides.yaml` (Name cleaning):**
```yaml
server_tags:
  github-custom: [git, github]

tools:
  github_custom__get_github_pull_request_info:
    name: get_github_pull_request_info
  filesystem__read_text_file:
    name: read_text_file
  # ... 30+ tool overrides
  
prompts:
  # ... 20+ prompt overrides
```
This converts ugly MCP names (`local_bridge_github_custom__get_pr_info`) into clean names (`get_github_pull_request_info`) for LLM consumption, and adds semantic tags.

### Why This Complexity Exists

**1. Docker Isolation Problem:**
The reasoning-api runs in Docker but many MCP servers (especially filesystem) need access to the host's filesystem at paths like `/Users/shanekercheval/repos`. Docker containers can't directly access arbitrary host paths without explicit volume mounts.

**2. MCP Protocol Limitation:**
Most useful MCP servers (filesystem, github tools) only support **stdio transport** (stdin/stdout pipes), not HTTP. These can't be called directly over the network from a Docker container.

**3. The Bridge Solution:**
The `mcp_bridge` was created to solve this: run stdio servers on the host machine (with native filesystem access) and expose them via HTTP so the Dockerized API can reach them. It's essentially a custom HTTP-to-stdio proxy.

**4. Name Parsing Complexity:**
MCP servers prefix tool names with server identifiers. After bridging through multiple layers:
- Original: `get_pr_info`
- After bridge: `github_custom__get_pr_info`  
- After API sees it: `local_bridge_github_custom__get_pr_info`

The `api/mcp.py` module contains 200+ lines of name parsing logic (`parse_mcp_name()`, `NamingConfig`, etc.) to clean these names and make them LLM-friendly.

### The Text-Only Response Problem

MCP tools return **text blobs** because the protocol is designed for LLMs:

```python
# MCP result (text):
result = await client.call_tool("read_file", {"path": "/foo/bar.py"})
# result.content[0].text = "def foo():\n    pass\n# 150 lines...\n"

# What you actually want (structured):
{
  "content": "def foo():\n    pass...",
  "path": "/foo/bar.py",
  "size_bytes": 4523,
  "modified_time": 1699564800,
  "line_count": 150,
  "encoding": "utf-8"
}
```

**Why this matters:**

1. **No caching:** You can't cache individual file contents efficiently. The cache key would be `tool_name + path`, but you only get back text, so you can't extract metadata for smarter invalidation.

2. **No reuse:** If you need to process the file (e.g., count imports, check file size before reading), you must parse the text response or call the tool again.

3. **No metadata:** File size, modification time, permissions—all lost. If you need these, you must make separate tool calls.

4. **Serialization overhead:** Everything converted to text (JSON → text → JSON), even for structured data like search results or git status.

### Operational Complexity

**Starting the system requires:**
1. Start Docker services: `docker compose up`
2. Start bridge server: `make mcp_bridge` (separate terminal, must stay running)
3. If bridge dies → all filesystem/github tools fail
4. Bridge logs mixed with application logs
5. Debugging requires checking: API logs, bridge logs, and MCP server stderr

**Developer experience issues:**
- New developer: "Why do I need to run `make mcp_bridge`? What is this?"
- Bridge crashes silently → tools fail with cryptic errors
- Can't debug tool execution without diving into MCP protocol
- Adding new tool requires editing 2-3 config files + name override

### When MCP Made Sense (Initially)

MCP was chosen for valid reasons:
- ✅ **Quick integration:** Existing ecosystem (filesystem, brave-search servers)
- ✅ **Standardized protocol:** Well-documented tool/prompt discovery
- ✅ **Community support:** Don't reinvent filesystem operations
- ✅ **FastMCP library:** Easy to build custom servers

This worked well for prototyping and getting the reasoning agent running quickly.

### Why We're Changing Now

The application has evolved beyond MCP's sweet spot:

**1. We don't need the ecosystem:**
- Brave Search has a direct API (simple HTTP calls)
- GitHub tools can use gh
- Filesystem operations can be 20 lines of Python (pathlib)
- No community servers we're using that we can't easily replace

**2. We need structured responses:**
- Caching tool results (especially file reads during conversations)
- Extracting metadata without re-parsing text
- Reusing structured data in application logic
- Better error handling with typed responses

**3. Architectural mismatch:**
- Building our own app, not integrating tools into third-party app (Claude Desktop)
- Need full control over tool lifecycle, caching, retries
- Want simple Docker Compose deployment (no "run this script separately")
- Performance matters (text serialization overhead adds up)

**4. Maintenance burden:**
- 3 config files to keep in sync
- 200+ lines of name parsing logic
- Bridge server that must stay running
- Complex debugging across multiple processes
- Hard to onboard new developers

### What We're Gaining

**New Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│                    Docker Compose                        │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │ reasoning-api (Container)                        │   │
│  │ - HTTP client to tools-api                       │   │
│  │ - Caching layer for tool responses               │   │
│  │ - Simple Tool abstraction                        │   │
│  └───────────────────┬──────────────────────────────┘   │
│                      │ HTTP (tools-api:8001)             │
│                      ▼                                   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ tools-api (Container)                            │   │
│  │ - FastAPI REST endpoints                         │   │
│  │ - Direct tool implementations (pathlib, httpx)   │   │
│  │ - Volume mounts: /mnt/repos, /mnt/downloads      │   │
│  │ - Returns structured JSON                        │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Benefits:**
1. **Single config:** Just Docker Compose volume mounts
2. **Structured responses:** JSON with metadata, ready for caching
3. **Simple deployment:** `docker compose up` (no separate bridge)
4. **Direct implementations:** 30 lines of Python instead of MCP protocol
5. **Better testing:** Mock HTTP endpoints instead of MCP servers
6. **Better debugging:** Standard HTTP logs, no protocol translation
7. **Easier onboarding:** Standard REST API, no MCP knowledge required

**What we remove:**
- `mcp_bridge/` directory (500+ lines)
- `api/mcp.py` (300+ lines of protocol/parsing logic)
- 3 configuration files
- Bridge server operational complexity
- Text serialization overhead
- Name parsing complexity

**Architecture Change:**
```
BEFORE:
reasoning-api (Docker) → mcp_bridge (host) → stdio MCP servers

AFTER:
reasoning-api (Docker) → tools-api (Docker) → direct implementations
                            ↓
                      volume mounts (/repos, /downloads)
```

**Key Principles:**
- No backwards compatibility required
- Structured responses over text blobs
- Clean abstractions over quick fixes
- Direct API clients over protocol layers
- Volume mounts over host networking

---

## Milestone 1: Create Tools API Service Scaffold ✅ COMPLETED

### Status
**COMPLETED** - 2025-11-23

The tools-api service has been created and integrated into the project. The service structure follows Python best practices with proper package organization.

### Success Criteria
- [x] `tools-api/` directory with FastAPI application
- [x] Docker Compose integration with volume mounts
- [x] Environment-based configuration (path configuration)
- [x] Path security configuration (blocked patterns for protected files)
- [x] API key configuration (GitHub, Brave Search)
- [x] Health check endpoint returns 200 OK
- [x] Service accessible from reasoning-api container
- [x] Tests passing for health check and path validation
- [x] Documentation in `tools-api/README.md`
- [x] Project structure reorganized with service-specific directories

### Actual Implementation

**Service structure created:**
```
tools-api/
├── tools_api/           # Package directory (flat layout)
│   ├── __init__.py
│   ├── main.py          # FastAPI app
│   ├── config.py        # Settings (base paths, path security)
│   ├── models.py        # Response models
│   ├── path_mapper.py   # Path translation (container ↔ host)
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── tools.py     # Tool endpoints
│   │   └── prompts.py   # Prompt endpoints
│   └── services/
│       ├── __init__.py
│       ├── base.py      # Base tool/prompt classes
│       ├── registry.py  # Tool/prompt registries
│       ├── tools/       # Tool implementations
│       │   ├── example_tool.py
│       │   ├── filesystem.py
│       │   ├── github_dev_tools.py
│       │   └── web_search_tool.py
│       ├── prompts/
│       │   └── example_prompt.py
│       └── web_search.py # Brave Search API client
├── tests/
│   ├── unit_tests/      # Fast tests, no external deps
│   └── integration_tests/  # Real HTTP tests
├── Dockerfile
└── README.md
```

**Additional changes made:**
- **Project reorganization:** Restructured entire repository with service-specific directories
  - `reasoning_api/reasoning_api/` - Reasoning agent package
  - `reasoning_api/tests/` - All reasoning agent tests
  - `reasoning_api/prompts/` - Prompt templates
  - `reasoning_api/alembic/` - Database migrations
  - `tools_api/tools_api/` - Tools API package
  - `tools_api/tests/` - All tools API tests

**Configuration updates:**
- **Makefile:** Added service-specific test targets (`reasoning_tests`, `tools_tests`)
- **docker-compose.yml:** Tools API service with volume mounts configured
- **pyproject.toml:** Flat layout configuration (no src/ directories)
- All imports updated from `api.*` to `reasoning_api.*`
- Test imports updated from `reasoning_api.tests.*` to `tests.*`

**Test results:**
- ✅ Reasoning API unit tests: 493/493 passing
- ✅ Reasoning API integration tests: 212/220 passing (8 MCP bridge failures expected)
- ✅ Tools API tests: 15/16 passing (1 skipped)

### Implementation Notes

The actual implementation includes several components that were built out beyond the basic scaffold:

1. **Path Security (`tools_api/config.py`):**
   - Read-write and read-only base paths configured
   - Blocked patterns for protected files (`.git/`, `node_modules/`, `.env`, etc.)
   - `is_readable()` and `is_writable()` validation methods
   - Path mapper for container ↔ host path translation

2. **Base Abstractions (`tools_api/services/base.py`):**
   - `BaseTool` abstract class with execute pattern
   - `BasePrompt` abstract class for prompt rendering
   - Automatic error handling and timing in base classes

3. **Registry Pattern (`tools_api/services/registry.py`):**
   - `ToolRegistry` for tool discovery
   - `PromptRegistry` for prompt discovery
   - Used by routers for listing and executing tools/prompts

4. **Tool Implementations:**
   - **Filesystem tools:** Complete implementation with structured responses
   - **GitHub tools:** Using subprocess and GitHub API
   - **Web search:** Brave Search API client (555 lines, production-ready)
   - **Example tool:** For testing patterns

5. **Docker Integration:**
   - Volume mounts for repos and downloads configured
   - Health check endpoint working
   - Service accessible from reasoning-api container
   - Network isolation via reasoning-network

**Key Design Decision:** Used **flat layout** instead of src/ subdirectories:
```
tools_api/
  tools_api/      # Package (what you import)
  tests/          # Tests (sibling to package)
```

This is simpler for Docker applications and avoids the complexity of src/ layout while maintaining proper package structure.

**3. Docker Compose integration:**
```yaml
# docker-compose.yml (add to existing)
services:
  tools-api:
    build: ./tools-api
    container_name: tools-api
    ports:
      - "8001:8001"
    volumes:
      # Read-write for code development (agent edits files, runs tests)
      - ${REPOS_PATH}:/mnt/repos:rw

      # Read-only for downloads (no need to edit)
      - ${DOWNLOADS_PATH}:/mnt/downloads:ro

      # Read-only for playbooks (YAML prompts)
      - ${PLAYBOOKS_PATH}:/mnt/playbooks:ro

      # Optional workspace for agent-generated artifacts
      - ./workspace:/mnt/workspace:rw
    environment:
      - BASE_REPOS_PATH=/mnt/repos
      - BASE_DOWNLOADS_PATH=/mnt/downloads
      - WORKSPACE_PATH=/mnt/workspace

      # API keys for tools
      - GITHUB_TOKEN=${GITHUB_TOKEN}
      - BRAVE_API_KEY=${BRAVE_API_KEY}
    networks:
      - reasoning-network
    restart: unless-stopped

  reasoning-api:
    # ... existing config ...
    environment:
      # Add tools-api URL
      - TOOLS_API_URL=http://tools-api:8001
```

**4. Configuration with path security:**
```python
# tools-api/config.py
from pydantic_settings import BaseSettings
from pathlib import Path
from fnmatch import fnmatch

class Settings(BaseSettings):
    # Mounted paths
    base_repos_path: Path = Path("/mnt/repos")
    base_downloads_path: Path = Path("/mnt/downloads")
    workspace_path: Path = Path("/mnt/workspace")

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

        # Sensitive files
        "*/.env", "*/.env.local", "*/secrets.yaml", "*/credentials.json",
    ]

    # Readable and writable locations
    readable_paths: list[Path] = None
    writable_paths: list[Path] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Define readable locations
        self.readable_paths = [
            self.base_repos_path,
            self.base_downloads_path,
            self.workspace_path,
        ]

        # Define writable locations (repos + workspace, NOT downloads)
        self.writable_paths = [
            self.base_repos_path,
            self.workspace_path,
        ]

    def is_path_blocked(self, path: Path) -> tuple[bool, str | None]:
        """Check if path matches blocked patterns."""
        path_str = str(path)
        for pattern in self.write_blocked_patterns:
            if fnmatch(path_str, pattern):
                return True, f"Path matches protected pattern: {pattern}"
        return False, None

    def is_readable(self, path: Path) -> tuple[bool, str | None]:
        """Check if path is in readable location."""
        try:
            path = path.resolve()
        except (OSError, RuntimeError) as e:
            return False, f"Invalid path: {e}"

        for allowed_path in self.readable_paths:
            try:
                if path.is_relative_to(allowed_path):
                    return True, None
            except ValueError:
                continue

        allowed = ", ".join(str(p) for p in self.readable_paths)
        return False, f"Path not in readable locations. Allowed: {allowed}"

    def is_writable(self, path: Path) -> tuple[bool, str | None]:
        """Check if path is writable (in writable location AND not blocked)."""
        try:
            path = path.resolve()
        except (OSError, RuntimeError) as e:
            return False, f"Invalid path: {e}"

        # First check if blocked by pattern
        is_blocked, block_reason = self.is_path_blocked(path)
        if is_blocked:
            return False, f"Write blocked: {block_reason}"

        # Check if in writable location
        for writable_path in self.writable_paths:
            try:
                if path.is_relative_to(writable_path):
                    if not path.parent.exists():
                        return False, f"Parent directory does not exist: {path.parent}"
                    return True, None
            except ValueError:
                continue

        allowed = ", ".join(str(p) for p in self.writable_paths)
        return False, f"Path not in writable locations. Allowed: {allowed}"

    class Config:
        env_file = ".env"

settings = Settings()
```

**5. Environment configuration file:**
```bash
# tools-api/.env.example
# ===========================================
# Path Configuration
# ===========================================
REPOS_PATH=/Users/yourname/repos
DOWNLOADS_PATH=/Users/yourname/Downloads
PLAYBOOKS_PATH=/Users/yourname/repos/playbooks

# ===========================================
# API Keys
# ===========================================
GITHUB_TOKEN=ghp_your_token_here
BRAVE_API_KEY=your_brave_api_key_here
```

### Testing Strategy

**What to test:**
- Health check returns correct response
- Docker container starts successfully
- Volume mounts are accessible (test file read from /mnt/repos)
- Service is reachable from reasoning-api container
- Path security validation (blocked patterns, writable checks)
- Environment configuration loading

**Example tests:**
```python
# tools-api/tests/test_health.py
import pytest
from httpx import AsyncClient
from main import app

@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_volume_mounts():
    """Verify volume mounts are accessible"""
    from config import settings
    assert settings.base_repos_path.exists()
    assert settings.base_repos_path.is_dir()

# tools-api/tests/test_path_security.py
from pathlib import Path
from config import settings

def test_blocked_patterns():
    """Test that protected patterns are blocked."""
    # .git directory should be blocked
    is_writable, error = settings.is_writable(Path("/mnt/repos/project/.git/config"))
    assert not is_writable
    assert "blocked" in error.lower()

    # node_modules should be blocked
    is_writable, error = settings.is_writable(Path("/mnt/repos/project/node_modules/package.json"))
    assert not is_writable

    # .env files should be blocked
    is_writable, error = settings.is_writable(Path("/mnt/repos/project/.env"))
    assert not is_writable

def test_normal_files_writable():
    """Test that normal files in repos are writable."""
    is_writable, error = settings.is_writable(Path("/mnt/repos/project/src/main.py"))
    # Should be writable if parent exists (may fail if parent doesn't exist)
    assert is_writable or "parent" in error.lower()

def test_downloads_not_writable():
    """Test that downloads directory is read-only."""
    is_readable, _ = settings.is_readable(Path("/mnt/downloads/file.txt"))
    assert is_readable

    is_writable, error = settings.is_writable(Path("/mnt/downloads/file.txt"))
    assert not is_writable
    assert "not in writable locations" in error.lower()
```

### Lessons Learned

1. **Flat Layout Works Well:** The flat layout (package at top level, not in src/) is simpler for Docker applications and causes fewer import issues.

2. **Path Security is Critical:** The path mapper and security validation prevent accidentally modifying protected files like `.git/` or `node_modules/`.

3. **Test Organization:** Separating unit and integration tests makes it easier to run fast tests during development.

4. **Import Path Updates:** Moving from `api.*` to `reasoning_api.*` required updating ~70 files, including string references in `patch()` calls.

5. **Alembic Configuration:** Migration commands need explicit `-c` flag to specify config file location after reorganization.

---

## Milestone 2: Define Tool/Prompt Abstractions ✅ COMPLETED

### Status
**COMPLETED** - 2025-11-23

Base abstractions, response models, and registry pattern implemented as part of Milestone 1.

### Success Criteria
- [x] `Tool` and `Prompt` base classes defined
- [x] Structured response models for tools and prompts
- [x] Tool/Prompt registry pattern implemented
- [x] Example implementations with tests
- [x] Documentation of patterns in tools-api/README.md

### Actual Implementation

All abstractions were implemented in `tools_api/services/base.py` and `tools_api/models.py`:

1. **Base Classes:**
   - `BaseTool` with abstract methods for name, description, parameters
   - `BasePrompt` with abstract methods for name, description, arguments
   - Automatic error handling and timing in `__call__()` methods

2. **Response Models:**
   - `ToolResult` with success flag, result data, error, execution_time_ms
   - `PromptResult` with success flag, messages, error
   - Models defined but flexibility for tool-specific response structures

3. **Registry:**
   - `ToolRegistry` and `PromptRegistry` for discovery
   - Used by router endpoints for listing and executing tools/prompts

4. **Example Tool:**
   - Example tool implementation demonstrating the pattern
   - Tests validating base class behavior

### Key Changes

**1. Response models:**
```python
# tools-api/models.py
from pydantic import BaseModel, Field
from typing import Any, Optional
from datetime import datetime

class ToolResult(BaseModel):
    """Structured tool execution result."""
    success: bool
    result: Any  # Tool-specific data structure
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float

class PromptResult(BaseModel):
    """Structured prompt rendering result."""
    success: bool
    messages: list[dict[str, str]]  # OpenAI format: [{"role": "user", "content": "..."}]
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class ToolDefinition(BaseModel):
    """Tool metadata for discovery."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for parameters
    tags: list[str] = Field(default_factory=list)

class PromptDefinition(BaseModel):
    """Prompt metadata for discovery."""
    name: str
    description: str
    arguments: list[dict[str, Any]]  # [{"name": "...", "required": bool, "description": "..."}]
    tags: list[str] = Field(default_factory=list)
```

**2. Base abstractions:**
```python
# tools-api/services/base.py
from abc import ABC, abstractmethod
from typing import Any
import time

class BaseTool(ABC):
    """Base class for all tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (used in API endpoint)."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        pass
    
    @property
    def tags(self) -> list[str]:
        """Optional tags for categorization."""
        return []
    
    @abstractmethod
    async def _execute(self, **kwargs) -> Any:
        """Execute tool and return structured result."""
        pass
    
    async def __call__(self, **kwargs) -> ToolResult:
        """Wrapper that handles timing and error handling."""
        start = time.time()
        try:
            result = await self._execute(**kwargs)
            execution_time_ms = (time.time() - start) * 1000
            return ToolResult(
                success=True,
                result=result,
                execution_time_ms=execution_time_ms,
            )
        except Exception as e:
            execution_time_ms = (time.time() - start) * 1000
            return ToolResult(
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=execution_time_ms,
            )

class BasePrompt(ABC):
    """Base class for all prompts."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Prompt name (used in API endpoint)."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Prompt description."""
        pass
    
    @property
    @abstractmethod
    def arguments(self) -> list[dict[str, Any]]:
        """Prompt arguments schema."""
        pass
    
    @property
    def tags(self) -> list[str]:
        """Optional tags for categorization."""
        return []
    
    @abstractmethod
    async def render(self, **kwargs) -> list[dict[str, str]]:
        """Render prompt with arguments and return messages."""
        pass
    
    async def __call__(self, **kwargs) -> PromptResult:
        """Wrapper that handles error handling."""
        try:
            messages = await self.render(**kwargs)
            return PromptResult(success=True, messages=messages)
        except Exception as e:
            return PromptResult(
                success=False,
                messages=[],
                error=str(e),
            )
```

**3. Registry pattern:**
```python
# tools-api/services/registry.py
from typing import Dict, Type
from .base import BaseTool, BasePrompt

class ToolRegistry:
    """Central registry for all tools."""
    _tools: Dict[str, BaseTool] = {}
    
    @classmethod
    def register(cls, tool: BaseTool):
        """Register a tool instance."""
        cls._tools[tool.name] = tool
    
    @classmethod
    def get(cls, name: str) -> BaseTool:
        """Get tool by name."""
        return cls._tools.get(name)
    
    @classmethod
    def list(cls) -> list[BaseTool]:
        """List all registered tools."""
        return list(cls._tools.values())

class PromptRegistry:
    """Central registry for all prompts."""
    _prompts: Dict[str, BasePrompt] = {}
    
    @classmethod
    def register(cls, prompt: BasePrompt):
        """Register a prompt instance."""
        cls._prompts[prompt.name] = prompt
    
    @classmethod
    def get(cls, name: str) -> BasePrompt:
        """Get prompt by name."""
        return cls._prompts.get(name)
    
    @classmethod
    def list(cls) -> list[BasePrompt]:
        """List all registered prompts."""
        return list(cls._prompts.values())
```

**4. Example tool implementation:**
```python
# tools-api/services/tools/example_tool.py
from ..base import BaseTool

class EchoTool(BaseTool):
    """Example tool for testing - echoes input."""
    
    @property
    def name(self) -> str:
        return "echo"
    
    @property
    def description(self) -> str:
        return "Echo the input message back"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to echo",
                },
            },
            "required": ["message"],
        }
    
    @property
    def tags(self) -> list[str]:
        return ["example", "test"]
    
    async def _execute(self, message: str) -> dict:
        return {
            "echo": message,
            "length": len(message),
        }
```

**5. Router endpoints:**
```python
# tools-api/routers/tools.py
from fastapi import APIRouter, HTTPException
from typing import Any
from ..models import ToolResult, ToolDefinition
from ..services.registry import ToolRegistry

router = APIRouter(prefix="/tools", tags=["tools"])

@router.get("/")
async def list_tools() -> list[ToolDefinition]:
    """List all available tools."""
    return [
        ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            tags=tool.tags,
        )
        for tool in ToolRegistry.list()
    ]

@router.post("/{tool_name}")
async def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
) -> ToolResult:
    """Execute a tool with the provided arguments."""
    tool = ToolRegistry.get(tool_name)
    if tool is None:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    
    return await tool(**arguments)
```

### Testing Strategy

**What to test:**
- Tool execution with valid/invalid arguments
- Error handling (missing required params, validation errors)
- Timing metadata in responses
- Registry operations (register, get, list)
- Example tool works end-to-end

**Example tests:**
```python
# tools-api/tests/test_base.py
import pytest
from services.tools.example_tool import EchoTool
from services.registry import ToolRegistry

@pytest.mark.asyncio
async def test_tool_execution():
    tool = EchoTool()
    result = await tool(message="hello")
    
    assert result.success is True
    assert result.result["echo"] == "hello"
    assert result.result["length"] == 5
    assert result.execution_time_ms > 0

@pytest.mark.asyncio
async def test_tool_error_handling():
    tool = EchoTool()
    # Missing required parameter
    result = await tool()
    
    assert result.success is False
    assert result.error is not None

def test_registry():
    tool = EchoTool()
    ToolRegistry.register(tool)
    
    retrieved = ToolRegistry.get("echo")
    assert retrieved is tool
    
    all_tools = ToolRegistry.list()
    assert tool in all_tools
```

### Dependencies
- Milestone 1 (service scaffold)

### Risk Factors
- JSON Schema validation may need additional library (e.g., `jsonschema`)
- Async/await patterns need to be consistent throughout
- Parameter validation should be done by Pydantic where possible

---

## Milestone 3: Migrate Filesystem Tools

### Goal
Migrate all filesystem tools from MCP to tools-api with structured responses and proper error handling.

### Success Criteria
- [ ] All filesystem tools implemented (read, write, list, search, etc.)
- [ ] Structured responses with file metadata
- [ ] Path validation (restricted to allowed directories)
- [ ] Comprehensive error handling
- [ ] Unit tests for all tools
- [ ] Integration tests with volume mounts

### Key Changes

**Tools to migrate:**
- `read_text_file` (was `filesystem__read_text_file`)
- `read_media_file` (was `filesystem__read_media_file`)
- `read_multiple_files` (was `filesystem__read_multiple_files`)
- `write_file` (was `filesystem__write_file`)
- `edit_file` (was `filesystem__edit_file`)
- `create_directory` (was `filesystem__create_directory`)
- `list_directory` (was `filesystem__list_directory`)
- `list_directory_with_sizes` (was `filesystem__list_directory_with_sizes`)
- `search_files` (was `filesystem__search_files`)
- `get_file_info` (was `filesystem__get_file_info`)
- `list_allowed_directories` (was `filesystem__list_allowed_directories`)
- `delete_file` (was `file_operations__delete_file`)
- `delete_directory` (was `file_operations__delete_directory`)
- `move_file` (was `filesystem__move_file`)

**Example implementation:**
```python
# tools-api/services/tools/filesystem.py
from pathlib import Path
from typing import Optional
import base64
from ..base import BaseTool
from ...config import settings

class ReadTextFileTool(BaseTool):
    """Read a text file with structured response."""
    
    @property
    def name(self) -> str:
        return "read_text_file"
    
    @property
    def description(self) -> str:
        return "Read the complete contents of a text file from the file system"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to read",
                },
                "head": {
                    "type": "integer",
                    "description": "If provided, returns only the first N lines",
                },
                "tail": {
                    "type": "integer",
                    "description": "If provided, returns only the last N lines",
                },
            },
            "required": ["path"],
        }
    
    @property
    def tags(self) -> list[str]:
        return ["filesystem", "read"]

    async def _execute(
        self,
        path: str,
        head: Optional[int] = None,
        tail: Optional[int] = None,
    ) -> dict:
        file_path = Path(path)

        # Validate readable using settings
        is_readable, error = settings.is_readable(file_path)
        if not is_readable:
            raise PermissionError(error)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if not file_path.is_file():
            raise ValueError(f"Not a file: {path}")
        
        # Read file
        content = file_path.read_text()
        lines = content.splitlines(keepends=True)
        
        # Apply head/tail if specified
        if head is not None:
            lines = lines[:head]
        elif tail is not None:
            lines = lines[-tail:]
        
        # Get file stats
        stat = file_path.stat()
        
        return {
            "path": str(file_path),
            "content": "".join(lines),
            "line_count": len(lines),
            "size_bytes": stat.st_size,
            "modified_time": stat.st_mtime,
            "is_truncated": head is not None or tail is not None,
        }

class WriteFileTool(BaseTool):
    """Write content to a file."""
    
    @property
    def name(self) -> str:
        return "write_file"
    
    @property
    def description(self) -> str:
        return "Create a new file or completely overwrite an existing file"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path where the file should be written",
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
        return ["filesystem", "write"]

    async def _execute(self, path: str, content: str) -> dict:
        file_path = Path(path)

        # Validate writable using settings (includes blocked pattern check)
        is_writable, error = settings.is_writable(file_path)
        if not is_writable:
            raise PermissionError(error)
        
        # Write file
        file_path.write_text(content)
        
        # Get stats after write
        stat = file_path.stat()
        
        return {
            "path": str(file_path),
            "size_bytes": stat.st_size,
            "line_count": len(content.splitlines()),
            "modified_time": stat.st_mtime,
        }

# ... implement remaining filesystem tools following same pattern
```

**Register tools on startup:**
```python
# tools-api/main.py
from fastapi import FastAPI
from routers import tools as tools_router, prompts as prompts_router
from services.registry import ToolRegistry
from services.tools.filesystem import (
    ReadTextFileTool,
    WriteFileTool,
    # ... import all filesystem tools
)

app = FastAPI(...)

# Register tools
ToolRegistry.register(ReadTextFileTool())
ToolRegistry.register(WriteFileTool())
# ... register remaining tools

# Include routers
app.include_router(tools_router.router)
app.include_router(prompts_router.router)
```

### Testing Strategy

**What to test:**
1. **Path validation:**
   - Allowed paths succeed
   - Paths outside allowed directories raise PermissionError
   - Path traversal attempts (../../etc/passwd) blocked

2. **Error conditions:**
   - File not found
   - Permission denied (on read-only volumes)
   - Invalid paths (null bytes, etc.)
   - Directory operations on files and vice versa

3. **Functionality:**
   - Read full file
   - Read with head/tail
   - Write new file
   - Overwrite existing file
   - List directory (files and subdirectories)
   - Search files with patterns
   - Delete operations

4. **Structured responses:**
   - Metadata (size, modified time) included
   - Line counts accurate




   **Example tests:**
```python
# tools-api/tests/test_filesystem_tools.py
import pytest
from pathlib import Path
from services.tools.filesystem import ReadTextFileTool, WriteFileTool
from services.registry import ToolRegistry

@pytest.fixture
def test_file(tmp_path):
    """Create a test file."""
    file_path = tmp_path / "test.txt"
    content = "line 1\nline 2\nline 3\n"
    file_path.write_text(content)
    return file_path

@pytest.mark.asyncio
async def test_read_text_file(test_file):
    tool = ReadTextFileTool()
    result = await tool(path=str(test_file))
    
    assert result.success is True
    assert result.result["content"] == "line 1\nline 2\nline 3\n"
    assert result.result["line_count"] == 3
    assert result.result["size_bytes"] > 0
    assert result.result["is_truncated"] is False

@pytest.mark.asyncio
async def test_read_text_file_with_head(test_file):
    tool = ReadTextFileTool()
    result = await tool(path=str(test_file), head=2)
    
    assert result.success is True
    assert result.result["line_count"] == 2
    assert result.result["is_truncated"] is True

@pytest.mark.asyncio
async def test_read_text_file_not_found():
    tool = ReadTextFileTool()
    result = await tool(path="/nonexistent/file.txt")
    
    assert result.success is False
    assert "not found" in result.error.lower()

@pytest.mark.asyncio
async def test_path_validation_outside_allowed():
    """Test that paths outside allowed directories are rejected."""
    tool = ReadTextFileTool()
    result = await tool(path="/etc/passwd")
    
    assert result.success is False
    assert "permission" in result.error.lower() or "denied" in result.error.lower()

@pytest.mark.asyncio
async def test_write_file(tmp_path):
    tool = WriteFileTool()
    file_path = tmp_path / "new_file.txt"
    
    result = await tool(
        path=str(file_path),
        content="hello world\n",
    )
    
    assert result.success is True
    assert result.result["size_bytes"] > 0
    assert result.result["line_count"] == 1
    
    # Verify file was actually written
    assert file_path.exists()
    assert file_path.read_text() == "hello world\n"

@pytest.mark.asyncio
async def test_integration_read_write(tmp_path):
    """Test full read-write cycle."""
    write_tool = WriteFileTool()
    read_tool = ReadTextFileTool()
    
    file_path = tmp_path / "integration.txt"
    content = "test content\nline 2\n"
    
    # Write
    write_result = await write_tool(path=str(file_path), content=content)
    assert write_result.success is True
    
    # Read back
    read_result = await read_tool(path=str(file_path))
    assert read_result.success is True
    assert read_result.result["content"] == content
```

You're right, let me refocus on the plan structure.

### Dependencies
- Milestone 1 (path security configuration)
- Milestone 2 (abstractions)

### Risk Factors
- Path security patterns may need refinement based on actual usage
- Path traversal security vulnerabilities (mitigated by path.resolve() checks)
- Large file handling (memory limits for very large files)
- Encoding issues with non-UTF-8 files

---

## Milestone 4: Migrate GitHub and Development Tools

### Goal
Migrate GitHub and development tools to tools-api using direct API clients and subprocess calls.

### Success Criteria
- [ ] GitHub PR info tool (using GitHub API)
- [ ] Local git changes tool (using subprocess)
- [ ] Directory tree tool (using subprocess/pathlib)
- [ ] All tools return structured responses
- [ ] Tests with mocked GitHub API and fixtures

### Key Changes

**Tools to migrate:**
- `get_github_pull_request_info` → Use PyGithub or httpx directly
- `get_local_git_changes_info` → Use subprocess to call git commands
- `get_directory_tree` → Use subprocess to call `tree` command or implement with pathlib

**Pattern for GitHub API:**
```python
# Use httpx to call GitHub API directly (no MCP)
# GitHub token configured in settings from GITHUB_TOKEN env var (Milestone 1)
from ...config import settings

async with httpx.AsyncClient() as client:
    headers = {}
    if settings.github_token:
        headers["Authorization"] = f"token {settings.github_token}"

    response = await client.get(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}",
        headers=headers,
    )

    if response.status_code == 401 and not settings.github_token:
        raise ValueError("GitHub API requires GITHUB_TOKEN environment variable")
```

**Pattern for git subprocess:**
```python
# Use asyncio.create_subprocess_exec for git commands
proc = await asyncio.create_subprocess_exec(
    "git", "status",
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
    cwd=repo_path,
)
stdout, stderr = await proc.communicate()
```

### Testing Strategy
- Mock GitHub API responses with `respx` or similar
- Use test git repositories in tmp_path fixtures
- Test error conditions (invalid PR URLs, non-git directories)
- Validate structured response schema

### Dependencies
- Milestone 1 (GitHub token configuration)
- Milestone 2 (abstractions)

### Risk Factors
- GitHub API rate limiting (mitigated by graceful error handling)
- Git command availability in Docker container (ensure git installed in Dockerfile)
- Subprocess security (shell injection - always use list form, never shell=True)

---

## Milestone 5: Migrate Web Search Tools

### Goal
Replace Brave Search MCP integration with direct API client.

### Success Criteria
- [ ] Web search tool using Brave Search API directly
- [ ] Structured search results (title, URL, snippet, rank)
- [ ] Rate limiting and error handling
- [ ] Tests with mocked API responses

### Key Changes

**Tool to migrate:**
- `web_search` → Use Brave Search API with httpx

**Pattern:**
```python
# Direct API call instead of MCP
# Brave API key configured in settings from BRAVE_API_KEY env var (Milestone 1)
from ...config import settings

if not settings.brave_api_key:
    raise ValueError("Brave Search requires BRAVE_API_KEY environment variable")

async with httpx.AsyncClient() as client:
    response = await client.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": settings.brave_api_key},
        params={"q": query},
    )
```

**Structured response includes:**
- Query metadata
- List of results with title, URL, snippet, rank
- Total results count
- Search metadata (time, filters applied)

### Testing Strategy
- Mock Brave API responses
- Test error handling (rate limits, timeouts)
- Validate response structure matches schema

### Dependencies
- Milestone 1 (Brave API key configuration)
- Milestone 2 (abstractions)

### Risk Factors
- Rate limiting from Brave API (implement graceful error handling)
- Response format changes from upstream (version API responses in tests)

---

## Milestone 6: Migrate Prompts from Playbooks

### Goal
Extract YAML prompt utilities from mcp-this and integrate with tools-api to support existing playbooks format.

### Success Criteria
- [ ] Extract reusable code from mcp-this (~314 lines: loader, renderer, parser, validator)
- [ ] All playbooks prompts loaded (meta, thinking, development)
- [ ] Template rendering works (Handlebars-like `{{var}}` and `{{#if}}` conditionals)
- [ ] Argument validation and substitution
- [ ] Tests ported from mcp-this + new integration tests

### Key Changes

**1. Extract mcp-this utilities (no MCP dependency):**
```
tools-api/yaml_tools/
├── __init__.py
├── loader.py         # load_config() - YAML/JSON loading with validation
├── renderer.py       # render_template() - {{var}} substitution and {{#if}} conditionals
├── models.py         # PromptInfo, PromptArgument, ToolInfo dataclasses
├── parser.py         # parse_prompts(), parse_tools() - YAML → objects
└── validator.py      # validate_config() - strict validation
```

**Source:** Copy from `/Users/shanekercheval/repos/mcp-this/src/mcp_this/`:
- `mcp_server.py` lines 22-55 (template rendering)
- `mcp_server.py` lines 57-150 (YAML loading/validation)
- `prompts.py` (PromptInfo, parsing, validation)
- Remove MCP-specific parts (`mcp.tool()` decorators, server registration)

**2. Existing playbooks format:**
```yaml
# /Users/shanekercheval/repos/playbooks/src/development/config.yaml
prompts:
  create-playbook:
    description: Create a new playbook
    template: |
      {{#if technology}}
      Create a playbook for {{technology}}
      {{else}}
      Create a general-purpose playbook
      {{/if}}

      Requirements: {{requirements}}
    arguments:
      technology:
        description: Technology or domain
        required: false
      requirements:
        description: Specific requirements
        required: true
```

**3. YAMLPrompt implementation:**
```python
# tools-api/services/prompts/yaml_prompt.py
from ..base import BasePrompt
from ...yaml_tools import PromptInfo, render_template

class YAMLPrompt(BasePrompt):
    """Prompt loaded from playbooks YAML."""

    def __init__(self, prompt_info: PromptInfo):
        self.info = prompt_info

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def description(self) -> str:
        return self.info.description

    @property
    def arguments(self) -> list[dict[str, Any]]:
        return [
            {
                "name": arg.name,
                "description": arg.description,
                "required": arg.required,
            }
            for arg in self.info.arguments
        ]

    async def render(self, **kwargs) -> list[dict[str, str]]:
        # Use mcp-this template renderer (proven, tested)
        content = render_template(self.info.template, kwargs)

        return [{"role": "user", "content": content}]
```

**4. Load playbooks on startup:**
```python
# tools-api/main.py
from services.prompts.yaml_prompt import YAMLPrompt
from services.registry import PromptRegistry
from yaml_tools import load_config, parse_prompts

# Load all playbooks YAML files
playbooks_dir = Path("/mnt/playbooks/src")
for yaml_file in playbooks_dir.glob("**/*.yaml"):
    config = load_config(yaml_file)
    prompt_infos = parse_prompts(config)

    for prompt_info in prompt_infos:
        prompt = YAMLPrompt(prompt_info)
        PromptRegistry.register(prompt)
```

**5. Volume mount playbooks:**
```yaml
# docker-compose.yml
services:
  tools-api:
    volumes:
      - /Users/shanekercheval/repos/playbooks:/mnt/playbooks:ro
```

### Testing Strategy

**1. Port mcp-this tests:**
- Copy tests for `render_template()` (variable substitution, conditionals)
- Copy tests for `load_config()` (YAML loading, validation)
- Copy tests for `parse_prompts()` (YAML → PromptInfo conversion)

**2. New integration tests:**
- Load actual playbooks YAML files
- Render templates with various arguments
- Validate required vs optional arguments
- Test error handling (missing required args, invalid YAML)

**3. Example test:**
```python
@pytest.mark.asyncio
async def test_yaml_prompt_rendering():
    """Test that YAML prompts render correctly."""
    # Load actual playbook
    config = load_config("/mnt/playbooks/src/development/config.yaml")
    prompts = parse_prompts(config)

    # Find create-playbook prompt
    create_playbook = next(p for p in prompts if p.name == "create-playbook")
    prompt = YAMLPrompt(create_playbook)

    # Render with arguments
    messages = await prompt.render(
        technology="Python",
        requirements="REST API with FastAPI"
    )

    assert len(messages) == 1
    assert "Python" in messages[0]["content"]
    assert "REST API" in messages[0]["content"]
```

### Benefits of Reusing mcp-this

**Advantages:**
- ✅ **Proven code** - mcp-this is tested and working
- ✅ **Zero reinvention** - ~314 lines of battle-tested utilities
- ✅ **Same syntax** - existing playbooks work without changes
- ✅ **Simple extraction** - just remove MCP-specific parts
- ✅ **Portable** - pure Python, no external dependencies for rendering

**What we're NOT reusing from mcp-this:**
- ❌ MCP server registration (`mcp.tool()` decorators)
- ❌ Dynamic function generation (`exec()` patterns)
- ❌ MCP protocol handling
- ❌ Subprocess tool execution (tools-api uses direct Python functions)

### Dependencies
- Milestone 1 (playbooks volume mount)
- Milestone 2 (abstractions)

### Risk Factors
- YAML parsing errors (mitigated: mcp-this already handles this)
- Template rendering edge cases (mitigated: extensive tests in mcp-this)
- Playbooks path configuration across environments (use env var)

---

## Milestone 7: Update Reasoning API Integration

### Goal
Remove MCP client from reasoning-api and integrate with tools-api REST endpoints.

### Success Criteria
- [ ] MCP client code removed
- [ ] HTTP client for tools-api implemented
- [ ] Tool/Prompt discovery via REST API
- [ ] Tool execution returns structured responses
- [ ] All existing tests passing with new integration
- [ ] Centralized OTEL tool logging in Tool class

### Key Changes

**1. Remove MCP dependencies:**
- Delete `api/mcp.py`
- Remove MCP client from dependencies
- Remove MCP config loading

**2. Create tools-api client:**
```python
# api/tools_client.py
class ToolsAPIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def list_tools(self) -> list[ToolDefinition]:
        response = await self.client.get(f"{self.base_url}/tools/")
        return [ToolDefinition(**t) for t in response.json()]
    
    async def execute_tool(self, name: str, arguments: dict) -> ToolResult:
        response = await self.client.post(
            f"{self.base_url}/tools/{name}",
            json=arguments,
        )
        return ToolResult(**response.json())
```

**3. Update Tool abstraction:**
```python
# api/tools.py - Update Tool class
class Tool:
    def __init__(self, definition: ToolDefinition, client: ToolsAPIClient):
        self.definition = definition
        self.client = client
    
    async def _execute(self, **kwargs) -> ToolResult:
        return await self.client.execute_tool(self.definition.name, kwargs)
```

**4. Centralize OTEL tool logging:**

**Current state:** Tool execution logging is only in `reasoning_agent.py`, causing:
- API tool endpoints (`/v1/mcp/tools/{tool_name}`) have NO tracing
- Duplicate instrumentation code
- Inconsistent logging across callers

**Solution:** Move OTEL to `Tool.__call__()` in `api/tools.py`

**Implementation:**
```python
# api/tools.py
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes
import json

tracer = trace.get_tracer(__name__)

class Tool(BaseModel):
    # ... existing fields ...

    async def __call__(self, **kwargs) -> ToolResult:
        """Execute tool with centralized OTEL logging."""

        # Create span for this tool execution
        with tracer.start_as_current_span(
            "tool.execute",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: "tool",
                SpanAttributes.TOOL_NAME: self.name,
                SpanAttributes.TOOL_PARAMETERS: json.dumps(kwargs),
                "tool.server_name": self.server_name,
                "tool.tags": self.tags,
            }
        ) as span:
            start = time.time()

            try:
                # Execute tool (existing logic)
                if asyncio.iscoroutinefunction(self.function):
                    result = await self.function(**kwargs)
                else:
                    result = await asyncio.to_thread(self.function, **kwargs)

                execution_time_ms = (time.time() - start) * 1000

                # Log success
                span.set_attribute("tool.success", True)
                span.set_attribute("tool.duration_ms", execution_time_ms)
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(result))
                span.set_status(trace.Status(trace.StatusCode.OK))

                return ToolResult(
                    success=True,
                    result=result,
                    execution_time_ms=execution_time_ms,
                )

            except Exception as e:
                execution_time_ms = (time.time() - start) * 1000
                error_msg = str(e)

                # Log failure
                span.set_attribute("tool.success", False)
                span.set_attribute("tool.duration_ms", execution_time_ms)
                span.set_attribute("tool.error", error_msg)
                span.set_status(trace.Status(
                    trace.StatusCode.ERROR,
                    description=error_msg
                ))

                return ToolResult(
                    success=False,
                    result=None,
                    error=error_msg,
                    execution_time_ms=execution_time_ms,
                )
```

**Cleanup in `reasoning_agent.py`:**
```python
# Remove _execute_single_tool_with_tracing() - no longer needed
# Simplify to direct tool calls:
async def _execute_tools_concurrently(self, predictions):
    tasks = [tool(**pred.arguments) for tool, pred in predictions]
    results = await asyncio.gather(*tasks)
    return results
```

**Benefits:**
- ✅ All tool calls automatically traced (reasoning agent, API endpoints, future orchestration)
- ✅ Consistent span attributes across all callers
- ✅ Phoenix UI shows complete tool execution journey
- ✅ Eliminates duplicate instrumentation code

**5. Error handling strategy:**

**tools-api returns structured errors:**
```python
# All tools return ToolResult with success flag
ToolResult(
    success=False,
    result=None,
    error="File not found: /path/to/file.txt",
    execution_time_ms=5.2
)
```

**reasoning-api surfaces errors to LLM:**
```python
# LLM receives error message and can respond appropriately
if not tool_result.success:
    messages.append({
        "role": "tool",
        "content": f"Error executing {tool_name}: {tool_result.error}"
    })
```

**No retries at tool level** - let LLM decide if retry is appropriate based on error message.

### Testing Strategy

**Integration tests using existing patterns:**

```python
# tests/integration_tests/conftest.py - Add tools-api fixtures

@pytest.fixture(scope="session")
def postgres_container_tools():
    """PostgreSQL for tools-api (if tools-api has database)."""
    with PostgresContainer("postgres:16", ...) as postgres:
        yield postgres.get_connection_url()

@pytest_asyncio.fixture(loop_scope="function")
async def tools_client():
    """Test client for tools-api using ASGITransport (in-process)."""
    from tools_api.main import app as tools_app

    async with AsyncClient(
        transport=ASGITransport(app=tools_app),
        base_url="http://test"
    ) as ac:
        yield ac

@pytest_asyncio.fixture(loop_scope="function")
async def reasoning_client_with_tools(tools_client):
    """Test reasoning-api with mocked tools-api responses."""
    from api.main import app

    # Mock tools-api HTTP client to return tools_client responses
    with patch('api.tools_client.httpx.AsyncClient') as mock:
        mock.return_value = tools_client
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as ac:
            yield ac
```

**Test scenarios:**
- Tool discovery via REST API
- Tool execution with valid/invalid arguments
- Error propagation (tools-api returns error → reasoning-api surfaces to LLM)
- OTEL span creation in Tool.__call__()
- Integration: reasoning-api → tools-api → tool execution

### Dependencies
- Milestones 3-6 (all tools/prompts migrated)

### Risk Factors
- Network latency between containers (should be minimal, same Docker network)
- Error propagation from tools-api (ensure ToolResult.error is properly surfaced to LLM)
- OTEL span duplication if not properly removed from reasoning_agent.py

---

## Milestone 8: Remove MCP Configuration and Bridge

### Goal
Clean up all MCP-related configuration and infrastructure.

### Success Criteria
- [ ] MCP bridge code deleted
- [ ] MCP config files deleted
- [ ] Docker Compose updated (no bridge service)
- [ ] Documentation updated
- [ ] No references to MCP in codebase

### Key Changes

**Files to delete:**
- `mcp_bridge/` (entire directory)
- `config/mcp_servers.json`
- `config/mcp_bridge_config.json`
- `config/mcp_overrides.yaml`
- `README_MCP_QUICKSTART.md`

**Update documentation:**
- `README.md` - Remove MCP setup sections
- Add tools-api documentation
- Update architecture diagrams

**Update Docker Compose:**
- Remove any MCP bridge service references
- Update environment variables
- Clean up network configuration if needed

### Testing Strategy
- Verify all tests still pass
- Verify docker compose up works
- Verify no broken documentation links
- Grep codebase for "mcp" references

### Dependencies
- Milestone 7 (reasoning-api integration complete)

### Risk Factors
- Missed references in documentation
- Circular dependencies in config

---

## Milestone 9: Documentation and Migration Guide

### Goal
Complete documentation for new architecture and provide migration guide.

### Success Criteria
- [ ] Architecture documentation in README.md
- [ ] Tools API documentation
- [ ] Deployment guide (local and cloud)
- [ ] Migration guide from MCP (if needed)
- [ ] API examples and tutorials

### Key Changes

**Documentation to create:**
1. **tools-api/README.md:**
   - Architecture overview
   - Adding new tools/prompts
   - Testing guidelines
   - Deployment instructions

2. **README.md updates:**
   - Remove MCP sections
   - Add tools-api sections
   - Update architecture diagrams
   - Update quick start guide

3. **docs/ARCHITECTURE.md:**
   - System architecture
   - Request flow diagrams
   - Tool execution lifecycle
   - Caching strategy

4. **docs/ADDING_TOOLS.md:**
   - Step-by-step guide for new tools
   - Code examples
   - Testing requirements
   - Best practices

**Architecture diagram:**
```
User Request
    ↓
reasoning-api (Docker)
    ↓ HTTP
tools-api (Docker)
    ↓ Volume Mounts
/mnt/repos (Host Filesystem)
```

### Testing Strategy
- Documentation review
- Test all code examples work
- Verify links and references
- Tutorial walkthrough

### Dependencies
- All previous milestones

### Risk Factors
- Documentation drift as code evolves
- Missing edge cases in examples

---

## Pre-Migration Testing & Findings

### Current MCP Setup (Tested 2025-11-23)

**MCP Bridge Status:** ✅ Running at `http://localhost:9000/mcp/`

**Total Resources:**
- **20 Tools** (across 7 MCP servers)
- **15 Prompts** (across 4 MCP servers)

**Tools Breakdown:**
1. **Filesystem** (11 tools via `@modelcontextprotocol/server-filesystem`):
   - Read: `read_text_file`, `read_media_file`, `read_multiple_files`, `get_file_info`, `list_directory`, `list_directory_with_sizes`, `search_files`, `list_allowed_directories`
   - Write: `write_file`, `edit_file`, `create_directory`, `move_file`

2. **GitHub/Dev** (3 tools via `mcp-this --preset github`):
   - `get_github_pull_request_info` - Fetch PR details via GitHub API
   - `get_local_git_changes_info` - Get git status/diff from local repo
   - `get_directory_tree` - Generate directory tree with tree command

3. **File Operations** (2 tools via `mcp-this`):
   - `delete_file`, `delete_directory`

4. **Web Search** (3 tools):
   - `web_search`, `web_search_location` (via `@modelcontextprotocol/server-brave-search`)
   - `web_scraper` (via `mcp-this` with lynx command)

**Prompts Breakdown:**
1. **Development** (9 prompts via `/Users/shanekercheval/repos/playbooks/src/development/config.yaml`):
   - `code_review`, `coding_guidelines`, `commit_message`, `create_playbook`, `implementation_guide`, `implementation_guide_review`, `python_coding_guidelines`, `unit_tests`, `update_documentation`

2. **Meta** (3 prompts via `/Users/shanekercheval/repos/playbooks/src/meta-prompts.yaml`):
   - `generate_playbook`, `generate_structured_prompt`, `update_playbooks`

3. **Thinking** (2 prompts via `/Users/shanekercheval/repos/playbooks/src/thinking/config.yaml`):
   - `explain_concept`, `transcript_summary`

4. **GitHub** (1 prompt via `mcp-this --preset github`):
   - `create_pr_description`

### Key Findings

**1. Brave Search Already Direct API:**
- ✅ **IMPORTANT:** `api/web_search.py` (555 lines) is already a complete Brave Search API client
- **Not** MCP-based - production-ready with:
  - Pydantic models for request/response validation
  - Automatic rate limiting and retry logic
  - Proper error handling (auth, rate limits, API errors)
  - Environment variable support (`BRAVE_SEARCH_API`)
- **Action:** Move entire module to `tools-api/` with minimal changes
- **Impact:** Milestone 5 is mostly done - just needs tool wrapper

**2. MCP Servers Architecture:**
```
reasoning-api (Docker)
  → MCP Bridge (localhost:9000)
    → 7 stdio MCP servers:
      1. filesystem (npx @modelcontextprotocol/server-filesystem)
      2. github-custom (uvx mcp-this --preset github)
      3. brave-search (npx @modelcontextprotocol/server-brave-search)
      4. tools (uvx mcp-this --config-path config/mcp-this-tools.yaml)
      5. file-operations (uvx mcp-this --config-path config/mcp-this-bridge-tools.yaml)
      6. meta (uvx mcp-this --config-path playbooks/src/meta-prompts.yaml)
      7. thinking (uvx mcp-this --config-path playbooks/src/thinking/config.yaml)
      8. dev (uvx mcp-this --config-path playbooks/src/development/config.yaml)
```

**3. Filesystem Access Paths:**
- Current: `/Users/shanekercheval/repos` (all projects) + `/Users/shanekercheval/Downloads`
- **Decision:** tools-api gets RW access to ALL of `/Users/shanekercheval/repos`
- Playbooks: `/Users/shanekercheval/repos/playbooks` (exists, contains YAML prompts)

**4. mcp-this Source:**
- ✅ Available locally at `/Users/shanekercheval/repos/mcp-this`
- Can extract YAML utilities for Milestone 6 (prompt rendering)

**5. Migration Scope Confirmed:**
- Migrating ALL 20 tools + 15 prompts (full 9-milestone plan)
- No backwards compatibility - delete MCP tests as we remove MCP code
- Clean slate approach

---

## Summary

**Total Milestones:** 9

**Key Benefits:**
- ✅ Structured tool responses (enables metadata extraction, reuse)
- ✅ Simpler architecture (no bridge, no MCP protocol)
- ✅ Direct Docker volume mounts (no host networking)
- ✅ Better testing (mock HTTP vs mock MCP, testcontainers)
- ✅ Better performance (less serialization overhead)
- ✅ Easier to extend (just add new tool class)
- ✅ Centralized observability (OTEL in Tool class)

**Key Design Decisions:**

1. **Volume Mount Strategy (Milestone 1):**
   - RW access to repos (agent can edit code, run tests)
   - Tool-level path security (blocked patterns for `.git/`, `node_modules/`, etc.)
   - Environment-based configuration (.env file for portability)

2. **API Key Management (Milestone 1):**
   - GitHub token and Brave API key in environment variables
   - Graceful error handling when keys are missing
   - Configured in settings, not hardcoded

3. **OTEL Centralization (Milestone 7):**
   - Move tool logging from reasoning_agent.py to Tool.__call__()
   - Single instrumentation point for all tool callers
   - Consistent span attributes across reasoning agent, API endpoints, orchestration
   - Eliminates duplicate OTEL code

4. **Integration Testing (Milestone 7):**
   - Use existing testcontainers pattern for databases
   - ASGITransport for in-process API testing (no real servers)
   - Transaction rollback for database isolation (no manual cleanup)

5. **Error Handling (Milestone 7):**
   - Structured ToolResult with success flag and error message
   - No automatic retries - let LLM decide based on error
   - Clean error propagation: tools-api → reasoning-api → LLM

**Critical Success Factors:**
1. Clean abstractions in Milestone 2 set pattern for everything
2. Comprehensive testing at each milestone (including path security and integration tests)
3. Path security validation (Milestone 1 & 3)
4. OTEL centralization (Milestone 7) for consistent observability
5. Complete documentation (Milestone 9) for maintainability

**Agent Instructions:**
- Complete each milestone fully before moving to next
- Ask clarifying questions before implementing
- Write meaningful tests (not just coverage)
- Stop after each milestone for review
- No backwards compatibility required - prioritize clean design
