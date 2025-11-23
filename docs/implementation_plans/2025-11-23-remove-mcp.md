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

## Milestone 1: Create Tools API Service Scaffold

[Continue with existing milestones...]
---

## Milestone 1: Create Tools API Service Scaffold

### Goal
Create new `tools-api` FastAPI service with Docker setup and basic project structure.

### Success Criteria
- [ ] `tools-api/` directory with FastAPI application
- [ ] Docker Compose integration with volume mounts
- [ ] Health check endpoint returns 200 OK
- [ ] Service accessible from reasoning-api container
- [ ] Tests passing for health check
- [ ] Documentation in `tools-api/README.md`

### Key Changes

**1. Create service structure:**
```
tools-api/
├── main.py              # FastAPI app
├── config.py            # Settings (base paths, etc.)
├── models.py            # Response models
├── routers/
│   ├── __init__.py
│   ├── tools.py         # Tool endpoints
│   └── prompts.py       # Prompt endpoints
├── services/
│   ├── __init__.py
│   ├── tools/           # Tool implementations
│   └── prompts/         # Prompt implementations
├── tests/
│   ├── test_health.py
│   └── conftest.py
├── Dockerfile
└── README.md
```

**2. Basic FastAPI application:**
```python
# tools-api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Tools API",
    description="Structured tool and prompt execution service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check() -> dict:
    return {"status": "healthy", "service": "tools-api"}
```

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
      # Read-only mounts for safety
      - /Users/shanekercheval/repos:/mnt/repos:ro
      - /Users/shanekercheval/Downloads:/mnt/downloads:ro
      # Playbooks for prompts
      - /Users/shanekercheval/repos/playbooks:/mnt/playbooks:ro
    environment:
      - BASE_REPOS_PATH=/mnt/repos
      - BASE_DOWNLOADS_PATH=/mnt/downloads
      - PLAYBOOKS_PATH=/mnt/playbooks
    networks:
      - reasoning-network
    restart: unless-stopped

  reasoning-api:
    # ... existing config ...
    environment:
      # Add tools-api URL
      - TOOLS_API_URL=http://tools-api:8001
```

**4. Configuration:**
```python
# tools-api/config.py
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    base_repos_path: Path = Path("/mnt/repos")
    base_downloads_path: Path = Path("/mnt/downloads")
    playbooks_path: Path = Path("/mnt/playbooks")
    
    # Allowed directories for filesystem operations
    allowed_directories: list[Path] = None
    
    def __post_init__(self):
        if self.allowed_directories is None:
            self.allowed_directories = [
                self.base_repos_path,
                self.base_downloads_path,
            ]
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### Testing Strategy

**What to test:**
- Health check returns correct response
- Docker container starts successfully
- Volume mounts are accessible (test file read from /mnt/repos)
- Service is reachable from reasoning-api container

**Example test:**
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
```

### Dependencies
None (first milestone)

### Risk Factors
- Volume mount paths may need adjustment for different environments
- Docker networking between services needs testing
- Read-only mounts may block certain operations (write_file should check permissions)

---

## Milestone 2: Define Tool/Prompt Abstractions

### Goal
Create clean abstractions for tools and prompts with structured response models, establishing patterns for all future implementations.

### Success Criteria
- [ ] `Tool` and `Prompt` base classes defined
- [ ] Structured response models for tools and prompts
- [ ] Tool/Prompt registry pattern implemented
- [ ] Example implementations with tests
- [ ] Documentation of patterns in tools-api/README.md

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
    
    def _validate_path(self, path: Path) -> None:
        """Validate path is within allowed directories."""
        path = path.resolve()
        allowed = any(
            path.is_relative_to(allowed_dir)
            for allowed_dir in settings.allowed_directories
        )
        if not allowed:
            raise PermissionError(
                f"Access denied: {path} is not within allowed directories"
            )
    
    async def _execute(
        self,
        path: str,
        head: Optional[int] = None,
        tail: Optional[int] = None,
    ) -> dict:
        file_path = Path(path)
        self._validate_path(file_path)
        
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
    
    def _validate_path(self, path: Path) -> None:
        """Validate path is within allowed directories and writable."""
        path = path.resolve()
        allowed = any(
            path.is_relative_to(allowed_dir)
            for allowed_dir in settings.allowed_directories
        )
        if not allowed:
            raise PermissionError(
                f"Access denied: {path} is not within allowed directories"
            )
        
        # Check parent directory exists and is writable
        if not path.parent.exists():
            raise FileNotFoundError(f"Parent directory does not exist: {path.parent}")
        
        if not path.parent.is_dir():
            raise ValueError(f"Parent is not a directory: {path.parent}")
    
    async def _execute(self, path: str, content: str) -> dict:
        file_path = Path(path)
        self._validate_path(file_path)
        
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
- Milestone 2 (abstractions)

### Risk Factors
- Volume mount permissions on read-only volumes
- Path traversal security vulnerabilities
- Large file handling (memory limits)
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
async with httpx.AsyncClient() as client:
    response = await client.get(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}",
        headers={"Authorization": f"token {github_token}"},
    )
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
- Milestone 2 (abstractions)

### Risk Factors
- GitHub API rate limiting
- Git command availability in Docker container
- Subprocess security (shell injection)

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
async with httpx.AsyncClient() as client:
    response = await client.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": brave_api_key},
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
- Milestone 2 (abstractions)

### Risk Factors
- API key management in environment variables
- Rate limiting from Brave API
- Response format changes from upstream

---

## Milestone 6: Migrate Prompts

### Goal
Convert MCP prompts to tools-api prompt implementations with YAML-based templates.

### Success Criteria
- [ ] All prompts migrated (meta, thinking, dev)
- [ ] YAML template loading and rendering
- [ ] Argument validation and substitution
- [ ] Tests for rendering and argument validation

### Key Changes

**Prompts to migrate:**
- Meta prompts: `generate_playbook`, `update_playbooks`, `generate_structured_prompt`
- Thinking prompts: `explain_concept`, `transcript_summary`
- Dev prompts: `code_review`, `implementation_guide`, `unit_tests`, etc.

**Pattern - YAML templates:**
```yaml
# /mnt/playbooks/src/code-review-prompt.yaml
name: code_review
description: Review code for best practices
arguments:
  - name: code
    required: true
    description: Code to review
  - name: language
    required: false
    description: Programming language

template: |
  Review the following {language} code:
  
  {code}
  
  Provide feedback on:
  - Best practices
  - Potential bugs
  - Performance issues
```

**Prompt implementation loads and renders YAML:**
```python
class CodeReviewPrompt(BasePrompt):
    def __init__(self, template_path: Path):
        self.template = yaml.safe_load(template_path.read_text())
    
    async def render(self, **kwargs) -> list[dict[str, str]]:
        # Validate arguments
        # Substitute into template
        # Return OpenAI message format
        pass
```

### Testing Strategy
- Test template loading and parsing
- Test argument validation (required vs optional)
- Test template rendering with various arguments
- Test error handling (missing required args)

### Dependencies
- Milestone 2 (abstractions)

### Risk Factors
- YAML template syntax errors
- Complex template logic (conditionals, loops)
- Argument validation complexity

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
- [ ] Caching layer for tool responses

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

**4. Add caching layer:**
- Cache successful tool responses (especially file reads)
- Use TTL-based cache (e.g., 5 minutes for file contents)
- Cache key: `tool_name:hash(arguments)`

### Testing Strategy
- Mock tools-api responses in tests
- Test tool discovery and execution
- Test error handling (tools-api down, tool not found)
- Test caching behavior
- Integration tests with real tools-api

### Dependencies
- Milestones 3-6 (all tools/prompts migrated)

### Risk Factors
- Network latency between containers (should be minimal)
- Error propagation from tools-api
- Cache invalidation strategy

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

## Milestone 9: Performance Testing and Optimization

### Goal
Validate performance improvements from structured responses and caching, identify bottlenecks.

### Success Criteria
- [ ] Benchmark suite implemented
- [ ] Performance comparison (MCP vs tools-api)
- [ ] Latency metrics for each tool
- [ ] Memory usage profiling
- [ ] Optimization recommendations documented

### Key Changes

**Benchmark scenarios:**
1. **Single file read:** Compare MCP bridge vs tools-api
2. **Multiple file reads:** Test caching effectiveness
3. **Complex reasoning task:** End-to-end with tool use
4. **Concurrent requests:** Load testing

**Metrics to collect:**
- Tool execution latency (p50, p95, p99)
- Cache hit rate
- Memory usage (reasoning-api, tools-api)
- Network latency between containers
- Full request latency (user → reasoning-api → tools-api → response)

**Pattern:**
```python
# tests/benchmarks/test_performance.py
@pytest.mark.benchmark
async def test_file_read_performance(benchmark):
    result = benchmark(lambda: tool.execute(path="/mnt/repos/file.py"))
    assert result < 50  # ms
```

### Testing Strategy
- Run benchmarks on consistent hardware
- Compare with baseline (MCP architecture)
- Test with realistic conversation scenarios
- Profile with py-spy or similar

### Dependencies
- All previous milestones (full implementation)

### Risk Factors
- Benchmark reliability (noise, variance)
- Fair comparison (different implementations)
- Container resource limits affecting results

---

## Milestone 10: Documentation and Migration Guide

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

## Summary

**Total Milestones:** 11

**Key Benefits:**
- ✅ Structured tool responses (enables caching, reuse)
- ✅ Simpler architecture (no bridge, no MCP protocol)
- ✅ Direct Docker volume mounts (no host networking)
- ✅ Better testing (mock HTTP vs mock MCP)
- ✅ Better performance (caching, less serialization)
- ✅ Easier to extend (just add new tool class)

**Critical Success Factors:**
1. Clean abstractions in Milestone 2 set pattern for everything
2. Comprehensive testing at each milestone
3. Path security validation (Milestone 3)
4. Caching strategy (Milestone 9) for performance
5. Complete documentation (Milestone 11) for maintainability

**Agent Instructions:**
- Complete each milestone fully before moving to next
- Ask clarifying questions before implementing
- Write meaningful tests (not just coverage)
- Stop after each milestone for review
- No backwards compatibility required - prioritize clean design
