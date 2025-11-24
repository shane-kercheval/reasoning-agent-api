# Tools API

Structured tool and prompt execution service for the reasoning agent.

## Overview

Tools API replaces the MCP (Model Context Protocol) architecture with a clean REST API that returns structured JSON responses. This enables:

- **Structured responses** with metadata (file stats, execution time, etc.)
- **Simpler architecture** (no bridge, no protocol layers)
- **Better testing** (mock HTTP endpoints instead of MCP servers)
- **Direct implementations** (filesystem via pathlib, GitHub via API, etc.)

## Architecture

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

## Project Structure

```
tools-api/
├── main.py              # FastAPI app
├── config.py            # Settings (base paths, path security)
├── models.py            # Response models (ToolResult, PromptResult, etc.)
├── routers/
│   ├── tools.py         # Tool endpoints (GET /tools, POST /tools/{name})
│   └── prompts.py       # Prompt endpoints (GET /prompts, POST /prompts/{name})
├── services/
│   ├── tools/           # Tool implementations (filesystem, github, search, etc.)
│   └── prompts/         # Prompt implementations (YAML-based)
├── tests/
│   ├── test_health.py
│   └── test_path_security.py
├── Dockerfile
└── README.md
```

## Configuration

### Environment Variables

Copy `tools-api/.env.example` to `.env` and configure:

```bash
# Path Configuration (Volume Mounts)
REPOS_PATH=/Users/yourname/repos           # RW access for code development
DOWNLOADS_PATH=/Users/yourname/Downloads   # Read-only
PLAYBOOKS_PATH=/Users/yourname/repos/playbooks  # Read-only (YAML prompts)

# API Keys
GITHUB_TOKEN=ghp_your_token_here          # For GitHub PR/git tools
BRAVE_API_KEY=your_brave_api_key_here     # For web search
```

### Path Security

Tools API implements multi-layer path security:

1. **Readable locations:** `/mnt/repos`, `/mnt/downloads`, `/mnt/workspace`, `/mnt/playbooks`
2. **Writable locations:** `/mnt/repos`, `/mnt/workspace` (NOT downloads/playbooks)
3. **Blocked patterns:** `.git/*`, `node_modules/*`, `.venv/*`, `__pycache__/*`, `.env`, etc.

Even in RW-mounted volumes, sensitive paths are blocked at the application level.

## API Endpoints

### Health Check
```bash
GET /health
→ {"status": "healthy", "service": "tools-api", "version": "1.0.0"}
```

### Tools
```bash
# List all tools
GET /tools/
→ [{"name": "read_text_file", "description": "...", "parameters": {...}, "tags": ["filesystem", "read"]}]

# Execute tool
POST /tools/read_text_file
{"path": "/mnt/repos/project/README.md"}
→ {
    "success": true,
    "result": {
      "path": "/mnt/repos/project/README.md",
      "content": "# Project...",
      "line_count": 42,
      "size_bytes": 1234,
      "modified_time": 1699564800.0
    },
    "execution_time_ms": 5.2
  }
```

### Prompts
```bash
# List all prompts
GET /prompts/
→ [{"name": "code_review", "description": "...", "arguments": [...], "tags": ["coding"]}]

# Render prompt
POST /prompts/code_review
{"code": "def foo():\n    pass"}
→ {
    "success": true,
    "messages": [
      {"role": "user", "content": "Review this code:\n\ndef foo():\n    pass"}
    ]
  }
```

## Development

### Running Tests

```bash
# Run all tests (use uv)
uv run pytest tools-api/tests/ -v

# Run specific test
uv run pytest tools-api/tests/test_health.py -v

# Run with coverage
uv run pytest tools-api/tests/ --cov=tools_api --cov-report=html
```

### Running Locally (Docker)

```bash
# Build and start all services
docker compose up --build

# Just tools-api
docker compose up tools-api

# View logs
docker compose logs -f tools-api

# Test health
curl http://localhost:8001/health
```

### Running Locally (Dev Mode)

```bash
# From tools-api directory
uv run uvicorn tools_api.main:app --reload --port 8001

# Or use make command (if added to Makefile)
make tools_api
```

## Adding New Tools

See implementation plan Milestone 2 for base abstractions and patterns.

Example tool structure:

```python
# tools-api/services/tools/example.py
from ...services.base import BaseTool

class ExampleTool(BaseTool):
    @property
    def name(self) -> str:
        return "example_tool"

    @property
    def description(self) -> str:
        return "Example tool description"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "param": {"type": "string", "description": "Parameter description"}
            },
            "required": ["param"]
        }

    async def _execute(self, param: str) -> dict:
        return {"result": f"Executed with {param}"}
```

## Migration from MCP

This service replaces:
- `mcp_bridge/` directory (500+ lines)
- `api/mcp.py` (300+ lines of protocol/parsing logic)
- 3 configuration files (mcp_servers.json, mcp_bridge_config.json, mcp_overrides.yaml)
- MCP stdio servers (filesystem, brave-search, mcp-this, etc.)

Benefits:
- **Structured responses:** JSON with metadata instead of text blobs
- **No bridge complexity:** Direct Docker networking instead of host.docker.internal
- **Better testing:** Standard HTTP mocking instead of MCP protocol simulation
- **Easier debugging:** Standard HTTP logs instead of multi-layer protocol translation
