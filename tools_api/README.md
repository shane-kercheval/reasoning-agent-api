# Tools API

Structured tool and prompt execution service for the reasoning agent.

## Overview

Tools API provides tool and prompt execution via two interfaces:

1. **REST API** - Clean JSON endpoints for direct HTTP integration
2. **MCP Protocol** - Model Context Protocol endpoint for MCP-compatible clients

Both interfaces expose the same tools and prompts with:

- **Structured responses** with metadata (file stats, execution time, etc.)
- **Direct implementations** (file-system via pathlib, GitHub via API, etc.)
- **Path security** with blocked patterns and access control

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
│  │ - MCP protocol endpoint (/mcp/)                  │   │
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
│   ├── tools/           # Tool implementations (file-system, github, search, etc.)
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
→ [{"name": "read_text_file", "description": "...", "parameters": {...}, "tags": ["file-system", "read"]}]

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

### MCP Protocol

The MCP endpoint exposes the same tools and prompts via the Model Context Protocol (JSON-RPC over HTTP).

```bash
# MCP health check
GET /mcp/health
→ {"status": "healthy", "transport": "streamable-http", "tools_count": 15, "prompts_count": 3}

# MCP protocol endpoint (for MCP clients)
POST /mcp/
# Accepts JSON-RPC requests per MCP specification

# Test with MCP Inspector
npx @modelcontextprotocol/inspector http://localhost:8001/mcp
```

MCP clients can use the `streamablehttp_client` transport to connect:

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async with streamablehttp_client("http://localhost:8001/mcp") as (read, write, _):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        result = await session.call_tool("list_allowed_directories", {})
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

## Adding Prompts

Prompts are markdown files with YAML frontmatter. Place them in the directory specified by `PROMPTS_HOST_PATH` (mounted to `/mnt/prompts` in the container).

### Prompt File Format

```markdown
---
name: my-prompt
description: What this prompt does
category: development
arguments:
  - name: input_var
    required: true
    description: Description of this argument
  - name: optional_var
    required: false
    description: Optional argument with default behavior
tags:
  - tag1
  - tag2
---
Your prompt content here with Jinja2 templating.

Use {{ input_var }} to insert arguments.

{% if optional_var %}
Conditional content based on optional argument.
{% endif %}
```

### Directory Structure

Prompts are loaded recursively from subdirectories:

```
prompts/
├── development/
│   ├── code-review.md
│   └── unit-tests.md
├── meta/
│   └── generate-prompt.md
└── thinking/
    └── explain.md
```

Each prompt's `name` field becomes its API endpoint (e.g., `code-review` → `POST /prompts/code-review`).

## Architecture Notes

This service provides both REST and MCP interfaces:

- **REST API**: Primary interface for reasoning-api integration, returns structured JSON with metadata
- **MCP Protocol**: Standard MCP endpoint for MCP-compatible clients (Claude Desktop, MCP Inspector, etc.)

Both interfaces use the same underlying tool and prompt implementations, ensuring consistent behavior.
