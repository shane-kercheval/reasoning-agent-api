# MCP Naming Conventions - Implementation Plan

**Date**: 2025-11-16
**Status**: Proposal
**Scope**: Tools and Prompts

## Problem Statement

Current MCP tool/prompt naming creates verbose, confusing names due to triple-layer prefixing:

1. **Bridge layer**: `github_custom__get_pr_info` (server + `__` + name)
2. **FastMCP Client layer**: `local_bridge_github_custom__get_pr_info` (adds bridge prefix with `_`)
3. **Result**: Long, confusing names that are hard for LLMs to select and users to call

**Issues**:
- Names polluted with infrastructure details (`local_bridge_`)
- Inconsistent separators (single `_` vs double `__`)
- Complexity added for rare edge case (duplicate names across servers)
- Poor UX for direct tool/prompt calls

## Objectives

1. **Clean, LLM-friendly names**: `get_pr_info` not `local_bridge_github_custom__get_pr_info`
2. **Fail fast on conflicts**: Explicit error when duplicate names exist
3. **Flexible overrides**: Manual control for edge cases via config
4. **Rich metadata**: Server source and semantic tags for context
5. **Reusable logic**: Same approach for both tools and prompts
6. **Simple defaults**: Works out-of-box, config only for customization

## Proposed Solution

### Enhanced Models

**Tool Model** (`api/tools.py`):
```python
class Tool(BaseModel):
    name: str                # "get_pr_info" - clean base name
    description: str
    input_schema: dict
    function: Callable

    # NEW: Metadata fields
    server_name: str | None  # "github-custom" - source MCP server
    tags: list[str]          # ["git", "pull-request"] - semantic grouping (validated)
    mcp_name: str = Field(exclude=True)  # Internal - not serialized in API responses

    # Note: mcp_name is "local_bridge_github_custom__get_pr_info" - original MCP identifier
    # Used internally for client.call_tool() but hidden from API consumers
```

**Prompt Model** (`api/prompts.py`):
```python
class Prompt(BaseModel):
    name: str                # "generate_playbook" - clean base name
    description: str
    arguments: list[dict]
    function: Callable

    # NEW: Metadata fields (same as Tool)
    server_name: str | None  # "meta" - source MCP server
    tags: list[str]          # ["automation", "templates"] - semantic grouping (validated)
    mcp_name: str = Field(exclude=True)  # Internal - not serialized in API responses
```

### Shared Processing Logic

**Core Parsing** (`api/mcp.py`):

```python
@dataclass
class ParsedMCPName:
    """Parsed components of an MCP tool/prompt name."""
    base_name: str           # "get_pr_info"
    server_name: str | None  # "github-custom"
    tags: list[str]          # ["git", "pull-request"]
    mcp_name: str           # Original full name

def parse_mcp_name(
    raw_name: str,
    config: NamingConfig,
    resource_type: str  # "tool" or "prompt"
) -> ParsedMCPName:
    """
    Parse MCP tool/prompt name into components.

    Process:
    1. Check manual override (exact match only - highest priority)
    2. Strip configured prefixes (MCP_STRIP_PREFIXES)
    3. Parse server__name pattern
    4. Merge server-level tags with resource-specific tags
    5. Validate all tags against format rules

    Returns parsed components for Tool/Prompt creation.
    """

def validate_tag(tag: str) -> None:
    """
    Validate tag format.

    Rules:
    - Lowercase alphanumeric with hyphens
    - Must start and end with alphanumeric
    - Pattern: ^[a-z0-9][a-z0-9-]*[a-z0-9]$

    Valid: "git", "pull-request", "code-review"
    Invalid: "Git", "pull request", "-git", "git_pull"
    """
```

**Reusable for both**:
- `to_tools()` calls `parse_mcp_name(name, config, "tool")`
- `to_prompts()` calls `parse_mcp_name(name, config, "prompt")`

### Configuration

**Environment Variables**:
```bash
# Prefixes to strip during parsing
MCP_STRIP_PREFIXES=local_bridge_,brave_search__

# Override config path
MCP_TOOL_OVERRIDES_PATH=config/mcp_overrides.yaml
```

**Override Config** (`config/mcp_overrides.yaml`):

```yaml
# Server-level default tags (merged with resource-specific tags)
# All tools/prompts from a server inherit these tags
server_tags:
  github-custom: [git, github, version-control]
  filesystem: [files, storage]
  meta: [automation, templates]
  thinking: [reasoning, analysis]

# Tool-specific overrides (EXACT MATCH ONLY - no regex patterns)
tools:
  # Match against original MCP name (before stripping)
  # Tags are MERGED with server_tags (inheritance)
  "local_bridge_github_custom__get_pr_info":
    name: get_pr_info
    tags: [pull-request]  # Result: [git, github, version-control, pull-request]

  # Resolve naming conflicts
  "local_bridge_filesystem__search":
    name: filesystem_search
    tags: [search]  # Result: [files, storage, search]

  "local_bridge_brave_search__search":
    name: web_search
    tags: [web, search]  # brave-search has no server_tags, so just these

# Prompt-specific overrides (EXACT MATCH ONLY - no regex patterns)
prompts:
  "local_bridge_meta__generate_playbook":
    name: generate_playbook
    tags: [playbook]  # Result: [automation, templates, playbook]

  "local_bridge_thinking__analyze":
    name: think
    tags: [deep-analysis]  # Result: [reasoning, analysis, deep-analysis]

# Tag Validation Rules:
# - Lowercase only: "git" not "Git"
# - Alphanumeric + hyphens: "pull-request" not "pull_request" or "pull request"
# - Start/end with alphanumeric: "git" not "-git" or "git-"
# - Pattern: ^[a-z0-9][a-z0-9-]*[a-z0-9]$
```

### Processing Flow

**For both tools and prompts**:

1. **Get from FastMCP**: Receive raw name (e.g., `local_bridge_github_custom__get_pr_info`)

2. **Check override** (priority #1):
   - Look in `tools` or `prompts` section of config
   - If found, use override values completely
   - If not found, continue to auto-parsing

3. **Auto-parse** (priority #2):
   - Strip configured prefixes: `github_custom__get_pr_info`
   - Parse pattern: `server_name="github-custom"`, `base_name="get_pr_info"`
   - Merge server-level tags with resource-specific tags

4. **Validate tags**:
   - Check all tags match pattern: `^[a-z0-9][a-z0-9-]*[a-z0-9]$`
   - Fail fast with clear error if invalid tags found

5. **Detect duplicates**:
   - Check if `name` already exists in collection
   - If duplicate: **always fail** with helpful error
   - Error shows conflicting items and suggests override config
   - No configuration option - duplicates always require explicit resolution

6. **Create object**:
   - Tool/Prompt with clean `name`, metadata fields, wrapper function
   - `mcp_name` excluded from serialization (internal only)
   - Function wrapper internally calls MCP using `mcp_name`

### Error Handling

**Duplicate Detection**:
```
Error: Duplicate tool name 'search' detected:
  - local_bridge_filesystem__search (from filesystem server)
  - local_bridge_brave_search__search (from brave-search server)

Resolve by adding overrides in config/mcp_overrides.yaml:

tools:
  "local_bridge_filesystem__search":
    name: filesystem_search
  "local_bridge_brave_search__search":
    name: web_search
```

## Implementation Changes

### Files to Modify

| File | Changes | Complexity |
|------|---------|------------|
| `api/tools.py` | Add `server_name`, `tags`, `mcp_name` fields to Tool | Low |
| `api/prompts.py` | Add `server_name`, `tags`, `mcp_name` fields to Prompt | Low |
| `api/mcp.py` | New `parse_mcp_name()`, `validate_tag()`, `NamingConfig` class | Medium |
| `api/mcp.py:to_tools()` | Use new parsing logic, tag validation, duplicate detection | Medium |
| `api/mcp.py:to_prompts()` | Use new parsing logic, tag validation, duplicate detection | Medium |
| `api/config.py` | Add `mcp_tool_overrides_path` setting | Low |

### New Files

| File | Purpose |
|------|---------|
| `config/mcp_overrides.yaml` | Manual naming overrides and server tag defaults |
| `config/mcp_overrides.example.yaml` | Example configuration with comments |

## Benefits

### For LLMs
- **Clean names**: `get_pr_info` vs `local_bridge_github_custom__get_pr_info`
- **Semantic tags**: Better tool selection via `["git", "pull-request"]`
- **Server context**: Know source without polluting name

### For Users
- **Simple calls**: `await get_pr_info(...)` not `await local_bridge_github_custom__get_pr_info(...)`
- **Clear organization**: Tools/prompts grouped by tags and server
- **Explicit errors**: Know immediately when conflicts exist

### For Developers
- **Reusable logic**: Same code handles tools and prompts
- **Configurable**: Override anything via YAML
- **Debuggable**: `mcp_name` and `server_name` trace back to source
- **Testable**: Duplicate detection prevents silent failures

## Migration Path

### Phase 1: Add Fields (Non-Breaking)
- Add new fields to Tool/Prompt models (optional/nullable)
- Existing code continues working

### Phase 2: Implement Parsing
- Add `parse_mcp_name()` function
- Update `to_tools()` and `to_prompts()` to use it
- Keep backward compatibility with environment variable stripping

### Phase 3: Enable by Default
- Document new naming convention
- Provide example override config
- Duplicate detection always enabled (no config needed)

## Example: Before & After

### Before
```json
{
  "tools": [
    {"name": "local_bridge_github_custom__get_pr_info", "description": "..."},
    {"name": "local_bridge_filesystem__read_file", "description": "..."}
  ]
}
```

### After
```json
{
  "tools": [
    {
      "name": "get_pr_info",
      "server_name": "github-custom",
      "tags": ["git", "github", "version-control", "pull-request"],
      "description": "..."
    },
    {
      "name": "read_file",
      "server_name": "filesystem",
      "tags": ["files", "storage"],
      "description": "..."
    }
  ]
}
```

**Note**: `mcp_name` field is NOT included in API responses (internal only)

## Design Decisions

### 1. Serialization
**Decision**: `mcp_name` is excluded from API responses (`Field(exclude=True)`)

**Rationale**: Internal implementation detail, not relevant to API consumers

### 2. Override Matching
**Decision**: Exact string matches only - no regex patterns

**Rationale**: Simpler, more predictable, easier to debug. Can add patterns later if needed.

### 3. Tag Inheritance
**Decision**: Merge server tags with resource-specific tags

**Example**:
```yaml
server_tags:
  github-custom: [git, github]

tools:
  "github_custom__get_pr":
    tags: [pull-request]
    # Result: [git, github, pull-request]  ← Merged
```

**Rationale**: DRY principle - don't repeat server tags for every tool

### 4. Tag Validation
**Decision**: Enforce format rules - lowercase alphanumeric with hyphens

**Pattern**: `^[a-z0-9][a-z0-9-]*[a-z0-9]$`

**Valid**: `git`, `pull-request`, `code-review`

**Invalid**: `Git`, `pull request`, `-git`, `git_pull`

**Rationale**: Consistency, URL-safe, prevents confusion

## Testing

See detailed testing plan: `2025-11-16-mcp-naming-testing-plan.md`

**Summary**:
- Unit tests for parsing, overrides, duplicate detection, tag validation
- Integration tests for end-to-end flow
- Existing tests modified to verify new metadata fields
- 95%+ code coverage target
- Mock-based for unit tests, real servers for integration

## Success Criteria

- [ ] Tool/prompt names reduced from avg 40+ chars to <20 chars
- [ ] Zero duplicate name conflicts in production
- [ ] Same parsing logic used for both tools and prompts
- [ ] Override config supports all edge cases (exact matches)
- [ ] Tag validation enforces format rules
- [ ] Tag inheritance merges server + resource tags
- [ ] `mcp_name` excluded from API responses
- [ ] Clear error messages guide users to resolution
- [ ] 95%+ test coverage
- [ ] Documentation updated with examples
