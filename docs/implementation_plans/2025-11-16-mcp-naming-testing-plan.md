# MCP Naming Conventions - Testing Plan

**Date**: 2025-11-16
**Status**: Proposal
**Related**: `2025-11-16-mcp-naming-conventions.md`

## Overview

Comprehensive testing strategy for MCP naming conventions feature, covering:
- Name parsing logic
- Prefix stripping
- Duplicate detection
- Override configuration
- Tool and Prompt conversion
- Integration with existing systems

## Testing Architecture

### Test Organization

```
tests/
├── unit_tests/
│   ├── test_mcp_name_parsing.py        # NEW: Core parsing logic
│   ├── test_mcp_override_config.py     # NEW: Override loading/application
│   ├── test_mcp_to_tool.py             # MODIFY: Add naming tests
│   ├── test_mcp_to_prompt.py           # MODIFY: Add naming tests
│   └── test_mcp_duplicate_detection.py # NEW: Duplicate detection
│
├── integration_tests/
│   ├── test_mcp_naming_e2e.py          # NEW: End-to-end naming flow
│   └── test_mcp_endpoints.py           # MODIFY: Verify API responses
│
└── fixtures/
    └── mcp_naming.py                   # NEW: Shared naming test fixtures
```

## Unit Tests

### 1. Name Parsing Tests (`test_mcp_name_parsing.py`)

**Pattern**: Pure function tests with parametrize for multiple cases

**Test Cases**:

```python
class TestParseMCPName:
    """Test core name parsing logic."""

    @pytest.mark.parametrize("raw_name,expected_base,expected_server", [
        # Basic patterns
        ("github_custom__get_pr_info", "get_pr_info", "github-custom"),
        ("filesystem__read_file", "read_file", "filesystem"),
        ("meta__generate_playbook", "generate_playbook", "meta"),

        # With bridge prefix (should be stripped)
        ("local_bridge_github_custom__get_pr_info", "get_pr_info", "github-custom"),
        ("local_bridge_filesystem__read", "read", "filesystem"),

        # Edge cases
        ("simple_tool", "simple_tool", None),  # No server prefix
        ("a__b__c", "c", "a__b"),  # Multiple __ separators
        ("tool_name", "name", "tool"),  # Single _ separator (fallback)
    ])
    def test_parse_basic_patterns(self, raw_name, expected_base, expected_server):
        """Test parsing of various MCP name patterns."""

    def test_parse_with_configured_strip_prefixes(self):
        """Test prefix stripping with MCP_STRIP_PREFIXES."""
        # Config with multiple prefixes

    def test_parse_without_strip_prefixes(self):
        """Test parsing when no strip prefixes configured."""

    def test_parse_empty_name(self):
        """Test error handling for empty/invalid names."""

    def test_server_name_normalization(self):
        """Test server name normalization (github-custom from github_custom)."""
```

**Mocking Pattern**: None needed - pure functions

**Fixtures Needed**:
- `naming_config` - NamingConfig instance with test settings

---

### 2. Override Configuration Tests (`test_mcp_override_config.py`)

**Pattern**: File-based config tests using tmp_path fixture (like `test_mcp_env_vars.py`)

**Test Cases**:

```python
class TestOverrideConfigLoading:
    """Test loading and parsing of override configuration."""

    def test_load_valid_config(self, tmp_path):
        """Test loading valid YAML override config."""
        # Create config with tools, prompts, server_tags
        # Verify NamingConfig loads correctly

    def test_load_empty_config(self, tmp_path):
        """Test loading empty/minimal config."""

    def test_load_invalid_yaml(self, tmp_path):
        """Test error handling for invalid YAML."""

    def test_missing_config_file(self):
        """Test behavior when override config doesn't exist."""
        # Should not fail, just use defaults

class TestOverrideApplication:
    """Test applying overrides to parsed names."""

    def test_tool_override_full_match(self):
        """Test exact match tool override."""
        # Override "local_bridge_github__tool" -> name: "custom_name"

    def test_prompt_override_full_match(self):
        """Test exact match prompt override."""

    def test_no_override_uses_parsed(self):
        """Test fallback to parsed name when no override."""

    def test_override_priority_over_parsing(self):
        """Test override takes priority over strip/parse."""
        # Even if strip would give different name, override wins

    def test_server_tags_applied(self):
        """Test server-level tags applied to tools/prompts."""
        # github-custom: [git, github] -> all github-custom tools get tags
```

**Mocking Pattern**: File system with tmp_path, no MCP client needed

**Fixtures Needed**:
- `sample_override_config` - Pre-built YAML config for testing
- `naming_config_factory` - Factory to create NamingConfig with custom settings

---

### 3. Duplicate Detection Tests (`test_mcp_duplicate_detection.py`)

**Pattern**: Mock MCP client like `test_mcp_to_tool.py`

**Test Cases**:

```python
class TestDuplicateDetection:
    """Test duplicate name detection and error handling."""

    @pytest.mark.asyncio
    async def test_duplicate_tools_raises_error(self):
        """Test that duplicate tool names raise ValueError."""
        # Mock client with two tools that parse to same name
        # e.g., "filesystem__search" and "brave_search__search" -> both "search"
        # Expect ValueError with helpful message

    @pytest.mark.asyncio
    async def test_duplicate_prompts_raises_error(self):
        """Test that duplicate prompt names raise ValueError."""

    @pytest.mark.asyncio
    async def test_no_duplicates_succeeds(self):
        """Test that unique names process successfully."""
        # Mock client with unique tool names
        # Should complete without error

    @pytest.mark.asyncio
    async def test_duplicate_error_message_format(self):
        """Test that error message includes helpful resolution info."""
        # Verify error shows both conflicting tools
        # Verify error suggests override config
        # Verify error shows full MCP names

    @pytest.mark.asyncio
    async def test_duplicate_across_servers(self):
        """Test duplicate detection across multiple MCP servers."""
        # Server A has "tool", Server B has "tool"
        # After stripping prefixes, both become "tool" -> error

    @pytest.mark.asyncio
    async def test_override_resolves_duplicate(self):
        """Test that override config resolves duplicates."""
        # Two tools would conflict, but override gives them different names
        # Should succeed
```

**Mocking Pattern**: Mock FastMCP Client (from `test_mcp_to_tool.py`)

**Fixtures Needed**:
- `duplicate_tools_mock_client` - Client that returns duplicate names
- `unique_tools_mock_client` - Client that returns unique names

---

### 4. Modified Tool Conversion Tests (`test_mcp_to_tool.py`)

**Pattern**: Add to existing test class

**New Test Cases**:

```python
class TestMCPToToolConversion:
    # Existing tests...

    @pytest.mark.asyncio
    async def test_tool_has_metadata_fields(self):
        """Test that converted tools include new metadata fields."""
        # Verify tool.server_name, tool.tags, tool.mcp_name present

    @pytest.mark.asyncio
    async def test_tool_name_stripped(self):
        """Test that tool name is stripped of prefixes."""
        # Mock tool with "local_bridge_github__tool"
        # Verify tool.name == "tool" (clean)
        # Verify tool.mcp_name == "local_bridge_github__tool" (original)

    @pytest.mark.asyncio
    async def test_tool_server_name_extracted(self):
        """Test server name extracted from MCP name."""
        # "github_custom__tool" -> server_name="github-custom"

    @pytest.mark.asyncio
    async def test_tool_tags_from_server_config(self):
        """Test tags applied from server_tags config."""

    @pytest.mark.asyncio
    async def test_tool_wrapper_uses_mcp_name(self):
        """Test that wrapper function calls MCP with original name."""
        # Tool name is "get_pr" but mcp_name is "local_bridge_github__get_pr"
        # When calling tool(), verify client.call_tool() gets mcp_name
```

**Mocking Pattern**: Existing pattern (Mock FastMCP Client)

---

### 5. Modified Prompt Conversion Tests (`test_mcp_to_prompt.py`)

**Pattern**: Mirror tool tests for prompts

**New Test Cases**: Same as tool tests but for prompts

---

## Integration Tests

### 6. End-to-End Naming Tests (`test_mcp_naming_e2e.py`)

**Pattern**: Use real FastMCP servers (like `test_mcp_to_tool_multiple_servers.py`)

**Test Cases**:

```python
@pytest.mark.integration
class TestNamingEndToEnd:
    """Integration tests with real MCP servers."""

    @pytest.mark.asyncio
    async def test_single_server_clean_names(self):
        """Test clean names with single MCP server."""
        # Use in-memory FastMCP server
        # Verify tool names are clean (no prefixes)

    @pytest.mark.asyncio
    async def test_multiple_servers_with_overrides(self, tmp_path):
        """Test multiple servers with override config."""
        # Create override config that handles potential duplicates
        # Create multiple in-memory servers
        # Verify overrides applied correctly

    @pytest.mark.asyncio
    async def test_bridge_scenario(self, tmp_path):
        """Test realistic bridge scenario."""
        # Simulate: API -> local_bridge -> multiple stdio servers
        # Verify triple-prefixing is stripped correctly

    @pytest.mark.asyncio
    async def test_tool_execution_with_clean_names(self):
        """Test that tools work correctly with clean names."""
        # Create tool with clean name
        # Call it via clean name
        # Verify underlying MCP call uses mcp_name
```

**Mocking Pattern**: Real FastMCP servers using mount/composition

**Fixtures Needed**:
- Reuse existing server fixtures from `test_mcp_to_tool_multiple_servers.py`

---

### 7. Modified API Endpoint Tests (`test_mcp_endpoints.py`)

**Pattern**: Add to existing endpoint tests

**New Test Cases**:

```python
@pytest.mark.integration
class TestMCPEndpoints:
    # Existing tests...

    @pytest.mark.asyncio
    async def test_tools_endpoint_returns_clean_names(self):
        """Test /tools endpoint returns clean names."""
        # Call /tools endpoint
        # Verify response has clean names, metadata fields

    @pytest.mark.asyncio
    async def test_tools_endpoint_includes_metadata(self):
        """Test /tools endpoint includes server_name, tags."""
        # Verify JSON response structure

    @pytest.mark.asyncio
    async def test_prompts_endpoint_returns_clean_names(self):
        """Test /prompts endpoint returns clean names."""
```

**Mocking Pattern**: FastAPI TestClient with real app

---

## Test Fixtures

### New Fixtures File (`tests/fixtures/mcp_naming.py`)

```python
"""Fixtures for MCP naming convention tests."""

import pytest
from pathlib import Path
from api.mcp import NamingConfig

@pytest.fixture
def sample_override_config(tmp_path: Path) -> Path:
    """Create sample override config for testing."""
    config_content = """
server_tags:
  github-custom: [git, github]
  filesystem: [files]

tools:
  "local_bridge_github__get_pr":
    name: get_pr_info
    tags: [git, pull-request]
"""
    config_file = tmp_path / "mcp_overrides.yaml"
    config_file.write_text(config_content)
    return config_file

@pytest.fixture
def naming_config(sample_override_config: Path) -> NamingConfig:
    """Create NamingConfig instance for testing."""
    return NamingConfig(
        override_path=sample_override_config,
        strip_prefixes=["local_bridge_"],
    )

@pytest.fixture
def duplicate_tools_scenario():
    """Mock data for duplicate tool testing."""
    return {
        "tool1": {"mcp_name": "local_bridge_filesystem__search", "base": "search"},
        "tool2": {"mcp_name": "local_bridge_brave__search", "base": "search"},
    }
```

---

## Testing Strategy Summary

### Coverage Goals

| Component | Coverage Target | Strategy |
|-----------|----------------|----------|
| Name parsing | 100% | Parametrize for all patterns |
| Override loading | 100% | File-based tests with tmp_path |
| Duplicate detection | 100% | Mock scenarios for all cases |
| Tool/Prompt conversion | 95%+ | Mock + integration tests |
| API endpoints | 90%+ | Integration with TestClient |

### Test Execution

**Fast unit tests** (non-integration):
```bash
make non_integration_tests
```

**Integration tests** (requires services):
```bash
make integration_tests
```

**Full test suite**:
```bash
make tests
```

### Key Testing Patterns

1. **Pure Functions** (`parse_mcp_name`)
   - No mocking needed
   - Use parametrize for comprehensive coverage
   - Test edge cases and error handling

2. **Config Loading** (`NamingConfig`)
   - Use tmp_path for file system
   - Test YAML parsing, validation
   - Test missing/invalid files

3. **MCP Integration** (`to_tools`, `to_prompts`)
   - Mock FastMCP Client with AsyncMock
   - Test metadata extraction
   - Verify wrapper functions work

4. **Duplicate Detection**
   - Mock scenarios with duplicate names
   - Verify error messages
   - Test resolution via overrides

5. **End-to-End**
   - Real FastMCP servers (in-memory)
   - Full flow from MCP -> API
   - Verify actual behavior

### Continuous Integration

- All tests run on PR/commit
- Fast unit tests run first (fail fast)
- Integration tests run after unit tests pass
- Coverage report generated and checked

---

## Migration Testing

### Backward Compatibility Tests

```python
class TestBackwardCompatibility:
    """Ensure new naming doesn't break existing functionality."""

    @pytest.mark.asyncio
    async def test_existing_tools_still_work(self):
        """Test that existing tool calls still function."""
        # Tools should work with old or new names

    @pytest.mark.asyncio
    async def test_env_var_strip_prefixes_still_works(self):
        """Test MCP_STRIP_PREFIXES environment variable."""
        # Existing env var should continue working
```

---

## Success Criteria

- [ ] All new test files created and passing
- [ ] Existing test files modified and passing
- [ ] 95%+ code coverage on new naming logic
- [ ] Integration tests verify end-to-end flow
- [ ] No regression in existing tests
- [ ] Tests run in <30 seconds (unit), <2 minutes (integration)
- [ ] Clear error messages in test failures
- [ ] Fixtures reusable across test files

---

## Implementation Order

1. **Phase 1: Core Parsing** - `test_mcp_name_parsing.py`
2. **Phase 2: Override Config** - `test_mcp_override_config.py`
3. **Phase 3: Duplicate Detection** - `test_mcp_duplicate_detection.py`
4. **Phase 4: Modify Existing** - Update `test_mcp_to_tool.py`, `test_mcp_to_prompt.py`
5. **Phase 5: Integration** - `test_mcp_naming_e2e.py`
6. **Phase 6: API Endpoints** - Update `test_mcp_endpoints.py`

---

## Example Test Structure

```python
"""Example test showing recommended patterns."""

import pytest
from unittest.mock import Mock, AsyncMock
from pathlib import Path

from api.mcp import parse_mcp_name, to_tools, NamingConfig

class TestMCPNameParsing:
    """Example unit test class."""

    @pytest.mark.parametrize("input,expected", [
        ("github__tool", ("tool", "github")),
        ("local_bridge_github__tool", ("tool", "github")),
    ])
    def test_parsing_patterns(self, input, expected):
        """Test multiple patterns with parametrize."""
        config = NamingConfig(strip_prefixes=["local_bridge_"])
        result = parse_mcp_name(input, config, "tool")
        assert result.base_name == expected[0]
        assert result.server_name == expected[1]

@pytest.mark.integration
class TestMCPNamingIntegration:
    """Example integration test class."""

    @pytest.mark.asyncio
    async def test_end_to_end(self, tmp_path: Path):
        """Test complete flow with config file."""
        # Create override config
        config_file = tmp_path / "overrides.yaml"
        config_file.write_text("tools: {}")

        # Create mock client
        mock_client = Mock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        # Test conversion
        tools = await to_tools(mock_client)
        assert len(tools) > 0
```
