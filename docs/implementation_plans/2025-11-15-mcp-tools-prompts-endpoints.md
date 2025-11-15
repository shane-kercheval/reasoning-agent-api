# Implementation Plan: MCP Prompts Support and Tools/Prompts Endpoints

**Created**: 2025-11-15
**Status**: Draft
**Goal**: Add MCP prompts support to the API (similar to existing MCP tools functionality) and create endpoints to expose both tools and prompts to end users.

---

## Background

Currently, the API supports MCP tools through:
- `api/tools.py`: Generic `Tool` abstraction for any callable
- `api/mcp.py`: Conversion from MCP client tools to `Tool` objects via `to_tools()`
- `api/dependencies.py`: `get_tools()` dependency that retrieves tools from MCP client
- `api/main.py`: `/tools` endpoint that lists available tool names

The MCP bridge (`mcp_bridge/server.py`) already supports both tools AND prompts from stdio MCP servers, proving that FastMCP provides prompt functionality via `list_prompts()` and `get_prompt()`.

**What's Missing:**
1. Prompt abstraction (similar to `Tool` class)
2. Conversion from MCP prompts to prompt objects (similar to `to_tools()`)
3. Dependency injection for prompts (similar to `get_tools()`)
4. API endpoints to expose tools and prompts to end users

**Important Distinction:**
- **PromptManager** (`api/prompt_manager.py`): Internal API prompts for reasoning agents (system prompts, templates, etc.)
- **MCP Prompts**: External prompts from MCP servers, designed for agents/end users to invoke
- These remain separate - this plan only addresses MCP prompts

---

## Key Documentation

**FastMCP Prompts:**
- Official docs: https://gofastmcp.com/servers/prompts
- Prompts are reusable message templates with parameters
- Defined via `@mcp.prompt` decorator in MCP servers
- Clients access via `client.list_prompts()` and `client.get_prompt(name, arguments)`
- Prompt objects have: `name`, `description`, `arguments` (list of argument specs)
- Results are `PromptMessage` objects with `role` and `content`

**Existing Patterns:**
- `tests/unit_tests/test_mcp_to_tool.py`: Testing patterns for MCP conversion
- `tests/unit_tests/test_api.py`: Testing patterns for API endpoints
- `tests/integration_tests/test_mcp_bridge_http.py`: Shows prompts working in bridge (lines 237-280)

---

## Milestones

### Milestone 1: Create Prompt Abstraction

**Goal**: Create a generic `Prompt` class (similar to `Tool`) to represent MCP prompts in a source-agnostic way.

**Success Criteria:**
- `Prompt` class in `api/prompts.py` with name, description, arguments, callable function
- `PromptResult` class to represent execution results
- `prompt_to_dict()` method for serialization
- `format_prompt_for_display()` utility for user-facing documentation
- Comprehensive unit tests in `tests/unit_tests/test_prompts.py`

**Key Changes:**

Create `api/prompts.py`:
```python
class PromptResult(BaseModel):
    """Result from prompt execution."""
    prompt_name: str
    success: bool
    messages: list[dict] | None  # MCP PromptMessage format
    error: str | None
    execution_time_ms: float

class Prompt(BaseModel):
    """
    Generic prompt interface abstracting MCP prompts.

    Similar to Tool, but for prompt templates instead of actions.
    Prompts return message sequences for LLM consumption.
    """
    name: str
    description: str
    arguments: list[dict[str, Any]]  # Argument specs (name, required, description)
    function: Callable  # Async function that returns PromptResult

    async def __call__(self, **kwargs) -> PromptResult:
        """Execute prompt with arguments, validate inputs, handle errors."""
        pass
```

**Testing Strategy:**
- Test prompt creation with various argument configurations
- Test async execution with valid/invalid arguments
- Test input validation (required arguments, types)
- Test error handling (missing args, execution errors)
- Test serialization (`prompt_to_dict()`)
- Test formatting for display

**Dependencies:** None

**Risk Factors:**
- Prompt argument validation may differ from tool validation
- MCP PromptMessage format must match exactly for LLM consumption

---

### Milestone 2: MCP Prompts Conversion

**Goal**: Add `to_prompts()` function to convert FastMCP client prompts to generic `Prompt` objects (analogous to `to_tools()`).

**Success Criteria:**
- `to_prompts(client)` function in `api/mcp.py`
- Wrapper functions that call `client.get_prompt()` with proper context management
- Handles missing descriptions/arguments gracefully
- Returns list of working `Prompt` objects
- Comprehensive unit tests in `tests/unit_tests/test_mcp_to_prompt.py`

**Key Changes:**

Add to `api/mcp.py`:
```python
async def to_prompts(client: Client) -> list[Prompt]:
    """
    Convert MCP client prompts to generic Prompt objects.

    Creates wrappers that manage client context when called.
    Prompt names are automatically prefixed by FastMCP if multiple servers.

    Args:
        client: Configured FastMCP Client instance

    Returns:
        List of Prompt objects from all connected servers
    """
    async with client:
        mcp_prompts = await client.list_prompts()

    prompts = []
    for mcp_prompt in mcp_prompts:
        # Create wrapper function
        def create_prompt_wrapper(prompt_name: str = mcp_prompt.name) -> Callable:
            async def wrapper(**kwargs) -> PromptResult:
                async with client:
                    result = await client.get_prompt(prompt_name, kwargs)
                    # Convert MCP result to PromptResult
                    return PromptResult(
                        prompt_name=prompt_name,
                        success=True,
                        messages=[msg.model_dump() for msg in result.messages],
                        execution_time_ms=...,
                    )
            return wrapper

        prompt = Prompt(
            name=mcp_prompt.name,
            description=mcp_prompt.description or "No description",
            arguments=mcp_prompt.arguments or [],
            function=create_prompt_wrapper(),
        )
        prompts.append(prompt)

    return prompts
```

**Testing Strategy:**
- Mock FastMCP client and MCP prompt objects
- Test basic conversion (single prompt)
- Test multiple prompts
- Test missing description/arguments
- Test prompt execution (wrapper calls client.get_prompt correctly)
- Test error handling (list_prompts fails, get_prompt fails)
- Test complex argument schemas
- Test prompt name prefixing (server__prompt format)

**Testing Pattern** (follow `test_mcp_to_tool.py`):
```python
def create_mock_client():
    """Create mock FastMCP client with async context manager."""
    mock_client = Mock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client

async def test_to_prompts_basic_conversion():
    mock_client = create_mock_client()

    # Mock MCP prompt
    mock_mcp_prompt = Mock()
    mock_mcp_prompt.name = "ask_question"
    mock_mcp_prompt.description = "Generate a question"
    mock_mcp_prompt.arguments = [
        {"name": "topic", "required": True, "description": "Topic to ask about"}
    ]

    # Mock prompt result
    mock_prompt_result = Mock()
    mock_prompt_result.messages = [
        Mock(role="user", content=Mock(text="What is Python?"))
    ]

    mock_client.list_prompts = AsyncMock(return_value=[mock_mcp_prompt])
    mock_client.get_prompt = AsyncMock(return_value=mock_prompt_result)

    prompts = await to_prompts(mock_client)

    assert len(prompts) == 1
    assert prompts[0].name == "ask_question"

    # Test prompt execution
    result = await prompts[0](topic="Python")
    assert result.success is True
    assert len(result.messages) == 1
```

**Dependencies:** Milestone 1 (Prompt abstraction)

**Risk Factors:**
- MCP prompt result format may vary by server
- PromptMessage objects need proper serialization
- Context manager usage with nested async calls

---

### Milestone 3: Prompts Dependency Injection

**Goal**: Add `get_prompts()` dependency to retrieve MCP prompts (analogous to `get_tools()`).

**Success Criteria:**
- `get_prompts()` async function in `api/dependencies.py`
- `PromptsDependency` type alias for endpoint signatures
- Handles missing MCP client gracefully (returns empty list)
- Error logging for failures
- Unit tests confirming dependency behavior

**Key Changes:**

Add to `api/dependencies.py`:
```python
async def get_prompts() -> list[Prompt]:
    """Get available prompts from MCP servers."""
    mcp_client = service_container.mcp_client

    if mcp_client is None:
        logger.info("No MCP client available, returning empty prompts list")
        return []

    try:
        async with mcp_client:
            prompts = await to_prompts(mcp_client)
            logger.info(f"Loaded {len(prompts)} prompts from MCP servers")
            return prompts
    except Exception as e:
        logger.error(f"Failed to load MCP prompts: {e}")
        return []

# Type alias
PromptsDependency = Annotated[list[Prompt], Depends(get_prompts)]
```

**Testing Strategy:**
- Test with valid MCP client (returns prompts)
- Test with no MCP client (returns empty list)
- Test with client.list_prompts() error (logs error, returns empty list)
- Test integration with service container lifecycle
- Verify logging messages

**Dependencies:** Milestone 2 (to_prompts conversion)

**Risk Factors:**
- Similar to get_tools(), must handle client lifecycle properly
- Error handling must be robust to not break API startup

---

### Milestone 4: API Endpoints for Tools and Prompts

**Goal**: Create REST API endpoints to expose available tools and prompts to end users.

**Success Criteria:**
- `GET /v1/mcp/tools` endpoint returning tool metadata
- `GET /v1/mcp/prompts` endpoint returning all prompts metadata
- `GET /v1/mcp/prompts/{prompt_name}` endpoint returning specific prompt details
- All endpoints require authentication
- Legacy `/tools` endpoint removed (breaking change)
- Proper error handling and logging
- OpenAPI documentation auto-generated
- Unit and integration tests

**Key Changes:**

Add to `api/main.py`:
```python
@app.get("/v1/mcp/tools")
async def list_mcp_tools(
    tools: ToolsDependency,
    _: bool = Depends(verify_token),
) -> dict[str, list[dict[str, Any]]]:
    """
    List available MCP tools with metadata.

    Returns tool names, descriptions, and input schemas for discovery.
    Clients can use this to understand what tools are available.

    Returns:
        {"tools": [{"name": "...", "description": "...", "input_schema": {...}}, ...]}
    """
    try:
        return {
            "tools": [tool.to_dict() for tool in tools]
        }
    except Exception as e:
        logger.error(f"Error listing MCP tools: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": str(e), "type": "internal_error"}},
        )

@app.get("/v1/mcp/prompts")
async def list_mcp_prompts(
    prompts: PromptsDependency,
    _: bool = Depends(verify_token),
) -> dict[str, list[dict[str, Any]]]:
    """
    List available MCP prompts with metadata.

    Returns prompt names, descriptions, and argument schemas for discovery.
    Clients can use this to understand what prompts are available.

    Returns:
        {"prompts": [{"name": "...", "description": "...", "arguments": [...]}, ...]}
    """
    try:
        return {
            "prompts": [prompt.to_dict() for prompt in prompts]
        }
    except Exception as e:
        logger.error(f"Error listing MCP prompts: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": str(e), "type": "internal_error"}},
        )

@app.get("/v1/mcp/prompts/{prompt_name}")
async def get_mcp_prompt(
    prompt_name: str,
    prompts: PromptsDependency,
    _: bool = Depends(verify_token),
) -> dict[str, Any]:
    """
    Get a specific MCP prompt by name.

    Returns detailed information about a single prompt including its
    name, description, and argument specifications.

    Args:
        prompt_name: Name of the prompt to retrieve (e.g., "server__prompt_name")

    Returns:
        {"name": "...", "description": "...", "arguments": [...]}

    Raises:
        404: Prompt not found
    """
    try:
        # Find prompt by name
        prompt = next((p for p in prompts if p.name == prompt_name), None)

        if prompt is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "message": f"Prompt '{prompt_name}' not found",
                        "type": "not_found_error",
                    }
                },
            )

        return prompt.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting MCP prompt '{prompt_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": str(e), "type": "internal_error"}},
        )
```

**Breaking Change - Remove Legacy `/tools` Endpoint:**
The existing `/tools` endpoint (line 1150 in `api/main.py`) returns only tool names: `{"tools": [list of names]}`.
This endpoint will be **REMOVED** and replaced with `/v1/mcp/tools` which:
1. Returns full metadata (names, descriptions, input schemas)
2. Uses consistent `/v1/` prefix with other API routes
3. Provides better discovery for clients

**Migration for clients:**
- Old: `GET /tools` → `{"tools": ["tool1", "tool2"]}`
- New: `GET /v1/mcp/tools` → `{"tools": [{"name": "tool1", "description": "...", "input_schema": {...}}, ...]}`

**Testing Strategy:**

Unit tests (`tests/unit_tests/test_api.py`):
```python
class TestMCPToolsEndpoint:
    """Test /v1/mcp/tools endpoint."""

    async def test_list_mcp_tools_with_tools_available(self):
        """Test endpoint returns tool metadata."""
        mock_tools = [
            Tool(name="search", description="Search", input_schema={...}),
            Tool(name="weather", description="Weather", input_schema={...}),
        ]

        result = await list_mcp_tools(tools=mock_tools, _=True)

        assert "tools" in result
        assert len(result["tools"]) == 2
        assert result["tools"][0]["name"] == "search"
        assert "description" in result["tools"][0]
        assert "input_schema" in result["tools"][0]

    async def test_list_mcp_tools_empty(self):
        result = await list_mcp_tools(tools=[], _=True)
        assert result == {"tools": []}

class TestMCPPromptsEndpoint:
    """Test /v1/mcp/prompts endpoints."""

    async def test_list_mcp_prompts_with_prompts_available(self):
        """Test list endpoint returns prompt metadata."""
        mock_prompts = [
            Prompt(name="ask", description="Ask question", arguments=[...]),
        ]

        result = await list_mcp_prompts(prompts=mock_prompts, _=True)

        assert "prompts" in result
        assert len(result["prompts"]) == 1
        assert result["prompts"][0]["name"] == "ask"
        assert "description" in result["prompts"][0]
        assert "arguments" in result["prompts"][0]

    async def test_get_mcp_prompt_found(self):
        """Test get endpoint returns specific prompt."""
        mock_prompts = [
            Prompt(name="server__ask", description="Ask question", arguments=[...]),
            Prompt(name="server__search", description="Search", arguments=[...]),
        ]

        result = await get_mcp_prompt(
            prompt_name="server__ask",
            prompts=mock_prompts,
            _=True
        )

        assert result["name"] == "server__ask"
        assert result["description"] == "Ask question"
        assert "arguments" in result

    async def test_get_mcp_prompt_not_found(self):
        """Test get endpoint raises 404 for unknown prompt."""
        mock_prompts = []

        with pytest.raises(HTTPException) as exc_info:
            await get_mcp_prompt(
                prompt_name="nonexistent",
                prompts=mock_prompts,
                _=True
            )

        assert exc_info.value.status_code == 404
        assert "not found" in str(exc_info.value.detail)
```

Integration tests (`tests/integration_tests/test_mcp_endpoints.py`):
```python
@pytest.mark.integration
class TestMCPEndpointsIntegration:
    """Integration tests for MCP tools/prompts endpoints."""

    def test_mcp_tools_endpoint_returns_real_tools(self):
        """Test /v1/mcp/tools with real MCP client."""
        # Requires MCP config with actual servers
        with TestClient(app) as client:
            response = client.get(
                "/v1/mcp/tools",
                headers={"Authorization": f"Bearer {settings.api_tokens[0]}"}
            )

            assert response.status_code == 200
            data = response.json()
            assert "tools" in data
            # Verify structure
            if data["tools"]:
                assert "name" in data["tools"][0]
                assert "description" in data["tools"][0]
                assert "input_schema" in data["tools"][0]

    def test_mcp_prompts_list_endpoint_returns_real_prompts(self):
        """Test /v1/mcp/prompts with real MCP client."""
        with TestClient(app) as client:
            response = client.get(
                "/v1/mcp/prompts",
                headers={"Authorization": f"Bearer {settings.api_tokens[0]}"}
            )

            assert response.status_code == 200
            data = response.json()
            assert "prompts" in data
            # Verify structure
            if data["prompts"]:
                assert "name" in data["prompts"][0]
                assert "description" in data["prompts"][0]
                assert "arguments" in data["prompts"][0]

    def test_mcp_prompts_get_endpoint_returns_specific_prompt(self):
        """Test /v1/mcp/prompts/{name} with real MCP client."""
        with TestClient(app) as client:
            # First get list to find a valid prompt name
            list_response = client.get(
                "/v1/mcp/prompts",
                headers={"Authorization": f"Bearer {settings.api_tokens[0]}"}
            )

            prompts = list_response.json()["prompts"]
            if prompts:
                # Test getting first prompt
                prompt_name = prompts[0]["name"]
                response = client.get(
                    f"/v1/mcp/prompts/{prompt_name}",
                    headers={"Authorization": f"Bearer {settings.api_tokens[0]}"}
                )

                assert response.status_code == 200
                data = response.json()
                assert data["name"] == prompt_name
                assert "description" in data
                assert "arguments" in data

    def test_mcp_prompts_get_endpoint_404_for_nonexistent(self):
        """Test /v1/mcp/prompts/{name} returns 404 for unknown prompt."""
        with TestClient(app) as client:
            response = client.get(
                "/v1/mcp/prompts/nonexistent_prompt_12345",
                headers={"Authorization": f"Bearer {settings.api_tokens[0]}"}
            )

            assert response.status_code == 404
            assert "not found" in response.json()["error"]["message"].lower()
```

**Dependencies:** Milestones 1-3

**Risk Factors:**
- API response size could be large with many tools/prompts
- Consider pagination if needed (future enhancement)
- Ensure sensitive information isn't leaked in schemas

---

## Testing Guidelines

### Unit Test Principles
1. **Mock External Dependencies**: Mock FastMCP Client, don't spawn real servers
2. **Test Edge Cases**: Empty lists, missing fields, None values, errors
3. **Follow Existing Patterns**: Use patterns from `test_mcp_to_tool.py` and `test_api.py`
4. **Test Isolation**: Each test should be independent
5. **Meaningful Coverage**: Focus on behavior, not just coverage percentage

### Integration Test Principles
1. **Real MCP Servers**: Use test fixtures with actual MCP servers (echo, math servers)
2. **End-to-End Flow**: Test full request�response cycle
3. **Error Scenarios**: Network failures, timeouts, invalid configs
4. **Performance**: Verify reasonable response times

### Test Organization
```
tests/
  unit_tests/
    test_prompts.py           # Milestone 1: Prompt abstraction
    test_mcp_to_prompt.py     # Milestone 2: MCP conversion
    test_api.py               # Milestone 4: Add endpoint tests
  integration_tests/
    test_mcp_endpoints.py     # Milestone 4: New integration tests
```

---

## Implementation Notes

### Breaking Changes
- **BREAKING**: Legacy `/tools` endpoint will be removed
  - Old endpoint: `GET /tools` → `{"tools": ["name1", "name2"]}`
  - New endpoint: `GET /v1/mcp/tools` → `{"tools": [{metadata}, {metadata}]}`
  - Clients must migrate to new endpoint for richer metadata
- All other changes are additive

### Security Considerations
- All endpoints require bearer token authentication
- Don't expose sensitive server connection details
- Sanitize error messages (no stack traces to clients)

### Performance Considerations
- MCP client connections are managed by service container (already initialized)
- Tools and prompts fetched per-request (no caching)
- Response size grows with number of tools/prompts (monitor in production)
- Consider pagination if response sizes become problematic (future enhancement)

### Error Handling Strategy
- All MCP operations wrapped in try/except
- Failures log errors and return empty lists (graceful degradation)
- API endpoints return proper HTTP error codes
- Error responses follow OpenAI-compatible format

### Future Enhancements (Not in this plan)
1. Pagination for tools/prompts endpoints (if response sizes become large)
2. Caching of tool/prompt lists with invalidation on MCP server changes
3. Filtering/search capabilities (e.g., `?search=weather`)
4. Prompt execution endpoint to invoke prompts via HTTP
5. WebSocket support for real-time updates when tools/prompts change
6. Tool execution endpoint (execute tools via HTTP - similar to prompts)

---

## Agent Instructions

### Before Starting
1. **Read FastMCP documentation**: https://gofastmcp.com/servers/prompts
2. **Review existing patterns**: Study `test_mcp_to_tool.py` and `test_api.py` thoroughly
3. **Understand the distinction**: MCP prompts (external) vs PromptManager (internal)

### During Implementation
- Complete each milestone fully (code + tests + docs) before proceeding
- Ask clarifying questions rather than make assumptions
- Follow existing code style and patterns exactly
- Run tests after each change: `make non_integration_tests` for speed
- Run integration tests before milestone completion: `make integration_tests`

### Testing Requirements
- Unit tests must pass without external dependencies
- Integration tests require `OPENAI_API_KEY` environment variable
- Use `pytest -k <test_name>` to run specific tests during development
- Aim for >90% code coverage on new code

### Code Quality
- Use type hints on all functions (including tests)
- No imports inside functions/classes
- Follow project's ruff linting rules
- Add docstrings to all public functions/classes
- Keep functions focused (single responsibility)

### Validation Checklist (per milestone)
- [ ] All new code has type hints
- [ ] All public functions have docstrings
- [ ] Unit tests pass: `make non_integration_tests`
- [ ] Integration tests pass: `make integration_tests`
- [ ] Linting passes: `make linting`
- [ ] No breaking changes to existing functionality
- [ ] Error handling covers all failure modes
- [ ] Logging is appropriate (info for success, error for failures)

---

## Design Decisions (Confirmed)

The following design decisions have been confirmed:

1. **Prompt Discovery**:
   - ✅ List all prompts via `GET /v1/mcp/prompts`
   - ✅ Get specific prompt via `GET /v1/mcp/prompts/{prompt_name}`
   - ❌ No prompt execution endpoint (future enhancement)

2. **Caching Strategy**:
   - ✅ Per-request fetching (no caching)
   - Simple, accurate, good for MVP
   - Caching can be added later if needed

3. **Response Format**:
   - ✅ Flat lists with prefixed names (e.g., `server__tool`, `server__prompt`)
   - Follows FastMCP's automatic naming convention
   - No additional grouping by server

4. **Legacy Endpoint**:
   - ✅ **Remove** `/tools` endpoint (breaking change acceptable)
   - Replace with `/v1/mcp/tools` (richer metadata)
   - Clean break for better API design

---

## Success Metrics

After completing all milestones:
- [ ] Can list all available MCP tools via `GET /v1/mcp/tools`
- [ ] Can list all available MCP prompts via `GET /v1/mcp/prompts`
- [ ] Can get specific prompt via `GET /v1/mcp/prompts/{prompt_name}`
- [ ] Legacy `/tools` endpoint removed
- [ ] All endpoints return full metadata (not just names)
- [ ] All endpoints require authentication
- [ ] Graceful degradation when MCP client unavailable
- [ ] Comprehensive test coverage (unit + integration)
- [ ] Documentation updated (OpenAPI auto-generated)
- [ ] Code follows project style guidelines
- [ ] All tests pass (`make tests`)
