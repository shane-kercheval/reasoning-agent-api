# Implementation Plan: Dynamic System Messages (No Storage)

## Overview

**Goal:** Remove system message storage from the database and make system messages purely dynamic (client-provided on each request).

**Rationale:** System messages are configuration/context for how to process a conversation, not part of the conversation content itself. This allows:
- Clients to change system instructions mid-conversation without affecting stored history
- Different clients to use different system messages for the same conversation
- The API to wrap/inject its own system instructions around user-provided system messages
- Cleaner separation: user/assistant messages = conversation content (stored), system message = processing instructions (dynamic)

**Key Principles:**
- Only user/assistant messages should be stored in the database
- System messages are extracted from the request, used for LLM call, then discarded
- No default system message (if client doesn't provide one, don't use one)
- Client is always responsible for providing system message if they want one
- API can inject/wrap user's system message with its own instructions

---

## Milestone 1: Database Schema Changes

### Goal
Remove `system_message` column from conversations table and update database models to reflect that system messages are not stored.

### Success Criteria
- [ ] New Alembic migration created and tested
- [ ] Migration successfully removes `system_message` column
- [ ] Migration handles existing data gracefully
- [ ] `ConversationDB` models updated (no system_message field)
- [ ] All database operations work without system_message
- [ ] Integration tests pass with new schema

### Key Changes

1. **Create Alembic Migration** (`alembic/versions/<timestamp>_remove_system_message_column.py`)
   - Remove `system_message` column from `conversations` table
   - No data migration needed (we're intentionally discarding this data)
   - Migration should be reversible for safety

2. **Update Database Models** (`api/database/conversation_db.py`)
   - Remove `system_message` field from `Conversation` dataclass
   - Update `create_conversation()` to not accept/store system_message
   - Update `get_conversation()` to not return system_message
   - Update `branch_conversation()` to not copy system_message

3. **Update SQL Queries**
   - Remove system_message from all SELECT queries
   - Remove system_message from INSERT queries
   - Update conversation branching logic (no system_message to copy)

4. Run migration against local database running in docker

### Testing Strategy

**Migration Testing:**
- **No automated migration tests exist** - migrations are run automatically by integration test fixtures
- Test manually by running against local docker database: `uv run alembic upgrade head`
- Verify migration runs without errors
- Check schema with `psql` to confirm `system_message` column is removed

**Unit Tests (`tests/unit_tests/test_conversation_utils.py`):**
- Update/remove tests that reference `system_message` in mock `Conversation` objects
- No database layer unit tests exist (integration tests cover ConversationDB)

**Integration Tests - Database Layer (`tests/integration_tests/test_conversation_db.py`):**
- **Pattern**: Uses testcontainers + automatic Alembic migration + transaction rollback per test
- **Fixture**: `conversation_db` provides ConversationDB with `_test_connection` for isolation
- **Update tests**:
  - Remove `system_message` parameter from `create_conversation()` calls (lines 26, 74, 86, 121-123, 143, 162, etc.)
  - Remove assertions on `conv.system_message` (lines 39, etc.)
  - Remove system_message from branch tests
- **Delete test**: `test_append_rejects_system_message` (line 160) - validation being removed

**Integration Tests - API Storage (`tests/integration_tests/test_conversation_storage_api.py`):**
- **Pattern**: Mocks LiteLLM, uses real database via `conversation_db` fixture + AsyncClient for API calls
- **Update tests**: Remove assertions on `conv.system_message` after database queries (lines ~158, 205, 296, etc.)

**Integration Tests - REST API (`tests/integration_tests/test_conversation_rest_api.py`):**
- **Pattern**: Tests REST endpoints with AsyncClient + mocked dependencies
- **Update tests**: Remove `system_message` from response structure checks (line 104, etc.)

### Dependencies
None - this is the foundation for all other changes.

### Risk Factors
- **Breaking change**: Any code that accesses `conversation.system_message` will break
- **Data loss**: Existing system messages will be permanently deleted

### Implementation Notes

**Example Migration Pattern:**
```python
def upgrade() -> None:
    # Drop system_message column from conversations table
    op.drop_column('conversations', 'system_message')

def downgrade() -> None:
    # Restore column with default value (for rollback)
    op.add_column(
        'conversations',
        sa.Column('system_message', sa.Text(), nullable=False,
                  server_default='You are a helpful assistant.')
    )
```

**Example Updated Dataclass:**
```python
@dataclass
class Conversation:
    id: UUID
    user_id: UUID | None
    title: str | None
    # system_message: str  # REMOVED
    created_at: str
    updated_at: str
    archived_at: str | None
    metadata: dict[str, Any]
    messages: list[Message]
    message_count: int | None = None
```

---

## Milestone 2: Core Conversation Logic Changes

### Goal
Update conversation utilities to treat system messages as dynamic request parameters rather than stored conversation properties.

### Success Criteria
- [ ] `SystemMessageInContinuationError` removed entirely
- [ ] `build_llm_messages()` extracts system message from request only (never from DB)
- [ ] `store_conversation_messages()` filters out system messages (doesn't store them)
- [ ] `extract_system_message()` continues to work for extracting from request
- [ ] All unit tests pass with new behavior

### Key Changes

1. **Remove System Message Validation** (`api/conversation_utils.py`)
   - Delete `SystemMessageInContinuationError` class (line 215-218)
   - Remove validation in `build_llm_messages()` that rejects system messages in continuation (lines 346-351)
   - Remove import/export of `SystemMessageInContinuationError`

2. **Update `build_llm_messages()`** (`api/conversation_utils.py:296-382`)
   - For STATELESS mode: use request messages as-is (no change)
   - For NEW mode: use request messages as-is (no change)
   - For CONTINUING mode:
     - Extract system message from REQUEST (not from DB)
     - Build: `[system_from_request] + [history] + [new_user_messages]`
     - If no system message in request, omit it: `[history] + [new_user_messages]`
     - Never load system_message from conversation object

3. **Update `store_conversation_messages()`** (`api/conversation_utils.py:384-436`)
   - Already filters system messages (line 417)
   - Verify this continues to work correctly
   - Add explicit test that system messages are never stored

4. **Keep `extract_system_message()`** (`api/openai_protocol.py:760-784`)
   - No changes needed - still used to extract from request
   - Used by `build_llm_messages()` to get system message from current request

### Testing Strategy

**Unit Tests to Update:**
- `test__build_llm_messages__continuing_rejects_system_message`: DELETE this test entirely
- `test__build_llm_messages__continuing_loads_history`: Update to show system message comes from REQUEST
- `test__build_llm_messages__stateless_passes_through`: No change (already correct)
- `test__build_llm_messages__new_passes_through`: No change (already correct)

**New Unit Tests to Add:**
- `test__build_llm_messages__continuing_with_system_message_in_request`: Verify system message from request is used
- `test__build_llm_messages__continuing_without_system_message`: Verify works fine without system message
- `test__build_llm_messages__continuing_system_message_placement`: Verify system message appears first in final list
- `test__store_conversation_messages__filters_system_messages`: Verify system messages are never stored

**Edge Cases:**
- Continuation request with system message (should work now)
- Continuation request without system message (should work)
- Multiple system messages in request (extract first one, as before)
- Empty messages array in continuation (regeneration case)

### Dependencies
- Milestone 1 must be complete (database schema updated)

### Risk Factors
- **Behavior change**: Continuing conversations can now include system messages
- **Test failures**: Many tests expect `SystemMessageInContinuationError` - need to update/delete
- **Existing client code**: Any clients that relied on stored system message will break

### Implementation Notes

**Updated `build_llm_messages()` Logic:**
```python
async def build_llm_messages(
    request_messages: list[dict],
    conversation_ctx: ConversationContext,
    conversation_db: ConversationDB | None,
) -> list[dict]:
    """Build complete message list for LLM based on conversation context."""

    # Extract system message from REQUEST (not from DB)
    system_message = extract_system_message(request_messages)

    # Stateless or new conversation: use request messages as-is
    if conversation_ctx.mode in (ConversationMode.STATELESS, ConversationMode.NEW):
        return request_messages

    # Continuing conversation
    if conversation_ctx.mode == ConversationMode.CONTINUING:
        # NO VALIDATION - system messages are allowed now

        # Load conversation history
        conversation = await conversation_db.get_conversation(conversation_ctx.conversation_id)

        # Build complete message list
        messages_for_llm = []

        # Add system message from REQUEST if provided
        if system_message is not None:
            messages_for_llm.append({"role": "system", "content": system_message})

        # Add historical messages (user/assistant only, no system)
        for msg in conversation.messages:
            messages_for_llm.append({"role": msg.role, "content": msg.content})

        # Add new user/assistant messages (filter out system since we already added it)
        user_messages = [m for m in request_messages if m.get("role") != "system"]
        messages_for_llm.extend(user_messages)

        return messages_for_llm
```

**Pattern for API Injecting System Instructions:**
(From our discussion - show how ReasoningAgent might do this)
```python
# Example: ReasoningAgent wrapping user's system message
synthesis_prompt = await self.prompt_manager.get_prompt("final_answer")
messages = deepcopy(request.messages)

if messages[0].get("role") == "system":
    # Wrap user's system message with API's instructions
    messages[0]["content"] = synthesis_prompt + "\n\n" + messages[0]["content"]
else:
    # Prepend API's system instructions
    messages.insert(0, {
        "role": "system",
        "content": synthesis_prompt,
    })
```

---

## Milestone 3: API Endpoint Changes

### Goal
Update FastAPI endpoints and response models to remove system_message field from all API responses.

### Success Criteria
- [ ] Exception handler for `SystemMessageInContinuationError` removed
- [ ] `create_conversation()` doesn't extract/store system_message
- [ ] `ConversationSummary` model doesn't include system_message
- [ ] `ConversationDetail` model doesn't include system_message
- [ ] `/v1/conversations` endpoint doesn't return system_message
- [ ] `/v1/conversations/{id}` endpoint doesn't return system_message
- [ ] All API integration tests pass

### Key Changes

1. **Remove Exception Handler** (`api/main.py:119-134`)
   - Delete `system_message_in_continuation_handler` exception handler
   - Remove `@app.exception_handler(SystemMessageInContinuationError)` decorator
   - Remove import of `SystemMessageInContinuationError`

2. **Update Chat Completions Endpoint** (`api/main.py:341-567`)
   - Remove `extract_system_message()` call (line 471)
   - Remove system_message parameter from `create_conversation()` call (line 473)
   - Remove title generation (line 472) - OR keep it using first user message
   - Update to pass title only: `conversation_id = await conversation_db.create_conversation(title=title)`

3. **Update Response Models** (`api/conversation_models.py`)
   - Remove `system_message` field from `ConversationSummary`
   - Remove `system_message` field from `ConversationDetail`
   - These models should only include: id, title, created_at, updated_at, archived_at, message_count

4. **Update Endpoint Response Builders** (`api/main.py`)
   - `list_conversations()` (lines 633-643): Don't include system_message in ConversationSummary
   - `get_conversation()` (lines 711-718): Don't include system_message in ConversationDetail
   - `update_conversation()` (lines 909-917): Don't include system_message in response
   - `branch_conversation()` (lines 1033-1040): Don't include system_message in response

### Testing Strategy

**Integration Tests to Update:**
- `test_conversation_storage_api.py`: Update all tests that check for system_message in responses
- `test_conversation_rest_api.py`: Remove assertions about system_message
- Verify conversation creation works without system_message
- Verify conversation listing doesn't include system_message
- Verify conversation detail doesn't include system_message

**New Integration Tests:**
- Test creating conversation without system message in request
- Test continuing conversation WITH system message (should work now)
- Test continuing conversation WITHOUT system message (should work)
- Test that system message in continuation request is used for LLM call but not stored

**API Contract Tests:**
- Verify response schemas match updated models
- Ensure no 400 errors for system messages in continuation
- Test regeneration with/without system message

### Dependencies
- Milestone 1: Database schema changes
- Milestone 2: Core logic changes

### Risk Factors
- **Breaking API change**: Clients expecting system_message in responses will break
- **OpenAPI schema**: Any generated clients will need regeneration
- **Frontend impact**: UI showing system message will need updates

### Implementation Notes

**Updated ConversationSummary:**
```python
class ConversationSummary(BaseModel):
    """Summary of a conversation for list views."""
    id: UUID
    title: str | None
    # system_message: str  # REMOVED
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None
    message_count: int
```

**Updated create_conversation call:**
```python
# OLD:
system_msg = extract_system_message(request.messages)
title = generate_title_from_messages(request.messages)
conversation_id = await conversation_db.create_conversation(
    system_message=system_msg,
    title=title,
)

# NEW:
title = generate_title_from_messages(request.messages)
conversation_id = await conversation_db.create_conversation(
    title=title,
)
```

**Title generation note:**
- Keep using `generate_title_from_messages()` - it extracts from first user message
- This is independent of system message and still works fine

---

## Milestone 4: Test Suite Updates

### Goal
Update all tests to reflect new behavior where system messages are dynamic and not stored.

### Success Criteria
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Tests verify system messages are NOT stored
- [ ] Tests verify system messages in continuation requests work correctly
- [ ] Test coverage maintained or improved
- [ ] Edge cases thoroughly tested

### Key Changes

1. **Delete Obsolete Tests**
   - Any test expecting `SystemMessageInContinuationError`
   - Any test checking for stored system_message in database
   - Any test asserting system_message in API responses

2. **Update Existing Tests**
   - Tests creating conversations: remove system_message assertions
   - Tests loading conversations: remove system_message from expected results
   - Tests calling API endpoints: remove system_message from response checks
   - Mock database tests: remove system_message from mock data

3. **Add New Tests**
   - System message in continuation request (positive test)
   - System message changes mid-conversation
   - No system message in request (verify it works)
   - API wrapping user's system message with its own instructions
   - Regeneration with/without system message

### Testing Strategy

**Tests to Delete:**
- `test_conversation_utils.py:test__build_llm_messages__continuing_rejects_system_message` (line ~303)
- `test_conversation_db.py:test_append_rejects_system_message` (line 160)

**Tests to Update:**

**Unit Tests (`tests/unit_tests/test_conversation_utils.py`):**
- `test__build_llm_messages__continuing_loads_history`: Update to show system message comes from request, not DB
- Remove `system_message` from all mock `Conversation` objects used in tests

**Integration Tests (`tests/integration_tests/test_conversation_db.py`):**
- **Pattern**: Direct ConversationDB method calls with testcontainers
- Update ~10+ tests that call `create_conversation(system_message=...)` - remove parameter
- Remove assertions on `conv.system_message` throughout

**Integration Tests (`tests/integration_tests/test_conversation_storage_api.py`):**
- **Pattern**: API calls via httpx.AsyncClient + mocked LiteLLM + real database
- Remove `system_message` assertions after `get_conversation()` calls
- Update tests that check conversation was created with system message

**Integration Tests (`tests/integration_tests/test_conversation_rest_api.py`):**
- **Pattern**: REST API endpoint tests with AsyncClient
- Line 104: Remove `assert "system_message" in conv` check from response validation
- Update all response structure assertions

**New Tests to Add:**

**In `test_conversation_storage_api.py`:**
```python
@pytest.mark.asyncio
async def test_continuing_conversation__with_system_message__allowed(
    client: AsyncClient,
    conversation_db: ConversationDB,
):
    """System messages can be provided when continuing conversations."""
    # Create conversation with initial request
    # Continue with DIFFERENT system message in second request
    # Verify: no error, LLM receives new system message, system message NOT stored
    # Pattern: Mock litellm.acompletion, check messages passed to it
```

```python
@pytest.mark.asyncio
async def test_continuing_conversation__system_message_not_stored(
    client: AsyncClient,
    conversation_db: ConversationDB,
):
    """Verify system messages are never stored in database."""
    # Send request with system message
    # Query database directly
    # Verify only user/assistant messages in db, no system messages
```

**In `test_conversation_utils.py`:**
```python
@pytest.mark.asyncio
async def test__build_llm_messages__continuing_with_system_from_request():
    """Continuing conversation uses system message from request."""
    # Mock conversation with history (no system_message field)
    # Build messages with system message in request
    # Verify: [system_from_request, history..., new_messages]
```

### Dependencies
- Milestones 1, 2, and 3 must be complete

### Risk Factors
- **Test coverage gaps**: Need to ensure new behavior is thoroughly tested
- **Regression risk**: Removing old tests might miss edge cases
- **Integration test complexity**: Mock behavior needs to match new reality

### Implementation Notes

**Test Organization:**
- Group tests by behavior (stateless, new, continuing)
- Use descriptive test names that explain the scenario
- Include docstrings explaining WHY the test exists
- Test both success and failure paths

**Key Test Scenarios:**

1. **Stateless Mode:**
   - With system message � pass through as-is 
   - Without system message � pass through as-is 

2. **New Conversation:**
   - With system message � use it, don't store it 
   - Without system message � works fine 

3. **Continuing Conversation:**
   - With system message � use from request 
   - Without system message � no system message in LLM call 
   - System message changes � each request uses its own 
   - Empty messages array (regeneration) � works with/without system 

4. **Storage Verification:**
   - User messages � stored 
   - Assistant messages � stored 
   - System messages � NEVER stored 

---

## Milestone 5: Documentation Updates

### Goal
Update all documentation to reflect new system message behavior.

### Success Criteria
- [ ] CLAUDE.md updated with new system message behavior
- [ ] API documentation reflects changes
- [ ] Migration guide for existing clients
- [ ] Examples show correct usage patterns

### Key Changes

1. **Update CLAUDE.md**
   - Remove references to stored system messages
   - Document that system messages are dynamic
   - Explain API can wrap/inject system instructions
   - Show example of changing system message mid-conversation

2. **Update API Documentation**
   - Document that system messages are not stored
   - Show examples of providing system message on each request
   - Explain regeneration behavior with/without system message

3. **Create Migration Guide**
   - Explain breaking changes for existing clients
   - Show before/after examples
   - Document that stored system messages will be deleted

### Testing Strategy
- No automated tests for documentation
- Manual review of clarity and accuracy
- Verify code examples actually work

### Dependencies
- All previous milestones complete

### Risk Factors
- **Outdated documentation**: Easy to miss references to old behavior
- **Client confusion**: Breaking change needs clear communication

### Implementation Notes

**Example Documentation Section:**

```markdown
## System Messages

System messages are **not stored** in the database. They are dynamic configuration parameters that control how the API processes your conversation.

**Key Points:**
- Provide system message in each request if you want one
- System message can change on every request
- The API may wrap your system message with its own instructions
- Only user/assistant messages are stored

**Example: Changing system message mid-conversation**

```python
# First request
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"}
    ],
    headers={"X-Conversation-ID": ""}
)
conversation_id = response.headers["X-Conversation-ID"]

# Later request with different system message
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are now a technical expert."},
        {"role": "user", "content": "Explain quantum computing"}
    ],
    headers={"X-Conversation-ID": conversation_id}
)
```
```

---

## Overall Implementation Order

1. **Milestone 1** (Database) - Foundation for all other changes
2. **Milestone 2** (Core Logic) - Depends on M1
3. **Milestone 3** (API) - Depends on M1 & M2
4. **Milestone 4** (Tests) - Depends on M1, M2 & M3
5. **Milestone 5** (Docs) - Depends on all previous milestones

## Validation Checklist

After all milestones complete, verify:
- [ ] `make tests` passes completely
- [ ] Database migration runs successfully on clean database
- [ ] Existing conversations load correctly (without system_message)
- [ ] New conversations work with/without system message
- [ ] Continuing conversations work with/without system message
- [ ] System messages are never stored in database
- [ ] API responses don't include system_message field
- [ ] Documentation accurately reflects new behavior
- [ ] No references to `SystemMessageInContinuationError` remain

## Breaking Changes Summary

**For Clients:**
- `system_message` field removed from all API responses
- Stored system messages will be deleted during migration
- Clients must provide system message on each request if they want one
- System messages in continuation requests are now allowed (no longer error)

**For Database:**
- `conversations.system_message` column removed
- Existing system message data will be lost
- Migration is one-way (downgrade restores column but not data)

## Questions for Human Review

Before starting implementation:
1. Should we keep `generate_title_from_messages()` using first user message? (Currently proposed: yes)
2. For regeneration (empty messages array), should we require system message in request or allow omitting it? (Currently proposed: client's choice)
3. Should the migration be reversible, or can we make it one-way? (Currently proposed: reversible with default value on downgrade)
4. Any specific error messages or warnings we should show to users during migration?
