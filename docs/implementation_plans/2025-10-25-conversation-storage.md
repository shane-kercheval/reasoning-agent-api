# Implementation Plan: Backend Conversation Storage

## Overview

Add postgres-backed conversation storage to the reasoning API, enabling:
- Persistent conversation history across sessions
- Cleaner client implementation (send only new messages, not full history)
- Foundation for future features (search, export, sharing, multi-device sync)

**Status**: ACTIVE - Required for Electron desktop client migration.

---

## Key Implementation Decisions

Based on comprehensive review and analysis, the following critical design decisions have been made:

### 1. Streaming Storage Strategy
**Problem**: Streaming architecture breaks atomicity - client receives response before DB write occurs.

**Solution**: Best-effort storage with error indication (Option C)
- Stream response normally for best UX
- After streaming completes, attempt to store in database
- If storage fails: log error + include `storage_failed: true` in metadata
- Desktop client shows toast notification: "⚠️ Response not saved to history"
- **Trade-off**: Small risk of client/DB inconsistency vs. maintaining streaming UX and simple implementation

### 2. Sequence Number Generation
**Problem**: Concurrent requests to same conversation could create duplicate sequence numbers.

**Solution**: Database-level atomic assignment
- Use `FOR UPDATE` row lock in transaction for atomic sequence number generation
- Add unique constraint: `UNIQUE (conversation_id, sequence_number)`
- Prevents race conditions without application-level complexity


### 3. Reasoning Events Storage
**Problem**: How to store reasoning steps in conversation history.

**Solution**: JSONB on final assistant message (not separate messages)
- Reasoning events are metadata about response generation, not conversation messages
- Store in `messages.reasoning_events` JSONB column
- Keeps conversation history clean (user → assistant → user pattern)
- Desktop client can render inline or in expandable section

### 4. Routing Mode Behavior
**Clarification**: `routing_mode` field purpose and constraints.

**Solution**: Analytics/debugging only, doesn't constrain behavior
- Stored from initial request's `X-Routing-Mode` header
- Used for observability and debugging
- Each request can override with new header or use default routing
- User can start with passthrough, switch to reasoning mid-conversation

---

## Smart Hybrid Approach

**Stateful Mode** (conversation storage):
- Request contains **only user messages** → Backend stores conversation in postgres
- `conversation_id` provided → Load history from DB, append new message
- `conversation_id` omitted → Create new conversation, return `conversation_id`
- Client sends only new user messages on subsequent requests

**Stateless Mode** (no storage):
- Request contains **system message** → Backend does NOT store conversation
- Client sends full message history with each request (current behavior)
- Useful for: custom system prompts, ephemeral chats, testing, programmatic use

**Rationale**: System message signals "I'm managing my own context". No system message signals "please manage my conversation history".

---

## Context

**Current Architecture** (Stateless):
- Client manages full conversation history
- Client sends entire message history with each request
- API has no conversation state (pure request/response)

**Proposed Architecture** (Stateful):
- Backend stores conversations in postgres
- API returns `conversation_id` on first message
- Client sends only new user messages + `conversation_id`
- API retrieves conversation history from DB

**Why This Is Complex**:
1. **Breaks stateless API design** - introduces session management
2. **Affects orchestration** - must work with future A2A protocol (M3-M4)
3. **Schema design** - needs to support reasoning events, tool calls, multi-agent conversations
4. **Lifecycle management** - when to archive/delete conversations?
5. **Authentication integration** - tie conversations to users (future multi-user support)

---

## Dependencies

**Prerequisites**:
- **New postgres instance required** - MUST create `postgres-reasoning` (cannot reuse phoenix/litellm)
- Alembic migrations setup for API schema changes (if not already present)

**Enables**:
- Desktop client with persistent conversations
- Multi-device conversation sync (future)
- Conversation search/export features (future)

**Future Considerations**:
- Orchestration architecture (M3-M4) may require schema updates for multi-agent flows
- User authentication would add user_id foreign key
- A2A protocol messages can be stored in same schema (messages.role = 'agent')

---

## High-Level Design (Tentative)

### Database Schema

**conversations table**:
```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,  -- Future: tie to user accounts
    title TEXT,  -- Auto-generated from first message
    routing_mode VARCHAR(50),  -- passthrough/reasoning/orchestration
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    archived_at TIMESTAMP,  -- Soft delete
    metadata JSONB DEFAULT '{}'  -- Extensible metadata
);

CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_created_at ON conversations(created_at DESC);
```

**messages table**:
```sql
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,  -- system/user/assistant/tool/agent
    content TEXT,
    reasoning_events JSONB,  -- Reasoning steps for assistant messages
    tool_calls JSONB,  -- Tool calls if applicable
    metadata JSONB DEFAULT '{}',  -- Extensible (model used, tokens, etc.)
    created_at TIMESTAMP DEFAULT NOW(),
    sequence_number INTEGER NOT NULL  -- Ordering within conversation
);

CREATE INDEX idx_messages_conversation_id ON messages(conversation_id, sequence_number);
```

### API Changes

**Modified Endpoint** (Main Change):
```python
POST /v1/chat/completions

Request Headers:
  X-Conversation-ID: <uuid>  # Optional - for continuing existing conversation

Request Body:
{
    "messages": [
        {"role": "user", "content": "Hello"}  # User messages only for stateful mode
        # OR
        {"role": "system", "content": "You are..."}, # Include system = stateless mode
        {"role": "user", "content": "Hello"}
    ],
    ...
}

Behavior Logic:
1. Check if messages contain system message:
   - YES → STATELESS MODE: Don't store, use messages as-is
   - NO → STATEFUL MODE: Check X-Conversation-ID header

2. Stateful mode logic:
   - If X-Conversation-ID header provided:
     - Load conversation from DB
     - Append new user message
     - Use full history for LLM call
   - If X-Conversation-ID header omitted:
     - Create new conversation in DB
     - Use user message for LLM call
     - Return conversation_id in response header

3. Response format (streaming):
   Response Headers:
     X-Conversation-ID: <uuid>  # Returned for new or existing conversations

   Response Body: Standard OpenAI SSE stream (no modifications needed)

   Desktop client:
   - Reads X-Conversation-ID from response headers
   - Stores conversation_id for next request
   - Includes in X-Conversation-ID header on subsequent requests
```

**New Endpoints** (Conversation Management):
```python
# List conversations (with pagination)
GET /v1/conversations?limit=50&offset=0
Response: {"conversations": [...], "total": 123}

# Get conversation details
GET /v1/conversations/{conversation_id}
Response: {"id": "uuid", "messages": [...], "title": "...", ...}

# Delete conversation
DELETE /v1/conversations/{conversation_id}
Response: {"deleted": true}

# Update conversation title
PATCH /v1/conversations/{conversation_id}
Request: {"title": "New title"}
Response: {"id": "uuid", "title": "New title", ...}
```

### Implementation Details

**Stateful vs Stateless Detection** (`api/main.py`):
```python
def is_stateless_request(request: OpenAIChatRequest) -> bool:
    """Check if request has system message (stateless mode indicator)."""
    return any(msg.get("role") == "system" for msg in request.messages)

async def chat_completions(
    request: OpenAIChatRequest,
    http_request: Request,
    ...
):
    # Determine mode
    if is_stateless_request(request):
        # Current behavior - use messages as-is, don't store
        messages = request.messages
        conversation_id = None
    else:
        # Stateful mode - manage conversation
        # Get conversation_id from request header
        conversation_id = http_request.headers.get("X-Conversation-ID")

        if conversation_id:
            # Load existing conversation
            conversation = await db.get_conversation(UUID(conversation_id))
            messages = conversation.messages + request.messages  # Append new
            await db.append_messages(UUID(conversation_id), request.messages)
        else:
            # Create new conversation
            conversation_id = await db.create_conversation(request.messages)
            messages = request.messages

    # Make LLM call with full message history
    # ... existing streaming logic ...

    # Return conversation_id in response header (if stateful)
    return StreamingResponse(
        span_aware_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Conversation-ID": str(conversation_id),  # Always include for stateful mode
        },
    )
```

**Conversation Title Generation**:
- Auto-generate from first user message (truncate to 50 chars)
- Update when user explicitly sets title via PATCH endpoint
- Example: "How do I use FastAPI?" → "How do I use FastAPI?"

**Reasoning Events Storage**:
- Store in `messages.reasoning_events` JSONB column
- Easy to query and serialize
- No need for separate table (JSONB is efficient)

**Performance Considerations**:
- Index on `conversation_id` for fast lookups
- Sequence numbers for message ordering
- No message limits - conversations can grow indefinitely
- Future: Consider pagination for UI rendering of very long conversations

**Backward Compatibility**:
- Stateless mode works exactly as before (system message present)
- No breaking changes for existing clients
- Desktop client opts into stateful mode (no system message)

---

## Implementation Milestones

### Milestone 0: Impact Analysis & Research

**Goal**: Analyze impact of stateful/stateless approach on existing code and tests.

**Success Criteria**:
- Document all code locations affected by system message detection
- Identify all existing tests and how they need to change
- Document API contract changes
- Create checklist of documentation that needs updating

**Tasks**:
- **Code Impact Analysis**:
  - Review all tests in `tests/` directory
  - Identify tests that send requests to `/v1/chat/completions`
  - Determine which tests should remain stateless (need system message added)
  - Document any code that builds OpenAI requests

- **Test Impact Assessment**:
  - List all integration tests that test chat completions
  - Determine if each test should be stateful or stateless
  - Document changes needed for stateless tests (add system message)
  - Identify new tests needed for conversation storage functionality

- **API Contract Changes**:
  - Document new `conversation_id` field in `OpenAIChatRequest`
  - Document response metadata changes (conversation_id in first chunk)
  - Document new conversation management endpoints

- **Documentation Checklist**:
  - README.md updates needed
  - .env.dev.example updates needed
  - .env.prod.example updates needed
  - Makefile updates needed
  - CLAUDE.md updates needed
  - API documentation updates needed

**Deliverable**: Create `docs/conversation_storage_impact_analysis.md` with:
- Complete list of affected files
- Test migration strategy
- Documentation update checklist

**Testing**: N/A (research only)

**Dependencies**: None

**Risks**: Missing edge cases in impact analysis

---

### Milestone 1: Database Setup & Migrations
**Goal**: Set up postgres schema for conversations

**Tasks**:
- Add Alembic to project (if not already present)
- Create migration for `conversations` and `messages` tables
- Add indexes for performance
- Add unique constraint: `UNIQUE (conversation_id, sequence_number)` to prevent race conditions
- Test migration locally

**Success Criteria**:
- `alembic upgrade head` creates tables
- Schema matches design above with unique constraint on sequence numbers
- docker-compose.yml updated with postgres-reasoning service (port 5434)
- .env.dev.example and .env.prod.example updated with REASONING_POSTGRES_PASSWORD
- Makefile updated if needed for database migrations

**Documentation Updates**:
- Update docker-compose.yml (add postgres-reasoning service)
- Update .env.dev.example (add REASONING_POSTGRES_PASSWORD=dev_password_here)
- Update .env.prod.example (add REASONING_POSTGRES_PASSWORD)
- Update README.md (mention new postgres instance)
- Update CLAUDE.md (document conversation storage architecture)

---

### Milestone 2: Database Layer
**Goal**: Create conversation CRUD operations

**Tasks**:
- Create `api/database/` module
- Implement `ConversationDB` class with async methods:
  - `create_conversation(messages, routing_mode) -> conversation_id`
  - `get_conversation(conversation_id) -> Conversation`
  - `append_messages(conversation_id, messages)` with atomic sequence numbering
  - `list_conversations(limit, offset) -> list[Conversation]`
  - `delete_conversation(conversation_id)`
- Use **asyncpg** for postgres access (raw SQL, no ORM)
- Handle connection pooling with asyncpg.create_pool()
- **Sequence number generation**: Use `FOR UPDATE` lock pattern for atomic assignment
- **Routing mode storage**: Store `routing_mode` from initial request for analytics/debugging only (doesn't constrain future requests)

**Success Criteria**:
- Unit tests pass for all CRUD operations
- Proper error handling (conversation not found, etc.)
- Atomic sequence number generation prevents race conditions
- Tests added incrementally as each method is implemented

**Testing Strategy**:
- Write unit tests AS YOU IMPLEMENT each CRUD method
- Test happy path and error cases for each method
- Mock postgres connection for unit tests
- Test connection pooling behavior
- **Test concurrent message appends** - verify no duplicate sequence numbers

**Implementation Details**:
```python
# Atomic sequence number generation
async def append_messages(conversation_id, messages):
    async with conn.transaction():
        # Get next sequence atomically with row lock
        result = await conn.fetchrow(
            """
            SELECT COALESCE(MAX(sequence_number), 0) + 1 as next_seq
            FROM messages WHERE conversation_id = $1
            FOR UPDATE  -- Prevents concurrent conflicts
            """,
            conversation_id
        )
        # Insert with sequential numbering...
```

---

### Milestone 3: Modify Chat Completions Endpoint
**Goal**: Add stateful/stateless mode detection and conversation management

**Tasks**:
- Implement `is_stateless_request()` helper (checks for system message)
- Modify `chat_completions()` endpoint logic:
  - Read `X-Conversation-ID` from request headers
  - Detect mode (stateless vs stateful)
  - Load/create conversation as needed
  - Build full message history for LLM call
  - Store assistant response in DB after streaming completes (stateful mode only)
  - Return `X-Conversation-ID` in response headers
- **Streaming storage strategy**: Best-effort storage with error indication
  - Buffer assistant message content during streaming
  - After streaming completes, try to store in DB
  - If storage fails, log error and include `storage_failed: true` in last chunk metadata
  - Client can display toast notification: "⚠️ Response not saved to history"

**Header-Based Design**:
- **Request header**: `X-Conversation-ID: <uuid>` (optional, for continuing conversation)
- **Response header**: `X-Conversation-ID: <uuid>` (always included for stateful mode)
- **Benefits**: Clean separation, no OpenAI spec pollution, works naturally with streaming

**Success Criteria**:
- Stateless mode works exactly as before (system message present)
- Stateful mode creates/loads conversations correctly
- Streaming storage works with error indication
- Integration tests cover both modes
- All existing tests updated to include system message (remain stateless)
- New tests added for stateful mode

**Testing Strategy**:
- **Update existing tests FIRST**: Add system message to all existing integration tests
  - Review tests identified in Milestone 0 impact analysis
  - Add `{"role": "system", "content": "Test system prompt"}` to keep tests stateless
  - Verify all existing tests still pass

- **Add new stateful tests**: Test conversation creation/loading
  - Test new conversation creation (no X-Conversation-ID header)
  - Test continuing conversation (with X-Conversation-ID header)
  - Test X-Conversation-ID in response headers
  - Test error cases (invalid conversation_id UUID, etc.)
  - Test storage failure scenario (mock DB error, verify metadata flag)

**Streaming Storage Implementation**:
```python
async def span_aware_stream():
    """Stream with best-effort message storage."""
    assistant_message_buffer = []
    storage_failed = False

    try:
        # Stream to client normally
        async for chunk in execute_passthrough_stream(...):
            # Buffer content for storage
            if chunk.choices[0].delta.content:
                assistant_message_buffer.append(chunk.choices[0].delta.content)
            yield chunk

        # After streaming completes, try to store
        if conversation_id:
            full_content = "".join(assistant_message_buffer)
            try:
                await db.append_messages(conversation_id, [{
                    "role": "assistant",
                    "content": full_content,
                }])
            except Exception as e:
                logger.error(f"Failed to store message for {conversation_id}: {e}")
                storage_failed = True

        # Indicate storage failure in last chunk if needed
        if storage_failed:
            yield create_sse({
                "choices": [...],
                "metadata": {"storage_failed": True}
            })

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        raise
```

**Reasoning Events Storage Format**:
- Reasoning events stored in `messages.reasoning_events` JSONB column
- NOT separate messages - part of final assistant message metadata
- Example structure:
```python
await db.append_messages(conversation_id, [{
    "role": "assistant",
    "content": "Final answer here",
    "reasoning_events": [
        {"step": 1, "type": "PLANNING", "content": "Analyzing query..."},
        {"step": 1, "type": "TOOL_EXECUTION_START", "tools": ["weather"]},
        {"step": 1, "type": "TOOL_RESULT", "results": [...]},
        {"step": 1, "type": "ITERATION_COMPLETE"},
        {"type": "REASONING_COMPLETE"},
    ]
}])
```

**Desktop Client Implementation**:
```typescript
// Sending request with conversation context
const conversationId = getCurrentConversationId(); // From state

const response = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-Conversation-ID': conversationId || '',  // Include if continuing conversation
    },
    body: JSON.stringify({
        messages: [{ role: "user", content: userInput }],
        stream: true,
    }),
});

// Extract conversation_id from response headers
const newConversationId = response.headers.get('X-Conversation-ID');
if (newConversationId) {
    setCurrentConversationId(newConversationId);  // Store for next request
}

// Consume SSE stream normally
const reader = response.body.getReader();
// ... standard streaming consumption ...
```

**Documentation Updates**:
- Update CLAUDE.md with stateful/stateless behavior
- Document header-based conversation_id approach
- Document streaming storage strategy and error indication
- Document reasoning events storage format
- Update API documentation (if exists)

---

### Milestone 4: Conversation Management Endpoints
**Goal**: Add REST endpoints for conversation management

**Tasks**:
- Implement `GET /v1/conversations` (list with pagination)
- Implement `GET /v1/conversations/{id}` (get details)
- Implement `DELETE /v1/conversations/{id}` (soft delete)
- Implement `PATCH /v1/conversations/{id}` (update title)
- Add authentication checks (require valid token)

**Success Criteria**:
- All endpoints work and return correct data
- Pagination works correctly
- Integration tests pass for each endpoint as it's implemented

**Testing Strategy**:
- Write integration test for each endpoint AS IT'S IMPLEMENTED
- Test authentication (valid/invalid tokens)
- Test pagination edge cases (empty list, single page, multiple pages)
- Test error cases (conversation not found, invalid parameters)

**Documentation Updates**:
- Update README.md with conversation management endpoints
- Update API documentation (if exists)
- Add examples to CLAUDE.md

---

### Milestone 5: Desktop Client Integration
**Goal**: Update Electron client to use backend conversation storage

**Tasks**:
- Modify Zustand store to track `conversation_id`
- Update `useChatActions` hook:
  - Send only user messages (no system prompt by default)
  - Include `conversation_id` in requests
  - Extract `conversation_id` from first chunk metadata
- Add conversation list UI (sidebar or menu)
- Add "New Conversation" button (clears `conversation_id`)
- Add delete conversation functionality

**Success Criteria**:
- Desktop client persists conversations across restarts
- User can view conversation list
- User can start new conversations
- User can delete conversations

---

### Milestone 6: Final Integration & Documentation
**Goal**: End-to-end testing and complete documentation

**Tasks**:
- **Final Integration Tests**:
  - Full conversation lifecycle (create → multiple messages → delete)
  - Test conversation with reasoning events
  - Test conversation with tool calls
  - **Test concurrent requests to same conversation** - verify no duplicate sequences
  - **Test streaming failure scenarios** - verify storage error indication
  - Performance testing (load conversation with 100+ messages)

- **Concurrency Testing** (explicit scenarios):
  - Rapid-fire messages to same conversation (no duplicate sequence numbers)
  - Simultaneous conversation creation (no UUID collisions)
  - Conversation loading during active write (read consistency)
  - Message ordering under concurrent appends

- **OpenTelemetry Integration**:
  - Add conversation_id to span attributes
  - Verify tracing works for stateful and stateless modes

- **Final Documentation**:
  - Complete README.md updates
  - Complete CLAUDE.md updates
  - Update Makefile (if needed for new commands)
  - Verify .env.example files are complete
  - Add troubleshooting section (including storage failure scenarios)

**Success Criteria**:
- All integration tests pass
- Performance acceptable (<100ms to load conversation)
- Tracing includes conversation metadata
- Documentation is complete and accurate
- Fresh developer can follow setup instructions successfully
- Concurrency edge cases handled correctly

**Note**: Most tests should already be written in earlier milestones. This milestone is for integration testing and final polish.

---

## Postgres Database Selection

**Decision**: Create new postgres instance for reasoning API

**Options**:
1. **New postgres instance** (reasoning-postgres) - Recommended
2. Reuse postgres-phoenix - Not recommended (separate concerns)
3. Reuse postgres-litellm - Not recommended (separate concerns)

**Recommendation**: New instance (`postgres-reasoning`)

**Rationale**:
- Clean separation of concerns (API data vs observability data vs LiteLLM data)
- Independent scaling and backup strategies
- Easier to migrate/upgrade independently
- Postgres is lightweight - container overhead is minimal
- Consistent naming: `postgres-reasoning` (matches `postgres-phoenix`, `postgres-litellm`)

**Docker Compose Addition**:
```yaml
postgres-reasoning:
  image: postgres:16
  container_name: reasoning-postgres
  environment:
    - POSTGRES_DB=reasoning
    - POSTGRES_USER=reasoning_user
    - POSTGRES_PASSWORD=${REASONING_POSTGRES_PASSWORD}
  ports:
    - "5434:5432"  # Avoid conflicts with 5432 (phoenix) and 5433 (litellm)
  volumes:
    - reasoning_postgres_data:/var/lib/postgresql/data
  networks:
    - reasoning-network
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U reasoning_user -d reasoning"]
    interval: 10s
    timeout: 5s
    retries: 5
```

---

## Technology Choices

**Database Access**: asyncpg (raw SQL)
- **Rationale**:
  - Simple CRUD operations - no complex ORM needed
  - Better performance than SQLAlchemy
  - More explicit control (matches current codebase style)
  - Less abstraction = easier debugging
  - Direct SQL is clearer for review and maintenance

**Migration Tool**: Alembic (works perfectly with asyncpg)

---

## Future Enhancements (Post-Electron)

**After orchestration design (M3-M4)**:
- Add support for agent-to-agent messages in schema
- Add multi-agent conversation flow tracking
- Update schema if A2A protocol requires changes

**User Authentication** (if needed):
- Add `user_id` foreign key to conversations
- Add user table
- Filter conversations by authenticated user

**Advanced Features**:
- Conversation search (full-text search on messages)
- Conversation export (JSON, Markdown, PDF)
- Conversation sharing (generate public links)
- Conversation forking (branch from specific message)
- Multi-device sync (real-time via WebSocket)
