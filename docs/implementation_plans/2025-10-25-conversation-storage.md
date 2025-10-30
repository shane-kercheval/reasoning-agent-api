# Implementation Plan: Backend Conversation Storage

## Overview

Add postgres-backed conversation storage to the reasoning API, enabling:
- Persistent conversation history across sessions
- Cleaner client implementation (send only new messages, not full history)
- Foundation for future features (search, export, sharing, multi-device sync)

**Status**: ACTIVE - Required for Electron desktop client migration.

---

## ⚠️ Important: Environment Management

**ALWAYS use `uv` for all package management and script execution**:

```bash
# Adding dependencies
uv add <package>          # NOT: pip install
uv add asyncpg alembic    # Example

# Running Python scripts
uv run python script.py   # NOT: python script.py

# Running tests
uv run make tests         # NOT: make tests
uv run pytest tests/      # NOT: pytest tests/

# Running database migrations
uv run alembic upgrade head    # NOT: alembic upgrade head
```

**Why `uv`?**
- Ensures consistent environment across development and CI
- Manages virtual environment automatically
- Faster than pip
- Prevents "works on my machine" issues

---

## Key Implementation Decisions

Based on comprehensive review and analysis, the following critical design decisions have been made:

### 1. Statefulness Detection
**Problem**: How to determine if a request should use conversation storage.

**Solution**: Header-based explicit opt-in
- **Header present** (`X-Conversation-ID`) → Stateful mode (store conversation)
- **Header absent** → Stateless mode (no storage, current behavior)
- **Rationale**: Explicit, intuitive, backward compatible - existing tests/clients work unchanged

### 2. System Message Handling
**Problem**: System messages come at start of conversation, can't change mid-conversation.

**Solution**: Store in conversations table, fail-fast on changes
- System message stored in `conversations.system_message` column (set once on creation)
- **New conversation** (`X-Conversation-ID: ""`) - system message allowed in request (optional)
- **Continuation** (`X-Conversation-ID: <uuid>`) - system message in request causes 400 error
- **Rationale**: Fail-fast prevents bugs, system message is immutable per conversation

### 3. Message Storage Format
**Problem**: How to store conversation messages in database.

**Solution**: Individual rows (not JSON blob)
- One row per user/assistant message in `messages` table
- System message NOT stored in messages table (only in `conversations.system_message`)
- **Rationale**: Easier to query, extend, index; atomic sequence numbering; better for reasoning events

### 4. Streaming Storage Strategy
**Problem**: Streaming architecture breaks atomicity - client receives response before DB write occurs.

**Solution**: Best-effort storage with error indication
- Stream response normally for best UX
- After streaming completes, attempt to store in database
- If storage fails: log error + include `storage_failed: true` in metadata
- Desktop client shows toast notification: "⚠️ Response not saved to history"
- **Trade-off**: Small risk of client/DB inconsistency vs. maintaining streaming UX and simple implementation

### 5. Sequence Number Generation
**Problem**: Concurrent requests to same conversation could create duplicate sequence numbers.

**Solution**: Database-level atomic assignment
- Use `FOR UPDATE` row lock in transaction for atomic sequence number generation
- Add unique constraint: `UNIQUE (conversation_id, sequence_number)`
- Prevents race conditions without application-level complexity


### 6. Reasoning Events Storage
**Problem**: How to store reasoning steps in conversation history.

**Solution**: JSONB on final assistant message (not separate messages)
- Reasoning events are metadata about response generation, not conversation messages
- Store in `messages.reasoning_events` JSONB column
- Keeps conversation history clean (user → assistant → user pattern)
- Desktop client can render inline or in expandable section

### 7. Routing Mode Behavior
**Clarification**: `routing_mode` field purpose and constraints.

**Solution**: Analytics/debugging only, doesn't constrain behavior
- Stored from initial request's `X-Routing-Mode` header
- Used for observability and debugging
- Each request can override with new header or use default routing
- User can start with passthrough, switch to reasoning mid-conversation

---

## Smart Hybrid Approach

**Stateful Mode** (conversation storage):
- Request includes `X-Conversation-ID` header → Backend stores conversation in postgres
- `X-Conversation-ID: ""` (empty) → Create new conversation, return `conversation_id`
- `X-Conversation-ID: <uuid>` → Load history from DB, append new messages
- Client sends only new messages on subsequent requests
- System message set once on creation, immutable thereafter

**Stateless Mode** (no storage):
- Request does NOT include `X-Conversation-ID` header → No storage (current behavior)
- Client sends full message history with each request
- Useful for: testing, programmatic use, one-off queries

**Rationale**: Header presence is explicit opt-in to conversation storage. No header means stateless (exactly like current behavior).

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
    system_message TEXT NOT NULL DEFAULT 'You are a helpful assistant.',  -- Set once on creation
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
    role VARCHAR(50) NOT NULL,  -- 'user' or 'assistant' (NOT 'system' - stored in conversations table)
    content TEXT,
    reasoning_events JSONB,  -- Reasoning steps for assistant messages
    tool_calls JSONB,  -- Tool calls if applicable
    metadata JSONB DEFAULT '{}',  -- Extensible (model used, tokens, etc.)
    created_at TIMESTAMP DEFAULT NOW(),
    sequence_number INTEGER NOT NULL,  -- Ordering within conversation

    CONSTRAINT unique_conversation_sequence UNIQUE (conversation_id, sequence_number)
);

CREATE INDEX idx_messages_conversation_id ON messages(conversation_id, sequence_number);
```

**Notes**:
- System messages stored in `conversations.system_message`, NOT in messages table
- One row per user/assistant message
- Unique constraint on (conversation_id, sequence_number) prevents race conditions

### API Changes

**Modified Endpoint** (Main Change):
```python
POST /v1/chat/completions

Request Headers:
  X-Conversation-ID: <uuid>  # Optional - for continuing existing conversation

Request Body:
{
    "messages": [
        {"role": "system", "content": "You are..."},  # Optional on new conversation, error on continuation
        {"role": "user", "content": "Hello"}
    ],
    ...
}

Behavior Logic:
1. Check if X-Conversation-ID header is present:
   - NO HEADER → STATELESS MODE: Use messages as-is, don't store
   - HEADER PRESENT → STATEFUL MODE: Check header value

2. Stateful mode logic:
   - If X-Conversation-ID is "" or "null" (new conversation):
     - Extract system message from request (optional)
     - Create new conversation in DB with system message
     - Return conversation_id in response header

   - If X-Conversation-ID is <uuid> (continuation):
     - Validate no system message in request (fail with 400 if present)
     - Load conversation from DB
     - Prepend stored system message to message history
     - Append new messages
     - Use full history for LLM call

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
def extract_system_message(messages: list[dict]) -> str | None:
    """Extract first system message from messages list."""
    system_msgs = [m for m in messages if m.get("role") == "system"]
    return system_msgs[0].get("content") if system_msgs else None

async def chat_completions(
    request: OpenAIChatRequest,
    http_request: Request,
    ...
):
    conversation_header = http_request.headers.get("X-Conversation-ID")

    # Determine mode based on header presence
    if conversation_header is None:
        # STATELESS MODE - no header, no storage
        messages = request.messages
        conversation_id = None

    else:
        # STATEFUL MODE - header present
        if conversation_header == "" or conversation_header == "null":
            # NEW CONVERSATION
            system_msg = extract_system_message(request.messages)
            user_messages = [m for m in request.messages if m.get("role") != "system"]

            conversation_id = await db.create_conversation(
                messages=user_messages,
                system_message=system_msg or "You are a helpful assistant.",
                routing_mode=routing_decision.routing_mode.value,
            )

            # Build messages: [system] + [new messages]
            messages = (
                [{"role": "system", "content": system_msg}] + user_messages
                if system_msg else user_messages
            )

        else:
            # CONTINUE EXISTING CONVERSATION
            # Fail fast if system message in continuation
            if any(m.get("role") == "system" for m in request.messages):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                "System messages not allowed when continuing a conversation. "
                                f"The system message for conversation {conversation_header} "
                                "was set during creation and cannot be changed."
                            ),
                            "type": "invalid_request_error",
                            "code": "system_message_in_continuation",
                        }
                    }
                )

            conversation_id = UUID(conversation_header)
            conversation = await db.get_conversation(conversation_id)

            # Build messages: [stored system] + [history] + [new messages]
            messages = (
                [{"role": "system", "content": conversation.system_message}]
                + conversation.messages
                + request.messages
            )

    # Make LLM call with full message history
    # ... existing streaming logic ...

    # Return conversation_id in response header (if stateful)
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    if conversation_id:
        headers["X-Conversation-ID"] = str(conversation_id)

    return StreamingResponse(
        span_aware_stream(),
        media_type="text/event-stream",
        headers=headers,
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

### Milestone 0: Impact Analysis & Research ✅ COMPLETE

**Goal**: Analyze impact of header-based stateful/stateless approach on existing code and tests.

**Key Finding**: **Header-based approach means ZERO changes to existing tests!**
- Existing tests don't send `X-Conversation-ID` header → stateless automatically
- No test updates needed (backward compatible)
- Only need new tests FOR conversation storage functionality

**Success Criteria**:
- ✅ Confirm existing tests remain stateless (no header = stateless)
- ✅ Document API contract changes (headers, system message handling)
- ✅ Create checklist of documentation that needs updating
- ✅ Identify new tests needed for conversation storage

**Deliverable**: ~~`docs/conversation_storage_impact_analysis.md`~~ **NOT NEEDED** - no existing test changes required

**Key Decisions**:
- ✅ Statefulness via `X-Conversation-ID` header (not system message detection)
- ✅ System message stored in `conversations.system_message` column
- ✅ Fail-fast on system message in continuation (400 error)
- ✅ Individual message rows (not JSON blob)
- ✅ Existing tests work unchanged (no header = stateless)

**Testing**: N/A (research only)

**Dependencies**: None

**Outcome**: Ready to proceed to Milestone 1

**Next Commands**:
```bash
# Add dependencies
uv add asyncpg alembic

# Initialize Alembic
uv run alembic init alembic

# Create initial migration (after editing alembic.ini and env.py)
uv run alembic revision --autogenerate -m "Add conversation storage tables"

# Run migration
uv run alembic upgrade head
```

---

### Milestone 1: Database Setup & Migrations
**Goal**: Set up postgres schema for conversations

**Implementation Decisions**:
- **Database Connection**: Alembic configured for `localhost:5434` (local dev workflow)
  - Migrations run from host Mac: `uv run alembic upgrade head`
  - Connects to Docker postgres container via exposed port
  - Database URL configurable via environment variable for flexibility
- **Environment Variables**:
  - `REASONING_POSTGRES_PASSWORD` - Configurable via .env (security best practice)
  - `REASONING_POSTGRES_DB=reasoning` - Hardcoded in docker-compose.yml (deployment config)
  - `REASONING_POSTGRES_USER=reasoning_user` - Hardcoded in docker-compose.yml (deployment config)
- **Alembic Configuration**:
  - Async engine with asyncpg (matches async FastAPI architecture)
  - Standard `alembic/versions/` directory structure
  - Environment-based database URL with sensible defaults
- **Initial Migration**:
  - Single migration with both `conversations` and `messages` tables
  - Includes all indexes and unique constraints from schema design
  - Includes default values (e.g., `system_message` default)
- **Testing Approach**:
  - Manual verification: `uv run alembic upgrade head` + SQL inspection
  - Automated migration tests deferred (appropriate for learning project)

**Tasks**:
- Add dependencies: `uv add asyncpg alembic` (**IMPORTANT**: Use `uv add`, not pip)
- Initialize Alembic: `uv run alembic init alembic`
- Configure Alembic for async asyncpg with environment-based database URL
- Create migration for `conversations` and `messages` tables
- Add indexes for performance
- Add unique constraint: `UNIQUE (conversation_id, sequence_number)` to prevent race conditions
- Test migration locally: `uv run alembic upgrade head`

**Success Criteria**:
- `uv run alembic upgrade head` creates tables successfully
- Schema matches design above with unique constraint on sequence numbers
- docker-compose.yml updated with postgres-reasoning service (port 5434)
- .env.dev.example and .env.prod.example updated with REASONING_POSTGRES_PASSWORD
- Makefile updated if needed for database migrations (commands should use `uv run`)

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
  - `create_conversation(messages, system_message, routing_mode) -> conversation_id`
  - `get_conversation(conversation_id) -> Conversation` (includes system_message)
  - `append_messages(conversation_id, messages)` with atomic sequence numbering
  - `list_conversations(limit, offset) -> list[Conversation]`
  - `delete_conversation(conversation_id)`
- Use **asyncpg** for postgres access (raw SQL, no ORM)
- Handle connection pooling with asyncpg.create_pool()
- **Sequence number generation**: Use `FOR UPDATE` lock pattern for atomic assignment (lock conversation row, not message rows)
- **Routing mode storage**: Store `routing_mode` from initial request for analytics/debugging only (doesn't constrain future requests)
- **OpenTelemetry integration**: Add spans for all database operations matching existing Phoenix/OTEL patterns

**Success Criteria**:
- Unit tests pass for all CRUD operations
- Proper error handling (conversation not found, etc.)
- Atomic sequence number generation prevents race conditions
- Tests added incrementally as each method is implemented
- OpenTelemetry spans added for all database operations

**Testing Strategy**:
- Write unit tests AS YOU IMPLEMENT each CRUD method
- Test happy path and error cases for each method
- Mock postgres connection for unit tests
- Test connection pooling behavior
- **Test concurrent message appends** - verify no duplicate sequence numbers
- Run tests with: `uv run pytest tests/unit_tests/test_conversation_db.py`

**Implementation Details**:
```python
# Atomic sequence number generation
# IMPORTANT: Lock the conversation row itself to prevent race conditions
# when no messages exist yet (FOR UPDATE on empty result set locks nothing)
async def append_messages(conversation_id, messages):
    async with conn.transaction():
        # Lock conversation row first to prevent concurrent appends
        await conn.fetchrow(
            "SELECT id FROM conversations WHERE id = $1 FOR UPDATE",
            conversation_id
        )

        # Now get next sequence number safely within transaction
        result = await conn.fetchrow(
            """
            SELECT COALESCE(MAX(sequence_number), 0) + 1 as next_seq
            FROM messages WHERE conversation_id = $1
            """,
            conversation_id
        )
        next_seq = result['next_seq']

        # Insert messages with sequential numbering...
        for i, message in enumerate(messages):
            await conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, sequence_number, metadata)
                VALUES ($1, $2, $3, $4, $5)
                """,
                conversation_id,
                message['role'],
                message.get('content'),
                next_seq + i,
                message.get('metadata', {})
            )
```

**OpenTelemetry Integration**:
```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

async def append_messages(conversation_id, messages):
    """Append messages with OTEL span for observability."""
    with tracer.start_as_current_span("db.append_messages") as span:
        span.set_attribute("db.conversation_id", str(conversation_id))
        span.set_attribute("db.message_count", len(messages))
        span.set_attribute("db.operation", "insert")

        async with conn.transaction():
            # Lock and insert logic...

        span.set_attribute("db.first_sequence_number", next_seq)
```

This pattern matches existing OTEL integration in `api/tracing.py` and `api/passthrough.py`.

---

### Milestone 3: Modify Chat Completions Endpoint
**Goal**: Add stateful/stateless mode detection and conversation management

**Tasks**:
- Implement `extract_system_message()` helper
- Modify `chat_completions()` endpoint logic:
  - Read `X-Conversation-ID` from request headers
  - **No header** → Stateless mode (use messages as-is, no storage)
  - **Header present** → Stateful mode:
    - Empty string/null → New conversation (system message allowed)
    - UUID → Continuation (fail if system message present)
  - Load/create conversation as needed
  - Build full message history (prepend stored system message for continuations)
  - Store assistant response in DB after streaming completes (stateful mode only)
  - Return `X-Conversation-ID` in response headers
- **System message validation**: Fail fast with 400 error if system message in continuation
- **Streaming storage strategy**: Best-effort storage with error indication
  - Buffer assistant message content during streaming
  - After streaming completes, try to store in DB
  - If storage fails, log error and include `storage_failed: true` in last chunk metadata
  - Client can display toast notification: "⚠️ Response not saved to history"

**Header-Based Design**:
- **Request header**:
  - Omitted → Stateless mode (no storage)
  - `X-Conversation-ID: ""` → New conversation
  - `X-Conversation-ID: <uuid>` → Continue conversation
- **Response header**: `X-Conversation-ID: <uuid>` (included for stateful mode)
- **Benefits**: Explicit opt-in, backward compatible, no existing test changes needed

**Success Criteria**:
- Stateless mode works exactly as before (no header)
- Stateful mode creates/loads conversations correctly
- System message validation works (fail on continuation)
- Streaming storage works with error indication
- Integration tests cover both modes
- **All existing tests work unchanged** (no header = stateless)
- New tests added for stateful mode

**Testing Strategy**:
- **Verify existing tests work unchanged**:
  - Run full test suite: `uv run make tests`
  - All existing tests should pass (no header = stateless)
  - No test updates needed ✅

- **Add new stateful tests**: Test conversation creation/loading
  - Test new conversation creation (`X-Conversation-ID: ""`)
  - Test continuing conversation (`X-Conversation-ID: <uuid>`)
  - Test X-Conversation-ID in response headers
  - Test system message on creation (optional)
  - Test system message on continuation (should fail with 400)
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
