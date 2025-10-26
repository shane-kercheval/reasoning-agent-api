# Implementation Plan: Backend Conversation Storage

## Overview

Add postgres-backed conversation storage to the reasoning API, enabling:
- Persistent conversation history across sessions
- Cleaner client implementation (send only new messages, not full history)
- Foundation for future features (search, export, sharing, multi-device sync)

**Status**: ACTIVE - Required for Electron desktop client migration.

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
- Postgres database available (already have `postgres-phoenix` and `postgres-litellm`)
- Alembic migrations setup for API schema changes

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
Request:
{
    "conversation_id": "uuid",  # Optional
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
   - NO → STATEFUL MODE: Check conversation_id

2. Stateful mode logic:
   - If conversation_id provided:
     - Load conversation from DB
     - Append new user message
     - Use full history for LLM call
   - If conversation_id omitted:
     - Create new conversation in DB
     - Use user message for LLM call
     - Return conversation_id in response metadata

3. Response format (streaming):
   - First chunk includes conversation_id in metadata
   - Desktop client stores conversation_id for next request
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

async def chat_completions(...):
    # Determine mode
    if is_stateless_request(request):
        # Current behavior - use messages as-is, don't store
        messages = request.messages
        conversation_id = None
    else:
        # Stateful mode - manage conversation
        conversation_id = request.model_dump().get("conversation_id")  # Custom field

        if conversation_id:
            # Load existing conversation
            conversation = await db.get_conversation(conversation_id)
            messages = conversation.messages + request.messages  # Append new
            await db.append_messages(conversation_id, request.messages)
        else:
            # Create new conversation
            conversation_id = await db.create_conversation(request.messages)
            messages = request.messages

    # Make LLM call with full message history
    # ... existing streaming logic ...

    # Include conversation_id in first chunk metadata (if stateful)
    if conversation_id and not is_stateless_request(request):
        # Add to first chunk
        first_chunk["metadata"] = {"conversation_id": conversation_id}
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
- No immediate truncation needed (start simple)
- Future: Pagination for very long conversations (e.g., >100 messages)

**Backward Compatibility**:
- Stateless mode works exactly as before (system message present)
- No breaking changes for existing clients
- Desktop client opts into stateful mode (no system message)

---

## Implementation Milestones

### Milestone 1: Database Setup & Migrations
**Goal**: Set up postgres schema for conversations

**Tasks**:
- Add Alembic to project (if not already present)
- Create migration for `conversations` and `messages` tables
- Add indexes for performance
- Test migration locally

**Success Criteria**:
- `alembic upgrade head` creates tables
- Schema matches design above

---

### Milestone 2: Database Layer
**Goal**: Create conversation CRUD operations

**Tasks**:
- Create `api/database/` module
- Implement `ConversationDB` class with async methods:
  - `create_conversation(messages) -> conversation_id`
  - `get_conversation(conversation_id) -> Conversation`
  - `append_messages(conversation_id, messages)`
  - `list_conversations(limit, offset) -> list[Conversation]`
  - `delete_conversation(conversation_id)`
- Use `asyncpg` or SQLAlchemy async for postgres access
- Handle connection pooling

**Success Criteria**:
- Unit tests pass for all CRUD operations
- Proper error handling (conversation not found, etc.)

---

### Milestone 3: Modify Chat Completions Endpoint
**Goal**: Add stateful/stateless mode detection and conversation management

**Tasks**:
- Add `conversation_id` field to `OpenAIChatRequest` (optional)
- Implement `is_stateless_request()` helper
- Modify `chat_completions()` endpoint logic:
  - Detect mode (stateless vs stateful)
  - Load/create conversation as needed
  - Build full message history for LLM call
  - Store assistant response in DB (stateful mode only)
- Include `conversation_id` in response metadata

**Success Criteria**:
- Stateless mode works exactly as before (system message present)
- Stateful mode creates/loads conversations correctly
- Integration tests cover both modes

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
- Integration tests pass

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

### Milestone 6: Testing & Polish
**Goal**: Comprehensive testing and edge case handling

**Tasks**:
- Integration tests for full conversation lifecycle
- Test conversation with reasoning events
- Test conversation with tool calls
- Test pagination for long conversations
- Test concurrent requests to same conversation
- Add OpenTelemetry span attributes for conversation_id
- Performance testing (load conversation with 100+ messages)

**Success Criteria**:
- All tests pass
- Performance acceptable (<100ms to load conversation)
- Tracing includes conversation metadata

---

## Postgres Database Selection

**Decision**: Create new postgres instance for reasoning API

**Options**:
1. **New postgres instance** (reasoning-postgres) - Recommended
2. Reuse postgres-phoenix - Not recommended (separate concerns)
3. Reuse postgres-litellm - Not recommended (separate concerns)

**Recommendation**: New instance

**Rationale**:
- Clean separation of concerns (API data vs observability data vs LiteLLM data)
- Independent scaling and backup strategies
- Easier to migrate/upgrade independently
- Postgres is lightweight - container overhead is minimal

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

**Database Access**:
- **Option A**: SQLAlchemy 2.0 async (recommended)
  - ORM benefits (type safety, migrations, relationships)
  - Alembic integration for schema migrations
  - Familiar to Python developers

- **Option B**: asyncpg (raw SQL)
  - Faster performance
  - More control
  - More verbose

**Recommendation**: SQLAlchemy 2.0 async for maintainability

**Migration Tool**: Alembic (standard with SQLAlchemy)

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
