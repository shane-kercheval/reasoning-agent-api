# Critical Concerns & Outstanding Questions
## Dapr Orchestration Implementation Plan

**Purpose:** Track all concerns, outstanding questions, and decisions needed before/during implementation.

---

## Critical Concerns Summary

### **STREAMING ARCHITECTURE (Milestone 5, 8, 9) - RESOLVED WITH A2A**
~~The current system supports streaming, which is critical for UX. However, the plan does not define:~~
- ✅ **SOLVED BY A2A**: Native Server-Sent Events (SSE) support
- ✅ **Pattern**: Each agent streams artifacts via A2A SSE, orchestrator aggregates, translation layer converts to OpenAI chunks
- ✅ **Multi-agent**: Orchestrator subscribes to multiple agent SSE streams, multiplexes/sequences as appropriate
- ✅ **Pause/Resume**: A2A task state transitions stream naturally (running → auth-required → running → completed)
- **ACTION REQUIRED:** Design translation layer for A2A artifacts → OpenAI streaming format

### **MCP INTEGRATION PATTERN (Milestone 3, 4, 7) - CLARIFIED WITH A2A**
~~The current system uses MCP for tool access, but the plan doesn't clarify integration:~~
- ✅ **A2A and MCP are complementary protocols**:
  - **A2A**: Agent-to-agent communication (orchestrator ↔ agents)
  - **MCP**: Agent-to-tools communication (agent ↔ external tools/data sources)
- ✅ **Pattern**: Each agent internally uses MCP client to access tools, externally exposes A2A endpoint
- ✅ **Example**: Reasoning Agent has A2A endpoint, internally uses MCP client for weather/search/etc tools
- ✅ **Planning Agent**: Discovers agent capabilities via Agent Cards, doesn't need to know internal MCP tools
- **ACTION REQUIRED:** Each agent maintains its own MCP client (if it needs tools), standardize MCP config pattern

### **COST TRACKING TOO LATE (Milestone 3-4 vs 13)**
- RAG and multi-agent workflows will multiply API costs significantly starting Milestone 7
- Milestone 13 addresses advanced cost management, but basic tracking needed earlier
- **ACTION REQUIRED:** Add basic token counting and cost attribution starting Milestone 3-4

### **ACTOR MODEL COMPLEXITY (Milestone 5) - LIKELY UNNECESSARY WITH A2A**
- ✅ **A2A tasks provide**: Lifecycle management, state persistence, pause/resume capabilities
- ✅ **Simpler approach**: Stateful orchestrator service using A2A tasks for workflow state
- ✅ **A2A task states** handle what actors provided: created → running → auth-required → completed/failed
- ⚠️ **Actors still useful if**: Need automatic distribution across instances for high-throughput scaling
- **DECISION POINT (Milestone 5):** Start with A2A tasks only, add actors later if scaling requires it

### **PLANNING AGENT AMBITION (Milestone 4) - SIMPLIFIED WITH A2A**
- ✅ **Agent Cards simplify discovery**: Agents self-describe capabilities via /.well-known/agent-card.json
- ✅ **Planning inputs**: Query agent cards to get available agents and their capabilities dynamically
- ⚠️ **LLM-based planning still ambitious**: Generating workflow DAGs with LLM unproven, high risk of incorrect plans
- **RECOMMENDATION:** Start with rule-based planning using Agent Card capabilities, evolve to LLM-based later
- **Example**: If Agent Card shows `"capabilities": ["rag", "knowledge_base"]` and query needs knowledge → use that agent

### **ERROR HANDLING PHILOSOPHY (Milestone 5, 8) - PARTIALLY ADDRESSED BY A2A**
- ✅ **A2A provides standardized task states**: completed, canceled, rejected, failed (clear error semantics)
- ✅ **Error propagation**: Failed task states propagate to orchestrator naturally
- ⚠️ **Partial failure policy still needed**: When 2 of 3 parallel agents succeed but 1 fails, what happens?
  - Option 1: Fail entire workflow (strict)
  - Option 2: Continue with partial results + error notification (graceful degradation)
  - Option 3: Retry failed task, fallback if still fails
- **ACTION REQUIRED:** Define partial failure handling philosophy (likely per-workflow configurable)

### **SECURITY TOPICS MISSING (Milestone 14)**
- No security review, mTLS strategy, secrets management, attack surface analysis
- Rate limiting strategy completely absent
- **ACTION REQUIRED:** Add security section and rate limiting to plan

### **VALUE DELIVERY TIMELINE (Milestone 8)**
- First real multi-agent value delivered at Milestone 8 of 14 (6+ months)
- Earlier milestones deliver infrastructure but limited user-facing value
- **CONSIDERATION:** Can simpler multi-agent workflows be delivered earlier for validation?

---

## Milestone 1: Dapr Foundation - Concerns

### **CONCERN: Milestone Scope Too Large**
- Consider splitting this into Milestone 1a (Dapr state only) and 1b (full sidecar setup) for incremental validation
- Would allow testing state operations before committing to full multi-service architecture

### **CONCERN: Testing Strategy Unclear**
- Need guidance on testing without full Dapr stack for faster iteration during development
- How to mock Dapr components for unit tests?
- Should we support running services standalone (without Dapr) for local development?

### **CONCERN: A2A Protocol Library Availability**
- A2A protocol is relatively new - mature Python libraries may not exist
- May need to implement custom A2A support based on specification
- Need to research available options before committing to implementation approach
- **ACTION:** Task #7 addresses this - research before proceeding with architecture decisions

---

## Milestone 2: State Management - Concerns

### **CONCERN: State Lifecycle and Cleanup Strategy Missing**
- No clear strategy for state archival, cleanup, or TTL management
- Sessions with full multi-agent context (added in later milestones) could grow very large
- Need explicit policy: When are sessions deleted? Archived? How long do we retain them?
- What happens when state store fills up?

### **CONCERN: State Growth with Multi-Agent Context**
- Current design stores "reasoning context (steps, tool results, current iteration)"
- In multi-agent workflows (Milestone 8), this could include results from 5+ agents with large tool outputs
- Need strategy for: State compression? Selective storage? Reference to external storage?

---

## Milestone 3: A2A Protocol - Concerns

### **RESOLVED: Basic Observability Included**
- ✅ Task #6 adds basic OpenTelemetry tracing and structured logging
- ✅ Trace context propagation through A2A protocol headers
- ✅ A2A task IDs provide natural correlation IDs

### **RESOLVED: Basic Cost Tracking Included**
- ✅ Task #7 adds basic token counting and cost attribution
- ✅ A2A artifacts include metadata for token usage
- ✅ Per-task tracking enabled

### **RESOLVED: MCP Integration Clarified**
- ✅ Each agent maintains its own MCP client internally
- ✅ Reasoning agent keeps current MCP integration when extracted
- ✅ MCP is agent-internal concern, not visible in A2A protocol
- ✅ Agent Cards describe capabilities, not implementation details

### **CONCERN: Translation Layer Complexity**
- OpenAI ↔ A2A translation is critical path and could be complex
- Streaming translation especially tricky (SSE → OpenAI chunks)
- Need comprehensive testing of edge cases
- **RECOMMENDATION:** Build translation layer incrementally with extensive tests

### **CONCERN: A2A Library/Implementation**
- Depending on Milestone 1 research, may need custom A2A implementation
- SSE handling in FastAPI requires careful async management
- **DECISION POINT:** Based on Milestone 1 research, decide custom vs library approach

---

## Milestone 4: Planning Agent - Concerns

### **CRITICAL: Planning Agent Ambition - LLM vs Rule-Based**
- Using LLM to generate workflow DAGs is ambitious and unproven
- Will require extensive prompt engineering iteration and testing
- High risk of incorrect plans causing orchestrator failures
- **RECOMMENDATION:** Start with rule-based planning (if query mentions "search" → use search agent) before attempting LLM-based planning
- Rule-based approach is more predictable, testable, and debuggable for MVP
- Can evolve to LLM-based planning in later phase once rule-based patterns are understood

### **CONCERN: MCP Tool Integration**
- Does planning agent need to know about available MCP tools when generating plans?
- Should MCP tool capabilities influence agent selection?
- How does planner discover what MCP tools are currently available?
- **DECISION NEEDED:** Define how MCP tools factor into planning decisions

### **CONCERN: Cost Tracking Should Be Included**
- If not already implemented in Milestone 3, basic token tracking must start here
- Planning agent itself will consume tokens for plan generation
- Need visibility into planning costs before adding more agents
- **RECOMMENDATION:** Ensure basic cost tracking operational before this milestone

### **CONCERN: Plan Quality Evaluation**
- No metrics defined for evaluating if generated plans are "good"
- How do we know if planner is working correctly beyond "doesn't crash"?
- Need test suite with known-good query→plan mappings
- **RECOMMENDATION:** Create evaluation dataset as part of this milestone

---

## Milestone 5: Orchestrator - Concerns

### **RESOLVED: Streaming Architecture with A2A**
- ✅ Task #4 implements multi-agent streaming aggregation
- ✅ Pattern: Subscribe to multiple agent SSE streams, multiplex artifacts, add metadata
- ✅ User sees real-time multiplexed stream with agent/step metadata
- ✅ Pause/resume: When auth-required, orchestrator stops streaming, resumes after approval
- **IMPLEMENTATION DETAIL:** Need to carefully handle async stream merging (asyncio.gather or async queue)

### **RESOLVED: No Actors for MVP**
- ✅ Using A2A task-based orchestration instead of Dapr actors
- ✅ State managed via Dapr state store
- ✅ Pause/resume via A2A auth-required state
- ✅ Simpler architecture, easier to debug
- ⚠️ **Future consideration:** Add actors only if scaling requires automatic distribution

### **CONCERN: Error Handling Philosophy Still Needs Definition**
- ⚠️ What happens when 2 of 3 parallel agents succeed but 1 fails?
- Options:
  1. Fail entire workflow (strict - easy to implement)
  2. Continue with partial results, flag failures (graceful - better UX)
  3. Retry failed steps with backoff
- **DECISION NEEDED:** Define policy before implementation
- **RECOMMENDATION:** Make configurable per workflow - some workflows require all steps, others can handle partial

### **CONCERN: Stream Merging Complexity**
- Merging multiple async SSE streams is non-trivial
- Need to handle: Different arrival rates, different completion times, errors in one stream
- **RECOMMENDATION:** Use async queue pattern or library for stream merging
- Test thoroughly with varying latencies and failure scenarios

### **CONCERN: Workflow State Size**
- Complex workflows with many steps and large artifacts could create large state objects
- State saved after each step - frequent Dapr state writes
- **CONSIDERATION:** Compression? Reference artifacts by ID rather than embedding full content?
- Monitor state size, implement limits if needed

---

## Milestone 6: Vector Database - Concerns

### **CONCERN: Vector Storage Growth**
- Embeddings (especially OpenAI's 1536-dimensional vectors) consume significant space
- Large document corpus + chunking = thousands of vectors
- No retention or cleanup policy defined
- **RECOMMENDATION:** Define vector storage limits, cleanup policies, and monitoring

### **CONCERN: Embedding Cost**
- OpenAI embeddings cost money per token
- Initial data ingestion of entire codebase could be expensive
- Re-embedding on document updates adds ongoing cost
- **RECOMMENDATION:** Calculate estimated embedding costs before proceeding; consider local models for development

---

## Milestone 7: RAG Agent - Concerns

### **CRITICAL: Cost Tracking Should Already Be Operational**
- RAG agent will multiply API costs: embeddings for every query + LLM calls with large context windows
- If cost tracking not implemented in Milestone 3-4, this is extremely risky
- **REQUIREMENT:** Basic token tracking and cost monitoring must be operational before RAG agent deployment

### **CONCERN: MCP Tool Access**
- Does RAG agent need access to MCP tools?
- Current reasoning agent uses MCP for tool calls - does RAG agent need same capability?
- Or is RAG agent purely retrieval + generation without tool use?
- **DECISION NEEDED:** Clarify RAG agent's relationship to MCP tools

### **CONCERN: Streaming with RAG**
- Retrieval happens before LLM generation
- Should user see: (1) "Searching..." → results → streaming generation, or (2) Buffer everything, stream at end?
- How does this fit into orchestrator's streaming architecture (from Milestone 5)?
- Need consistency in streaming UX across all agents

---

## Milestone 8: Multi-Agent Integration - Concerns

### **CRITICAL: First Real User Value Delivered Too Late**
- This is Milestone 8 of 14 - user doesn't see multi-agent benefits until 6+ months in
- Earlier milestones deliver infrastructure but limited user-facing value
- **RECOMMENDATION:** Consider if simpler multi-agent workflows could be delivered earlier
- Could we have basic 2-agent orchestration in Milestone 5 as proof-of-value?

### **RESOLVED: Streaming with Multiple Agents via A2A**
- ✅ Resolved in Milestone 5 Task #4 - multi-agent streaming aggregation
- ✅ Pattern: Orchestrator subscribes to multiple agent SSE streams, multiplexes artifacts
- ✅ User sees single stream with metadata identifying agent/step
- ✅ A2A artifacts include metadata for source tracking

### **CONCERN: Partial Failure Handling Philosophy**
- "Error propagation: Failures handled at all levels" - but HOW?
- If RAG agent fails but reasoning agent succeeds, what does user get?
- Does system show partial results? Retry the failed agent? Fail entire request?
- **DECISION NEEDED:** Define partial failure UX before implementing complex workflows

### **CONCERN: Result Aggregation Complexity**
- "Results in different formats" - need standardized agent output schema
- "Conflicting information" - who arbitrates? Orchestrator? Another agent?
- "Citation merging" - how to combine citations from RAG + search agents?
- This task (#5) may need its own detailed design document

### **CONCERN: Token Usage Attribution**
- "Token usage correctly attributed" - attributed to what? Per-agent? Per-workflow? Per-user?
- Need clear attribution model defined before testing

---

## Milestone 9: Human-in-the-Loop - Concerns

### **RESOLVED: Streaming During Pause/Resume with A2A**
- ✅ A2A pattern: When task status changes to auth-required, SSE stream sends status update, then pauses
- ✅ After approval (PUT /tasks/{task_id}), task status → running, SSE stream resumes
- ✅ Client sees: stream → status:auth-required → pause → approval → stream resumes
- **IMPLEMENTATION**: SSE connection stays open during pause, sends periodic keepalives

### **RESOLVED: UI Real-Time Updates via A2A SSE**
- ✅ Use A2A's SSE streaming for real-time notifications
- ✅ Web client subscribes to orchestration task stream
- ✅ When status→auth-required, stream sends artifact with approval context
- ✅ No need for WebSocket or polling - SSE is built into A2A protocol

### **RESOLVED: Long-Running Approval State Management**
- ✅ A2A task state persisted in Dapr state store (not in-memory)
- ✅ Task with auth-required status and auth_context survives service restarts
- ✅ Client can reconnect to SSE stream at any time via GET /tasks/{task_id}/stream
- **CONSIDERATION**: Set reasonable TTL for auth-required tasks (e.g., 7 days) to prevent indefinite waiting

---

## Milestone 10: Observability - Concerns

### **CONCERN: Observability Should Have Started in Milestone 2-3**
- This milestone is appropriate for ADVANCED observability features
- But basic tracing and structured logging should have been implemented in Milestone 2-3
- If not already done, this is a problem for debugging Milestones 4-9
- **RECOMMENDATION:** If basic observability not yet implemented, prioritize it immediately

### **CONCERN: Metrics vs Phoenix Traces**
- Phoenix provides tracing; this milestone adds metrics (Prometheus/OTel)
- Are we running separate Prometheus instance? Or using Phoenix's built-in metrics?
- Need clarity on observability stack architecture
- **DECISION NEEDED:** Define complete observability stack (tracing + metrics + logs)

---

## Milestone 11: Performance Optimization - Concerns

### **CONCERN: Timing Appropriate**
- Performance optimization at this stage makes sense
- System should be functionally complete before optimizing
- No major concerns for this milestone

---

## Milestone 12: Advanced RAG - Concerns

### **CONCERN: Advanced Features Appropriate Here**
- These are genuinely advanced features that should come after basic RAG is working
- Timing is good - Milestone 7 handles basics, this handles advanced
- No major concerns for this milestone

### **CONCERN: Cost Impact**
- Reranking, query expansion, and contextual retrieval all add API costs
- Contextual retrieval especially: Need LLM call to generate context for EACH chunk during ingestion
- Ensure cost tracking (Milestone 13 or earlier) is monitoring these features
- May want A/B testing to measure quality improvement vs cost increase

---

## Milestone 13: Cost Management - Concerns

### **CRITICAL: Cost Tracking Should Have Started in Milestone 3-4**
- This milestone for ADVANCED cost management is appropriately timed
- But BASIC token tracking should have started much earlier (Milestone 3-4)
- By this point (after RAG, multi-agent, advanced RAG), costs could already be significant
- **RECOMMENDATION:** If basic tracking not yet implemented, this is high priority
- Advanced features here (budgets, dashboards, model selection) are good additions to basic tracking

### **CONCERN: Model Selection in Planning Agent**
- Task #4 says "Planning agent includes model selection in workflow plan"
- This is architectural change to Planning Agent (Milestone 4)
- Should this have been part of Milestone 4, or is it correctly placed here as enhancement?
- **CLARIFICATION NEEDED:** Is this a new capability or enhancement to existing planning?

---

## Questions to Ask Before Proceeding (by Milestone)

### Milestone 1
1. Should we use Redis or PostgreSQL for the Dapr state store from the start? (Both will be available)
2. What should be the password strategy for app-postgres? (Store in .env, use secrets management?)
3. Should we enable mTLS between services now or wait until production?

### Milestone 2
1. What should be the default TTL for sessions?
2. Should we implement session state compression for large contexts?
3. Do we want a session listing/search endpoint?
4. How should we handle session cleanup (automatic vs manual)?

### Milestone 3
1. Should we add authentication between services beyond mTLS?
2. What should be the default timeout for reasoning service calls?
3. Do we want versioning in the service interface?
4. Should we add load balancing configuration for multiple reasoning instances?

### Milestone 4
1. Should planning agent cache plans for similar queries?
2. How should we handle ambiguous queries (multiple possible plans)?
3. Should users be able to provide custom workflow plans?
4. Do we want plan versioning for iterative refinement?

### Milestone 5
1. How should we handle partial workflow failures (fail all vs continue)?
2. What's the appropriate timeout for each step?
3. Should we support workflow checkpointing (save intermediate results)?
4. Do we want workflow retry at orchestrator level or just step level?

### Milestone 6
1. Should we use OpenAI embeddings or local model?
2. What chunking strategy for long documents?
3. Do we need real-time indexing or batch updates?
4. What metadata fields should we support?

### Milestone 7
1. Should we implement reranking from the start?
2. What's the right tradeoff between precision and recall?
3. Do we want user feedback on retrieval quality?
4. Should we cache embeddings for common queries?

### Milestone 8
1. Should we add workflow templates for common patterns?
2. Do we want automatic workflow optimization learning?
3. How should we handle conflicting results from different agents?

### Milestone 9
1. Should approvals be revocable after submission?
2. What's appropriate default timeout for approvals?
3. Do we need multi-level approvals (different approval authorities)?
4. Should we support delegated approvals?

### Milestone 10
1. What metrics retention policy should we use?
2. Which metrics are most critical for alerts?
3. Should we implement rate limiting at this stage?
4. Do we need log aggregation beyond Phoenix?

### Milestone 11
1. What's acceptable latency for different query types?
2. Should we implement adaptive rate limiting?
3. Do we need request prioritization?
4. What cache hit rate indicates success?

### Milestone 12
1. Is reranking worth the latency cost?
2. Should hybrid search be default or opt-in?
3. How many query variations for expansion?
4. Should contextual retrieval be applied to all documents?

### Milestone 13
1. What should be default budget limits?
2. Should we support cost alerts?
3. How should we handle over-budget requests?
4. Do we need per-user cost isolation?

---

## Action Items Summary

**Immediate (Before Starting):**
- [ ] Research A2A protocol Python implementation options (M1)
- [ ] Define partial failure handling philosophy (M5, M8)
- [ ] Add basic cost tracking plan to M3 (don't wait for M13)
- [ ] Add security section to plan (M14 or earlier)

**Milestone 1:**
- [ ] Enable mTLS decision
- [ ] App-postgres password strategy
- [ ] A2A library vs custom implementation

**Milestone 2:**
- [ ] Session TTL and cleanup strategy
- [ ] State compression strategy for large contexts

**Milestone 3:**
- [ ] Implement basic cost tracking (Task #7)
- [ ] Standardize MCP config pattern per agent

**Milestone 4:**
- [ ] Decide: LLM-based or rule-based planning (recommend rule-based)
- [ ] MCP tools in planning decisions
- [ ] Create plan evaluation dataset

**Milestone 5:**
- [ ] Define partial failure policy (strict vs graceful)
- [ ] Implement stream merging pattern (async queue)
- [ ] Monitor workflow state size

**Milestone 6:**
- [ ] Vector storage limits and cleanup policy
- [ ] Calculate embedding costs (OpenAI vs local)

**Milestone 7:**
- [ ] Verify cost tracking operational
- [ ] RAG agent MCP tool access decision
- [ ] RAG streaming UX consistency

**Milestone 10:**
- [ ] Define complete observability stack
- [ ] If basic observability missing, add immediately
