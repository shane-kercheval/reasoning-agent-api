# Critical Concerns & Outstanding Questions
## Dapr Orchestration Implementation Plan

**Purpose:** Track all concerns, outstanding questions, and decisions needed before/during implementation.

---

## Milestone 1: Dapr Foundation

### Concerns

**Milestone Scope Too Large**
- Consider splitting this into Milestone 1a (Dapr state only) and 1b (full sidecar setup) for incremental validation
- Would allow testing state operations before committing to full multi-service architecture

**Testing Strategy Unclear**
- Need guidance on testing without full Dapr stack for faster iteration during development
- How to mock Dapr components for unit tests?
- Should we support running services standalone (without Dapr) for local development?

**A2A Protocol Library Availability**
- A2A protocol is relatively new - mature Python libraries may not exist
- May need to implement custom A2A support based on specification
- Need to research available options before committing to implementation approach
- **ACTION:** Task #7 addresses this - research before proceeding with architecture decisions

### Questions
1. Should we use Redis or PostgreSQL for the Dapr state store from the start? (Both will be available)
2. What should be the password strategy for app-postgres? (Store in .env, use secrets management?)
3. Should we enable mTLS between services now or wait until production?

### Action Items
- [ ] Research A2A protocol Python implementation options
- [ ] Enable mTLS decision
- [ ] App-postgres password strategy
- [ ] A2A library vs custom implementation

---

## Milestone 2: State Management

### Concerns

**State Lifecycle and Cleanup Strategy Missing**
- No clear strategy for state archival, cleanup, or TTL management
- Sessions with full multi-agent context (added in later milestones) could grow very large
- Need explicit policy: When are sessions deleted? Archived? How long do we retain them?
- What happens when state store fills up?

**State Growth with Multi-Agent Context**
- Current design stores "reasoning context (steps, tool results, current iteration)"
- In multi-agent workflows (Milestone 8), this could include results from 5+ agents with large tool outputs
- Need strategy for: State compression? Selective storage? Reference to external storage?

### Questions
1. What should be the default TTL for sessions?
2. Should we implement session state compression for large contexts?
3. Do we want a session listing/search endpoint?
4. How should we handle session cleanup (automatic vs manual)?

### Action Items
- [ ] Session TTL and cleanup strategy
- [ ] State compression strategy for large contexts

---

## Milestone 3: A2A Protocol Implementation

### Concerns

**Translation Layer Complexity**
- OpenAI ↔ A2A translation is critical path and could be complex
- Streaming translation especially tricky (SSE → OpenAI chunks)
- Need comprehensive testing of edge cases
- **RECOMMENDATION:** Build translation layer incrementally with extensive tests

**A2A Library/Implementation**
- Depending on Milestone 1 research, may need custom A2A implementation
- SSE handling in FastAPI requires careful async management
- **DECISION POINT:** Based on Milestone 1 research, decide custom vs library approach

**MCP Integration Pattern**
- Each agent maintains its own MCP client (if it needs tools)
- Need to standardize MCP config pattern across agents
- **ACTION REQUIRED:** Standardize MCP config pattern per agent

**Cost Tracking TOO LATE**
- RAG and multi-agent workflows will multiply API costs significantly starting Milestone 7
- Milestone 13 addresses advanced cost management, but basic tracking needed earlier
- **ACTION REQUIRED:** Add basic token counting and cost attribution starting Milestone 3-4

### Questions
1. Should we add authentication between services beyond mTLS?
2. What should be the default timeout for reasoning service calls?
3. Do we want versioning in the service interface?
4. Should we add load balancing configuration for multiple reasoning instances?

### Action Items
- [ ] Implement basic cost tracking (Task #7)
- [ ] Standardize MCP config pattern per agent
- [ ] Design translation layer for A2A artifacts → OpenAI streaming format

---

## Milestone 4: Planning Agent

### Concerns

**CRITICAL: Planning Agent Ambition - LLM vs Rule-Based**
- Using LLM to generate workflow DAGs is ambitious and unproven
- Will require extensive prompt engineering iteration and testing
- High risk of incorrect plans causing orchestrator failures
- **RECOMMENDATION:** Start with rule-based planning (if query mentions "search" → use search agent) before attempting LLM-based planning
- Rule-based approach is more predictable, testable, and debuggable for MVP
- Can evolve to LLM-based planning in later phase once rule-based patterns are understood

**MCP Tool Integration**
- Does planning agent need to know about available MCP tools when generating plans?
- Should MCP tool capabilities influence agent selection?
- How does planner discover what MCP tools are currently available?
- **DECISION NEEDED:** Define how MCP tools factor into planning decisions

**Cost Tracking Should Be Operational**
- If not already implemented in Milestone 3, basic token tracking must start here
- Planning agent itself will consume tokens for plan generation
- Need visibility into planning costs before adding more agents
- **RECOMMENDATION:** Ensure basic cost tracking operational before this milestone

**Plan Quality Evaluation**
- No metrics defined for evaluating if generated plans are "good"
- How do we know if planner is working correctly beyond "doesn't crash"?
- Need test suite with known-good query→plan mappings
- **RECOMMENDATION:** Create evaluation dataset as part of this milestone

### Questions
1. Should planning agent cache plans for similar queries?
2. How should we handle ambiguous queries (multiple possible plans)?
3. Should users be able to provide custom workflow plans?
4. Do we want plan versioning for iterative refinement?

### Action Items
- [ ] Decide: LLM-based or rule-based planning (recommend rule-based)
- [ ] MCP tools in planning decisions
- [ ] Create plan evaluation dataset

---

## Milestone 5: Orchestrator Implementation

### Concerns

**Error Handling Philosophy Still Needs Definition**
- ⚠️ What happens when 2 of 3 parallel agents succeed but 1 fails?
- Options:
  1. Fail entire workflow (strict - easy to implement)
  2. Continue with partial results, flag failures (graceful - better UX)
  3. Retry failed steps with backoff
- **DECISION NEEDED:** Define policy before implementation
- **RECOMMENDATION:** Make configurable per workflow - some workflows require all steps, others can handle partial
- **NOTE:** This affects Milestone 8 as well

**Stream Merging Complexity**
- Merging multiple async SSE streams is non-trivial
- Need to handle: Different arrival rates, different completion times, errors in one stream
- **RECOMMENDATION:** Use async queue pattern or library for stream merging
- Test thoroughly with varying latencies and failure scenarios
- **IMPLEMENTATION DETAIL:** Need to carefully handle async stream merging (asyncio.gather or async queue)

**Workflow State Size**
- Complex workflows with many steps and large artifacts could create large state objects
- State saved after each step - frequent Dapr state writes
- **CONSIDERATION:** Compression? Reference artifacts by ID rather than embedding full content?
- Monitor state size, implement limits if needed

**Actor Model Complexity**
- **DECISION POINT:** Start with A2A tasks only, add actors later if scaling requires it
- Actors still useful if need automatic distribution across instances for high-throughput scaling

### Questions
1. How should we handle partial workflow failures (fail all vs continue)?
2. What's the appropriate timeout for each step?
3. Should we support workflow checkpointing (save intermediate results)?
4. Do we want workflow retry at orchestrator level or just step level?

### Action Items
- [ ] Define partial failure policy (strict vs graceful)
- [ ] Implement stream merging pattern (async queue)
- [ ] Monitor workflow state size

---

## Milestone 6: Vector Database Setup

### Concerns

**Vector Storage Growth**
- Embeddings (especially OpenAI's 1536-dimensional vectors) consume significant space
- Large document corpus + chunking = thousands of vectors
- No retention or cleanup policy defined
- **RECOMMENDATION:** Define vector storage limits, cleanup policies, and monitoring

**Embedding Cost**
- OpenAI embeddings cost money per token
- Initial data ingestion of entire codebase could be expensive
- Re-embedding on document updates adds ongoing cost
- **RECOMMENDATION:** Calculate estimated embedding costs before proceeding; consider local models for development

### Questions
1. Should we use OpenAI embeddings or local model?
2. What chunking strategy for long documents?
3. Do we need real-time indexing or batch updates?
4. What metadata fields should we support?

### Action Items
- [ ] Vector storage limits and cleanup policy
- [ ] Calculate embedding costs (OpenAI vs local)

---

## Milestone 7: RAG Agent Implementation

### Concerns

**CRITICAL: Cost Tracking Should Already Be Operational**
- RAG agent will multiply API costs: embeddings for every query + LLM calls with large context windows
- If cost tracking not implemented in Milestone 3-4, this is extremely risky
- **REQUIREMENT:** Basic token tracking and cost monitoring must be operational before RAG agent deployment

**MCP Tool Access**
- Does RAG agent need access to MCP tools?
- Current reasoning agent uses MCP for tool calls - does RAG agent need same capability?
- Or is RAG agent purely retrieval + generation without tool use?
- **DECISION NEEDED:** Clarify RAG agent's relationship to MCP tools

**Streaming with RAG**
- Retrieval happens before LLM generation
- Should user see: (1) "Searching..." → results → streaming generation, or (2) Buffer everything, stream at end?
- How does this fit into orchestrator's streaming architecture (from Milestone 5)?
- Need consistency in streaming UX across all agents

### Questions
1. Should we implement reranking from the start?
2. What's the right tradeoff between precision and recall?
3. Do we want user feedback on retrieval quality?
4. Should we cache embeddings for common queries?

### Action Items
- [ ] Verify cost tracking operational
- [ ] RAG agent MCP tool access decision
- [ ] RAG streaming UX consistency

---

## Milestone 8: Multi-Agent Integration

### Concerns

**CRITICAL: First Real User Value Delivered Too Late**
- This is Milestone 8 of 14 - user doesn't see multi-agent benefits until 6+ months in
- Earlier milestones deliver infrastructure but limited user-facing value
- **RECOMMENDATION:** Consider if simpler multi-agent workflows could be delivered earlier
- Could we have basic 2-agent orchestration in Milestone 5 as proof-of-value?

**Partial Failure Handling Philosophy**
- "Error propagation: Failures handled at all levels" - but HOW?
- If RAG agent fails but reasoning agent succeeds, what does user get?
- Does system show partial results? Retry the failed agent? Fail entire request?
- **DECISION NEEDED:** Define partial failure UX before implementing complex workflows
- **NOTE:** Related to Milestone 5 error handling philosophy

**Result Aggregation Complexity**
- "Results in different formats" - need standardized agent output schema
- "Conflicting information" - who arbitrates? Orchestrator? Another agent?
- "Citation merging" - how to combine citations from RAG + search agents?
- This task (#5) may need its own detailed design document

**Token Usage Attribution**
- "Token usage correctly attributed" - attributed to what? Per-agent? Per-workflow? Per-user?
- Need clear attribution model defined before testing

### Questions
1. Should we add workflow templates for common patterns?
2. Do we want automatic workflow optimization learning?
3. How should we handle conflicting results from different agents?

### Action Items
- [ ] Define result aggregation approach for conflicting information and citation merging
- [ ] Define token usage attribution model (per-agent, per-workflow, per-user)

---

## Milestone 9: Human-in-the-Loop UI

### Concerns

**Approval Timeout TTL**
- Set reasonable TTL for auth-required tasks (e.g., 7 days) to prevent indefinite waiting

### Questions
1. Should approvals be revocable after submission?
2. What's appropriate default timeout for approvals?
3. Do we need multi-level approvals (different approval authorities)?
4. Should we support delegated approvals?

### Action Items
- [ ] Define approval timeout TTL policy for auth-required tasks

---

## Milestone 10: Observability and Production Readiness

### Concerns

**Observability Should Have Started in Milestone 2-3**
- This milestone is appropriate for ADVANCED observability features
- But basic tracing and structured logging should have been implemented in Milestone 2-3
- If not already done, this is a problem for debugging Milestones 4-9
- **RECOMMENDATION:** If basic observability not yet implemented, prioritize it immediately

**Metrics vs Phoenix Traces**
- Phoenix provides tracing; this milestone adds metrics (Prometheus/OTel)
- Are we running separate Prometheus instance? Or using Phoenix's built-in metrics?
- Need clarity on observability stack architecture
- **DECISION NEEDED:** Define complete observability stack (tracing + metrics + logs)

### Questions
1. What metrics retention policy should we use?
2. Which metrics are most critical for alerts?
3. Should we implement rate limiting at this stage?
4. Do we need log aggregation beyond Phoenix?

### Action Items
- [ ] Define complete observability stack architecture (Phoenix vs Prometheus, metrics retention)
- [ ] Decide on rate limiting implementation approach
- [ ] If basic observability missing, add immediately

---

## Milestone 11: Performance Optimization

### Concerns

**Timing Appropriate**
- Performance optimization at this stage makes sense
- System should be functionally complete before optimizing
- No major concerns for this milestone

### Questions
1. What's acceptable latency for different query types?
2. Should we implement adaptive rate limiting?
3. Do we need request prioritization?
4. What cache hit rate indicates success?

### Action Items
- None specific

---

## Milestone 12: Advanced RAG Features

### Concerns

**Cost Impact**
- Reranking, query expansion, and contextual retrieval all add API costs
- Contextual retrieval especially: Need LLM call to generate context for EACH chunk during ingestion
- Ensure cost tracking (Milestone 13 or earlier) is monitoring these features
- May want A/B testing to measure quality improvement vs cost increase

### Questions
1. Is reranking worth the latency cost?
2. Should hybrid search be default or opt-in?
3. How many query variations for expansion?
4. Should contextual retrieval be applied to all documents?

### Action Items
- [ ] Evaluate cost-benefit tradeoff for advanced features (reranking latency, contextual retrieval cost)

---

## Milestone 13: Cost Management and Optimization

### Concerns

**CRITICAL: Cost Tracking Should Have Started in Milestone 3-4**
- This milestone for ADVANCED cost management is appropriately timed
- But BASIC token tracking should have started much earlier (Milestone 3-4)
- By this point (after RAG, multi-agent, advanced RAG), costs could already be significant
- **RECOMMENDATION:** If basic tracking not yet implemented, this is high priority
- Advanced features here (budgets, dashboards, model selection) are good additions to basic tracking

**Model Selection in Planning Agent**
- Task #4 says "Planning agent includes model selection in workflow plan"
- This is architectural change to Planning Agent (Milestone 4)
- Should this have been part of Milestone 4, or is it correctly placed here as enhancement?
- **CLARIFICATION NEEDED:** Is this a new capability or enhancement to existing planning?

### Questions
1. What should be default budget limits?
2. Should we support cost alerts?
3. How should we handle over-budget requests?
4. Do we need per-user cost isolation?

### Action Items
- [ ] Define default budget limits and over-budget handling approach

---

## Milestone 14: System Documentation and Deployment

### Concerns

**CRITICAL: Security Topics Missing**
- Production readiness checklist doesn't include security items
- Missing from plan:
  - **mTLS**: Mentioned as optional in Milestone 1, but should it be required for production?
  - **Secrets Management**: How are API keys, tokens stored/rotated? Using Docker secrets? External vault?
  - **Attack Surface Analysis**: What are the external-facing endpoints? Rate limiting? DDoS protection?
  - **Input Validation**: How do we prevent malicious inputs to planning agent, prompts, etc.?
  - **Service-to-Service Auth**: Beyond mTLS, do we need service tokens/credentials?
- **REQUIREMENT:** Add security review section to this milestone

**Rate Limiting Strategy Missing**
- No rate limiting mentioned anywhere in plan
- Critical for: (1) Cost control, (2) Abuse prevention, (3) Fair usage
- Where should rate limiting be implemented? API gateway? Per-service? Per-user?
- **DECISION NEEDED:** Define rate limiting strategy and add to appropriate milestone

**Service Versioning and Deployment Strategy**
- How to deploy updates without downtime?
- How to handle breaking changes in service interfaces?
- Blue-green deployment? Rolling updates? Canary deployments?
- What's the rollback procedure if deployment fails?
- **DECISION NEEDED:** Define deployment strategy before production

**State Lifecycle and Archival**
- State cleanup strategy mentioned in Milestone 2 but never resolved
- What's the long-term retention policy for:
  - Session state
  - Workflow history
  - Approval decisions
  - Vector embeddings
  - Cost/usage metrics
- Need archival strategy for compliance, debugging, analytics
- **DECISION NEEDED:** Define comprehensive data retention and archival policy

### Questions
- None additional (concerns cover the questions)

### Action Items
- [ ] Add security review section (mTLS, secrets management, attack surface, input validation, service auth)
- [ ] Define deployment strategy (zero-downtime updates, rollback procedures)
- [ ] Define comprehensive data retention and archival policy for all state types
- [ ] Define rate limiting strategy

---

## Cross-Cutting Immediate Actions (Before Starting)

These should be addressed before beginning implementation:

- [ ] Research A2A protocol Python implementation options (affects M1)
- [ ] Define partial failure handling philosophy (affects M5, M8)
- [ ] Add basic cost tracking plan to M3 (don't wait for M13)
- [ ] Add security section to plan (M14 or earlier)
