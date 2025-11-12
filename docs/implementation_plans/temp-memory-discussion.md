Wow, this opens up some really important architectural questions! Let me think through this systematically.

  Node Results & Context Passing

  I think you're right on several key points:

  1. NodeResult Structure

  class NodeResult(BaseModel):
      node_id: str
      status: Literal["succeeded", "failed"]

      # The primary output (flexible type)
      result: str | dict | BaseModel | None

      # Artifacts/resources produced during execution
      context: dict[str, Any] = {}  # e.g. {"files_read": [...], "urls_scraped": [...], "citations": [...]}

      # Execution metadata
      metadata: dict[str, Any] = {}  # duration_ms, token_count, cost, model_used, etc.
      error_message: str | None = None

  Key insight: Streaming events are for UI/observability, but the assembled result is what gets passed to dependents.

  2. Context Passing to Dependent Nodes

  I see a few levels of context a node might need:

  A. User's Original Objective (string)
  - "Plan a 3-day trip to Paris for 2 people in June"
  - Every node should know the top-level goal

  B. Dependency Results (structured data)
  - Node's direct dependencies: {"research_flights": NodeResult(...), "research_hotels": NodeResult(...)}
  - The orchestrator/executor decides what to extract and pass

  C. Shared Artifacts/Resources (dict accumulation)
  - As nodes execute, they add to a shared resource pool
  - Example: {"flights": [...], "hotels": [...], "weather_data": {...}}
  - Nodes can reference artifacts by key without re-fetching

  D. Conversation History (messages list)
  - Multi-turn conversations need prior context
  - This is where Postgres comes in

  3. How Orchestrator Passes Context

  I think the orchestrator could build context messages like:

  # For node "create_itinerary" that depends on ["research_flights", "research_hotels"]
  messages = [
      {"role": "system", "content": agent_card.prompt},
      {"role": "user", "content": f"Overall objective: {original_request}"},
      {"role": "user", "content": f"""
  Context from dependencies:

  [research_flights]:
  Result: {dep_results["research_flights"].result}
  Resources: {dep_results["research_flights"].context}

  [research_hotels]:
  Result: {dep_results["research_hotels"].result}
  Resources: {dep_results["research_hotels"].context}
  """},
      {"role": "user", "content": node.objective}  # This specific node's task
  ]

  But your question is key: Does the orchestrator need to be smart about what to extract/pass? Or just dump everything?

  Memory System Architecture

  This is fascinating and I think you're identifying a fundamental challenge. Let me break down the options:

  Option 1: Orchestrator-Managed Memory

  - Orchestrator tracks what's been learned/fetched
  - Decides what each node needs
  - Problem: Orchestrator becomes too complex/intelligent

  Option 2: Memory Agent (Your First Idea)

  - Standalone agent that fields natural language queries
  - Stores conversation history, scraped data, etc.
  - Problem: Other agents don't know what's available → lots of "I don't have that" responses → inefficient

  Option 3: Oracle Agent (Your Second Idea)

  - Multi-capability agent with access to:
    - Conversation history (Postgres)
    - Short-term memory (session context, accumulated data)
    - Web search (can fetch missing info on-demand)
    - MCP tools (file access, etc.)
  - Can answer questions AND go fetch data if needed
  - Provides citations/references with responses

  I like Option 3 best because:
  - Single interface ("ask the oracle")
  - Oracle decides: Do I have this? If not, where can I get it?
  - Oracle can use tools (web search, file read, database query)
  - Provides structured responses with citations

  How Oracle Would Work

  # Node needs information
  oracle_node = DAGNode(
      id="get_paris_weather",
      agent="oracle",  # Special multi-tool agent
      objective="What's the weather in Paris in June?",
      depends_on=[]
  )

  # Oracle checks:
  # 1. Conversation history (Postgres) - any prior Paris weather discussions?
  # 2. Short-term memory (session context) - already scraped today?
  # 3. If not found → use web_search tool
  # 4. Return: {"answer": "...", "sources": ["url1", "url2"], "cached": false}

  Alternative pattern: Oracle is NOT a DAG node, but a tool that any agent can call:

  # travel_planner agent has access to oracle tool
  agent_card = AgentCard(
      name="travel_planner",
      tools=["oracle_query", "flight_search", "hotel_search"]
  )

  # Agent calls oracle during execution
  oracle_query("What's the weather in Paris in June?")
  # Oracle returns: structured response with data + sources

  Key Design Questions

  1. Memory Storage Schema (Postgres):
    - Conversations table (user messages, assistant responses, session_id)
    - Artifacts table (files read, URLs scraped, extracted data, session_id)
    - How to query efficiently? Vector embeddings? Full-text search?
  2. Oracle Scope:
    - Is oracle a DAG node (orchestrator uses it explicitly)?
    - Or a tool (any agent can call it)?
    - Or both?
  3. Context Accumulation:
    - Should ExecutionState maintain a shared artifacts dict?
    - Nodes add to it as they execute?
    - Downstream nodes get access automatically?
  4. Orchestrator Intelligence:
    - Simple mode: Pass all dependency results as-is
    - Smart mode: Extract relevant fields, summarize if too long
    - Which for Phase 1?

  My Recommendations

  For Phase 1 (M2-M3):

  1. NodeResult: Use the structure above (result, context, metadata, status)
  2. Context Passing: Simple orchestrator - pass full dependency results + original objective
    - Don't try to be smart about extraction yet
    - Let the receiving agent's prompt + LLM figure out what's relevant
  3. Memory: Defer full memory system to Phase 4
    - Phase 1: In-memory only (no Postgres yet)
    - Nodes can accumulate artifacts in ExecutionState.artifacts dict
    - Pass artifacts dict to nodes that need it
  4. Oracle Agent: Design the interface now, stub for Phase 1, implement in Phase 4
    - Phase 1: No oracle, nodes work with passed context only
    - Phase 4: Add oracle as both a specialized agent AND a tool

  For Phase 4+ (Memory & Oracle):

  1. Postgres Schema:
    - conversations table (session_id, role, content, timestamp)
    - session_artifacts table (session_id, key, value_json, artifact_type, sources)
    - Consider vector embeddings for semantic search
  2. Oracle Agent:
    - Hybrid: Can be a DAG node OR a tool
    - Has access to: conversation history, session artifacts, web search, MCP tools
    - Returns structured responses with citations
  3. Smart Orchestrator:
    - Learns to summarize/extract relevant context
    - Maybe uses LLM to decide what to pass to next node?

  What Do You Think?

  Does this align with your vision? Key questions:

  1. Phase 1 scope: Agree with simple context passing (dump everything) and defer memory/oracle?
  2. Oracle design: Node vs. tool vs. both?
  3. Artifacts dict: Should ExecutionState accumulate a shared artifacts pool that all nodes can reference?
  4. Postgres schema: Does the tables structure I suggested make sense for conversation + artifact storage?