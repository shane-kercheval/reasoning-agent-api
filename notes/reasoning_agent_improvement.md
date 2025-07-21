> in @api/reasoning_agent.py do you have any recommendations for best practices for LLM calls or the reasoning agent. For example, the message order is currently System->User->Assistant and i'm guessing it 
  should probably be System->Assistant->User. Those types of things?
  ⎿  Read api/reasoning_agent.py (973 lines)

⏺ Looking at the reasoning_agent.py file, I can identify several best practices and improvements for the LLM calls and reasoning agent architecture:

⏺ Update Todos
  ⎿  ☐ Review message order in reasoning agent (System->User->Assistant vs System->Assistant->User)
     ☐ Check prompt construction patterns for best practices
     ☐ Analyze error handling and retry patterns
     ☐ Review streaming and token usage optimization

⏺ Key Recommendations:

  1. Message Order (Critical Fix)

  You're correct - the current pattern is problematic. In _generate_reasoning_step (lines 680-736), the message construction follows:
  - System
  - User
  - Assistant (context)
  - Assistant (tools)
  - Assistant (schema)

  This violates OpenAI's requirement that messages must alternate between roles. The fix:

  # Lines 680-736 should be restructured to:
  messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": f"Original request: {last_6_messages}"},
  ]

  # Combine ALL assistant content into a single message
  assistant_content_parts = []
  if context["steps"]:
      context_summary = # ... existing code
      assistant_content_parts.append(f"Previous reasoning:\n\n```\n{context_summary}\n```")

  if context["tool_results"]:
      tool_summary = # ... existing code  
      assistant_content_parts.append(f"Tool execution results:\n\n```\n{tool_summary}\n```")

  # ... add tools and schema to assistant_content_parts

  if assistant_content_parts:
      messages.append({
          "role": "assistant",
          "content": "\n\n".join(assistant_content_parts)
      })

  # Then add a final user message to trigger the response
  messages.append({
      "role": "user",
      "content": "Based on the above context, provide your next reasoning step in JSON format."
  })

  2. Token Optimization

  - Line 676-679: You're including the last 6 messages, but this could be optimized by:
    - Implementing a token counter to include as many messages as fit within a budget
    - Summarizing older messages instead of including them verbatim

  3. Error Handling Improvements

  - Lines 759-791: The fallback for malformed JSON is good, but consider:
    - Adding retry logic with exponential backoff
    - Implementing a more sophisticated JSON repair mechanism
    - Logging the raw response for debugging

  4. Streaming Best Practices

  - The unified event flow is excellent, but consider:
    - Adding backpressure handling for slow clients
    - Implementing heartbeat events to prevent timeouts
    - Adding stream position tracking for resume capability

  5. Prompt Engineering

  - Lines 726-736: The JSON schema instructions could be improved:
    - Add examples of valid responses
    - Include explicit instructions about required vs optional fields
    - Consider using OpenAI's new structured outputs feature instead of JSON mode

  6. Context Management

  - The reasoning context grows unbounded - consider:
    - Implementing a sliding window for steps
    - Summarizing older steps to reduce context size
    - Adding token budgeting for context inclusion

⏺ Update Todos
  ⎿  ☒ Review message order in reasoning agent (System->User->Assistant vs System->Assistant->User)
     ☒ Check prompt construction patterns for best practices
     ☒ Analyze error handling and retry patterns
     ☒ Review streaming and token usage optimization
     