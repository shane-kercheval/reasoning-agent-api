1. Do we really want to use OpenAI interface?
    - If so, do we want to support the new "response" api format?
2. How do agents pass back "reasoning" events to the orchestrator and ultimately to the client? We are in a A2A agent and the event needs to get back to the original client that is using OpenAI interface.
3. How do we handle state?
    - OpenAI interface is stateless, so that implies they pass entire conversation history each time. 
    - To what degree do we do that? We will have to have state management for a session which corresponds to the summary, etc. Does the client pass a session id? If so why do we need to have them pass entire history? But then that breaks the OpenAI interface.
    - Can we use hashes of prior messages to identify sessions?
