# LiteLLM Migration Research Findings

**Date:** 2025-10-25
**Research Scripts:** `research_litellm.py`, `research_litellm_pooling.py`, `research_litellm_errors.py`

## Executive Summary

✅ **Migration is FEASIBLE and STRAIGHTFORWARD**

LiteLLM's `acompletion()` is highly compatible with AsyncOpenAI and provides excellent built-in connection pooling. The migration can proceed as planned with minimal adjustments.

---

## Key Findings

### 1. ✅ API Compatibility (Task #1)

**Status:** EXCELLENT compatibility

#### Parameter Support
All required parameters are supported by `litellm.acompletion()`:
- ✓ `model`, `messages`, `temperature`, `max_tokens`, `top_p`, `n`
- ✓ `stop`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `user`
- ✓ `stream`, `stream_options`, `extra_headers`, `response_format`

#### Response Format
```python
# AsyncOpenAI approach (current)
response = await openai_client.chat.completions.create(...)

# LiteLLM approach (new)
response = await litellm.acompletion(...)

# Both return the same response structure!
```

**Key Difference:** Function call vs. client method
- AsyncOpenAI: `await client.chat.completions.create(...)`
- LiteLLM: `await litellm.acompletion(...)`

---

### 2. ✅ Streaming Response Format (Task #2)

**Status:** FULLY COMPATIBLE

#### Response Object Structure
```python
# Non-streaming
response: litellm.types.utils.ModelResponse
  - .model_dump()  ✓ EXISTS
  - .dict()        ✓ EXISTS
  - .choices       ✓ SAME STRUCTURE
  - .usage         ✓ SAME STRUCTURE
  - .model, .id, .created  ✓ ALL PRESENT

# Streaming chunks
chunk: litellm.types.utils.ModelResponseStream
  - .model_dump()  ✓ EXISTS
  - .choices[0].delta.content  ✓ SAME STRUCTURE
  - .choices[0].delta.role     ✓ SAME STRUCTURE
  - .choices[0].finish_reason  ✓ SAME STRUCTURE
```

**Streaming test results:**
- Total chunks: 7 (includes role chunk, content chunks, finish chunk, usage chunk)
- `stream_options={"include_usage": True}` works correctly
- Chunk structure is identical to OpenAI SDK

**NO CHANGES NEEDED** to streaming response processing code!

---

### 3. ⚠️ Structured Outputs with Pydantic (Task #7)

**Status:** WORKS but requires manual parsing

#### Current Code (AsyncOpenAI with `.parse()`)
```python
# request_router.py:301 - Uses .parse() method
response = await openai_client.chat.completions.parse(
    model=settings.routing_classifier_model,
    messages=[...],
    response_format=ClassifierRoutingDecision,  # Pydantic model
)
# Response has .choices[0].message.parsed attribute
decision = response.choices[0].message.parsed
```

#### LiteLLM Approach
```python
# litellm supports response_format with Pydantic models
response = await litellm.acompletion(
    model=settings.routing_classifier_model,
    messages=[...],
    response_format=ClassifierRoutingDecision,  # Pydantic model
)
# BUT: No .parsed attribute - returns JSON string in .content
content = response.choices[0].message.content  # JSON string
decision = ClassifierRoutingDecision.model_validate_json(content)
```

**Required Change:**
- In `request_router.py`, manually parse the JSON response
- Change from `response.choices[0].message.parsed` to:
  ```python
  content = response.choices[0].message.content
  decision = ClassifierRoutingDecision.model_validate_json(content)
  ```

---

### 4. ✅ Connection Pooling & Performance (Task #4)

**Status:** EXCELLENT - Built-in connection pooling

#### Performance Test Results

**LiteLLM (5 concurrent calls):**
- Total time: 1.594s
- Efficiency ratio: **0.99** (excellent!)

**AsyncOpenAI (5 concurrent calls):**
- Total time: 1.703s
- Efficiency ratio: **1.00** (excellent!)

**Conclusion:** LiteLLM has **built-in connection pooling** that performs on par with AsyncOpenAI.

#### Internal HTTP Client Management

LiteLLM uses:
- `litellm.llms.custom_httpx.http_handler.HTTPHandler`
- Module-level clients: `module_level_aclient`, `aclient_session`
- Automatic cleanup: `close_litellm_async_clients()`, `register_async_client_cleanup()`

**No explicit client management needed!** LiteLLM handles it internally.

---

### 5. ✅ Error Handling (Task #6)

**Status:** COMPATIBLE with OpenAI SDK exceptions

#### Exception Hierarchy

LiteLLM exceptions are **compatible with OpenAI SDK**:

```python
# LiteLLM exceptions (all in litellm.exceptions)
litellm.AuthenticationError  -> status_code=401
litellm.BadRequestError      -> status_code=400
litellm.RateLimitError       -> status_code=429
litellm.Timeout              -> status_code=408
litellm.InternalServerError  -> status_code=500
litellm.APIConnectionError
litellm.ContentPolicyViolationError
litellm.ContextWindowExceededError
```

**All exceptions have:**
- `.status_code` attribute
- `.message` attribute
- `.response` attribute (httpx.Response object)

#### Current Code Compatibility

**Current code in `reasoning_agent.py:712`:**
```python
except httpx.HTTPStatusError as http_error:
    logger.error(f"OpenAI API error during reasoning: {http_error}")
```

**After migration:**
```python
except litellm.APIError as api_error:
    logger.error(f"LLM API error during reasoning: {api_error}")
    # api_error.status_code available
    # api_error.message available
```

**Required Changes:**
- Change `httpx.HTTPStatusError` → `litellm.APIError` (or specific subclasses)
- Update error messages to reference LiteLLM instead of "OpenAI API"

---

### 6. ✅ Trace Context Propagation (Task #3)

**Status:** FULLY SUPPORTED

#### Extra Headers Support

```python
# Inject W3C TraceContext headers
carrier: dict[str, str] = {}
propagate.inject(carrier)

# Pass to litellm - WORKS IDENTICALLY
response = await litellm.acompletion(
    model="gpt-4o-mini",
    messages=[...],
    extra_headers=carrier,  # ✓ Supported
)
```

**No changes needed** to trace propagation code!

---

### 7. ⚠️ Configuration Management (Task #5)

**Status:** Environment variables vs. explicit parameters

#### Current Approach (AsyncOpenAI)
```python
# config.py
llm_api_key: str = Field(alias="LITELLM_API_KEY")
llm_base_url: str = Field(alias="LITELLM_BASE_URL")

# dependencies.py
openai_client = AsyncOpenAI(
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
)
```

#### LiteLLM Approach (Two Options)

**Option A: Explicit parameters (RECOMMENDED)**
```python
response = await litellm.acompletion(
    model="gpt-4o-mini",
    messages=[...],
    api_key=settings.llm_api_key,    # Pass explicitly
    base_url=settings.llm_base_url,  # Pass explicitly
)
```

**Option B: Environment variables**
```python
# LiteLLM checks these env vars automatically:
# - OPENAI_API_KEY
# - OPENAI_API_BASE
# But we want to use LITELLM_API_KEY and LITELLM_BASE_URL

# Set up litellm module config at startup
import litellm
litellm.api_key = settings.llm_api_key
litellm.api_base = settings.llm_base_url
```

**Recommendation:** Use **Option A** (explicit parameters) for:
- Better testability (can override per-call)
- Explicit dependency on settings
- No global state modification

---

### 8. ❌ ServiceContainer Simplification (Task #8)

**Status:** Can remove AsyncOpenAI client

#### Current State
```python
class ServiceContainer:
    self.http_client: httpx.AsyncClient  # Used for MCP
    self.openai_client: AsyncOpenAI      # Used for LLM calls
    self.mcp_client                       # Used for tools
    self.prompt_manager_initialized       # Used for prompts
```

#### After Migration
```python
class ServiceContainer:
    self.http_client: httpx.AsyncClient  # Keep for MCP
    # REMOVE: self.openai_client (no longer needed!)
    self.mcp_client                       # Keep
    self.prompt_manager_initialized       # Keep
```

**Changes needed:**
- Remove `openai_client` from ServiceContainer
- Remove `get_openai_client()` dependency function
- Update `get_reasoning_agent()` to not inject `openai_client`
- Update `api/main.py` endpoint signatures

---

## Migration Plan Updates

Based on findings, here are the **CONCRETE ANSWERS** to the critical questions:

### ✅ Critical Question #1: Connection Pooling
**Answer:** LiteLLM has **built-in connection pooling** that works automatically. No configuration needed. Performance is on-par with AsyncOpenAI.

### ✅ Critical Question #2: Exceptions
**Answer:** LiteLLM raises **OpenAI-compatible exceptions** under `litellm.exceptions` module. All have `.status_code`, `.message`, and `.response` attributes. Replace `httpx.HTTPStatusError` with `litellm.APIError`.

### ✅ Critical Question #3: Configuration
**Answer:** Use **explicit parameters** (`api_key`, `base_url`) on each `acompletion()` call. Access via `settings.llm_api_key` and `settings.llm_base_url`.

### ✅ Critical Question #4: Test Mocking
**Answer:** Use **Option C** (both):
- Unit tests: Mock `litellm.acompletion` function directly with `pytest-mock`
- Integration tests: Use real LiteLLM proxy (already working in CI)

---

## Updated Migration Tasks

### Task #1: API Compatibility Verification ✅ COMPLETE
- [x] Confirmed all parameters supported
- [x] Verified response object structure identical
- [x] Documented response_format difference (no `.parsed`, need manual parsing)

### Task #2: Streaming Response Verification ✅ COMPLETE
- [x] Verified async iterator works identically
- [x] Confirmed chunk structure matches OpenAI
- [x] Tested `stream_options={"include_usage": True}`
- [x] Confirmed `model_dump()` method exists

### Task #3: Trace Context Propagation ✅ COMPLETE
- [x] Verified `extra_headers` parameter exists
- [x] Confirmed W3C TraceContext pattern works identically

### Task #4: Connection Pooling & Performance ✅ COMPLETE
- [x] Confirmed built-in connection pooling (0.99 efficiency ratio)
- [x] No explicit configuration needed
- [x] Performance on-par with AsyncOpenAI

### Task #5: Configuration Management ✅ COMPLETE
- [x] Decision: Use explicit `api_key` and `base_url` parameters
- [x] Pass `settings.llm_api_key` and `settings.llm_base_url` to each call
- [x] No global state modification

### Task #6: Error Handling Updates ✅ COMPLETE
- [x] Documented exception types (litellm.APIError and subclasses)
- [x] Confirmed all have `.status_code`, `.message`, `.response`
- [x] Migration: `httpx.HTTPStatusError` → `litellm.APIError`

### Task #7: Structured Output Support ⚠️ PARTIAL
- [x] Verified `response_format=PydanticModel` works
- [ ] **ACTION NEEDED:** Update `request_router.py` to manually parse JSON
  ```python
  # Change from:
  decision = response.choices[0].message.parsed

  # To:
  content = response.choices[0].message.content
  decision = ClassifierRoutingDecision.model_validate_json(content)
  ```

### Task #8: Code Migration 🔄 READY TO START
- [ ] Update `api/passthrough.py`
- [ ] Update `api/reasoning_agent.py`
- [ ] Update `api/request_router.py` (structured output parsing)
- [ ] Update `api/dependencies.py` (remove AsyncOpenAI)
- [ ] Update imports throughout

### Task #9: Test Updates 🔄 READY TO START
- [ ] Update unit test mocks to `litellm.acompletion`
- [ ] Update exception assertions (`httpx.HTTPStatusError` → `litellm.APIError`)
- [ ] Verify integration tests still work

### Task #10: Documentation & Cleanup 🔄 READY TO START
- [ ] Update CLAUDE.md
- [ ] Remove AsyncOpenAI references

---

## Migration Pattern Examples

### Example 1: Passthrough Path (passthrough.py)

**BEFORE:**
```python
from openai import AsyncOpenAI

stream = await openai_client.chat.completions.create(
    model=request.model,
    messages=request.messages,
    max_tokens=request.max_tokens,
    temperature=request.temperature,
    top_p=request.top_p,
    n=request.n,
    stop=request.stop,
    presence_penalty=request.presence_penalty,
    frequency_penalty=request.frequency_penalty,
    logit_bias=request.logit_bias,
    user=request.user,
    stream=True,
    stream_options={"include_usage": True},
    extra_headers=carrier,
)
```

**AFTER:**
```python
import litellm
from .config import settings

stream = await litellm.acompletion(
    model=request.model,
    messages=request.messages,
    max_tokens=request.max_tokens,
    temperature=request.temperature,
    top_p=request.top_p,
    n=request.n,
    stop=request.stop,
    presence_penalty=request.presence_penalty,
    frequency_penalty=request.frequency_penalty,
    logit_bias=request.logit_bias,
    user=request.user,
    stream=True,
    stream_options={"include_usage": True},
    extra_headers=carrier,
    # Add LiteLLM-specific config
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
)
```

### Example 2: Reasoning Agent (reasoning_agent.py)

**BEFORE:**
```python
from openai import AsyncOpenAI

class ReasoningAgent:
    def __init__(self, openai_client: AsyncOpenAI, ...):
        self.openai_client = openai_client

    async def _generate_reasoning_step(self, ...):
        response = await self.openai_client.chat.completions.create(
            model=request.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=request.temperature or DEFAULT_TEMPERATURE,
            extra_headers=carrier,
        )
```

**AFTER:**
```python
import litellm
from .config import settings

class ReasoningAgent:
    def __init__(self, tools: list[Tool], ...):
        # Remove openai_client parameter!
        self.tools = {tool.name: tool for tool in tools}

    async def _generate_reasoning_step(self, ...):
        response = await litellm.acompletion(
            model=request.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=request.temperature or DEFAULT_TEMPERATURE,
            extra_headers=carrier,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
```

### Example 3: Request Router (request_router.py)

**BEFORE:**
```python
response = await openai_client.chat.completions.parse(
    model=settings.routing_classifier_model,
    messages=[...],
    temperature=settings.routing_classifier_temperature,
    response_format=ClassifierRoutingDecision,
    extra_headers=carrier,
)

decision = response.choices[0].message.parsed  # ← Uses .parsed attribute
```

**AFTER:**
```python
response = await litellm.acompletion(
    model=settings.routing_classifier_model,
    messages=[...],
    temperature=settings.routing_classifier_temperature,
    response_format=ClassifierRoutingDecision,
    extra_headers=carrier,
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
)

# Manually parse JSON response
content = response.choices[0].message.content
decision = ClassifierRoutingDecision.model_validate_json(content)
```

### Example 4: Error Handling

**BEFORE:**
```python
import httpx

try:
    response = await openai_client.chat.completions.create(...)
except httpx.HTTPStatusError as http_error:
    logger.error(f"OpenAI API error: {http_error}")
    raise ReasoningError(
        f"OpenAI API error: {http_error.response.status_code}",
        details={"http_status": http_error.response.status_code}
    )
```

**AFTER:**
```python
import litellm

try:
    response = await litellm.acompletion(...)
except litellm.APIError as api_error:
    logger.error(f"LLM API error: {api_error}")
    raise ReasoningError(
        f"LLM API error: {api_error.status_code}",
        details={"http_status": api_error.status_code}
    )
```

---

## Risk Assessment

### ✅ LOW RISK
- Streaming response format (identical structure)
- Connection pooling (built-in, tested)
- Trace propagation (identical pattern)
- Error handling (compatible exceptions)

### ⚠️ MEDIUM RISK
- Structured outputs (requires manual parsing - small change)
- Test mocking (need to update mock targets)

### ❌ NO HIGH RISKS IDENTIFIED

---

## Recommended Migration Order

1. **Update `api/config.py`** - No changes needed (already has `llm_api_key`, `llm_base_url`)

2. **Update `api/request_router.py`** - Smallest file, test structured output parsing
   - Replace `openai_client.chat.completions.parse()` with `litellm.acompletion()`
   - Add manual JSON parsing for structured output
   - Update error handling

3. **Update `api/passthrough.py`** - Simple streaming case
   - Replace `openai_client.chat.completions.create()` with `litellm.acompletion()`
   - Update error handling
   - Update imports

4. **Update `api/reasoning_agent.py`** - Most complex, but well-tested
   - Remove `openai_client` from `__init__`
   - Replace all `self.openai_client.chat.completions.create()` calls
   - Update error handling

5. **Update `api/dependencies.py`** - Remove AsyncOpenAI client
   - Remove `openai_client` from ServiceContainer
   - Remove `get_openai_client()` function
   - Update `get_reasoning_agent()` to not inject client

6. **Update tests** - Mock `litellm.acompletion` instead of HTTP responses
   - Update `tests/unit_tests/test_reasoning_agent.py`
   - Update `tests/unit_tests/test_request_router.py`
   - Update exception assertions

7. **Run full test suite** - Verify everything works
   - `make non_integration_tests`
   - `make integration_tests`
   - `make evaluations`

8. **Update documentation**
   - Update CLAUDE.md
   - Remove AsyncOpenAI references

---

## Estimated Effort

- **Code changes:** ~2-3 hours
- **Test updates:** ~1-2 hours
- **Testing & verification:** ~1 hour
- **Documentation:** ~30 minutes

**Total:** ~5-7 hours for complete migration

---

## Conclusion

✅ **PROCEED WITH MIGRATION**

The migration from AsyncOpenAI to litellm.acompletion is:
- **Feasible:** All required functionality is supported
- **Low risk:** Response formats are identical, connection pooling is built-in
- **Straightforward:** Mostly mechanical changes with one gotcha (structured outputs)
- **Well-tested:** Research confirms compatibility

The only significant change is **manual JSON parsing for structured outputs** in `request_router.py`.

**Next step:** Begin code migration following the recommended order above.
