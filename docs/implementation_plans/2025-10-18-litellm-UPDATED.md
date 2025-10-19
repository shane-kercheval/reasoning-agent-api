# LiteLLM Proxy Integration - UPDATED Implementation Plan

## ⚠️ IMPORTANT: No Backwards Compatibility Required

**This project is NOT currently deployed to production.** Therefore:
- ✅ Breaking changes are acceptable and expected
- ✅ No migration paths needed for existing data
- ✅ No support for legacy configurations
- ✅ Clean slate approach - optimize for future, not past

All implementation steps assume a fresh start. If you have existing Phoenix data or configurations, they will be replaced.

---

## Overview

**Goal**: Replace direct OpenAI API calls with LiteLLM proxy to enable unified observability and component-level tracking across all LLM requests.

**Architecture Change**:
```
BEFORE: code → OpenAI API → OpenAI
AFTER:  code → virtual key → LiteLLM proxy → real OpenAI key → OpenAI
```

**Key Benefits**:
- Environment-based tracking via virtual keys (dev, test, eval, prod)
- Component-based tracking via OTEL metadata (reasoning-agent vs passthrough vs classifier)
- Centralized observability through Phoenix integration
- Unified request flow for ALL code (production and tests)

**Critical Design Decisions**:
- ✅ No backwards compatibility required (project not deployed)
- ✅ ALL code uses LiteLLM (production AND tests)
- ✅ No budget/rate limits (unlimited by default)
- ✅ Clean variable naming (no OPENAI_API_KEY confusion)
- ✅ Remove low-value edge case tests

---

## Environment Variable Strategy

**Clear Separation**:

```bash
# Real OpenAI API key - ONLY used by LiteLLM service
OPENAI_API_KEY=sk-proj-...

# Virtual key - used by ALL application code (reasoning-api, tests, etc)
LITELLM_API_KEY=sk-...

# LiteLLM admin key - used by setup scripts to generate virtual keys
LITELLM_MASTER_KEY=sk-...
```

**Flow**:
```
reasoning-api reads LLM_API_KEY → gets LITELLM_API_KEY → calls LiteLLM proxy
                                                           ↓
LiteLLM proxy reads OPENAI_API_KEY → calls OpenAI with real key
```

---

## Milestone 1: LiteLLM Service Infrastructure

### Goal

Set up LiteLLM proxy service with PostgreSQL backend, ensuring it starts cleanly and integrates with Phoenix OTEL.

### Success Criteria

- ✅ LiteLLM container starts successfully with database connection
- ✅ Health check passes: `curl http://localhost:4000/health/readiness`
- ✅ LiteLLM dashboard accessible at `http://localhost:4000`
- ✅ PostgreSQL database `postgres-litellm` runs on port 5433
- ✅ OTEL traces appear in Phoenix UI
- ✅ Can view LiteLLM logs: `docker compose logs litellm`

### Implementation

#### 1. Create `config/litellm_config.yaml`

```yaml
# Model definitions - maps model names to OpenAI backend
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      num_retries: 3
      timeout: 600  # 10 minutes for long requests

  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY
      num_retries: 3
      timeout: 600

  - model_name: o1-mini
    litellm_params:
      model: openai/o1-mini
      api_key: os.environ/OPENAI_API_KEY
      num_retries: 3
      timeout: 600

# Proxy authentication and database
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
  store_model_in_db: true

# OTEL integration for Phoenix observability
# Note: OTEL endpoint is configured via environment variables (see docker-compose.yml)
litellm_settings:
  callbacks: ["otel"]
  success_callback: ["otel"]
  failure_callback: ["otel"]
```

**Note**: OTEL configuration is handled via environment variables for cleaner separation of concerns.

#### 2. Update `docker-compose.yml`

**Add postgres-litellm service**:

```yaml
# PostgreSQL Database for LiteLLM
postgres-litellm:
  image: postgres:16
  container_name: litellm-postgres
  environment:
    - POSTGRES_DB=litellm
    - POSTGRES_USER=litellm_user
    - POSTGRES_PASSWORD=${LITELLM_POSTGRES_PASSWORD}
  ports:
    - "5433:5432"  # External port 5433 to avoid conflict with Phoenix postgres
  volumes:
    - litellm_postgres_data:/var/lib/postgresql/data
  networks:
    - reasoning-network
  restart: unless-stopped
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U litellm_user -d litellm"]
    interval: 10s
    timeout: 5s
    retries: 5
    start_period: 10s
```

**Add litellm service**:

```yaml
# LiteLLM Proxy - Unified LLM Gateway
litellm:
  image: ghcr.io/berriai/litellm:main-latest
  container_name: litellm-proxy
  ports:
    - "4000:4000"
  environment:
    # Real OpenAI API Key - ONLY LiteLLM uses this
    - OPENAI_API_KEY=${OPENAI_API_KEY}

    # Database connection
    - DATABASE_URL=postgresql://litellm_user:${LITELLM_POSTGRES_PASSWORD}@postgres-litellm:5432/litellm

    # Master key for admin operations (creating virtual keys)
    - LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY}

    # OTEL Configuration for Phoenix integration
    - OTEL_EXPORTER=otlp_grpc
    - OTEL_EXPORTER_OTLP_ENDPOINT=http://phoenix:4317
    - OTEL_EXPORTER_OTLP_PROTOCOL=grpc

    # Optional: Enable for OTEL debugging
    - OTEL_DEBUG=false
  volumes:
    - ./config/litellm_config.yaml:/app/config.yaml:ro
  command: ["--config", "/app/config.yaml", "--port", "4000", "--num_workers", "1"]
  depends_on:
    postgres-litellm:
      condition: service_healthy
    phoenix:
      condition: service_healthy
  networks:
    - reasoning-network
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:4000/health/readiness"]
    interval: 30s
    timeout: 10s
    retries: 5
    start_period: 30s
```

**Rename existing postgres service to postgres-phoenix**:

```yaml
# PostgreSQL Database for Phoenix
postgres-phoenix:  # Renamed from 'postgres'
  image: postgres:16
  container_name: phoenix-postgres
  environment:
    - POSTGRES_DB=${PHOENIX_POSTGRES_DB:-phoenix}
    - POSTGRES_USER=${POSTGRES_USER:-phoenix_user}
    - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
  ports:
    - "5432:5432"
  volumes:
    - phoenix_postgres_data:/var/lib/postgresql/data  # Keep same volume name
  networks:
    - reasoning-network
  restart: unless-stopped
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-phoenix_user} -d ${PHOENIX_POSTGRES_DB:-phoenix}"]
    interval: 10s
    timeout: 5s
    retries: 5
    start_period: 10s
```

**Update phoenix service dependency**:

```yaml
phoenix:
  # ... existing config ...
  environment:
    - PHOENIX_SQL_DATABASE_URL=postgresql://phoenix_user:${POSTGRES_PASSWORD}@postgres-phoenix:5432/phoenix
  depends_on:
    postgres-phoenix:  # Changed from 'postgres'
      condition: service_healthy
```

**Add new volume**:

```yaml
volumes:
  phoenix_postgres_data:  # Keep existing name for data preservation
    driver: local
  litellm_postgres_data:  # New volume for LiteLLM
    driver: local
```

#### 3. Update `.env.dev.example`

Add new LiteLLM section at the top:

```bash
# =============================================================================
# LITELLM PROXY CONFIGURATION
# =============================================================================

# LiteLLM Master Key (for admin operations like creating virtual keys)
# Used by: scripts/setup_litellm_keys.sh, LiteLLM proxy admin API
# Generate: python -c "import secrets; print('sk-' + secrets.token_urlsafe(32))"
LITELLM_MASTER_KEY=

# LiteLLM PostgreSQL Password
# Used by: postgres-litellm service, LiteLLM database connection
# Generate: python -c "import secrets; print(secrets.token_urlsafe(16))"
LITELLM_POSTGRES_PASSWORD=

# LiteLLM Virtual Keys (generated by setup script - LEAVE EMPTY initially)
# After running `make litellm_setup`, copy generated keys here

# LITELLM_API_KEY: Virtual key for all application code (dev, tests, etc)
# This is what your code uses to call LiteLLM
LITELLM_API_KEY=

# LITELLM_EVAL_KEY: Virtual key for LLM evaluations (separate tracking)
LITELLM_EVAL_KEY=

# =============================================================================
# SHARED CONFIGURATION
# =============================================================================

# CRITICAL BEHAVIOR NOTE:
# - OPENAI_API_KEY is ONLY used by LiteLLM proxy to call OpenAI
# - Application code (reasoning-api, tests) uses LITELLM_API_KEY (virtual key)
#
# Flow: reasoning-api → LITELLM_API_KEY → litellm proxy → OPENAI_API_KEY → OpenAI

# OpenAI API Configuration
# Required: Your REAL OpenAI API key (used ONLY by LiteLLM proxy)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your-openai-api-key-here

# ... rest of existing config
```

Update workflow section:

```bash
# =============================================================================
# DEVELOPMENT WORKFLOW
# =============================================================================

# 1. Copy this file: cp .env.dev.example .env
# 2. Add your OpenAI API key to OPENAI_API_KEY
# 3. Generate LiteLLM master key: python -c "import secrets; print('sk-' + secrets.token_urlsafe(32))"
# 4. Set LITELLM_MASTER_KEY and LITELLM_POSTGRES_PASSWORD
# 5. Start services: docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
# 6. Generate virtual keys: make litellm_setup
# 7. Copy generated keys to LITELLM_API_KEY and LITELLM_EVAL_KEY in .env
# 8. Restart reasoning-api: docker compose restart reasoning-api
# 9. Access web interface: http://localhost:8080
# 10. Access API docs: http://localhost:8000/docs
# 11. Access Phoenix UI: http://localhost:6006
# 12. Access LiteLLM dashboard: http://localhost:4000
```

### Testing Strategy

Manual verification (no automated tests for this milestone):

```bash
# 1. Start services
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# 2. Check all services healthy
docker compose ps

# 3. Verify LiteLLM logs show successful startup
docker compose logs litellm | grep -i "connected to database"

# 4. Check database connections
docker compose exec postgres-phoenix pg_isready
docker compose exec postgres-litellm pg_isready

# 5. Access LiteLLM dashboard
open http://localhost:4000

# 6. Verify Phoenix still works
open http://localhost:6006

# 7. Check health endpoint
curl http://localhost:4000/health/readiness
```

### Dependencies

None - this is the foundation.

### Risk Factors

1. **OTEL endpoint configuration** - Verify traces appear in Phoenix UI
2. **Database schema initialization** - LiteLLM auto-creates schema on first startup
3. **Port conflicts** - Developer may have services on 4000 or 5433
4. **Volume naming** - Ensure Phoenix data is preserved after postgres rename

---

## Milestone 2: Virtual Key Generation Script

### Goal

Create automated script to generate virtual keys for different use cases (dev, eval) with proper health checks and error handling.

### Success Criteria

- ✅ Script waits for LiteLLM to be ready before creating keys
- ✅ Creates 2 virtual keys with unlimited access (no budgets/limits)
- ✅ Outputs keys in copy-pasteable format for .env
- ✅ Script is executable: `chmod +x scripts/setup_litellm_keys.sh`
- ✅ `make litellm_setup` successfully generates keys
- ✅ Keys are visible in LiteLLM dashboard

### Implementation

#### 1. Create `scripts/setup_litellm_keys.sh`

```bash
#!/bin/bash

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== LiteLLM Virtual Keys Setup ===${NC}\n"

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo -e "${RED}Error: .env file not found${NC}"
    echo "Please create .env from .env.dev.example"
    exit 1
fi

# Check if LITELLM_MASTER_KEY is set
if [ -z "$LITELLM_MASTER_KEY" ]; then
    echo -e "${RED}Error: LITELLM_MASTER_KEY not set in .env${NC}"
    echo "Generate one with: python -c \"import secrets; print('sk-' + secrets.token_urlsafe(32))\""
    exit 1
fi

LITELLM_URL="http://localhost:4000"
MAX_RETRIES=30
RETRY_INTERVAL=2

# Wait for LiteLLM to be ready
echo "Waiting for LiteLLM to be ready..."
for i in $(seq 1 $MAX_RETRIES); do
    if curl -sf "${LITELLM_URL}/health/readiness" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ LiteLLM is ready${NC}\n"
        break
    fi

    if [ $i -eq $MAX_RETRIES ]; then
        echo -e "${RED}Error: LiteLLM did not become ready after ${MAX_RETRIES} attempts${NC}"
        echo "Check logs with: docker compose logs litellm"
        exit 1
    fi

    echo "Attempt $i/$MAX_RETRIES - waiting ${RETRY_INTERVAL}s..."
    sleep $RETRY_INTERVAL
done

# Function to generate a virtual key
generate_key() {
    local KEY_ALIAS=$1
    local ENV_NAME=$2

    echo "Generating key: ${KEY_ALIAS}..."

    # IMPORTANT: No max_budget or rate limiting parameters
    # Omitting max_budget creates UNLIMITED keys (no spending caps)
    # No max_parallel_requests means unlimited concurrency
    # This is intentional for development/research environment
    RESPONSE=$(curl -s -X POST "${LITELLM_URL}/key/generate" \
        -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
        -H "Content-Type: application/json" \
        -d "{
            \"key_alias\": \"${KEY_ALIAS}\",
            \"models\": [\"gpt-4o\", \"gpt-4o-mini\", \"o1-mini\"],
            \"metadata\": {\"environment\": \"${ENV_NAME}\"}
        }")

    # Extract key from response (assumes {"key": "sk-..."} format)
    KEY=$(echo $RESPONSE | grep -o '"key":"[^"]*' | grep -o 'sk-[^"]*' || echo "")

    if [ -z "$KEY" ]; then
        echo -e "${RED}Error generating ${KEY_ALIAS}${NC}"
        echo "Response: $RESPONSE"
        return 1
    fi

    echo -e "${GREEN}✓ Generated ${KEY_ALIAS}${NC}"
    echo "$KEY"
}

echo -e "${YELLOW}Generating virtual keys...${NC}\n"

# Generate keys
echo "# Copy these to your .env file"
echo "#"

DEV_KEY=$(generate_key "litellm-dev" "development")
if [ $? -eq 0 ]; then
    echo "LITELLM_API_KEY=${DEV_KEY}"
fi

echo ""

EVAL_KEY=$(generate_key "litellm-eval" "evaluation")
if [ $? -eq 0 ]; then
    echo "LITELLM_EVAL_KEY=${EVAL_KEY}"
fi

echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "Next steps:"
echo "1. Copy the keys above to your .env file"
echo "2. Restart reasoning-api: docker compose restart reasoning-api"
echo "3. View keys in LiteLLM dashboard: http://localhost:4000"
```

Make executable:
```bash
chmod +x scripts/setup_litellm_keys.sh
```

#### 2. Update `Makefile`

Add LiteLLM management targets:

```makefile
####
# LiteLLM Management
####

.PHONY: litellm_setup
litellm_setup: ## Setup LiteLLM virtual keys (run after docker compose up)
	@echo "Setting up LiteLLM virtual keys..."
	@./scripts/setup_litellm_keys.sh

.PHONY: litellm_ui
litellm_ui: ## Open LiteLLM dashboard in browser
	@echo "Opening LiteLLM dashboard..."
	@open http://localhost:4000 || xdg-open http://localhost:4000 || echo "Please open http://localhost:4000 in your browser"

.PHONY: litellm_logs
litellm_logs: ## Show LiteLLM proxy logs
	@docker compose logs -f litellm

.PHONY: litellm_restart
litellm_restart: ## Restart LiteLLM service
	@docker compose restart litellm

.PHONY: litellm_reset
litellm_reset: ## Reset LiteLLM database and regenerate keys (DESTRUCTIVE)
	@echo "WARNING: This will delete all virtual keys and usage data!"
	@echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
	@sleep 5
	@docker compose stop litellm
	@docker volume rm reasoning-agent-api_litellm_postgres_data || true
	@docker compose up -d litellm
	@echo "Waiting for LiteLLM to initialize..."
	@sleep 10
	@make litellm_setup
```

Update help section to include LiteLLM targets:

```makefile
help:
	@echo "Available commands:"
	@echo ""
	# ... existing sections ...
	@echo "LiteLLM Management:"
	@echo "  make litellm_setup           - Generate virtual keys (run after docker_up)"
	@echo "  make litellm_ui              - Open LiteLLM dashboard in browser"
	@echo "  make litellm_logs            - Show LiteLLM proxy logs"
	@echo "  make litellm_restart         - Restart LiteLLM service"
	@echo "  make litellm_reset           - Reset LiteLLM database (DESTRUCTIVE)"
```

### Testing Strategy

Manual verification:

```bash
# 1. Start services
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# 2. Run setup script
make litellm_setup

# 3. Verify output shows 2 keys generated
# Expected output:
# LITELLM_API_KEY=sk-...
# LITELLM_EVAL_KEY=sk-...

# 4. Copy keys to .env file

# 5. Check LiteLLM dashboard shows 2 keys
open http://localhost:4000

# 6. Test a key with curl
curl -X POST 'http://localhost:4000/chat/completions' \
  -H 'Authorization: Bearer <LITELLM_API_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Dependencies

- Milestone 1 complete (LiteLLM service running)

### Risk Factors

1. **Master key format** - Must start with `sk-` or LiteLLM may reject
2. **API response format** - Key extraction assumes `{"key": "sk-..."}` format
3. **Network timing** - May need to increase wait time for slower systems

---

## Milestone 3: Configuration Changes

### Goal

Update application configuration to use LiteLLM proxy with clean variable naming and no backwards compatibility concerns.

### Success Criteria

- ✅ `api/config.py` has `llm_api_key` and `llm_base_url` fields
- ✅ All code references updated to use new field names
- ✅ Docker configuration uses clear variable separation
- ✅ Tests use LiteLLM (no direct OpenAI calls)
- ✅ Configuration loads successfully

### Implementation

#### 1. Update `api/config.py`

Replace OpenAI-specific fields with generic LLM fields:

```python
class Settings(BaseSettings):
    """Application settings for the Reasoning Agent API."""

    # LLM API Configuration (LiteLLM proxy)
    llm_api_key: str = Field(
        default="",
        alias="LLM_API_KEY",
        description="API key for LLM requests (virtual key from LiteLLM)",
    )
    llm_base_url: str = Field(
        default="http://litellm:4000",
        alias="LLM_BASE_URL",
        description="Base URL for LLM API (LiteLLM proxy in production)",
    )

    # ... rest of existing fields remain the same
```

**Remove these old fields**:
```python
# DELETE:
openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
reasoning_agent_base_url: str = Field(...)
```

#### 2. Update `api/dependencies.py`

**Add shared AsyncOpenAI client for connection pooling:**

```python
from openai import AsyncOpenAI

class ServiceContainer:
    """Container for application services with proper lifecycle management."""

    def __init__(self):
        self.http_client: httpx.AsyncClient | None = None
        self.openai_client: AsyncOpenAI | None = None  # ADD THIS
        self.mcp_client = None
        self.prompt_manager_initialized: bool = False

    async def initialize(self) -> None:
        """Initialize services during app startup."""
        # Create ONE http client for the entire app lifetime
        self.http_client = create_production_http_client()

        # CREATE ONE AsyncOpenAI client for entire app lifetime
        # This enables connection pooling to LiteLLM proxy
        self.openai_client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )

        # ... rest of initialization (prompt manager, MCP)

    async def cleanup(self) -> None:
        """Cleanup services during app shutdown."""
        if self.http_client:
            await self.http_client.aclose()
        if self.openai_client:
            await self.openai_client.close()  # ADD THIS
        # ... rest of cleanup
```

**Add dependency function for AsyncOpenAI client:**

```python
async def get_openai_client() -> AsyncOpenAI:
    """Get shared AsyncOpenAI client dependency."""
    if service_container.openai_client is None:
        raise RuntimeError(
            "Service container not initialized. "
            "AsyncOpenAI client should be available after app startup."
        )
    return service_container.openai_client
```

**Update `get_reasoning_agent()` to inject shared client:**

```python
async def get_reasoning_agent(
    openai_client: Annotated[AsyncOpenAI, Depends(get_openai_client)],  # ADD THIS
    tools: Annotated[list[Tool], Depends(get_tools)],
    prompt_manager: Annotated[PromptManager, Depends(get_prompt_manager)],
) -> ReasoningAgent:
    """Get reasoning agent dependency with injected dependencies."""
    return ReasoningAgent(
        openai_client=openai_client,  # Pass shared client instead of base_url/api_key
        tools=tools,
        prompt_manager=prompt_manager,
    )
```

**Modify `ReasoningAgent.__init__` to accept client instead of creating one:**

```python
# In api/reasoning_agent.py

class ReasoningAgent:
    def __init__(
        self,
        openai_client: AsyncOpenAI,  # CHANGED: Accept client instead of base_url/api_key
        tools: list[Tool],
        prompt_manager: PromptManager,
        max_reasoning_iterations: int = 20,
    ):
        """Initialize the reasoning agent."""
        self.openai_client = openai_client  # Use injected client
        self.tools = {tool.name: tool for tool in tools}
        self.prompt_manager = prompt_manager
        # ... rest of initialization

        # REMOVE these lines:
        # self.base_url = base_url.rstrip("/")
        # self.api_key = api_key
        # self.openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
```

**Why this matters**: ReasoningAgent instances are created per-request. Without this change, each request creates a new AsyncOpenAI client, defeating connection pooling.

#### 3. Update `api/passthrough.py`

**Change function signature to accept shared AsyncOpenAI client:**

```python
async def execute_passthrough_stream(
    request: OpenAIChatRequest,
    openai_client: AsyncOpenAI,  # ADD THIS - injected from dependency
    parent_span: trace.Span | None = None,
    check_disconnected: Callable[[], bool] | None = None,
) -> AsyncGenerator[str]:
    """Execute a streaming chat completion via direct OpenAI API call."""
    # ... rest of function implementation
    # REMOVE the AsyncOpenAI instantiation - use the injected client instead
```

**Update endpoint in `api/main.py` to inject client:**

```python
from api.dependencies import get_openai_client

@app.post("/v1/chat/completions")
async def chat_completions(
    request: OpenAIChatRequest,
    openai_client: Annotated[AsyncOpenAI, Depends(get_openai_client)],  # ADD THIS
    # ... other dependencies
):
    # Pass openai_client to execute_passthrough_stream
    return StreamingResponse(
        execute_passthrough_stream(request, openai_client, ...),
        media_type="text/event-stream"
    )
```

#### 4. Update `api/request_router.py`

**Change `_classify_with_llm()` to accept shared AsyncOpenAI client:**

```python
async def _classify_with_llm(
    request: OpenAIChatRequest,
    openai_client: AsyncOpenAI,  # ADD THIS - injected parameter
) -> dict[str, Any]:
    """Use LLM with structured outputs to classify routing path."""
    # ... rest of function
    # REMOVE the AsyncOpenAI instantiation - use the injected client

    # Use structured outputs with Pydantic schema
    response = await openai_client.chat.completions.parse(
        model=settings.routing_classifier_model,
        messages=[...],
        temperature=settings.routing_classifier_temperature,
        response_format=ClassifierRoutingDecision,
    )
```

**Update `determine_routing()` to accept and pass client:**

```python
async def determine_routing(
    request: OpenAIChatRequest,
    headers: dict[str, str] | None = None,
    openai_client: AsyncOpenAI | None = None,  # ADD THIS - optional for backwards compat during migration
) -> RoutingDecision:
    """Determine routing path for a chat completion request."""
    # ... tier 1 and 2 logic ...

    if routing_mode_header == "auto":
        # Use LLM classifier - pass the client
        decision = await _classify_with_llm(request, openai_client)
        return RoutingDecision(...)
```

**Update endpoint in `api/main.py` to inject client:**

```python
@app.post("/v1/chat/completions")
async def chat_completions(
    request: OpenAIChatRequest,
    openai_client: Annotated[AsyncOpenAI, Depends(get_openai_client)],
    # ... other dependencies
):
    # Pass openai_client to determine_routing
    routing_decision = await determine_routing(request, headers, openai_client)
```

#### 5. Update `docker-compose.yml`

Update reasoning-api service environment:

```yaml
reasoning-api:
  # ... existing config ...
  environment:
    # LLM API Configuration (via LiteLLM proxy)
    - LLM_API_KEY=${LITELLM_API_KEY}
    - LLM_BASE_URL=http://litellm:4000

    # Remove these old variables:
    # - OPENAI_API_KEY=${OPENAI_API_KEY}
    # - REASONING_AGENT_BASE_URL=${REASONING_AGENT_BASE_URL:-https://api.openai.com/v1}

    # ... rest of existing environment variables ...

  # Add litellm to dependencies with health check
  depends_on:
    fake-mcp-server:
      condition: service_started
    litellm:
      condition: service_healthy  # Wait for LiteLLM to be ready
```

#### 6. Update `docker-compose.dev.yml`

Keep development overrides minimal (auth disabled, debug logging):

```yaml
reasoning-api:
  # ... existing volume mounts ...
  environment:
    # ... existing dev settings ...

    # LLM configuration is already set in base docker-compose.yml
    # No need to override here - virtual key comes from .env

    # Remove any OPENAI_API_KEY overrides if present
```

#### 7. Update `.env.dev.example`

Update the shared configuration section:

```bash
# =============================================================================
# SHARED CONFIGURATION
# =============================================================================

# CRITICAL BEHAVIOR NOTE:
# - OPENAI_API_KEY is ONLY used by LiteLLM proxy to call OpenAI
# - Application code uses LiteLLM virtual key (LITELLM_API_KEY)
# - Tests also use LiteLLM (no direct OpenAI calls)
#
# Request Flow:
# reasoning-api → LLM_API_KEY (virtual) → litellm proxy → OPENAI_API_KEY (real) → OpenAI

# OpenAI API Configuration
# Required: Your REAL OpenAI API key (used ONLY by LiteLLM proxy)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your-openai-api-key-here

# Remove old REASONING_AGENT_BASE_URL section if present
```

#### 8. Update `.env.prod.example`

Add complete LiteLLM section (similar to dev):

```bash
# =============================================================================
# LITELLM PROXY CONFIGURATION
# =============================================================================

# LiteLLM Master Key (for admin operations)
# CRITICAL: Generate a secure key for production!
# Generate: python -c "import secrets; print('sk-' + secrets.token_urlsafe(32))"
LITELLM_MASTER_KEY=

# LiteLLM PostgreSQL Password
# CRITICAL: Use a strong password for production!
LITELLM_POSTGRES_PASSWORD=

# LiteLLM Production Virtual Key
# Generate this key in production by running: make litellm_setup
LITELLM_API_KEY=

# =============================================================================
# SHARED CONFIGURATION
# =============================================================================

# OpenAI API Configuration
# Required: Your REAL OpenAI API key (used ONLY by LiteLLM proxy to call OpenAI)
# Get from: https://platform.openai.com/api-keys
# IMPORTANT: Application code uses LITELLM_API_KEY, not this key
OPENAI_API_KEY=

# ... rest of config
```

#### 9. Update test fixtures in `tests/conftest.py`

Update fixtures to use shared AsyncOpenAI client with LiteLLM:

```python
import os
from openai import AsyncOpenAI
from api.config import settings

# Find all ReasoningAgent instantiations and update them

# OLD (conftest.py line 262):
yield ReasoningAgent(
    base_url="https://api.openai.com/v1",  # Hardcoded - bypasses LiteLLM!
    api_key="test-api-key",
    tools=tools,
    prompt_manager=mock_prompt_manager,
)

# NEW:
# Create shared AsyncOpenAI client for tests
test_openai_client = AsyncOpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:4000"),
    api_key=os.getenv("LLM_API_KEY", "test-key"),
)

yield ReasoningAgent(
    openai_client=test_openai_client,  # Pass client instead of base_url/api_key
    tools=tools,
    prompt_manager=mock_prompt_manager,
)
```

**Important**: Tests now use LiteLLM by default. To run tests:
```bash
# Ensure LiteLLM is running and LITELLM_API_KEY is set in .env
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
make integration_tests
```

### Testing Strategy

#### Unit Tests

Update all tests that reference old config fields:

```bash
# Find all references to update
grep -r "openai_api_key" tests/
grep -r "reasoning_agent_base_url" tests/
grep -r "OPENAI_API_KEY" tests/
grep -r "REASONING_AGENT_BASE_URL" tests/

# Update all occurrences to use:
# - llm_api_key / LLM_API_KEY
# - llm_base_url / LLM_BASE_URL
```

Example test update:

```python
# tests/unit_tests/test_config.py
def test_config_defaults():
    settings = Settings()
    assert settings.llm_base_url == "http://litellm:4000"  # New default
    assert settings.llm_api_key == ""  # Empty until configured
```

#### Integration Tests

Integration tests now require LiteLLM to be running:

```bash
# 1. Start services (including LiteLLM)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# 2. Ensure LITELLM_API_KEY is set in .env

# 3. Run integration tests
make integration_tests
```

**Note**: Tests now verify the full production path (through LiteLLM proxy).

### Dependencies

- Milestone 1 complete (LiteLLM service available)
- Milestone 2 complete (virtual keys generated)

### Risk Factors

1. **Breaking changes** - All existing deployments need .env updates (acceptable - not deployed yet)
2. **Test failures** - Tests may need LiteLLM running (document this clearly)
3. **Missing virtual key** - Tests will fail if LITELLM_API_KEY not set (expected behavior)

---

## Milestone 4: Component Tracking via OTEL Metadata

### Goal

Add component-level tracking to LLM calls so Phoenix can distinguish between reasoning-agent, passthrough, and classifier.

### Success Criteria

- ✅ All LLM calls include metadata with component name
- ✅ Phoenix traces show component information
- ✅ Can filter Phoenix traces by component
- ✅ Metadata appears in LiteLLM logs

### Implementation

#### 1. Update `api/reasoning_agent.py`

Add metadata to all LLM calls using `extra_body` parameter:

```python
# In _generate_reasoning_step() method (around line 659)
response = await self.openai_client.chat.completions.create(
    model=request.model,
    messages=messages,
    response_format={"type": "json_object"},
    temperature=request.temperature or DEFAULT_TEMPERATURE,
    # Add component metadata
    extra_body={
        "metadata": {
            "component": "reasoning-agent",
            "operation": "step_generation",
        }
    },
)

# In _stream_final_synthesis() method (around line 524)
stream = await self.openai_client.chat.completions.create(
    model=request.model,
    messages=messages,
    stream=True,
    temperature=request.temperature or DEFAULT_TEMPERATURE,
    stream_options={"include_usage": True},
    # Add component metadata
    extra_body={
        "metadata": {
            "component": "reasoning-agent",
            "operation": "synthesis",
        }
    },
)
```

#### 2. Update `api/passthrough.py`

Add metadata to passthrough requests:

```python
# In execute_passthrough_stream() function (around line 87)
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
    # Add component metadata
    extra_body={
        "metadata": {
            "component": "passthrough",
            "routing_path": "passthrough",
        }
    },
)
```

#### 3. Update `api/request_router.py`

Add metadata to classifier LLM calls:

```python
# In _classify_with_llm() function (around line 285)
response = await openai_client.chat.completions.parse(
    model=settings.routing_classifier_model,
    messages=[
        {"role": "system", "content": classification_prompt},
        {"role": "user", "content": f"Query to classify: {last_user_message}"},
    ],
    temperature=settings.routing_classifier_temperature,
    response_format=ClassifierRoutingDecision,
    # Add component metadata
    extra_body={
        "metadata": {
            "component": "routing-classifier",
            "operation": "auto_routing",
        }
    },
)
```

#### 4. Verify OTEL Configuration

Ensure `config/litellm_config.yaml` has OTEL callbacks enabled (should already be set from Milestone 1):

```yaml
litellm_settings:
  callbacks: ["otel"]
  success_callback: ["otel"]
  failure_callback: ["otel"]
```

#### 5. CRITICAL: Enable OTEL Trace Context Propagation

**Problem**: Your application creates Phoenix OTEL spans, and LiteLLM creates separate OTEL spans. Without trace context propagation, these will appear as disconnected trace trees instead of unified parent-child traces.

**Solution**: Propagate trace context from your application spans to LiteLLM proxy calls by injecting the `traceparent` header in each LLM request.

**Implementation - Add propagation in each LLM call:**

Since trace context changes per request, inject it at call time:

```python
# In api/passthrough.py
from opentelemetry import propagate  # Use global propagator (simpler)

async def execute_passthrough_stream(...):
    # Inject current trace context into headers
    trace_headers = {}
    propagate.inject(trace_headers)  # Injects 'traceparent' header automatically

    # Make request with trace propagation
    stream = await openai_client.chat.completions.create(
        model=request.model,
        messages=request.messages,
        # ... other params ...
        extra_headers=trace_headers,  # Propagate trace context to LiteLLM
        extra_body={
            "metadata": {
                "component": "passthrough",
                "routing_path": "passthrough",
            }
        },
    )
```

**Note**: `propagate.inject()` uses the global propagator (W3C TraceContext by default) and is the recommended approach per OpenTelemetry Python docs. This is simpler than manually creating a `TraceContextTextMapPropagator()` instance.

**Apply the same pattern to:**
- `api/reasoning_agent.py` - Both `_generate_reasoning_step()` and `_stream_final_synthesis()`
- `api/request_router.py` - `_classify_with_llm()`

**Result**: LiteLLM spans will appear as children of your application spans in Phoenix, creating a unified trace tree.

### Testing Strategy

Manual verification:

```bash
# 1. Start services with LiteLLM
make docker_up

# 2. Make a request through reasoning path
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Routing-Mode: reasoning" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'

# 3. Check Phoenix UI (http://localhost:6006)
# - Find the trace for this request
# - Verify metadata includes "component": "reasoning-agent"
# - Verify there are spans for both step generation and synthesis

# 4. Make a request through passthrough
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Routing-Mode: passthrough" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# 5. Verify metadata includes "component": "passthrough"

# 6. Make an auto-routing request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Routing-Mode: auto" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Research quantum computing"}]
  }'

# 7. Verify classifier call has "component": "routing-classifier"

# 8. Check LiteLLM logs for OTEL activity
make litellm_logs | grep -i otel
```

### Dependencies

- Milestone 1 complete (LiteLLM with OTEL configured)
- Milestone 3 complete (all code using LiteLLM)

### Risk Factors

1. **Metadata forwarding** - Verify LiteLLM actually forwards extra_body.metadata to OTEL
2. **Streaming compatibility** - Ensure metadata works with streaming responses
3. **Phoenix display** - Verify Phoenix UI properly displays custom metadata

---

## Milestone 5: Documentation and Production Configuration

### Goal

Create production-ready configuration examples and comprehensive documentation.

### Success Criteria

- ✅ `.env.prod.example` has complete LiteLLM configuration
- ✅ `CLAUDE.md` documents LiteLLM architecture
- ✅ All Makefile targets documented
- ✅ Clear setup instructions for new developers

### Implementation

#### 1. Update `CLAUDE.md`

Add LiteLLM architecture section after "Architecture Overview":

```markdown
### LiteLLM Proxy Integration

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│                      LLM Request Flow                            │
└─────────────────────────────────────────────────────────────────┘

Application Code → Virtual Key → LiteLLM Proxy → Real OpenAI Key → OpenAI API

Components:
- reasoning-agent: Uses virtual key for reasoning LLM calls
- passthrough: Uses virtual key for direct OpenAI forwarding
- request-router: Uses virtual key for routing classification
- integration tests: Use virtual key (full production path)
```

**Key Concepts:**

- **Virtual Keys:** Environment-specific API keys (dev, eval, prod) tracked by LiteLLM
- **Real Key:** Single OpenAI API key used by LiteLLM proxy (never used by application code)
- **Component Metadata:** OTEL metadata tags LLM calls by component (reasoning vs passthrough vs classifier)
- **Unified Observability:** All LLM requests (including tests) traced in Phoenix with usage data

**Configuration:**

- `LLM_BASE_URL`: LiteLLM proxy endpoint (default: `http://litellm:4000`)
- `LLM_API_KEY`: Virtual key for application code (generated by `make litellm_setup`)
- `OPENAI_API_KEY`: Real OpenAI key (ONLY used by LiteLLM proxy)

**Local Development:**

```bash
# 1. Start services (includes LiteLLM)
make docker_up

# 2. Generate virtual keys
make litellm_setup

# 3. Copy keys to .env
# LITELLM_API_KEY=sk-...
# LITELLM_EVAL_KEY=sk-...

# 4. Restart reasoning-api to load new keys
docker compose restart reasoning-api

# 5. Access dashboards
make litellm_ui  # LiteLLM dashboard
# Phoenix: http://localhost:6006
```

**Monitoring:**

- **LiteLLM Dashboard** (http://localhost:4000): Virtual keys, usage metrics
- **Phoenix Dashboard** (http://localhost:6006): Request traces, latency, errors

**Testing:**

All tests (unit, integration, evaluations) use LiteLLM proxy for consistency with production.

```bash
# Ensure LiteLLM is running
make docker_up

# Run integration tests (uses LiteLLM)
make integration_tests

# Run evaluations (uses LiteLLM)
make evaluations
```
```

#### 2. Update `README.md`

Add LiteLLM quick start information (if README has setup instructions):

```markdown
## Quick Start

1. **Clone and configure**:
   ```bash
   git clone <repo-url>
   cd reasoning-agent-api
   cp .env.dev.example .env
   # Edit .env: add your OPENAI_API_KEY
   ```

2. **Generate LiteLLM credentials**:
   ```bash
   # Generate master key and add to .env
   python -c "import secrets; print('sk-' + secrets.token_urlsafe(32))"
   # Set LITELLM_MASTER_KEY and LITELLM_POSTGRES_PASSWORD in .env
   ```

3. **Start services**:
   ```bash
   make docker_up
   ```

4. **Setup LiteLLM virtual keys**:
   ```bash
   make litellm_setup
   # Copy generated keys to .env LITELLM_API_KEY and LITELLM_EVAL_KEY
   docker compose restart reasoning-api
   ```

5. **Access services**:
   - Web UI: http://localhost:8080
   - API Docs: http://localhost:8000/docs
   - Phoenix: http://localhost:6006
   - LiteLLM: http://localhost:4000
```

#### 3. Add production deployment notes

Create `docs/deployment/litellm-production.md`:

```markdown
# LiteLLM Production Deployment

## Architecture

```
Production Flow:
reasoning-api → LLM_API_KEY (virtual) → litellm proxy → OPENAI_API_KEY (real) → OpenAI
```

## Required Environment Variables

### LiteLLM Service

```bash
# Real OpenAI key (only LiteLLM uses this)
OPENAI_API_KEY=sk-proj-...

# LiteLLM admin credentials
LITELLM_MASTER_KEY=<secure-random-key>
LITELLM_POSTGRES_PASSWORD=<secure-password>

# OTEL endpoint
OTEL_EXPORTER=otlp_grpc
OTEL_EXPORTER_OTLP_ENDPOINT=http://phoenix:4317
OTEL_EXPORTER_OTLP_PROTOCOL=grpc
```

### Application Service

```bash
# Virtual key (generated via make litellm_setup)
LLM_API_KEY=<virtual-key-from-litellm>
LLM_BASE_URL=http://litellm:4000
```

## Setup Steps

1. Deploy PostgreSQL databases (Phoenix + LiteLLM)
2. Deploy Phoenix service
3. Deploy LiteLLM proxy with config
4. Generate production virtual key: `make litellm_setup`
5. Configure reasoning-api with virtual key
6. Deploy reasoning-api

## Security Considerations

- **Never commit real keys** to version control
- **Rotate virtual keys** periodically
- **Monitor usage** in LiteLLM dashboard
- **Restrict LITELLM_MASTER_KEY** access (only for key generation)
- **Use separate virtual keys** per environment (dev/staging/prod)

## Monitoring

- **LiteLLM Dashboard**: http://your-domain:4000 - Usage metrics, key management
- **Phoenix Dashboard**: http://your-domain:6006 - Traces, performance analysis

## Troubleshooting

### LiteLLM won't start
- Check PostgreSQL connection: `docker compose exec postgres-litellm pg_isready`
- Check logs: `docker compose logs litellm`
- Verify DATABASE_URL format

### Traces not appearing in Phoenix
- Check OTEL environment variables in LiteLLM service
- Verify Phoenix is healthy: `curl http://phoenix:6006`
- Enable OTEL_DEBUG=true in LiteLLM

### Virtual keys not working
- Verify key was created: check LiteLLM dashboard
- Check key is set in reasoning-api environment
- Verify LiteLLM proxy is reachable from reasoning-api
```

### Testing Strategy

Documentation verification:

```bash
# 1. Follow setup instructions from scratch in a new environment
# 2. Verify all Makefile commands work as documented
# 3. Ensure CLAUDE.md accurately reflects current architecture
# 4. Verify README quick start works for new developers
```

### Dependencies

- All previous milestones complete

---

## Post-Implementation Verification

After all milestones are complete:

### 1. End-to-End Test (Manual)

```bash
# Start from clean state
make docker_down
docker volume prune -f

# Set up from scratch
cp .env.dev.example .env
# Edit .env: add OPENAI_API_KEY, LITELLM_MASTER_KEY, LITELLM_POSTGRES_PASSWORD

# Start services
make docker_up

# Generate keys
make litellm_setup

# Copy keys to .env
# Edit .env: paste LITELLM_API_KEY and LITELLM_EVAL_KEY

# Restart reasoning-api
docker compose restart reasoning-api

# Test passthrough path
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Routing-Mode: passthrough" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Test passthrough"}]}'

# Test reasoning path
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Routing-Mode: reasoning" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Test reasoning"}]}'

# Verify in Phoenix (http://localhost:6006)
# - Check traces appear
# - Check metadata includes component tags
# - Check usage data is tracked

# Verify in LiteLLM Dashboard (http://localhost:4000)
# - Check virtual keys show usage
```

### 2. Integration Test Suite

```bash
# Ensure services are running
make docker_up

# Ensure LITELLM_API_KEY is set in .env

# Run all tests (now using LiteLLM)
make integration_tests
```

### 3. Observability Verification

- ✅ Phoenix shows traces from all components (reasoning, passthrough, classifier)
- ✅ Metadata includes correct component tags
- ✅ LiteLLM dashboard shows usage per virtual key
- ✅ OTEL traces include LiteLLM spans

### 4. Performance Check

Verify that LiteLLM proxy doesn't introduce unacceptable latency:

```bash
# Test passthrough latency (should be similar to direct OpenAI + small proxy overhead)
time curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Routing-Mode: passthrough" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hello"}]}'

# Test reasoning path latency
time curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Routing-Mode: reasoning" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "What is 2+2?"}]}'

# Check LiteLLM resource usage
docker stats litellm-proxy

# Verify connection pooling is working
docker compose logs litellm | grep -i "connection"
```

**Acceptance Criteria**:
- ✅ Passthrough latency ≈ direct OpenAI + ~50-100ms proxy overhead
- ✅ Reasoning path latency is acceptable for use case
- ✅ No connection pool exhaustion under normal load
- ✅ LiteLLM proxy doesn't become bottleneck (CPU/memory usage reasonable)

---

## Success Metrics

Implementation is complete when:

- ✅ All LLM requests route through LiteLLM proxy
- ✅ Virtual keys provide environment-based tracking (dev/eval/prod)
- ✅ Component metadata enables filtering by component
- ✅ Phoenix traces include complete request flow with LiteLLM spans
- ✅ Integration tests pass using LiteLLM
- ✅ Documentation is complete and accurate
- ✅ Setup process is reproducible from clean state

---

## Critical Implementation Notes

### 1. ✅ Use AsyncOpenAI with LiteLLM Proxy (Correct Approach)

**Confirmed from docs**: LiteLLM proxy is OpenAI-compatible. You use the standard `AsyncOpenAI` SDK and point it to the LiteLLM proxy URL. This is the correct and recommended approach.

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=LITELLM_API_KEY,  # Virtual key
    base_url="http://litellm:4000"  # LiteLLM proxy
)
```

**DO NOT use the LiteLLM SDK** - that's for direct model calls without the proxy.

### 2. 🔄 Client Instance Reuse (Performance Critical)

**Issue**: Creating new `AsyncOpenAI()` instances per request wastes connection pooling.

**Solution**: Create ONE client instance at app startup in `ServiceContainer` and reuse it via dependency injection (see Milestone 3, Section 2).

**Why this matters**: Each AsyncOpenAI instance has its own httpx connection pool. Reusing the client instance enables HTTP connection reuse to the LiteLLM proxy.

### 3. 🔗 OTEL Trace Context Propagation (Observability Critical)

**Issue**: Without trace propagation, your application spans and LiteLLM spans appear as disconnected trees in Phoenix.

**Solution**: Inject trace context headers in every LLM call using `extra_headers` (see Milestone 4, Section 5).

**Impact**: Enables unified parent-child trace visualization showing the complete request flow.

### 4. 🚫 No Rate Limiting or Budgets

**Confirmed**: Virtual keys created without `max_budget` parameter have unlimited spending caps. No `max_parallel_requests` means unlimited concurrency.

This is intentional for development/research environments. See explicit comments in Milestone 2 key generation script.

### 5. ⚡ Latency Overhead: 40-50ms

**Benchmark data**: LiteLLM proxy adds approximately 40-50ms per request under optimal configuration.

**Your timeout settings**: `HTTP_READ_TIMEOUT=60s` is more than sufficient. This timeout applies to receiving each chunk during streaming, not total duration.

### 6. 🎯 AsyncOpenAI Client Pattern Summary

**OLD (current) - Creates new client per request:**
```python
# passthrough.py, request_router.py
openai_client = AsyncOpenAI(api_key=..., base_url=...)  # ❌ Per-request instantiation
```

**NEW (optimized) - Reuses shared client:**
```python
# dependencies.py - ONE client for entire app
service_container.openai_client = AsyncOpenAI(api_key=..., base_url=...)

# passthrough.py, request_router.py - Inject via dependency
async def execute_passthrough_stream(
    openai_client: Annotated[AsyncOpenAI, Depends(get_openai_client)],  # ✅ Injected
    ...
):
```

---

## References

- LiteLLM Docker Quick Start: https://docs.litellm.ai/docs/proxy/docker_quick_start
- LiteLLM Virtual Keys: https://docs.litellm.ai/docs/proxy/virtual_keys
- LiteLLM Configuration: https://docs.litellm.ai/docs/proxy/configs
- LiteLLM Config Settings: https://docs.litellm.ai/docs/proxy/config_settings
- LiteLLM OpenTelemetry: https://docs.litellm.ai/docs/observability/opentelemetry_integration
- LiteLLM Health Checks: https://docs.litellm.ai/docs/proxy/health
- LiteLLM Benchmarks: https://docs.litellm.ai/docs/benchmarks
