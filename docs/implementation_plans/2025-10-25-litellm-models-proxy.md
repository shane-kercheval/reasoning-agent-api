# Implementation Plan: LiteLLM Models Proxy

## Overview

Update the `/v1/models` endpoint to proxy LiteLLM's model discovery instead of returning a hardcoded list. This enables the desktop client to dynamically discover available models.

---

## Context

**Current Implementation** (`api/main.py` lines 117-139):
```python
@app.get("/v1/models")
async def list_models(_: bool = Depends(verify_token)) -> ModelsResponse:
    return ModelsResponse(
        data=[
            ModelInfo(id="gpt-4o", created=int(time.time()), owned_by="openai"),
            ModelInfo(id="gpt-4o-mini", created=int(time.time()), owned_by="openai"),
        ],
    )
```

**Problem**:
- Hardcoded model list becomes stale when `config/litellm_config.yaml` changes
- No way for clients to discover available models dynamically
- Source of truth (LiteLLM config) and API response are disconnected

**Solution**:
- Proxy to LiteLLM's `/v1/models` endpoint (`http://litellm:4000/v1/models`)
- Forward LiteLLM's response to client
- Let LiteLLM be the single source of truth for available models

---

## Reference Documentation

**LiteLLM Model Discovery**:
- Docs: https://docs.litellm.ai/docs/proxy/model_discovery
- Endpoint: `GET http://<litellm-host>:4000/v1/models`
- Auth: Requires `Authorization: Bearer <LITELLM_API_KEY>` header
- Response: OpenAI-compatible JSON with `data` array of model objects

**Response Format**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-4o",
      "object": "model",
      "created": 1234567890,
      "owned_by": "openai"
    },
    {
      "id": "gpt-4o-mini",
      "object": "model",
      "created": 1234567890,
      "owned_by": "openai"
    }
  ]
}
```

---

## Implementation

### Goal
Replace hardcoded model list with proxy to LiteLLM's `/v1/models` endpoint.

### Success Criteria
- `GET /v1/models` returns dynamic list from LiteLLM
- Response format matches OpenAI spec (clients don't break)
- Authentication works (uses API's `LITELLM_API_KEY`)
- Error handling for LiteLLM downtime
- Integration test validates response structure

### Key Changes

**Update Endpoint** (`api/main.py`):
```python
@app.get("/v1/models")
async def list_models(
    openai_client: Annotated[AsyncOpenAI, Depends(get_openai_client)],
    _: bool = Depends(verify_token),
) -> ModelsResponse:
    """
    List available models from LiteLLM proxy.

    Proxies to LiteLLM's /v1/models endpoint to provide dynamic model discovery.
    Uses dependency-injected AsyncOpenAI client configured for LiteLLM proxy.
    """
    try:
        # Call LiteLLM proxy's /v1/models endpoint
        # AsyncOpenAI client is already configured with LITELLM_BASE_URL and LITELLM_API_KEY
        response = await openai_client.models.list()

        # Convert OpenAI SDK response to our ModelsResponse format
        # SDK returns a SyncPage[Model] object with model objects
        models_data = [
            ModelInfo(
                id=model.id,
                created=model.created,
                owned_by=model.owned_by,
            )
            for model in response.data
        ]

        return ModelsResponse(data=models_data)

    except Exception as e:
        logger.error(f"Failed to fetch models from LiteLLM: {e}")
        # Fallback to hardcoded defaults if LiteLLM unavailable
        return ModelsResponse(
            data=[
                ModelInfo(id="gpt-4o", created=int(time.time()), owned_by="openai"),
                ModelInfo(id="gpt-4o-mini", created=int(time.time()), owned_by="openai"),
            ],
        )
```

**Why this works**:
- `AsyncOpenAI` client is already injected via `Depends(get_openai_client)`
- Client is already configured with `LITELLM_BASE_URL` and `LITELLM_API_KEY` (from `api/dependencies.py`)
- OpenAI SDK's `client.models.list()` calls `GET /v1/models` on the base URL (LiteLLM proxy)
- LiteLLM proxy implements OpenAI-compatible `/v1/models` endpoint

### Testing Strategy

**Unit Test** (`tests/unit_tests/test_api.py`):
```python
@pytest.mark.asyncio
async def test_list_models_success(mock_openai_client):
    """Test /v1/models returns models from LiteLLM."""
    # Mock OpenAI client's models.list() response
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(id="gpt-4o", created=1234567890, owned_by="openai"),
        MagicMock(id="gpt-4o-mini", created=1234567890, owned_by="openai"),
    ]
    mock_openai_client.models.list = AsyncMock(return_value=mock_response)

    # Call endpoint
    response = await list_models(openai_client=mock_openai_client, _=True)

    # Verify
    assert len(response.data) == 2
    assert response.data[0].id == "gpt-4o"
    assert response.data[1].id == "gpt-4o-mini"

@pytest.mark.asyncio
async def test_list_models_fallback_on_error(mock_openai_client):
    """Test /v1/models falls back to defaults when LiteLLM unavailable."""
    # Mock client raising exception
    mock_openai_client.models.list = AsyncMock(side_effect=Exception("Connection failed"))

    # Call endpoint
    response = await list_models(openai_client=mock_openai_client, _=True)

    # Verify fallback models returned
    assert len(response.data) == 2
    assert response.data[0].id == "gpt-4o"
```

**Integration Test** (`tests/integration_tests/test_litellm_models.py`):
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_models_from_litellm(test_client, litellm_running):
    """Test /v1/models fetches real models from LiteLLM proxy."""
    response = test_client.get(
        "/v1/models",
        headers={"Authorization": f"Bearer {settings.api_tokens[0]}"},
    )

    assert response.status_code == 200
    data = response.json()

    # Verify OpenAI-compatible response format
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0

    # Verify model objects have required fields
    for model in data["data"]:
        assert "id" in model
        assert "object" in model
        assert model["object"] == "model"
        assert "created" in model
        assert "owned_by" in model

    # Verify models from litellm_config.yaml are present
    model_ids = [m["id"] for m in data["data"]]
    assert "gpt-4o" in model_ids
    assert "gpt-4o-mini" in model_ids
```

**Manual Test**:
```bash
# Start services
make docker_up

# Test endpoint
curl -X GET http://localhost:8000/v1/models \
  -H "Authorization: Bearer ${API_TOKEN}" | jq

# Expected: JSON list of models from litellm_config.yaml
```

### Dependencies
- None (uses existing AsyncOpenAI client and LiteLLM proxy)

### Risks
- **LiteLLM downtime**: If LiteLLM proxy is down, fallback to hardcoded defaults
- **Response format changes**: LiteLLM should maintain OpenAI compatibility, but test thoroughly
- **Performance**: Extra network hop (API → LiteLLM), but models list is small and rarely changes (could cache)

---

## Future Enhancements (Optional)

**Caching** (if performance becomes issue):
```python
from functools import lru_cache
from datetime import datetime, timedelta

models_cache = {"data": None, "expires": None}

@app.get("/v1/models")
async def list_models(...):
    # Check cache
    now = datetime.now()
    if models_cache["data"] and models_cache["expires"] > now:
        return models_cache["data"]

    # Fetch from LiteLLM
    response = await openai_client.models.list()
    result = ModelsResponse(...)

    # Cache for 5 minutes
    models_cache["data"] = result
    models_cache["expires"] = now + timedelta(minutes=5)

    return result
```

**Wildcard Models** (if using LiteLLM wildcards):
- Enable in `litellm_config.yaml`: `check_provider_endpoint: true`
- LiteLLM will discover models from provider endpoints (e.g., `xai/*` → actual xAI models)

---

## Validation

After implementation:
1. ✅ `/v1/models` returns models from LiteLLM
2. ✅ Response format matches OpenAI spec
3. ✅ Adding model to `litellm_config.yaml` appears in response (without code changes)
4. ✅ Desktop client can fetch and display models dynamically
5. ✅ Fallback works when LiteLLM unavailable

---

## Notes

- **Small change**: ~20 lines of code modified
- **High value**: Unblocks desktop client dynamic model selection
- **Safe**: Fallback to hardcoded defaults on error
- **Maintainable**: LiteLLM config is single source of truth
