#!/bin/bash

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== LiteLLM Virtual Keys Setup ===${NC}\n"

# Configurable via environment variables
ENV_FILE="${ENV_FILE:-.env}"
LITELLM_PORT="${LITELLM_PORT:-4000}"
KEY_PREFIX="${KEY_PREFIX:-agentic}"

# Load environment variables
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
    echo "Using env file: $ENV_FILE"
else
    echo -e "${RED}Error: $ENV_FILE file not found${NC}"
    exit 1
fi

# Check if LITELLM_MASTER_KEY is set
if [ -z "$LITELLM_MASTER_KEY" ]; then
    echo -e "${RED}Error: LITELLM_MASTER_KEY not set in $ENV_FILE${NC}"
    echo "Generate one with: python -c \"import secrets; print('sk-' + secrets.token_urlsafe(32))\""
    exit 1
fi

LITELLM_URL="http://localhost:${LITELLM_PORT}"
MAX_RETRIES=30
RETRY_INTERVAL=2

echo "Using LiteLLM URL: $LITELLM_URL"
echo "Using key prefix: $KEY_PREFIX"
echo ""

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
    local ENV_VAR_NAME=$3

    echo "Generating key: ${KEY_ALIAS}..."

    # IMPORTANT: No max_budget or rate limiting parameters
    # Omitting max_budget creates UNLIMITED keys (no spending caps)
    # No max_parallel_requests means unlimited concurrency
    # No models restriction = access to ALL models configured in LiteLLM
    # This is intentional for development/research environment
    RESPONSE=$(curl -s -X POST "${LITELLM_URL}/key/generate" \
        -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
        -H "Content-Type: application/json" \
        -d "{
            \"key_alias\": \"${KEY_ALIAS}\",
            \"metadata\": {\"environment\": \"${ENV_NAME}\"}
        }")

    # Extract key from response
    KEY=$(echo "$RESPONSE" | jq -r '.key // empty' 2>/dev/null)

    # If no key, show error and continue
    if [ -z "$KEY" ] || [ "$KEY" = "null" ]; then
        echo -e "${YELLOW}Error:${NC}"
        echo "$RESPONSE" | jq -C '.' 2>/dev/null || echo "$RESPONSE"
        echo ""
        return 1
    fi

    echo -e "${GREEN}✓ Success${NC}"
    echo "${ENV_VAR_NAME}=${KEY}"
    echo ""
}

echo -e "${YELLOW}Generating virtual keys...${NC}\n"
echo "# Copy these to your $ENV_FILE file"
echo "#"

# Don't exit on errors - continue generating all keys even if some fail
set +e

generate_key "${KEY_PREFIX}-dev" "development" "LITELLM_API_KEY"
generate_key "${KEY_PREFIX}-test" "testing" "LITELLM_TEST_KEY"
generate_key "${KEY_PREFIX}-eval" "evaluation" "LITELLM_EVAL_KEY"

set -e

echo -e "${GREEN}=== Setup Complete ===${NC}"
