#!/bin/bash
# Setup script for personal deployment
# Creates external Docker volumes that persist across `docker compose down -v`

set -e

echo "Creating external volumes for personal deployment..."

# Create external volumes (idempotent - won't error if they exist)
docker volume create personal_phoenix_postgres 2>/dev/null && echo "Created: personal_phoenix_postgres" || echo "Exists:  personal_phoenix_postgres"
docker volume create personal_litellm_postgres 2>/dev/null && echo "Created: personal_litellm_postgres" || echo "Exists:  personal_litellm_postgres"
docker volume create personal_reasoning_postgres 2>/dev/null && echo "Created: personal_reasoning_postgres" || echo "Exists:  personal_reasoning_postgres"

echo ""
echo "Setup complete! External volumes are now ready."
echo ""
echo "These volumes will NOT be deleted by 'docker compose down -v'"
echo ""
echo "Next steps:"
echo "  1. Start personal environment: make personal_up"
echo "  2. Setup LiteLLM keys:         make personal_litellm_setup"
echo "  3. Run migrations:             make personal_migrate"
echo ""
