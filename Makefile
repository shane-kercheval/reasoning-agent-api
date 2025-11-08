.PHONY: tests api cleanup client

# Help command
help:
	@echo "Available commands:"
	@echo ""
	@echo "Testing:"
	@echo "  make tests                   - Run linting + all tests (recommended)"
	@echo "  make all_tests              - Run all tests (non-integration + integration)"
	@echo "  make non_integration_tests   - Run only non-integration tests (fast)"
	@echo "  make integration_tests       - Run only integration tests"
	@echo "  make evaluations             - Run LLM behavioral evaluations"
	@echo "  make linting                 - Run code linting/formatting"
	@echo ""
	@echo "Development:"
	@echo "  make dev                     - Install all dependencies for development"
	@echo "  make api                     - Start the reasoning agent API server"
	@echo "  make demo_mcp_server         - Start the demo MCP server with fake tools"
	@echo "  make demo                    - Run the complete demo (requires API + MCP server)"
	@echo ""
	@echo "Desktop Client:"
	@echo "  make client                  - Start desktop client (requires Node.js 18+)"
	@echo "  make client_tests             - Run desktop client tests"
	@echo "  make client_type_check       - Run TypeScript type checking"
	@echo "  make client_build            - Build desktop client for current platform"
	@echo "  make client_install          - Install client dependencies"
	@echo ""
	@echo "Docker:"
	@echo "  make docker_build            - Build all Docker services"
	@echo "  make docker_up               - Start all services with Docker Compose (dev mode)"
	@echo "  make docker_down             - Stop all Docker services"
	@echo "  make docker_logs             - View Docker service logs"
	@echo "  make docker_test             - Run tests in Docker container"
	@echo "  make docker_restart          - Restart all Docker services"
	@echo "  make docker_rebuild          - Rebuild all services with no cache and restart"
	@echo "  make docker_rebuild_service SERVICE=<name> - Rebuild single service (e.g. SERVICE=reasoning-api)"
	@echo "  make docker_clean            - Clean up Docker containers and images"
	@echo ""
	@echo "Phoenix Data Management:"
	@echo "  make phoenix_reset_data      - Delete all Phoenix trace data (preserves database)"
	@echo "  make phoenix_backup_data     - Backup Phoenix database to ./backups/"
	@echo "  make phoenix_restore_data    - Restore Phoenix database from backup"
	@echo ""
	@echo "LiteLLM Management:"
	@echo "  make litellm_setup           - Generate virtual keys (run after docker_up)"
	@echo "  make litellm_ui              - Open LiteLLM dashboard in browser"
	@echo "  make litellm_logs            - Show LiteLLM proxy logs"
	@echo "  make litellm_restart         - Restart LiteLLM service"
	@echo "  make litellm_reset           - Reset LiteLLM database (DESTRUCTIVE)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make cleanup                 - Kill any leftover test servers"

####
# Environment Setup
####

# Install all dependencies for local development
dev:
	@echo "Installing all dependencies for local development..."
	uv sync --all-groups

####
# Linting and Testing
####
linting_examples:
	uv run ruff check examples --fix --unsafe-fixes

linting_mcp_servers:
	uv run ruff check mcp_servers --fix --unsafe-fixes

linting_api:
	uv run ruff check api --fix --unsafe-fixes

linting_tests:
	uv run ruff check tests --fix --unsafe-fixes

linting:
	uv run ruff check api tests examples mcp_servers --fix --unsafe-fixes

# Fast unit tests (no external dependencies)
non_integration_tests:
	uv run pytest --durations=0 --durations-min=0.1 -m "not integration and not evaluation" tests

# Integration tests
integration_tests:
	uv run pytest -m "integration and not evaluation" tests/ -v --timeout=300

# All tests with coverage
all_tests:
	uv run coverage run -m pytest --durations=0 --durations-min=0.1 -m "not evaluation" tests
	uv run coverage html

# LLM behavioral evaluations (require: docker compose up -d litellm postgres-litellm && make litellm_setup)
evaluations:
	uv run pytest tests/evaluations/ -v

# Main command - linting + all tests
tests: linting all_tests

####
# Development Servers
####
# Demo API server with specific MCP configuration
demo_api:
	@echo "Starting reasoning agent with demo MCP configuration..."
	MCP_CONFIG_PATH=examples/configs/demo_complete.json uv run python -m api.main

demo_mcp_server:
	@echo "Starting demo MCP server with fake tools..."
	@echo "Server will be available at http://localhost:8001/mcp/"
	@echo "Press Ctrl+C to stop"
	uv run python mcp_servers/fake_server.py

demo:
	@echo "Running complete demo with MCP tools..."
	@echo "Make sure both servers are running:"
	@echo "  Terminal 1: make demo_api"
	@echo "  Terminal 2: make demo_mcp_server"
	@echo ""
	uv run python examples/demo_complete.py

####
# Docker Commands
####

docker_build:
	@echo "Building all Docker services..."
	docker compose -f docker-compose.yml -f docker-compose.dev.yml build

docker_up:
	@echo "Starting all services with Docker Compose (dev mode with hot reload)..."
	@echo "Services will be available at:"
	@echo "  - API: http://localhost:8000"
	@echo "  - MCP Server: http://localhost:8001"
	@echo "  - LiteLLM Dashboard: http://localhost:4000/ui/"
	@echo ""
	@echo "Hot reloading is enabled - changes to source files will auto-restart services"
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

docker_down:
	@echo "Stopping all Docker services..."
	docker compose -f docker-compose.yml -f docker-compose.dev.yml down

docker_logs:
	@echo "Viewing Docker service logs..."
	docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f

docker_test:
	@echo "Running tests in Docker container..."
	docker compose -f docker-compose.yml -f docker-compose.dev.yml exec reasoning-api uv run pytest --durations=0 --durations-min=0.1 -m "not integration and not evaluation" tests

docker_restart: docker_down docker_up

docker_rebuild:
	# data volumes are preserved
	@echo "Rebuilding all Docker services with no cache..."
	docker compose -f docker-compose.yml -f docker-compose.dev.yml down
	docker compose -f docker-compose.yml -f docker-compose.dev.yml build --no-cache
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Rebuild a single Docker service with no cache
# Usage: make docker_rebuild_service SERVICE=reasoning-api
docker_rebuild_service:
	@echo "Rebuilding Docker service: $(SERVICE)..."
	docker compose -f docker-compose.yml -f docker-compose.dev.yml stop $(SERVICE)
	docker compose -f docker-compose.yml -f docker-compose.dev.yml build --no-cache $(SERVICE)
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d $(SERVICE)

docker_clean:
	# data volumes are removed
	@echo "Cleaning up Docker containers and images..."
	docker compose -f docker-compose.yml -f docker-compose.dev.yml down -v
	docker system prune -f

####
# Phoenix Data Management
####

phoenix_reset_data:
	@echo "WARNING: This will delete all Phoenix trace data!"
	@echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
	@sleep 5
	@echo "Resetting Phoenix data..."
	@docker compose exec -T postgres psql -U phoenix_user -d phoenix -c "DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;" 2>/dev/null || echo "Database may not be running or may not have data yet"
	@echo "Phoenix data reset complete."

phoenix_backup_data:
	@echo "Creating backup directory if it doesn't exist..."
	@mkdir -p ./backups
	@echo "Backing up Phoenix database..."
	@BACKUP_FILE="./backups/phoenix_backup_$$(date +%Y%m%d_%H%M%S).sql" && \
	docker compose exec -T postgres pg_dump -U phoenix_user phoenix > $$BACKUP_FILE && \
	echo "Backup saved to: $$BACKUP_FILE"

phoenix_restore_data:
	@echo "Available backups:"
	@ls -la ./backups/phoenix_backup_*.sql 2>/dev/null || (echo "No backups found in ./backups/" && exit 1)
	@echo ""
	@echo "To restore a specific backup, run:"
	@echo "  cat ./backups/phoenix_backup_YYYYMMDD_HHMMSS.sql | docker compose exec -T postgres psql -U phoenix_user phoenix"

####
# LiteLLM Management
####

litellm_setup: ## Setup LiteLLM virtual keys (run after docker compose up)
	@echo "Setting up LiteLLM virtual keys..."
	@./scripts/setup_litellm_keys.sh

litellm_ui: ## Open LiteLLM dashboard in browser
	@echo "Opening LiteLLM dashboard..."
	@open http://localhost:4000 || xdg-open http://localhost:4000 || echo "Please open http://localhost:4000 in your browser"

litellm_logs: ## Show LiteLLM proxy logs
	@docker compose logs -f litellm

litellm_restart: ## Restart LiteLLM service
	@docker compose restart litellm

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

####
# Desktop Client
####

client: ## Start desktop client (requires Node.js 18+)
	@echo "🖥️  Starting Electron desktop client..."
	@echo "Note: Ensure backend services are running (make docker_up)"
	cd client && npm install && npm run dev

client_tests: ## Run desktop client tests
	@echo "🧪 Running desktop client tests..."
	cd client && npm test

client_type_check: ## Run TypeScript type checking
	@echo "🔍 Running TypeScript type checking..."
	cd client && npm run type-check

client_build: ## Build desktop client for current platform
	@echo "🔨 Building desktop client..."
	cd client && npm install && npm run build

client_install: ## Install client dependencies
	@echo "📦 Installing client dependencies..."
	cd client && npm install

####
# Cleanup
####
cleanup:
	@echo "Cleaning up any leftover test servers..."
	@lsof -ti :8000 | xargs -r kill -9 2>/dev/null || true
	@lsof -ti :8080 | xargs -r kill -9 2>/dev/null || true
	@lsof -ti :8001 | xargs -r kill -9 2>/dev/null || true
	@lsof -ti :8002 | xargs -r kill -9 2>/dev/null || true
	@lsof -ti :9000 | xargs -r kill -9 2>/dev/null || true
	@lsof -ti :9080 | xargs -r kill -9 2>/dev/null || true
	@lsof -ti :9001 | xargs -r kill -9 2>/dev/null || true
	@lsof -ti :9002 | xargs -r kill -9 2>/dev/null || true
	
	@echo "Cleanup complete."
