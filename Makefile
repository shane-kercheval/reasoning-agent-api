.PHONY: tests api cleanup

# Help command
help:
	@echo "Available commands:"
	@echo ""
	@echo "Testing:"
	@echo "  make tests                   - Run linting + all tests (recommended)"
	@echo "  make unit_tests              - Run all tests (non-integration + integration)"
	@echo "  make non_integration_tests   - Run only non-integration tests (fast)"
	@echo "  make integration_tests       - Run only integration tests (needs OPENAI_API_KEY)"
	@echo "  make evaluations             - Run LLM behavioral evaluations (needs OPENAI_API_KEY)"
	@echo "  make linting                 - Run code linting/formatting"
	@echo ""
	@echo "Development:"
	@echo "  make dev                     - Install all dependencies for development"
	@echo "  make api                     - Start the reasoning agent API server"
	@echo "  make web_client              - Start the MonsterUI web client"
	@echo "  make demo_mcp_server         - Start the demo MCP server with fake tools"
	@echo "  make demo                    - Run the complete demo (requires API + MCP server)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker_build            - Build all Docker services"
	@echo "  make docker_up               - Start all services with Docker Compose (dev mode)"
	@echo "  make docker_down             - Stop all Docker services"
	@echo "  make docker_logs             - View Docker service logs"
	@echo "  make docker_test             - Run tests in Docker container"
	@echo "  make docker_restart          - Restart all Docker services"
	@echo "  make docker_rebuild          - Rebuild all services with no cache and restart"
	@echo "  make docker_clean            - Clean up Docker containers and images"
	@echo ""
	@echo "Phoenix Data Management:"
	@echo "  make phoenix_reset_data      - Delete all Phoenix trace data (preserves database)"
	@echo "  make phoenix_backup_data     - Backup Phoenix database to ./backups/"
	@echo "  make phoenix_restore_data    - Restore Phoenix database from backup"
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

linting_web_client:
	uv run ruff check web-client --fix --unsafe-fixes

linting:
	uv run ruff check api tests examples mcp_servers web-client --fix --unsafe-fixes

# Non-integration tests only (fast for development)
non_integration_tests:
	uv run pytest --durations=0 --durations-min=0.1 -m "not integration and not evaluation" tests

# Integration tests only (require OPENAI_API_KEY, auto-start servers)
integration_tests:
	@echo "Running integration tests (auto-start servers)..."
	@echo "Note: Requires OPENAI_API_KEY environment variable"
	uv run pytest -m "integration and not evaluation" tests/ -v

# All tests (non-integration + integration)
unit_tests:
	@echo "Running ALL tests (non-integration + integration)..."
	@echo "Note: Integration tests require OPENAI_API_KEY environment variable"
	uv run pytest --durations=0 --durations-min=0.1 -m "not evaluation" tests

# LLM behavioral evaluations using flex-evals (opt-in only)
evaluations:
	@echo "Running LLM behavioral evaluations with flex-evals..."
	@echo "Note: Requires OPENAI_API_KEY environment variable"
	@echo "Note: These test real LLM behavior with statistical thresholds"
	uv run pytest -m evaluation tests/evaluations/eval_reasoning_agent.py --asyncio-mode=auto -v

# Main command - linting + all tests
tests: linting unit_tests

####
# Development Servers
####
api:
	uv run python -m api.main

# Web client
web_client:
	@echo "Starting MonsterUI web client on port 8080..."
	@echo "Make sure the API is running on port 8000 (make api)"
	@echo "Web interface: http://localhost:8080"
	@echo "Note: Web client uses root .env file for configuration"
	@echo "Note: For development with auto-reload, use: cd web-client && uvicorn main:app --reload --port 8080"
	cd web-client && uv run python main.py

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
	docker compose build

docker_up:
	@echo "Starting all services with Docker Compose (dev mode with hot reload)..."
	@echo "Services will be available at:"
	@echo "  - Web Interface: http://localhost:8080"
	@echo "  - API: http://localhost:8000"
	@echo "  - MCP Server: http://localhost:8001"
	@echo ""
	@echo "Hot reloading is enabled - changes to source files will auto-restart services"
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

docker_down:
	@echo "Stopping all Docker services..."
	docker compose down

docker_logs:
	@echo "Viewing Docker service logs..."
	docker compose logs -f

docker_test:
	@echo "Running tests in Docker container..."
	docker compose exec reasoning-api uv run pytest --durations=0 --durations-min=0.1 -m "not integration and not evaluation" tests

docker_restart: docker_down docker_up

docker_rebuild:
	@echo "Rebuilding all Docker services with no cache..."
	docker compose down
	docker compose build --no-cache
	docker compose up -d

docker_clean:
	@echo "Cleaning up Docker containers and images..."
	docker compose down -v
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
