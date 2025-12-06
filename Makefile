.PHONY: tests client

# Load .env file if it exists (for PERSONAL_BACKUP_DIR)
-include .env
export

# Help command
help:
	@echo "Available commands:"
	@echo ""
	@echo "Testing:"
	@echo "  make tests                   - Run linting + all tests (recommended)"
	@echo "  make non_integration_tests   - Run only non-integration tests (fast)"
	@echo "  make integration_tests       - Run only integration tests"
	@echo "  make evaluations             - Run LLM behavioral evaluations"
	@echo "  make linting                 - Run code linting/formatting"
	@echo ""
	@echo "Development:"
	@echo "  make dev                     - Install all dependencies for development"
	@echo ""
	@echo "Desktop Client:"
	@echo "  make client                  - Start desktop client (connects to dev: localhost:8000)"
	@echo "  make client_personal         - Start desktop client (connects to personal: localhost:18000)"
	@echo "  make client_tests            - Run desktop client tests"
	@echo "  make client_type_check       - Run TypeScript type checking"
	@echo "  make client_build            - Build desktop client for current platform"
	@echo "  make client_install          - Install client dependencies"
	@echo ""
	@echo "Docker:"
	@echo "  make docker_build            - Build all Docker services"
	@echo "  make docker_up               - Start all services with Docker Compose (dev mode)"
	@echo "  make docker_down             - Stop all Docker services"
	@echo "  make docker_logs             - View Docker service logs"
	@echo "  make docker_restart          - Restart all Docker services"
	@echo "  make docker_rebuild          - Rebuild all services with no cache and restart"
	@echo "  make docker_rebuild_service SERVICE=<name> - Rebuild single service (e.g. SERVICE=reasoning-api)"
	@echo "  make docker_clean            - Clean up Docker containers and images"
	@echo ""
	@echo "Database:"
	@echo "  make reasoning_migrate       - Run database migrations"
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
	@echo "  make kill_servers            - Kill any leftover test servers"
	@echo ""
	@echo "Personal Deployment (protected data, see README_USAGE.md):"
	@echo "  make personal_setup          - Create external volumes (run once)"
	@echo "  make personal_up             - Start personal environment"
	@echo "  make personal_down           - Stop personal environment (data preserved)"
	@echo "  make personal_logs           - View personal environment logs"
	@echo "  make personal_backup         - Backup all databases"
	@echo "  make personal_restore        - Restore database from backup"
	@echo "  make personal_litellm_setup  - Generate virtual keys for personal env"
	@echo "  make personal_migrate        - Run database migrations"

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
linting:
	uv run ruff check reasoning_api tools_api examples --fix --unsafe-fixes

# Reasoning API Tests
reasoning_unit_tests:
	uv run pytest reasoning_api/tests/unit_tests/ -v --durations=0 --durations-min=0.1

reasoning_integration_tests:
	uv run pytest reasoning_api/tests/integration_tests/ -v --timeout=300

reasoning_evaluations:
	LITELLM_BASE_URL=http://localhost:4000 uv run pytest reasoning_api/tests/evaluations/ -v

reasoning_tests: reasoning_unit_tests reasoning_integration_tests

# Tools API Tests
tools_unit_tests:
	uv run pytest tools_api/tests/unit_tests/ -v

tools_integration_tests:
	uv run pytest tools_api/tests/integration_tests/ -v

tools_tests: tools_unit_tests tools_integration_tests

# Fast unit tests (no external dependencies) - backward compatibility
non_integration_tests: reasoning_unit_tests tools_unit_tests

# Integration tests - backward compatibility
integration_tests: reasoning_integration_tests tools_integration_tests

# LLM behavioral evaluations (require: docker compose up -d litellm postgres-litellm && make litellm_setup)
evaluations: reasoning_evaluations

# Alembic/Migration Commands
reasoning_migrate:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml exec reasoning-api uv run alembic -c alembic.ini upgrade head

reasoning_migrate_create:
	@read -p "Enter migration message: " msg; \
	uv run alembic -c reasoning_api/alembic.ini revision --autogenerate -m "$$msg"

reasoning_migrate_history:
	uv run alembic -c reasoning_api/alembic.ini history

tests_only: reasoning_tests tools_tests

# Main command - linting + all tests
tests: linting reasoning_tests tools_tests

####
# Docker Commands
####

docker_build:
	@echo "Building all Docker services..."
	docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.override.yml build

docker_up:
	@echo "Starting all services with Docker Compose (dev mode with hot reload)..."
	@echo "Services will be available at:"
	@echo "  - Reasoning API: http://localhost:8000"
	@echo "  - Tools API: http://localhost:8001"
	@echo "  - LiteLLM Dashboard: http://localhost:4000"
	@echo "  - Phoenix UI: http://localhost:6006"
	@echo ""
	@echo "Hot reloading is enabled - changes to source files will auto-restart services"
	docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.override.yml up -d

docker_down:
	@echo "Stopping all Docker services..."
	docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.override.yml down

docker_logs:
	@echo "Viewing Docker service logs..."
	docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.override.yml logs -f

docker_test:
	@echo "Running tests in Docker container..."
	docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.override.yml exec reasoning-api uv run pytest --durations=0 --durations-min=0.1 -m "not integration and not evaluation" tests

docker_restart: docker_down docker_up

docker_rebuild:
	# data volumes are preserved
	@echo "Rebuilding all Docker services with no cache..."
	docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.override.yml down
	docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.override.yml build --no-cache
	docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.override.yml up -d

# Rebuild a single Docker service with no cache
# Usage: make docker_rebuild_service SERVICE=reasoning-api
docker_rebuild_service:
	@echo "Rebuilding Docker service: $(SERVICE)..."
	docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.override.yml stop $(SERVICE)
	docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.override.yml build --no-cache $(SERVICE)
	docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.override.yml up -d $(SERVICE)

docker_clean:
	# data volumes are removed
	@echo "Cleaning up Docker containers and images..."
	docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.override.yml down -v
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
	@ENV_FILE=.env LITELLM_PORT=4000 KEY_PREFIX=agentic ./scripts/setup_litellm_keys.sh

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

client_personal: ## Start desktop client connected to personal environment
	@echo "Starting Electron client (connected to personal environment)..."
	@echo "Personal API: http://localhost:18000"
	@echo ""
	@if [ ! -f .env.personal ]; then \
		echo "WARNING: .env.personal not found. Using defaults."; \
	fi
	cd client && \
		REASONING_API_URL=http://localhost:18000 \
		TOOLS_API_URL_CLIENT=http://localhost:18001 \
		npm run dev

####
# Cleanup
####
kill_servers:
	@echo "Cleaning up any leftover test servers..."
	@lsof -ti :8000 | xargs -r kill -9 2>/dev/null || true
	@lsof -ti :8080 | xargs -r kill -9 2>/dev/null || true
	@lsof -ti :8001 | xargs -r kill -9 2>/dev/null || true
	@lsof -ti :8002 | xargs -r kill -9 2>/dev/null || true
	@lsof -ti :9000 | xargs -r kill -9 2>/dev/null || true
	@lsof -ti :9080 | xargs -r kill -9 2>/dev/null || true
	@lsof -ti :9001 | xargs -r kill -9 2>/dev/null || true
	@lsof -ti :9002 | xargs -r kill -9 2>/dev/null || true

	@echo "Kill Servers complete."

####
# Personal Deployment
# Uses external volumes that survive `docker compose down -v`
# Uses different ports so dev and personal can run simultaneously
# See README_USAGE.md for full documentation
####

# Compose files for personal deployment (standalone, uses .env.personal)
# Uses same override file as dev for volume mounts (same machine, same paths)
PERSONAL_COMPOSE=--env-file .env.personal -p personal -f docker-compose.personal.yml -f docker-compose.override.yml

# Default backup directory (override in .env.personal with PERSONAL_BACKUP_DIR)
PERSONAL_BACKUP_DIR ?= ./backups/personal

# Load only PERSONAL_BACKUP_DIR from .env.personal (not all vars, to avoid overriding dev settings)
ifneq (,$(wildcard .env.personal))
    PERSONAL_BACKUP_DIR := $(shell grep '^PERSONAL_BACKUP_DIR=' .env.personal 2>/dev/null | cut -d'=' -f2)
endif

personal_setup:
	@echo "Setting up personal deployment..."
	@./scripts/setup_personal.sh

personal_up:
	@if [ ! -f .env.personal ]; then \
		echo "ERROR: .env.personal not found!"; \
		echo "Run: cp .env.personal.example .env.personal"; \
		echo "Then edit .env.personal with your secrets."; \
		exit 1; \
	fi
	@echo "Starting personal environment (with protected volumes)..."
	@echo ""
	@echo "Services will be available at:"
	@echo "  - Reasoning API:    http://localhost:18000"
	@echo "  - Tools API:        http://localhost:18001"
	@echo "  - LiteLLM Dashboard: http://localhost:14000"
	@echo "  - Phoenix UI:       http://localhost:16006"
	@echo ""
	@echo "(Dev environment uses ports 8000, 8001, 4000, 6006)"
	@echo ""
	docker compose $(PERSONAL_COMPOSE) up -d

personal_down:
	@echo "Stopping personal environment (data preserved in external volumes)..."
	docker compose $(PERSONAL_COMPOSE) down

personal_logs:
	docker compose $(PERSONAL_COMPOSE) logs -f

personal_restart: personal_down personal_up

personal_litellm_setup:
	@echo "Setting up LiteLLM virtual keys for personal environment..."
	@ENV_FILE=.env.personal LITELLM_PORT=14000 KEY_PREFIX=personal ./scripts/setup_litellm_keys.sh

personal_migrate:
	@echo "Running database migrations for personal environment..."
	docker compose $(PERSONAL_COMPOSE) exec reasoning-api uv run alembic -c alembic.ini upgrade head

personal_backup:
	@mkdir -p "$(PERSONAL_BACKUP_DIR)"
	@echo "Backing up personal databases to $(PERSONAL_BACKUP_DIR)..."
	@echo ""
	@echo "Backing up reasoning database..."
	@docker compose $(PERSONAL_COMPOSE) exec -T postgres-reasoning pg_dump -U reasoning_user reasoning | gzip > "$(PERSONAL_BACKUP_DIR)/reasoning_$$(date +%Y%m%d_%H%M%S).sql.gz"
	@echo "Backing up litellm database..."
	@docker compose $(PERSONAL_COMPOSE) exec -T postgres-litellm pg_dump -U litellm_user litellm | gzip > "$(PERSONAL_BACKUP_DIR)/litellm_$$(date +%Y%m%d_%H%M%S).sql.gz"
	@echo "Backing up phoenix database..."
	@docker compose $(PERSONAL_COMPOSE) exec -T postgres-phoenix pg_dump -U phoenix_user phoenix | gzip > "$(PERSONAL_BACKUP_DIR)/phoenix_$$(date +%Y%m%d_%H%M%S).sql.gz"
	@echo ""
	@echo "Backup complete!"
	@ls -lh "$(PERSONAL_BACKUP_DIR)"/*.sql.gz 2>/dev/null | tail -6

personal_restore:
	@echo "Available backups in $(PERSONAL_BACKUP_DIR):"
	@echo ""
	@ls -la "$(PERSONAL_BACKUP_DIR)"/*.sql.gz 2>/dev/null || (echo "No backups found." && exit 1)
	@echo ""
	@echo "To restore a specific backup, run:"
	@echo ""
	@echo "  # Reasoning database:"
	@echo "  gunzip -c $(PERSONAL_BACKUP_DIR)/reasoning_YYYYMMDD_HHMMSS.sql.gz | docker compose $(PERSONAL_COMPOSE) exec -T postgres-reasoning psql -U reasoning_user reasoning"
	@echo ""
	@echo "  # LiteLLM database:"
	@echo "  gunzip -c $(PERSONAL_BACKUP_DIR)/litellm_YYYYMMDD_HHMMSS.sql.gz | docker compose $(PERSONAL_COMPOSE) exec -T postgres-litellm psql -U litellm_user litellm"
	@echo ""
	@echo "  # Phoenix database:"
	@echo "  gunzip -c $(PERSONAL_BACKUP_DIR)/phoenix_YYYYMMDD_HHMMSS.sql.gz | docker compose $(PERSONAL_COMPOSE) exec -T postgres-phoenix psql -U phoenix_user phoenix"

personal_backup_clean:
	@echo "Cleaning backups older than 30 days..."
	@find "$(PERSONAL_BACKUP_DIR)" -name "*.sql.gz" -mtime +30 -delete 2>/dev/null || true
	@echo "Done."

personal_status:
	@echo "Personal deployment status:"
	@docker compose $(PERSONAL_COMPOSE) ps
	@echo ""
	@echo "Volume sizes:"
	@docker system df -v 2>/dev/null | grep personal_ || echo "No personal volumes found. Run: make personal_setup"
