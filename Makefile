.PHONY: tests api

# Help command
help:
	@echo "Available commands:"
	@echo "  make tests                   - Run linting + all tests (recommended)"
	@echo "  make unit_tests              - Run all tests (non-integration + integration)"
	@echo "  make non_integration_tests   - Run only non-integration tests (fast)"
	@echo "  make integration_tests       - Run only integration tests (needs OPENAI_API_KEY)"
	@echo "  make linting                 - Run code linting/formatting"
	@echo "  make api                     - Start the API server"

####
# Environment
####
linting_api:
	uv run ruff check api --fix --unsafe-fixes

linting_tests:
	uv run ruff check tests --fix --unsafe-fixes

linting:
	uv run ruff check api tests --fix --unsafe-fixes

# Non-integration tests only (fast for development)
non_integration_tests:
	uv run pytest --durations=0 --durations-min=0.1 -m "not integration" tests

# Integration tests only (require OPENAI_API_KEY, auto-start servers)
integration_tests:
	@echo "Running integration tests (auto-start servers)..."
	@echo "Note: Requires OPENAI_API_KEY environment variable"
	uv run pytest -m integration tests/ -v

# All tests (non-integration + integration)
unit_tests:
	@echo "Running ALL tests (non-integration + integration)..."
	@echo "Note: Integration tests require OPENAI_API_KEY environment variable"
	uv run pytest --durations=0 --durations-min=0.1 tests

# Main command - linting + all tests
tests: linting unit_tests

####
# Development Servers
####
api:
	uv run python -m api.main
