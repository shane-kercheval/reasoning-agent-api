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
	@echo "  make linting                 - Run code linting/formatting"
	@echo ""
	@echo "Development:"
	@echo "  make api                     - Start the reasoning agent API server"
	@echo "  make web_client              - Start the MonsterUI web client"
	@echo "  make dev                     - Start both API and web client (requires 2 terminals)"
	@echo "  make demo_mcp_server         - Start the demo MCP server with fake tools"
	@echo "  make demo                    - Run the complete demo (requires API + MCP server)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make cleanup                 - Kill any leftover test servers"

####
# Environment
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
	cd web-client && uv run ruff check . --fix --unsafe-fixes

linting:
	uv run ruff check api tests examples mcp_servers --fix --unsafe-fixes

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

# Web client
web_client:
	@echo "Starting MonsterUI web client on port 8080..."
	@echo "Make sure the API is running on port 8000 (make api)"
	@echo "Web interface: http://localhost:8080"
	@echo "Note: For development with auto-reload, use: cd web-client && uvicorn main:app --reload --port 8080"
	cd web-client && uv run python main.py

# Start both services (requires two terminals)
dev:
	@echo "To start both services, run these commands in separate terminals:"
	@echo "  Terminal 1: make api"
	@echo "  Terminal 2: make web_client"
	@echo ""
	@echo "Or use your terminal's split pane feature"

# Demo API server with specific MCP configuration
demo_api:
	@echo "Starting reasoning agent with demo MCP configuration..."
	MCP_CONFIG_PATH=examples/configs/demo_complete.yaml uv run python -m api.main

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
# Cleanup
####
cleanup:
	@echo "Cleaning up any leftover test servers..."
	@lsof -ti :8000 | xargs -r kill -9 2>/dev/null || true
	@lsof -ti :8080 | xargs -r kill -9 2>/dev/null || true
	@echo "Cleanup complete."
