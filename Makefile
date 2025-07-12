.PHONY: tests

####
# Environment
####
linting_api:
	uv run ruff check api --fix --unsafe-fixes

linting_mcp_server:
	uv run ruff check mcp_server --fix --unsafe-fixes

linting_tests:
	uv run ruff check tests --fix --unsafe-fixes

linting:
	# Run all linters on source, examples, and tests
	uv run ruff check api mcp_server tests --fix --unsafe-fixes

unittests:
	uv run pytest --durations=0 --durations-min=0.1 tests

tests: linting unittests

####
# Development Servers
####
api:
	# Run the main API server
	uv run python -m api.main

mcp-server:
	# Run the MCP server
	uv run python mcp_server/server.py

dev:
	# Run both servers in development (requires separate terminals)
	@echo "Run 'make api' in one terminal and 'make mcp-server' in another"

install:
	# Install dependencies
	uv sync
