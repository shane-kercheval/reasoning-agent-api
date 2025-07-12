.PHONY: tests

####
# Environment
####
linting_api:
	uv run ruff check api --fix --unsafe-fixes

# MCP server linting - commented out temporarily
# linting_mcp_server:
# 	uv run ruff check mcp_server --fix --unsafe-fixes

linting_tests:
	uv run ruff check tests --fix --unsafe-fixes

linting:
	# Run all linters on source and tests (mcp_server excluded temporarily)
	uv run ruff check api tests --fix --unsafe-fixes

unittests:
	uv run pytest --durations=0 --durations-min=0.1 tests

# Integration tests that require OPENAI_API_KEY
integration_tests:
	uv run pytest -m integration tests/test_integration.py

tests: linting unittests

####
# Development Servers
####
api:
	# Run the main API server
	uv run python -m api.main

# MCP server - commented out temporarily
# mcp-server:
# 	# Run the MCP server
# 	uv run python mcp_server/server.py

# Development setup - simplified to just API server
dev:
	# Run the API server in development mode
	@echo "Starting API server..."
	make api

install:
	# Install dependencies
	uv sync