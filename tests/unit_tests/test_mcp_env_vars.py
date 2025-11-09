"""Unit tests for MCP configuration environment variable expansion."""

import json
import os
from pathlib import Path
from unittest.mock import patch

from api.mcp import load_mcp_config, _expand_env_vars


class TestEnvVarExpansion:
    """Test environment variable expansion in MCP config."""

    def test_expand_simple_var(self) -> None:
        """Test expansion of simple ${VAR} syntax."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = _expand_env_vars("${TEST_VAR}")
            assert result == "test_value"

    def test_expand_var_with_default(self) -> None:
        """Test expansion of ${VAR:-default} syntax."""
        # Var not set - should use default
        with patch.dict(os.environ, {}, clear=True):
            result = _expand_env_vars("${MISSING_VAR:-default_value}")
            assert result == "default_value"

        # Var set - should use var value
        with patch.dict(os.environ, {"PRESENT_VAR": "actual_value"}):
            result = _expand_env_vars("${PRESENT_VAR:-default_value}")
            assert result == "actual_value"

    def test_expand_var_in_url(self) -> None:
        """Test expansion within a URL string."""
        with patch.dict(os.environ, {"MCP_HOST": "localhost", "MCP_PORT": "9000"}):
            result = _expand_env_vars("http://${MCP_HOST}:${MCP_PORT}/mcp/")
            assert result == "http://localhost:9000/mcp/"

    def test_expand_dict(self) -> None:
        """Test expansion in nested dictionaries."""
        with patch.dict(os.environ, {"API_KEY": "secret123"}):
            config = {
                "url": "${API_KEY}",
                "nested": {
                    "key": "${API_KEY}",
                },
            }
            result = _expand_env_vars(config)
            assert result["url"] == "secret123"
            assert result["nested"]["key"] == "secret123"

    def test_expand_list(self) -> None:
        """Test expansion in lists."""
        with patch.dict(os.environ, {"VAR": "value"}):
            config = ["${VAR}", "static", {"key": "${VAR}"}]
            result = _expand_env_vars(config)
            assert result == ["value", "static", {"key": "value"}]

    def test_expand_non_string_passthrough(self) -> None:
        """Test that non-string values pass through unchanged."""
        config = {
            "number": 42,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
        }
        result = _expand_env_vars(config)
        assert result == config

    def test_expand_missing_var_without_default(self) -> None:
        """Test that missing var without default becomes empty string."""
        with patch.dict(os.environ, {}, clear=True):
            result = _expand_env_vars("${MISSING_VAR}")
            assert result == ""


class TestMCPConfigLoading:
    """Test MCP config loading with environment variables."""

    def test_load_config_with_env_vars(self, tmp_path: Path) -> None:
        """Test loading config with environment variable expansion."""
        config_file = tmp_path / "config.json"
        config_data = {
            "mcpServers": {
                "bridge": {
                    "url": "${MCP_BRIDGE_URL:-http://localhost:9000/mcp/}",
                    "transport": "http",
                },
            },
        }
        config_file.write_text(json.dumps(config_data))

        # Test with env var not set - should use default
        with patch.dict(os.environ, {}, clear=True):
            config = load_mcp_config(config_file)
            assert config["mcpServers"]["bridge"]["url"] == "http://localhost:9000/mcp/"

        # Test with env var set - should use env var
        with patch.dict(os.environ, {"MCP_BRIDGE_URL": "http://custom:8080/mcp/"}):
            config = load_mcp_config(config_file)
            assert config["mcpServers"]["bridge"]["url"] == "http://custom:8080/mcp/"

    def test_load_config_docker_host(self, tmp_path: Path) -> None:
        """Test loading config with Docker host networking."""
        config_file = tmp_path / "config.json"
        config_data = {
            "mcpServers": {
                "bridge": {
                    "url": "${MCP_BRIDGE_URL:-http://localhost:9000/mcp/}",
                    "transport": "http",
                },
            },
        }
        config_file.write_text(json.dumps(config_data))

        # Simulate Docker environment with host.docker.internal
        with patch.dict(os.environ, {"MCP_BRIDGE_URL": "http://host.docker.internal:9000/mcp/"}):
            config = load_mcp_config(config_file)
            assert config["mcpServers"]["bridge"]["url"] == "http://host.docker.internal:9000/mcp/"
