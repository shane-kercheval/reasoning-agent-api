"""
Tests for the updated JSON-only configuration loader.

This module tests loading MCP server configuration from JSON files,
ensuring compatibility with Claude Desktop and FastMCP standards.
"""

import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from api.mcp_config_loader import MCPConfigLoader, ConfigurationError, load_mcp_config
from api.mcp import MCPServersConfig


class TestMCPConfigLoader:
    """Test the JSON configuration loader."""

    def test_load_valid_json_config(self):
        """Test loading a valid JSON configuration."""
        config_data = {
            "mcpServers": {
                "demo_tools": {
                    "url": "http://localhost:8001/mcp/",
                    "transport": "http",
                    "enabled": True,
                },
                "web_search": {
                    "url": "https://api.example.com/mcp",
                    "transport": "http",
                    "auth_env_var": "API_KEY",
                    "enabled": False,
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = MCPConfigLoader.load_from_file(temp_path)

            assert isinstance(config, MCPServersConfig)
            assert len(config.servers) == 2

            # Check first server
            demo_server = next(s for s in config.servers if s.name == "demo_tools")
            assert demo_server.url == "http://localhost:8001/mcp/"
            assert demo_server.enabled is True
            assert demo_server.auth_env_var is None

            # Check second server
            web_server = next(s for s in config.servers if s.name == "web_search")
            assert web_server.url == "https://api.example.com/mcp"
            assert web_server.enabled is False
            assert web_server.auth_env_var == "API_KEY"

        finally:
            Path(temp_path).unlink()

    def test_load_stdio_server_config(self):
        """Test loading configuration with stdio server."""
        config_data = {
            "mcpServers": {
                "local_server": {
                    "command": "python",
                    "args": ["server.py", "--verbose"],
                    "env": {"DEBUG": "true"},
                    "transport": "stdio",
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = MCPConfigLoader.load_from_file(temp_path)

            assert len(config.servers) == 1
            server = config.servers[0]
            assert server.name == "local_server"
            assert server.command == "python"
            assert server.args == ["server.py", "--verbose"]
            assert server.env == {"DEBUG": "true"}
            assert server.url is None

        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file(self):
        """Test loading from a nonexistent file raises error."""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            MCPConfigLoader.load_from_file("/nonexistent/config.json")

    def test_load_invalid_json(self):
        """Test loading invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json}')  # Invalid JSON
            temp_path = f.name

        try:
            with pytest.raises(ConfigurationError, match="Invalid JSON"):
                MCPConfigLoader.load_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_find_default_config_env_var(self):
        """Test finding config via MCP_CONFIG_PATH environment variable."""
        config_data = {"mcpServers": {}}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            with patch.dict(os.environ, {"MCP_CONFIG_PATH": temp_path}):
                found_path = MCPConfigLoader._find_default_config()
                assert found_path == temp_path
        finally:
            Path(temp_path).unlink()

    def test_find_default_config_standard_locations(self):
        """Test finding config in standard locations."""
        # Create a temporary config file in a "config" directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            config_dir.mkdir()
            config_file = config_dir / "mcp_servers.json"

            with open(config_file, 'w') as f:
                json.dump({"mcpServers": {}}, f)

            # Change to temp directory so relative path works
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                found_path = MCPConfigLoader._find_default_config()
                assert found_path == "config/mcp_servers.json"
            finally:
                os.chdir(original_cwd)

    def test_load_from_env_simple(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            "MCP_SERVERS": "server1,server2",
            "MCP_SERVER1_URL": "http://localhost:8001/mcp/",
            "MCP_SERVER1_ENABLED": "true",
            "MCP_SERVER2_URL": "https://api.example.com/mcp",
            "MCP_SERVER2_AUTH_ENV": "API_KEY",
            "MCP_SERVER2_ENABLED": "false",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = MCPConfigLoader.load_from_env()

            assert len(config.servers) == 2

            server1 = next(s for s in config.servers if s.name == "server1")
            assert server1.url == "http://localhost:8001/mcp/"
            assert server1.enabled is True

            server2 = next(s for s in config.servers if s.name == "server2")
            assert server2.url == "https://api.example.com/mcp"
            assert server2.enabled is False
            assert server2.auth_env_var == "API_KEY"

    def test_load_from_env_no_servers(self):
        """Test loading from environment with no servers defined."""
        with patch.dict(os.environ, {}, clear=True):
            config = MCPConfigLoader.load_from_env()
            assert len(config.servers) == 0

    def test_load_from_env_missing_url(self):
        """Test that servers without URLs are skipped."""
        env_vars = {
            "MCP_SERVERS": "good_server,bad_server",
            "MCP_GOOD_SERVER_URL": "http://localhost:8001/mcp/",
            # Missing MCP_BAD_SERVER_URL
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = MCPConfigLoader.load_from_env()

            assert len(config.servers) == 1
            assert config.servers[0].name == "good_server"

    def test_validate_config_file_valid(self):
        """Test validating a valid configuration file."""
        config_data = {
            "mcpServers": {
                "test_server": {
                    "url": "http://localhost:8001/mcp/",
                    "transport": "http",
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            assert MCPConfigLoader.validate_config_file(temp_path) is True
        finally:
            Path(temp_path).unlink()

    def test_validate_config_file_invalid(self):
        """Test validating an invalid configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('invalid json')
            temp_path = f.name

        try:
            assert MCPConfigLoader.validate_config_file(temp_path) is False
        finally:
            Path(temp_path).unlink()

    def test_get_config_schema(self):
        """Test getting the configuration schema."""
        schema = MCPConfigLoader.get_config_schema()

        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "mcpServers" in schema["properties"]


class TestLoadMCPConfig:
    """Test the convenience load_mcp_config function."""

    def test_load_with_explicit_path(self):
        """Test loading with explicit config file path."""
        config_data = {
            "mcpServers": {
                "explicit_server": {
                    "url": "http://explicit.example.com/mcp/",
                    "transport": "http",
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_mcp_config(temp_path)
            assert len(config.servers) == 1
            assert config.servers[0].name == "explicit_server"
        finally:
            Path(temp_path).unlink()

    def test_load_fallback_to_env(self):
        """Test fallback to environment variables when no file found."""
        env_vars = {
            "MCP_SERVERS": "env_server",
            "MCP_ENV_SERVER_URL": "http://env.example.com/mcp/",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            # Clear any existing config files
            with patch.object(MCPConfigLoader, '_find_default_config', return_value=None):
                config = load_mcp_config()

                assert len(config.servers) == 1
                assert config.servers[0].name == "env_server"

    def test_load_fallback_to_empty(self):
        """Test fallback to empty configuration when nothing found."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(MCPConfigLoader, '_find_default_config', return_value=None):
                config = load_mcp_config()

                assert len(config.servers) == 0


if __name__ == "__main__":
    pytest.main([__file__])
