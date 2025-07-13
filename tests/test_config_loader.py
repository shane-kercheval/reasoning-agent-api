"""Tests for MCP configuration loader."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

from api.config_loader import ConfigurationError, MCPConfigLoader, load_mcp_config


class TestMCPConfigLoader:
    """Test MCP configuration loading functionality."""

    def test_load_valid_yaml_config(self):
        """Test loading a valid YAML configuration file."""
        config_content = """
servers:
  - name: test_server
    url: http://localhost:8001
    enabled: true
  - name: web_server
    url: https://example.com
    auth_env_var: API_KEY
    enabled: false
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config = MCPConfigLoader.load_from_file(config_path)

            assert len(config.servers) == 2
            assert config.servers[0].name == "test_server"
            assert config.servers[0].url == "http://localhost:8001"
            assert config.servers[0].enabled is True

            assert config.servers[1].name == "web_server"
            assert config.servers[1].url == "https://example.com"
            assert config.servers[1].auth_env_var == "API_KEY"
            assert config.servers[1].enabled is False

        finally:
            os.unlink(config_path)

    def test_load_empty_config_file(self):
        """Test loading an empty configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")  # Empty file
            config_path = f.name

        try:
            config = MCPConfigLoader.load_from_file(config_path)
            assert len(config.servers) == 0
        finally:
            os.unlink(config_path)

    def test_load_nonexistent_file(self):
        """Test loading a nonexistent configuration file."""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            MCPConfigLoader.load_from_file("/nonexistent/path/config.yaml")

    def test_load_invalid_yaml(self):
        """Test loading a file with invalid YAML syntax."""
        invalid_yaml = """
servers:
  - name: test
    transport: stdio
    invalid_yaml: [unclosed bracket
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            config_path = f.name

        try:
            with pytest.raises(ConfigurationError, match="Invalid YAML"):
                MCPConfigLoader.load_from_file(config_path)
        finally:
            os.unlink(config_path)

    def test_load_invalid_configuration(self):
        """Test loading YAML with invalid configuration structure."""
        invalid_config = """
servers:
  - name: test_server
    # Missing required url field
    auth_env_var: API_KEY
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_config)
            config_path = f.name

        try:
            with pytest.raises(ConfigurationError, match="Configuration validation failed"):
                MCPConfigLoader.load_from_file(config_path)
        finally:
            os.unlink(config_path)

    @patch.dict(os.environ, {
        "MCP_CONFIG_PATH": "/custom/path/config.yaml",
    })
    def test_find_config_from_env_var(self):
        """Test finding configuration file from environment variable."""
        # Create a temporary file at the expected path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("servers: []")
            config_path = f.name

        try:
            with patch.dict(os.environ, {"MCP_CONFIG_PATH": config_path}):
                found_path = MCPConfigLoader._find_default_config()
                assert found_path == config_path
        finally:
            os.unlink(config_path)

    def test_find_config_standard_locations(self):
        """Test finding configuration in standard locations."""
        # Create a config file in current directory
        config_path = Path("mcp_servers.yaml")
        config_path.write_text("servers: []")

        try:
            # Mock environment to not find the config/ file first
            with patch.dict(os.environ, {}, clear=True):
                found_path = MCPConfigLoader._find_default_config()
                # Could find either our test file or the config/ file - just check a file was found
                assert found_path is not None
                assert "mcp_servers.yaml" in found_path
        finally:
            if config_path.exists():
                config_path.unlink()

    def test_find_config_not_found(self):
        """Test behavior when no configuration file is found."""
        # Temporarily move the actual config file if it exists
        actual_config = Path("config/mcp_servers.yaml")
        temp_location = None

        if actual_config.exists():
            temp_location = Path("config/mcp_servers.yaml.bak")
            actual_config.rename(temp_location)

        try:
            with patch.dict(os.environ, {}, clear=True):
                found_path = MCPConfigLoader._find_default_config()
                assert found_path is None
        finally:
            # Restore the actual config file if we moved it
            if temp_location and temp_location.exists():
                temp_location.rename(actual_config)

    @patch.dict(os.environ, {
        "MCP_SERVERS": "server1,server2",
        "MCP_SERVER1_URL": "http://localhost:8001",
        "MCP_SERVER1_ENABLED": "true",
        "MCP_SERVER2_URL": "https://example.com",
        "MCP_SERVER2_AUTH_ENV": "API_TOKEN",
        "MCP_SERVER2_ENABLED": "false",
    })
    def test_load_from_env_variables(self):
        """Test loading configuration from environment variables."""
        config = MCPConfigLoader.load_from_env()

        assert len(config.servers) == 2

        # Check server1
        server1 = next(s for s in config.servers if s.name == "server1")
        assert server1.url == "http://localhost:8001"
        assert server1.auth_env_var is None
        assert server1.enabled is True

        # Check server2
        server2 = next(s for s in config.servers if s.name == "server2")
        assert server2.url == "https://example.com"
        assert server2.auth_env_var == "API_TOKEN"
        assert server2.enabled is False

    @patch.dict(os.environ, {
        "MCP_SERVERS": "",
    }, clear=True)
    def test_load_from_env_no_servers(self):
        """Test loading from environment when no servers are configured."""
        config = MCPConfigLoader.load_from_env()
        assert len(config.servers) == 0

    @patch.dict(os.environ, {
        "MCP_SERVERS": "incomplete_server",
        "MCP_INCOMPLETE_SERVER_ENABLED": "true",
        # Missing URL - should be skipped
    })
    def test_load_from_env_incomplete_server(self):
        """Test that incomplete server configurations are skipped."""
        config = MCPConfigLoader.load_from_env()
        assert len(config.servers) == 0

    def test_validate_config_file_valid(self):
        """Test validation of a valid configuration file."""
        config_content = """
servers:
  - name: test_server
    url: http://localhost:8001
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            assert MCPConfigLoader.validate_config_file(config_path) is True
        finally:
            os.unlink(config_path)

    def test_validate_config_file_invalid(self):
        """Test validation of an invalid configuration file."""
        invalid_config = """
servers:
  - name: test_server
    # Missing required url
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_config)
            config_path = f.name

        try:
            assert MCPConfigLoader.validate_config_file(config_path) is False
        finally:
            os.unlink(config_path)

    def test_get_config_schema(self):
        """Test getting the configuration JSON schema."""
        schema = MCPConfigLoader.get_config_schema()

        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "servers" in schema["properties"]
        assert schema["properties"]["servers"]["type"] == "array"


class TestLoadMCPConfig:
    """Test the convenience function for loading configuration."""

    def test_load_with_explicit_path(self):
        """Test loading with an explicit configuration file path."""
        config_content = """
servers:
  - name: explicit_server
    url: http://localhost:8001
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config = load_mcp_config(config_path)
            assert len(config.servers) == 1
            assert config.servers[0].name == "explicit_server"
        finally:
            os.unlink(config_path)

    def test_load_from_default_location(self):
        """Test loading from default file location."""
        config_content = """
servers:
  - name: default_server
    url: http://localhost:8001
"""

        # Create config in current directory (but use different name to avoid conflict)
        config_path = Path("test_mcp_servers.yaml")
        config_path.write_text(config_content)

        try:
            # Mock the default config finder to return our test file
            with patch.object(MCPConfigLoader, '_find_default_config', return_value=str(config_path)):
                config = load_mcp_config()
                assert len(config.servers) == 1
                assert config.servers[0].name == "default_server"
        finally:
            config_path.unlink()

    @patch.dict(os.environ, {
        "MCP_SERVERS": "env_server",
        "MCP_ENV_SERVER_URL": "http://localhost:8001",
    })
    def test_load_fallback_to_env(self):
        """Test fallback to environment variables when no file is found."""
        # Ensure no config files exist
        with patch.object(MCPConfigLoader, '_find_default_config', return_value=None):
            config = load_mcp_config()
            assert len(config.servers) == 1
            assert config.servers[0].name == "env_server"

    def test_load_fallback_to_empty(self):
        """Test fallback to empty configuration when nothing is found."""
        with patch.object(MCPConfigLoader, '_find_default_config', return_value=None):
            with patch.dict(os.environ, {}, clear=True):
                config = load_mcp_config()
                assert len(config.servers) == 0
