"""
Tests for application configuration settings.

Tests the Pydantic BaseSettings configuration including HTTP client settings,
authentication configuration, and environment variable parsing.
"""

from unittest.mock import patch
from api.dependencies import create_production_http_client
from api.config import settings, Settings


class TestSettings:
    """Test application settings configuration."""

    @patch.dict('os.environ', {}, clear=True)
    def test__default_values__are_production_ready(self):
        """Test that default configuration values are suitable for production."""
        with patch.object(Settings, 'model_config', {**Settings.model_config, 'env_file': None}):
            settings = Settings()

        # HTTP timeouts should be reasonable
        assert settings.http_connect_timeout == 5.0  # Fast failure
        assert settings.http_read_timeout == 60.0    # Reasonable for AI responses
        assert settings.http_write_timeout == 10.0   # Reasonable for uploads

        # Connection limits should be conservative
        assert settings.http_max_connections == 20
        assert settings.http_max_keepalive_connections == 5
        assert settings.http_keepalive_expiry == 30.0

        # Auth should be enabled by default
        assert settings.require_auth is True
        assert settings.api_tokens == ""

        # Server defaults
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000

    def test__allowed_tokens__parses_comma_separated_values(self):
        """Test that allowed_tokens property correctly parses comma-separated tokens."""
        settings = Settings()

        # Empty tokens
        settings.api_tokens = ""
        assert settings.allowed_tokens == []

        # Single token
        settings.api_tokens = "token1"
        assert settings.allowed_tokens == ["token1"]

        # Multiple tokens
        settings.api_tokens = "token1,token2,token3"
        assert settings.allowed_tokens == ["token1", "token2", "token3"]

        # Tokens with spaces (should be stripped)
        settings.api_tokens = " token1 , token2 , token3 "
        assert settings.allowed_tokens == ["token1", "token2", "token3"]

        # Empty tokens in list (should be filtered out)
        settings.api_tokens = "token1,,token2,"
        assert settings.allowed_tokens == ["token1", "token2"]

    def test__allowed_tokens__handles_edge_cases(self):
        """Test that allowed_tokens handles edge cases gracefully."""
        settings = Settings()

        # Only commas
        settings.api_tokens = ",,,"
        assert settings.allowed_tokens == []

        # Only spaces
        settings.api_tokens = "   "
        assert settings.allowed_tokens == []

        # Mixed valid and empty
        settings.api_tokens = "valid1,  , valid2,,"
        assert settings.allowed_tokens == ["valid1", "valid2"]

    @patch.dict('os.environ', {
        'HTTP_CONNECT_TIMEOUT': '10.0',
        'HTTP_READ_TIMEOUT': '60.0',
        'API_TOKENS': 'prod-token-1,prod-token-2',
        'REQUIRE_AUTH': 'false',
    })
    def test__environment_variables__override_defaults(self):
        """Test that environment variables properly override default values."""
        settings = Settings()

        # HTTP settings should be overridden
        assert settings.http_connect_timeout == 10.0
        assert settings.http_read_timeout == 60.0

        # Auth settings should be overridden
        assert settings.api_tokens == "prod-token-1,prod-token-2"
        assert settings.allowed_tokens == ["prod-token-1", "prod-token-2"]
        assert settings.require_auth is False

    def test__field_descriptions__are_present(self):
        """Test that important fields have helpful descriptions."""
        # Check that key fields have descriptions for documentation
        field_info = Settings.model_fields

        assert field_info['http_connect_timeout'].description is not None
        assert field_info['http_read_timeout'].description is not None
        assert field_info['api_tokens'].description is not None
        assert field_info['require_auth'].description is not None
        assert field_info['mcp_config_path'].description is not None

        # Descriptions should be helpful
        assert "timeout" in field_info['http_connect_timeout'].description.lower()
        assert "token" in field_info['api_tokens'].description.lower()
        assert "auth" in field_info['require_auth'].description.lower()
        assert "mcp" in field_info['mcp_config_path'].description.lower()

    def test__http_configuration__has_reasonable_bounds(self):
        """Test that HTTP configuration values are within reasonable bounds."""
        settings = Settings()

        # Timeouts should be positive and reasonable
        assert 0 < settings.http_connect_timeout < 60  # Not too long
        assert 0 < settings.http_read_timeout < 300    # Not excessive
        assert 0 < settings.http_write_timeout < 60    # Reasonable upload time

        # Connection limits should be positive
        assert settings.http_max_connections > 0
        assert settings.http_max_keepalive_connections > 0
        assert settings.http_keepalive_expiry > 0


class TestSettingsIntegration:
    """Test settings integration with the application."""

    def test__settings_can_be_imported__without_errors(self):
        """Test that settings can be imported and instantiated without errors."""
        # Should be able to access all properties
        assert isinstance(settings.http_connect_timeout, float)
        assert isinstance(settings.allowed_tokens, list)
        assert isinstance(settings.require_auth, bool)
        assert isinstance(settings.mcp_config_path, str)

    def test__mcp_config_path__default_value(self):
        """Test MCP config path has correct default value."""
        settings = Settings()
        assert settings.mcp_config_path == "config/mcp_servers.json"

    def test__mcp_config_path__environment_override(self, monkeypatch):  # noqa: ANN001
        """Test MCP config path can be overridden via environment variable."""
        # Set environment variable
        monkeypatch.setenv("MCP_CONFIG_PATH", "custom/path/config.yaml")

        # Create settings with custom environment
        settings = Settings(_env_file=None)
        assert settings.mcp_config_path == "custom/path/config.yaml"

    def test__mcp_config_path__supports_json_files(self, monkeypatch):  # noqa: ANN001
        """Test MCP config path supports JSON files."""
        # Set environment variable to JSON path
        monkeypatch.setenv("MCP_CONFIG_PATH", "config/servers.json")

        # Create settings with custom environment
        settings = Settings(_env_file=None)
        assert settings.mcp_config_path == "config/servers.json"

    def test__settings_work_with_dependencies__import(self):
        """Test that settings work correctly with dependency injection."""
        # Should be able to create HTTP client using settings
        client = create_production_http_client()
        assert client is not None

        # Verify it uses settings values
        assert client.timeout.connect == settings.http_connect_timeout
