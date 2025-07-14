"""
Configuration settings for the Reasoning Agent API.

This module defines the application settings using Pydantic v2 BaseSettings,
including API keys, server URLs, and runtime configuration options.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings for the Reasoning Agent API."""

    # OpenAI API Configuration
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    reasoning_agent_base_url: str = Field(
        default="https://api.openai.com/v1",
        alias="REASONING_AGENT_BASE_URL",
    )

    # MCP Server URLs (optional, for future use)
    mcp_web_search_url: str = Field(
        default="http://localhost:8001",
        alias="MCP_WEB_SEARCH_URL",
    )
    mcp_weather_url: str = Field(
        default="http://localhost:8002",
        alias="MCP_WEATHER_URL",
    )
    mcp_filesystem_url: str = Field(
        default="http://localhost:8003",
        alias="MCP_FILESYSTEM_URL",
    )

    # HTTP Client Configuration
    http_connect_timeout: float = Field(
        default=5.0,
        alias="HTTP_CONNECT_TIMEOUT",
        description="HTTP connection timeout in seconds",
    )
    http_read_timeout: float = Field(
        default=30.0,
        alias="HTTP_READ_TIMEOUT",
        description="HTTP read timeout in seconds",
    )
    http_write_timeout: float = Field(
        default=10.0,
        alias="HTTP_WRITE_TIMEOUT",
        description="HTTP write timeout in seconds",
    )
    http_max_connections: int = Field(
        default=20,
        alias="HTTP_MAX_CONNECTIONS",
        description="Maximum total HTTP connections",
    )
    http_max_keepalive_connections: int = Field(
        default=5,
        alias="HTTP_MAX_KEEPALIVE_CONNECTIONS",
        description="Maximum keep-alive HTTP connections",
    )
    http_keepalive_expiry: float = Field(
        default=30.0,
        alias="HTTP_KEEPALIVE_EXPIRY",
        description="Keep-alive connection expiry time in seconds",
    )

    # Authentication Configuration
    api_tokens: str = Field(
        default="",
        alias="API_TOKENS",
        description="Comma-separated list of allowed bearer tokens",
    )
    require_auth: bool = Field(
        default=True,
        alias="REQUIRE_AUTH",
        description="Whether to require authentication for protected endpoints",
    )

    # API Configuration
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # MCP Configuration
    mcp_config_path: str = Field(
        default="config/mcp_servers.yaml",
        alias="MCP_CONFIG_PATH",
        description="Path to MCP server configuration file (YAML or JSON)",
    )

    # Development
    debug: bool = Field(default=False, alias="DEBUG")

    model_config = {
        'env_file': '.env',
        'env_file_encoding': 'utf-8',
        'case_sensitive': False,
    }

    @property
    def allowed_tokens(self) -> list[str]:
        """
        Parse comma-separated API tokens into a list.

        Returns:
            List of valid API tokens, empty if none configured.

        Example:
            >>> settings.api_tokens = "token1,token2,token3"
            >>> settings.allowed_tokens
            ['token1', 'token2', 'token3']
        """
        if not self.api_tokens:
            return []
        return [token.strip() for token in self.api_tokens.split(',') if token.strip()]


settings = Settings()
