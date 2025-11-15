"""
Configuration settings for the Reasoning Agent API.

This module defines the application settings using Pydantic v2 BaseSettings,
including API keys, server URLs, and runtime configuration options.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings for the Reasoning Agent API."""

    # LLM API Configuration (LiteLLM proxy)
    llm_api_key: str = Field(
        default="",
        alias="LITELLM_API_KEY",
        description="API key for LLM requests (virtual key from LiteLLM)",
    )
    llm_base_url: str = Field(
        default="http://litellm:4000",
        alias="LITELLM_BASE_URL",
        description="Base URL for LLM API (LiteLLM proxy in production)",
    )


    # HTTP Client Configuration
    http_connect_timeout: float = Field(
        default=5.0,
        alias="HTTP_CONNECT_TIMEOUT",
        description="HTTP connection timeout in seconds",
    )
    http_read_timeout: float = Field(
        default=60.0,
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
        default="config/mcp_servers.json",
        alias="MCP_CONFIG_PATH",
        description="Path to MCP server configuration file (YAML or JSON)",
    )
    mcp_filter_deprecated: bool = Field(
        default=True,
        alias="MCP_FILTER_DEPRECATED",
        description="Filter out tools marked as deprecated (containing 'DEPRECATED' in description)",  # noqa: E501
    )
    mcp_strip_prefixes: str = Field(
        default="",
        alias="MCP_STRIP_PREFIXES",
        description="Comma-separated list of prefixes to strip from MCP tool/prompt names (e.g., 'local_bridge_,proxy_')",  # noqa: E501
    )

    # Database Configuration
    reasoning_database_url: str = Field(
        default="postgresql+asyncpg://reasoning_user:reasoning_dev_password123@localhost:5434/reasoning",
        alias="REASONING_DATABASE_URL",
        description="PostgreSQL database URL for conversation storage",
    )

    # Development
    debug: bool = Field(default=False, alias="DEBUG")

    # Request Routing Configuration
    routing_classifier_model: str = Field(
        default="gpt-4o-mini",
        alias="ROUTING_CLASSIFIER_MODEL",
        description="Model to use for request complexity classification",
    )
    routing_classifier_temperature: float = Field(
        default=0.0,
        alias="ROUTING_CLASSIFIER_TEMPERATURE",
        description="Temperature for routing classifier (0.0 for deterministic)",
    )

    # Phoenix Tracing Configuration
    phoenix_collector_endpoint: str = Field(
        default="http://localhost:4317",
        alias="PHOENIX_COLLECTOR_ENDPOINT",
        description="Phoenix OTLP collector endpoint for tracing",
    )
    phoenix_project_name: str = Field(
        default="reasoning-agent",
        alias="PHOENIX_PROJECT_NAME",
        description="Project name for organizing traces in Phoenix",
    )
    phoenix_api_key: str = Field(
        default="",
        alias="PHOENIX_API_KEY",
        description="Optional API key for Phoenix authentication",
    )
    enable_tracing: bool = Field(
        default=False,
        alias="ENABLE_TRACING",
        description="Whether to enable OpenTelemetry tracing",
    )

    enable_console_tracing: bool = Field(
        default=False,
        alias="ENABLE_CONSOLE_TRACING",
        description="Whether to output traces to console for debugging",
    )

    model_config = {
        'env_file': '.env',
        'env_file_encoding': 'utf-8',
        'case_sensitive': False,
        'extra': 'ignore',  # Ignore extra fields from shared .env file
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

    @property
    def mcp_prefixes_to_strip(self) -> list[str]:
        """
        Parse comma-separated MCP prefixes into a list.

        Returns:
            List of prefixes to strip from MCP tool/prompt names, empty if none configured.

        Example:
            >>> settings.mcp_strip_prefixes = "local_bridge_,proxy_"
            >>> settings.mcp_prefixes_to_strip
            ['local_bridge_', 'proxy_']
        """
        if not self.mcp_strip_prefixes:
            return []
        return [prefix.strip() for prefix in self.mcp_strip_prefixes.split(',') if prefix.strip()]


settings = Settings()
