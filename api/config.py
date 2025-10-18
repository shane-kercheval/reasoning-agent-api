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


settings = Settings()
