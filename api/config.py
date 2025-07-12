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

    # API Configuration
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # Development
    debug: bool = Field(default=False, alias="DEBUG")

    model_config = {
        'env_file': '.env',
        'env_file_encoding': 'utf-8',
        'case_sensitive': False,
    }


settings = Settings()
