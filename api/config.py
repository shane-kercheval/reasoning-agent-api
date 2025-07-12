"""
Configuration settings for the Reasoning Agent API.

This module defines the application settings using Pydantic BaseSettings,
including API keys, server URLs, and runtime configuration options.
"""

import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings for the Reasoning Agent API."""

    # OpenAI API
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # MCP Server URLs (will be remote deployed servers)
    MCP_WEB_SEARCH_URL: str = os.getenv("MCP_WEB_SEARCH_URL", "http://localhost:8001")
    MCP_WEATHER_URL: str = os.getenv("MCP_WEATHER_URL", "http://localhost:8002")
    MCP_FILESYSTEM_URL: str = os.getenv("MCP_FILESYSTEM_URL", "http://localhost:8003")

    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Development
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    class Config:
        """Configuration for settings."""

        env_file = ".env"

settings = Settings()
