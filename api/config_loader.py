"""Configuration loader for MCP servers with validation."""
import os
from pathlib import Path

import yaml
from pydantic import ValidationError
from .reasoning_models import MCPServersConfig, MCPServerConfig


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""

    pass


class MCPConfigLoader:
    """Loads and validates MCP server configuration from YAML files."""

    @staticmethod
    def load_from_file(config_path: str | None = None) -> MCPServersConfig:
        """
        Load MCP server configuration from a YAML file.

        Args:
            config_path: Path to the configuration file. If None, uses default paths.

        Returns:
            Validated MCP servers configuration

        Raises:
            ConfigurationError: If configuration file is not found, invalid YAML,
                               or fails Pydantic validation
        """
        if config_path is None:
            config_path = MCPConfigLoader._find_default_config()

        if not config_path or not Path(config_path).exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, encoding='utf-8') as file:
                config_data = yaml.safe_load(file)

            if config_data is None:
                config_data = {"servers": []}

            # Validate configuration using Pydantic
            return MCPServersConfig(**config_data)

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}")
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

    @staticmethod
    def _find_default_config() -> str | None:
        """
        Find the default configuration file in standard locations.

        Returns:
            Path to the first found configuration file, or None if not found
        """
        # Check environment variable first
        env_config_path = os.getenv("MCP_CONFIG_PATH")
        if env_config_path and Path(env_config_path).exists():
            return env_config_path

        # Check standard locations
        possible_paths = [
            "config/mcp_servers.yaml",
            "config/mcp_servers.yml",
            "mcp_servers.yaml",
            "mcp_servers.yml",
            "/etc/reasoning-agent/mcp_servers.yaml",
            "~/.config/reasoning-agent/mcp_servers.yaml",
        ]

        for path in possible_paths:
            expanded_path = Path(path).expanduser()
            if expanded_path.exists():
                return str(expanded_path)

        return None

    @staticmethod
    def load_from_env() -> MCPServersConfig:
        """
        Load configuration from environment variables.

        This method creates a minimal configuration based on environment
        variables, useful for simple deployments or testing.

        Environment variables:
        - MCP_SERVERS: Comma-separated list of server names
        - MCP_{SERVER_NAME}_URL: HTTP URL for the MCP server
        - MCP_{SERVER_NAME}_AUTH_ENV: Auth environment variable name
        - MCP_{SERVER_NAME}_ENABLED: Whether server is enabled (default: true)

        Returns:
            Configuration created from environment variables
        """
        server_names = os.getenv("MCP_SERVERS", "").split(",")
        server_names = [name.strip() for name in server_names if name.strip()]

        if not server_names:
            return MCPServersConfig(servers=[])

        servers = []
        for server_name in server_names:
            server_prefix = f"MCP_{server_name.upper()}"

            url = os.getenv(f"{server_prefix}_URL")
            if not url:
                continue  # Skip servers without URL

            config = {
                "name": server_name,
                "url": url,
                "enabled": os.getenv(f"{server_prefix}_ENABLED", "true").lower() == "true",
            }

            # Add authentication
            auth_env = os.getenv(f"{server_prefix}_AUTH_ENV")
            if auth_env:
                config["auth_env_var"] = auth_env

            try:
                server_config = MCPServerConfig(**config)
                servers.append(server_config)
            except ValidationError:
                # Skip invalid server configurations
                continue

        return MCPServersConfig(servers=servers)

    @staticmethod
    def validate_config_file(config_path: str) -> bool:
        """
        Validate a configuration file without loading it.

        Args:
            config_path: Path to the configuration file to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            MCPConfigLoader.load_from_file(config_path)
            return True
        except ConfigurationError:
            return False

    @staticmethod
    def get_config_schema() -> dict:
        """
        Get the JSON schema for the configuration format.

        Returns:
            JSON schema dictionary for the configuration
        """
        return MCPServersConfig.model_json_schema()


def load_mcp_config(config_path: str | None = None) -> MCPServersConfig:
    """
    Convenience function to load MCP configuration.

    This function tries multiple sources in order:
    1. Explicit config file path
    2. Default file locations
    3. Environment variables
    4. Empty configuration (fallback)

    Args:
        config_path: Optional path to configuration file

    Returns:
        Loaded and validated MCP configuration
    """
    # Try explicit file path or default locations
    if config_path or MCPConfigLoader._find_default_config():
        try:
            return MCPConfigLoader.load_from_file(config_path)
        except ConfigurationError:
            pass  # Fall back to environment variables

    # Try environment variables
    try:
        config = MCPConfigLoader.load_from_env()
        if config.servers:  # Only use if we found some servers
            return config
    except Exception:
        pass  # Fall back to empty configuration

    # Fallback to empty configuration
    return MCPServersConfig(servers=[])
