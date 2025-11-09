"""
Unit tests for MCP Bridge server.

Tests configuration loading, validation, and error handling without spawning
actual server processes.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from mcp_bridge.server import load_config, create_bridge


class TestConfigLoading:
    """Test configuration file loading and validation."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """Test loading a valid configuration file."""
        config_file = tmp_path / "config.json"
        config_data = {
            "mcpServers": {
                "test-server": {
                    "command": "python",
                    "args": ["server.py"],
                    "transport": "stdio",
                },
            },
        }
        config_file.write_text(json.dumps(config_data))

        config = load_config(config_file)

        assert "mcpServers" in config
        assert "test-server" in config["mcpServers"]
        assert config["mcpServers"]["test-server"]["command"] == "python"

    def test_load_config_file_not_found(self, tmp_path: Path) -> None:
        """Test loading non-existent configuration file."""
        config_file = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(config_file)

    def test_load_config_invalid_json(self, tmp_path: Path) -> None:
        """Test loading invalid JSON."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("{ invalid json }")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_config(config_file)

    def test_load_config_missing_mcp_servers(self, tmp_path: Path) -> None:
        """Test loading config without mcpServers key."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"other": "data"}))

        with pytest.raises(ValueError, match="must contain 'mcpServers'"):
            load_config(config_file)

    def test_load_config_empty_servers(self, tmp_path: Path) -> None:
        """Test loading config with empty mcpServers."""
        config_file = tmp_path / "config.json"
        config_data = {"mcpServers": {}}
        config_file.write_text(json.dumps(config_data))

        config = load_config(config_file)

        assert "mcpServers" in config
        assert len(config["mcpServers"]) == 0


class TestBridgeCreation:
    """Test bridge server creation."""

    @patch("mcp_bridge.server.FastMCP.as_proxy")
    def test_create_bridge_basic(self, mock_as_proxy: Mock) -> None:
        """Test basic bridge creation."""
        mock_bridge = Mock()
        mock_as_proxy.return_value = mock_bridge

        config = {
            "mcpServers": {
                "test-server": {
                    "command": "python",
                    "args": ["server.py"],
                    "transport": "stdio",
                },
            },
        }

        bridge = create_bridge(config, name="Test Bridge")

        mock_as_proxy.assert_called_once_with(config, name="Test Bridge")
        assert bridge == mock_bridge

    @patch("mcp_bridge.server.FastMCP.as_proxy")
    def test_create_bridge_multiple_servers(self, mock_as_proxy: Mock) -> None:
        """Test bridge creation with multiple servers."""
        mock_bridge = Mock()
        mock_as_proxy.return_value = mock_bridge

        config = {
            "mcpServers": {
                "server1": {
                    "command": "python",
                    "args": ["server1.py"],
                    "transport": "stdio",
                },
                "server2": {
                    "command": "python",
                    "args": ["server2.py"],
                    "transport": "stdio",
                },
            },
        }

        bridge = create_bridge(config)

        mock_as_proxy.assert_called_once()
        assert bridge == mock_bridge

    @patch("mcp_bridge.server.FastMCP.as_proxy")
    def test_create_bridge_empty_servers(self, mock_as_proxy: Mock) -> None:
        """Test bridge creation with no servers."""
        mock_bridge = Mock()
        mock_as_proxy.return_value = mock_bridge

        config = {"mcpServers": {}}

        bridge = create_bridge(config)

        mock_as_proxy.assert_called_once_with(config, name="MCP Bridge")
        assert bridge == mock_bridge
