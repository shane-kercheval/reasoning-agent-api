"""
Unit tests for MCP Bridge server.

Tests configuration loading, validation, and error handling without spawning
actual server processes.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

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

    @pytest.mark.asyncio
    @patch("mcp_bridge.server.Client")
    @patch("mcp_bridge.server.FastMCP")
    async def test_create_bridge_basic(
        self, mock_fastmcp: Mock, mock_client_class: Mock,
    ) -> None:
        """Test basic bridge creation with one enabled server."""
        # Mock FastMCP instance
        mock_bridge = Mock()
        mock_fastmcp.return_value = mock_bridge

        # Mock Client instance and tool listing
        mock_client = AsyncMock()
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        mock_tool.inputSchema = {"properties": {}, "required": []}
        mock_client.list_tools = AsyncMock(return_value=[mock_tool])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value = mock_client

        config = {
            "mcpServers": {
                "test-server": {
                    "command": "python",
                    "args": ["server.py"],
                    "transport": "stdio",
                    "enabled": True,
                },
            },
        }

        bridge = await create_bridge(config, name="Test Bridge")

        # Verify FastMCP was created with correct name
        mock_fastmcp.assert_called_once_with(name="Test Bridge")

        # Verify Client was created and tools were listed
        mock_client_class.assert_called_once()
        mock_client.list_tools.assert_called_once()

        # Verify bridge.tool was called to register the tool
        mock_bridge.tool.assert_called_once()

        assert bridge == mock_bridge

    @pytest.mark.asyncio
    @patch("mcp_bridge.server.Client")
    @patch("mcp_bridge.server.FastMCP")
    async def test_create_bridge_filters_disabled_servers(
        self, mock_fastmcp: Mock, mock_client_class: Mock,
    ) -> None:
        """Test that disabled servers are not initialized."""
        mock_bridge = Mock()
        mock_fastmcp.return_value = mock_bridge

        config = {
            "mcpServers": {
                "enabled-server": {
                    "command": "python",
                    "args": ["enabled.py"],
                    "transport": "stdio",
                    "enabled": True,
                },
                "disabled-server": {
                    "command": "python",
                    "args": ["disabled.py"],
                    "transport": "stdio",
                    "enabled": False,
                },
            },
        }

        # Mock client for enabled server only
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value = mock_client

        bridge = await create_bridge(config)

        # Should only create one client (for enabled server)
        mock_client_class.assert_called_once()
        assert bridge == mock_bridge

    @pytest.mark.asyncio
    @patch("mcp_bridge.server.Client")
    @patch("mcp_bridge.server.FastMCP")
    async def test_create_bridge_empty_servers(
        self, mock_fastmcp: Mock, mock_client_class: Mock,
    ) -> None:
        """Test bridge creation with no servers."""
        mock_bridge = Mock()
        mock_fastmcp.return_value = mock_bridge

        config = {"mcpServers": {}}

        bridge = await create_bridge(config)

        # No clients should be created
        mock_client_class.assert_not_called()

        # Bridge should still be created
        mock_fastmcp.assert_called_once_with(name="MCP Bridge")
        assert bridge == mock_bridge
