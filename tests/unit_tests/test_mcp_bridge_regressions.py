"""
Regression tests for MCP Bridge bugs.

These tests ensure specific bugs stay fixed.
"""

import inspect
import pytest
from unittest.mock import AsyncMock, Mock, patch

from mcp_bridge.server import create_bridge


class TestToolNameSanitization:
    """Regression tests for tool name sanitization (hyphens/dots in names)."""

    @pytest.mark.asyncio
    @patch("mcp_bridge.server.Client")
    @patch("mcp_bridge.server.FastMCP")
    async def test_hyphens_in_server_and_tool_names_are_sanitized(
        self, mock_fastmcp: Mock, mock_client_class: Mock,
    ) -> None:
        """
        Test that hyphens in server/tool names become underscores.

        Regression test for bug where 'test-server' + 'echo-message' created
        invalid function name 'test-server_echo-message' (hyphens not allowed
        in Python identifiers), causing SyntaxError.
        """
        mock_bridge = Mock()
        mock_fastmcp.return_value = mock_bridge

        mock_client = AsyncMock()
        mock_tool = Mock()
        mock_tool.name = "echo-message"  # Hyphen in tool name
        mock_tool.description = "Echo a message"
        mock_tool.inputSchema = {
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }
        mock_client.list_tools = AsyncMock(return_value=[mock_tool])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value = mock_client

        config = {
            "mcpServers": {
                "echo-server": {  # Hyphen in server name
                    "command": "python",
                    "args": ["server.py"],
                    "transport": "stdio",
                    "enabled": True,
                },
            },
        }

        # Should not raise SyntaxError
        await create_bridge(config)

        # Verify tool was registered with sanitized name
        mock_bridge.tool.assert_called_once()
        registered_func = mock_bridge.tool.call_args[0][0]

        # Function name should be sanitized: echo_server_echo_message
        assert registered_func.__name__ == "echo_server_echo_message"
        assert "-" not in registered_func.__name__  # No hyphens

    @pytest.mark.asyncio
    @patch("mcp_bridge.server.Client")
    @patch("mcp_bridge.server.FastMCP")
    async def test_dots_in_names_are_sanitized(
        self, mock_fastmcp: Mock, mock_client_class: Mock,
    ) -> None:
        """Test that dots in server/tool names become underscores."""
        mock_bridge = Mock()
        mock_fastmcp.return_value = mock_bridge

        mock_client = AsyncMock()
        mock_tool = Mock()
        mock_tool.name = "io.read"  # Dot in tool name
        mock_tool.description = "Read file"
        mock_tool.inputSchema = {"properties": {}, "required": []}
        mock_client.list_tools = AsyncMock(return_value=[mock_tool])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value = mock_client

        config = {
            "mcpServers": {
                "fs.server": {  # Dot in server name
                    "command": "python",
                    "args": ["server.py"],
                    "transport": "stdio",
                    "enabled": True,
                },
            },
        }

        await create_bridge(config)

        registered_func = mock_bridge.tool.call_args[0][0]
        assert registered_func.__name__ == "fs_server_io_read"
        assert "." not in registered_func.__name__  # No dots


class TestToolsWithNoParameters:
    """Regression tests for tools with empty inputSchema."""

    @pytest.mark.asyncio
    @patch("mcp_bridge.server.Client")
    @patch("mcp_bridge.server.FastMCP")
    async def test_tool_with_no_parameters_generates_valid_code(
        self, mock_fastmcp: Mock, mock_client_class: Mock,
    ) -> None:
        """
        Test tools with empty properties don't cause syntax errors.

        Regression test to ensure tools without parameters generate valid
        parameterless functions.
        """
        mock_bridge = Mock()
        mock_fastmcp.return_value = mock_bridge

        mock_client = AsyncMock()
        mock_tool = Mock()
        mock_tool.name = "ping"
        mock_tool.description = "Health check"
        mock_tool.inputSchema = {"properties": {}, "required": []}  # No parameters
        mock_client.list_tools = AsyncMock(return_value=[mock_tool])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value = mock_client

        config = {
            "mcpServers": {
                "health": {
                    "command": "python",
                    "args": ["server.py"],
                    "transport": "stdio",
                    "enabled": True,
                },
            },
        }

        # Should not raise SyntaxError for parameterless function
        await create_bridge(config)

        # Verify function has no parameters
        registered_func = mock_bridge.tool.call_args[0][0]
        sig = inspect.signature(registered_func)
        assert len(sig.parameters) == 0


class TestSpecialCharactersInDescriptions:
    """Regression tests for descriptions with quotes and special characters."""

    @pytest.mark.asyncio
    @patch("mcp_bridge.server.Client")
    @patch("mcp_bridge.server.FastMCP")
    async def test_quotes_in_description_dont_break_fstring(
        self, mock_fastmcp: Mock, mock_client_class: Mock,
    ) -> None:
        """
        Test descriptions with quotes don't cause f-string syntax errors.

        Regression test for nested f-string issues with quotes in descriptions.
        """
        mock_bridge = Mock()
        mock_fastmcp.return_value = mock_bridge

        mock_client = AsyncMock()
        mock_tool = Mock()
        mock_tool.name = "greet"
        # Description with both single and double quotes
        mock_tool.description = "Say 'hello' to user with \"quotes\""
        mock_tool.inputSchema = {
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        mock_client.list_tools = AsyncMock(return_value=[mock_tool])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value = mock_client

        config = {
            "mcpServers": {
                "greeter": {
                    "command": "python",
                    "args": ["server.py"],
                    "transport": "stdio",
                    "enabled": True,
                },
            },
        }

        # Should not raise SyntaxError from f-string nesting
        await create_bridge(config)

        # Verify the docstring made it through
        registered_func = mock_bridge.tool.call_args[0][0]
        assert "hello" in registered_func.__doc__


class TestMixedParameterTypes:
    """Test tools with various parameter types."""

    @pytest.mark.asyncio
    @patch("mcp_bridge.server.Client")
    @patch("mcp_bridge.server.FastMCP")
    async def test_tools_with_all_json_types(
        self, mock_fastmcp: Mock, mock_client_class: Mock,
    ) -> None:
        """Test tools with string, number, integer, boolean parameters."""
        mock_bridge = Mock()
        mock_fastmcp.return_value = mock_bridge

        mock_client = AsyncMock()
        mock_tool = Mock()
        mock_tool.name = "calculate"
        mock_tool.description = "Perform calculation"
        mock_tool.inputSchema = {
            "properties": {
                "value": {"type": "number"},  # float
                "count": {"type": "integer"},  # int
                "enabled": {"type": "boolean"},  # bool
                "label": {"type": "string"},  # str
            },
            "required": ["value", "count"],  # Some required, some optional
        }
        mock_client.list_tools = AsyncMock(return_value=[mock_tool])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value = mock_client

        config = {
            "mcpServers": {
                "calc": {
                    "command": "python",
                    "args": ["server.py"],
                    "transport": "stdio",
                    "enabled": True,
                },
            },
        }

        await create_bridge(config)

        # Verify function signature has correct types and defaults
        registered_func = mock_bridge.tool.call_args[0][0]
        sig = inspect.signature(registered_func)

        # Check required parameters have no defaults
        assert sig.parameters["value"].default == inspect.Parameter.empty
        assert sig.parameters["count"].default == inspect.Parameter.empty

        # Check optional parameters have None defaults
        assert sig.parameters["enabled"].default is None
        assert sig.parameters["label"].default is None
