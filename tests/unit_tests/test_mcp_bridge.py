"""
Unit tests for MCP Bridge server.

Tests configuration loading, validation, and error handling without spawning
actual server processes.
"""

import inspect
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

        # Function name should be sanitized: echo_server__echo_message
        assert registered_func.__name__ == "echo_server__echo_message"
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
        assert registered_func.__name__ == "fs_server__io_read"
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


class TestPromptRegistration:
    """Test prompt registration from stdio servers."""

    @pytest.mark.asyncio
    @patch("mcp_bridge.server.Client")
    @patch("mcp_bridge.server.FastMCP")
    async def test_create_bridge_registers_prompts(
        self, mock_fastmcp: Mock, mock_client_class: Mock,
    ) -> None:
        """Test that prompts from stdio servers are registered on the bridge."""
        mock_bridge = Mock()
        mock_fastmcp.return_value = mock_bridge

        mock_client = AsyncMock()
        # Mock tools (empty for this test)
        mock_client.list_tools = AsyncMock(return_value=[])

        # Mock prompts
        mock_prompt = Mock()
        mock_prompt.name = "generate_playbook"
        mock_prompt.description = "Generate a comprehensive playbook"
        # Create argument mocks with .name and .required attributes
        arg1 = Mock()
        arg1.name = "topic"
        arg1.required = True
        arg2 = Mock()
        arg2.name = "instructions"
        arg2.required = False
        mock_prompt.arguments = [arg1, arg2]
        mock_client.list_prompts = AsyncMock(return_value=[mock_prompt])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value = mock_client

        config = {
            "mcpServers": {
                "meta": {
                    "command": "uvx",
                    "args": ["mcp-this", "--config", "meta.yaml"],
                    "transport": "stdio",
                    "enabled": True,
                },
            },
        }

        bridge = await create_bridge(config)

        # Verify prompts were listed
        mock_client.list_prompts.assert_called_once()

        # Verify bridge.prompt was called to register the prompt
        mock_bridge.prompt.assert_called_once()

        # Verify the registered function has correct name
        registered_func = mock_bridge.prompt.call_args[0][0]
        assert registered_func.__name__ == "meta__generate_playbook"

        assert bridge == mock_bridge

    @pytest.mark.asyncio
    @patch("mcp_bridge.server.Client")
    @patch("mcp_bridge.server.FastMCP")
    async def test_prompt_with_required_and_optional_arguments(
        self, mock_fastmcp: Mock, mock_client_class: Mock,
    ) -> None:
        """Test prompt registration with both required and optional arguments."""
        mock_bridge = Mock()
        mock_fastmcp.return_value = mock_bridge

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])

        mock_prompt = Mock()
        mock_prompt.name = "analyze"
        mock_prompt.description = "Analyze content"
        # Create argument mocks with .name and .required attributes
        arg1 = Mock()
        arg1.name = "content"
        arg1.required = True
        arg2 = Mock()
        arg2.name = "depth"
        arg2.required = False
        arg3 = Mock()
        arg3.name = "format"
        arg3.required = False
        mock_prompt.arguments = [arg1, arg2, arg3]
        mock_client.list_prompts = AsyncMock(return_value=[mock_prompt])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value = mock_client

        config = {
            "mcpServers": {
                "thinking": {
                    "command": "uvx",
                    "args": ["mcp-this"],
                    "transport": "stdio",
                    "enabled": True,
                },
            },
        }

        await create_bridge(config)

        # Verify function signature
        registered_func = mock_bridge.prompt.call_args[0][0]
        sig = inspect.signature(registered_func)

        # Required parameter has no default
        assert sig.parameters["content"].default == inspect.Parameter.empty

        # Optional parameters have None defaults
        assert sig.parameters["depth"].default is None
        assert sig.parameters["format"].default is None

    @pytest.mark.asyncio
    @patch("mcp_bridge.server.Client")
    @patch("mcp_bridge.server.FastMCP")
    async def test_prompt_with_no_arguments(
        self, mock_fastmcp: Mock, mock_client_class: Mock,
    ) -> None:
        """Test prompt registration with no arguments."""
        mock_bridge = Mock()
        mock_fastmcp.return_value = mock_bridge

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])

        mock_prompt = Mock()
        mock_prompt.name = "hello"
        mock_prompt.description = "Simple greeting prompt"
        mock_prompt.arguments = []
        mock_client.list_prompts = AsyncMock(return_value=[mock_prompt])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value = mock_client

        config = {
            "mcpServers": {
                "greetings": {
                    "command": "python",
                    "args": ["server.py"],
                    "transport": "stdio",
                    "enabled": True,
                },
            },
        }

        await create_bridge(config)

        # Verify function has no parameters
        registered_func = mock_bridge.prompt.call_args[0][0]
        sig = inspect.signature(registered_func)
        assert len(sig.parameters) == 0

    @pytest.mark.asyncio
    @patch("mcp_bridge.server.Client")
    @patch("mcp_bridge.server.FastMCP")
    async def test_prompt_name_sanitization(
        self, mock_fastmcp: Mock, mock_client_class: Mock,
    ) -> None:
        """Test that hyphens and dots in prompt names are converted to underscores."""
        mock_bridge = Mock()
        mock_fastmcp.return_value = mock_bridge

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])

        mock_prompt = Mock()
        mock_prompt.name = "generate-detailed-analysis"  # Hyphens in name
        mock_prompt.description = "Generate analysis"
        mock_prompt.arguments = []
        mock_client.list_prompts = AsyncMock(return_value=[mock_prompt])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value = mock_client

        config = {
            "mcpServers": {
                "meta-prompts": {  # Hyphen in server name
                    "command": "uvx",
                    "args": ["mcp-this"],
                    "transport": "stdio",
                    "enabled": True,
                },
            },
        }

        await create_bridge(config)

        # Verify name sanitization (hyphens -> underscores)
        registered_func = mock_bridge.prompt.call_args[0][0]
        assert registered_func.__name__ == "meta_prompts__generate_detailed_analysis"
        assert "-" not in registered_func.__name__

    @pytest.mark.asyncio
    @patch("mcp_bridge.server.Client")
    @patch("mcp_bridge.server.FastMCP")
    async def test_multiple_prompts_registered(
        self, mock_fastmcp: Mock, mock_client_class: Mock,
    ) -> None:
        """Test registration of multiple prompts from a single server."""
        mock_bridge = Mock()
        mock_fastmcp.return_value = mock_bridge

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])

        # Multiple prompts
        mock_prompt1 = Mock()
        mock_prompt1.name = "prompt_one"
        mock_prompt1.description = "First prompt"
        mock_prompt1.arguments = []

        mock_prompt2 = Mock()
        mock_prompt2.name = "prompt_two"
        mock_prompt2.description = "Second prompt"
        mock_prompt2.arguments = []

        mock_client.list_prompts = AsyncMock(return_value=[mock_prompt1, mock_prompt2])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value = mock_client

        config = {
            "mcpServers": {
                "multi": {
                    "command": "python",
                    "args": ["server.py"],
                    "transport": "stdio",
                    "enabled": True,
                },
            },
        }

        await create_bridge(config)

        # Should register both prompts
        assert mock_bridge.prompt.call_count == 2

    @pytest.mark.asyncio
    @patch("mcp_bridge.server.Client")
    @patch("mcp_bridge.server.FastMCP")
    async def test_prompts_and_tools_registered_together(
        self, mock_fastmcp: Mock, mock_client_class: Mock,
    ) -> None:
        """Test that both tools and prompts are registered from the same server."""
        mock_bridge = Mock()
        mock_fastmcp.return_value = mock_bridge

        mock_client = AsyncMock()

        # Mock tool
        mock_tool = Mock()
        mock_tool.name = "read_file"
        mock_tool.description = "Read a file"
        mock_tool.inputSchema = {"properties": {}, "required": []}
        mock_client.list_tools = AsyncMock(return_value=[mock_tool])

        # Mock prompt
        mock_prompt = Mock()
        mock_prompt.name = "analyze_file"
        mock_prompt.description = "Analyze file content"
        mock_prompt.arguments = []
        mock_client.list_prompts = AsyncMock(return_value=[mock_prompt])

        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value = mock_client

        config = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["@modelcontextprotocol/server-filesystem"],
                    "transport": "stdio",
                    "enabled": True,
                },
            },
        }

        await create_bridge(config)

        # Both tool and prompt should be registered
        mock_bridge.tool.assert_called_once()
        mock_bridge.prompt.assert_called_once()

    @pytest.mark.asyncio
    @patch("mcp_bridge.server.Client")
    @patch("mcp_bridge.server.FastMCP")
    async def test_prompt_with_quotes_in_description(
        self, mock_fastmcp: Mock, mock_client_class: Mock,
    ) -> None:
        """Test that prompts with quotes in descriptions don't cause syntax errors."""
        mock_bridge = Mock()
        mock_fastmcp.return_value = mock_bridge

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])

        mock_prompt = Mock()
        mock_prompt.name = "greet"
        # Description with both single and double quotes
        mock_prompt.description = "Say 'hello' to user with \"quotes\""
        mock_prompt.arguments = []
        mock_client.list_prompts = AsyncMock(return_value=[mock_prompt])
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

        # Should not raise SyntaxError
        await create_bridge(config)

        # Verify the docstring made it through
        registered_func = mock_bridge.prompt.call_args[0][0]
        assert "hello" in registered_func.__doc__
