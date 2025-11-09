"""
MCP Bridge Server - Custom HTTP proxy for stdio MCP servers.

This bridge allows stdio-based MCP servers (like filesystem, mcp-this, etc.)
to be accessed via HTTP. It runs locally and spawns stdio server processes,
exposing their tools through a unified HTTP endpoint.

Architecture:
    API (Docker) --HTTP--> Bridge (localhost) --stdio--> MCP Servers (processes)

Implementation:
    - Uses FastMCP server for HTTP endpoint
    - Uses FastMCP Client for stdio connections
    - Dynamically registers tools from stdio servers
    - Manages server lifecycle (startup/shutdown)

Usage:
    uv run python mcp_bridge/server.py
    uv run python mcp_bridge/server.py --config path/to/config.json
    uv run python mcp_bridge/server.py --port 9000
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

from fastmcp import FastMCP, Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Global state for cleanup
_active_clients: list[Client] = []
_shutdown_event = asyncio.Event()


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load MCP server configuration from JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary with mcpServers

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid JSON or missing mcpServers
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}") from e

    if "mcpServers" not in config:
        raise ValueError("Config must contain 'mcpServers' key")

    return config


async def create_bridge(config: dict[str, Any], name: str = "MCP Bridge") -> FastMCP:
    """
    Create custom MCP bridge server.

    Connects to stdio servers and registers their tools on an HTTP server.

    Args:
        config: MCP server configuration dictionary
        name: Name for the bridge server

    Returns:
        FastMCP HTTP server instance with registered tools
    """
    logger.info(f"Creating custom bridge server: {name}")

    # Create HTTP server
    bridge = FastMCP(name=name)

    # Get server configs
    servers = config.get("mcpServers", {})
    enabled_servers = {
        name: cfg for name, cfg in servers.items()
        if cfg.get("enabled", True)
    }

    logger.info(f"Found {len(enabled_servers)} enabled servers")

    # Connect to each stdio server and register tools
    for server_name, server_config in enabled_servers.items():
        try:
            logger.info(f"Connecting to server: {server_name}")

            # Create client config
            client_config = {
                "mcpServers": {
                    server_name: server_config
                }
            }

            # Create client and connect
            client = Client(client_config)
            await client.__aenter__()  # Start client session
            _active_clients.append(client)  # Track for cleanup

            # List available tools
            tools = await client.list_tools()
            logger.info(f"  Found {len(tools)} tools from {server_name}")

            # Register each tool on bridge
            for tool in tools:
                # Create prefixed tool name
                # Sanitize names to be valid Python identifiers (replace hyphens/invalid chars)
                safe_server_name = server_name.replace("-", "_").replace(".", "_")
                safe_tool_name = tool.name.replace("-", "_").replace(".", "_")
                tool_name = f"{safe_server_name}_{safe_tool_name}"

                # Get input schema parameters
                input_schema = tool.inputSchema or {}
                properties = input_schema.get("properties", {})
                required_params = set(input_schema.get("required", []))

                # Create function with proper signature
                # Build parameter string for exec()
                params_list = []
                for param_name, param_schema in properties.items():
                    param_type = param_schema.get("type", "string")
                    # Map JSON types to Python types
                    type_map = {
                        "string": "str",
                        "number": "float",
                        "integer": "int",
                        "boolean": "bool",
                    }
                    py_type = type_map.get(param_type, "Any")

                    if param_name in required_params:
                        params_list.append(f"{param_name}: {py_type}")
                    else:
                        # Optional parameter with default
                        params_list.append(f"{param_name}: {py_type} = None")

                params_str = ", ".join(params_list) if params_list else ""

                # Create function dynamically with proper closure
                # Use exec to build function with correct signature
                kwargs_lines = "\n".join(
                    f'    if {p.split(":")[0].strip()} is not None: kwargs["{p.split(":")[0].strip()}"] = {p.split(":")[0].strip()}'
                    for p in params_list
                ) if params_list else "    pass"

                # Pre-compute description to avoid f-string nesting issues
                tool_description = tool.description or f"Tool from {server_name}"

                func_code = f"""
async def {tool_name}({params_str}) -> str:
    '''{tool_description}'''
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Calling tool: {tool_name}")
    # Build kwargs dict from parameters
    kwargs = {{}}
{kwargs_lines}
    try:
        result = await _client_ref.call_tool(_original_tool_name, kwargs)
        if hasattr(result, 'content') and result.content:
            return result.content[0].text
        return str(result)
    except Exception as e:
        logger.error(f"Tool call failed: {tool_name} - {{e}}")
        raise
"""

                # Execute the function definition
                namespace = {
                    "_client_ref": client,
                    "_original_tool_name": tool.name,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "Any": Any,
                }
                try:
                    exec(func_code, namespace)
                except SyntaxError as e:
                    logger.error(f"Syntax error in generated code for {tool_name}:\n{func_code}")
                    raise
                tool_func = namespace[tool_name]

                # Use decorator as function to register tool
                bridge.tool(tool_func)

                logger.debug(f"  Registered: {tool_name}")

        except Exception as e:
            logger.error(f"Failed to connect to {server_name}: {e}")
            # Continue with other servers

    logger.info("Bridge server created successfully")
    return bridge


async def cleanup() -> None:
    """Clean up active clients on shutdown."""
    logger.info("Cleaning up stdio clients...")
    for client in _active_clients:
        try:
            await client.__aexit__(None, None, None)
        except Exception as e:
            logger.error(f"Error closing client: {e}")
    _active_clients.clear()
    logger.info("Cleanup complete")


def handle_shutdown(signum: int, frame: Any) -> None:
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    _shutdown_event.set()


async def run_bridge(bridge: FastMCP, host: str, port: int) -> None:
    """
    Run the bridge server with proper lifecycle management.

    Args:
        bridge: Configured FastMCP server
        host: Host to bind to
        port: Port to listen on
    """
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        logger.info(f"Starting bridge server on {host}:{port}")
        logger.info(f"Bridge accessible at: http://{host}:{port}/mcp/")
        logger.info("Press Ctrl+C to stop")

        # Run bridge in background
        server_task = asyncio.create_task(
            bridge.run_http_async(host=host, port=port)
        )

        # Wait for shutdown signal
        await _shutdown_event.wait()

        # Cancel server
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    finally:
        await cleanup()


async def async_main(config_path: Path, host: str, port: int) -> None:
    """
    Async entry point.

    Args:
        config_path: Path to configuration file
        host: Host to bind to
        port: Port to listen on
    """
    # Load configuration
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # Create bridge server
    bridge = await create_bridge(config)

    # Run bridge
    await run_bridge(bridge, host, port)


def main() -> None:
    """Run the MCP bridge server."""
    parser = argparse.ArgumentParser(
        description="MCP Bridge - Custom HTTP proxy for stdio MCP servers"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("mcp_bridge/config.json"),
        help="Path to MCP server configuration file (default: mcp_bridge/config.json)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MCP_BRIDGE_PORT", "9000")),
        help="Port to listen on (default: 9000, env: MCP_BRIDGE_PORT)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("MCP_BRIDGE_HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0, env: MCP_BRIDGE_HOST)",
    )

    args = parser.parse_args()

    try:
        asyncio.run(async_main(args.config, args.host, args.port))
    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down bridge server...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
