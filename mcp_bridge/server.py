"""
MCP Bridge Server - HTTP Proxy for stdio MCP servers.

This bridge allows stdio-based MCP servers (like filesystem, mcp-this, etc.)
to be accessed via HTTP. It runs locally and spawns stdio server processes,
exposing their tools through a unified HTTP endpoint.

Architecture:
    API (Docker) --HTTP--> Bridge (localhost) --stdio--> MCP Servers (processes)

Usage:
    uv run python mcp_bridge/server.py
    uv run python mcp_bridge/server.py --config path/to/config.json
    uv run python mcp_bridge/server.py --port 9000
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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


def create_bridge(config: dict[str, Any], name: str = "MCP Bridge") -> FastMCP:
    """
    Create MCP bridge server from configuration.

    Args:
        config: MCP server configuration dictionary
        name: Name for the bridge server

    Returns:
        FastMCP proxy server instance
    """
    logger.info(f"Creating bridge server: {name}")
    logger.info(f"Found {len(config.get('mcpServers', {}))} configured servers")

    for server_name in config.get("mcpServers", {}).keys():
        logger.info(f"  - {server_name}")

    # Create proxy server using FastMCP
    bridge = FastMCP.as_proxy(config, name=name)

    logger.info("Bridge server created successfully")
    return bridge


def main() -> None:
    """Run the MCP bridge server."""
    parser = argparse.ArgumentParser(description="MCP Bridge - HTTP proxy for stdio MCP servers")
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
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)

        # Create bridge server
        bridge = create_bridge(config)

        # Run as HTTP server
        logger.info(f"Starting bridge server on {args.host}:{args.port}")
        logger.info(f"Bridge accessible at: http://{args.host}:{args.port}/mcp/")
        logger.info("Press Ctrl+C to stop")

        bridge.run(transport="http", host=args.host, port=args.port)

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
