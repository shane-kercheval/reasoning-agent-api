"""
Logging configuration for the Reasoning Agent API.

This module configures Python's warning system and logging to suppress
harmless warnings from third-party libraries while preserving important logs.
"""

import warnings
import logging


def configure_warnings() -> None:
    """
    Configure warning filters to suppress harmless third-party warnings.

    These warnings are from dependencies (aiohttp, litellm, opentelemetry)
    and don't indicate actual problems with our code.
    """
    # Suppress aiohttp deprecation warning (fixed in Python 3.13)
    warnings.filterwarnings(
        "ignore",
        message="enable_cleanup_closed ignored because",
        category=DeprecationWarning,
        module="aiohttp.connector",
    )

    # Suppress LiteLLM Pydantic deprecation warning (third-party issue)
    warnings.filterwarnings(
        "ignore",
        message="Accessing the 'model_fields' attribute on the instance is deprecated",
        category=Warning,
        module="litellm.litellm_core_utils.core_helpers",
    )


def configure_logging() -> None:
    """
    Configure logging to reduce verbosity of known harmless errors.

    Sets up filters and handlers to suppress OpenTelemetry context detach errors
    and reduce noise from MCP connection failures (which are expected when MCP
    servers are disabled).
    """

    class ContextDetachFilter(logging.Filter):
        """Filter out OpenTelemetry context detach errors."""

        def filter(self, record: logging.LogRecord) -> bool:
            # Filter out the known OpenTelemetry context error
            msg = record.getMessage()
            if "Failed to detach context" in msg:
                return False
            if "was created in a different Context" in msg:
                return False

            # Also check exception info if present
            if record.exc_info:
                exc_type, exc_value, _ = record.exc_info
                if exc_type and exc_value:
                    exc_str = str(exc_value)
                    if "was created in a different Context" in exc_str:
                        return False

            return True

    # Add filter to root logger to catch stderr output
    root_logger = logging.getLogger()
    root_logger.addFilter(ContextDetachFilter())

    # Add filter specifically to OpenTelemetry context logger
    # (this is where the detach errors are actually logged)
    otel_context_logger = logging.getLogger("opentelemetry.context")
    otel_context_logger.addFilter(ContextDetachFilter())
    otel_context_logger.setLevel(logging.ERROR)  # Suppress lower-level messages

    # Reduce MCP warning verbosity (expected when servers are disabled)
    mcp_logger = logging.getLogger("fastmcp")
    if mcp_logger:
        # Don't suppress completely, but keep them at WARNING level
        # Users can set MCP_LOG_LEVEL=ERROR to hide completely
        pass


def initialize_logging() -> None:
    """
    Initialize all logging and warning configurations.

    Should be called once during application startup, before any
    other code that might generate warnings or logs.
    """
    configure_warnings()
    configure_logging()
