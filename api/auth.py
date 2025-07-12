"""
Authentication module for the Reasoning Agent API.

Provides simple bearer token authentication for protecting API endpoints.
Supports enabling/disabling authentication and configurable token lists.
"""

from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .config import settings

# Security scheme for bearer token authentication
security = HTTPBearer(auto_error=False)


async def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> bool:
    """
    Verify bearer token authentication.

    Validates that the provided bearer token is in the list of allowed tokens.
    Can be disabled for development by setting REQUIRE_AUTH=false.

    Args:
        credentials: Bearer token credentials from the Authorization header.

    Returns:
        True if authentication is successful or disabled.

    Raises:
        HTTPException: 401 if authentication fails or token is invalid.
        HTTPException: 500 if no tokens are configured when auth is required.

    Example:
        >>> # In endpoint
        >>> @app.get("/protected")
        >>> async def protected_endpoint(
        ...     _: bool = Depends(verify_token)
        ... ):
        ...     return {"message": "Access granted"}
    """
    # Skip authentication if disabled (useful for development)
    if not settings.require_auth:
        return True

    # Check if any tokens are configured
    allowed_tokens = settings.allowed_tokens
    if not allowed_tokens:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "No API tokens configured on server",
                    "type": "server_configuration_error",
                    "code": "no_tokens_configured",
                },
            },
        )

    # Check if credentials were provided
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Missing authentication token",
                    "type": "authentication_error",
                    "code": "missing_token",
                },
            },
        )

    # Verify token is in allowed list
    if credentials.credentials not in allowed_tokens:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid authentication token",
                    "type": "authentication_error",
                    "code": "invalid_token",
                },
            },
        )

    return True


def is_auth_enabled() -> bool:
    """
    Check if authentication is enabled.

    Returns:
        True if authentication is required, False otherwise.

    Example:
        >>> if is_auth_enabled():
        ...     print("Authentication is required")
    """
    return settings.require_auth


def get_configured_token_count() -> int:
    """
    Get the number of configured authentication tokens.

    Returns:
        Number of valid tokens configured.

    Example:
        >>> count = get_configured_token_count()
        >>> print(f"Server has {count} valid tokens configured")
    """
    return len(settings.allowed_tokens)
