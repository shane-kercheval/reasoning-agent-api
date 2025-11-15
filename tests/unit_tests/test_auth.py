"""
Tests for authentication system.

Tests bearer token authentication, token validation, and integration with
FastAPI endpoints. Covers both unit tests for auth functions and integration
tests for protected endpoints.
"""

import pytest
import respx
import httpx
from fastapi.testclient import TestClient
from fastapi.security import HTTPAuthorizationCredentials
from fastapi import HTTPException

from api.auth import verify_token, get_configured_token_count
from api.config import settings
from api.main import app


class TestAuthFunctions:
    """Test authentication functions in isolation."""

    @pytest.mark.asyncio
    async def test__verify_token__allows_access_when_auth_disabled(self):
        """Test that verify_token allows access when authentication is disabled."""
        # Save original state
        original_require_auth = settings.require_auth

        try:
            # Disable authentication
            settings.require_auth = False

            # Should return True regardless of credentials
            result = await verify_token(None)
            assert result is True

            # Should also work with invalid credentials
            fake_creds = HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials="invalid-token",
            )
            result = await verify_token(fake_creds)
            assert result is True
        finally:
            # Restore original state
            settings.require_auth = original_require_auth

    @pytest.mark.asyncio
    async def test__verify_token__raises_500_when_no_tokens_configured(self):
        """Test that verify_token raises 500 when no tokens are configured."""
        # Save original state
        original_require_auth = settings.require_auth
        original_tokens = settings.api_tokens

        try:
            # Enable auth but provide no tokens
            settings.require_auth = True
            settings.api_tokens = ""

            with pytest.raises(HTTPException) as exc_info:
                await verify_token(None)

            assert exc_info.value.status_code == 500
            assert "no_tokens_configured" in str(exc_info.value.detail)
        finally:
            # Restore original state
            settings.require_auth = original_require_auth
            settings.api_tokens = original_tokens

    @pytest.mark.asyncio
    async def test__verify_token__raises_401_when_no_credentials_provided(self):
        """Test that verify_token raises 401 when no credentials are provided."""
        # Save original state
        original_require_auth = settings.require_auth
        original_tokens = settings.api_tokens

        try:
            # Enable auth with valid tokens
            settings.require_auth = True
            settings.api_tokens = "valid-token-1,valid-token-2"

            with pytest.raises(HTTPException) as exc_info:
                await verify_token(None)

            assert exc_info.value.status_code == 401
            assert "missing_token" in str(exc_info.value.detail)
        finally:
            # Restore original state
            settings.require_auth = original_require_auth
            settings.api_tokens = original_tokens

    @pytest.mark.asyncio
    async def test__verify_token__raises_401_when_invalid_token_provided(self):
        """Test that verify_token raises 401 when invalid token is provided."""
        # Save original state
        original_require_auth = settings.require_auth
        original_tokens = settings.api_tokens

        try:
            # Enable auth with valid tokens
            settings.require_auth = True
            settings.api_tokens = "valid-token-1,valid-token-2"

            invalid_creds = HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials="invalid-token",
            )

            with pytest.raises(HTTPException) as exc_info:
                await verify_token(invalid_creds)

            assert exc_info.value.status_code == 401
            assert "invalid_token" in str(exc_info.value.detail)
        finally:
            # Restore original state
            settings.require_auth = original_require_auth
            settings.api_tokens = original_tokens

    @pytest.mark.asyncio
    async def test__verify_token__allows_access_with_valid_token(self):
        """Test that verify_token allows access with valid token."""
        # Save original state
        original_require_auth = settings.require_auth
        original_tokens = settings.api_tokens

        try:
            # Enable auth with valid tokens
            settings.require_auth = True
            settings.api_tokens = "valid-token-1,valid-token-2"

            valid_creds = HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials="valid-token-1",
            )

            result = await verify_token(valid_creds)
            assert result is True
        finally:
            # Restore original state
            settings.require_auth = original_require_auth
            settings.api_tokens = original_tokens

    @pytest.mark.asyncio
    async def test__verify_token__handles_whitespace_in_tokens(self):
        """Test that verify_token handles whitespace correctly in token configuration."""
        # Save original state
        original_require_auth = settings.require_auth
        original_tokens = settings.api_tokens

        try:
            # Enable auth with tokens that have whitespace
            settings.require_auth = True
            settings.api_tokens = " token1 , token2 ,token3, "

            valid_creds = HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials="token2",  # Should work despite whitespace in config
            )

            result = await verify_token(valid_creds)
            assert result is True
        finally:
            # Restore original state
            settings.require_auth = original_require_auth
            settings.api_tokens = original_tokens


    def test__get_configured_token_count__returns_correct_count(self):
        """Test that get_configured_token_count returns correct count."""
        # Save original state
        original_tokens = settings.api_tokens

        try:
            # Test no tokens
            settings.api_tokens = ""
            assert get_configured_token_count() == 0

            # Test single token
            settings.api_tokens = "token1"
            assert get_configured_token_count() == 1

            # Test multiple tokens
            settings.api_tokens = "token1,token2,token3"
            assert get_configured_token_count() == 3

            # Test tokens with whitespace
            settings.api_tokens = " token1 , token2 "
            assert get_configured_token_count() == 2
        finally:
            # Restore original state
            settings.api_tokens = original_tokens


class TestAuthenticationIntegration:
    """Test authentication integration with FastAPI endpoints."""

    def test__protected_endpoints__require_auth_when_enabled(self):
        """Test that protected endpoints require authentication when enabled."""
        # Save original state
        original_require_auth = settings.require_auth
        original_tokens = settings.api_tokens

        try:
            # Enable auth with valid tokens
            settings.require_auth = True
            settings.api_tokens = "test-token-123"

            with TestClient(app) as client:
                # Test chat completions endpoint
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": "test"}],
                    },
                )
                assert response.status_code == 401

                # Test models endpoint
                response = client.get("/v1/models")
                assert response.status_code == 401

                # Test tools endpoint
                response = client.get("/v1/mcp/tools")
                assert response.status_code == 401

                # Health endpoint should still be public
                response = client.get("/health")
                assert response.status_code == 200
        finally:
            # Restore original state
            settings.require_auth = original_require_auth
            settings.api_tokens = original_tokens

    @respx.mock
    def test__protected_endpoints__allow_access_with_valid_token(self, respx_mock: respx.Router):
        """Test that protected endpoints allow access with valid token."""
        # Mock LiteLLM /v1/models endpoint
        respx_mock.get(f"{settings.llm_base_url}/v1/models").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"id": "gpt-4o", "created": 1234567890, "owned_by": "openai"}]},
            ),
        )

        # Save original state
        original_require_auth = settings.require_auth
        original_tokens = settings.api_tokens

        try:
            # Enable auth with valid tokens
            settings.require_auth = True
            settings.api_tokens = "test-token-123"

            with TestClient(app) as client:
                headers = {"Authorization": "Bearer test-token-123"}

                # Test models endpoint (should work with auth)
                response = client.get("/v1/models", headers=headers)
                assert response.status_code == 200

                # Test tools endpoint (should work with auth)
                response = client.get("/v1/mcp/tools", headers=headers)
                assert response.status_code == 200

                # Chat completions would need mocked reasoning agent,
                # but we can test that auth doesn't block it
                # (it would fail later due to missing reasoning agent setup)
        finally:
            # Restore original state
            settings.require_auth = original_require_auth
            settings.api_tokens = original_tokens

    @respx.mock
    def test__protected_endpoints__work_when_auth_disabled(self, respx_mock: respx.Router):
        """Test that protected endpoints work when authentication is disabled."""
        # Mock LiteLLM /v1/models endpoint
        respx_mock.get(f"{settings.llm_base_url}/v1/models").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"id": "gpt-4o", "created": 1234567890, "owned_by": "openai"}]},
            ),
        )

        # Save original state
        original_require_auth = settings.require_auth

        try:
            # Disable authentication
            settings.require_auth = False

            with TestClient(app) as client:
                # Test models endpoint (should work without auth)
                response = client.get("/v1/models")
                assert response.status_code == 200

                # Test tools endpoint (should work without auth)
                response = client.get("/v1/mcp/tools")
                assert response.status_code == 200

                # Test health endpoint (should always work)
                response = client.get("/health")
                assert response.status_code == 200
        finally:
            # Restore original state
            settings.require_auth = original_require_auth

    def test__invalid_token_format__returns_proper_error(self):
        """Test that invalid token format returns proper error structure."""
        # Save original state
        original_require_auth = settings.require_auth
        original_tokens = settings.api_tokens

        try:
            # Enable auth with valid tokens
            settings.require_auth = True
            settings.api_tokens = "valid-token"

            with TestClient(app) as client:
                headers = {"Authorization": "Bearer invalid-token"}

                response = client.get("/v1/models", headers=headers)
                assert response.status_code == 401

                data = response.json()
                assert "detail" in data
                assert "error" in data["detail"]
                assert data["detail"]["error"]["code"] == "invalid_token"
                assert "authentication_error" in data["detail"]["error"]["type"]
        finally:
            # Restore original state
            settings.require_auth = original_require_auth
            settings.api_tokens = original_tokens

    def test__malformed_authorization_header__returns_proper_error(self):
        """Test that malformed authorization headers return proper errors."""
        # Save original state
        original_require_auth = settings.require_auth
        original_tokens = settings.api_tokens

        try:
            # Enable auth with valid tokens
            settings.require_auth = True
            settings.api_tokens = "valid-token"

            with TestClient(app) as client:
                # Test malformed header (missing Bearer prefix)
                headers = {"Authorization": "invalid-token"}
                response = client.get("/v1/models", headers=headers)
                assert response.status_code == 401

                # Test empty authorization header
                headers = {"Authorization": ""}
                response = client.get("/v1/models", headers=headers)
                assert response.status_code == 401

                # Test wrong auth scheme
                headers = {"Authorization": "Basic dGVzdA=="}
                response = client.get("/v1/models", headers=headers)
                assert response.status_code == 401
        finally:
            # Restore original state
            settings.require_auth = original_require_auth
            settings.api_tokens = original_tokens


class TestAuthEdgeCases:
    """Test edge cases and error conditions for authentication."""

    @pytest.mark.asyncio
    async def test__empty_token_in_list__is_ignored(self):
        """Test that empty tokens in the token list are ignored."""
        # Save original state
        original_require_auth = settings.require_auth
        original_tokens = settings.api_tokens

        try:
            # Configure with empty tokens mixed in
            settings.require_auth = True
            settings.api_tokens = "valid1,,valid2,,"

            # Should only have 2 valid tokens
            assert get_configured_token_count() == 2

            valid_creds = HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials="valid2",
            )

            # Should work with valid token
            result = await verify_token(valid_creds)
            assert result is True
        finally:
            # Restore original state
            settings.require_auth = original_require_auth
            settings.api_tokens = original_tokens

    def test__only_whitespace_tokens__are_filtered_out(self):
        """Test that tokens with only whitespace are filtered out."""
        # Save original state
        original_tokens = settings.api_tokens

        try:
            # Configure with only whitespace tokens
            settings.api_tokens = "  ,   ,\t,\n"

            # Should have no valid tokens
            assert get_configured_token_count() == 0
            assert settings.allowed_tokens == []
        finally:
            # Restore original state
            settings.api_tokens = original_tokens
