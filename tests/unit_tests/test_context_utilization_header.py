"""Unit tests for X-Context-Utilization header parsing."""

import pytest

from api.dependencies import parse_context_utilization_header
from api.context_manager import ContextUtilization


class TestContextUtilizationHeader:
    """Test X-Context-Utilization header parsing."""

    def test_defaults_to_full_when_missing(self) -> None:
        """Should default to FULL utilization when header not provided."""
        result = parse_context_utilization_header(None)
        assert result == ContextUtilization.FULL

    def test_parses_lowercase(self) -> None:
        """Should parse 'low', 'medium', 'full'."""
        assert parse_context_utilization_header("low") == ContextUtilization.LOW
        assert parse_context_utilization_header("medium") == ContextUtilization.MEDIUM
        assert parse_context_utilization_header("full") == ContextUtilization.FULL

    def test_parses_uppercase(self) -> None:
        """Should parse 'LOW', 'MEDIUM', 'FULL' (case-insensitive)."""
        assert parse_context_utilization_header("LOW") == ContextUtilization.LOW
        assert parse_context_utilization_header("MEDIUM") == ContextUtilization.MEDIUM
        assert parse_context_utilization_header("FULL") == ContextUtilization.FULL

    def test_parses_mixed_case(self) -> None:
        """Should parse 'Low', 'Medium', 'Full'."""
        assert parse_context_utilization_header("Low") == ContextUtilization.LOW
        assert parse_context_utilization_header("Medium") == ContextUtilization.MEDIUM
        assert parse_context_utilization_header("Full") == ContextUtilization.FULL

    def test_strips_whitespace(self) -> None:
        """Should handle ' low ', ' medium ', ' full '."""
        assert parse_context_utilization_header(" low ") == ContextUtilization.LOW
        assert parse_context_utilization_header(" medium ") == ContextUtilization.MEDIUM
        assert parse_context_utilization_header(" full ") == ContextUtilization.FULL

    def test_raises_on_invalid_value(self) -> None:
        """Should raise ValueError with helpful message for invalid values."""
        with pytest.raises(ValueError, match="Invalid context utilization: invalid"):
            parse_context_utilization_header("invalid")

        with pytest.raises(ValueError, match="Must be one of: low, medium, full"):
            parse_context_utilization_header("bad_value")

    def test_raises_on_empty_string(self) -> None:
        """Should raise ValueError for empty string."""
        with pytest.raises(ValueError, match="Invalid context utilization"):
            parse_context_utilization_header("")

    def test_error_message_includes_valid_values(self) -> None:
        """Error message should list all valid values."""
        try:
            parse_context_utilization_header("wrong")
        except ValueError as e:
            error_msg = str(e)
            assert "low" in error_msg
            assert "medium" in error_msg
            assert "full" in error_msg
