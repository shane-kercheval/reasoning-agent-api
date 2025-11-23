"""
Unit tests for MCP name parsing logic.

Tests the core parsing functions: parse_mcp_name() and validate_tag()
"""

import pytest
from api.mcp import parse_mcp_name, validate_tag, NamingConfig


class TestValidateTag:
    """Test tag validation rules."""

    @pytest.mark.parametrize("valid_tag", [
        "git",
        "pull-request",
        "code-review",
        "a",
        "a1",
        "version-2",
        "my-tag-123",
        "web",
        "github",
        "1",
        "123",
        "a-b-c-d-e-f",
    ])
    def test_valid_tags(self, valid_tag: str) -> None:
        """Test that valid tags pass validation."""
        # Should not raise
        validate_tag(valid_tag)

    @pytest.mark.parametrize(("invalid_tag", "error_substring"), [
        ("Git", "lowercase"),
        ("pull request", "lowercase"),
        ("pull_request", "lowercase"),
        ("-git", "alphanumeric"),
        ("git-", "alphanumeric"),
        ("my tag", "lowercase"),
        ("TAG", "lowercase"),
        ("a_b", "lowercase"),
        ("", "lowercase"),
        ("123-", "alphanumeric"),
        ("-123", "alphanumeric"),
    ])
    def test_invalid_tags(self, invalid_tag: str, error_substring: str) -> None:
        """Test that invalid tags raise ValueError with helpful message."""
        with pytest.raises(ValueError, match="Invalid tag format") as exc_info:
            validate_tag(invalid_tag)
        assert error_substring in str(exc_info.value).lower()


class TestParseMCPName:
    """Test MCP name parsing logic."""

    @pytest.mark.parametrize(("raw_name", "expected_base", "expected_server"), [
        # Basic server__name pattern
        ("github_custom__get_pr_info", "get_pr_info", "github-custom"),
        ("filesystem__read_file", "read_file", "filesystem"),
        ("meta__generate_playbook", "generate_playbook", "meta"),

        # No server prefix (simple names)
        ("simple_tool", "simple_tool", None),
        ("my_function", "my_function", None),

        # Multiple underscores - last __ is separator
        # Note: a__b becomes "a--b" because each __ in server part stays as __
        ("a__b__c", "c", "a--b"),
        # Last __ separates server from tool name
        ("server_name__tool__subtool", "subtool", "server-name--tool"),
    ])
    def test_parse_basic_patterns(
        self,
        raw_name: str,
        expected_base: str,
        expected_server: str | None,
    ) -> None:
        """Test parsing of various MCP name patterns without prefix stripping."""
        config = NamingConfig()
        result = parse_mcp_name(raw_name, config, "tool")

        assert result.base_name == expected_base
        assert result.server_name == expected_server
        assert result.mcp_name == raw_name
        assert result.tags == []

    def test_server_name_normalization(self) -> None:
        """Test server name normalization (underscore to hyphen)."""
        config = NamingConfig()

        test_cases = [
            ("github_custom__tool", "github-custom"),
            ("my_server_name__tool", "my-server-name"),
            ("a_b_c__tool", "a-b-c"),
        ]

        for raw_name, expected_server in test_cases:
            result = parse_mcp_name(raw_name, config, "tool")
            assert result.server_name == expected_server

    def test_parse_with_server_tags(self) -> None:
        """Test that server tags are applied from configuration."""
        config = NamingConfig(
            server_tags={
                "github-custom": ["git", "github", "version-control"],
                "filesystem": ["files", "storage"],
            },
        )

        # GitHub tool gets github tags
        result = parse_mcp_name("github_custom__get_pr", config, "tool")
        assert result.tags == ["git", "github", "version-control"]

        # Filesystem tool gets filesystem tags
        result = parse_mcp_name("filesystem__read", config, "tool")
        assert result.tags == ["files", "storage"]

        # Unknown server gets no tags
        result = parse_mcp_name("unknown__tool", config, "tool")
        assert result.tags == []

    def test_parse_with_override_exact_match(self) -> None:
        """Test that override takes priority over auto-parsing."""
        config = NamingConfig(
            server_tags={
                "github-custom": ["git", "github"],
            },
            tool_overrides={
                "github_custom__get_pr": {
                    "name": "get_pr_info",
                    "tags": ["pull-request", "code-review"],
                },
            },
        )

        result = parse_mcp_name("github_custom__get_pr", config, "tool")

        # Override name used
        assert result.base_name == "get_pr_info"

        # Server tags + override tags merged
        assert set(result.tags) == {"git", "github", "pull-request", "code-review"}

        # Server name still extracted
        assert result.server_name == "github-custom"

    def test_override_for_prompts(self) -> None:
        """Test that prompt overrides are separate from tool overrides."""
        config = NamingConfig(
            tool_overrides={
                "tool_name": {"name": "renamed_tool"},
            },
            prompt_overrides={
                "prompt_name": {"name": "renamed_prompt"},
            },
        )

        # Tool override applies to tools
        result = parse_mcp_name("tool_name", config, "tool")
        assert result.base_name == "renamed_tool"

        # Prompt override applies to prompts
        result = parse_mcp_name("prompt_name", config, "prompt")
        assert result.base_name == "renamed_prompt"

        # Tool override doesn't apply to prompts
        result = parse_mcp_name("tool_name", config, "prompt")
        assert result.base_name == "tool_name"

    def test_tag_deduplication(self) -> None:
        """Test that duplicate tags are removed while preserving order."""
        config = NamingConfig(
            server_tags={
                "github-custom": ["git", "github"],
            },
            tool_overrides={
                "github_custom__tool": {
                    "name": "tool",
                    "tags": ["git", "pull-request"],  # "git" is duplicate
                },
            },
        )

        result = parse_mcp_name("github_custom__tool", config, "tool")

        # Should have: git (from server), github (from server), pull-request (from override)
        # "git" appears only once
        assert result.tags == ["git", "github", "pull-request"]

    def test_tag_validation_in_parsing(self) -> None:
        """Test that invalid tags in config cause parsing to fail."""
        config = NamingConfig(
            server_tags={
                "github-custom": ["Git", "invalid tag"],  # Invalid tags
            },
        )

        with pytest.raises(ValueError, match="Invalid tag format"):
            parse_mcp_name("github_custom__tool", config, "tool")

    def test_override_without_server_pattern(self) -> None:
        """Test override for tools without server__name pattern."""
        config = NamingConfig(
            tool_overrides={
                "simple_tool": {
                    "name": "renamed_simple",
                    "tags": ["utility"],
                },
            },
        )

        result = parse_mcp_name("simple_tool", config, "tool")

        assert result.base_name == "renamed_simple"
        assert result.server_name is None
        assert result.tags == ["utility"]

    def test_no_override_uses_auto_parsing(self) -> None:
        """Test that auto-parsing is used when no override matches."""
        config = NamingConfig(
            tool_overrides={
                "other_tool": {"name": "renamed"},
            },
        )

        # This tool has no override, should use auto-parsing
        result = parse_mcp_name("github__my_tool", config, "tool")

        assert result.base_name == "my_tool"
        assert result.server_name == "github"
        assert result.tags == []

    def test_disable_flag_in_override(self) -> None:
        """Test that disable flag is correctly parsed from override."""
        config = NamingConfig(
            tool_overrides={
                "unwanted_tool": {"disable": True},
                "wanted_tool": {"disable": False},
                "normal_tool": {"name": "renamed"},
            },
        )

        # Tool with disable: true
        result = parse_mcp_name("unwanted_tool", config, "tool")
        assert result.disabled is True
        assert result.base_name == "unwanted_tool"

        # Tool with explicit disable: false
        result = parse_mcp_name("wanted_tool", config, "tool")
        assert result.disabled is False

        # Tool without disable field (should default to False)
        result = parse_mcp_name("normal_tool", config, "tool")
        assert result.disabled is False

    def test_disable_without_name_override(self) -> None:
        """Test that you can disable a tool without renaming it."""
        config = NamingConfig(
            tool_overrides={
                "github__unwanted": {"disable": True},
            },
        )

        result = parse_mcp_name("github__unwanted", config, "tool")
        assert result.disabled is True
        assert result.base_name == "unwanted"
        assert result.server_name == "github"

    def test_auto_parsed_tools_not_disabled(self) -> None:
        """Test that auto-parsed tools (no override) default to not disabled."""
        config = NamingConfig()

        result = parse_mcp_name("github__my_tool", config, "tool")
        assert result.disabled is False
