"""
Unit tests for MCP naming override configuration loading.

Tests NamingConfig class and YAML loading functionality.
"""

import pytest
from pathlib import Path
from api.mcp import NamingConfig


class TestNamingConfigLoading:
    """Test loading and parsing of override configuration."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """Test loading valid YAML override config."""
        config_content = """
server_tags:
  github-custom: [git, github]
  filesystem: [files]

tools:
  "local_bridge_github__get_pr":
    name: get_pr_info
    tags: [pull-request]

prompts:
  "local_bridge_meta__generate":
    name: generate_playbook
    tags: [automation]
"""
        config_file = tmp_path / "mcp_overrides.yaml"
        config_file.write_text(config_content)

        config = NamingConfig.load_from_yaml(config_file)

        assert config.server_tags == {
            "github-custom": ["git", "github"],
            "filesystem": ["files"],
        }
        assert "local_bridge_github__get_pr" in config.tool_overrides
        assert config.tool_overrides["local_bridge_github__get_pr"] == {
            "name": "get_pr_info",
            "tags": ["pull-request"],
        }
        assert "local_bridge_meta__generate" in config.prompt_overrides
        assert config.prompt_overrides["local_bridge_meta__generate"] == {
            "name": "generate_playbook",
            "tags": ["automation"],
        }

    def test_load_empty_config(self, tmp_path: Path) -> None:
        """Test loading empty/minimal config."""
        config_file = tmp_path / "mcp_overrides.yaml"
        config_file.write_text("")

        config = NamingConfig.load_from_yaml(config_file)

        assert config.server_tags == {}
        assert config.tool_overrides == {}
        assert config.prompt_overrides == {}

    def test_load_partial_config(self, tmp_path: Path) -> None:
        """Test loading config with only some sections."""
        config_content = """
server_tags:
  github-custom: [git]

# No tools or prompts sections
"""
        config_file = tmp_path / "mcp_overrides.yaml"
        config_file.write_text(config_content)

        config = NamingConfig.load_from_yaml(config_file)

        assert config.server_tags == {"github-custom": ["git"]}
        assert config.tool_overrides == {}
        assert config.prompt_overrides == {}

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        """Test error handling for invalid YAML."""
        config_file = tmp_path / "mcp_overrides.yaml"
        config_file.write_text("invalid: yaml: content: [[[")

        with pytest.raises(ValueError, match="Invalid YAML"):
            NamingConfig.load_from_yaml(config_file)

    def test_missing_config_file(self, tmp_path: Path) -> None:
        """Test behavior when override config doesn't exist."""
        config_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError) as exc_info:
            NamingConfig.load_from_yaml(config_file)

        assert "not found" in str(exc_info.value).lower()

    def test_override_name_only(self, tmp_path: Path) -> None:
        """Test override with only name (no tags)."""
        config_content = """
tools:
  "tool_name":
    name: renamed_tool
"""
        config_file = tmp_path / "mcp_overrides.yaml"
        config_file.write_text(config_content)

        config = NamingConfig.load_from_yaml(config_file)

        assert config.tool_overrides["tool_name"]["name"] == "renamed_tool"
        # Tags key should not be present or be empty
        assert config.tool_overrides["tool_name"].get("tags", []) == []

    def test_override_tags_only(self, tmp_path: Path) -> None:
        """Test override with only tags (no name)."""
        config_content = """
tools:
  "tool_name":
    tags: [utility, helper]
"""
        config_file = tmp_path / "mcp_overrides.yaml"
        config_file.write_text(config_content)

        config = NamingConfig.load_from_yaml(config_file)

        # Name key should not be present
        assert "name" not in config.tool_overrides["tool_name"]
        assert config.tool_overrides["tool_name"]["tags"] == ["utility", "helper"]

    def test_complex_config_structure(self, tmp_path: Path) -> None:
        """Test loading complex nested config structure."""
        config_content = """
server_tags:
  github-custom:
    - git
    - github
    - version-control
  filesystem:
    - files
    - storage
    - io
  meta:
    - automation
    - templates

tools:
  "local_bridge_github_custom__get_pr_info":
    name: get_pr_info
    tags:
      - pull-request
      - code-review
  "local_bridge_filesystem__search":
    name: filesystem_search
    tags: [search, find]
  "local_bridge_brave_search__search":
    name: web_search
    tags: [search, query, web]

prompts:
  "local_bridge_meta__generate_playbook":
    name: generate_playbook
    tags: [playbook, orchestration]
"""
        config_file = tmp_path / "mcp_overrides.yaml"
        config_file.write_text(config_content)

        config = NamingConfig.load_from_yaml(config_file)

        # Verify server tags
        assert len(config.server_tags) == 3
        assert config.server_tags["github-custom"] == ["git", "github", "version-control"]

        # Verify tool overrides
        assert len(config.tool_overrides) == 3
        pr_info_tool = config.tool_overrides["local_bridge_github_custom__get_pr_info"]
        assert pr_info_tool["name"] == "get_pr_info"
        assert config.tool_overrides["local_bridge_filesystem__search"]["tags"] == [
            "search",
            "find",
        ]

        # Verify prompt overrides
        assert len(config.prompt_overrides) == 1
        playbook_prompt = config.prompt_overrides["local_bridge_meta__generate_playbook"]
        assert playbook_prompt["tags"] == ["playbook", "orchestration"]

    def test_programmatic_config_creation(self) -> None:
        """Test creating NamingConfig programmatically (not from file)."""
        config = NamingConfig(
            server_tags={
                "github-custom": ["git", "github"],
            },
            tool_overrides={
                "tool1": {"name": "renamed1"},
            },
            prompt_overrides={
                "prompt1": {"name": "renamed_prompt1"},
            },
        )

        assert config.server_tags["github-custom"] == ["git", "github"]
        assert config.tool_overrides["tool1"]["name"] == "renamed1"
        assert config.prompt_overrides["prompt1"]["name"] == "renamed_prompt1"

    def test_empty_sections_in_yaml(self, tmp_path: Path) -> None:
        """Test YAML with empty sections."""
        config_content = """
server_tags:

tools:

prompts:
"""
        config_file = tmp_path / "mcp_overrides.yaml"
        config_file.write_text(config_content)

        config = NamingConfig.load_from_yaml(config_file)

        # Empty sections should result in empty dicts
        assert config.server_tags == {} or config.server_tags is None
        assert config.tool_overrides == {} or config.tool_overrides is None
        assert config.prompt_overrides == {} or config.prompt_overrides is None

    def test_disable_field_in_config(self, tmp_path: Path) -> None:
        """Test that disable field is properly loaded from YAML."""
        config_content = """
tools:
  "enabled_tool":
    name: renamed_tool
    disable: false
  "disabled_tool":
    disable: true
  "normal_tool":
    name: another_tool

prompts:
  "disabled_prompt":
    disable: true
"""
        config_file = tmp_path / "mcp_overrides.yaml"
        config_file.write_text(config_content)

        config = NamingConfig.load_from_yaml(config_file)

        # Tool with explicit disable: false
        assert config.tool_overrides["enabled_tool"]["disable"] is False
        assert config.tool_overrides["enabled_tool"]["name"] == "renamed_tool"

        # Tool with disable: true
        assert config.tool_overrides["disabled_tool"]["disable"] is True

        # Tool without disable field (should not have the key, will default in parse_mcp_name)
        assert "disable" not in config.tool_overrides["normal_tool"]

        # Prompt with disable: true
        assert config.prompt_overrides["disabled_prompt"]["disable"] is True
