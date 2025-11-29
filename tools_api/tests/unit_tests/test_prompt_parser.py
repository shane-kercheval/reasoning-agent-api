"""Tests for prompt file parser."""

import asyncio
import pytest
from pathlib import Path
from tools_api.services.prompts.parser import parse_prompt_file


@pytest.fixture
def valid_prompt_content() -> str:
    """Valid prompt file content with all fields."""
    return """---
name: code_review
description: Review code for quality and best practices
category: development
arguments:
  - name: language
    required: true
    description: Programming language of the code
  - name: focus
    required: false
    description: Specific areas to focus on
tags:
  - code
  - review
---
You are a senior {{ language }} developer performing a code review.

{% if focus %}
Focus specifically on: {{ focus }}
{% endif %}

Review the code for quality, bugs, and best practices.
"""


@pytest.fixture
def minimal_prompt_content() -> str:
    """Minimal valid prompt file content with only required fields."""
    return """---
name: simple
description: A simple prompt
---
This is the template content.
"""


def test_parse_valid_file_all_fields(tmp_path: Path, valid_prompt_content: str) -> None:
    """Test parsing valid file with all fields populated."""
    file_path = tmp_path / "code_review.md"
    file_path.write_text(valid_prompt_content)

    prompt = parse_prompt_file(file_path)

    assert prompt.name == "code_review"
    assert prompt.description == "Review code for quality and best practices"
    assert prompt.category == "development"
    assert len(prompt.arguments) == 2
    assert prompt.arguments[0]["name"] == "language"
    assert prompt.arguments[0]["required"] is True
    assert prompt.arguments[1]["name"] == "focus"
    assert prompt.arguments[1]["required"] is False
    assert prompt.tags == ["code", "review"]
    assert prompt.source_path == file_path
    assert "senior" in prompt._template_str
    assert "{{ language }}" in prompt._template_str


def test_parse_valid_file_optional_fields_omitted(
    tmp_path: Path, minimal_prompt_content: str,
) -> None:
    """Test parsing valid file with optional fields omitted."""
    file_path = tmp_path / "simple.md"
    file_path.write_text(minimal_prompt_content)

    prompt = parse_prompt_file(file_path)

    assert prompt.name == "simple"
    assert prompt.description == "A simple prompt"
    assert prompt.category is None
    assert prompt.arguments == []
    assert prompt.tags == []
    assert prompt.source_path == file_path


def test_parse_missing_name_raises_error(tmp_path: Path) -> None:
    """Test that missing required field 'name' raises ValueError."""
    content = """---
description: No name field
---
Template content
"""
    file_path = tmp_path / "no_name.md"
    file_path.write_text(content)

    with pytest.raises(ValueError) as exc_info:  # noqa: PT011
        parse_prompt_file(file_path)

    assert "name" in str(exc_info.value).lower()
    assert str(file_path) in str(exc_info.value)


def test_parse_missing_description_raises_error(tmp_path: Path) -> None:
    """Test that missing required field 'description' raises ValueError."""
    content = """---
name: no_description
---
Template content
"""
    file_path = tmp_path / "no_desc.md"
    file_path.write_text(content)

    with pytest.raises(ValueError) as exc_info:  # noqa: PT011
        parse_prompt_file(file_path)

    assert "description" in str(exc_info.value).lower()
    assert str(file_path) in str(exc_info.value)


def test_parse_empty_template_body(tmp_path: Path) -> None:
    """Test parsing file with empty template body (valid)."""
    content = """---
name: empty_body
description: Has empty body
---
"""
    file_path = tmp_path / "empty_body.md"
    file_path.write_text(content)

    prompt = parse_prompt_file(file_path)

    assert prompt.name == "empty_body"
    assert prompt._template_str == ""


def test_parse_file_no_frontmatter_raises_error(tmp_path: Path) -> None:
    """Test that file with no frontmatter raises error."""
    content = """This is just plain text with no frontmatter."""
    file_path = tmp_path / "no_frontmatter.md"
    file_path.write_text(content)

    with pytest.raises(ValueError) as exc_info:  # noqa: PT011
        parse_prompt_file(file_path)

    # Should fail on missing 'name' field since there's no frontmatter
    assert "name" in str(exc_info.value).lower()


def test_parse_unicode_content(tmp_path: Path) -> None:
    """Test parsing file with unicode content works."""
    content = """---
name: unicode_test
description: Tests unicode handling
---
Hello {{ name }}! Your emoji is {{ emoji }}.
Special characters: Muller, cafe, nino
"""
    file_path = tmp_path / "unicode.md"
    file_path.write_text(content, encoding="utf-8")

    prompt = parse_prompt_file(file_path)

    assert prompt.name == "unicode_test"
    assert "Muller" in prompt._template_str
    assert "cafe" in prompt._template_str


def test_parse_sets_source_path(tmp_path: Path, minimal_prompt_content: str) -> None:
    """Test that source_path is set on returned PromptTemplate."""
    file_path = tmp_path / "with_path.md"
    file_path.write_text(minimal_prompt_content)

    prompt = parse_prompt_file(file_path)

    assert prompt.source_path is not None
    assert prompt.source_path == file_path


def test_parsed_template_renders_correctly(
    tmp_path: Path, valid_prompt_content: str,
) -> None:
    """Test that parsed template can render correctly."""
    file_path = tmp_path / "renderable.md"
    file_path.write_text(valid_prompt_content)

    prompt = parse_prompt_file(file_path)

    # Test sync render via pytest.mark.asyncio
    result = asyncio.run(prompt.render(language="Python", focus="security"))

    assert "Python" in result
    assert "security" in result
