"""Tests for prompt file loader."""

from pathlib import Path

import pytest

from tools_api.services.prompts.loader import (
    load_prompts_from_directory,
    register_prompts_from_directory,
)
from tools_api.services.registry import PromptRegistry


@pytest.fixture(autouse=True)
def clear_registry() -> None: # type: ignore
    """Clear prompt registry before and after each test."""
    PromptRegistry.clear()
    yield
    PromptRegistry.clear()


@pytest.fixture
def valid_prompt_1() -> str:
    """First valid prompt content."""
    return """---
name: prompt_one
description: First test prompt
category: test
---
Hello from prompt one!
"""


@pytest.fixture
def valid_prompt_2() -> str:
    """Second valid prompt content."""
    return """---
name: prompt_two
description: Second test prompt
---
Hello from prompt two!
"""


@pytest.fixture
def invalid_prompt() -> str:
    """Invalid prompt content (missing required field)."""
    return """---
description: Missing name field
---
This won't parse correctly.
"""


def test_load_directory_with_multiple_valid_files(
    tmp_path: Path, valid_prompt_1: str, valid_prompt_2: str,
) -> None:
    """Test loading directory with multiple valid .md files."""
    (tmp_path / "prompt1.md").write_text(valid_prompt_1)
    (tmp_path / "prompt2.md").write_text(valid_prompt_2)

    prompts = load_prompts_from_directory(tmp_path)

    assert len(prompts) == 2
    names = {p.name for p in prompts}
    assert "prompt_one" in names
    assert "prompt_two" in names


def test_load_nested_subdirectories_are_scanned(
    tmp_path: Path, valid_prompt_1: str, valid_prompt_2: str,
) -> None:
    """Test that nested subdirectories are scanned (recursive)."""
    # Create nested structure
    (tmp_path / "prompt1.md").write_text(valid_prompt_1)
    subdir = tmp_path / "subdir" / "nested"
    subdir.mkdir(parents=True)
    (subdir / "prompt2.md").write_text(valid_prompt_2)

    prompts = load_prompts_from_directory(tmp_path)

    assert len(prompts) == 2
    names = {p.name for p in prompts}
    assert "prompt_one" in names
    assert "prompt_two" in names


def test_load_ignores_non_md_files(tmp_path: Path, valid_prompt_1: str) -> None:
    """Test that non-.md files are ignored."""
    (tmp_path / "prompt.md").write_text(valid_prompt_1)
    (tmp_path / "readme.txt").write_text("This is not a prompt")
    (tmp_path / "config.yaml").write_text("key: value")
    (tmp_path / "script.py").write_text("print('hello')")

    prompts = load_prompts_from_directory(tmp_path)

    assert len(prompts) == 1
    assert prompts[0].name == "prompt_one"


def test_load_invalid_file_logs_warning_and_continues(
    tmp_path: Path, valid_prompt_1: str, invalid_prompt: str, caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that invalid file logs warning and continues loading others."""
    (tmp_path / "valid.md").write_text(valid_prompt_1)
    (tmp_path / "invalid.md").write_text(invalid_prompt)

    prompts = load_prompts_from_directory(tmp_path)

    # Should have loaded the valid one
    assert len(prompts) == 1
    assert prompts[0].name == "prompt_one"

    # Should have logged a warning about the invalid one
    assert any("Failed to load" in record.message for record in caplog.records)


def test_load_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    """Test that empty directory returns empty list."""
    prompts = load_prompts_from_directory(tmp_path)

    assert prompts == []


def test_load_nonexistent_directory_raises_error(tmp_path: Path) -> None:
    """Test that non-existent directory raises FileNotFoundError."""
    nonexistent = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError) as exc_info:
        load_prompts_from_directory(nonexistent)

    assert "not found" in str(exc_info.value).lower()


def test_register_prompts_from_directory(
    tmp_path: Path, valid_prompt_1: str, valid_prompt_2: str,
) -> None:
    """Test registering prompts from directory into registry."""
    (tmp_path / "prompt1.md").write_text(valid_prompt_1)
    (tmp_path / "prompt2.md").write_text(valid_prompt_2)

    count = register_prompts_from_directory(tmp_path)

    assert count == 2
    assert PromptRegistry.get("prompt_one") is not None
    assert PromptRegistry.get("prompt_two") is not None


def test_register_duplicate_names_raises_error(tmp_path: Path) -> None:
    """Test that duplicate prompt names raise ValueError during registration."""
    # Create two files with the same prompt name
    content1 = """---
name: duplicate
description: First instance
---
First content
"""
    content2 = """---
name: duplicate
description: Second instance
---
Second content
"""
    (tmp_path / "file1.md").write_text(content1)
    (tmp_path / "file2.md").write_text(content2)

    with pytest.raises(ValueError) as exc_info:  # noqa: PT011
        register_prompts_from_directory(tmp_path)

    assert "duplicate" in str(exc_info.value).lower()


def test_register_returns_count(tmp_path: Path, valid_prompt_1: str) -> None:
    """Test that register returns correct count of registered prompts."""
    (tmp_path / "prompt.md").write_text(valid_prompt_1)

    count = register_prompts_from_directory(tmp_path)

    assert count == 1


def test_register_nonexistent_directory_raises_error(tmp_path: Path) -> None:
    """Test that registering from non-existent directory raises FileNotFoundError."""
    nonexistent = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError):
        register_prompts_from_directory(nonexistent)
