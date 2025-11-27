"""Loader for prompt files from directories."""

import logging
from pathlib import Path

from tools_api.services.prompts.parser import parse_prompt_file
from tools_api.services.prompts.template import PromptTemplate
from tools_api.services.registry import PromptRegistry

logger = logging.getLogger(__name__)


def load_prompts_from_directory(directory: Path) -> list[PromptTemplate]:
    """
    Scan directory recursively for .md files and return PromptTemplate instances.

    Logs warnings for invalid files but continues loading others.

    Args:
        directory: Path to prompts directory

    Returns:
        List of successfully parsed PromptTemplate instances

    Raises:
        FileNotFoundError: If directory does not exist
    """
    if not directory.exists():
        raise FileNotFoundError(f"Prompts directory not found: {directory}")

    prompts = []
    for file_path in directory.rglob("*.md"):
        try:
            prompt = parse_prompt_file(file_path)
            prompts.append(prompt)
            logger.info(f"Loaded prompt '{prompt.name}' from {file_path}")
        except Exception as e:
            logger.warning(f"Failed to load prompt from {file_path}: {e}")

    return prompts


def register_prompts_from_directory(directory: Path) -> int:
    """
    Load prompts from directory and register in PromptRegistry.

    Duplicate prompt names will cause startup failure (fail-fast).

    Args:
        directory: Path to prompts directory

    Returns:
        Count of successfully registered prompts

    Raises:
        FileNotFoundError: If directory does not exist
        ValueError: On duplicate prompt names (fails startup)
    """
    prompts = load_prompts_from_directory(directory)
    for prompt in prompts:
        PromptRegistry.register(prompt)  # Raises ValueError on duplicate
    return len(prompts)
