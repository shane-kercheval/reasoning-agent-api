"""Parser for markdown files with YAML frontmatter."""

from pathlib import Path

import frontmatter

from tools_api.services.prompts.template import PromptTemplate


def parse_prompt_file(file_path: Path) -> PromptTemplate:
    """
    Parse a markdown file with YAML frontmatter into a PromptTemplate.

    Required frontmatter fields: name, description
    Optional frontmatter fields: arguments, tags, category

    Args:
        file_path: Path to the markdown file

    Returns:
        PromptTemplate instance with source_path set

    Raises:
        ValueError: If required fields are missing or file is malformed
    """
    post = frontmatter.load(file_path)

    # Validate required fields
    if "name" not in post.metadata:
        raise ValueError(f"Missing required field 'name' in {file_path}")
    if "description" not in post.metadata:
        raise ValueError(f"Missing required field 'description' in {file_path}")

    return PromptTemplate(
        name=post.metadata["name"],
        description=post.metadata["description"],
        template=post.content,
        arguments=post.metadata.get("arguments", []),
        category=post.metadata.get("category"),
        tags=post.metadata.get("tags", []),
        source_path=file_path,
    )
