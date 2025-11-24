"""Prompts router - REST endpoints for prompt rendering."""

from typing import Any

from fastapi import APIRouter, HTTPException

from tools_api.models import PromptDefinition, PromptResult
from tools_api.services.registry import PromptRegistry

router = APIRouter(prefix="/prompts", tags=["prompts"])


@router.get("/")
async def list_prompts() -> list[PromptDefinition]:
    """List all available prompts."""
    return [
        PromptDefinition(
            name=prompt.name,
            description=prompt.description,
            arguments=prompt.arguments,
            tags=prompt.tags,
        )
        for prompt in PromptRegistry.list()
    ]


@router.post("/{prompt_name}")
async def render_prompt(
    prompt_name: str,
    arguments: dict[str, Any],
) -> PromptResult:
    """
    Render a prompt with the provided arguments.

    Args:
        prompt_name: Name of the prompt to render
        arguments: Prompt-specific arguments

    Returns:
        PromptResult with success status and rendered messages

    Raises:
        HTTPException: 404 if prompt not found
    """
    prompt = PromptRegistry.get(prompt_name)
    if prompt is None:
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_name}' not found")

    return await prompt(**arguments)
