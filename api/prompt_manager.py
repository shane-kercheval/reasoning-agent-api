"""Simple prompt manager that loads markdown files."""
from pathlib import Path


class PromptManager:
    """Simple prompt manager that loads markdown files from disk."""

    def __init__(self, prompts_dir: str = "prompts"):
        """
        Initialize the prompt manager.

        Args:
            prompts_dir: Directory containing prompt markdown files
        """
        self.prompts_dir = Path(prompts_dir)
        self._prompts: dict[str, str] = {}

    async def initialize(self) -> None:
        """Load all prompts from the prompts directory."""
        if not self.prompts_dir.exists():
            return

        for prompt_file in self.prompts_dir.glob("*.md"):
            prompt_name = prompt_file.stem
            content = prompt_file.read_text(encoding='utf-8').strip()
            self._prompts[prompt_name] = content

    async def get_prompt(self, prompt_name: str) -> str:
        """
        Get a prompt by name.

        Args:
            prompt_name: Name of the prompt (without .md extension)

        Returns:
            The prompt content as a string

        Raises:
            KeyError: If the prompt doesn't exist
        """
        if prompt_name not in self._prompts:
            raise KeyError(f"Prompt '{prompt_name}' not found")
        return self._prompts[prompt_name]

    def get_available_prompts(self) -> list[str]:
        """Get a list of available prompt names."""
        return list(self._prompts.keys())

    async def cleanup(self) -> None:
        """Cleanup - nothing to do for simple implementation."""
        pass


# Global prompt manager instance
prompt_manager = PromptManager()


async def get_prompt_manager() -> PromptManager:
    """Get the global prompt manager instance."""
    return prompt_manager
