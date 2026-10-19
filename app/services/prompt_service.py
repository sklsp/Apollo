"""Prompt service for managing prompt templates."""

from datetime import UTC, datetime
from typing import TypedDict


class Prompt(TypedDict):
    """Prompt template structure."""

    id: str
    name: str
    content: str
    created_at: str
    updated_at: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class PromptService:
    """Manage prompt templates with create, read, update, and delete operations."""

    def __init__(self) -> None:
        self._storage: dict[str, Prompt] = {}
        self._prompt_counter = 0
        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        """Initialize with default prompt templates."""
        self.create_prompt(
            "default",
            "You are a helpful AI assistant. Answer the user's question clearly and concisely.",
        )
        self.create_prompt(
            "code_helper",
            "You are an expert programmer. Help the user with code questions, provide clean examples, and explain concepts clearly.\n\nUser request: {input}",
        )
        self.create_prompt(
            "document_analyzer",
            "You are a document analysis expert. Analyze the provided documents and answer questions based on their content.\n\nQuestion: {input}",
        )
        self.create_prompt(
            "summarizer",
            "Summarize the following clearly and concisely:\n\n{input}",
        )

    def create_prompt(self, name: str, content: str) -> str:
        """Create a new prompt template."""
        prompt_id = f"prompt_{self._prompt_counter}"
        self._prompt_counter += 1
        now = _utc_now()

        self._storage[prompt_id] = {
            "id": prompt_id,
            "name": name,
            "content": content,
            "created_at": now,
            "updated_at": now,
        }
        return prompt_id

    def get_prompt(self, prompt_id: str) -> Prompt | None:
        """Retrieve a prompt by ID."""
        return self._storage.get(prompt_id)

    def get_prompt_by_name(self, name: str) -> Prompt | None:
        """Retrieve a prompt by name."""
        for prompt in self._storage.values():
            if prompt["name"] == name:
                return prompt
        return None

    def list_prompts(self) -> list[Prompt]:
        """List all prompts sorted by name."""
        return sorted(self._storage.values(), key=lambda prompt: prompt["name"].lower())

    def update_prompt(
        self,
        prompt_id: str,
        name: str | None = None,
        content: str | None = None,
    ) -> bool:
        """Update an existing prompt."""
        if prompt_id not in self._storage:
            return False

        if name is not None:
            self._storage[prompt_id]["name"] = name
        if content is not None:
            self._storage[prompt_id]["content"] = content
        self._storage[prompt_id]["updated_at"] = _utc_now()
        return True

    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt template."""
        if prompt_id in self._storage:
            del self._storage[prompt_id]
            return True
        return False

    def apply_template(self, prompt_id: str, user_input: str) -> str:
        """Merge a saved prompt template with the user's message.

        Replaces ``{input}`` in the template when present; otherwise appends the user message.
        """
        prompt = self.get_prompt(prompt_id)
        if prompt is None:
            raise KeyError(prompt_id)

        template = prompt["content"]
        if "{input}" in template:
            return template.replace("{input}", user_input)
        return f"{template}\n\n{user_input}"
