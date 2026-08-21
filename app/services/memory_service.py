"""Memory service for managing chat history per session."""

from datetime import datetime, timezone
from typing import TypedDict

from app.core.persistence import JsonPersist

# Keep persisted sessions bounded: the most recent 50 sessions, 200 messages each.
MAX_PERSISTED_SESSIONS = 50
MAX_PERSISTED_MESSAGES = 200


class Message(TypedDict):
    """Chat message structure."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: str  # ISO format


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryService:
    """Persistent chat history storage.

    Sessions survive application restarts via ``data/sessions.json``.
    """

    def __init__(self) -> None:
        # Structure: {session_id: [messages]}
        self._storage: dict[str, list[Message]] = {}
        self._session_settings: dict[str, dict[str, bool]] = {}
        self._persist = JsonPersist("sessions.json")
        self._load()

    def _load(self) -> None:
        stored = self._persist.load(default={})
        if not isinstance(stored, dict):
            return
        sessions = stored.get("sessions")
        if isinstance(sessions, dict):
            for session_id, messages in sessions.items():
                if isinstance(messages, list):
                    self._storage[str(session_id)] = [
                        message for message in messages
                        if isinstance(message, dict) and "role" in message
                    ]
        settings_map = stored.get("settings")
        if isinstance(settings_map, dict):
            for session_id, flags in settings_map.items():
                if isinstance(flags, dict):
                    self._session_settings[str(session_id)] = dict(flags)

    def _save(self) -> None:
        # Trim to the most recent sessions so the file stays small.
        recent = list(self._storage.items())[-MAX_PERSISTED_SESSIONS:]
        trimmed = {
            session_id: messages[-MAX_PERSISTED_MESSAGES:]
            for session_id, messages in recent
        }
        self._persist.save({
            "sessions": trimmed,
            "settings": self._session_settings,
        })

    def get_use_documents(self, session_id: str, default: bool = True) -> bool:
        """Return whether RAG document retrieval is enabled for a session."""
        return self._session_settings.get(session_id, {}).get("use_documents", default)

    def set_use_documents(self, session_id: str, enabled: bool) -> None:
        """Persist the RAG document toggle for a session."""
        if session_id not in self._session_settings:
            self._session_settings[session_id] = {}
        self._session_settings[session_id]["use_documents"] = enabled
        self._save()

    def add_message(self, session_id: str, role: str, content: str) -> Message:
        """Add a message to session history.

        Args:
            session_id: Unique session identifier
            role: "user" or "assistant"
            content: Message content

        Returns:
            The created message
        """
        if session_id not in self._storage:
            self._storage[session_id] = []

        message: Message = {
            "role": role,
            "content": content,
            "timestamp": _utc_now(),
        }

        self._storage[session_id].append(message)
        self._save()
        return message

    def get_history(self, session_id: str) -> list[Message]:
        """Get all messages for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            List of messages in chronological order
        """
        return self._storage.get(session_id, [])

    def clear_history(self, session_id: str) -> None:
        """Clear all messages for a session.

        Args:
            session_id: Unique session identifier
        """
        if session_id in self._storage:
            del self._storage[session_id]
            self._save()

    def get_session_ids(self) -> list[str]:
        """Get all session IDs.

        Returns:
            List of active session IDs
        """
        return list(self._storage.keys())

    def format_history_for_prompt(self, session_id: str) -> str:
        """Format chat history as a string for inclusion in prompts.

        Args:
            session_id: Unique session identifier

        Returns:
            Formatted history string
        """
        history = self.get_history(session_id)
        if not history:
            return ""

        lines = []
        for msg in history:
            role_label = "USER" if msg["role"] == "user" else "ASSISTANT"
            lines.append(f"{role_label}: {msg['content']}")

        return "\n".join(lines)
