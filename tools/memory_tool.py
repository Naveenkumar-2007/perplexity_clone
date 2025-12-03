from typing import Dict, List


class MemoryTool:
    """Simple in-memory workspace chat history."""

    def __init__(self) -> None:
        self.store: Dict[str, List[Dict[str, str]]] = {}
        self.profile: Dict[str, Dict[str, str]] = {}  # Store user metadata like name

    def add(self, workspace_id: str, role: str, content: str) -> None:
        self.store.setdefault(workspace_id, []).append(
            {"role": role, "content": content}
        )

    def get_context(self, workspace_id: str, max_messages: int = 10) -> str:
        msgs = self.store.get(workspace_id, [])[-max_messages:]
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in msgs)

    def get_recent_messages(self, workspace_id: str, limit: int = 6) -> List[Dict[str, str]]:
        """Get recent messages for LLM context (default last 6 messages)."""
        return self.store.get(workspace_id, [])[-limit:]

    def get_long_chat(self, workspace_id: str) -> List[Dict[str, str]]:
        """Get entire chat history for long-term memory context."""
        return self.store.get(workspace_id, [])

    def set_name(self, workspace_id: str, name: str) -> None:
        """Store user's name in profile."""
        self.profile[workspace_id] = {"name": name}

    def get_name(self, workspace_id: str) -> str:
        """Retrieve user's name from profile."""
        return self.profile.get(workspace_id, {}).get("name")
