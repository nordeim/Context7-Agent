# File: src/history.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List

from pydantic_ai import ChatMessage


class History:
    """
    Wrapper that syncs chat history to a JSON file on disk.

    Each entry is a dict {role:str, content:str}
    """
    def __init__(self, path: Path):
        self.path = path
        self.messages: List[ChatMessage] = []
        self.load()  # attempt to restore previous session

    # --------------------------------------------------------------------- I/O
    def load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.messages = [ChatMessage(**m) for m in data]
            except Exception:
                self.messages = []

    def save(self):
        self.path.write_text(
            json.dumps([m.dict() for m in self.messages], indent=2)
        )

    # ----------------------------------------------------------------- Helpers
    def add(self, role: str, content: str):
        self.messages.append(ChatMessage(role=role, content=content))
        self.save()

    def to_model_messages(self) -> List[ChatMessage]:
        return self.messages

    # ---------------------------------------------------------- fancy features
    def last_user_message(self) -> str | None:
        for m in reversed(self.messages):
            if m.role == "user":
                return m.content
        return None
