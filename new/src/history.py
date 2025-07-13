# File: src/history.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class History:
    """
    Persistent JSON history store compatible with pydantic-ai ChatMessage.
    """

    def __init__(self, file: Path):
        self.file = file
        self.messages: List[ChatMessage] = []
        self.load()

    # ---------------------------------------------------------------- load/save
    def load(self):
        if self.file.exists():
            try:
                raw = json.loads(self.file.read_text(encoding="utf-8"))
                self.messages = [ChatMessage(**m) for m in raw]
            except Exception:  # pragma: no cover
                self.messages = []

    def save(self):
        # Use model_dump for Pydantic v2+ compatibility
        self.file.write_text(
            json.dumps([m.model_dump() for m in self.messages], indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ helpers
    def add(self, role: str, content: str):
        self.messages.append(ChatMessage(role=role, content=content))
        self.save()

    def to_model_messages(self) -> List[ChatMessage]:
        return self.messages
