# File: src/history.py
"""
Conversation history management.
This version is a simple, robust implementation that correctly returns a 
plain list[dict] for maximum stability with pydantic-ai v0.4.2.
"""
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiofiles

from .config import config

class HistoryManager:
    """Manages conversation history with a simple, robust, and compliant message schema."""
    def __init__(self):
        self.history_path = config.history_file
        self.max_history = config.max_history
        self._history: Dict[str, List[Dict[str, Any]]] = {}

    async def load(self):
        """Loads conversation history from disk asynchronously."""
        try:
            if self.history_path.exists():
                async with aiofiles.open(self.history_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    if content:
                        self._history = json.loads(content)
        except Exception as e:
            print(f"Warning: Could not load history: {e}. Starting fresh.")
            self._history = {}

    async def save(self):
        """Saves conversation history to disk asynchronously."""
        try:
            config.history_file.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(self.history_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self._history, indent=2))
        except Exception as e:
            print(f"Warning: Could not save history: {e}")

    async def add_message(self, conversation_id: str, role: str, content: str):
        """Adds a message to the history, respecting the max history limit."""
        if conversation_id not in self._history:
            self._history[conversation_id] = []
        
        self._history[conversation_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self._history[conversation_id]) > self.max_history:
            self._history[conversation_id] = self._history[conversation_id][-self.max_history:]
        
        await self.save()

    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Gets messages for a conversation in the exact list[dict] format the Agent needs."""
        messages = self._history.get(conversation_id, [])
        return [{"role": msg["role"], "content": msg["content"]} for msg in messages]

    def get_conversations(self) -> List[Dict[str, Any]]:
        """Gets metadata for all conversations, for use in a UI."""
        conversations = []
        for conv_id, messages in self._history.items():
            if messages:
                last_message = messages[-1].get("content", "")
                conversations.append({
                    "id": conv_id,
                    "last_message": last_message[:50] + "..." if len(last_message) > 50 else last_message,
                    "message_count": len(messages)
                })
        return sorted(conversations, key=lambda x: self._history.get(x["id"], [{}])[-1].get("timestamp", ""), reverse=True)

    async def clear(self, conversation_id: Optional[str] = None):
        """Clears history for a specific conversation or all conversations."""
        if conversation_id and conversation_id in self._history:
            del self._history[conversation_id]
        elif conversation_id is None:
            self._history.clear()
        await self.save()
