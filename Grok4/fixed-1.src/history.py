# File: src/history.py
"""
History module for the Context7 Agent.

Handles JSON-based persistence for conversations, searches, bookmarks, and sessions.
"""

import json
import os
from typing import List, Dict

HISTORY_FILE = "history.json"

class History:
    def __init__(self):
        self.data = {"conversations": [], "searches": [], "bookmarks": [], "sessions": {}}
        self.load()

    def load(self):
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r") as f:
                    self.data = json.load(f)
            except json.JSONDecodeError:
                # Corrupted or empty JSON: reset history
                self.data = {
                    "conversations": [],
                    "searches": [],
                    "bookmarks": [],
                    "sessions": {}
                }
                self.save()

    def save(self):
        with open(HISTORY_FILE, "w") as f:
            json.dump(self.data, f, indent=4)

    def add_message(self, role: str, content: str):
        self.data["conversations"].append({"role": role, "content": content})
        self.save()

    def get_conversation(self) -> List[Dict]:
        return self.data["conversations"]

    def add_search(self, query: str, results: List[Dict]):
        self.data["searches"].append({"query": query, "results": results})
        self.save()

    def get_searches(self) -> List[Dict]:
        return self.data["searches"]

    def add_bookmark(self, doc: Dict):
        self.data["bookmarks"].append(doc)
        self.save()

    # Add more methods for sessions, analytics...
