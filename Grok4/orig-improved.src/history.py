# File: src/history.py
"""
History module for the Context7 Agent.

Handles JSON-based persistence for conversations, searches, bookmarks, and sessions.
"""

import json
import os
from typing import List, Dict, Any

HISTORY_FILE = "history.json"
SESSION_FILE = "session.json"

class History:
    def __init__(self):
        self.data: Dict[str, List[Any]] = {
            "conversations": [],
            "searches": [],
            "bookmarks": [],
        }
        self.load()

    def load(self):
        """Loads history from file, handling corruption."""
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r") as f:
                    self.data = json.load(f)
            except json.JSONDecodeError:
                # On corruption, reset to a clean state and save.
                self.data = {"conversations": [], "searches": [], "bookmarks": []}
                self.save()

    def save(self):
        """Saves the current history to a JSON file."""
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
        if doc not in self.data["bookmarks"]:
            self.data["bookmarks"].append(doc)
            self.save()

    def get_bookmarks(self) -> List[Dict]:
        return self.data["bookmarks"]

    def save_session(self, state: Dict):
        """Saves the UI state to a session file."""
        with open(SESSION_FILE, "w") as f:
            json.dump(state, f, indent=4)

    def load_session(self) -> Dict:
        """Loads UI state, handling missing or corrupt files."""
        if os.path.exists(SESSION_FILE):
            try:
                with open(SESSION_FILE, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                # On any error, return a default empty state.
                return {}
        return {}
