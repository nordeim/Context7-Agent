Below is a focused patch that catches and recovers from invalid JSON in your history file so you no longer get a crash on startup. It only touches `src/history.py`, wrapping the `json.load` in a `try/except` to handle `JSONDecodeError`, resetting the in-memory data and writing a fresh file when the stored JSON is corrupted.

```diff
diff --git a/src/history.py b/src/history.py
index e3f5a2b..4a9c8d1 100644
--- a/src/history.py
+++ b/src/history.py
@@ class History:
     def load(self):
-        if os.path.exists(HISTORY_FILE):
-            with open(HISTORY_FILE, "r") as f:
-                self.data = json.load(f)
+        if os.path.exists(HISTORY_FILE):
+            try:
+                with open(HISTORY_FILE, "r") as f:
+                    self.data = json.load(f)
+            except json.JSONDecodeError:
+                # Corrupted or empty JSON: reset history
+                self.data = {
+                    "conversations": [],
+                    "searches": [],
+                    "bookmarks": [],
+                    "sessions": {}
+                }
+                self.save()

     def save(self):
         with open(HISTORY_FILE, "w") as f:
             json.dump(self.data, f, indent=4)
```

Full updated `src/history.py` for clarity:

```python
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
        self.data = {
            "conversations": [],
            "searches": [],
            "bookmarks": [],
            "sessions": {}
        }
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
```

With this in place:

1. If `history.json` is missing, you start with a clean state.
2. If it exists but is invalid JSON (empty file or partial writes), it’s safely reinitialized and overwritten.
3. No more `JSONDecodeError` on startup.

Restart your CLI, and the agent will boot past history loading every time.
