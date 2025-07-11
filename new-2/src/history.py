# src/history.py
"""Conversation history management for Context7 Agent."""

import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import uuid


class ConversationHistory:
    """Manages conversation history storage and retrieval."""
    
    def __init__(self, history_file: Path):
        """
        Initialize conversation history manager.
        
        Args:
            history_file: Path to the history JSON file
        """
        self.history_file = history_file
        self.current_session_id = str(uuid.uuid4())
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Ensure the history file exists."""
        if not self.history_file.exists():
            self.history_file.write_text(json.dumps({
                "sessions": {},
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }, indent=2))
    
    def _load_history(self) -> Dict:
        """Load history from file."""
        try:
            return json.loads(self.history_file.read_text())
        except Exception:
            return {"sessions": {}, "metadata": {}}
    
    def _save_history(self, data: Dict):
        """Save history to file."""
        self.history_file.write_text(json.dumps(data, indent=2))
    
    def start_session(self, title: Optional[str] = None) -> str:
        """
        Start a new conversation session.
        
        Args:
            title: Optional session title
            
        Returns:
            Session ID
        """
        self.current_session_id = str(uuid.uuid4())
        history = self._load_history()
        
        history["sessions"][self.current_session_id] = {
            "id": self.current_session_id,
            "title": title or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "metadata": {}
        }
        
        self._save_history(history)
        return self.current_session_id
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        Add a message to the current session.
        
        Args:
            role: Message role (user/assistant)
            content: Message content
            metadata: Optional metadata
        """
        history = self._load_history()
        
        if self.current_session_id not in history["sessions"]:
            self.start_session()
            history = self._load_history()
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        history["sessions"][self.current_session_id]["messages"].append(message)
        self._save_history(history)
    
    def get_session_messages(self, session_id: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Get messages from a session.
        
        Args:
            session_id: Session ID (uses current if not provided)
            
        Returns:
            List of messages
        """
        history = self._load_history()
        session_id = session_id or self.current_session_id
        
        if session_id in history["sessions"]:
            return history["sessions"][session_id]["messages"]
        return []
    
    def get_all_sessions(self) -> List[Dict]:
        """
        Get all conversation sessions.
        
        Returns:
            List of session summaries
        """
        history = self._load_history()
        sessions = []
        
        for session_id, session_data in history["sessions"].items():
            sessions.append({
                "id": session_id,
                "title": session_data.get("title", "Untitled"),
                "created_at": session_data.get("created_at"),
                "message_count": len(session_data.get("messages", [])),
                "last_message": session_data.get("messages", [])[-1] if session_data.get("messages") else None
            })
        
        # Sort by creation date (newest first)
        sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return sessions
    
    def search_history(self, query: str) -> List[Dict]:
        """
        Search through conversation history.
        
        Args:
            query: Search query
            
        Returns:
            List of matching messages with session info
        """
        history = self._load_history()
        results = []
        query_lower = query.lower()
        
        for session_id, session_data in history["sessions"].items():
            for message in session_data.get("messages", []):
                if query_lower in message["content"].lower():
                    results.append({
                        "session_id": session_id,
                        "session_title": session_data.get("title"),
                        "message": message,
                        "context": self._get_message_context(session_data["messages"], message)
                    })
        
        return results
    
    def _get_message_context(self, messages: List[Dict], target_message: Dict, context_size: int = 2) -> List[Dict]:
        """Get surrounding context for a message."""
        try:
            index = messages.index(target_message)
            start = max(0, index - context_size)
            end = min(len(messages), index + context_size + 1)
            return messages[start:end]
        except ValueError:
            return [target_message]
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a conversation session.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if deleted, False otherwise
        """
        history = self._load_history()
        
        if session_id in history["sessions"]:
            del history["sessions"][session_id]
            self._save_history(history)
            return True
        
        return False
    
    def export_session(self, session_id: str, format: str = "json") -> Optional[str]:
        """
        Export a session in various formats.
        
        Args:
            session_id: Session ID to export
            format: Export format (json, markdown, txt)
            
        Returns:
            Exported content or None
        """
        history = self._load_history()
        
        if session_id not in history["sessions"]:
            return None
        
        session = history["sessions"][session_id]
        
        if format == "json":
            return json.dumps(session, indent=2)
        
        elif format == "markdown":
            content = f"# {session.get('title', 'Conversation')}\n\n"
            content += f"*Created: {session.get('created_at', 'Unknown')}*\n\n"
            
            for msg in session.get("messages", []):
                role = "**You**" if msg["role"] == "user" else "**Assistant**"
                content += f"{role}: {msg['content']}\n\n"
            
            return content
        
        elif format == "txt":
            content = f"{session.get('title', 'Conversation')}\n"
            content += f"{'=' * len(session.get('title', 'Conversation'))}\n\n"
            
            for msg in session.get("messages", []):
                role = "You" if msg["role"] == "user" else "Assistant"
                content += f"{role}: {msg['content']}\n\n"
            
            return content
        
        return None


class BookmarkManager:
    """Manages document bookmarks."""
    
    def __init__(self, bookmarks_file: Path):
        """Initialize bookmark manager."""
        self.bookmarks_file = bookmarks_file
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Ensure the bookmarks file exists."""
        if not self.bookmarks_file.exists():
            self.bookmarks_file.write_text(json.dumps({
                "bookmarks": [],
                "tags": [],
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }, indent=2))
    
    def add_bookmark(self, document: Dict, tags: Optional[List[str]] = None, notes: Optional[str] = None) -> str:
        """Add a bookmark."""
        data = json.loads(self.bookmarks_file.read_text())
        
        bookmark_id = str(uuid.uuid4())
        bookmark = {
            "id": bookmark_id,
            "document": document,
            "tags": tags or [],
            "notes": notes,
            "created_at": datetime.now().isoformat()
        }
        
        data["bookmarks"].append(bookmark)
        
        # Update tags list
        if tags:
            data["tags"] = list(set(data["tags"] + tags))
        
        self.bookmarks_file.write_text(json.dumps(data, indent=2))
        return bookmark_id
    
    def get_bookmarks(self, tag: Optional[str] = None) -> List[Dict]:
        """Get bookmarks, optionally filtered by tag."""
        data = json.loads(self.bookmarks_file.read_text())
        bookmarks = data.get("bookmarks", [])
        
        if tag:
            bookmarks = [b for b in bookmarks if tag in b.get("tags", [])]
        
        return bookmarks
    
    def remove_bookmark(self, bookmark_id: str) -> bool:
        """Remove a bookmark."""
        data = json.loads(self.bookmarks_file.read_text())
        original_count = len(data["bookmarks"])
        
        data["bookmarks"] = [b for b in data["bookmarks"] if b["id"] != bookmark_id]
        
        if len(data["bookmarks"]) < original_count:
            self.bookmarks_file.write_text(json.dumps(data, indent=2))
            return True
        
        return False
