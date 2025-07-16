"""
Conversation history management for the Context7 Agent.

Handles persistent storage, retrieval, and analysis of chat conversations
with automatic session management and search functionality.
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class MessageRole(Enum):
    """Enumeration for message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

@dataclass
class ChatMessage:
    """Represents a single chat message."""
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.metadata is None:
            self.metadata = {}
        
        # Ensure timestamp is datetime object
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for JSON serialization."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create message from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )

@dataclass
class SearchResult:
    """Represents a search result with metadata."""
    query: str
    results: List[Dict[str, Any]]
    timestamp: datetime
    filters: Dict[str, Any] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.filters is None:
            self.filters = {}
        
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "results": self.results,
            "timestamp": self.timestamp.isoformat(),
            "filters": self.filters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResult':
        """Create from dictionary."""
        return cls(
            query=data["query"],
            results=data["results"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            filters=data.get("filters", {})
        )

class ConversationHistory:
    """
    Manages conversation history with persistent storage and analytics.
    
    Features:
    - Persistent JSON storage
    - Conversation search and filtering
    - Analytics and insights
    - Session management
    - Auto-save functionality
    """
    
    def __init__(self, data_dir: Path, max_messages: int = 1000):
        """
        Initialize conversation history manager.
        
        Args:
            data_dir: Directory for storing history files
            max_messages: Maximum messages to keep in memory
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.max_messages = max_messages
        self.messages: List[ChatMessage] = []
        self.search_history: List[SearchResult] = []
        self.bookmarks: List[Dict[str, Any]] = []
        
        # File paths
        self.history_file = self.data_dir / "conversation_history.json"
        self.search_file = self.data_dir / "search_history.json"
        self.bookmarks_file = self.data_dir / "bookmarks.json"
        
        # Load existing data
        self.load_all()
    
    def add_message(self, role: MessageRole, content: str, metadata: Dict[str, Any] = None) -> ChatMessage:
        """
        Add a new message to the conversation history.
        
        Args:
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Created ChatMessage object
        """
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.messages.append(message)
        
        # Trim history if needed
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        
        # Auto-save
        asyncio.create_task(self._auto_save_messages())
        
        return message
    
    def add_user_message(self, content: str, metadata: Dict[str, Any] = None) -> ChatMessage:
        """Add a user message."""
        return self.add_message(MessageRole.USER, content, metadata)
    
    def add_assistant_message(self, content: str, metadata: Dict[str, Any] = None) -> ChatMessage:
        """Add an assistant message."""
        return self.add_message(MessageRole.ASSISTANT, content, metadata)
    
    def add_system_message(self, content: str, metadata: Dict[str, Any] = None) -> ChatMessage:
        """Add a system message."""
        return self.add_message(MessageRole.SYSTEM, content, metadata)
    
    def add_search_result(self, query: str, results: List[Dict[str, Any]], filters: Dict[str, Any] = None) -> SearchResult:
        """
        Add a search result to history.
        
        Args:
            query: Search query
            results: Search results
            filters: Applied filters
            
        Returns:
            Created SearchResult object
        """
        search_result = SearchResult(
            query=query,
            results=results,
            timestamp=datetime.now(),
            filters=filters or {}
        )
        
        self.search_history.append(search_result)
        
        # Auto-save
        asyncio.create_task(self._auto_save_search())
        
        return search_result
    
    def get_messages(self, limit: Optional[int] = None, role_filter: Optional[MessageRole] = None) -> List[ChatMessage]:
        """
        Get messages with optional filtering.
        
        Args:
            limit: Maximum number of messages to return
            role_filter: Filter by message role
            
        Returns:
            List of ChatMessage objects
        """
        messages = self.messages
        
        if role_filter:
            messages = [msg for msg in messages if msg.role == role_filter]
        
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def get_recent_messages(self, hours: int = 24) -> List[ChatMessage]:
        """Get messages from the last N hours."""
        cutoff = datetime.now().timestamp() - (hours * 3600)
        return [
            msg for msg in self.messages
            if msg.timestamp.timestamp() > cutoff
        ]
    
    def search_messages(self, query: str, case_sensitive: bool = False) -> List[ChatMessage]:
        """
        Search messages by content.
        
        Args:
            query: Search query
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            List of matching messages
        """
        if not case_sensitive:
            query = query.lower()
        
        matching_messages = []
        for message in self.messages:
            content = message.content if case_sensitive else message.content.lower()
            if query in content:
                matching_messages.append(message)
        
        return matching_messages
    
    def get_search_history(self, limit: Optional[int] = None) -> List[SearchResult]:
        """Get search history with optional limit."""
        if limit:
            return self.search_history[-limit:]
        return self.search_history
    
    def add_bookmark(self, document_id: str, title: str, notes: str = "", tags: List[str] = None) -> Dict[str, Any]:
        """
        Add a bookmark.
        
        Args:
            document_id: Document identifier
            title: Bookmark title
            notes: Optional notes
            tags: Optional tags
            
        Returns:
            Created bookmark dictionary
        """
        bookmark = {
            "id": document_id,
            "title": title,
            "notes": notes,
            "tags": tags or [],
            "timestamp": datetime.now().isoformat(),
            "metadata": {}
        }
        
        self.bookmarks.append(bookmark)
        
        # Auto-save
        asyncio.create_task(self._auto_save_bookmarks())
        
        return bookmark
    
    def remove_bookmark(self, document_id: str) -> bool:
        """Remove a bookmark by document ID."""
        initial_length = len(self.bookmarks)
        self.bookmarks = [b for b in self.bookmarks if b["id"] != document_id]
        
        if len(self.bookmarks) < initial_length:
            asyncio.create_task(self._auto_save_bookmarks())
            return True
        return False
    
    def get_bookmarks(self, tag_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get bookmarks with optional tag filtering."""
        if tag_filter:
            return [b for b in self.bookmarks if tag_filter in b.get("tags", [])]
        return self.bookmarks
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get conversation analytics and insights."""
        total_messages = len(self.messages)
        user_messages = len([m for m in self.messages if m.role == MessageRole.USER])
        assistant_messages = len([m for m in self.messages if m.role == MessageRole.ASSISTANT])
        
        # Calculate average message length
        if self.messages:
            avg_message_length = sum(len(m.content) for m in self.messages) / len(self.messages)
        else:
            avg_message_length = 0
        
        # Recent activity (last 24 hours)
        recent_messages = self.get_recent_messages(24)
        
        # Search analytics
        search_count = len(self.search_history)
        recent_searches = [
            s for s in self.search_history
            if (datetime.now() - s.timestamp).total_seconds() < 86400
        ]
        
        return {
            "total_messages": total_messages,
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "average_message_length": avg_message_length,
            "recent_messages_24h": len(recent_messages),
            "total_searches": search_count,
            "recent_searches_24h": len(recent_searches),
            "total_bookmarks": len(self.bookmarks),
            "first_message_date": self.messages[0].timestamp.isoformat() if self.messages else None,
            "last_message_date": self.messages[-1].timestamp.isoformat() if self.messages else None
        }
    
    def clear_all(self):
        """Clear all history data."""
        self.messages.clear()
        self.search_history.clear()
        self.bookmarks.clear()
        
        # Save empty data
        asyncio.create_task(self._auto_save_all())
    
    def export_conversation(self, filepath: Path, format: str = "json"):
        """
        Export conversation history to file.
        
        Args:
            filepath: Output file path
            format: Export format (json, txt)
        """
        if format == "json":
            self._export_json(filepath)
        elif format == "txt":
            self._export_text(filepath)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, filepath: Path):
        """Export as JSON."""
        export_data = {
            "messages": [msg.to_dict() for msg in self.messages],
            "search_history": [search.to_dict() for search in self.search_history],
            "bookmarks": self.bookmarks,
            "analytics": self.get_analytics(),
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def _export_text(self, filepath: Path):
        """Export as plain text."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("Context7 Agent Conversation Export\n")
            f.write("=" * 40 + "\n\n")
            
            for message in self.messages:
                role_name = message.role.value.title()
                timestamp = message.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                
                f.write(f"[{timestamp}] {role_name}:\n")
                f.write(f"{message.content}\n")
                f.write("-" * 40 + "\n\n")
    
    async def _auto_save_messages(self):
        """Auto-save messages to file."""
        try:
            data = [msg.to_dict() for msg in self.messages]
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving messages: {e}")
    
    async def _auto_save_search(self):
        """Auto-save search history to file."""
        try:
            data = [search.to_dict() for search in self.search_history]
            with open(self.search_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving search history: {e}")
    
    async def _auto_save_bookmarks(self):
        """Auto-save bookmarks to file."""
        try:
            with open(self.bookmarks_file, 'w', encoding='utf-8') as f:
                json.dump(self.bookmarks, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving bookmarks: {e}")
    
    async def _auto_save_all(self):
        """Auto-save all data."""
        await asyncio.gather(
            self._auto_save_messages(),
            self._auto_save_search(),
            self._auto_save_bookmarks(),
            return_exceptions=True
        )
    
    def load_all(self):
        """Load all data from files."""
        self._load_messages()
        self._load_search_history()
        self._load_bookmarks()
    
    def _load_messages(self):
        """Load messages from file."""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.messages = [ChatMessage.from_dict(msg_data) for msg_data in data]
        except Exception as e:
            print(f"Error loading messages: {e}")
            self.messages = []
    
    def _load_search_history(self):
        """Load search history from file."""
        try:
            if self.search_file.exists():
                with open(self.search_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.search_history = [SearchResult.from_dict(search_data) for search_data in data]
        except Exception as e:
            print(f"Error loading search history: {e}")
            self.search_history = []
    
    def _load_bookmarks(self):
        """Load bookmarks from file."""
        try:
            if self.bookmarks_file.exists():
                with open(self.bookmarks_file, 'r', encoding='utf-8') as f:
                    self.bookmarks = json.load(f)
        except Exception as e:
            print(f"Error loading bookmarks: {e}")
            self.bookmarks = []
