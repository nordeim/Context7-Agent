# File: src/history.py
"""
History and session management for the Context7 Agent.

Handles conversation history, search history, bookmarks, and session persistence.
"""

import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from .utils import load_json_file, save_json_file, format_timestamp
from .config import config

@dataclass
class Message:
    """Represents a conversation message."""
    id: str
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create(cls, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> 'Message':
        """Create a new message."""
        return cls(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class SearchEntry:
    """Represents a search history entry."""
    id: str
    query: str
    results_count: int
    timestamp: float
    session_id: str
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create(cls, query: str, results_count: int, session_id: str, metadata: Optional[Dict[str, Any]] = None) -> 'SearchEntry':
        """Create a new search entry."""
        return cls(
            id=str(uuid.uuid4()),
            query=query,
            results_count=results_count,
            timestamp=time.time(),
            session_id=session_id,
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchEntry':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class Bookmark:
    """Represents a bookmarked document."""
    id: str
    title: str
    file_path: str
    description: str
    tags: List[str]
    timestamp: float
    session_id: str
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create(cls, title: str, file_path: str, description: str, tags: List[str], session_id: str, metadata: Optional[Dict[str, Any]] = None) -> 'Bookmark':
        """Create a new bookmark."""
        return cls(
            id=str(uuid.uuid4()),
            title=title,
            file_path=file_path,
            description=description,
            tags=tags,
            timestamp=time.time(),
            session_id=session_id,
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Bookmark':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class Session:
    """Represents a user session."""
    id: str
    name: str
    created_at: float
    last_activity: float
    theme: str
    settings: Dict[str, Any]
    
    @classmethod
    def create(cls, name: str, theme: str = "cyberpunk", settings: Optional[Dict[str, Any]] = None) -> 'Session':
        """Create a new session."""
        now = time.time()
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            created_at=now,
            last_activity=now,
            theme=theme,
            settings=settings or {}
        )
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """Create from dictionary."""
        return cls(**data)

class ConversationHistory:
    """Manages conversation history with persistence."""
    
    def __init__(self):
        self.messages: List[Message] = []
        self.current_session_id: Optional[str] = None
        self._loaded = False
    
    async def load(self) -> None:
        """Load conversation history from file."""
        if self._loaded:
            return
        
        data = await load_json_file(config.history_file, {"messages": []})
        self.messages = [Message.from_dict(msg_data) for msg_data in data.get("messages", [])]
        self._loaded = True
    
    async def save(self) -> bool:
        """Save conversation history to file."""
        try:
            # Keep only recent messages to prevent file from growing too large
            recent_messages = self.messages[-config.max_history:] if len(self.messages) > config.max_history else self.messages
            
            data = {
                "messages": [msg.to_dict() for msg in recent_messages],
                "saved_at": time.time()
            }
            
            return await save_json_file(config.history_file, data)
        except Exception:
            return False
    
    async def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add a new message to history."""
        await self.load()
        
        message = Message.create(role, content, metadata)
        if self.current_session_id:
            message.metadata = message.metadata or {}
            message.metadata["session_id"] = self.current_session_id
        
        self.messages.append(message)
        await self.save()
        return message
    
    async def get_recent_messages(self, limit: int = 50) -> List[Message]:
        """Get recent messages."""
        await self.load()
        return self.messages[-limit:] if len(self.messages) > limit else self.messages
    
    async def get_session_messages(self, session_id: str) -> List[Message]:
        """Get messages for a specific session."""
        await self.load()
        return [msg for msg in self.messages if msg.metadata and msg.metadata.get("session_id") == session_id]
    
    async def search_messages(self, query: str, limit: int = 20) -> List[Message]:
        """Search messages by content."""
        await self.load()
        query_lower = query.lower()
        
        matching_messages = [
            msg for msg in self.messages
            if query_lower in msg.content.lower()
        ]
        
        return matching_messages[-limit:] if len(matching_messages) > limit else matching_messages
    
    async def clear_session_messages(self, session_id: str) -> None:
        """Clear messages for a specific session."""
        await self.load()
        self.messages = [msg for msg in self.messages if not (msg.metadata and msg.metadata.get("session_id") == session_id)]
        await self.save()

class SearchHistory:
    """Manages search history with persistence."""
    
    def __init__(self):
        self.searches: List[SearchEntry] = []
        self._loaded = False
    
    async def load(self) -> None:
        """Load search history from file."""
        if self._loaded:
            return
        
        data = await load_json_file(config.history_file.replace("history.json", "search_history.json"), {"searches": []})
        self.searches = [SearchEntry.from_dict(search_data) for search_data in data.get("searches", [])]
        self._loaded = True
    
    async def save(self) -> bool:
        """Save search history to file."""
        try:
            # Keep only recent searches
            recent_searches = self.searches[-config.max_history:] if len(self.searches) > config.max_history else self.searches
            
            data = {
                "searches": [search.to_dict() for search in recent_searches],
                "saved_at": time.time()
            }
            
            filepath = config.history_file.replace("history.json", "search_history.json")
            return await save_json_file(filepath, data)
        except Exception:
            return False
    
    async def add_search(self, query: str, results_count: int, session_id: str, metadata: Optional[Dict[str, Any]] = None) -> SearchEntry:
        """Add a new search entry."""
        await self.load()
        
        search_entry = SearchEntry.create(query, results_count, session_id, metadata)
        self.searches.append(search_entry)
        await self.save()
        return search_entry
    
    async def get_recent_searches(self, limit: int = 20) -> List[SearchEntry]:
        """Get recent searches."""
        await self.load()
        return self.searches[-limit:] if len(self.searches) > limit else self.searches
    
    async def get_popular_queries(self, limit: int = 10) -> List[str]:
        """Get most popular search queries."""
        await self.load()
        
        query_counts = {}
        for search in self.searches:
            query_counts[search.query] = query_counts.get(search.query, 0) + 1
        
        # Sort by count and return top queries
        sorted_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
        return [query for query, count in sorted_queries[:limit]]

class BookmarkManager:
    """Manages bookmarks with persistence."""
    
    def __init__(self):
        self.bookmarks: List[Bookmark] = []
        self._loaded = False
    
    async def load(self) -> None:
        """Load bookmarks from file."""
        if self._loaded:
            return
        
        data = await load_json_file(config.bookmarks_file, {"bookmarks": []})
        self.bookmarks = [Bookmark.from_dict(bookmark_data) for bookmark_data in data.get("bookmarks", [])]
        self._loaded = True
    
    async def save(self) -> bool:
        """Save bookmarks to file."""
        try:
            data = {
                "bookmarks": [bookmark.to_dict() for bookmark in self.bookmarks],
                "saved_at": time.time()
            }
            
            return await save_json_file(config.bookmarks_file, data)
        except Exception:
            return False
    
    async def add_bookmark(self, title: str, file_path: str, description: str, tags: List[str], session_id: str, metadata: Optional[Dict[str, Any]] = None) -> Bookmark:
        """Add a new bookmark."""
        await self.load()
        
        bookmark = Bookmark.create(title, file_path, description, tags, session_id, metadata)
        self.bookmarks.append(bookmark)
        await self.save()
        return bookmark
    
    async def get_bookmarks(self, tag_filter: Optional[str] = None) -> List[Bookmark]:
        """Get bookmarks, optionally filtered by tag."""
        await self.load()
        
        if tag_filter:
            return [bookmark for bookmark in self.bookmarks if tag_filter in bookmark.tags]
        
        return self.bookmarks
    
    async def remove_bookmark(self, bookmark_id: str) -> bool:
        """Remove a bookmark by ID."""
        await self.load()
        
        original_count = len(self.bookmarks)
        self.bookmarks = [bookmark for bookmark in self.bookmarks if bookmark.id != bookmark_id]
        
        if len(self.bookmarks) < original_count:
            await self.save()
            return True
        
        return False
    
    async def search_bookmarks(self, query: str) -> List[Bookmark]:
        """Search bookmarks by title, description, or tags."""
        await self.load()
        query_lower = query.lower()
        
        matching_bookmarks = []
        for bookmark in self.bookmarks:
            if (query_lower in bookmark.title.lower() or 
                query_lower in bookmark.description.lower() or 
                any(query_lower in tag.lower() for tag in bookmark.tags)):
                matching_bookmarks.append(bookmark)
        
        return matching_bookmarks

class SessionManager:
    """Manages user sessions with persistence."""
    
    def __init__(self):
        self.sessions: List[Session] = []
        self.current_session: Optional[Session] = None
        self._loaded = False
    
    async def load(self) -> None:
        """Load sessions from file."""
        if self._loaded:
            return
        
        data = await load_json_file(config.sessions_file, {"sessions": []})
        self.sessions = [Session.from_dict(session_data) for session_data in data.get("sessions", [])]
        self._loaded = True
    
    async def save(self) -> bool:
        """Save sessions to file."""
        try:
            data = {
                "sessions": [session.to_dict() for session in self.sessions],
                "current_session_id": self.current_session.id if self.current_session else None,
                "saved_at": time.time()
            }
            
            return await save_json_file(config.sessions_file, data)
        except Exception:
            return False
    
    async def create_session(self, name: str, theme: str = "cyberpunk", settings: Optional[Dict[str, Any]] = None) -> Session:
        """Create a new session."""
        await self.load()
        
        session = Session.create(name, theme, settings)
        self.sessions.append(session)
        self.current_session = session
        await self.save()
        return session
    
    async def switch_session(self, session_id: str) -> Optional[Session]:
        """Switch to an existing session."""
        await self.load()
        
        for session in self.sessions:
            if session.id == session_id:
                session.update_activity()
                self.current_session = session
                await self.save()
                return session
        
        return None
    
    async def get_sessions(self) -> List[Session]:
        """Get all sessions."""
        await self.load()
        return sorted(self.sessions, key=lambda s: s.last_activity, reverse=True)
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        await self.load()
        
        original_count = len(self.sessions)
        self.sessions = [session for session in self.sessions if session.id != session_id]
        
        if len(self.sessions) < original_count:
            if self.current_session and self.current_session.id == session_id:
                self.current_session = None
            await self.save()
            return True
        
        return False
    
    async def auto_load_session(self) -> Optional[Session]:
        """Auto-load the most recent session."""
        await self.load()
        
        if not self.sessions:
            return await self.create_session("Default Session", config.default_theme)
        
        # Load most recent session
        most_recent = max(self.sessions, key=lambda s: s.last_activity)
        self.current_session = most_recent
        most_recent.update_activity()
        await self.save()
        return most_recent
