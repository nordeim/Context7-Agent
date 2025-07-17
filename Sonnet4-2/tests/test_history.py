"""
Tests for history and session management classes.
"""

import pytest
import tempfile
import os
from unittest.mock import patch, AsyncMock

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.history import (
    Message, SearchEntry, Bookmark, Session,
    ConversationHistory, SearchHistory, BookmarkManager, SessionManager
)

class TestMessage:
    """Test cases for Message class."""
    
    def test_message_creation(self):
        """Test message creation."""
        message = Message.create("user", "Hello world!")
        
        assert message.role == "user"
        assert message.content == "Hello world!"
        assert message.id is not None
        assert message.timestamp is not None
        assert message.metadata == {}
    
    def test_message_with_metadata(self):
        """Test message creation with metadata."""
        metadata = {"session_id": "test-session"}
        message = Message.create("assistant", "Hello!", metadata)
        
        assert message.metadata == metadata
    
    def test_message_serialization(self):
        """Test message to/from dict conversion."""
        message = Message.create("user", "Test message")
        message_dict = message.to_dict()
        
        assert "id" in message_dict
        assert "role" in message_dict
        assert "content" in message_dict
        assert "timestamp" in message_dict
        
        # Test deserialization
        restored_message = Message.from_dict(message_dict)
        assert restored_message.id == message.id
        assert restored_message.role == message.role
        assert restored_message.content == message.content

class TestConversationHistory:
    """Test cases for ConversationHistory class."""
    
    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)
    
    @pytest.mark.asyncio
    async def test_conversation_history_init(self):
        """Test conversation history initialization."""
        history = ConversationHistory()
        
        assert history.messages == []
        assert history.current_session_id is None
        assert history._loaded is False
    
    @pytest.mark.asyncio
    async def test_add_message(self, temp_file):
        """Test adding messages to history."""
        with patch('src.history.config') as mock_config:
            mock_config.history_file = temp_file
            mock_config.max_history = 1000
            
            history = ConversationHistory()
            
            message = await history.add_message("user", "Test message")
            
            assert message.role == "user"
            assert message.content == "Test message"
            assert len(history.messages) == 1
    
    @pytest.mark.asyncio
    async def test_get_recent_messages(self, temp_file):
        """Test getting recent messages."""
        with patch('src.history.config') as mock_config:
            mock_config.history_file = temp_file
            mock_config.max_history = 1000
            
            history = ConversationHistory()
            
            # Add multiple messages
            for i in range(10):
                await history.add_message("user", f"Message {i}")
            
            recent = await history.get_recent_messages(5)
            assert len(recent) == 5
            assert recent[-1].content == "Message 9"
    
    @pytest.mark.asyncio
    async def test_search_messages(self, temp_file):
        """Test searching messages."""
        with patch('src.history.config') as mock_config:
            mock_config.history_file = temp_file
            mock_config.max_history = 1000
            
            history = ConversationHistory()
            
            await history.add_message("user", "Tell me about Python")
            await history.add_message("assistant", "Python is a programming language")
            await history.add_message("user", "What about Java?")
            
            # Search for Python-related messages
            results = await history.search_messages("Python")
            assert len(results) == 2
            
            # Search for non-existent term
            results = await history.search_messages("nonexistent")
            assert len(results) == 0

class TestSessionManager:
    """Test cases for SessionManager class."""
    
    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)
    
    @pytest.mark.asyncio
    async def test_create_session(self, temp_file):
        """Test session creation."""
        with patch('src.history.config') as mock_config:
            mock_config.sessions_file = temp_file
            
            manager = SessionManager()
            
            session = await manager.create_session("Test Session", "cyberpunk")
            
            assert session.name == "Test Session"
            assert session.theme == "cyberpunk"
            assert session.id is not None
            assert manager.current_session == session
    
    @pytest.mark.asyncio
    async def test_switch_session(self, temp_file):
        """Test switching between sessions."""
        with patch('src.history.config') as mock_config:
            mock_config.sessions_file = temp_file
            
            manager = SessionManager()
            
            # Create two sessions
            session1 = await manager.create_session("Session 1")
            session2 = await manager.create_session("Session 2")
            
            # Switch back to session 1
            switched = await manager.switch_session(session1.id)
            
            assert switched == session1
            assert manager.current_session == session1
    
    @pytest.mark.asyncio
    async def test_delete_session(self, temp_file):
        """Test session deletion."""
        with patch('src.history.config') as mock_config:
            mock_config.sessions_file = temp_file
            
            manager = SessionManager()
            
            session = await manager.create_session("Test Session")
            session_id = session.id
            
            # Delete the session
            result = await manager.delete_session(session_id)
            
            assert result is True
            assert manager.current_session is None
            assert len(manager.sessions) == 0

class TestBookmarkManager:
    """Test cases for BookmarkManager class."""
    
    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)
    
    @pytest.mark.asyncio
    async def test_add_bookmark(self, temp_file):
        """Test adding bookmarks."""
        with patch('src.history.config') as mock_config:
            mock_config.bookmarks_file = temp_file
            
            manager = BookmarkManager()
            
            bookmark = await manager.add_bookmark(
                title="Test Document",
                file_path="/test/path.md",
                description="A test document",
                tags=["test", "python"],
                session_id="test-session"
            )
            
            assert bookmark.title == "Test Document"
            assert bookmark.file_path == "/test/path.md"
            assert "test" in bookmark.tags
            assert "python" in bookmark.tags
    
    @pytest.mark.asyncio
    async def test_search_bookmarks(self, temp_file):
        """Test searching bookmarks."""
        with patch('src.history.config') as mock_config:
            mock_config.bookmarks_file = temp_file
            
            manager = BookmarkManager()
            
            # Add multiple bookmarks
            await manager.add_bookmark("Python Guide", "/python.md", "Python tutorial", ["python"], "session1")
            await manager.add_bookmark("Java Tutorial", "/java.md", "Java basics", ["java"], "session1")
            await manager.add_bookmark("Python Advanced", "/advanced.md", "Advanced Python", ["python", "advanced"], "session1")
            
            # Search for Python bookmarks
            results = await manager.search_bookmarks("python")
            assert len(results) == 2
            
            # Search by tag
            python_bookmarks = await manager.get_bookmarks("python")
            assert len(python_bookmarks) == 2
    
    @pytest.mark.asyncio
    async def test_remove_bookmark(self, temp_file):
        """Test removing bookmarks."""
        with patch('src.history.config') as mock_config:
            mock_config.bookmarks_file = temp_file
            
            manager = BookmarkManager()
            
            bookmark = await manager.add_bookmark("Test", "/test.md", "Test", ["test"], "session1")
            bookmark_id = bookmark.id
            
            # Remove the bookmark
            result = await manager.remove_bookmark(bookmark_id)
            
            assert result is True
            assert len(manager.bookmarks) == 0
