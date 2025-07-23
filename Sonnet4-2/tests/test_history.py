# File: tests/test_history.py
"""Comprehensive tests for the HistoryManager."""
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from freezegun import freeze_time

from src.history import HistoryManager


class TestHistoryManager:
    """Test suite for HistoryManager."""
    
    @pytest.mark.asyncio
    async def test_load_empty_history(self, history_manager: HistoryManager):
        """Test loading history when file doesn't exist."""
        await history_manager.load()
        assert history_manager._history == {}
    
    @pytest.mark.asyncio
    async def test_load_existing_history(self, history_manager: HistoryManager, sample_history_data):
        """Test loading existing history from file."""
        # Create test file
        history_manager.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_manager.history_path, 'w') as f:
            json.dump(sample_history_data, f)
        
        await history_manager.load()
        assert history_manager._history == sample_history_data
    
    @pytest.mark.asyncio
    async def test_save_history(self, history_manager: HistoryManager):
        """Test saving history to file."""
        history_manager._history = {"test": [{"role": "user", "content": "test"}]}
        await history_manager.save()
        
        assert history_manager.history_path.exists()
        with open(history_manager.history_path) as f:
            data = json.load(f)
            assert data == history_manager._history
    
    @pytest.mark.asyncio
    async def test_add_message_new_conversation(self, history_manager: HistoryManager):
        """Test adding message to new conversation."""
        with freeze_time("2024-01-01 12:00:00"):
            await history_manager.add_message("new_conv", "user", "Hello")
        
        messages = history_manager._history["new_conv"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[0]["timestamp"] == "2024-01-01T12:00:00"
    
    @pytest.mark.asyncio
    async def test_add_message_existing_conversation(self, history_manager: HistoryManager):
        """Test adding message to existing conversation."""
        history_manager._history["existing"] = [{"role": "user", "content": "First"}]
        
        await history_manager.add_message("existing", "assistant", "Response")
        
        messages = history_manager._history["existing"]
        assert len(messages) == 2
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Response"
    
    @pytest.mark.asyncio
    async def test_history_limit_enforcement(self, history_manager: HistoryManager):
        """Test that history limit is enforced."""
        history_manager.max_history = 3
        
        # Add 5 messages
        for i in range(5):
            await history_manager.add_message("test", "user", f"Message {i}")
        
        messages = history_manager._history["test"]
        assert len(messages) == 3  # Should be limited
        assert messages[0]["content"] == "Message 2"
        assert messages[2]["content"] == "Message 4"
    
    def test_get_messages_format(self, history_manager: HistoryManager):
        """Test that get_messages returns correct format."""
        history_manager._history["test"] = [
            {"role": "user", "content": "Hello", "timestamp": "2024-01-01"},
            {"role": "assistant", "content": "Hi", "timestamp": "2024-01-01"}
        ]
        
        messages = history_manager.get_messages("test")
        assert len(messages) == 2
        
        # Should not include timestamp
        for msg in messages:
            assert "role" in msg
            assert "content" in msg
            assert "timestamp" not in msg
    
    def test_get_conversations(self, history_manager: HistoryManager):
        """Test conversation metadata retrieval."""
        history_manager._history = {
            "conv1": [
                {"role": "user", "content": "Short message", "timestamp": "2024-01-01T10:00:00"},
                {"role": "assistant", "content": "Short response", "timestamp": "2024-01-01T10:01:00"}
            ],
            "conv2": [
                {"role": "user", "content": "This is a very long message that should be truncated", "timestamp": "2024-01-01T09:00:00"}
            ]
        }
        
        conversations = history_manager.get_conversations()
        assert len(conversations) == 2
        
        # Check ordering (newest first)
        assert conversations[0]["id"] == "conv1"
        assert conversations[1]["id"] == "conv2"
        
        # Check truncation
        assert conversations[1]["last_message"] == "This is a very long message that should be tr..."
        assert conversations[1]["message_count"] == 1
    
    @pytest.mark.asyncio
    async def test_load_corrupted_file(self, history_manager: HistoryManager, capsys):
        """Test handling of corrupted history file."""
        # Create corrupted file
        history_manager.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_manager.history_path, 'w') as f:
            f.write("invalid json")
        
        await history_manager.load()
        assert history_manager._history == {}
        
        # Check warning message
        captured = capsys.readouterr()
        assert "Warning" in captured.out
    
    @pytest.mark.asyncio
    async def test_save_permission_error(self, history_manager: HistoryManager, capsys):
        """Test handling of permission errors during save."""
        # Mock file system error
        with patch('aiofiles.open', side_effect=PermissionError("Access denied")):
            await history_manager.save()
        
        captured = capsys.readouterr()
        assert "Warning" in captured.out
