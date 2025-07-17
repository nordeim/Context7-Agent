"""
Tests for the Context7Agent class.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import Context7Agent
from src.config import config

class TestContext7Agent:
    """Test cases for Context7Agent."""
    
    @pytest.fixture
    async def agent(self):
        """Create a test agent instance."""
        with patch('src.agent.config') as mock_config:
            mock_config.openai_api_key = "test-key"
            mock_config.openai_base_url = "https://api.openai.com/v1"
            mock_config.openai_model = "gpt-4o-mini"
            mock_config.to_mcp_config.return_value = {
                "command": "npx",
                "args": ["-y", "@upstash/context7-mcp@latest"]
            }
            mock_config.validate.return_value = None
            
            agent = Context7Agent()
            yield agent
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent is not None
        assert agent.agent is not None
        assert agent.llm is not None
        assert agent.provider is not None
    
    @pytest.mark.asyncio
    async def test_intent_detection_search(self, agent):
        """Test intent detection for search queries."""
        message = "Tell me about quantum computing"
        intent = await agent.detect_intent(message)
        
        assert intent["intent"] == "search"
        assert "quantum computing" in intent["query"]
        assert "keywords" in intent
    
    @pytest.mark.asyncio
    async def test_intent_detection_command(self, agent):
        """Test intent detection for commands."""
        message = "/help"
        intent = await agent.detect_intent(message)
        
        assert intent["intent"] == "command"
        assert intent["command"] == "help"
        assert intent["confidence"] == 1.0
    
    @pytest.mark.asyncio
    async def test_intent_detection_conversation(self, agent):
        """Test intent detection for general conversation."""
        message = "Hello, how are you?"
        intent = await agent.detect_intent(message)
        
        assert intent["intent"] == "conversation"
        assert intent["query"] == message
    
    @pytest.mark.asyncio
    async def test_search_documents(self, agent):
        """Test document search functionality."""
        with patch.object(agent.agent, 'run_mcp_servers') as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock()
            mock_mcp.return_value.__aexit__ = AsyncMock()
            
            with patch.object(agent.agent, 'run') as mock_run:
                mock_run.return_value.data = "Sample search results"
                
                results = await agent.search_documents("test query")
                
                assert isinstance(results, list)
                # Results should be parsed from the mock response
                if results:
                    assert "title" in results[0]
                    assert "file_path" in results[0]
    
    @pytest.mark.asyncio
    async def test_generate_response_search_intent(self, agent):
        """Test response generation for search intent."""
        with patch.object(agent, 'search_documents') as mock_search:
            mock_search.return_value = [
                {
                    "title": "Test Document",
                    "content_preview": "This is a test document about quantum computing.",
                    "file_path": "/test/path.md"
                }
            ]
            
            response = await agent.generate_response("Tell me about quantum computing")
            
            assert "Test Document" in response
            assert "quantum computing" in response
    
    @pytest.mark.asyncio
    async def test_generate_response_command(self, agent):
        """Test response generation for commands."""
        response = await agent.generate_response("/help")
        
        assert "Available commands" in response
        assert "/theme" in response
        assert "/bookmark" in response
    
    @pytest.mark.asyncio
    async def test_save_conversation_message(self, agent):
        """Test saving conversation messages."""
        with patch.object(agent.conversation_history, 'add_message') as mock_add:
            mock_add.return_value = Mock()
            
            await agent.save_conversation_message("user", "test message")
            
            mock_add.assert_called_once_with("user", "test message", None)
    
    @pytest.mark.asyncio
    async def test_create_bookmark(self, agent):
        """Test bookmark creation."""
        agent.current_session_id = "test-session"
        
        with patch.object(agent.bookmark_manager, 'add_bookmark') as mock_add:
            mock_add.return_value = Mock()
            
            result = await agent.create_bookmark(
                title="Test Bookmark",
                file_path="/test/path.md", 
                description="Test description",
                tags=["test"]
            )
            
            assert result is True
            mock_add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_document(self, agent):
        """Test document analysis."""
        with patch.object(agent.agent, 'run_mcp_servers') as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock()
            mock_mcp.return_value.__aexit__ = AsyncMock()
            
            with patch.object(agent.agent, 'run') as mock_run:
                mock_run.return_value.data = "Document analysis results"
                
                result = await agent.analyze_document("/test/path.md")
                
                assert "summary" in result
                assert "key_topics" in result
                assert "file_path" in result
                assert result["file_path"] == "/test/path.md"
    
    @pytest.mark.asyncio
    async def test_get_similar_documents(self, agent):
        """Test finding similar documents."""
        with patch.object(agent, 'search_documents') as mock_search:
            mock_search.return_value = [
                {
                    "id": "1",
                    "title": "Similar Doc 1",
                    "content_preview": "This document discusses machine learning algorithms.",
                    "file_path": "/test/doc1.md"
                },
                {
                    "id": "2", 
                    "title": "Similar Doc 2",
                    "content_preview": "This document covers neural networks.",
                    "file_path": "/test/doc2.md"
                }
            ]
            
            similar_docs = await agent.get_similar_documents("machine learning tutorial", limit=5)
            
            assert isinstance(similar_docs, list)
            assert len(similar_docs) <= 5
            
            # Check that similarity scores were calculated
            for doc in similar_docs:
                assert "similarity_score" in doc
