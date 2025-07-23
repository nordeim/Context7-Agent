# File: tests/test_agent.py
"""Comprehensive tests for the Context7Agent."""
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from pydantic_ai import RunResult

from src.agent import Context7Agent, AGENT_SYSTEM_PROMPT


class TestContext7Agent:
    """Test suite for Context7Agent."""
    
    def test_agent_initialization(self, agent_instance: Context7Agent):
        """Test agent initialization with correct configuration."""
        assert agent_instance.agent is not None
        assert "openai:gpt-4o-mini" in str(agent_instance.agent.model)
        assert AGENT_SYSTEM_PROMPT in agent_instance.agent.system_prompt
    
    @pytest.mark.asyncio
    async def test_agent_chat_success(self, agent_instance: Context7Agent):
        """Test successful agent chat interaction."""
        # Mock the agent.run method
        mock_result = MagicMock(spec=RunResult)
        mock_result.data = "Test response from AI"
        
        with patch.object(agent_instance.agent, 'run', return_value=mock_result):
            with patch.object(agent_instance.history, 'add_message', new_callable=AsyncMock) as mock_add:
                with patch.object(agent_instance.agent, 'run_mcp_servers', return_value=AsyncMock()):
                    result = await agent_instance.chat("test message", "test_conv")
        
        assert result["type"] == "complete"
        assert result["data"] == "Test response from AI"
        assert "timestamp" in result
        assert mock_add.call_count == 2  # User and assistant messages
    
    @pytest.mark.asyncio
    async def test_agent_chat_error_handling(self, agent_instance: Context7Agent):
        """Test error handling in agent chat."""
        with patch.object(agent_instance.agent, 'run', side_effect=Exception("API Error")):
            with patch.object(agent_instance.agent, 'run_mcp_servers', return_value=AsyncMock()):
                result = await agent_instance.chat("test message", "test_conv")
        
        assert result["type"] == "error"
        assert "API Error" in result["data"]
        assert "timestamp" in result
    
    @pytest.mark.asyncio
    async def test_agent_with_history(self, agent_instance: Context7Agent):
        """Test agent interaction with conversation history."""
        # Add test history
        await agent_instance.history.add_message("test_conv", "user", "Previous message")
        await agent_instance.history.add_message("test_conv", "assistant", "Previous response")
        
        # Mock the agent
        mock_result = MagicMock(spec=RunResult)
        mock_result.data = "Response considering history"
        
        with patch.object(agent_instance.agent, 'run', return_value=mock_result):
            with patch.object(agent_instance.history, 'add_message', new_callable=AsyncMock):
                with patch.object(agent_instance.agent, 'run_mcp_servers', return_value=AsyncMock()):
                    result = await agent_instance.chat("new message", "test_conv")
        
        assert result["type"] == "complete"
        history = agent_instance.history.get_messages("test_conv")
        assert len(history) == 2  # Should have previous messages
    
    def test_get_conversations(self, agent_instance: Context7Agent):
        """Test conversation listing functionality."""
        # Add some test data
        agent_instance.history._history = {
            "conv1": [{"role": "user", "content": "test", "timestamp": "2024-01-01"}],
            "conv2": [{"role": "user", "content": "test2", "timestamp": "2024-01-02"}]
        }
        
        conversations = agent_instance.get_conversations()
        assert len(conversations) == 2
        assert conversations[0]["id"] in ["conv1", "conv2"]
    
    @pytest.mark.asyncio
    async def test_clear_history(self, agent_instance: Context7Agent):
        """Test clearing conversation history."""
        # Add test data
        await agent_instance.history.add_message("test_conv", "user", "test")
        assert len(agent_instance.history.get_messages("test_conv")) > 0
        
        # Clear specific conversation
        await agent_instance.clear_history("test_conv")
        assert len(agent_instance.history.get_messages("test_conv")) == 0
        
        # Clear all conversations
        await agent_instance.history.add_message("test_conv2", "user", "test")
        await agent_instance.clear_history()
        assert len(agent_instance.history._history) == 0
    
    def test_system_prompt_content(self):
        """Test that system prompt contains required elements."""
        assert "Context7" in AGENT_SYSTEM_PROMPT
        assert "search" in AGENT_SYSTEM_PROMPT
        assert "FORBIDDEN" in AGENT_SYSTEM_PROMPT
        assert "MUST" in AGENT_SYSTEM_PROMPT
