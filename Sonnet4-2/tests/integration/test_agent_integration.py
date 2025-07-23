# File: tests/integration/test_agent_integration.py
"""Integration tests for the complete agent workflow."""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.agent import Context7Agent


class TestAgentIntegration:
    """Integration tests for full agent workflows."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_conversation_workflow(self):
        """Test complete conversation workflow."""
        agent = Context7Agent()
        
        # Initialize
        await agent.initialize()
        
        # Mock MCP and OpenAI
        with patch.object(agent.agent, 'run_mcp_servers', return_value=AsyncMock()):
            with patch.object(agent.agent, 'run', new_callable=AsyncMock) as mock_run:
                mock_run.return_value.data = "Integration test response"
                
                # Start conversation
                result1 = await agent.chat("Hello", "test_conv")
                assert result1["type"] == "complete"
                
                # Continue conversation
                result2 = await agent.chat("Tell me more", "test_conv")
                assert result2["type"] == "complete"
                
                # Check history
                history = agent.get_conversations()
                assert len(history) >= 1
                
                # Clear conversation
                await agent.clear_history("test_conv")
                
                # Verify cleared
                messages = agent.history.get_messages("test_conv")
                assert len(messages) == 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multiple_conversations(self):
        """Test handling multiple concurrent conversations."""
        agent = Context7Agent()
        await agent.initialize()
        
        with patch.object(agent.agent, 'run_mcp_servers', return_value=AsyncMock()):
            with patch.object(agent.agent, 'run', new_callable=AsyncMock) as mock_run:
                mock_run.return_value.data = "Response"
                
                # Multiple conversations
                results = await asyncio.gather(
                    agent.chat("Query 1", "conv1"),
                    agent.chat("Query 2", "conv2"),
                    agent.chat("Query 3", "conv3")
                )
                
                assert all(r["type"] == "complete" for r in results)
                
                # Check all conversations exist
                conversations = agent.get_conversations()
                conversation_ids = [c["id"] for c in conversations]
                assert "conv1" in conversation_ids
                assert "conv2" in conversation_ids
                assert "conv3" in conversation_ids
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_conversation_persistence(self):
        """Test that conversations persist across sessions."""
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            history_file = Path(tmp_dir) / "test_history.json"
            
            # First agent instance
            agent1 = Context7Agent()
            agent1.history.history_path = history_file
            await agent1.initialize()
            
            with patch.object(agent1.agent, 'run_mcp_servers', return_value=AsyncMock()):
                with patch.object(agent1.agent, 'run', new_callable=AsyncMock):
                    await agent1.chat("Persistent message", "persistent_conv")
            
            # Second agent instance
            agent2 = Context7Agent()
            agent2.history.history_path = history_file
            await agent2.initialize()
            
            # Should load existing history
            messages = agent2.history.get_messages("persistent_conv")
            assert len(messages) > 0
