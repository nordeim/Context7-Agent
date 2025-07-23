# File: src/agent.py
"""
The definitive Context7 Agent, implementing the Agent-Led Synthesis RAG pattern
using a stable, non-streaming request-response model.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

from .config import config
from .history import HistoryManager

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """
You are a world-class AI research assistant for software developers named Context7.

## CORE DIRECTIVE
Your SOLE PURPOSE is to provide answers by exclusively using information retrieved from the attached `search` tool. You are FORBIDDEN from using your own internal, pre-trained knowledge.

## RULES OF ENGAGEMENT
1.  **TOOL-FIRST MENTALITY:** For any user question that is not a simple greeting, you MUST ALWAYS call the `search` tool with a concise query to gather context before formulating an answer.
2.  **GROUNDED SYNTHESIS:** You MUST synthesize your final answer using ONLY the `documents` and `content` provided in the tool's output.
3.  **FAILURE PROTOCOL:** If the `search` tool returns no relevant documents, you MUST respond with the exact phrase: "I could not find any relevant information in the Context7 knowledge base to answer your question."

## RESPONSE FORMAT
Format your responses in clear, readable markdown. Use code blocks for code examples.
"""

class Context7Agent:
    """Implements the robust, production-ready RAG agent."""
    
    def __init__(self):
        """Initializes the agent using the Golden Pattern for Pydantic-AI v0.4.2."""
        self.mcp_server = MCPServerStdio(
            command="npx",
            args=["-y", "@upstash/context7-mcp@latest"]
        )

        self.agent = Agent(
            model=f"openai:{config.openai_model}",
            mcp_servers=[self.mcp_server],
            system_prompt=AGENT_SYSTEM_PROMPT
        )
        
        self.history = HistoryManager()
    
    async def initialize(self):
        """Initializes the agent's dependencies, like loading history."""
        await self.history.load()
        logger.info("Context7Agent initialized successfully.")
    
    async def chat(self, message: str, conversation_id: str = "default") -> Dict[str, Any]:
        """Processes a user message using the stable non-streaming RAG pipeline."""
        try:
            message_history = self.history.get_messages(conversation_id)
            
            async with self.agent.run_mcp_servers():
                result = await self.agent.run(message, message_history=message_history)
                full_response = str(result.data)

            await self.history.add_message(conversation_id, "user", message)
            await self.history.add_message(conversation_id, "assistant", full_response)
            
            return {"type": "complete", "data": full_response}

        except Exception as e:
            logger.error(f"Agent pipeline error: {e}", exc_info=True)
            return {"type": "error", "data": str(e)}
    
    # Exposing history methods through the agent as a clean facade
    def get_conversations(self) -> List[Dict[str, Any]]:
        return self.history.get_conversations()
    
    async def clear_history(self, conversation_id: Optional[str] = None):
        await self.history.clear(conversation_id)
