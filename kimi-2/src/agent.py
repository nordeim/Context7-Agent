"""
Core agent implementation with Pydantic AI and MCP integration.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
import logging

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import ModelMessage, ModelResponse, UserPromptPart

from .config import config
from .history import HistoryManager

logger = logging.getLogger(__name__)

class Context7Agent:
    """Advanced AI agent with Context7 MCP integration."""
    
    def __init__(self):
        """Initialize the agent with MCP and OpenAI."""
        self.provider = OpenAIProvider(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )
        
        self.model = OpenAIModel(
            model_name=config.openai_model,
            provider=self.provider
        )
        
        self.mcp_config = config.get_mcp_config()
        self.mcp_server = MCPServerStdio(
            command="npx",
            args=["-y", "@upstash/context7-mcp@latest"]
        )
        
        self.agent = Agent(
            model=self.model,
            mcp_servers=[self.mcp_server],
            system_prompt="""You are Context7, an advanced AI assistant with access to a vast knowledge base.
            You can search, analyze, and present documents in real-time. Be helpful, concise, and engaging.
            When users ask about topics, use the search tools to find relevant documents and provide summaries."""
        )
        
        self.history = HistoryManager()
        self._conversation_id = None
    
    async def initialize(self):
        """Initialize the agent and MCP connection."""
        await self.history.load()
    
    async def chat_stream(
        self, 
        message: str, 
        conversation_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat responses with MCP integration."""
        
        try:
            async with self.agent.run_mcp_servers():
                messages = self.history.get_messages(conversation_id or "default")
                
                # Add user message
                messages.append({
                    "role": "user",
                    "content": message,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Stream response
                full_response = ""
                async for chunk in self.agent.run_stream(message):
                    content = chunk.data if hasattr(chunk, 'data') else str(chunk)
                    full_response += content
                    
                    yield {
                        "type": "content",
                        "data": content,
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Save to history
                messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": datetime.now().isoformat()
                })
                
                await self.history.save_message(
                    conversation_id or "default",
                    message,
                    full_response
                )
                
                yield {
                    "type": "complete",
                    "data": full_response,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Agent error: {e}")
            yield {
                "type": "error",
                "data": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def search_documents(
        self, 
        query: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search documents using MCP tools."""
        
        try:
            async with self.agent.run_mcp_servers():
                result = await self.agent.run(
                    f"Search for documents about: {query}. Return up to {limit} results."
                )
                
                # Parse MCP tool results
                if hasattr(result, 'data'):
                    return json.loads(result.data) if isinstance(result.data, str) else result.data
                
                return [{"title": "Search Result", "content": str(result)}]
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            return [{"error": str(e)}]
    
    def get_conversations(self) -> List[Dict[str, Any]]:
        """Get all conversation histories."""
        return self.history.get_conversations()
    
    async def clear_history(self, conversation_id: Optional[str] = None):
        """Clear conversation history."""
        await self.history.clear(conversation_id)
