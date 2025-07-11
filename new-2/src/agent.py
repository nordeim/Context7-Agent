# src/agent.py
"""
Agent module for the Context7 Agent.

This module implements a Pydantic AI agent with Context7 MCP server integration.
"""

import os
import sys
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from mcp import StdioServerParameters, stdio_client

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config


class Context7Agent:
    """
    Context7 Agent implementation using Pydantic AI.
    
    This agent integrates with the Context7 MCP server for enhanced context management
    and uses an OpenAI model as the underlying LLM provider.
    """
    
    def __init__(self):
        """Initialize the Context7 Agent."""
        # Validate configuration
        error = config.validate()
        if error:
            raise ValueError(f"Configuration error: {error}")
        
        # Initialize OpenAI model
        self.model = OpenAIModel(
            model=config.openai_model,
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
        )
        
        # Initialize the agent
        self.agent = Agent(
            model=self.model,
            system_prompt=self._get_system_prompt(),
        )
        
        # Initialize MCP client
        self.mcp_client = None
        self._init_mcp_client()
        
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        return """You are Context7 Agent, an AI assistant with advanced document search and management capabilities.
        
You have access to the Context7 MCP server which provides:
- Intelligent document discovery and search
- Content analysis and understanding
- Contextual search based on meaning
- Document recommendations

You can help users:
- Search for documents using natural language
- Find similar documents
- Manage bookmarks and sessions
- Analyze document content
- Provide insights based on document collections

Be helpful, precise, and proactive in assisting users with their document management needs."""

    def _init_mcp_client(self):
        """Initialize the MCP client for Context7."""
        try:
            # Create MCP server parameters
            server_params = StdioServerParameters(
                command="npx",
                args=["-y", "@upstash/context7-mcp@latest"]
            )
            
            # Initialize the client
            self.mcp_client = stdio_client(server_params)
            
        except Exception as e:
            print(f"Warning: Could not initialize Context7 MCP server: {e}")
            self.mcp_client = None
    
    async def search_documents(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for documents using Context7.
        
        Args:
            query: Search query
            filters: Optional filters (file_type, date_range, size, tags)
            
        Returns:
            List of matching documents
        """
        if not self.mcp_client:
            return []
        
        try:
            # Prepare search parameters
            params = {
                "query": query,
                "limit": 20,
            }
            
            if filters:
                params.update(filters)
            
            # Call Context7 search
            response = await self.mcp_client.call_tool("search", params)
            
            return response.get("results", [])
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    async def get_similar_documents(self, document_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get similar documents using Context7's AI capabilities.
        
        Args:
            document_id: ID of the reference document
            limit: Maximum number of similar documents to return
            
        Returns:
            List of similar documents
        """
        if not self.mcp_client:
            return []
        
        try:
            response = await self.mcp_client.call_tool("find_similar", {
                "document_id": document_id,
                "limit": limit
            })
            
            return response.get("similar_documents", [])
            
        except Exception as e:
            print(f"Similar documents error: {e}")
            return []
    
    async def analyze_document(self, document_id: str) -> Dict[str, Any]:
        """
        Analyze a document using Context7.
        
        Args:
            document_id: ID of the document to analyze
            
        Returns:
            Analysis results including summary, keywords, entities, etc.
        """
        if not self.mcp_client:
            return {}
        
        try:
            response = await self.mcp_client.call_tool("analyze", {
                "document_id": document_id
            })
            
            return response.get("analysis", {})
            
        except Exception as e:
            print(f"Document analysis error: {e}")
            return {}
    
    async def chat(self, message: str, context: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Chat with the agent.
        
        Args:
            message: User message
            context: Optional conversation context
            
        Returns:
            Agent response
        """
        try:
            # If context is provided, format it for the agent
            messages = []
            if context:
                for msg in context:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Add current message
            messages.append({
                "role": "user",
                "content": message
            })
            
            # Get response from agent
            result = await self.agent.run(message, message_history=messages[:-1] if context else None)
            
            return result.data
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def index_directory(self, path: str) -> Dict[str, Any]:
        """
        Index a directory for Context7.
        
        Args:
            path: Directory path to index
            
        Returns:
            Indexing results
        """
        if not self.mcp_client:
            return {"error": "MCP client not initialized"}
        
        try:
            response = await self.mcp_client.call_tool("index", {
                "path": path,
                "recursive": True
            })
            
            return response
            
        except Exception as e:
            return {"error": str(e)}
