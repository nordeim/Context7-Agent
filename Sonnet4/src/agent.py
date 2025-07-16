"""
Core Context7 Agent implementation using Pydantic AI.

This module implements the main AI agent with Context7 MCP server integration,
using proper Pydantic AI patterns and OpenAI model configuration.
"""

import os
import sys
import json
import asyncio
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerStdio
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.messages import ModelMessagesTypeAdapter
    from pydantic_core import to_jsonable_python
    import openai
except ImportError as e:
    print(f"Error importing Pydantic AI: {e}")
    print("Please install with: pip install pydantic-ai[openai]")
    sys.exit(1)

from src.config import config

class Context7Agent:
    """
    Advanced Context7 Agent with Pydantic AI and MCP integration.
    
    Features:
    - Conversational AI with OpenAI integration
    - Context7 MCP server for document search
    - Intent detection for automatic searches
    - Real-time streaming responses
    - Document analysis and recommendations
    """
    
    def __init__(self):
        """Initialize the Context7 Agent with proper error handling."""
        self.config = config
        
        # Validate configuration
        error = self.config.validate()
        if error:
            raise ValueError(f"Configuration error: {error}")
        
        # Initialize OpenAI components
        self._setup_openai_components()
        
        # Initialize Pydantic AI agent with MCP
        self._setup_pydantic_agent()
        
        # Agent state
        self.conversation_history = []
        self.search_results = []
        self.bookmarks = []
        self.session_data = {}
    
    def _setup_openai_components(self):
        """Set up OpenAI provider and model components."""
        try:
            # Create OpenAI provider
            self.provider = OpenAIProvider(
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_base_url
            )
            
            # Create OpenAI model for Pydantic AI
            self.llm = OpenAIModel(
                model_name=self.config.openai_model,
                provider=self.provider
            )
            
            # Create async OpenAI client for direct operations
            self.async_client = openai.AsyncOpenAI(
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_base_url
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI components: {e}")
    
    def _setup_pydantic_agent(self):
        """Set up the Pydantic AI agent with MCP server."""
        try:
            # Create MCP server configuration
            mcp_config = self.config.get_mcp_config()
            
            # Initialize MCP server
            self.mcp_server = MCPServerStdio(**mcp_config)
            
            # Create the Pydantic AI agent
            self.agent = Agent(
                model=self.llm,
                mcp_servers=[self.mcp_server],
                system_prompt=self._get_system_prompt()
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pydantic AI agent: {e}")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        return """You are Context7, an advanced AI agent specialized in contextual document discovery and analysis.

Your capabilities include:
1. Conversational interaction about any topic
2. Intelligent document search using Context7 MCP server
3. Document analysis and content understanding
4. Providing contextual recommendations
5. Bookmarking and organizing information

When users discuss topics, automatically search for relevant documents using your MCP tools.
Provide helpful, contextual responses and suggest related documents.
Be proactive in offering document insights and connections.

Always maintain a friendly, helpful, and professional tone.
"""
    
    async def chat(self, user_message: str) -> str:
        """
        Main chat interface with intent detection and MCP integration.
        
        Args:
            user_message: The user's input message
            
        Returns:
            The agent's response
        """
        try:
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            
            # Use Pydantic AI agent with MCP server lifecycle
            async with self.agent.run_mcp_servers():
                response = await self.agent.run(user_message)
                
                # Extract response content
                response_content = response.data if hasattr(response, 'data') else str(response)
                
                # Add to conversation history
                self.conversation_history.append({"role": "assistant", "content": response_content})
                
                return response_content
                
        except Exception as e:
            error_msg = f"Chat error: {str(e)}"
            # Add error to history for context
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            return error_msg
    
    async def search_documents(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Search for documents using Context7 MCP server.
        
        Args:
            query: Search query
            filters: Optional search filters
            
        Returns:
            List of search results
        """
        try:
            # This would use MCP tools through the agent
            search_prompt = f"Search for documents related to: {query}"
            if filters:
                search_prompt += f" with filters: {json.dumps(filters)}"
            
            async with self.agent.run_mcp_servers():
                response = await self.agent.run(search_prompt)
                
                # Parse search results (this would depend on actual MCP response format)
                results = self._parse_search_results(response)
                self.search_results = results
                
                return results
                
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def _parse_search_results(self, response) -> List[Dict]:
        """Parse search results from MCP response."""
        # This is a placeholder - actual implementation would depend on
        # the specific format returned by Context7 MCP server
        try:
            if hasattr(response, 'data'):
                return response.data if isinstance(response.data, list) else []
            else:
                return []
        except:
            return []
    
    async def detect_intent(self, message: str) -> Dict[str, Any]:
        """
        Detect user intent to trigger automatic actions.
        
        Args:
            message: User message to analyze
            
        Returns:
            Intent analysis results
        """
        try:
            intent_prompt = f"""Analyze this message for search intent: "{message}"

Determine if the user is asking about a topic that would benefit from document search.
Return a JSON response with:
{{
    "has_search_intent": boolean,
    "search_query": "extracted search terms",
    "confidence": 0.0-1.0,
    "suggested_filters": {{"file_type": "pdf", "date_range": "recent"}}
}}"""

            response = await self.async_client.chat.completions.create(
                model=self.config.openai_model,
                messages=[{"role": "user", "content": intent_prompt}],
                temperature=0.1
            )
            
            # Parse JSON response
            intent_text = response.choices[0].message.content
            try:
                intent_data = json.loads(intent_text)
                return intent_data
            except json.JSONDecodeError:
                return {"has_search_intent": False, "confidence": 0.0}
                
        except Exception as e:
            print(f"Intent detection error: {e}")
            return {"has_search_intent": False, "confidence": 0.0}
    
    async def get_document_preview(self, document_id: str) -> Optional[str]:
        """Get a preview of a specific document."""
        try:
            preview_prompt = f"Get preview for document ID: {document_id}"
            
            async with self.agent.run_mcp_servers():
                response = await self.agent.run(preview_prompt)
                return str(response)
                
        except Exception as e:
            print(f"Preview error: {e}")
            return None
    
    def add_bookmark(self, document_id: str, title: str, notes: str = ""):
        """Add a document to bookmarks."""
        bookmark = {
            "id": document_id,
            "title": title,
            "notes": notes,
            "timestamp": str(asyncio.get_event_loop().time())
        }
        self.bookmarks.append(bookmark)
    
    def remove_bookmark(self, document_id: str) -> bool:
        """Remove a bookmark."""
        initial_length = len(self.bookmarks)
        self.bookmarks = [b for b in self.bookmarks if b["id"] != document_id]
        return len(self.bookmarks) < initial_length
    
    def get_bookmarks(self) -> List[Dict]:
        """Get all bookmarks."""
        return self.bookmarks.copy()
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """Get search analytics and patterns."""
        return {
            "total_conversations": len(self.conversation_history),
            "total_searches": len(self.search_results),
            "total_bookmarks": len(self.bookmarks),
            "session_data": self.session_data
        }
    
    async def generate_similar_recommendations(self, document_id: str) -> List[Dict]:
        """Generate similar document recommendations."""
        try:
            rec_prompt = f"Find similar documents to ID: {document_id}"
            
            async with self.agent.run_mcp_servers():
                response = await self.agent.run(rec_prompt)
                return self._parse_search_results(response)
                
        except Exception as e:
            print(f"Recommendations error: {e}")
            return []
    
    def save_session(self, filepath: Path):
        """Save current session data."""
        session_data = {
            "conversation_history": self.conversation_history,
            "bookmarks": self.bookmarks,
            "search_results": self.search_results,
            "session_data": self.session_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def load_session(self, filepath: Path):
        """Load session data."""
        try:
            with open(filepath, 'r') as f:
                session_data = json.load(f)
            
            self.conversation_history = session_data.get("conversation_history", [])
            self.bookmarks = session_data.get("bookmarks", [])
            self.search_results = session_data.get("search_results", [])
            self.session_data = session_data.get("session_data", {})
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Session load error: {e}")

# Create global agent instance
def create_agent() -> Context7Agent:
    """Factory function to create a Context7 Agent instance."""
    return Context7Agent()
