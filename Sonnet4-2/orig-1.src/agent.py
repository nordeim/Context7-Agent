# File: src/agent.py
"""
Core Pydantic AI agent with Context7 MCP server integration.

This module implements the main AI agent that handles conversations,
intent detection, and document search via the Context7 MCP server.
"""

import os
import sys
import asyncio
from typing import Dict, Any, Optional, List, Union, AsyncIterator

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_core import to_jsonable_python

# Add the project root to the Python path to enable absolute imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .config import config
from .history import ConversationHistory, SearchHistory, BookmarkManager, SessionManager
from .utils import extract_keywords, calculate_similarity, parse_hotkey_command

class Context7Agent:
    """
    Context7 Agent implementation using Pydantic AI.

    This agent integrates with the Context7 MCP server for enhanced context management
    and uses an OpenAI model with OpenAIProvider as the underlying LLM provider.
    """

    def __init__(self):
        """
        Initialize the Context7 Agent with configuration from environment variables.

        Sets up the OpenAI model with OpenAIProvider and Context7 MCP server integration.
        """
        # Validate configuration
        error = config.validate()
        if error:
            raise ValueError(f"Configuration error: {error}")

        # Initialize OpenAI provider
        self.provider = OpenAIProvider(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )

        # Initialize OpenAI model
        self.llm = OpenAIModel(
            model_name=config.openai_model,
            provider=self.provider
        )

        # Initialize MCP server configuration
        mcp_config = config.to_mcp_config()
        
        # Initialize the agent with MCP server
        self.agent = Agent(
            model=self.llm,
            mcp_servers=[MCPServerStdio(**mcp_config)]
        )

        # Initialize history and session managers
        self.conversation_history = ConversationHistory()
        self.search_history = SearchHistory()
        self.bookmark_manager = BookmarkManager()
        self.session_manager = SessionManager()
        
        # Agent state
        self.current_session_id: Optional[str] = None
        self.is_running = False

    async def initialize(self) -> bool:
        """
        Initialize the agent and load necessary data.
        
        Returns:
            bool: True if initialization successful, False otherwise.
        """
        try:
            # Load current session
            current_session = await self.session_manager.auto_load_session()
            if current_session:
                self.current_session_id = current_session.id
                self.conversation_history.current_session_id = current_session.id

            # Load history
            await self.conversation_history.load()
            await self.search_history.load()
            await self.bookmark_manager.load()

            return True
        except Exception as e:
            print(f"Initialization error: {e}")
            return False

    async def detect_intent(self, message: str) -> Dict[str, Any]:
        """
        Detect user intent from message.
        
        Args:
            message: User input message
            
        Returns:
            Dict containing intent type and extracted parameters
        """
        # Check for hotkey commands first
        command, args = parse_hotkey_command(message)
        if command:
            return {
                "intent": "command",
                "command": command,
                "args": args,
                "confidence": 1.0
            }

        # Simple intent detection patterns
        message_lower = message.lower()
        
        # Search intents
        search_triggers = [
            "search for", "find", "look for", "tell me about", 
            "what is", "explain", "show me", "information about"
        ]
        
        if any(trigger in message_lower for trigger in search_triggers):
            keywords = extract_keywords(message)
            return {
                "intent": "search",
                "query": message,
                "keywords": keywords,
                "confidence": 0.8
            }

        # Bookmark intents
        if any(word in message_lower for word in ["bookmark", "save", "remember"]):
            return {
                "intent": "bookmark",
                "query": message,
                "confidence": 0.7
            }

        # General conversation
        return {
            "intent": "conversation",
            "query": message,
            "confidence": 0.6
        }

    async def search_documents(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for documents using the Context7 MCP server.
        
        Args:
            query: Search query
            filters: Optional search filters
            
        Returns:
            List of search results
        """
        try:
            # Use the agent's MCP server integration for search
            async with self.agent.run_mcp_servers():
                # Create a search prompt
                search_prompt = f"Search for documents related to: {query}"
                if filters:
                    search_prompt += f" with filters: {filters}"

                # Run the agent with the search prompt
                result = await self.agent.run(search_prompt)
                
                # Parse and return results
                # Note: This is a simplified implementation
                # In practice, you'd need to handle the specific MCP response format
                search_results = self._parse_search_results(result.data)
                
                # Record search in history
                if self.current_session_id:
                    await self.search_history.add_search(
                        query=query,
                        results_count=len(search_results),
                        session_id=self.current_session_id,
                        metadata={"filters": filters}
                    )

                return search_results

        except Exception as e:
            print(f"Search error: {e}")
            return []

    def _parse_search_results(self, raw_results: Any) -> List[Dict[str, Any]]:
        """
        Parse raw search results from MCP server.
        
        Args:
            raw_results: Raw results from MCP server
            
        Returns:
            Parsed and formatted search results
        """
        # This is a placeholder implementation
        # In practice, you'd parse the actual MCP response format
        if isinstance(raw_results, str):
            return [
                {
                    "id": "1",
                    "title": "Sample Document",
                    "file_path": "/path/to/document.md",
                    "content_preview": raw_results[:200] + "..." if len(raw_results) > 200 else raw_results,
                    "relevance_score": 0.95,
                    "file_type": "markdown",
                    "size": len(raw_results),
                    "metadata": {}
                }
            ]
        
        return []

    async def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a document using the AI agent.
        
        Args:
            file_path: Path to the document to analyze
            
        Returns:
            Document analysis results
        """
        try:
            async with self.agent.run_mcp_servers():
                analysis_prompt = f"Analyze the document at: {file_path}. Provide a summary, key topics, and insights."
                result = await self.agent.run(analysis_prompt)
                
                return {
                    "summary": result.data if isinstance(result.data, str) else str(result.data),
                    "key_topics": extract_keywords(result.data if isinstance(result.data, str) else str(result.data)),
                    "analysis_timestamp": asyncio.get_event_loop().time(),
                    "file_path": file_path
                }

        except Exception as e:
            print(f"Document analysis error: {e}")
            return {
                "summary": f"Error analyzing document: {e}",
                "key_topics": [],
                "analysis_timestamp": asyncio.get_event_loop().time(),
                "file_path": file_path
            }

    async def get_similar_documents(self, reference_doc: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find documents similar to a reference document.
        
        Args:
            reference_doc: Reference document content or path
            limit: Maximum number of similar documents to return
            
        Returns:
            List of similar documents
        """
        try:
            # Extract keywords from reference document
            keywords = extract_keywords(reference_doc)
            
            # Search for documents with similar keywords
            similar_docs = []
            for keyword in keywords[:3]:  # Use top 3 keywords
                results = await self.search_documents(keyword)
                similar_docs.extend(results)
            
            # Remove duplicates and calculate similarity scores
            unique_docs = {}
            for doc in similar_docs:
                doc_id = doc.get("id", doc.get("file_path", ""))
                if doc_id not in unique_docs:
                    # Calculate similarity score
                    doc_content = doc.get("content_preview", "")
                    similarity = calculate_similarity(reference_doc, doc_content)
                    doc["similarity_score"] = similarity
                    unique_docs[doc_id] = doc
            
            # Sort by similarity and return top results
            sorted_docs = sorted(
                unique_docs.values(),
                key=lambda x: x.get("similarity_score", 0),
                reverse=True
            )
            
            return sorted_docs[:limit]

        except Exception as e:
            print(f"Similar documents error: {e}")
            return []

    async def generate_response(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a conversational response to user message.
        
        Args:
            message: User message
            context: Optional conversation context
            
        Returns:
            AI-generated response
        """
        try:
            # Detect intent
            intent_data = await self.detect_intent(message)
            
            # Handle different intent types
            if intent_data["intent"] == "search":
                # Perform search and generate response
                search_results = await self.search_documents(intent_data["query"])
                
                if search_results:
                    response = f"I found {len(search_results)} documents related to your query. Here are the highlights:\n\n"
                    for i, result in enumerate(search_results[:3], 1):
                        response += f"{i}. **{result.get('title', 'Untitled')}**\n"
                        response += f"   {result.get('content_preview', 'No preview available')}\n\n"
                    response += "Would you like me to provide more details about any of these documents?"
                else:
                    response = "I couldn't find any documents matching your query. Could you try rephrasing or using different keywords?"
                
                return response

            elif intent_data["intent"] == "command":
                return await self._handle_command(intent_data["command"], intent_data["args"])

            else:
                # General conversation
                async with self.agent.run_mcp_servers():
                    # Include conversation history for context
                    recent_messages = await self.conversation_history.get_recent_messages(10)
                    context_prompt = ""
                    
                    if recent_messages:
                        context_prompt = "Previous conversation:\n"
                        for msg in recent_messages[-5:]:  # Last 5 messages
                            context_prompt += f"{msg.role}: {msg.content}\n"
                        context_prompt += "\n"

                    full_prompt = f"{context_prompt}User: {message}\n\nPlease provide a helpful and conversational response."
                    
                    result = await self.agent.run(full_prompt)
                    return result.data if isinstance(result.data, str) else str(result.data)

        except Exception as e:
            return f"I apologize, but I encountered an error while processing your message: {e}"

    async def _handle_command(self, command: str, args: str) -> str:
        """Handle hotkey commands."""
        if command == "help":
            return self._get_help_text()
        elif command == "theme":
            return f"Theme command received with args: {args}"
        elif command == "bookmark":
            return f"Bookmark command received with args: {args}"
        elif command == "history":
            recent_searches = await self.search_history.get_recent_searches(5)
            if recent_searches:
                response = "Recent searches:\n"
                for search in recent_searches:
                    response += f"• {search.query} ({search.results_count} results)\n"
                return response
            else:
                return "No search history found."
        elif command == "sessions":
            sessions = await self.session_manager.get_sessions()
            if sessions:
                response = "Available sessions:\n"
                for session in sessions:
                    status = " (current)" if session.id == self.current_session_id else ""
                    response += f"• {session.name}{status}\n"
                return response
            else:
                return "No sessions found."
        elif command == "analytics":
            return await self._get_analytics()
        else:
            return f"Unknown command: {command}. Type /help for available commands."

    def _get_help_text(self) -> str:
        """Get help text for available commands."""
        return """
Available commands:
• /help - Show this help message
• /theme [theme_name] - Change visual theme (cyberpunk, ocean, forest, sunset)
• /bookmark [title] - Bookmark current document or search result
• /history - Show recent search history
• /sessions - Show available sessions
• /analytics - Show usage analytics
• /exit - Exit the application

You can also chat naturally! Ask questions like:
• "Tell me about quantum computing"
• "Find documents about machine learning"
• "Search for Python tutorials"
        """

    async def _get_analytics(self) -> str:
        """Get usage analytics."""
        try:
            recent_searches = await self.search_history.get_recent_searches(100)
            bookmarks = await self.bookmark_manager.get_bookmarks()
            popular_queries = await self.search_history.get_popular_queries(5)
            
            analytics = f"""
📊 Usage Analytics:

🔍 Search Activity:
• Total searches: {len(recent_searches)}
• Popular queries: {', '.join(popular_queries) if popular_queries else 'None'}

📑 Bookmarks:
• Total bookmarks: {len(bookmarks)}

💬 Current Session:
• Session ID: {self.current_session_id or 'None'}
            """
            
            return analytics
        except Exception as e:
            return f"Error generating analytics: {e}"

    async def save_conversation_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save a message to conversation history."""
        await self.conversation_history.add_message(role, content, metadata)

    async def create_bookmark(self, title: str, file_path: str, description: str, tags: List[str]) -> bool:
        """Create a new bookmark."""
        try:
            if self.current_session_id:
                await self.bookmark_manager.add_bookmark(
                    title=title,
                    file_path=file_path,
                    description=description,
                    tags=tags,
                    session_id=self.current_session_id
                )
                return True
        except Exception:
            pass
        return False

    async def switch_session(self, session_name: str) -> bool:
        """Switch to a different session."""
        try:
            sessions = await self.session_manager.get_sessions()
            for session in sessions:
                if session.name.lower() == session_name.lower():
                    await self.session_manager.switch_session(session.id)
                    self.current_session_id = session.id
                    self.conversation_history.current_session_id = session.id
                    return True
            return False
        except Exception:
            return False

    async def cleanup(self) -> None:
        """Cleanup resources and save data."""
        try:
            await self.conversation_history.save()
            await self.search_history.save()
            await self.bookmark_manager.save()
            await self.session_manager.save()
        except Exception as e:
            print(f"Cleanup error: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        if hasattr(self, 'is_running') and self.is_running:
            # Note: This is not ideal for async cleanup, but provides a fallback
            pass
