# src/agent.py
```py
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

```

# src/cli.py
```py
"""
Beautiful Terminal User Interface (TUI) for the Context7 Agent.

Provides an immersive, conversational experience with stunning themes,
smooth animations, and interactive elements.
"""

import asyncio
import sys
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

import anyio
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.columns import Columns

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import Context7Agent
from src.themes import get_theme, list_themes, AnimationEffects, Theme
from src.config import config
from src.utils import format_timestamp, truncate_text, highlight_search_terms, parse_hotkey_command

class Context7CLI:
    """
    Beautiful CLI interface for the Context7 Agent.
    
    Features stunning visual themes, smooth animations, and an immersive
    conversational experience with real-time document search capabilities.
    """
    
    def __init__(self):
        """Initialize the CLI with default settings."""
        self.console = Console()
        self.agent = Context7Agent()
        self.current_theme = get_theme(config.default_theme)
        self.is_running = False
        self.search_results: List[Dict[str, Any]] = []
        self.animation_speed = config.animation_speed
        
        # Layout components
        self.layout = Layout()
        self._setup_layout()
    
    def _setup_layout(self):
        """Setup the main layout structure."""
        self.layout.split_column(
            Layout(name="header", size=6),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        self.layout["main"].split_row(
            Layout(name="chat", ratio=2),
            Layout(name="results", ratio=1)
        )
    
    async def start(self):
        """Start the CLI application."""
        self.is_running = True
        
        try:
            # Initialize the agent
            await self.agent.initialize()
            
            # Show welcome screen with animation
            await self._show_welcome_screen()
            
            # Main interaction loop
            await self._main_loop()
            
        except KeyboardInterrupt:
            await self._handle_exit()
        except Exception as e:
            self.console.print(f"\n[red]Fatal error: {e}[/red]")
        finally:
            await self._cleanup()
    
    async def _show_welcome_screen(self):
        """Display animated welcome screen with ASCII art."""
        self.console.clear()
        
        # Show theme ASCII art
        welcome_panel = Panel(
            self.current_theme.ascii_art,
            title=f"[bold {self.current_theme.accent}]Welcome to Context7 Agent[/]",
            subtitle=f"[{self.current_theme.secondary}]Theme: {self.current_theme.name}[/]",
            border_style=self.current_theme.primary,
            padding=(1, 2)
        )
        
        self.console.print(welcome_panel, justify="center")
        
        # Loading animation
        with Progress(
            SpinnerColumn(spinner_name="dots12", style=self.current_theme.accent),
            TextColumn(f"[{self.current_theme.text}]Initializing AI Agent..."),
            console=self.console,
            transient=True
        ) as progress:
            task = progress.add_task("Loading", total=100)
            for i in range(100):
                progress.update(task, advance=1)
                await asyncio.sleep(0.02)
        
        # Show initial help
        help_text = """
🎯 **Getting Started:**
• Chat naturally: "Tell me about quantum computing"
• Use commands: /help, /theme, /bookmark, /analytics
• Search documents: "Find Python tutorials"
• Switch themes: /theme ocean

Type your message below to begin!
        """
        
        help_panel = Panel(
            Markdown(help_text),
            title=f"[bold {self.current_theme.accent}]Quick Start Guide[/]",
            border_style=self.current_theme.secondary,
            padding=(1, 2)
        )
        
        self.console.print(help_panel)
        self.console.print()
    
    async def _main_loop(self):
        """Main interaction loop with live updates."""
        conversation_messages = []
        
        while self.is_running:
            try:
                # Create live display
                with Live(self._create_main_display(conversation_messages), 
                         console=self.console, refresh_per_second=10) as live:
                    
                    # Get user input in a separate thread to avoid blocking
                    user_input = await anyio.to_thread.run_sync(
                        lambda: Prompt.ask(
                            f"[bold {self.current_theme.accent}]You[/]",
                            console=self.console
                        )
                    )
                    
                    if not user_input.strip():
                        continue
                    
                    # Handle exit commands
                    if user_input.lower() in ['/exit', '/quit', 'exit', 'quit']:
                        await self._handle_exit()
                        break
                    
                    # Add user message to conversation
                    conversation_messages.append({
                        "role": "user",
                        "content": user_input,
                        "timestamp": datetime.now()
                    })
                    
                    # Update display with user message
                    live.update(self._create_main_display(conversation_messages))
                    
                    # Show typing indicator
                    conversation_messages.append({
                        "role": "assistant",
                        "content": "🤖 Thinking...",
                        "timestamp": datetime.now(),
                        "is_typing": True
                    })
                    live.update(self._create_main_display(conversation_messages))
                    
                    # Process user input
                    await self._process_user_input(user_input, conversation_messages, live)
                    
            except KeyboardInterrupt:
                await self._handle_exit()
                break
            except Exception as e:
                error_msg = f"Error: {e}"
                conversation_messages.append({
                    "role": "system",
                    "content": error_msg,
                    "timestamp": datetime.now(),
                    "is_error": True
                })
    
    def _create_main_display(self, conversation_messages: List[Dict[str, Any]]) -> Layout:
        """Create the main display layout."""
        # Update header
        self.layout["header"].update(self._create_header())
        
        # Update chat area
        self.layout["chat"].update(self._create_chat_panel(conversation_messages))
        
        # Update results area
        self.layout["results"].update(self._create_results_panel())
        
        # Update footer
        self.layout["footer"].update(self._create_footer())
        
        return self.layout
    
    def _create_header(self) -> Panel:
        """Create the header panel."""
        current_time = datetime.now().strftime("%H:%M:%S")
        session_info = f"Session: {self.agent.current_session_id[:8] if self.agent.current_session_id else 'None'}"
        
        header_table = Table.grid(padding=1)
        header_table.add_column(justify="left")
        header_table.add_column(justify="center")
        header_table.add_column(justify="right")
        
        header_table.add_row(
            f"[{self.current_theme.accent}]Context7 Agent[/]",
            f"[{self.current_theme.secondary}]🤖 AI Document Assistant[/]",
            f"[{self.current_theme.text}]{current_time} | {session_info}[/]"
        )
        
        return Panel(
            header_table,
            style=self.current_theme.primary,
            border_style=self.current_theme.accent
        )
    
    def _create_chat_panel(self, messages: List[Dict[str, Any]]) -> Panel:
        """Create the chat conversation panel."""
        chat_content = Text()
        
        # Show recent messages (last 20)
        recent_messages = messages[-20:] if len(messages) > 20 else messages
        
        for i, message in enumerate(recent_messages):
            timestamp = message["timestamp"].strftime("%H:%M")
            role = message["role"]
            content = message["content"]
            
            # Style based on role
            if role == "user":
                chat_content.append(f"[{timestamp}] ", style=self.current_theme.secondary)
                chat_content.append("You: ", style=f"bold {self.current_theme.accent}")
                chat_content.append(f"{content}\n\n", style=self.current_theme.text)
            elif role == "assistant":
                if message.get("is_typing"):
                    chat_content.append(f"[{timestamp}] ", style=self.current_theme.secondary)
                    chat_content.append("🤖 Assistant: ", style=f"bold {self.current_theme.primary}")
                    chat_content.append(f"{content}\n\n", style=self.current_theme.accent)
                else:
                    chat_content.append(f"[{timestamp}] ", style=self.current_theme.secondary)
                    chat_content.append("🤖 Assistant: ", style=f"bold {self.current_theme.primary}")
                    chat_content.append(f"{content}\n\n", style=self.current_theme.text)
            elif role == "system":
                if message.get("is_error"):
                    chat_content.append(f"[{timestamp}] ", style=self.current_theme.secondary)
                    chat_content.append("⚠️  System: ", style=f"bold {self.current_theme.error}")
                    chat_content.append(f"{content}\n\n", style=self.current_theme.error)
                else:
                    chat_content.append(f"[{timestamp}] ", style=self.current_theme.secondary)
                    chat_content.append("ℹ️  System: ", style=f"bold {self.current_theme.warning}")
                    chat_content.append(f"{content}\n\n", style=self.current_theme.warning)
        
        return Panel(
            chat_content,
            title=f"[bold {self.current_theme.accent}]💬 Conversation[/]",
            border_style=self.current_theme.primary,
            padding=(1, 2)
        )
    
    def _create_results_panel(self) -> Panel:
        """Create the search results panel."""
        if not self.search_results:
            empty_content = Text(
                "🔍 Search results will appear here\n\n" +
                "Try asking:\n" +
                "• 'Tell me about Python'\n" +
                "• 'Find documentation on APIs'\n" +
                "• 'Search for tutorials'",
                style=self.current_theme.secondary
            )
            return Panel(
                empty_content,
                title=f"[bold {self.current_theme.accent}]📄 Search Results[/]",
                border_style=self.current_theme.secondary,
                padding=(1, 2)
            )
        
        results_content = Text()
        for i, result in enumerate(self.search_results[:5], 1):
            title = result.get("title", "Untitled")
            preview = truncate_text(result.get("content_preview", "No preview"), 80)
            file_type = result.get("file_type", "unknown")
            relevance = result.get("relevance_score", 0.0)
            
            results_content.append(f"{i}. ", style=self.current_theme.accent)
            results_content.append(f"{title}\n", style=f"bold {self.current_theme.text}")
            results_content.append(f"   {preview}\n", style=self.current_theme.secondary)
            results_content.append(f"   Type: {file_type} | Relevance: {relevance:.1%}\n\n", style=self.current_theme.warning)
        
        return Panel(
            results_content,
            title=f"[bold {self.current_theme.accent}]📄 Search Results ({len(self.search_results)})[/]",
            border_style=self.current_theme.primary,
            padding=(1, 2)
        )
    
    def _create_footer(self) -> Panel:
        """Create the footer panel with hotkeys."""
        footer_table = Table.grid(padding=1)
        footer_table.add_column(justify="left")
        footer_table.add_column(justify="right")
        
        hotkeys = "/help | /theme | /bookmark | /analytics | /exit"
        theme_info = f"Theme: {self.current_theme.name}"
        
        footer_table.add_row(
            f"[{self.current_theme.secondary}]Hotkeys: {hotkeys}[/]",
            f"[{self.current_theme.accent}]{theme_info}[/]"
        )
        
        return Panel(
            footer_table,
            style=self.current_theme.secondary,
            border_style=self.current_theme.accent
        )
    
    async def _process_user_input(self, user_input: str, conversation_messages: List[Dict[str, Any]], live):
        """Process user input and generate response."""
        try:
            # Save user message to history
            await self.agent.save_conversation_message("user", user_input)
            
            # Detect intent and handle special commands
            command, args = parse_hotkey_command(user_input)
            
            if command:
                response = await self._handle_command(command, args)
            else:
                # Generate AI response
                response = await self.agent.generate_response(user_input)
                
                # If this was a search query, update search results
                intent_data = await self.agent.detect_intent(user_input)
                if intent_data["intent"] == "search":
                    self.search_results = await self.agent.search_documents(intent_data["query"])
            
            # Remove typing indicator
            conversation_messages = [msg for msg in conversation_messages if not msg.get("is_typing")]
            
            # Add assistant response
            conversation_messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now()
            })
            
            # Save assistant message to history
            await self.agent.save_conversation_message("assistant", response)
            
            # Update live display
            live.update(self._create_main_display(conversation_messages))
            
        except Exception as e:
            # Remove typing indicator
            conversation_messages = [msg for msg in conversation_messages if not msg.get("is_typing")]
            
            # Add error message
            error_response = f"I apologize, but I encountered an error: {e}"
            conversation_messages.append({
                "role": "system",
                "content": error_response,
                "timestamp": datetime.now(),
                "is_error": True
            })
            
            # Update live display
            live.update(self._create_main_display(conversation_messages))
    
    async def _handle_command(self, command: str, args: str) -> str:
        """Handle special commands."""
        if command == "theme":
            return await self._handle_theme_command(args)
        elif command == "help":
            return self._get_help_text()
        elif command == "analytics":
            return await self.agent._get_analytics()
        elif command == "history":
            return await self._show_search_history()
        elif command == "bookmark":
            return await self._handle_bookmark_command(args)
        elif command == "sessions":
            return await self._show_sessions()
        elif command == "clear":
            self.console.clear()
            return "Screen cleared!"
        else:
            return f"Unknown command: /{command}. Type /help for available commands."
    
    async def _handle_theme_command(self, theme_name: str) -> str:
        """Handle theme change command."""
        if not theme_name:
            available_themes = ", ".join(list_themes())
            return f"Available themes: {available_themes}\nUsage: /theme <theme_name>"
        
        theme_name = theme_name.strip().lower()
        if theme_name in list_themes():
            self.current_theme = get_theme(theme_name)
            return f"Theme changed to: {self.current_theme.name}"
        else:
            available_themes = ", ".join(list_themes())
            return f"Unknown theme: {theme_name}\nAvailable themes: {available_themes}"
    
    async def _handle_bookmark_command(self, args: str) -> str:
        """Handle bookmark command."""
        if not args:
            return "Usage: /bookmark <title> - Bookmark the current search result or conversation"
        
        # Simple bookmark creation
        success = await self.agent.create_bookmark(
            title=args,
            file_path="conversation",
            description=f"Bookmarked from conversation: {args}",
            tags=["conversation"]
        )
        
        if success:
            return f"Bookmark created: {args}"
        else:
            return "Failed to create bookmark"
    
    async def _show_search_history(self) -> str:
        """Show recent search history."""
        try:
            recent_searches = await self.agent.search_history.get_recent_searches(10)
            if not recent_searches:
                return "No search history found."
            
            history_text = "Recent Searches:\n\n"
            for i, search in enumerate(recent_searches, 1):
                timestamp = format_timestamp(search.timestamp)
                history_text += f"{i}. {search.query} ({search.results_count} results) - {timestamp}\n"
            
            return history_text
        except Exception as e:
            return f"Error retrieving search history: {e}"
    
    async def _show_sessions(self) -> str:
        """Show available sessions."""
        try:
            sessions = await self.agent.session_manager.get_sessions()
            if not sessions:
                return "No sessions found."
            
            sessions_text = "Available Sessions:\n\n"
            for i, session in enumerate(sessions, 1):
                status = " (current)" if session.id == self.agent.current_session_id else ""
                last_activity = format_timestamp(session.last_activity)
                sessions_text += f"{i}. {session.name}{status} - Last activity: {last_activity}\n"
            
            return sessions_text
        except Exception as e:
            return f"Error retrieving sessions: {e}"
    
    def _get_help_text(self) -> str:
        """Get comprehensive help text."""
        return """
🎯 **Context7 Agent - Help Guide**

**💬 Natural Conversation:**
• Ask questions: "Tell me about quantum computing"
• Search documents: "Find Python tutorials"
• Request analysis: "Explain this code"

**⌨️  Commands:**
• `/help` - Show this help message
• `/theme [name]` - Change theme (cyberpunk, ocean, forest, sunset)
• `/bookmark [title]` - Save current result
• `/history` - Show search history
• `/sessions` - Show available sessions
• `/analytics` - View usage statistics
• `/clear` - Clear screen
• `/exit` - Exit application

**🔍 Search Tips:**
• Use natural language for better results
• Be specific about what you're looking for
• Use keywords from your domain

**🎨 Themes:**
• Cyberpunk - Neon colors and futuristic feel
• Ocean - Blue tones and calm atmosphere
• Forest - Green colors and natural vibe
• Sunset - Warm colors and cozy feel

**⚡ Pro Tips:**
• Type naturally - the AI understands context
• Use commands for quick actions
• Bookmark important findings
• Switch themes to match your mood!
        """
    
    async def _handle_exit(self):
        """Handle graceful exit."""
        self.is_running = False
        
        # Show exit animation
        exit_panel = Panel(
            Text("Thank you for using Context7 Agent!\n🚀 Happy exploring!", 
                 style=self.current_theme.accent, justify="center"),
            title=f"[bold {self.current_theme.primary}]Goodbye![/]",
            border_style=self.current_theme.accent,
            padding=(1, 2)
        )
        
        self.console.print(exit_panel, justify="center")
        
        # Cleanup
        await self._cleanup()
    
    async def _cleanup(self):
        """Cleanup resources."""
        try:
            await self.agent.cleanup()
        except Exception as e:
            self.console.print(f"[red]Cleanup error: {e}[/red]")

def main():
    """Main entry point for the CLI application."""
    try:
        cli = Context7CLI()
        anyio.run(cli.start)
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    main()

```

# src/config.py
```py
"""
Configuration module for the Context7 Agent.

Handles environment variables, API keys, and application settings.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Configuration class for the Context7 Agent."""
    
    # OpenAI Configuration
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    
    # Context7 MCP Configuration
    mcp_server_command: str
    mcp_server_args: list[str]
    
    # Application Settings
    max_history: int
    session_timeout: int
    default_theme: str
    animation_speed: float
    
    # File Paths
    history_file: str
    sessions_file: str
    bookmarks_file: str
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        return cls(
            # OpenAI Configuration
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            
            # Context7 MCP Configuration
            mcp_server_command=os.getenv("MCP_SERVER_COMMAND", "npx"),
            mcp_server_args=os.getenv("MCP_SERVER_ARGS", "-y,@upstash/context7-mcp@latest").split(","),
            
            # Application Settings
            max_history=int(os.getenv("MAX_HISTORY", "1000")),
            session_timeout=int(os.getenv("SESSION_TIMEOUT", "3600")),
            default_theme=os.getenv("DEFAULT_THEME", "cyberpunk"),
            animation_speed=float(os.getenv("ANIMATION_SPEED", "0.05")),
            
            # File Paths
            history_file=os.getenv("HISTORY_FILE", "data/history.json"),
            sessions_file=os.getenv("SESSIONS_FILE", "data/sessions.json"),
            bookmarks_file=os.getenv("BOOKMARKS_FILE", "data/bookmarks.json"),
        )
    
    def validate(self) -> Optional[str]:
        """Validate configuration and return error message if invalid."""
        if not self.openai_api_key:
            return "OpenAI API key is required"
        
        if not self.openai_model:
            return "OpenAI model is required"
            
        if not self.mcp_server_command:
            return "MCP server command is required"
            
        return None
    
    def to_mcp_config(self) -> Dict[str, Any]:
        """Convert to MCP server configuration format."""
        return {
            "command": self.mcp_server_command,
            "args": self.mcp_server_args
        }

# Global configuration instance
config = Config.from_env()

```

# src/history.py
```py
"""
History and session management for the Context7 Agent.

Handles conversation history, search history, bookmarks, and session persistence.
"""

import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from .utils import load_json_file, save_json_file, format_timestamp
from .config import config

@dataclass
class Message:
    """Represents a conversation message."""
    id: str
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create(cls, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> 'Message':
        """Create a new message."""
        return cls(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class SearchEntry:
    """Represents a search history entry."""
    id: str
    query: str
    results_count: int
    timestamp: float
    session_id: str
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create(cls, query: str, results_count: int, session_id: str, metadata: Optional[Dict[str, Any]] = None) -> 'SearchEntry':
        """Create a new search entry."""
        return cls(
            id=str(uuid.uuid4()),
            query=query,
            results_count=results_count,
            timestamp=time.time(),
            session_id=session_id,
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchEntry':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class Bookmark:
    """Represents a bookmarked document."""
    id: str
    title: str
    file_path: str
    description: str
    tags: List[str]
    timestamp: float
    session_id: str
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create(cls, title: str, file_path: str, description: str, tags: List[str], session_id: str, metadata: Optional[Dict[str, Any]] = None) -> 'Bookmark':
        """Create a new bookmark."""
        return cls(
            id=str(uuid.uuid4()),
            title=title,
            file_path=file_path,
            description=description,
            tags=tags,
            timestamp=time.time(),
            session_id=session_id,
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Bookmark':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class Session:
    """Represents a user session."""
    id: str
    name: str
    created_at: float
    last_activity: float
    theme: str
    settings: Dict[str, Any]
    
    @classmethod
    def create(cls, name: str, theme: str = "cyberpunk", settings: Optional[Dict[str, Any]] = None) -> 'Session':
        """Create a new session."""
        now = time.time()
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            created_at=now,
            last_activity=now,
            theme=theme,
            settings=settings or {}
        )
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """Create from dictionary."""
        return cls(**data)

class ConversationHistory:
    """Manages conversation history with persistence."""
    
    def __init__(self):
        self.messages: List[Message] = []
        self.current_session_id: Optional[str] = None
        self._loaded = False
    
    async def load(self) -> None:
        """Load conversation history from file."""
        if self._loaded:
            return
        
        data = await load_json_file(config.history_file, {"messages": []})
        self.messages = [Message.from_dict(msg_data) for msg_data in data.get("messages", [])]
        self._loaded = True
    
    async def save(self) -> bool:
        """Save conversation history to file."""
        try:
            # Keep only recent messages to prevent file from growing too large
            recent_messages = self.messages[-config.max_history:] if len(self.messages) > config.max_history else self.messages
            
            data = {
                "messages": [msg.to_dict() for msg in recent_messages],
                "saved_at": time.time()
            }
            
            return await save_json_file(config.history_file, data)
        except Exception:
            return False
    
    async def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add a new message to history."""
        await self.load()
        
        message = Message.create(role, content, metadata)
        if self.current_session_id:
            message.metadata = message.metadata or {}
            message.metadata["session_id"] = self.current_session_id
        
        self.messages.append(message)
        await self.save()
        return message
    
    async def get_recent_messages(self, limit: int = 50) -> List[Message]:
        """Get recent messages."""
        await self.load()
        return self.messages[-limit:] if len(self.messages) > limit else self.messages
    
    async def get_session_messages(self, session_id: str) -> List[Message]:
        """Get messages for a specific session."""
        await self.load()
        return [msg for msg in self.messages if msg.metadata and msg.metadata.get("session_id") == session_id]
    
    async def search_messages(self, query: str, limit: int = 20) -> List[Message]:
        """Search messages by content."""
        await self.load()
        query_lower = query.lower()
        
        matching_messages = [
            msg for msg in self.messages
            if query_lower in msg.content.lower()
        ]
        
        return matching_messages[-limit:] if len(matching_messages) > limit else matching_messages
    
    async def clear_session_messages(self, session_id: str) -> None:
        """Clear messages for a specific session."""
        await self.load()
        self.messages = [msg for msg in self.messages if not (msg.metadata and msg.metadata.get("session_id") == session_id)]
        await self.save()

class SearchHistory:
    """Manages search history with persistence."""
    
    def __init__(self):
        self.searches: List[SearchEntry] = []
        self._loaded = False
    
    async def load(self) -> None:
        """Load search history from file."""
        if self._loaded:
            return
        
        data = await load_json_file(config.history_file.replace("history.json", "search_history.json"), {"searches": []})
        self.searches = [SearchEntry.from_dict(search_data) for search_data in data.get("searches", [])]
        self._loaded = True
    
    async def save(self) -> bool:
        """Save search history to file."""
        try:
            # Keep only recent searches
            recent_searches = self.searches[-config.max_history:] if len(self.searches) > config.max_history else self.searches
            
            data = {
                "searches": [search.to_dict() for search in recent_searches],
                "saved_at": time.time()
            }
            
            filepath = config.history_file.replace("history.json", "search_history.json")
            return await save_json_file(filepath, data)
        except Exception:
            return False
    
    async def add_search(self, query: str, results_count: int, session_id: str, metadata: Optional[Dict[str, Any]] = None) -> SearchEntry:
        """Add a new search entry."""
        await self.load()
        
        search_entry = SearchEntry.create(query, results_count, session_id, metadata)
        self.searches.append(search_entry)
        await self.save()
        return search_entry
    
    async def get_recent_searches(self, limit: int = 20) -> List[SearchEntry]:
        """Get recent searches."""
        await self.load()
        return self.searches[-limit:] if len(self.searches) > limit else self.searches
    
    async def get_popular_queries(self, limit: int = 10) -> List[str]:
        """Get most popular search queries."""
        await self.load()
        
        query_counts = {}
        for search in self.searches:
            query_counts[search.query] = query_counts.get(search.query, 0) + 1
        
        # Sort by count and return top queries
        sorted_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
        return [query for query, count in sorted_queries[:limit]]

class BookmarkManager:
    """Manages bookmarks with persistence."""
    
    def __init__(self):
        self.bookmarks: List[Bookmark] = []
        self._loaded = False
    
    async def load(self) -> None:
        """Load bookmarks from file."""
        if self._loaded:
            return
        
        data = await load_json_file(config.bookmarks_file, {"bookmarks": []})
        self.bookmarks = [Bookmark.from_dict(bookmark_data) for bookmark_data in data.get("bookmarks", [])]
        self._loaded = True
    
    async def save(self) -> bool:
        """Save bookmarks to file."""
        try:
            data = {
                "bookmarks": [bookmark.to_dict() for bookmark in self.bookmarks],
                "saved_at": time.time()
            }
            
            return await save_json_file(config.bookmarks_file, data)
        except Exception:
            return False
    
    async def add_bookmark(self, title: str, file_path: str, description: str, tags: List[str], session_id: str, metadata: Optional[Dict[str, Any]] = None) -> Bookmark:
        """Add a new bookmark."""
        await self.load()
        
        bookmark = Bookmark.create(title, file_path, description, tags, session_id, metadata)
        self.bookmarks.append(bookmark)
        await self.save()
        return bookmark
    
    async def get_bookmarks(self, tag_filter: Optional[str] = None) -> List[Bookmark]:
        """Get bookmarks, optionally filtered by tag."""
        await self.load()
        
        if tag_filter:
            return [bookmark for bookmark in self.bookmarks if tag_filter in bookmark.tags]
        
        return self.bookmarks
    
    async def remove_bookmark(self, bookmark_id: str) -> bool:
        """Remove a bookmark by ID."""
        await self.load()
        
        original_count = len(self.bookmarks)
        self.bookmarks = [bookmark for bookmark in self.bookmarks if bookmark.id != bookmark_id]
        
        if len(self.bookmarks) < original_count:
            await self.save()
            return True
        
        return False
    
    async def search_bookmarks(self, query: str) -> List[Bookmark]:
        """Search bookmarks by title, description, or tags."""
        await self.load()
        query_lower = query.lower()
        
        matching_bookmarks = []
        for bookmark in self.bookmarks:
            if (query_lower in bookmark.title.lower() or 
                query_lower in bookmark.description.lower() or 
                any(query_lower in tag.lower() for tag in bookmark.tags)):
                matching_bookmarks.append(bookmark)
        
        return matching_bookmarks

class SessionManager:
    """Manages user sessions with persistence."""
    
    def __init__(self):
        self.sessions: List[Session] = []
        self.current_session: Optional[Session] = None
        self._loaded = False
    
    async def load(self) -> None:
        """Load sessions from file."""
        if self._loaded:
            return
        
        data = await load_json_file(config.sessions_file, {"sessions": []})
        self.sessions = [Session.from_dict(session_data) for session_data in data.get("sessions", [])]
        self._loaded = True
    
    async def save(self) -> bool:
        """Save sessions to file."""
        try:
            data = {
                "sessions": [session.to_dict() for session in self.sessions],
                "current_session_id": self.current_session.id if self.current_session else None,
                "saved_at": time.time()
            }
            
            return await save_json_file(config.sessions_file, data)
        except Exception:
            return False
    
    async def create_session(self, name: str, theme: str = "cyberpunk", settings: Optional[Dict[str, Any]] = None) -> Session:
        """Create a new session."""
        await self.load()
        
        session = Session.create(name, theme, settings)
        self.sessions.append(session)
        self.current_session = session
        await self.save()
        return session
    
    async def switch_session(self, session_id: str) -> Optional[Session]:
        """Switch to an existing session."""
        await self.load()
        
        for session in self.sessions:
            if session.id == session_id:
                session.update_activity()
                self.current_session = session
                await self.save()
                return session
        
        return None
    
    async def get_sessions(self) -> List[Session]:
        """Get all sessions."""
        await self.load()
        return sorted(self.sessions, key=lambda s: s.last_activity, reverse=True)
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        await self.load()
        
        original_count = len(self.sessions)
        self.sessions = [session for session in self.sessions if session.id != session_id]
        
        if len(self.sessions) < original_count:
            if self.current_session and self.current_session.id == session_id:
                self.current_session = None
            await self.save()
            return True
        
        return False
    
    async def auto_load_session(self) -> Optional[Session]:
        """Auto-load the most recent session."""
        await self.load()
        
        if not self.sessions:
            return await self.create_session("Default Session", config.default_theme)
        
        # Load most recent session
        most_recent = max(self.sessions, key=lambda s: s.last_activity)
        self.current_session = most_recent
        most_recent.update_activity()
        await self.save()
        return most_recent

```

# requirements.txt
```txt
aiofiles==24.1.0
anyio==4.9.0
asyncclick==8.1.8.0
click==8.2.1
openai==1.95.1
pydantic==2.11.7
pydantic-ai==0.4.2
pydantic-settings==2.10.1
pytest==8.4.1
pytest-asyncio==1.1.0
python-dotenv==1.1.1
rich==14.0.0
rich-gradient==0.3.2
```

# .env.example
```example
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

# Context7 MCP Server Configuration
MCP_SERVER_COMMAND=npx
MCP_SERVER_ARGS=-y,@upstash/context7-mcp@latest

# Application Settings
MAX_HISTORY=1000
SESSION_TIMEOUT=3600
DEFAULT_THEME=cyberpunk
ANIMATION_SPEED=0.05

# File Paths (relative to project root)
HISTORY_FILE=data/history.json
SESSIONS_FILE=data/sessions.json
BOOKMARKS_FILE=data/bookmarks.json
```

