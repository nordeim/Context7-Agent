"""
Stunning Terminal User Interface for the Context7 Agent.

Features rich visual themes, animations, real-time chat interface,
and comprehensive document management capabilities.
"""

import os
import sys
import asyncio
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Rich imports for stunning TUI
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.columns import Columns
from rich.prompt import Prompt, Confirm
from rich.progress import track
from rich.live import Live
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.tree import Tree
from rich import box

# Async imports
import anyio
from anyio import to_thread

# Local imports
from src.agent import Context7Agent, create_agent
from src.history import ConversationHistory, MessageRole
from src.themes import ThemeManager
from src.config import config

class Context7CLI:
    """
    Stunning Terminal User Interface for Context7 Agent.
    
    Features:
    - Beautiful themed interface with 4 visual themes
    - Real-time chat with streaming responses
    - Document search and management
    - Interactive commands and hotkeys
    - Session management and analytics
    - Smooth animations and visual effects
    """
    
    def __init__(self):
        """Initialize the CLI interface."""
        self.console = Console()
        self.theme_manager = ThemeManager(self.console)
        
        # Initialize agent and history
        self.agent: Optional[Context7Agent] = None
        self.history: Optional[ConversationHistory] = None
        
        # CLI state
        self.running = True
        self.current_session = "default"
        self.auto_search = True
        
        # Available commands
        self.commands = {
            "/help": self.show_help,
            "/theme": self.change_theme,
            "/themes": self.show_themes,
            "/search": self.manual_search,
            "/preview": self.preview_document,
            "/bookmark": self.bookmark_document,
            "/bookmarks": self.show_bookmarks,
            "/history": self.show_history,
            "/analytics": self.show_analytics,
            "/clear": self.clear_conversation,
            "/export": self.export_conversation,
            "/session": self.manage_session,
            "/quit": self.quit_application,
            "/exit": self.quit_application
        }
    
    async def run(self):
        """Main application entry point."""
        try:
            # Initialize components
            await self.initialize()
            
            # Show welcome screen
            self.theme_manager.show_welcome_screen()
            
            # Main interaction loop
            await self.main_loop()
            
        except KeyboardInterrupt:
            self.console.print("\n👋 Goodbye!", style="bold cyan")
        except Exception as e:
            self.theme_manager.show_error(f"Application error: {e}")
        finally:
            await self.cleanup()
    
    async def initialize(self):
        """Initialize agent and history components."""
        try:
            # Show initialization progress
            with self.console.status("[bold green]Initializing Context7 Agent...") as status:
                # Initialize agent
                status.update("[bold green]Setting up AI agent...")
                self.agent = await to_thread.run_sync(create_agent)
                
                # Initialize history
                status.update("[bold green]Loading conversation history...")
                self.history = ConversationHistory(config.data_dir)
                
                status.update("[bold green]Connecting to Context7 MCP server...")
                await asyncio.sleep(1)  # Simulate connection time
            
            self.theme_manager.show_success("Context7 Agent initialized successfully!")
            
        except Exception as e:
            self.theme_manager.show_error(f"Initialization failed: {e}")
            raise
    
    async def main_loop(self):
        """Main chat interaction loop."""
        # Show interface layout
        self.show_main_interface()
        
        while self.running:
            try:
                # Get user input with themed prompt
                user_input = await self.get_user_input()
                
                if not user_input.strip():
                    continue
                
                # Check for commands
                if user_input.startswith('/'):
                    await self.handle_command(user_input)
                    continue
                
                # Process chat message
                await self.process_chat_message(user_input)
                
            except KeyboardInterrupt:
                if await self.confirm_exit():
                    break
            except Exception as e:
                self.theme_manager.show_error(f"Error: {e}")
    
    def show_main_interface(self):
        """Display the main interface layout."""
        theme = self.theme_manager.get_theme()
        
        # Create main layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # Header
        header_text = self.theme_manager.create_gradient_text("Context7 AI Agent", "gradient")
        header = Panel(
            header_text,
            subtitle=f"Theme: {self.theme_manager.current_theme.title()} | Session: {self.current_session}",
            border_style=theme["primary"]
        )
        layout["header"].update(header)
        
        # Footer with commands
        footer_text = "Commands: /help /theme /search /bookmarks /analytics /quit"
        footer = Panel(
            footer_text,
            title="Quick Commands",
            border_style=theme["secondary"]
        )
        layout["footer"].update(footer)
        
        # Main area
        welcome_text = """
🚀 Welcome to Context7 AI Agent!

I'm your intelligent document discovery companion. You can:
• Chat naturally about any topic
• Ask questions and I'll search for relevant documents
• Use commands like /search, /bookmark, /theme
• Get AI-powered document recommendations

Try asking me about something you're curious about!
        """
        
        main_panel = Panel(
            welcome_text,
            title="💬 Ready to Chat",
            border_style=theme["accent"],
            padding=(1, 2)
        )
        layout["main"].update(main_panel)
        
        self.console.print(layout)
    
    async def get_user_input(self) -> str:
        """Get user input with themed styling."""
        theme = self.theme_manager.get_theme()
        
        # Create prompt with theme colors
        prompt_text = Text()
        prompt_text.append("You", style=f"bold {theme['primary']}")
        prompt_text.append(" > ", style=theme["accent"])
        
        # Get input in thread to avoid blocking
        user_input = await to_thread.run_sync(
            lambda: Prompt.ask(prompt_text, console=self.console)
        )
        
        return user_input
    
    async def process_chat_message(self, user_message: str):
        """Process a chat message with intent detection and response."""
        try:
            # Add user message to history
            self.history.add_user_message(user_message)
            
            # Show typing animation
            with self.console.status("[bold cyan]🤖 Context7 is thinking...") as status:
                # Detect intent for automatic search
                if self.auto_search:
                    status.update("[bold cyan]🔍 Analyzing intent...")
                    intent = await self.agent.detect_intent(user_message)
                    
                    if intent.get("has_search_intent", False) and intent.get("confidence", 0) > 0.7:
                        status.update("[bold cyan]🔍 Searching documents...")
                        search_query = intent.get("search_query", user_message)
                        search_results = await self.agent.search_documents(search_query)
                        
                        if search_results:
                            self.history.add_search_result(search_query, search_results)
                            self.show_search_results(search_results, search_query)
                
                # Get AI response
                status.update("[bold cyan]🤖 Generating response...")
                response = await self.agent.chat(user_message)
            
            # Add assistant response to history
            self.history.add_assistant_message(response)
            
            # Display response with theme styling
            self.show_chat_response(response)
            
        except Exception as e:
            error_msg = f"Chat error: {e}"
            self.theme_manager.show_error(error_msg)
            self.history.add_assistant_message(error_msg)
    
    def show_chat_response(self, response: str):
        """Display the AI response with beautiful formatting."""
        theme = self.theme_manager.get_theme()
        
        # Create response panel
        response_panel = Panel(
            response,
            title="🤖 Context7",
            border_style=theme["secondary"],
            padding=(1, 2)
        )
        
        self.console.print(response_panel)
    
    def show_search_results(self, results: List[Dict], query: str):
        """Display search results in a beautiful table."""
        theme = self.theme_manager.get_theme()
        
        # Create results table
        table = Table(title=f"🔍 Search Results for: '{query}'")
        table.add_column("ID", style=theme["primary"], width=8)
        table.add_column("Title", style=theme["text"], width=40)
        table.add_column("Type", style=theme["accent"], width=10)
        table.add_column("Score", style=theme["secondary"], width=8)
        
        for i, result in enumerate(results[:10]):  # Show top 10 results
            table.add_row(
                str(i + 1),
                result.get("title", "Unknown"),
                result.get("type", "document"),
                f"{result.get('score', 0):.2f}"
            )
        
        search_panel = Panel(
            table,
            title="📄 Document Results",
            border_style=theme["accent"]
        )
        
        self.console.print(search_panel)
        
        # Show command hints
        hint_text = "💡 Use /preview <id> to view a document, /bookmark <id> to save it"
        hint_panel = Panel(hint_text, border_style=theme["success"])
        self.console.print(hint_panel)
    
    async def handle_command(self, command_input: str):
        """Handle command input."""
        parts = command_input.split()
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if command in self.commands:
            try:
                await self.commands[command](args)
            except Exception as e:
                self.theme_manager.show_error(f"Command error: {e}")
        else:
            self.theme_manager.show_error(f"Unknown command: {command}")
            await self.show_help([])
    
    async def show_help(self, args: List[str]):
        """Show help information."""
        theme = self.theme_manager.get_theme()
        
        help_table = Table(title="📚 Context7 Agent Commands")
        help_table.add_column("Command", style=theme["primary"], width=15)
        help_table.add_column("Description", style=theme["text"], width=50)
        help_table.add_column("Example", style=theme["accent"], width=20)
        
        commands_info = [
            ("/help", "Show this help message", "/help"),
            ("/theme <name>", "Change visual theme", "/theme ocean"),
            ("/themes", "Show available themes", "/themes"),
            ("/search <query>", "Manual document search", "/search AI ethics"),
            ("/preview <id>", "Preview document", "/preview 1"),
            ("/bookmark <id>", "Bookmark document", "/bookmark 1"),
            ("/bookmarks", "Show bookmarks", "/bookmarks"),
            ("/history", "Show conversation history", "/history"),
            ("/analytics", "Show usage analytics", "/analytics"),
            ("/clear", "Clear conversation", "/clear"),
            ("/export", "Export conversation", "/export"),
            ("/session <name>", "Manage sessions", "/session work"),
            ("/quit", "Exit application", "/quit")
        ]
        
        for cmd, desc, example in commands_info:
            help_table.add_row(cmd, desc, example)
        
        help_panel = Panel(
            help_table,
            title="🆘 Help & Commands",
            border_style=theme["primary"]
        )
        
        self.console.print(help_panel)
    
    async def change_theme(self, args: List[str]):
        """Change the visual theme."""
        if not args:
            await self.show_themes([])
            return
        
        theme_name = args[0].lower()
        if self.theme_manager.set_theme(theme_name):
            self.theme_manager.show_success(f"Theme changed to: {theme_name}")
            
            # Show theme preview
            self.theme_manager.particle_animation(2.0)
        else:
            available = ", ".join(self.theme_manager.get_available_themes())
            self.theme_manager.show_error(f"Unknown theme. Available: {available}")
    
    async def show_themes(self, args: List[str]):
        """Show available themes."""
        theme = self.theme_manager.get_theme()
        
        themes_table = Table(title="🎨 Available Themes")
        themes_table.add_column("Theme", style=theme["primary"])
        themes_table.add_column("Description", style=theme["text"])
        themes_table.add_column("Colors", style=theme["accent"])
        
        theme_descriptions = {
            "cyberpunk": ("🔮 Cyberpunk", "Futuristic neon colors"),
            "ocean": ("🌊 Ocean", "Calm blue tones"),
            "forest": ("🌲 Forest", "Natural green palette"),
            "sunset": ("🌅 Sunset", "Warm orange hues")
        }
        
        for theme_name in self.theme_manager.get_available_themes():
            desc, colors = theme_descriptions.get(theme_name, (theme_name.title(), "Unknown"))
            current = " (current)" if theme_name == self.theme_manager.current_theme else ""
            themes_table.add_row(f"{desc}{current}", colors, f"/theme {theme_name}")
        
        self.console.print(Panel(themes_table, border_style=theme["accent"]))
    
    async def manual_search(self, args: List[str]):
        """Perform manual document search."""
        if not args:
            query = await to_thread.run_sync(
                lambda: Prompt.ask("Enter search query", console=self.console)
            )
        else:
            query = " ".join(args)
        
        if not query.strip():
            self.theme_manager.show_error("Search query cannot be empty")
            return
        
        try:
            with self.console.status(f"[bold cyan]🔍 Searching for: {query}..."):
                results = await self.agent.search_documents(query)
            
            if results:
                self.history.add_search_result(query, results)
                self.show_search_results(results, query)
            else:
                self.theme_manager.show_error("No documents found")
                
        except Exception as e:
            self.theme_manager.show_error(f"Search failed: {e}")
    
    async def preview_document(self, args: List[str]):
        """Preview a document."""
        if not args:
            self.theme_manager.show_error("Usage: /preview <document_id>")
            return
        
        doc_id = args[0]
        
        try:
            with self.console.status(f"[bold cyan]📄 Loading preview for document {doc_id}..."):
                preview = await self.agent.get_document_preview(doc_id)
            
            if preview:
                # Show preview in a panel with syntax highlighting
                syntax = Syntax(preview, "markdown", theme="monokai", line_numbers=True)
                preview_panel = Panel(
                    syntax,
                    title=f"📄 Document Preview - ID: {doc_id}",
                    border_style=self.theme_manager.get_theme()["accent"]
                )
                self.console.print(preview_panel)
            else:
                self.theme_manager.show_error(f"Could not load preview for document {doc_id}")
                
        except Exception as e:
            self.theme_manager.show_error(f"Preview failed: {e}")
    
    async def bookmark_document(self, args: List[str]):
        """Bookmark a document."""
        if not args:
            self.theme_manager.show_error("Usage: /bookmark <document_id> [notes]")
            return
        
        doc_id = args[0]
        notes = " ".join(args[1:]) if len(args) > 1 else ""
        
        try:
            # Get document title (this would come from search results)
            title = f"Document {doc_id}"
            
            # Add to agent bookmarks
            self.agent.add_bookmark(doc_id, title, notes)
            
            # Add to history bookmarks
            self.history.add_bookmark(doc_id, title, notes)
            
            self.theme_manager.show_success(f"Bookmarked document {doc_id}")
            
        except Exception as e:
            self.theme_manager.show_error(f"Bookmark failed: {e}")
    
    async def show_bookmarks(self, args: List[str]):
        """Show bookmarks."""
        bookmarks = self.history.get_bookmarks()
        
        if not bookmarks:
            self.theme_manager.show_error("No bookmarks found")
            return
        
        theme = self.theme_manager.get_theme()
        
        bookmarks_table = Table(title="🔖 Your Bookmarks")
        bookmarks_table.add_column("ID", style=theme["primary"])
        bookmarks_table.add_column("Title", style=theme["text"])
        bookmarks_table.add_column("Notes", style=theme["accent"])
        bookmarks_table.add_column("Date", style=theme["secondary"])
        
        for bookmark in bookmarks[-20:]:  # Show last 20 bookmarks
            bookmarks_table.add_row(
                bookmark["id"],
                bookmark["title"],
                bookmark.get("notes", "")[:30] + "..." if len(bookmark.get("notes", "")) > 30 else bookmark.get("notes", ""),
                bookmark["timestamp"][:10]  # Just the date
            )
        
        self.console.print(Panel(bookmarks_table, border_style=theme["accent"]))
    
    async def show_history(self, args: List[str]):
        """Show conversation history."""
        limit = 10
        if args and args[0].isdigit():
            limit = int(args[0])
        
        messages = self.history.get_messages(limit=limit)
        
        if not messages:
            self.theme_manager.show_error("No conversation history found")
            return
        
        theme = self.theme_manager.get_theme()
        
        history_table = Table(title=f"💬 Conversation History (Last {len(messages)} messages)")
        history_table.add_column("Time", style=theme["secondary"], width=10)
        history_table.add_column("Role", style=theme["primary"], width=10)
        history_table.add_column("Message", style=theme["text"], width=60)
        
        for message in messages:
            role_emoji = {"user": "👤", "assistant": "🤖", "system": "⚙️"}
            emoji = role_emoji.get(message.role.value, "❓")
            
            content = message.content[:60] + "..." if len(message.content) > 60 else message.content
            
            history_table.add_row(
                message.timestamp.strftime("%H:%M:%S"),
                f"{emoji} {message.role.value}",
                content
            )
        
        self.console.print(Panel(history_table, border_style=theme["accent"]))
    
    async def show_analytics(self, args: List[str]):
        """Show usage analytics."""
        analytics = self.history.get_analytics()
        agent_analytics = self.agent.get_search_analytics()
        
        theme = self.theme_manager.get_theme()
        
        # Create analytics panels
        stats_table = Table(title="📊 Usage Statistics")
        stats_table.add_column("Metric", style=theme["primary"])
        stats_table.add_column("Value", style=theme["accent"])
        
        stats_data = [
            ("Total Messages", str(analytics["total_messages"])),
            ("User Messages", str(analytics["user_messages"])),
            ("Assistant Messages", str(analytics["assistant_messages"])),
            ("Average Message Length", f"{analytics['average_message_length']:.1f} chars"),
            ("Recent Messages (24h)", str(analytics["recent_messages_24h"])),
            ("Total Searches", str(analytics["total_searches"])),
            ("Recent Searches (24h)", str(analytics["recent_searches_24h"])),
            ("Total Bookmarks", str(analytics["total_bookmarks"])),
        ]
        
        for metric, value in stats_data:
            stats_table.add_row(metric, value)
        
        analytics_panel = Panel(
            stats_table,
            title="📈 Analytics Dashboard",
            border_style=theme["primary"]
        )
        
        self.console.print(analytics_panel)
    
    async def clear_conversation(self, args: List[str]):
        """Clear conversation history."""
        confirm = await to_thread.run_sync(
            lambda: Confirm.ask("Are you sure you want to clear the conversation?", console=self.console)
        )
        
        if confirm:
            self.history.clear_all()
            self.agent.clear_conversation_history()
            self.theme_manager.show_success("Conversation history cleared")
    
    async def export_conversation(self, args: List[str]):
        """Export conversation to file."""
        format_type = args[0] if args else "json"
        
        if format_type not in ["json", "txt"]:
            self.theme_manager.show_error("Supported formats: json, txt")
            return
        
        filename = f"context7_export_{self.current_session}.{format_type}"
        filepath = config.data_dir / filename
        
        try:
            self.history.export_conversation(filepath, format_type)
            self.theme_manager.show_success(f"Exported to: {filepath}")
        except Exception as e:
            self.theme_manager.show_error(f"Export failed: {e}")
    
    async def manage_session(self, args: List[str]):
        """Manage sessions."""
        if not args:
            self.theme_manager.show_error("Usage: /session <session_name>")
            return
        
        session_name = args[0]
        
        # Save current session
        session_file = config.data_dir / f"session_{self.current_session}.json"
        self.agent.save_session(session_file)
        
        # Switch to new session
        self.current_session = session_name
        
        # Load new session if exists
        new_session_file = config.data_dir / f"session_{session_name}.json"
        if new_session_file.exists():
            self.agent.load_session(new_session_file)
            self.theme_manager.show_success(f"Loaded session: {session_name}")
        else:
            self.theme_manager.show_success(f"Created new session: {session_name}")
    
    async def quit_application(self, args: List[str]):
        """Quit the application."""
        self.running = False
        
        # Save session before quitting
        session_file = config.data_dir / f"session_{self.current_session}.json"
        try:
            self.agent.save_session(session_file)
        except Exception:
            pass  # Ignore save errors on exit
        
        self.theme_manager.show_success("Session saved. Goodbye! 👋")
    
    async def confirm_exit(self) -> bool:
        """Confirm application exit."""
        return await to_thread.run_sync(
            lambda: Confirm.ask("Do you want to exit Context7 Agent?", console=self.console)
        )
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Save final session
            if self.agent and self.current_session:
                session_file = config.data_dir / f"session_{self.current_session}.json"
                self.agent.save_session(session_file)
        except Exception:
            pass  # Ignore cleanup errors

# Application entry point
async def main():
    """Main application entry point."""
    cli = Context7CLI()
    await cli.run()

if __name__ == "__main__":
    # Use anyio for async execution (compatible with Pydantic AI)
    anyio.run(main)
