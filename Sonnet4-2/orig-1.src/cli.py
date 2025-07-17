# File: src/cli.py
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
