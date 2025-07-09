# File: src/explorer_cli.py
"""
Main CLI application for Context7 Document Explorer.
"""

import asyncio
import os
from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.prompt import Prompt
import click
from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from src.ui.dashboard import Dashboard
from src.context7_integration import Context7Manager, SearchQuery, Document
from src.data.history_manager import HistoryManager
from src.data.bookmarks import BookmarkManager
from src.data.session_manager import SessionManager
from src.config import config


class Context7Explorer:
    """Main application class for Context7 Document Explorer."""
    
    def __init__(self):
        self.console = Console()
        self.dashboard = Dashboard(self.console)
        self.context7 = Context7Manager()
        
        # Data managers
        self.history = HistoryManager(config.data_dir / config.history_file)
        self.bookmarks = BookmarkManager(config.data_dir / config.bookmarks_file)
        self.sessions = SessionManager(config.data_dir / config.sessions_dir)
        
        # Application state
        self.running = True
        self.current_session = None
        
        # Key bindings
        self.kb = self._create_key_bindings()
    
    def _create_key_bindings(self) -> KeyBindings:
        """Create keyboard shortcuts."""
        kb = KeyBindings()
        
        @kb.add('/')
        def search_mode(event):
            """Enter search mode."""
            asyncio.create_task(self.enter_search_mode())
        
        @kb.add('escape')
        def go_back(event):
            """Go back to previous view."""
            asyncio.create_task(self.go_back())
        
        @kb.add('enter')
        def select_item(event):
            """Select current item."""
            asyncio.create_task(self.select_current())
        
        @kb.add('up')
        def move_up(event):
            """Move selection up."""
            self.dashboard.selected_index = max(0, self.dashboard.selected_index - 1)
        
        @kb.add('down')
        def move_down(event):
            """Move selection down."""
            max_index = len(self.dashboard.search_results) - 1
            self.dashboard.selected_index = min(max_index, self.dashboard.selected_index + 1)
        
        @kb.add('c-b')
        def show_bookmarks(event):
            """Show bookmarks."""
            asyncio.create_task(self.show_bookmarks())
        
        @kb.add('c-h')
        def show_history(event):
            """Show search history."""
            asyncio.create_task(self.show_history())
        
        @kb.add('c-s')
        def save_session(event):
            """Save current session."""
            asyncio.create_task(self.save_session())
        
        @kb.add('c-q')
        def quit_app(event):
            """Quit application."""
            self.running = False
        
        return kb
    
    async def initialize(self):
        """Initialize the application."""
        self.console.clear()
        
        # Show splash screen
        if config.animations_enabled:
            await self._show_splash_screen()
        
        # Initialize Context7
        self.console.print("[cyan]Initializing Context7 integration...[/cyan]")
        success = await self.context7.initialize()
        
        if not success:
            self.console.print("[red]Failed to initialize Context7. Running in offline mode.[/red]")
        else:
            self.console.print("[green]✓ Context7 initialized successfully![/green]")
        
        # Load last session if available
        last_session = self.sessions.get_last_session()
        if last_session:
            self.current_session = last_session
            self.console.print(f"[dim]Restored session: {last_session.name}[/dim]")
    
    async def _show_splash_screen(self):
        """Show animated splash screen."""
        frames = [
            "⚡", "⚡C", "⚡CO", "⚡CON", "⚡CONT", "⚡CONTE", "⚡CONTEX", "⚡CONTEXT", 
            "⚡CONTEXT7", "⚡CONTEXT7 ⚡"
        ]
        
        for frame in frames:
            self.console.clear()
            self.console.print(
                f"\n\n\n[bold cyan]{frame}[/bold cyan]",
                justify="center"
            )
            await asyncio.sleep(0.1)
        
        await asyncio.sleep(0.5)
    
    async def enter_search_mode(self):
        """Enter search mode."""
        self.dashboard.current_view = "search"
        
        # Get search input
        query = await self._get_search_input()
        
        if query:
            await self.perform_search(query)
    
    async def _get_search_input(self) -> Optional[str]:
        """Get search input from user."""
        # In a real implementation, this would use prompt_toolkit for better input handling
        # For now, using simple input
        self.console.print("\n[cyan]Enter search query:[/cyan]")
        query = input("> ").strip()
        
        return query if query else None
    
    async def perform_search(self, query: str):
        """Perform document search."""
        self.dashboard.search.current_query = query
        self.dashboard.is_searching = True
        self.dashboard.current_view = "results"
        
        # Add to history
        self.history.add_search(query)
        
        # Create search query
        search_query = SearchQuery(
            query=query,
            limit=config.max_results,
            include_content=False
        )
        
        # Perform search
        results = await self.context7.search_documents(search_query)
        
        # Convert to display format
        self.dashboard.search_results = [
            {
                "id": doc.id,
                "title": doc.title,
                "path": doc.path,
                "preview": doc.preview,
                "score": doc.score,
                "metadata": doc.metadata
            }
            for doc in results
        ]
        
        self.dashboard.is_searching = False
        self.dashboard.selected_index = 0
        
        # Show results notification
        if results:
            self.dashboard.status_bar.update("Status", f"Found {len(results)} documents")
        else:
            self.dashboard.status_bar.update("Status", "No documents found")
    
    async def select_current(self):
        """Select the currently highlighted item."""
        if self.dashboard.current_view == "results" and self.dashboard.search_results:
            if 0 <= self.dashboard.selected_index < len(self.dashboard.search_results):
                doc = self.dashboard.search_results[self.dashboard.selected_index]
                await self.view_document(doc["id"])
    
    async def view_document(self, doc_id: str):
        """View a specific document."""
        self.dashboard.current_view = "document"
        
        # Load full document content
        content = await self.context7.get_document_content(doc_id)
        
        if content:
            # Update the selected document's content
            for doc in self.dashboard.search_results:
                if doc["id"] == doc_id:
                    doc["content"] = content
                    break
    
    async def go_back(self):
        """Go back to previous view."""
        if self.dashboard.current_view == "document":
            self.dashboard.current_view = "results"
        elif self.dashboard.current_view in ["search", "results"]:
            self.dashboard.current_view = "welcome"
    
    async def show_bookmarks(self):
        """Show bookmarked documents."""
        bookmarks = self.bookmarks.get_all()
        
        if bookmarks:
            # Convert bookmarks to search results format
            self.dashboard.search_results = [
                {
                    "id": b.doc_id,
                    "title": b.title,
                    "path": b.path,
                    "preview": b.notes or "Bookmarked document",
                    "score": 1.0,
                    "metadata": {"bookmarked": True}
                }
                for b in bookmarks
            ]
            
            self.dashboard.current_view = "results"
            self.dashboard.search.current_query = "Bookmarks"
        else:
            self.dashboard.status_bar.update("Status", "No bookmarks found")
    
    async def show_history(self):
        """Show search history."""
        history = self.history.get_recent_searches(20)
        
        if history:
            # Display history in a special view
            self.console.print("\n[bold]Recent Searches:[/bold]")
            for i, search in enumerate(history, 1):
                self.console.print(
                    f"{i}. {search.query} [dim]({search.timestamp.strftime('%Y-%m-%d %H:%M')})[/dim]"
                )
            
            # Allow selection
            choice = Prompt.ask("Select a search to repeat (or press Enter to cancel)")
            if choice.isdigit() and 1 <= int(choice) <= len(history):
                selected = history[int(choice) - 1]
                await self.perform_search(selected.query)
    
    async def save_session(self):
        """Save current session."""
        if self.dashboard.search_results:
            session_name = Prompt.ask("Session name", default="Quick Save")
            
            session_data = {
                "query": self.dashboard.search.current_query,
                "results": self.dashboard.search_results,
                "selected_index": self.dashboard.selected_index,
                "view": self.dashboard.current_view
            }
            
            self.sessions.save_session(session_name, session_data)
            self.dashboard.status_bar.update("Status", f"Session '{session_name}' saved")
    
    async def run(self):
        """Run the main application loop."""
        await self.initialize()
        
        # Create async tasks
        dashboard_task = asyncio.create_task(self.dashboard.run())
        input_task = asyncio.create_task(self._handle_input())
        
        try:
            # Run until quit
            while self.running:
                await asyncio.sleep(0.1)
        finally:
            # Cleanup
            dashboard_task.cancel()
            input_task.cancel()
            await self.cleanup()
    
    async def _handle_input(self):
        """Handle keyboard input."""
        # In a real implementation, this would integrate with prompt_toolkit
        # For now, using a simple approach
        while self.running:
            await asyncio.sleep(0.1)
    
    async def cleanup(self):
        """Clean up resources."""
        await self.context7.cleanup()
        self.console.print("\n[cyan]Thanks for using Context7 Explorer! 👋[/cyan]")


@click.command()
@click.option('--theme', type=click.Choice(['cyberpunk', 'ocean', 'forest', 'sunset']), 
              default='cyberpunk', help='UI theme')
@click.option('--no-animations', is_flag=True, help='Disable animations')
@click.option('--index-path', type=click.Path(), help='Path to document index')
def main(theme: str, no_animations: bool, index_path: Optional[str]):
    """
    Context7 Document Explorer - AI-Powered Document Search
    
    Search through your documents with intelligent AI assistance.
    """
    # Update configuration
    if theme:
        config.theme = theme
    if no_animations:
        config.animations_enabled = False
    if index_path:
        config.context7_index_path = index_path
    
    # Create and run application
    app = Context7Explorer()
    
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
