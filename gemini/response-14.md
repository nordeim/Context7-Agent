You are absolutely right to be frustrated. The application is crashing again, and the error messages clearly indicate that my previous refactoring was incomplete. I sincerely apologize for this repeated failure. It is unacceptable.

I have performed an extremely detailed analysis of the traceback. The error is an `AttributeError`, which confirms that one part of the code was updated while another part that depends on it was not.

### In-Depth Analysis of the Core Problem

1.  **The Error**: `AttributeError: 'SearchInterface' object has no attribute 'current_query'`.
2.  **The Cause**: In a recent fix, I correctly upgraded the `SearchInterface` component in `src/ui/components.py` to use a modern, interactive `input_buffer` for handling text. As part of this, I deleted the old, simple string attribute `current_query`. However, I failed to update the `Dashboard` class in `src/ui/dashboard.py`, which was still trying to access the old, deleted `current_query` attribute.
3.  **The Incomplete Refactoring**: This is a classic and inexcusable refactoring error. The change was not propagated to all dependent code. A full search reveals that `current_query` was also still being used in `src/explorer_cli.py`.

### The Definitive Plan to Fix the Refactoring

I will now execute a comprehensive and definitive fix across all affected files. This is not a patch; it is the completion of the necessary architectural upgrade.

1.  **`src/ui/dashboard.py`**: I will update the `update_header` and `_display_results` methods to stop referencing the non-existent `current_query` and instead correctly call the updated components.
2.  **`src/explorer_cli.py`**: I will go through this file and replace every remaining reference to `self.dashboard.search.current_query` with the new, correct `self.dashboard.search.input_buffer.text`. This will align the entire application with the new interactive input model.

This two-file change will synchronize the codebase, eliminate the `AttributeError`, and finally deliver the stable, interactive UI that has been the goal of the last several fixes.

### The Corrected and Fully-Featured Code

Here are the complete, updated, and synchronized versions of the files.

#### Corrected `src/ui/dashboard.py`

```python
# File: src/ui/dashboard.py
"""
Main dashboard layout for Context7 Document Explorer.
"""

from typing import Optional, List, Dict, Any, Callable
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
import asyncio

from src.ui.components import (
    SearchInterface,
    DocumentCard,
    StatusBar,
    LoadingAnimation,
    WelcomeScreen,
    AnimatedText
)
from src.ui.themes import get_theme
from src.config import config


class Dashboard:
    """Main dashboard interface."""
    
    def __init__(self, console: Console):
        self.console = console
        self.theme = get_theme(config.theme)
        self.layout = self._create_layout()
        
        # Components
        self.search = SearchInterface(self.theme)
        self.status_bar = StatusBar(self.theme)
        self.loading = LoadingAnimation(self.theme)
        self.welcome = WelcomeScreen(self.theme)
        
        # State
        self.current_view = "welcome"  # welcome, search, results, document
        self.search_results: List[Dict[str, Any]] = []
        self.selected_index = 0
        self.is_searching = False
        
    def _create_layout(self) -> Layout:
        """Create the main layout structure."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=7),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="sidebar", size=30),
            Layout(name="main", ratio=2)
        )
        
        return layout
    
    def update_header(self):
        """Update the header section."""
        if self.current_view == "search" or self.current_view == "results":
            # --- BEGIN MODIFICATION ---
            # The render_search_box method no longer takes a 'query' argument
            # as it's now handled by the internal buffer.
            self.layout["header"].update(
                self.search.render_search_box(
                    focused=self.current_view == "search"
                )
            )
            # --- END MODIFICATION ---
        else:
            # Show a cool banner
            banner = AnimatedText.gradient_text(
                "⚡ CONTEXT7 DOCUMENT EXPLORER ⚡",
                self.theme.gradient_start,
                self.theme.gradient_end
            )
            self.layout["header"].update(
                Panel(
                    Align.center(banner, vertical="middle"),
                    border_style=self.theme.primary,
                    height=7
                )
            )
    
    def update_sidebar(self):
        """Update the sidebar section."""
        if self.current_view == "results" and self.search_results:
            content = []
            content.append(Text(f"Found {len(self.search_results)} documents", style=self.theme.success))
            content.append(Text(""))
            content.append(Text("Filters:", style=f"bold {self.theme.text}"))
            content.append(Text("📄 File Types", style=self.theme.text_dim))
            content.append(Text("📅 Date Range", style=self.theme.text_dim))
            content.append(Text("📏 Size", style=self.theme.text_dim))
            content.append(Text(""))
            content.append(Text("Recent Searches:", style=f"bold {self.theme.text}"))
            # Use the search history from the search component
            for query in self.search.search_history[-5:]:
                content.append(Text(f"  • {query}", style=self.theme.text_dim))
            
            panel = Panel("\n".join(str(c) for c in content), title="[bold]📊 Search Info[/bold]", border_style=self.theme.surface)
            self.layout["sidebar"].update(panel)
        else:
            tips = [
                "🔍 Search Tips:", "", "• Use quotes for exact match", "• AND/OR for boolean search",
                "• * for wildcards", "• ~n for fuzzy search", "", "⌨️  Shortcuts:", "", "• / - Focus search",
                "• ↑↓ - Navigate results", "• Enter - Open document", "• Esc - Go back", "• Ctrl+B - Bookmarks", "• Ctrl+H - History"
            ]
            panel = Panel("\n".join(tips), title="[bold]💡 Quick Help[/bold]", border_style=self.theme.surface)
            self.layout["sidebar"].update(panel)
    
    def update_main(self):
        """Update the main content area."""
        if self.current_view == "welcome":
            self.layout["main"].update(self.welcome.render())
        elif self.is_searching:
            spinner_text = self.loading.render_spinner("Searching documents...")
            loading_panel = Panel(Align.center(spinner_text, vertical="middle"), border_style=self.theme.accent, height=10)
            self.layout["main"].update(loading_panel)
        elif self.current_view == "results":
            if not self.search_results:
                no_results = Panel(Align.center(Text("No documents found 😔\nTry different keywords", style=self.theme.text_dim), vertical="middle"), border_style=self.theme.warning)
                self.layout["main"].update(no_results)
            else:
                self._display_results()
        elif self.current_view == "document":
            self._display_document()
    
    def _display_results(self):
        """Display search results as cards."""
        cards = []
        for i, result in enumerate(self.search_results[:10]):
            card = DocumentCard(self.theme)
            cards.append(card.render(
                title=result.get("title", "Untitled"), path=result.get("path", ""),
                preview=result.get("preview", ""), score=result.get("score", 0.0),
                highlighted=(i == self.selected_index)
            ))
        
        from rich.columns import Columns
        results_view = Columns(cards, equal=True, expand=True)
        
        # --- BEGIN MODIFICATION ---
        # The panel title now gets the query text from the input_buffer
        panel_title = f"[bold]📄 Search Results - '{self.search.input_buffer.text}'[/bold]"
        # --- END MODIFICATION ---
        
        self.layout["main"].update(
            Panel(results_view, title=panel_title, border_style=self.theme.primary)
        )
    
    def _display_document(self):
        """Display the selected document."""
        if self.selected_index < len(self.search_results):
            doc = self.search_results[self.selected_index]
            content = doc.get("content", "")
            file_ext = doc.get("path", "").split(".")[-1]
            
            if file_ext in ["py", "js", "java", "cpp", "c", "rs", "go"]:
                from rich.syntax import Syntax
                content_display = Syntax(content, file_ext, theme="monokai", line_numbers=True)
            elif file_ext in ["md", "markdown"]:
                from rich.markdown import Markdown
                content_display = Markdown(content)
            else:
                content_display = Text(content, style=self.theme.text)
            
            doc_panel = Panel(
                content_display, title=f"[bold]📄 {doc.get('title', 'Document')}[/bold]",
                subtitle=f"[dim]{doc.get('path', '')}[/dim]", border_style=self.theme.primary
            )
            self.layout["main"].update(doc_panel)
    
    def update_footer(self):
        """Update the footer/status bar."""
        self.status_bar.update("Mode", self.current_view.title())
        self.status_bar.update("Results", str(len(self.search_results)))
        self.layout["footer"].update(self.status_bar.render())
    
    def refresh(self):
        """Refresh all layout sections."""
        self.update_header()
        self.update_sidebar()
        self.update_main()
        self.update_footer()
```

#### Corrected `src/explorer_cli.py`

```python
# File: src/explorer_cli.py
"""
Main CLI application for Context7 Document Explorer.
"""

import asyncio
import os
import sys
import io
from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.prompt import Prompt
import click

try:
    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.layout import Layout, Window, ConditionalContainer, HSplit
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.filters import Condition
except ModuleNotFoundError:
    sys.exit("Error: prompt-toolkit is not installed. Please run 'pip install -r requirements.txt'")

from src.ui.dashboard import Dashboard
from src.context7_integration import Context7Manager, SearchQuery, Document
from src.data.history_manager import HistoryManager
from src.data.bookmarks import BookmarkManager
from src.data.session_manager import SessionManager
from src.config import config


class Context7Explorer:
    """Main application class for Context7 Document Explorer."""
    
    def __init__(self):
        self.console = Console(record=True, file=io.StringIO())
        self.real_console = Console()
        self.dashboard = Dashboard(self.console)
        self.context7 = Context7Manager()
        self.history = HistoryManager(config.data_dir / config.history_file)
        self.bookmarks = BookmarkManager(config.data_dir / config.bookmarks_file)
        self.sessions = SessionManager(config.data_dir / config.sessions_dir)
        self.running = True
        self.current_session = None
        self.kb = self._create_key_bindings()
    
    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add('/')
        def search_mode(event):
            self.dashboard.current_view = "search"
            event.app.layout.focus(self.dashboard.search.buffer_control)

        @kb.add('escape')
        def go_back(event):
            if self.dashboard.current_view == "search":
                self.dashboard.search.input_buffer.reset()
                self.dashboard.current_view = "welcome"
            else:
                asyncio.create_task(self.go_back())
            event.app.layout.focus(None)

        @kb.add('enter')
        def on_enter(event):
            if self.dashboard.current_view == "search":
                query = self.dashboard.search.input_buffer.text
                if query:
                    self.dashboard.search.search_history.append(query)
                    self.history.add_search(query)
                    asyncio.create_task(self.perform_search(query))
                self.dashboard.search.input_buffer.reset()
                self.dashboard.current_view = "results"
                event.app.layout.focus(None)
            elif self.dashboard.current_view == "results":
                asyncio.create_task(self.select_current())

        @kb.add('up')
        def move_up(event):
            if self.dashboard.current_view == "results":
                self.dashboard.selected_index = max(0, self.dashboard.selected_index - 1)

        @kb.add('down')
        def move_down(event):
            if self.dashboard.current_view == "results" and self.dashboard.search_results:
                max_index = len(self.dashboard.search_results) - 1
                self.dashboard.selected_index = min(max_index, self.dashboard.selected_index + 1)
        
        @kb.add('c-b')
        def show_bookmarks(event):
            asyncio.create_task(self.show_bookmarks())

        @kb.add('c-h')
        def show_history(event):
            asyncio.create_task(self.show_history())

        @kb.add('c-s')
        def save_session(event):
            event.app.exit(result="save_session")

        @kb.add('c-q')
        def quit_app(event):
            self.running = False
            event.app.exit()

        return kb
    
    async def initialize(self):
        """Initialize the application and print startup messages."""
        if config.animations_enabled:
            await self._show_splash_screen()
        
        self.real_console.print("[cyan]Initializing Context7 integration...[/cyan]")
        success = await self.context7.initialize()
        
        if not success:
            self.real_console.print("[red]Failed to initialize Context7. Running in offline mode.[/red]")
        else:
            self.real_console.print("[green]✓ Context7 initialized successfully![/green]")
        
        last_session = self.sessions.get_last_session()
        if last_session:
            self.current_session = last_session
            self.real_console.print(f"[dim]Restored session: {last_session.name}[/dim]")

    async def _show_splash_screen(self):
        """Show animated splash screen on the real console."""
        console = self.real_console
        frames = ["⚡", "⚡C", "⚡CO", "⚡CON", "⚡CONT", "⚡CONTE", "⚡CONTEX", "⚡CONTEXT", "⚡CONTEXT7", "⚡CONTEXT7 ⚡"]
        for frame in frames:
            console.clear()
            console.print(f"\n\n\n[bold cyan]{frame}[/bold cyan]", justify="center")
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.5)
        console.clear()

    async def perform_search(self, query: str):
        """Perform document search."""
        self.dashboard.is_searching = True
        self.dashboard.refresh()
        
        results = await self.context7.search_documents(SearchQuery(query=query))
        
        self.dashboard.is_searching = False
        self.dashboard.search_results = [
            {"id": doc.id, "title": doc.title, "path": doc.path, "preview": doc.preview,
             "score": doc.score, "metadata": doc.metadata} for doc in results
        ]
        self.dashboard.current_view = "results"
        self.dashboard.selected_index = 0
        self.dashboard.status_bar.update("Status", f"Found {len(results)} results")

    async def select_current(self):
        """Select the currently highlighted item to view its content."""
        if self.dashboard.current_view == "results" and self.dashboard.search_results:
            await self.view_document(self.dashboard.search_results[self.dashboard.selected_index]["id"])

    async def view_document(self, doc_id: str):
        """View a specific document."""
        self.dashboard.current_view = "document"
        content = await self.context7.get_document_content(doc_id)
        if content:
            for doc in self.dashboard.search_results:
                if doc["id"] == doc_id:
                    doc["content"] = content
                    break

    async def go_back(self):
        """Go back to the previous view."""
        if self.dashboard.current_view == "document":
            self.dashboard.current_view = "results"
        elif self.dashboard.current_view in ["search", "results"]:
            self.dashboard.current_view = "welcome"

    async def show_bookmarks(self):
        """Display bookmarked documents."""
        bookmarks = self.bookmarks.get_all()
        if bookmarks:
            self.dashboard.search_results = [
                {"id": b.doc_id, "title": b.title, "path": b.path, "preview": b.notes or "Bookmarked", "score": 1.0, "metadata": {}}
                for b in bookmarks
            ]
            self.dashboard.current_view = "results"
            # --- BEGIN MODIFICATION ---
            # Use the input buffer to set the title
            self.dashboard.search.input_buffer.text = "Bookmarks"
            # --- END MODIFICATION ---

    async def show_history(self):
        """Display search history."""
        history = self.history.get_recent_searches(20)
        if history:
            self.dashboard.search_results = [
                {"id": f"hist_{i}", "title": item.query, "path": item.timestamp.strftime('%Y-%m-%d %H:%M'), "preview": f"{item.results_count} results", "score": 1.0, "metadata": {}}
                for i, item in enumerate(history)
            ]
            self.dashboard.current_view = "results"
            # --- BEGIN MODIFICATION ---
            # Use the input buffer to set the title
            self.dashboard.search.input_buffer.text = "History"
            # --- END MODIFICATION ---

    async def save_session(self):
        """Save the current session state."""
        if not self.dashboard.search_results:
            self.dashboard.status_bar.update("Status", "Nothing to save.")
            return
        
        session_name = Prompt.ask("Save session as", default="Quick Save", console=self.real_console)
        if session_name:
            # --- BEGIN MODIFICATION ---
            # Use the input buffer to get the query for saving
            session_data = {
                "query": self.dashboard.search.input_buffer.text, 
                "results": self.dashboard.search_results,
                "selected_index": self.dashboard.selected_index, 
                "view": self.dashboard.current_view
            }
            # --- END MODIFICATION ---
            self.sessions.save_session(session_name, session_data)
            self.dashboard.status_bar.update("Status", f"Session '{session_name}' saved.")

    async def run(self):
        """Run the main application loop."""
        await self.initialize()

        while self.running:
            def get_content():
                self.dashboard.refresh()
                return self.console.export_text()

            # The main layout is a single window that displays the rich output.
            # The search box is part of the header rendered by the dashboard.
            pt_layout = Layout(Window(FormattedTextControl(text=get_content, show_cursor=False)))
            
            pt_app = Application(
                layout=pt_layout,
                key_bindings=self.kb,
                full_screen=True,
                mouse_support=False,
                refresh_interval=0.1
            )
            
            result = await pt_app.run_async()

            if result == "save_session":
                await self.save_session()
            
            if not self.running:
                break
        
        await self.cleanup()

    async def cleanup(self):
        """Clean up resources before exiting."""
        await self.context7.cleanup()
        self.real_console.clear()
        self.real_console.print("\n[cyan]Thanks for using Context7 Explorer! 👋[/cyan]")

@click.command()
@click.option('--theme', type=click.Choice(['cyberpunk', 'ocean', 'forest', 'sunset']), default='cyberpunk', help='UI theme')
@click.option('--no-animations', is_flag=True, help='Disable animations')
@click.option('--index-path', type=click.Path(), help='Path to document index')
def main(theme: str, no_animations: bool, index_path: Optional[str]):
    """Context7 Document Explorer - AI-Powered Document Search"""
    if theme: config.theme = theme
    if no_animations: config.animations_enabled = False
    if index_path: config.context7_index_path = index_path
    
    app = Context7Explorer()
    try:
        asyncio.run(app.run())
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        print("\nApplication exited cleanly.")

if __name__ == "__main__":
    main()
```

