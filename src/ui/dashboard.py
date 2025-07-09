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
            self.layout["header"].update(
                self.search.render_search_box(
                    self.search.current_query,
                    focused=self.current_view == "search"
                )
            )
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
            # Show search filters and statistics
            content = []
            
            # Results count
            content.append(
                Text(f"Found {len(self.search_results)} documents", 
                     style=self.theme.success)
            )
            content.append(Text(""))
            
            # Filters
            content.append(Text("Filters:", style=f"bold {self.theme.text}"))
            content.append(Text("📄 File Types", style=self.theme.text_dim))
            content.append(Text("📅 Date Range", style=self.theme.text_dim))
            content.append(Text("📏 Size", style=self.theme.text_dim))
            content.append(Text(""))
            
            # Recent searches
            content.append(Text("Recent Searches:", style=f"bold {self.theme.text}"))
            for query in self.search.search_history[-5:]:
                content.append(Text(f"  • {query}", style=self.theme.text_dim))
            
            panel = Panel(
                "\n".join(str(c) for c in content),
                title="[bold]📊 Search Info[/bold]",
                border_style=self.theme.surface
            )
            self.layout["sidebar"].update(panel)
        else:
            # Show tips or navigation
            tips = [
                "🔍 Search Tips:",
                "",
                "• Use quotes for exact match",
                "• AND/OR for boolean search",
                "• * for wildcards",
                "• ~n for fuzzy search",
                "",
                "⌨️  Shortcuts:",
                "",
                "• / - Focus search",
                "• ↑↓ - Navigate results",
                "• Enter - Open document",
                "• Esc - Go back",
                "• Ctrl+B - Bookmarks",
                "• Ctrl+H - History"
            ]
            
            panel = Panel(
                "\n".join(tips),
                title="[bold]💡 Quick Help[/bold]",
                border_style=self.theme.surface
            )
            self.layout["sidebar"].update(panel)
    
    def update_main(self):
        """Update the main content area."""
        if self.current_view == "welcome":
            self.layout["main"].update(self.welcome.render())
            
        elif self.is_searching:
            # Show loading animation
            spinner_text = self.loading.render_spinner("Searching documents...")
            loading_panel = Panel(
                Align.center(spinner_text, vertical="middle"),
                border_style=self.theme.accent,
                height=10
            )
            self.layout["main"].update(loading_panel)
            
        elif self.current_view == "results":
            if not self.search_results:
                # No results found
                no_results = Panel(
                    Align.center(
                        Text("No documents found 😔\nTry different keywords",
                             style=self.theme.text_dim),
                        vertical="middle"
                    ),
                    border_style=self.theme.warning
                )
                self.layout["main"].update(no_results)
            else:
                # Show search results
                self._display_results()
                
        elif self.current_view == "document":
            # Show document preview
            self._display_document()
    
    def _display_results(self):
        """Display search results as cards."""
        cards = []
        
        for i, result in enumerate(self.search_results[:10]):
            card = DocumentCard(self.theme)
            cards.append(
                card.render(
                    title=result.get("title", "Untitled"),
                    path=result.get("path", ""),
                    preview=result.get("preview", ""),
                    score=result.get("score", 0.0),
                    highlighted=(i == self.selected_index)
                )
            )
        
        # Create a scrollable view
        from rich.columns import Columns
        results_view = Columns(cards, equal=True, expand=True)
        
        self.layout["main"].update(
            Panel(
                results_view,
                title=f"[bold]📄 Search Results - '{self.search.current_query}'[/bold]",
                border_style=self.theme.primary
            )
        )
    
    def _display_document(self):
        """Display the selected document."""
        if self.selected_index < len(self.search_results):
            doc = self.search_results[self.selected_index]
            
            # Document content with syntax highlighting
            content = doc.get("content", "")
            
            # Try to detect file type for syntax highlighting
            file_ext = doc.get("path", "").split(".")[-1]
            
            if file_ext in ["py", "js", "java", "cpp", "c", "rs", "go"]:
                from rich.syntax import Syntax
                syntax = Syntax(content, file_ext, theme="monokai", line_numbers=True)
                content_display = syntax
            elif file_ext in ["md", "markdown"]:
                from rich.markdown import Markdown
                content_display = Markdown(content)
            else:
                content_display = Text(content, style=self.theme.text)
            
            doc_panel = Panel(
                content_display,
                title=f"[bold]📄 {doc.get('title', 'Document')}[/bold]",
                subtitle=f"[dim]{doc.get('path', '')}[/dim]",
                border_style=self.theme.primary
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
    
    async def run(self):
        """Run the dashboard with live updates."""
        with Live(self.layout, console=self.console, refresh_per_second=10) as live:
            while True:
                self.refresh()
                await asyncio.sleep(0.1)
