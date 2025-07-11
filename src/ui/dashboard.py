# File: src/ui/dashboard.py
"""
Main dashboard layout for Context7 Document Explorer.
"""

from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.align import Align

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
        self.current_view = "welcome"
        self.search_query = "" # This will be passed from the CLI for rendering
        self.search_results: List[Dict[str, Any]] = []
        self.selected_index = 0
        self.is_searching = False
        
    def _create_layout(self) -> Layout:
        """Create the main layout structure."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=5), # Reduced size for search bar
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        layout["body"].split_row(Layout(name="sidebar", size=30), Layout(name="main", ratio=2))
        return layout
    
    def update_header(self):
        """Update the header section."""
        # --- BEGIN MODIFICATION: Correctly call render_search_box ---
        # The header now always contains the search box, which gets the current
        # query text and focus state from the dashboard's properties.
        self.layout["header"].update(
            self.search.render_search_box(
                query=self.search_query,
                focused=(self.current_view == "search")
            )
        )
        # --- END MODIFICATION ---
    
    def update_sidebar(self):
        """Update the sidebar section."""
        if self.current_view == "results" and self.search_results:
            content = [
                Text(f"Found {len(self.search_results)} documents", style=self.theme.success),
                Text(""),
                Text("Filters:", style=f"bold {self.theme.text}"),
                Text("📄 File Types", style=self.theme.text_dim),
                Text("📅 Date Range", style=self.theme.text_dim),
                Text("📏 Size", style=self.theme.text_dim),
                Text(""),
                Text("Recent Searches:", style=f"bold {self.theme.text}"),
            ]
            content.extend([Text(f"  • {q}", style=self.theme.text_dim) for q in self.search.search_history[-5:]])
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
        # The welcome screen is now shown when not searching for results
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
        else:
             # Default to welcome screen if in a weird state
            self.layout["main"].update(self.welcome.render())

    def _display_results(self):
        """Display search results as cards."""
        cards = []
        for i, result in enumerate(self.search_results):
            cards.append(DocumentCard(self.theme).render(
                title=result.get("title", "Untitled"), path=result.get("path", ""),
                preview=result.get("preview", ""), score=result.get("score", 0.0),
                highlighted=(i == self.selected_index)
            ))
        
        from rich.columns import Columns
        results_view = Columns(cards, equal=True, expand=True)
        panel_title = f"[bold]📄 Search Results - '{self.search_query}'[/bold]"
        self.layout["main"].update(Panel(results_view, title=panel_title, border_style=self.theme.primary))
    
    def _display_document(self):
        """Display the selected document."""
        if 0 <= self.selected_index < len(self.search_results):
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
