I'll create an exceptionally cool and user-friendly CLI agent that transforms document search into an amazing experience! Let me design a reimagined application with a beautiful interface, smooth animations, and powerful Context7 integration.

**src/config.py**
```python
"""
Enhanced configuration module for Context7 Document Explorer.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path


@dataclass
class Config:
    """Configuration for the Context7 Document Explorer."""
    
    # OpenAI configuration
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    
    # Context7 configuration
    context7_workspace: str = "default"
    context7_index_path: str = "./documents"
    
    # UI configuration
    theme: str = "cyberpunk"  # cyberpunk, ocean, forest, sunset
    animations_enabled: bool = True
    sound_effects: bool = True
    
    # Search configuration
    max_results: int = 20
    preview_lines: int = 5
    highlight_matches: bool = True
    fuzzy_search: bool = True
    
    # Application settings
    data_dir: Path = field(default_factory=lambda: Path.home() / ".context7_explorer")
    history_file: str = "search_history.json"
    bookmarks_file: str = "bookmarks.json"
    sessions_dir: str = "sessions"
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config = cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4"),
            context7_workspace=os.getenv("CONTEXT7_WORKSPACE", "default"),
            context7_index_path=os.getenv("CONTEXT7_INDEX_PATH", "./documents"),
            theme=os.getenv("THEME", "cyberpunk"),
            animations_enabled=os.getenv("ANIMATIONS_ENABLED", "true").lower() == "true",
            sound_effects=os.getenv("SOUND_EFFECTS", "false").lower() == "true",
        )
        
        # Ensure directories exist
        config.data_dir.mkdir(parents=True, exist_ok=True)
        (config.data_dir / config.sessions_dir).mkdir(exist_ok=True)
        
        return config
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY environment variable is required")
        
        if not Path(self.context7_index_path).exists():
            Path(self.context7_index_path).mkdir(parents=True, exist_ok=True)
        
        return errors


# Global config instance
config = Config.from_env()
```

**src/ui/themes.py**
```python
"""
Theme definitions for the Context7 Document Explorer.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class Theme:
    """Theme configuration for the UI."""
    
    name: str
    primary: str
    secondary: str
    accent: str
    background: str
    surface: str
    text: str
    text_dim: str
    success: str
    warning: str
    error: str
    info: str
    
    # Gradient colors for backgrounds
    gradient_start: str
    gradient_end: str
    
    # Special effects
    glow_color: str
    shadow_color: str
    
    # ASCII art style
    border_style: str
    ascii_art_style: str


THEMES: Dict[str, Theme] = {
    "cyberpunk": Theme(
        name="Cyberpunk",
        primary="#ff006e",
        secondary="#3a0ca3",
        accent="#8338ec",
        background="#0a0a0a",
        surface="#1a1a2e",
        text="#f72585",
        text_dim="#7209b7",
        success="#06ffa5",
        warning="#ffbe0b",
        error="#ff006e",
        info="#4cc9f0",
        gradient_start="#3a0ca3",
        gradient_end="#f72585",
        glow_color="#ff006e",
        shadow_color="#3a0ca3",
        border_style="double",
        ascii_art_style="cyberpunk"
    ),
    
    "ocean": Theme(
        name="Ocean Breeze",
        primary="#006ba6",
        secondary="#0496ff",
        accent="#1d7874",
        background="#001524",
        surface="#15616d",
        text="#78c3fb",
        text_dim="#4a7c7e",
        success="#06ffa5",
        warning="#ffd60a",
        error="#d62828",
        info="#89d2dc",
        gradient_start="#001524",
        gradient_end="#0496ff",
        glow_color="#0496ff",
        shadow_color="#001524",
        border_style="rounded",
        ascii_art_style="waves"
    ),
    
    "forest": Theme(
        name="Forest Deep",
        primary="#2d6a4f",
        secondary="#40916c",
        accent="#52b788",
        background="#081c15",
        surface="#1b4332",
        text="#95d5b2",
        text_dim="#74c69d",
        success="#b7e4c7",
        warning="#f4a261",
        error="#e76f51",
        info="#90e0ef",
        gradient_start="#081c15",
        gradient_end="#52b788",
        glow_color="#52b788",
        shadow_color="#081c15",
        border_style="heavy",
        ascii_art_style="trees"
    ),
    
    "sunset": Theme(
        name="Sunset Glow",
        primary="#f77f00",
        secondary="#fcbf49",
        accent="#eae2b7",
        background="#003049",
        surface="#d62828",
        text="#fcbf49",
        text_dim="#f77f00",
        success="#06ffa5",
        warning="#fcbf49",
        error="#d62828",
        info="#eae2b7",
        gradient_start="#003049",
        gradient_end="#f77f00",
        glow_color="#f77f00",
        shadow_color="#003049",
        border_style="ascii",
        ascii_art_style="sunset"
    )
}


def get_theme(name: str) -> Theme:
    """Get theme by name, default to cyberpunk if not found."""
    return THEMES.get(name, THEMES["cyberpunk"])
```

**src/ui/components.py**
```python
"""
Beautiful UI components for the Context7 Document Explorer.
"""

import asyncio
from typing import List, Optional, Callable, Any, Dict
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich.text import Text
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.columns import Columns
from rich.box import Box, ROUNDED, DOUBLE, HEAVY, ASCII
import random
import time
from datetime import datetime

from src.ui.themes import Theme, get_theme
from src.config import config


class AnimatedText:
    """Create animated text effects."""
    
    @staticmethod
    def typewriter(text: str, console: Console, delay: float = 0.05):
        """Typewriter effect for text."""
        for char in text:
            console.print(char, end="")
            time.sleep(delay)
        console.print()
    
    @staticmethod
    def glow_text(text: str, theme: Theme) -> Text:
        """Create glowing text effect."""
        styled_text = Text(text)
        styled_text.stylize(f"bold {theme.glow_color}")
        return styled_text
    
    @staticmethod
    def gradient_text(text: str, start_color: str, end_color: str) -> Text:
        """Create gradient text effect."""
        styled_text = Text()
        length = len(text)
        
        for i, char in enumerate(text):
            # Simple gradient interpolation
            ratio = i / max(length - 1, 1)
            r1, g1, b1 = int(start_color[1:3], 16), int(start_color[3:5], 16), int(start_color[5:7], 16)
            r2, g2, b2 = int(end_color[1:3], 16), int(end_color[3:5], 16), int(end_color[5:7], 16)
            
            r = int(r1 + (r2 - r1) * ratio)
            g = int(g1 + (g2 - g1) * ratio)
            b = int(b1 + (b2 - b1) * ratio)
            
            color = f"#{r:02x}{g:02x}{b:02x}"
            styled_text.append(char, style=f"bold {color}")
        
        return styled_text


class SearchInterface:
    """Beautiful search interface component."""
    
    def __init__(self, theme: Theme):
        self.theme = theme
        self.search_history: List[str] = []
        self.current_query = ""
        self.suggestions: List[str] = []
    
    def render_search_box(self, query: str = "", focused: bool = True) -> Panel:
        """Render the search input box."""
        border_style = self.theme.primary if focused else self.theme.text_dim
        
        # Create search prompt with icon
        search_icon = "🔍" if focused else "🔎"
        prompt = Text(f"{search_icon} ", style="bold")
        
        # Add the query text with cursor
        if focused:
            cursor = "█" if int(time.time() * 2) % 2 == 0 else "_"
            query_text = Text(query + cursor, style=self.theme.text)
        else:
            query_text = Text(query, style=self.theme.text_dim)
        
        content = Group(
            Text(""),
            prompt + query_text,
            Text(""),
        )
        
        box_style = DOUBLE if self.theme.border_style == "double" else ROUNDED
        
        return Panel(
            content,
            title="[bold]⚡ Context7 Search[/bold]",
            title_align="center",
            border_style=border_style,
            box=box_style,
            padding=(0, 2),
        )
    
    def render_suggestions(self, suggestions: List[str]) -> Optional[Panel]:
        """Render search suggestions."""
        if not suggestions:
            return None
        
        table = Table(show_header=False, show_edge=False, padding=0)
        table.add_column("", style=self.theme.text_dim)
        
        for i, suggestion in enumerate(suggestions[:5]):
            prefix = "→" if i == 0 else " "
            table.add_row(f"{prefix} {suggestion}")
        
        return Panel(
            table,
            border_style=self.theme.secondary,
            box=ROUNDED,
            padding=(0, 1),
        )


class DocumentCard:
    """Beautiful document card component."""
    
    def __init__(self, theme: Theme):
        self.theme = theme
    
    def render(
        self,
        title: str,
        path: str,
        preview: str,
        score: float,
        highlighted: bool = False
    ) -> Panel:
        """Render a document card."""
        # Score indicator
        score_bar = self._create_score_bar(score)
        
        # Title with gradient
        title_text = AnimatedText.gradient_text(
            title,
            self.theme.primary,
            self.theme.secondary
        )
        
        # Path
        path_text = Text(f"📁 {path}", style=self.theme.text_dim)
        
        # Preview with syntax highlighting
        preview_lines = preview.split('\n')[:3]
        preview_text = Text()
        for line in preview_lines:
            preview_text.append(line + "\n", style=self.theme.text)
        
        # Combine all elements
        content = Group(
            title_text,
            path_text,
            Text(""),
            preview_text,
            Text(""),
            score_bar
        )
        
        border_style = self.theme.accent if highlighted else self.theme.surface
        
        return Panel(
            content,
            border_style=border_style,
            box=ROUNDED,
            padding=(1, 2),
        )
    
    def _create_score_bar(self, score: float) -> Text:
        """Create a visual score bar."""
        bar_length = 20
        filled = int(score * bar_length)
        
        bar = Text()
        bar.append("Relevance: ", style=self.theme.text_dim)
        bar.append("█" * filled, style=self.theme.success)
        bar.append("░" * (bar_length - filled), style=self.theme.surface)
        bar.append(f" {score:.0%}", style=self.theme.text)
        
        return bar


class StatusBar:
    """Status bar component."""
    
    def __init__(self, theme: Theme):
        self.theme = theme
        self.items: Dict[str, str] = {}
    
    def update(self, key: str, value: str):
        """Update a status item."""
        self.items[key] = value
    
    def render(self) -> Panel:
        """Render the status bar."""
        columns = []
        
        # Time
        time_str = datetime.now().strftime("%H:%M:%S")
        columns.append(Text(f"🕐 {time_str}", style=self.theme.info))
        
        # Custom items
        for key, value in self.items.items():
            columns.append(Text(f"{key}: {value}", style=self.theme.text_dim))
        
        # Mode indicator
        columns.append(Text("● READY", style=self.theme.success))
        
        return Panel(
            Columns(columns, expand=True),
            height=3,
            border_style=self.theme.surface,
            box=ASCII,
        )


class LoadingAnimation:
    """Cool loading animations."""
    
    def __init__(self, theme: Theme):
        self.theme = theme
        self.frames = self._get_frames()
        self.current_frame = 0
    
    def _get_frames(self) -> List[str]:
        """Get animation frames based on theme."""
        if self.theme.name == "Cyberpunk":
            return ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        elif self.theme.name == "Ocean Breeze":
            return ["🌊", "🌊", "🌊", "💧", "💧", "💧"]
        elif self.theme.name == "Forest Deep":
            return ["🌱", "🌿", "🌳", "🌲", "🌳", "🌿"]
        else:
            return ["🌅", "☀️", "🌤️", "⛅", "🌥️", "☁️"]
    
    def next_frame(self) -> str:
        """Get the next frame of the animation."""
        frame = self.frames[self.current_frame]
        self.current_frame = (self.current_frame + 1) % len(self.frames)
        return frame
    
    def render_spinner(self, message: str) -> Text:
        """Render a spinner with message."""
        spinner = Text()
        spinner.append(self.next_frame(), style=self.theme.accent)
        spinner.append(f" {message}", style=self.theme.text)
        return spinner


class WelcomeScreen:
    """Stunning welcome screen."""
    
    def __init__(self, theme: Theme):
        self.theme = theme
    
    def render(self) -> Panel:
        """Render the welcome screen."""
        # ASCII art based on theme
        ascii_art = self._get_ascii_art()
        
        # Title with gradient
        title = AnimatedText.gradient_text(
            "CONTEXT7 DOCUMENT EXPLORER",
            self.theme.gradient_start,
            self.theme.gradient_end
        )
        
        # Subtitle
        subtitle = Text(
            "Intelligent Document Search Powered by AI",
            style=f"italic {self.theme.text_dim}"
        )
        
        # Tips
        tips = [
            "💡 Press '/' to start searching",
            "📚 Use '@' to search by document type",
            "🏷️  Use '#' to search by tags",
            "⚡ Press 'Tab' for autocomplete"
        ]
        
        tips_text = Text()
        for tip in tips:
            tips_text.append(f"\n  {tip}", style=self.theme.info)
        
        # Combine everything
        content = Group(
            Align.center(Text(ascii_art, style=self.theme.accent)),
            Text(""),
            Align.center(title),
            Align.center(subtitle),
            Text(""),
            tips_text,
        )
        
        return Panel(
            Align.center(content, vertical="middle"),
            border_style=self.theme.primary,
            box=DOUBLE if self.theme.border_style == "double" else ROUNDED,
            padding=(2, 4),
        )
    
    def _get_ascii_art(self) -> str:
        """Get theme-specific ASCII art."""
        if self.theme.name == "Cyberpunk":
            return """
    ╔═══════════════════════════════╗
    ║  ░▒▓█ CONTEXT-7 █▓▒░         ║
    ║  ┌─┐┌─┐┌┐┌┌┬┐┌─┐─┐ ┬┌┬┐     ║
    ║  │  │ ││││ │ ├┤ ┌┴┬┘ │      ║
    ║  └─┘└─┘┘└┘ ┴ └─┘┴ └─ ┴      ║
    ╚═══════════════════════════════╝
            """
        elif self.theme.name == "Ocean Breeze":
            return """
    ～～～～～～～～～～～～～～～～～
     🌊 CONTEXT-7 EXPLORER 🌊
    ～～～～～～～～～～～～～～～～～
            """
        elif self.theme.name == "Forest Deep":
            return """
        🌲🌲🌲🌲🌲🌲🌲🌲🌲
         CONTEXT-7 SEARCH
          🌿 EXPLORER 🌿
        🌲🌲🌲🌲🌲🌲🌲🌲🌲
            """
        else:
            return """
     ☀️ ════════════════════ ☀️
        CONTEXT-7 EXPLORER
      SUNSET DOCUMENT SEARCH
     ☀️ ════════════════════ ☀️
            """
```

**src/ui/dashboard.py**
```python
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
```

**src/context7_integration.py**
```python
"""
Deep integration with Context7 MCP server for document management.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
import hashlib

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import Message
from pydantic_ai.models.openai import OpenAIModel

from src.config import config


class Document(BaseModel):
    """Document model for Context7."""
    
    id: str
    title: str
    path: str
    content: str
    preview: str
    metadata: Dict[str, Any]
    score: float = 0.0
    last_modified: datetime
    size: int
    file_type: str
    tags: List[str] = []


class SearchQuery(BaseModel):
    """Search query model."""
    
    query: str
    filters: Dict[str, Any] = {}
    limit: int = 20
    offset: int = 0
    sort_by: str = "relevance"
    include_content: bool = False


class Context7Manager:
    """Manager for Context7 MCP server integration."""
    
    def __init__(self):
        self.agent = None
        self.mcp_client = None
        self.index_path = Path(config.context7_index_path)
        self.workspace = config.context7_workspace
        self._ensure_directories()
        
        # Document cache
        self._document_cache: Dict[str, Document] = {}
        self._index_metadata: Dict[str, Any] = {}
        
        # Initialize OpenAI model
        self.model = OpenAIModel(
            model=config.openai_model,
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        (self.index_path / ".context7").mkdir(exist_ok=True)
    
    async def initialize(self):
        """Initialize Context7 MCP connection and agent."""
        try:
            # Import MCP-related modules
            from pydantic_ai.tools.mcp import MCPClient
            
            # Create MCP client for Context7
            self.mcp_client = MCPClient(
                command="npx",
                args=["-y", "@upstash/context7-mcp@latest"],
                env={
                    **os.environ.copy(),
                    "CONTEXT7_WORKSPACE": self.workspace,
                    "CONTEXT7_INDEX_PATH": str(self.index_path)
                }
            )
            
            # Connect to the MCP server
            await self.mcp_client.connect()
            
            # Initialize the agent with Context7 tools
            self.agent = Agent(
                model=self.model,
                system_prompt="""You are a document search and analysis expert with access to 
                Context7 MCP server. You help users find relevant documents, extract insights, 
                and provide intelligent summaries. You can search through documents, analyze 
                their content, and identify relationships between different documents."""
            )
            
            # Register Context7 tools
            tools = await self.mcp_client.list_tools()
            for tool in tools:
                self.agent.register_tool(tool)
            
            # Load index metadata
            await self._load_index_metadata()
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize Context7: {e}")
            return False
    
    async def _load_index_metadata(self):
        """Load metadata about the document index."""
        metadata_file = self.index_path / ".context7" / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self._index_metadata = json.load(f)
    
    async def search_documents(
        self, 
        query: SearchQuery,
        progress_callback: Optional[Callable] = None
    ) -> List[Document]:
        """Search documents using Context7."""
        try:
            # Use the agent to search with Context7 tools
            search_prompt = f"""
            Search for documents matching: {query.query}
            
            Apply these filters if specified:
            {json.dumps(query.filters, indent=2)}
            
            Return up to {query.limit} results sorted by {query.sort_by}.
            Include document preview and metadata.
            """
            
            result = await self.agent.run(search_prompt)
            
            # Parse the results
            documents = self._parse_search_results(result.data)
            
            # Apply client-side filtering and scoring
            documents = self._apply_filters(documents, query.filters)
            documents = self._calculate_scores(documents, query.query)
            
            # Sort and limit results
            documents.sort(key=lambda d: d.score, reverse=True)
            documents = documents[:query.limit]
            
            # Cache results
            for doc in documents:
                self._document_cache[doc.id] = doc
            
            return documents
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def _parse_search_results(self, raw_results: str) -> List[Document]:
        """Parse search results from Context7."""
        documents = []
        
        try:
            # Parse JSON response from Context7
            results = json.loads(raw_results)
            
            for item in results.get("documents", []):
                doc = Document(
                    id=self._generate_doc_id(item["path"]),
                    title=item.get("title", Path(item["path"]).stem),
                    path=item["path"],
                    content=item.get("content", ""),
                    preview=self._generate_preview(item.get("content", "")),
                    metadata=item.get("metadata", {}),
                    last_modified=datetime.fromisoformat(
                        item.get("last_modified", datetime.now().isoformat())
                    ),
                    size=item.get("size", 0),
                    file_type=Path(item["path"]).suffix[1:],
                    tags=item.get("tags", [])
                )
                documents.append(doc)
                
        except Exception as e:
            print(f"Error parsing results: {e}")
        
        return documents
    
    def _generate_doc_id(self, path: str) -> str:
        """Generate unique document ID from path."""
        return hashlib.md5(path.encode()).hexdigest()
    
    def _generate_preview(self, content: str, max_length: int = 200) -> str:
        """Generate document preview."""
        if not content:
            return "No preview available"
        
        # Clean and truncate content
        preview = content.strip()
        if len(preview) > max_length:
            preview = preview[:max_length] + "..."
        
        return preview
    
    def _apply_filters(
        self, 
        documents: List[Document], 
        filters: Dict[str, Any]
    ) -> List[Document]:
        """Apply filters to document list."""
        filtered = documents
        
        # File type filter
        if "file_types" in filters:
            allowed_types = filters["file_types"]
            filtered = [d for d in filtered if d.file_type in allowed_types]
        
        # Date range filter
        if "date_from" in filters:
            date_from = datetime.fromisoformat(filters["date_from"])
            filtered = [d for d in filtered if d.last_modified >= date_from]
        
        if "date_to" in filters:
            date_to = datetime.fromisoformat(filters["date_to"])
            filtered = [d for d in filtered if d.last_modified <= date_to]
        
        # Size filter
        if "min_size" in filters:
            filtered = [d for d in filtered if d.size >= filters["min_size"]]
        
        if "max_size" in filters:
            filtered = [d for d in filtered if d.size <= filters["max_size"]]
        
        # Tag filter
        if "tags" in filters:
            required_tags = set(filters["tags"])
            filtered = [
                d for d in filtered 
                if required_tags.intersection(set(d.tags))
            ]
        
        return filtered
    
    def _calculate_scores(self, documents: List[Document], query: str) -> List[Document]:
        """Calculate relevance scores for documents."""
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        for doc in documents:
            score = 0.0
            
            # Title match (highest weight)
            title_lower = doc.title.lower()
            if query_lower in title_lower:
                score += 0.5
            else:
                title_terms = set(title_lower.split())
                score += 0.3 * len(query_terms.intersection(title_terms)) / len(query_terms)
            
            # Content match
            content_lower = doc.content.lower()
            if query_lower in content_lower:
                score += 0.3
            else:
                content_terms = set(content_lower.split())
                score += 0.2 * len(query_terms.intersection(content_terms)) / len(query_terms)
            
            # Path match
            if query_lower in doc.path.lower():
                score += 0.1
            
            # Tag match
            tag_terms = set(" ".join(doc.tags).lower().split())
            score += 0.1 * len(query_terms.intersection(tag_terms)) / max(len(query_terms), 1)
            
            doc.score = min(score, 1.0)
        
        return documents
    
    async def get_document_content(self, doc_id: str) -> Optional[str]:
        """Get full document content."""
        if doc_id in self._document_cache:
            doc = self._document_cache[doc_id]
            
            # Load full content if not already loaded
            if not doc.content or len(doc.content) <= len(doc.preview):
                try:
                    file_path = Path(doc.path)
                    if file_path.exists():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            doc.content = f.read()
                except Exception as e:
                    print(f"Error reading document: {e}")
            
            return doc.content
        
        return None
    
    async def analyze_document(self, doc_id: str) -> Dict[str, Any]:
        """Analyze document using AI."""
        content = await self.get_document_content(doc_id)
        if not content:
            return {}
        
        analysis_prompt = f"""
        Analyze this document and provide:
        1. Key topics and themes
        2. Summary (2-3 sentences)
        3. Important entities (people, places, organizations)
        4. Suggested tags
        5. Related document suggestions
        
        Document content:
        {content[:2000]}...
        """
        
        result = await self.agent.run(analysis_prompt)
        
        # Parse and return analysis
        return self._parse_analysis(result.data)
    
    def _parse_analysis(self, raw_analysis: str) -> Dict[str, Any]:
        """Parse document analysis results."""
        # Simple parsing - in production, use structured output
        return {
            "analysis": raw_analysis,
            "timestamp": datetime.now().isoformat()
        }
    
    async def find_similar_documents(self, doc_id: str, limit: int = 5) -> List[Document]:
        """Find documents similar to the given document."""
        if doc_id not in self._document_cache:
            return []
        
        source_doc = self._document_cache[doc_id]
        
        similarity_prompt = f"""
        Find documents similar to: {source_doc.title}
        
        Content preview: {source_doc.preview}
        Tags: {', '.join(source_doc.tags)}
        
        Return up to {limit} similar documents.
        """
        
        result = await self.agent.run(similarity_prompt)
        
        # Parse and return similar documents
        return self._parse_search_results(result.data)
    
    async def cleanup(self):
        """Clean up resources."""
        if self.mcp_client:
            try:
                await self.mcp_client.disconnect()
            except Exception as e:
                print(f"Error disconnecting MCP client: {e}")
```

Now let's create the main CLI application:

**src/explorer_cli.py**
```python
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
```

Let's create the data management modules:

**src/data/history_manager.py**
```python
"""
Search history management for Context7 Explorer.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import threading


@dataclass
class SearchEntry:
    """Search history entry."""
    query: str
    timestamp: datetime
    results_count: int = 0
    execution_time: float = 0.0
    filters: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "timestamp": self.timestamp.isoformat(),
            "results_count": self.results_count,
            "execution_time": self.execution_time,
            "filters": self.filters or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchEntry":
        """Create from dictionary."""
        return cls(
            query=data["query"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            results_count=data.get("results_count", 0),
            execution_time=data.get("execution_time", 0.0),
            filters=data.get("filters", {})
        )


class HistoryManager:
    """Manages search history with analytics."""
    
    def __init__(self, history_file: Path):
        self.history_file = history_file
        self._lock = threading.Lock()
        self._ensure_file()
        self._cache: List[SearchEntry] = []
        self._load_history()
    
    def _ensure_file(self):
        """Ensure history file exists."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_file.exists():
            self._save_history([])
    
    def _load_history(self):
        """Load history from file."""
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                self._cache = [SearchEntry.from_dict(item) for item in data]
        except Exception:
            self._cache = []
    
    def _save_history(self, entries: List[Dict[str, Any]]):
        """Save history to file."""
        with open(self.history_file, 'w') as f:
            json.dump(entries, f, indent=2)
    
    def add_search(
        self, 
        query: str, 
        results_count: int = 0,
        execution_time: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ):
        """Add a search to history."""
        with self._lock:
            entry = SearchEntry(
                query=query,
                timestamp=datetime.now(),
                results_count=results_count,
                execution_time=execution_time,
                filters=filters
            )
            
            self._cache.append(entry)
            
            # Keep only last 1000 entries
            if len(self._cache) > 1000:
                self._cache = self._cache[-1000:]
            
            # Save to file
            self._save_history([e.to_dict() for e in self._cache])
    
    def get_recent_searches(self, limit: int = 10) -> List[SearchEntry]:
        """Get recent search entries."""
        with self._lock:
            return list(reversed(self._cache[-limit:]))
    
    def get_popular_searches(self, days: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
        """Get popular searches in the last N days."""
        with self._lock:
            cutoff = datetime.now().timestamp() - (days * 86400)
            recent = [e for e in self._cache if e.timestamp.timestamp() > cutoff]
            
            # Count occurrences
            query_counts = {}
            for entry in recent:
                query = entry.query.lower()
                query_counts[query] = query_counts.get(query, 0) + 1
            
            # Sort by count
            popular = sorted(
                query_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:limit]
            
            return [{"query": q, "count": c} for q, c in popular]
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """Get search analytics."""
        with self._lock:
            if not self._cache:
                return {
                    "total_searches": 0,
                    "unique_queries": 0,
                    "avg_results": 0,
                    "avg_execution_time": 0
                }
            
            unique_queries = len(set(e.query.lower() for e in self._cache))
            avg_results = sum(e.results_count for e in self._cache) / len(self._cache)
            avg_time = sum(e.execution_time for e in self._cache) / len(self._cache)
            
            return {
                "total_searches": len(self._cache),
                "unique_queries": unique_queries,
                "avg_results": round(avg_results, 1),
                "avg_execution_time": round(avg_time, 3)
            }
    
    def search_history(self, query: str) -> List[SearchEntry]:
        """Search through history."""
        with self._lock:
            query_lower = query.lower()
            return [
                e for e in self._cache 
                if query_lower in e.query.lower()
            ]
    
    def clear_history(self):
        """Clear all history."""
        with self._lock:
            self._cache = []
            self._save_history([])
```

**src/data/bookmarks.py**
```python
"""
Bookmark management for Context7 Explorer.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import threading


@dataclass
class Bookmark:
    """Document bookmark."""
    doc_id: str
    title: str
    path: str
    timestamp: datetime
    tags: List[str]
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "path": self.path,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Bookmark":
        """Create from dictionary."""
        return cls(
            doc_id=data["doc_id"],
            title=data["title"],
            path=data["path"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            tags=data.get("tags", []),
            notes=data.get("notes", "")
        )


class BookmarkManager:
    """Manages document bookmarks."""
    
    def __init__(self, bookmarks_file: Path):
        self.bookmarks_file = bookmarks_file
        self._lock = threading.Lock()
        self._ensure_file()
        self._bookmarks: Dict[str, Bookmark] = {}
        self._load_bookmarks()
    
    def _ensure_file(self):
        """Ensure bookmarks file exists."""
        self.bookmarks_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.bookmarks_file.exists():
            self._save_bookmarks({})
    
    def _load_bookmarks(self):
        """Load bookmarks from file."""
        try:
            with open(self.bookmarks_file, 'r') as f:
                data = json.load(f)
                self._bookmarks = {
                    k: Bookmark.from_dict(v) 
                    for k, v in data.items()
                }
        except Exception:
            self._bookmarks = {}
    
    def _save_bookmarks(self, bookmarks: Dict[str, Dict[str, Any]]):
        """Save bookmarks to file."""
        with open(self.bookmarks_file, 'w') as f:
            json.dump(bookmarks, f, indent=2)
    
    def add_bookmark(
        self,
        doc_id: str,
        title: str,
        path: str,
        tags: List[str] = None,
        notes: str = ""
    ) -> Bookmark:
        """Add a bookmark."""
        with self._lock:
            bookmark = Bookmark(
                doc_id=doc_id,
                title=title,
                path=path,
                timestamp=datetime.now(),
                tags=tags or [],
                notes=notes
            )
            
            self._bookmarks[doc_id] = bookmark
            
            # Save to file
            self._save_bookmarks({
                k: v.to_dict() 
                for k, v in self._bookmarks.items()
            })
            
            return bookmark
    
    def remove_bookmark(self, doc_id: str) -> bool:
        """Remove a bookmark."""
        with self._lock:
            if doc_id in self._bookmarks:
                del self._bookmarks[doc_id]
                
                # Save to file
                self._save_bookmarks({
                    k: v.to_dict() 
                    for k, v in self._bookmarks.items()
                })
                
                return True
            return False
    
    def get_bookmark(self, doc_id: str) -> Optional[Bookmark]:
        """Get a specific bookmark."""
        with self._lock:
            return self._bookmarks.get(doc_id)
    
    def is_bookmarked(self, doc_id: str) -> bool:
        """Check if document is bookmarked."""
        with self._lock:
            return doc_id in self._bookmarks
    
    def get_all(self) -> List[Bookmark]:
        """Get all bookmarks."""
        with self._lock:
            return list(self._bookmarks.values())
    
    def get_by_tag(self, tag: str) -> List[Bookmark]:
        """Get bookmarks by tag."""
        with self._lock:
            return [
                b for b in self._bookmarks.values()
                if tag in b.tags
            ]
    
    def update_notes(self, doc_id: str, notes: str):
        """Update bookmark notes."""
        with self._lock:
            if doc_id in self._bookmarks:
                self._bookmarks[doc_id].notes = notes
                
                # Save to file
                self._save_bookmarks({
                    k: v.to_dict() 
                    for k, v in self._bookmarks.items()
                })
```

**src/data/session_manager.py**
```python
"""
Session management for Context7 Explorer.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import threading


@dataclass
class Session:
    """Application session."""
    name: str
    timestamp: datetime
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data["data"]
        )


class SessionManager:
    """Manages application sessions."""
    
    def __init__(self, sessions_dir: Path):
        self.sessions_dir = Path(sessions_dir)
        self._lock = threading.Lock()
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Ensure sessions directory exists."""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
    
    def save_session(self, name: str, data: Dict[str, Any]) -> Session:
        """Save a session."""
        with self._lock:
            session = Session(
                name=name,
                timestamp=datetime.now(),
                data=data
            )
            
            # Generate filename
            safe_name = "".join(c if c.isalnum() else "_" for c in name)
            filename = f"{safe_name}_{session.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            
            # Save session
            session_file = self.sessions_dir / filename
            with open(session_file, 'w') as f:
                json.dump(session.to_dict(), f, indent=2)
            
            return session
    
    def load_session(self, filename: str) -> Optional[Session]:
        """Load a specific session."""
        with self._lock:
            session_file = self.sessions_dir / filename
            
            if session_file.exists():
                try:
                    with open(session_file, 'r') as f:
                        data = json.load(f)
                        return Session.from_dict(data)
                except Exception:
                    pass
            
            return None
    
    def get_all_sessions(self) -> List[Session]:
        """Get all saved sessions."""
        with self._lock:
            sessions = []
            
            for file in self.sessions_dir.glob("*.json"):
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        sessions.append(Session.from_dict(data))
                except Exception:
                    pass
            
            # Sort by timestamp (newest first)
            sessions.sort(key=lambda s: s.timestamp, reverse=True)
            
            return sessions
    
    def get_last_session(self) -> Optional[Session]:
        """Get the most recent session."""
        sessions = self.get_all_sessions()
        return sessions[0] if sessions else None
    
    def delete_session(self, filename: str) -> bool:
        """Delete a session."""
        with self._lock:
            session_file = self.sessions_dir / filename
            
            if session_file.exists():
                session_file.unlink()
                return True
            
            return False
    
    def cleanup_old_sessions(self, days: int = 30):
        """Remove sessions older than specified days."""
        with self._lock:
            cutoff = datetime.now().timestamp() - (days * 86400)
            
            for file in self.sessions_dir.glob("*.json"):
                if file.stat().st_mtime < cutoff:
                    file.unlink()
```

Let's create the main entry point:

**main.py**
```python
#!/usr/bin/env python3
"""
Context7 Document Explorer - AI-Powered Document Search

An incredibly cool and user-friendly CLI for searching documents
with Context7 MCP server integration.
"""

from src.explorer_cli import main

if __name__ == "__main__":
    main()
```

And create the necessary init files:

**src/__init__.py**
```python
"""Context7 Document Explorer - AI-Powered Document Search System."""

__version__ = "1.0.0"
__author__ = "Context7 Explorer Team"
```

**src/ui/__init__.py**
```python
"""UI components for Context7 Explorer."""
```

**src/data/__init__.py**
```python
"""Data management modules for Context7 Explorer."""
```

Finally, let's create an updated requirements.txt:

**requirements.txt**
```txt
# Core dependencies
pydantic-ai>=0.0.5
pydantic>=2.0.0
openai>=1.0.0

# UI dependencies
rich>=13.0.0
click>=8.0.0
prompt-toolkit>=3.0.0
textual>=0.40.0

# Data handling
aiofiles>=23.0.0
python-dotenv>=1.0.0
orjson>=3.9.0

# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0

# Development
black>=23.0.0
isort>=5.0.0
flake8>=6.0.0
mypy>=1.0.0
```

And a comprehensive README:

**README.md**
```markdown
# 🚀 Context7 Document Explorer

<div align="center">

![Context7 Explorer](https://img.shields.io/badge/Context7-Explorer-ff006e?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![AI Powered](https://img.shields.io/badge/AI-Powered-8338ec?style=for-the-badge)

*An incredibly cool and beautiful CLI for intelligent document search*

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Themes](#themes) • [Contributing](#contributing)

</div>

---

## ✨ Features

### 🎨 Stunning Visual Interface
- **4 Beautiful Themes**: Cyberpunk, Ocean, Forest, and Sunset
- **Smooth Animations**: Fluid transitions and loading effects
- **Rich Terminal UI**: Gradients, glowing text, and modern layouts
- **ASCII Art**: Theme-specific welcome screens

### 🔍 Powerful Search Capabilities
- **AI-Powered Search**: Intelligent document discovery with Context7
- **Real-time Results**: Live search with instant feedback
- **Advanced Filters**: File type, date range, size, and tags
- **Fuzzy Matching**: Find documents even with typos
- **Search Analytics**: Track and analyze your search patterns

### 📚 Document Management
- **Smart Previews**: Syntax-highlighted document previews
- **Bookmarks**: Save and organize important documents
- **Search History**: Access and replay previous searches
- **Session Management**: Save and restore your work sessions
- **Similar Documents**: AI-powered document recommendations

### ⚡ Context7 Integration
- **MCP Server**: Deep integration with Context7 Model Context Protocol
- **Document Analysis**: AI-powered content understanding
- **Contextual Search**: Find documents based on meaning, not just keywords
- **Auto-indexing**: Automatic document discovery and indexing

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Node.js and npm (for Context7 MCP server)
- OpenAI API key or compatible endpoint

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/context7-explorer.git
cd context7-explorer
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Run the explorer**
```bash
python main.py
```

## 🎯 Usage

### Basic Commands

```bash
# Start with default theme
python main.py

# Choose a different theme
python main.py --theme ocean

# Disable animations for faster performance
python main.py --no-animations

# Specify custom document directory
python main.py --index-path /path/to/documents
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `/` | Enter search mode |
| `↑/↓` | Navigate results |
| `Enter` | View document |
| `Esc` | Go back |
| `Ctrl+B` | Show bookmarks |
| `Ctrl+H` | Search history |
| `Ctrl+S` | Save session |
| `Ctrl+Q` | Quit |

### Search Syntax

- **Basic Search**: `python tutorial`
- **Exact Match**: `"exact phrase"`
- **Boolean**: `python AND (tutorial OR guide)`
- **Wildcards**: `pyth*` or `doc?.pdf`
- **Fuzzy Search**: `pythn~2` (allows 2 character differences)
- **File Type**: `@pdf financial report`
- **Tags**: `#important #urgent meeting notes`

## 🎨 Themes

### 🌆 Cyberpunk
Neon colors with a futuristic feel. Perfect for night coding sessions.

### 🌊 Ocean Breeze
Calming blues and aqua tones. Great for long document reading.

### 🌲 Forest Deep
Natural greens with earthy accents. Reduces eye strain.

### 🌅 Sunset Glow
Warm oranges and purples. Energizing and vibrant.

## 🔧 Configuration

Create a `.env` file with:

```env
# OpenAI Configuration
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4

# Context7 Configuration
CONTEXT7_WORKSPACE=default
CONTEXT7_INDEX_PATH=./documents

# UI Configuration
THEME=cyberpunk
ANIMATIONS_ENABLED=true
SOUND_EFFECTS=false

# Search Configuration
MAX_RESULTS=20
FUZZY_SEARCH=true
```

## 📊 Architecture

```
context7-explorer/
├── src/
│   ├── ui/              # Beautiful UI components
│   │   ├── themes.py    # Theme definitions
│   │   ├── components.py # UI widgets
│   │   └── dashboard.py  # Main layout
│   ├── data/            # Data management
│   │   ├── history.py   # Search history
│   │   ├── bookmarks.py # Bookmarks
│   │   └── sessions.py  # Session management
│   ├── context7_integration.py  # Context7 MCP integration
│   ├── explorer_cli.py          # Main CLI application
│   └── config.py                # Configuration
├── tests/               # Test suite
├── docs/               # Documentation
└── main.py            # Entry point
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Pydantic AI](https://github.com/pydantic/pydantic-ai)
- Powered by [Context7 MCP](https://github.com/upstash/context7-mcp)
- Beautiful terminal UI with [Rich](https://github.com/Textualize/rich)

---

<div align="center">

Made with ❤️ by the Context7 Explorer Team

⭐ Star us on GitHub!

</div>
```

This reimagined Context7 Document Explorer features:

1. **Stunning Visual Design**:
   - 4 beautiful themes with gradients and effects
   - Smooth animations and transitions
   - ASCII art and visual elements
   - Modern dashboard layout

2. **Powerful Search**:
   - AI-powered document discovery
   - Advanced search syntax
   - Real-time results
   - Smart filtering and sorting

3. **Great User Experience**:
   - Intuitive keyboard shortcuts
   - Search history and analytics
   - Bookmarks and sessions
   - Document previews with syntax highlighting

4. **Deep Context7 Integration**:
   - MCP server connection
   - Document analysis
   - Similar document suggestions
   - Contextual search

The application provides an incredibly user-friendly and visually appealing way to search and explore documents using the Context7 MCP server!
