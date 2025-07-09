# File: src/ui/components.py
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
