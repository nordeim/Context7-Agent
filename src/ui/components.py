# File: src/ui/components.py
"""
Beautiful UI components for the Context7 Document Explorer.
"""

import time
from datetime import datetime
from typing import List, Optional, Dict, Any

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.columns import Columns
from rich.table import Table
from rich.text import Text
from rich.box import ROUNDED, DOUBLE, ASCII

from src.ui.themes import Theme


class AnimatedText:
    """Create animated text effects."""
    @staticmethod
    def gradient_text(text: str, start_color: str, end_color: str) -> Text:
        styled_text = Text()
        length = len(text)
        for i, char in enumerate(text):
            ratio = i / max(length - 1, 1)
            r1,g1,b1 = int(start_color[1:3],16),int(start_color[3:5],16),int(start_color[5:7],16)
            r2,g2,b2 = int(end_color[1:3],16),int(end_color[3:5],16),int(end_color[5:7],16)
            r,g,b = int(r1+(r2-r1)*ratio),int(g1+(g2-g1)*ratio),int(b1+(b2-b1)*ratio)
            styled_text.append(char, style=f"#{r:02x}{g:02x}{b:02x}")
        return styled_text

class SearchInterface:
    """Component for the search input area."""
    def __init__(self, theme: Theme):
        self.theme = theme
        self.search_history: List[str] = []

    def render_search_box(self, query: str, focused: bool = True) -> Panel:
        """Renders the search box with a manually drawn 'fake' cursor."""
        border_style = self.theme.primary if focused else self.theme.text_dim
        search_icon = "🔍" if focused else "🔎"
        
        prompt = Text(f"{search_icon} ", style="bold")
        query_text = Text(query, style=self.theme.text)
        
        if focused:
            # Manually append a blinking cursor character using rich's style
            cursor = "█" if int(time.time() * 2) % 2 == 0 else " "
            query_text.append(cursor, style="bold")

        content = Group(Text(""), prompt + query_text, Text(""))

        return Panel(
            content, title="[bold]⚡ Context7 Search[/bold]", title_align="center",
            border_style=border_style, box=DOUBLE, padding=(0, 2), height=5
        )

class DocumentCard:
    """Component for a single document result."""
    def __init__(self, theme: Theme): self.theme = theme
    def render(self, title: str, path: str, preview: str, score: float, highlighted: bool = False) -> Panel:
        score_bar = self._create_score_bar(score)
        title_text = AnimatedText.gradient_text(title, self.theme.primary, self.theme.secondary)
        path_text = Text(f"📁 {path}", style=self.theme.text_dim)
        preview_text = Text()
        for line in preview.split('\n')[:3]:
            preview_text.append(line + "\n", style=self.theme.text)
        content = Group(title_text, path_text, Text(""), preview_text, Text(""), score_bar)
        border_style = self.theme.accent if highlighted else self.theme.surface
        return Panel(content, border_style=border_style, box=ROUNDED, padding=(1, 2))
    
    def _create_score_bar(self, score: float) -> Text:
        bar = Text(); bar.append("Relevance: ", style=self.theme.text_dim)
        filled = int(score * 20)
        bar.append("█" * filled, style=self.theme.success)
        bar.append("░" * (20 - filled), style=self.theme.surface)
        bar.append(f" {score:.0%}", style=self.theme.text)
        return bar

class StatusBar:
    """Component for the status bar at the bottom."""
    def __init__(self, theme: Theme): self.theme = theme; self.items: Dict[str, str] = {}
    def update(self, key: str, value: str): self.items[key] = value
    def render(self) -> Panel:
        time_str = datetime.now().strftime("%H:%M:%S")
        cols = [Text(f"🕐 {time_str}", style=self.theme.info)]
        cols.extend([Text(f"{k}: {v}", style=self.theme.text_dim) for k, v in self.items.items()])
        cols.append(Text("● READY", style=self.theme.success))
        return Panel(Columns(cols, expand=True), height=3, border_style=self.theme.surface, box=ASCII)

class LoadingAnimation:
    """Component for loading animations."""
    def __init__(self, theme: Theme): self.theme = theme; self.frames = self._get_frames(); self.current_frame = 0
    def _get_frames(self) -> List[str]:
        if self.theme.name == "Cyberpunk": return ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        if self.theme.name == "Ocean Breeze": return ["🌊","🌊","🌊","💧","💧","💧"]
        if self.theme.name == "Forest Deep": return ["🌱","🌿","🌳","🌲","🌳","🌿"]
        return ["🌅","☀️","🌤️","⛅","🌥️","☁️"]
    def next_frame(self) -> str:
        frame = self.frames[self.current_frame]; self.current_frame = (self.current_frame + 1) % len(self.frames)
        return frame
    def render_spinner(self, message: str) -> Text:
        spinner = Text(); spinner.append(self.next_frame(), style=self.theme.accent); spinner.append(f" {message}", style=self.theme.text)
        return spinner

class WelcomeScreen:
    """Component for the welcome screen."""
    def __init__(self, theme: Theme): self.theme = theme
    def render(self) -> Panel:
        art = self._get_ascii_art()
        title = AnimatedText.gradient_text("CONTEXT7 DOCUMENT EXPLORER", self.theme.gradient_start, self.theme.gradient_end)
        subtitle = Text("Intelligent Document Search Powered by AI", style=f"italic {self.theme.text_dim}")
        tips = Text()
        for tip in ["💡 Press '/' to start searching", "📚 Use '@' to search by document type", "🏷️  Use '#' to search by tags", "⚡ Press 'Tab' for autocomplete"]:
            tips.append(f"\n  {tip}", style=self.theme.info)
        content = Group(Align.center(Text(art, style=self.theme.accent)), Text(""), Align.center(title), Align.center(subtitle), Text(""), tips)
        return Panel(Align.center(content, vertical="middle"), border_style=self.theme.primary, box=DOUBLE, padding=(2, 4))
    def _get_ascii_art(self) -> str:
        if self.theme.name == "Cyberpunk": return """
    ╔═══════════════════════════════╗
    ║  ░▒▓█ CONTEXT-7 █▓▒░         ║
    ║  ┌─┐┌─┐┌┐┌┌┬┐┌─┐─┐ ┬┌┬┐     ║
    ║  │  │ ││││ │ ├┤ ┌┴┬┘ │      ║
    ║  └─┘└─┘┘└┘ ┴ └─┘┴ └─ ┴      ║
    ╚═══════════════════════════════╝
            """
        if self.theme.name == "Ocean Breeze": return "🌊 CONTEXT-7 EXPLORER 🌊"
        if self.theme.name == "Forest Deep": return "🌲 CONTEXT-7 EXPLORER 🌲"
        return "☀️ CONTEXT-7 EXPLORER ☀️"
