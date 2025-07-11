# src/cli.py
"""Interactive CLI for Context7 Agent."""

import asyncio
import sys
from typing import Optional, Dict, List
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich.text import Text
from rich import box
import aioconsole

from src.agent import Context7Agent
from src.history import ConversationHistory, BookmarkManager
from src.config import config
from src.themes import THEMES
from src.utils import (
    format_document_preview,
    create_gradient_text,
    animate_text,
    clear_screen
)


class Context7CLI:
    """Interactive CLI for Context7 Agent."""
    
    def __init__(self, theme_name: str = None):
        """Initialize the CLI."""
        self.theme_name = theme_name or config.default_theme
        self.theme = THEMES.get(self.theme_name, THEMES["cyberpunk"])
        self.console = Console()
        self.agent = Context7Agent()
        self.history = ConversationHistory(config.history_file)
        self.bookmarks = BookmarkManager(config.bookmarks_file)
        self.current_session_id = None
        
    def display_welcome(self):
        """Display welcome screen with theme-specific ASCII art."""
        clear_screen()
        
        # Display ASCII art with gradient
        art_lines = self.theme.ascii_art.strip().split('\n')
        for i, line in enumerate(art_lines):
            gradient_line = create_gradient_text(
                line,
                self.theme.gradient_start,
                self.theme.gradient_end,
                i / len(art_lines)
            )
            self.console.print(gradient_line, justify="center")
        
        # Display welcome message
        welcome_text = f"Welcome to Context7 Agent - {self.theme.name} Theme"
        self.console.print(
            Panel(
                Align.center(welcome_text),
                style=f"bold {self.theme.primary}",
                box=box.ROUNDED
            )
        )
        
        if 
