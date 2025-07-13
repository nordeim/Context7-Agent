# File: src/utils.py
from __future__ import annotations
from itertools import cycle

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .themes import THEMES
from .config import settings

_console = Console(theme=THEMES[settings.theme])
_theme_iter = cycle(THEMES.keys())


def get_console() -> Console:
    """Return singleton console, re-instantiated if theme changed."""
    return _console


def banner() -> None:
    c = get_console()
    art = Text(
        """
 ██████╗ ██████╗ ██████╗ ███████╗████████╗ ██████╗██╗  ██╗
██╔════╝██╔═══██╗██╔══██╗██╔════╝╚══██╔══╝██╔════╝██║ ██╔╝
██║     ██║   ██║██████╔╝█████╗     ██║   ██║     █████╔╝ 
██║     ██║   ██║██╔══██╗██╔══╝     ██║   ██║     ██╔═██╗ 
╚██████╗╚██████╔╝██║  ██║███████╗   ██║   ╚██████╗██║  ██╗
 ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
""",
        style="banner",
    )
    c.print(Panel(art, border_style="border"))


def switch_theme() -> str:
    """
    Cycle through available themes and return the new theme name.
    """
    global _console
    next_theme = next(_theme_iter)
    _console = Console(theme=THEMES[next_theme])
    return next_theme
