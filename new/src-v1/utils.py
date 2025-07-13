# File: src/utils.py
from rich.console import Console
from rich.panel import Panel
from .themes import THEMES
from .config import settings

console = Console(theme=THEMES[settings.theme])

def banner() -> None:
    console.print(
        Panel.fit(
            "[primary]⚡ Context7 Agent ⚡[/primary]\n"
            "[secondary]Conversational MCP Librarian[/secondary]",
            border_style="border",
        )
    )
