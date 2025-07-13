# File: src/cli.py
from __future__ import annotations
import asyncio
import textwrap
from typing import Optional

import typer
from rich.align import Align
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .agent import stream_reply
from .config import settings
from .history import History
from .utils import banner, get_console, switch_theme

app = typer.Typer()
history = History(settings.history_path)  # <-- changed from history_file to history_path


# --------------------------------------------------------------------------- UI
class ChatLayout:
    """
    Handle Rich Live layout: header, chat body, results panel, footer.
    """

    def __init__(self):
        self.chat_lines: list[str] = []
        self.results: Optional[str] = None
        self.status: str = "Ready"

    # Pretty renderable --------------------------------------------------------
    def _render_header(self):
        return Panel("⚡  Context7 Agent  ⚡", style="sys", border_style="border")

    def _render_chat(self):
        if not self.chat_lines:
            placeholder = Align.center("[dim]Start chatting![/dim]", vertical="middle")
            return Panel(placeholder, title="Conversation", border_style="border")

        md = Markdown("\n\n".join(self.chat_lines))
        return Panel(md, title="Conversation", border_style="border")

    def _render_results(self):
        if self.results is None:
            return Panel("[dim]No results yet[/dim]", title="Results", border_style="border")
        return Panel(self.results, title="Results", border_style="border")

    def _render_footer(self):
        return Panel(self.status, border_style="border")

    def __rich__(self):
        return Group(
            self._render_header(),
            self._render_chat(),
            self._render_results(),
            self._render_footer(),
        )


# ---------------------------------------------------------------------- helpers
async def handle_user_input(user_text: str, layout: ChatLayout):
    """
    Detect slash commands or treat as normal user message.
    """
    c = get_console()

    if user_text.lower() == "/theme":
        new_theme = switch_theme()
        c.print(f"[sys]Theme switched to {new_theme}[/sys]")
        return

    if user_text.lower() == "/help":
        help_md = Markdown(
            textwrap.dedent(
                """
                **Slash Commands**
                • `/theme`   – cycle visual themes  
                • `/help`    – this message  
                • `/exit`    – quit the program  
                """
            )
        )
        c.print(help_md)
        return

    # normal message → history → stream reply
    history.add("user", user_text)
    layout.chat_lines.append(f"**You:** {user_text}")
    layout.status = "Thinking…"

    async for role, payload in stream_reply(history):
        if role == "assistant":
            layout.chat_lines.append(f"**AI:** {payload}")
        elif role == "mcp":
            layout.results = payload
        layout.status = "Ready"
        yield  # let Live refresh


# --------------------------------------------------------------------------- Typer
@app.command()
def chat():
    """
    Launch the interactive CLI with Live layout and streaming responses.
    """
    banner()
    c = get_console()
    layout = ChatLayout()

    with Live(layout, console=c, auto_refresh=False, screen=False) as live:
        while True:
            live.refresh()
            try:
                user_input = Prompt.ask("[user]❯")
            except (EOFError, KeyboardInterrupt):
                c.print("\n[sys]Bye![/sys]")
                raise typer.Exit()

            if user_input.strip().lower() == "/exit":
                c.print("[sys]Session saved. Goodbye.[/sys]")
                break

            async def _consume():
                async for _ in handle_user_input(user_input, layout):
                    live.refresh()

            asyncio.run(_consume())
            live.refresh()


if __name__ == "__main__":
    app()
