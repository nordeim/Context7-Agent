# File: src/cli.py
"""
CLI module for the Context7 Agent.

Provides a re-imagined, immersive terminal interface with split-screen layout,
live streaming, advanced animations, and enhanced interactivity.
"""

import os
import sys
from typing import AsyncGenerator

import anyio
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Context7Agent
from themes import THEMES, ASCII_ART, get_theme_styles

console = Console()


class CLI:
    def __init__(self):
        self.agent = Context7Agent()
        self.conversation = []
        self.current_theme = "cyberpunk"
        self.styles = get_theme_styles(self.current_theme)
        self.results = []  # For sidebar
        self.bookmarks = self.agent.history.get_bookmarks()
        self.status = "Ready"
        
        # Load session state at startup
        self.session_state = self.agent.history.load_session()
        if self.session_state:
            self.conversation = self.session_state.get("conversation", [])
            self.agent.history.data["conversations"] = self.conversation
            self.current_theme = self.session_state.get("theme", "cyberpunk")
            self.styles = get_theme_styles(self.current_theme)

    def make_layout(self) -> Layout:
        """Create dynamic split-screen layout."""
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="header", size=7),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )
        layout["main"].split_row(Layout(name="chat", ratio=3), Layout(name="sidebar", ratio=1))
        return layout

    def update_header(self, layout: Layout):
        art = Text.from_markup(ASCII_ART.get(self.current_theme, ""), justify="center")
        layout["header"].update(Panel(art, style=self.styles["header"]))

    def update_chat(self, layout: Layout, message_to_render: str = ""):
        chat_text = Text()
        for msg in self.conversation[-15:]:  # Scrollable view
            style = self.styles["chat_user"] if msg["role"] == "user" else self.styles["chat_agent"]
            chat_text.append(f"{msg['role'].capitalize()}: ", style=f"bold {style}")
            chat_text.append(f"{msg['content']}\n")
        
        if message_to_render:
            chat_text.append("Agent: ", style=f"bold {self.styles['chat_agent']}")
            chat_text.append(Text(message_to_render, style=self.styles["response"]))

        layout["chat"].update(Panel(chat_text, title="Chat", style=self.styles["panel"]))

    def update_sidebar(self, layout: Layout):
        sidebar = Layout(name="sidebar")
        sidebar.split_column(Layout(name="results", ratio=1), Layout(name="bookmarks", ratio=1))

        results_table = Table(title="Live Results", style=self.styles["result"], expand=True)
        results_table.add_column("ID", width=4)
        results_table.add_column("Title")
        for res in self.results:
            if "recommendation" in res:
                results_table.add_row("", Text(f"Rec: {res['recommendation']}", style="italic yellow"))
            else:
                results_table.add_row(str(res["id"]), res["title"])
        sidebar["results"].update(Panel(results_table, title="Search Results", style=self.styles["panel"]))

        bookmarks_text = Text()
        for doc in self.bookmarks[-10:]:
            bookmarks_text.append(f"{doc['id']}: {doc['title']}\n")
        sidebar["bookmarks"].update(Panel(bookmarks_text, title="Bookmarks", style=self.styles["panel"]))

        layout["sidebar"].update(sidebar)

    def update_footer(self, layout: Layout):
        hotkeys = "Hotkeys: /help /search /preview <id> /bookmark <id> /theme <name> /exit"
        footer_text = f"{hotkeys}\nStatus: {self.status}"
        layout["footer"].update(Panel(footer_text, style=self.styles["footer"]))

    async def typing_animation(self, text: str, live: Live, layout: Layout):
        """Typing animation with live updates."""
        current = ""
        for char in text:
            current += char
            self.update_chat(layout, message_to_render=current)
            live.refresh()
            await anyio.sleep(0.02)
        self.agent.history.add_message("assistant", text)
        self.conversation = self.agent.history.get_conversation()

    async def particle_loader(self, live: Live, layout: Layout):
        """Particle burst loader."""
        progress = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True)
        task = progress.add_task(self.styles["particle"], total=None)
        layout["chat"].update(Panel(progress, title="Chat", style=self.styles["panel"]))
        live.refresh()
        await anyio.sleep(2)  # Simulate loader duration
        progress.stop_task(task)

    async def display_error(self, message: str, live: Live, layout: Layout):
        """Styled error alert."""
        original_status = self.status
        self.status = f"ERROR: {message}"
        self.update_footer(layout)
        live.refresh()
        await anyio.sleep(3)
        self.status = original_status

    async def handle_streamed_response(self, generator: AsyncGenerator[Dict, None], live: Live, layout: Layout):
        """Handle live streaming of MCP results."""
        self.results = []
        async for item in generator:
            self.results.append(item)
            self.update_sidebar(layout)
            live.refresh()
            await anyio.sleep(0.1)
        
        rec_count = sum(1 for r in self.results if "recommendation" in r)
        doc_count = len(self.results) - rec_count
        self.status = f"Search complete: {doc_count} docs found"
        self.update_footer(layout)
        summary = f"Results streamed into the sidebar. Use /preview <id> for details."
        self.agent.history.add_message("assistant", summary)
        self.conversation = self.agent.history.get_conversation()

    async def run(self):
        """Main async execution loop."""
        layout = self.make_layout()
        
        async with self.agent.agent.run_mcp_servers():
            with Live(layout, console=console, screen=True, refresh_per_second=10) as live:
                self.update_header(layout)
                self.update_chat(layout)
                self.update_sidebar(layout)
                self.update_footer(layout)

                await self.typing_animation("Welcome! I am your Context7 agent. How can I help?", live, layout)
                
                while True:
                    try:
                        self.status = "Ready"
                        self.update_footer(layout)
                        # Use a thread for blocking input to not halt the event loop
                        user_input = await anyio.to_thread.run_sync(lambda: console.input("[bold]You > [/]"))

                        if user_input.lower() == "/exit":
                            break
                        
                        self.status = "Processing..."
                        self.update_footer(layout)
                        self.agent.history.add_message("user", user_input)
                        self.conversation = self.agent.history.get_conversation()
                        self.update_chat(layout)
                        live.refresh()

                        if user_input.startswith("/"):
                            if user_input.startswith("/preview"):
                                doc_id = int(user_input.split()[-1])
                                preview = self.agent.preview_document(doc_id)
                                await self.typing_animation(preview, live, layout)
                            elif user_input.startswith("/theme"):
                                theme = user_input.split()[-1]
                                if theme in THEMES:
                                    self.current_theme = theme
                                    self.styles = get_theme_styles(theme)
                                    self.update_header(layout)
                                    self.status = f"Theme switched to {theme}!"
                            elif user_input.startswith("/bookmark"):
                                self.status = self.agent.handle_command(user_input)
                                self.bookmarks = self.agent.history.get_bookmarks()
                            else: # /help, /analytics etc.
                                response = self.agent.handle_command(user_input)
                                await self.typing_animation(response, live, layout)
                        else: # It's a prompt for the AI
                            response = await self.agent.generate_response(user_input, self.conversation)

                            if isinstance(response, str):
                                await self.typing_animation(response, live, layout)
                            elif isinstance(response, AsyncGenerator):
                                await self.particle_loader(live, layout)
                                await self.handle_streamed_response(response, live, layout)
                            else:
                                await self.display_error("Unexpected response type.", live, layout)

                    except Exception as e:
                        await self.display_error(str(e), live, layout)

        # Auto-save session on exit
        state = {"conversation": self.conversation, "theme": self.current_theme}
        self.agent.history.save_session(state)
        console.print("[green]Session saved. Goodbye![/green]")

if __name__ == "__main__":
    try:
        anyio.run(CLI().run)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
