# File: src/cli.py
"""
CLI module for the Context7 Agent.

Provides a user-friendly terminal interface for chatting with the agent,
with dynamic TUI, themes, animations, and hotkeys.
"""

import os
import sys
import time
import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.prompt import Prompt
from rich import print as rprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Context7Agent
from themes import get_theme_styles, THEMES, ASCII_ART

console = Console()

class CLI:
    def __init__(self):
        self.agent = Context7Agent()
        self.conversation = self.agent.history.get_conversation()
        self.current_theme = "cyberpunk"
        self.styles = get_theme_styles(self.current_theme)

    def typing_animation(self, text: str):
        """Simple typing effect animation."""
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.05)
        print()

    def particle_loader(self, duration: int = 3):
        """Simple particle loader animation."""
        with Live(console=console, refresh_per_second=4) as live:
            for _ in range(duration * 4):
                live.update(Text("Loading... •", style=self.styles["loader"]))
                time.sleep(0.25)

    def display_welcome(self):
        """Theme-specific ASCII art welcome screen with animation."""
        art = ASCII_ART.get(self.current_theme, "")
        rprint(Panel(art, title="Welcome to Context7 Agent", style=self.styles["panel"]))
        self.typing_animation("Hello! Chat naturally or use /commands. Try 'Tell me about AI ethics'.")

    async def handle_input(self):
        while True:
            user_input = await asyncio.to_thread(Prompt.ask, "[bold]You[/bold]", console=console)
            if user_input == "/exit":
                await asyncio.to_thread(self.agent.history.save)
                break
            if user_input.startswith("/theme"):
                theme = user_input.split()[-1]
                if theme in THEMES:
                    self.current_theme = theme
                    self.styles = get_theme_styles(theme)
                    rprint("[green]Theme changed![/green]")
                continue
            if user_input.startswith("/preview"):
                doc_id = int(user_input.split()[-1])
                preview = await asyncio.to_thread(self.agent.preview_document, doc_id)
                rprint(Panel(preview, title="Document Preview", style=self.styles["preview"]))
                continue
            # Add more hotkeys...

            self.agent.history.add_message("user", user_input)
            await asyncio.to_thread(self.particle_loader)

            response = await self.agent.generate_response(user_input, self.conversation)
            self.agent.history.add_message("assistant", response)
            self.conversation = self.agent.history.get_conversation()

            rprint(Panel(response, title="Agent", style=self.styles["response"]))
            await asyncio.to_thread(self.typing_animation, "")

    async def run(self):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        layout["header"].update(Panel("Context7 Agent", style=self.styles["header"]))
        layout["footer"].update(Panel("Hotkeys: /help, /exit", style=self.styles["footer"]))

        with console:
            self.display_welcome()
            # Use the agent's async context manager to run the MCP server
            async with self.agent.agent.run_mcp_servers():
                await self.handle_input()

if __name__ == "__main__":
    asyncio.run(CLI().run())
