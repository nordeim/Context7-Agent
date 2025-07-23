# File: src/cli.py
"""
Beautiful Terminal User Interface (TUI) for the Context7 Agent.
This version has been refactored for stability to use a non-streaming
request-response model with the refactored agent core.
"""
import anyio
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.markdown import Markdown

from src.agent import Context7Agent
from src.themes import get_theme, list_themes, Theme # Assuming themes.py exists and is compatible

class Context7CLI:
    """A stable and beautiful CLI for the Context7 Agent."""
    
    def __init__(self):
        self.console = Console()
        self.agent = Context7Agent()
        self.current_theme: Theme = get_theme("cyberpunk") # Assuming themes.py provides this
        self.current_conversation = "default"
        self.is_running = True

    async def start(self):
        """Start the CLI application lifecycle."""
        try:
            await self.agent.initialize()
            self._show_welcome()
            await self._main_loop()
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n\n[bold yellow]Goodbye![/bold yellow]")
        finally:
            self.console.print("[dim]Shutting down...[/dim]")

    def _show_welcome(self):
        """Display the welcome banner and initial help text."""
        self.console.clear()
        self.console.print(Panel(
            self.current_theme.ascii_art,
            title="[bold]Welcome to Context7 AI[/]",
            subtitle=f"[dim]Theme: {self.current_theme.name}[/dim]",
            border_style=self.current_theme.primary
        ))
        self.console.print("\n[bold green]Agent is ready![/bold green] Type a query or /help for commands.")
        self.console.print()

    async def _main_loop(self):
        """The main interactive loop for the user."""
        while self.is_running:
            user_input = await anyio.to_thread.run_sync(
                lambda: Prompt.ask(f"[bold {self.current_theme.accent}]You[/]")
            )
            
            if not user_input.strip():
                continue

            if user_input.lower().startswith('/'):
                await self._handle_command(user_input)
            else:
                await self._process_message(user_input)

    async def _process_message(self, message: str):
        """Handle a standard user message with a status indicator."""
        self.console.print()
        with self.console.status("[bold green]Assistant is thinking...", spinner="dots"):
            response = await self.agent.chat(message, self.current_conversation)
        
        self.console.print(f"[bold {self.current_theme.primary}]Assistant:[/]")
        if response["type"] == "complete":
            self.console.print(Markdown(response["data"]))
        else: # Error case
            self.console.print(Panel(
                f"[bold]Error:[/]\n{response['data']}",
                border_style="red"
            ))
        self.console.print()

    async def _handle_command(self, user_input: str):
        """Handle slash commands."""
        parts = user_input.strip().split()
        command = parts[0].lower()
        args = parts[1:]

        if command == "/exit":
            self.is_running = False
            self.console.print("[bold yellow]Exiting...[/bold yellow]")
        elif command == "/help":
            self._show_help()
        elif command == "/clear":
            await self.agent.clear_history(self.current_conversation)
            self.console.print("[green]Current conversation history cleared.[/green]")
        elif command == "/history":
            self._show_history()
        elif command == "/theme":
            self._handle_theme_command(args)
        else:
            self.console.print(f"[red]Unknown command: {command}[/red]")
        self.console.print()

    def _handle_theme_command(self, args: list[str]):
        """Handle theme switching."""
        if not args:
            themes = ", ".join(list_themes())
            self.console.print(f"Available themes: {themes}. Usage: /theme <name>")
            return
        
        theme_name = args[0].lower()
        try:
            self.current_theme = get_theme(theme_name)
            self.console.print(f"Theme switched to [bold {self.current_theme.accent}]{theme_name}[/].")
            self._show_welcome() # Redraw with new theme
        except KeyError:
            self.console.print(f"[red]Theme '{theme_name}' not found.[/red]")

    def _show_history(self):
        """Display conversation history summary."""
        conversations = self.agent.get_conversations()
        if not conversations:
            self.console.print("[yellow]No conversation history found.[/yellow]")
            return
        
        table = Table(title="Conversation Summary", border_style=self.current_theme.secondary)
        table.add_column("ID", style=self.current_theme.accent)
        table.add_column("Message Count")
        table.add_column("Last Message")

        for conv in conversations:
            table.add_row(conv['id'], str(conv['message_count']), conv['last_message'])
        
        self.console.print(table)

    def _show_help(self):
        """Display the help panel."""
        help_text = """
        **Commands:**
        *   `/help`      - Shows this help message.
        *   `/exit`      - Exits the application.
        *   `/clear`     - Clears the current conversation history.
        *   `/history`   - Shows a summary of all conversations.
        *   `/theme [name]`- Switches the visual theme.
        """
        self.console.print(Panel(
            Markdown(help_text),
            title="[bold]Help Guide[/]",
            border_style=self.current_theme.primary
        ))

def main():
    """Main entry point for the CLI application."""
    cli = Context7CLI()
    anyio.run(cli.start)

if __name__ == "__main__":
    main()
