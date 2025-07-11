# File: src/explorer_cli.py
"""
Main CLI application for Context7 Document Explorer.
"""

import asyncio
import io
import os
import sys
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.prompt import Prompt

try:
    from prompt_toolkit import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
except ModuleNotFoundError:
    sys.exit("Error: prompt-toolkit is not installed. Please run 'pip install -r requirements.txt'")

from src.config import config
from src.context7_integration import Context7Manager, Document, SearchQuery
from src.data.bookmarks import BookmarkManager
from src.data.history_manager import HistoryManager
from src.data.session_manager import SessionManager
from src.ui.dashboard import Dashboard


class Context7Explorer:
    """Main application class for Context7 Document Explorer."""
    
    def __init__(self):
        self.console = Console(record=True, file=io.StringIO())
        self.real_console = Console()
        self.dashboard = Dashboard(self.console)
        self.context7 = Context7Manager()
        self.history = HistoryManager(config.data_dir / config.history_file)
        self.bookmarks = BookmarkManager(config.data_dir / config.bookmarks_file)
        self.sessions = SessionManager(config.data_dir / config.sessions_dir)
        self.running = True
        self.current_session = None
        self.search_buffer = Buffer()
        self.kb = self._create_key_bindings()

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("/")
        def _(event): self.dashboard.current_view = "search"

        @kb.add("escape")
        def _(event):
            if self.dashboard.current_view == "search": self.search_buffer.reset()
            self.dashboard.current_view = "welcome"

        @kb.add("enter")
        def _(event):
            if self.dashboard.current_view == "search":
                query = self.search_buffer.text
                if query:
                    self.dashboard.search.search_history.append(query)
                    asyncio.create_task(self.perform_search(query))
                self.search_buffer.reset()
                self.dashboard.current_view = "results"
            elif self.dashboard.current_view == "results":
                asyncio.create_task(self.select_current())

        @kb.add("up")
        def _(event):
            if self.dashboard.current_view == "results": self.dashboard.selected_index = max(0, self.dashboard.selected_index - 1)

        @kb.add("down")
        def _(event):
            if self.dashboard.current_view == "results" and self.dashboard.search_results:
                self.dashboard.selected_index = min(len(self.dashboard.search_results) - 1, self.dashboard.selected_index + 1)

        @kb.add("c-b")
        def _(event): asyncio.create_task(self.show_bookmarks())
        @kb.add("c-h")
        def _(event): asyncio.create_task(self.show_history())
        @kb.add("c-s")
        def _(event): event.app.exit(result="save_session")
        @kb.add("c-q")
        def _(event): self.running = False; event.app.exit()

        @kb.add("backspace")
        def _(event):
            if self.dashboard.current_view == "search": self.search_buffer.delete_before_cursor(1)

        @kb.add("<any>")
        def _(event):
            if self.dashboard.current_view == "search": self.search_buffer.insert_text(event.data)

        return kb

    async def initialize(self):
        if config.animations_enabled: await self._show_splash_screen()
        self.real_console.print("[cyan]Initializing Context7 integration...[/cyan]")
        if not await self.context7.initialize():
            self.real_console.print("[red]Failed to initialize Context7. Running in offline mode.[/red]")
        else:
            self.real_console.print("[green]✓ Context7 initialized successfully![/green]")
        if last_session := self.sessions.get_last_session():
            self.current_session = last_session
            self.real_console.print(f"[dim]Restored session: {last_session.name}[/dim]")

    async def _show_splash_screen(self):
        console = self.real_console
        frames = ["⚡","⚡C","⚡CO","⚡CON","⚡CONT","⚡CONTE","⚡CONTEX","⚡CONTEXT","⚡CONTEXT7","⚡CONTEXT7 ⚡"]
        for frame in frames:
            console.clear(); console.print(f"\n\n\n[bold cyan]{frame}[/bold cyan]", justify="center"); await asyncio.sleep(0.1)
        await asyncio.sleep(0.5); console.clear()

    async def perform_search(self, query: str):
        self.dashboard.is_searching = True
        self.history.add_search(query, results_count=0) # Add to history immediately
        results = await self.context7.search_documents(SearchQuery(query=query))
        self.history.update_search(query, len(results)) # Update with result count
        self.dashboard.is_searching = False
        self.dashboard.search_results = [{"id":d.id,"title":d.title,"path":d.path,"preview":d.preview,"score":d.score,"metadata":d.metadata} for d in results]
        self.dashboard.current_view = "results"
        self.dashboard.selected_index = 0
        self.dashboard.status_bar.update("Status", f"Found {len(results)} results")

    async def select_current(self):
        if self.dashboard.current_view == "results" and 0 <= self.dashboard.selected_index < len(self.dashboard.search_results):
            await self.view_document(self.dashboard.search_results[self.dashboard.selected_index]["id"])

    async def view_document(self, doc_id: str):
        self.dashboard.current_view = "document"
        if content := await self.context7.get_document_content(doc_id):
            for doc in self.dashboard.search_results:
                if doc["id"] == doc_id: doc["content"] = content; break

    async def go_back(self):
        if self.dashboard.current_view == "document": self.dashboard.current_view = "results"
        elif self.dashboard.current_view in ["search", "results"]: self.dashboard.current_view = "welcome"

    async def show_bookmarks(self):
        if bookmarks := self.bookmarks.get_all():
            self.dashboard.search_results = [{"id":b.doc_id,"title":b.title,"path":b.path,"preview":b.notes or "Bookmarked","score":1.0,"metadata":{}} for b in bookmarks]
            self.dashboard.current_view = "results"
            self.dashboard.search_query = "Bookmarks"

    async def show_history(self):
        if history := self.history.get_recent_searches(20):
            self.dashboard.search_results = [{"id":f"hist_{i}","title":item.query,"path":item.timestamp.strftime('%Y-%m-%d %H:%M'),"preview":f"{item.results_count} results","score":1.0,"metadata":{}} for i, item in enumerate(history)]
            self.dashboard.current_view = "results"
            self.dashboard.search_query = "History"

    async def save_session(self):
        if not self.dashboard.search_results:
            self.dashboard.status_bar.update("Status", "Nothing to save.")
            return
        if session_name := Prompt.ask("Save session as", default="Quick Save", console=self.real_console):
            session_data = {"query":self.dashboard.search_query,"results":self.dashboard.search_results,"selected_index":self.dashboard.selected_index,"view":self.dashboard.current_view}
            self.sessions.save_session(session_name, session_data)
            self.dashboard.status_bar.update("Status", f"Session '{session_name}' saved.")

    async def run(self):
        await self.initialize()
        
        def get_content():
            self.dashboard.search_query = self.search_buffer.text
            self.dashboard.refresh()
            return self.console.export_text()

        pt_layout = Layout(Window(FormattedTextControl(text=get_content, show_cursor=False)))
        pt_app = Application(layout=pt_layout, key_bindings=self.kb, full_screen=True, mouse_support=False, refresh_interval=0.1)

        while self.running:
            result = await pt_app.run_async()
            if result == "save_session": await self.save_session()
            if not self.running: break
        
        await self.cleanup()

    async def cleanup(self):
        await self.context7.cleanup()
        self.real_console.clear()
        self.real_console.print("\n[cyan]Thanks for using Context7 Explorer! 👋[/cyan]")

@click.command()
@click.option('--theme', type=click.Choice(['cyberpunk', 'ocean', 'forest', 'sunset']), default='cyberpunk', help='UI theme')
@click.option('--no-animations', is_flag=True, help='Disable animations')
@click.option('--index-path', type=click.Path(), help='Path to document index')
def main(theme: str, no_animations: bool, index_path: Optional[str]):
    if theme: config.theme = theme
    if no_animations: config.animations_enabled = False
    if index_path: config.context7_index_path = index_path
    app = Context7Explorer()
    try: asyncio.run(app.run())
    except (KeyboardInterrupt, EOFError): pass
    finally: print("\nApplication exited cleanly.")

if __name__ == "__main__":
    main()
