"""
Futuristic librarian TUI
pip install textual rich httpx websockets
python tui.py
"""
import asyncio, json, os, sys, typing as T
from textual.app import App, ComposeResult
from textual.widgets import Input, Markdown, Footer, DataTable, Static
from textual.binding import Binding
from rich.markdown import Markdown as RichMarkdown
from textual.reactive import reactive
from agent import AgentDeps, agent, MCPClient
from textual.containers import Horizontal, Vertical
from textual.screen import Screen

URI = "ws://localhost:8766"

class PreviewPane(Static):
    """Shows a live scrollable preview of the selected doc."""
    content = reactive("")

    def watch_content(self, val: str) -> None:
        self.update(RichMarkdown(val))

class ChatScreen(Screen):
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+d", "toggle_dark", "Toggle theme"),
        Binding("b", "bookmark", "Bookmark"),
        ("enter", "submit", "Submit"),
    ]

    def __init__(self):
        super().__init__()
        self.mcp = MCPClient(URI)
        self.agent_deps = AgentDeps(mcp=self.mcp)

    async def on_mount(self) -> None:
        await self.mcp.connect()
        self.query_one("#search", Input).focus()

    def compose(self) -> ComposeResult:
        yield Footer()
        with Horizontal():
            with Vertical(id="left"):
                yield Input(placeholder="Ask me anything…", id="search")
                yield DataTable(id="results")
            with Vertical(id="right"):
                yield PreviewPane(id="preview")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return
        self.query_one("#results").clear(columns=True)
        self.query_one("#results").add_columns("Title", "URL")
        self.query_one("#preview").content = "_Searching…_"
        # Stream agent response
        async with agent.run_stream(query, deps=self.agent_deps) as run:
            async for partial in run.stream():
                if partial.tool_return:
                    docs = partial.tool_return
                    table = self.query_one("#results", DataTable)
                    table.clear()
                    for d in docs:
                        table.add_row(d.title, d.url, key=d.url)
                    if docs:
                        table.move_cursor(row=0)
                        self.query_one("#preview").content = docs[0].snippet

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        url = event.data_table.get_cell(event.row_key, 1)
        self.query_one("#preview").content = f"[Selected] {url}"

    async def action_bookmark(self) -> None:
        table = self.query_one("#results", DataTable)
        if table.cursor_row is None:
            return
        url = table.get_cell_at(table.cursor_row, 1)
        await self.mcp.call_tool("bookmark_doc", {"url": url, "note": "via TUI"})
        self.notify("Bookmarked 🧷")

    def action_toggle_dark(self) -> None:
        self.app.dark = not self.app.dark

class LibrarianApp(App):
    CSS = """
    Screen { background: #111827; }
    #left { width: 60%; }
    #right { width: 40%; }
    DataTable { height: 1fr; }
    #preview { border-left: panel #374151; padding: 1; }
    """

    def on_mount(self):
        self.push_screen(ChatScreen())

if __name__ == "__main__":
    LibrarianApp().run()
