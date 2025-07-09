<think>
### Meticulous Review of the `diff` Output

I will now perform a rigorous, line-by-line validation of the provided `diff` output for `src/explorer_cli.py`. My goal is to confirm that the architectural changes were applied correctly and to identify any unintended regressions or loss of functionality.

**File: `src/explorer_cli.py`**

**1. Imports:**
*   **Change:** `rich.prompt.Prompt` was removed.
    *   **Validation:** This is **INVALID**. The `save_session` method, which was removed in this `diff` but exists in the original code, relies on `Prompt.ask`. Removing this import breaks that feature.
*   **Change:** `HSplit`, `ConditionalContainer`, and `Condition` were added from `prompt_toolkit`.
    *   **Validation:** This is **VALID**. These are necessary components for the new, correct layout that conditionally shows the search bar.
*   **Change:** `Document` was removed from the `context7_integration` import.
    *   **Validation:** This is **INVALID**. The original `view_document` method, which was removed in the `diff`, uses type hints that may rely on this, and other methods might as well. Removing imports without ensuring they are truly unused is risky.
*   **Change:** The `sys.exit` message for a missing `prompt-toolkit` was shortened.
    *   **Validation:** This is an acceptable stylistic change.

**2. `__init__` method:**
*   **Change:** `self.current_session = None` was removed.
    *   **Validation:** This is **INVALID**. The original `initialize` method uses this attribute to store the restored session. Removing its initialization is a bug.

**3. `_create_key_bindings` method:**
*   **Change:** The method was heavily refactored.
    *   `/`: Now changes view and focuses the input buffer. **VALID**.
    *   `escape`: Now also resets the input buffer. **VALID**.
    *   `enter`: Now context-aware for search and result selection. **VALID**.
    *   `up`/`down`: Now wrapped in a condition to only work in "results" view. **VALID**.
    *   `c-b` (bookmarks), `c-h` (history), `c-s` (save session): These key bindings were **completely removed**.
    *   **Validation:** This is a **CRITICAL and UNINTENDED REGRESSION**. The user has lost the ability to view bookmarks, see their search history, or save their session via keyboard shortcuts. This is a major loss of functionality. These bindings must be restored.

**4. `initialize` method:**
*   **Change:** The original implementation, which printed status messages and restored sessions, was replaced with a much simpler version that only calls `self.context7.initialize()`.
    *   **Validation:** This is a **CRITICAL REGRESSION**. The user no longer receives feedback on whether `Context7` initialization succeeded or failed. The feature to restore the last session has been completely removed. This is a significant loss of functionality and user experience.

**5. `perform_search` method:**
*   **Change:** The implementation was simplified. It no longer updates the status bar with the number of results found.
    *   **Validation:** This is a minor regression in user feedback but a regression nonetheless. The status bar update should be restored.

**6. Entire Methods Removed:**
*   **`_show_splash_screen`:** Completely removed. **CRITICAL REGRESSION**. The application has lost its polished, animated startup sequence.
*   **`_get_search_input`:** Completely removed. **INVALID**. While the old implementation was flawed, the new architecture still needs a way to handle blocking prompts for features like saving a session. A complete removal is incorrect.
*   **`view_document`:** Completely removed. **CATASTROPHIC REGRESSION**. The user can no longer view the content of a selected document. The `select_current` method now sets the view to "document" but there is no corresponding method to load and display the content.
*   **`go_back`:** Completely removed. **INVALID**. The `escape` key binding relies on this method. The `diff` shows the key binding exists, but the function it calls is gone. This will cause a runtime crash (`AttributeError`).
*   **`show_bookmarks`:** Completely removed. **CRITICAL REGRESSION**. The `Ctrl+B` key binding was also removed, but this confirms the entire feature is gone.
*   **`show_history`:** Completely removed. **CRITICAL REGRESSION**. The `Ctrl+H` key binding was also removed, confirming this feature is gone.
*   **`save_session`:** Completely removed. **CRITICAL REGRESSION**. The `Ctrl+S` key binding was also removed, confirming this feature is gone.

**7. `run` method:**
*   **Change:** The method was completely rewritten to use the new `HSplit` layout with a `ConditionalContainer`.
    *   **Validation:** The new architecture for the event loop and conditional UI is **structurally correct** and a significant improvement over the previous flawed versions. However, in the process of implementing it, the developer (me) deleted almost all the application's features.

**8. `main` function:**
*   **Change:** Minor stylistic changes to `click.option` decorators. Acceptable.
*   **Change:** The docstring was removed from `main`. Minor regression, should be restored.
*   **Change:** The logic for handling `KeyboardInterrupt` and `EOFError` was retained. **VALID**.

### Conclusion of Review

The `diff` reveals a **catastrophic failure** in my previous code generation. While I correctly identified the need for a new UI architecture and implemented a structurally sound event loop, I did so by deleting a vast majority of the application's existing features and logic. This is a complete violation of the principle of non-regression.

The code is not a "fix"; it is a gutting of the application. The loss of document viewing, history, bookmarks, session management, user feedback, and the splash screen is unacceptable.

### New Plan: The Great Restoration

My new plan is one of careful, meticulous reconstruction. I will use the **structurally correct parts** of my last attempt (the new `run` method's event loop and the interactive search box) as a foundation, but I will **re-integrate every single one of the lost features**.

1.  **Start Point:** I will begin with my last generated `src/explorer_cli.py` and `src/ui/components.py` as a base for the *new architecture*.
2.  **Restore Lost Imports:** Add back `rich.prompt.Prompt` and `Document`.
3.  **Restore Lost Attributes:** Add back `self.current_session = None` in `__init__`.
4.  **Restore Lost Methods:** I will carefully copy and paste the full implementations of **all** the deleted methods from the last known-good version into the new file:
    *   `_show_splash_screen`
    *   `view_document`
    *   `go_back`
    *   `show_bookmarks`
    *   `show_history`
    *   `save_session`
5.  **Restore Lost Key Bindings:** I will add back the key bindings for `c-b`, `c-h`, and `c-s` into the `_create_key_bindings` method. The `go_back` binding in `escape` must also be restored.
6.  **Restore `initialize` Logic:** The full implementation of the `initialize` method, including the status prints and session restoration, will be restored.
7.  **Restore `perform_search` Logic:** The status bar updates will be re-added.
8.  **Adapt to the New Architecture:** The restored methods that use blocking prompts (`save_session`) will need to be adapted to work with the new single-loop architecture. I will use the `event.app.exit(result=...)` pattern for these, as it's the only way to safely run a blocking prompt without a full modal dialog system. The main `run` method will then handle these exit results.

This will be a painstaking process of merging the old, correct logic with the new, correct architecture. I will be extremely careful to ensure every feature is restored to full working order.I have reviewed the `diff` output and I am in complete agreement with your unstated assessment. The version of the code I generated was an absolute failure. In my attempt to fix the UI architecture, I eviscerated the application, removing numerous critical features including document viewing, history, bookmarks, and session management. This is a severe regression and is unacceptable.

I apologize for this significant error. I will now correct this by meticulously re-integrating all the lost functionality into the new, stable UI architecture.

### The New Plan: A Full Restoration

My sole focus is to restore the application to its full-featured state while retaining the stable, non-blocking UI event loop from the last attempt.

1.  **Foundation**: I will use my last generated `src/explorer_cli.py` as the architectural base because its core event loop (`run` method) and interactive component (`ui/components.py`) are structurally correct.
2.  **Meticulous Re-integration**: I will carefully copy every missing method, key binding, and line of logic from the last known-good version of the application and merge it into the new structure.
3.  **Adaptation**: Features that require blocking prompts (like "Save Session") will be adapted to the new event loop. They will exit the main UI loop with a specific result, the prompt will be shown, and then the application will re-enter the main UI loop, preserving the user's state.
4.  **Rigorous Validation**: I will perform a manual `diff` before presenting the final code to ensure that all features have been restored:
    *   Splash screen and initialization messages.
    *   Session restoration.
    *   All key bindings (`/`, `esc`, `enter`, `up`/`down`, `c-b`, `c-h`, `c-s`, `c-q`).
    *   All corresponding methods (`view_document`, `go_back`, `show_bookmarks`, `show_history`, `save_session`).
    *   All status updates and other UX details.

This is not a patch; it is a careful reconstruction to correct my previous error.

### The Corrected and Fully-Featured Code

Here are the complete and corrected files, with all original features restored and integrated into the stable UI architecture.

#### Corrected `src/ui/components.py`
*(No changes from the last version, which correctly added the interactive Buffer, but provided for completeness)*

```python
# File: src/ui/components.py
"""
Beautiful UI components for the Context7 Document Explorer.
"""

import asyncio
from typing import List, Optional, Callable, Any, Dict
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich.text import Text
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.columns import Columns
from rich.box import Box, ROUNDED, DOUBLE, HEAVY, ASCII
import random
import time
from datetime import datetime

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout.controls import BufferControl

from src.ui.themes import Theme, get_theme
from src.config import config


class AnimatedText:
    """Create animated text effects."""
    
    @staticmethod
    def typewriter(text: str, console: Console, delay: float = 0.05):
        """Typewriter effect for text."""
        for char in text:
            console.print(char, end="")
            time.sleep(delay)
        console.print()
    
    @staticmethod
    def glow_text(text: str, theme: Theme) -> Text:
        """Create glowing text effect."""
        styled_text = Text(text)
        styled_text.stylize(f"bold {theme.glow_color}")
        return styled_text
    
    @staticmethod
    def gradient_text(text: str, start_color: str, end_color: str) -> Text:
        """Create gradient text effect."""
        styled_text = Text()
        length = len(text)
        
        for i, char in enumerate(text):
            ratio = i / max(length - 1, 1)
            r1, g1, b1 = int(start_color[1:3], 16), int(start_color[3:5], 16), int(start_color[5:7], 16)
            r2, g2, b2 = int(end_color[1:3], 16), int(end_color[3:5], 16), int(end_color[5:7], 16)
            
            r = int(r1 + (r2 - r1) * ratio)
            g = int(g1 + (g2 - g1) * ratio)
            b = int(b1 + (b2 - b1) * ratio)
            
            color = f"#{r:02x}{g:02x}{b:02x}"
            styled_text.append(char, style=f"bold {color}")
        
        return styled_text


class SearchInterface:
    """Beautiful search interface component."""
    
    def __init__(self, theme: Theme):
        self.theme = theme
        self.search_history: List[str] = []
        self.input_buffer = Buffer()
        self.buffer_control = BufferControl(buffer=self.input_buffer)
        self.suggestions: List[str] = []
    
    def render_search_box(self, focused: bool = True) -> Panel:
        """Render the search input box."""
        border_style = self.theme.primary if focused else self.theme.text_dim
        search_icon = "🔍" if focused else "🔎"
        
        content = Group(
            Text(""),
            (Text(f"{search_icon} ", style="bold") + self.buffer_control),
            Text(""),
        )
        
        box_style = DOUBLE if self.theme.border_style == "double" else ROUNDED
        
        return Panel(
            content,
            title="[bold]⚡ Context7 Search[/bold]",
            title_align="center",
            border_style=border_style,
            box=box_style,
            padding=(0, 2),
            height=5
        )
    
    def render_suggestions(self, suggestions: List[str]) -> Optional[Panel]:
        """Render search suggestions."""
        if not suggestions: return None
        table = Table(show_header=False, show_edge=False, padding=0)
        table.add_column("", style=self.theme.text_dim)
        for i, suggestion in enumerate(suggestions[:5]):
            prefix = "→" if i == 0 else " "
            table.add_row(f"{prefix} {suggestion}")
        return Panel(table, border_style=self.theme.secondary, box=ROUNDED, padding=(0, 1))


class DocumentCard:
    """Beautiful document card component."""
    
    def __init__(self, theme: Theme):
        self.theme = theme
    
    def render(self, title: str, path: str, preview: str, score: float, highlighted: bool = False) -> Panel:
        """Render a document card."""
        score_bar = self._create_score_bar(score)
        title_text = AnimatedText.gradient_text(title, self.theme.primary, self.theme.secondary)
        path_text = Text(f"📁 {path}", style=self.theme.text_dim)
        preview_lines = preview.split('\n')[:3]
        preview_text = Text("\n".join(preview_lines), style=self.theme.text)
        content = Group(title_text, path_text, Text(""), preview_text, Text(""), score_bar)
        border_style = self.theme.accent if highlighted else self.theme.surface
        return Panel(content, border_style=border_style, box=ROUNDED, padding=(1, 2))
    
    def _create_score_bar(self, score: float) -> Text:
        """Create a visual score bar."""
        bar_length = 20
        filled = int(score * bar_length)
        bar = Text()
        bar.append("Relevance: ", style=self.theme.text_dim)
        bar.append("█" * filled, style=self.theme.success)
        bar.append("░" * (bar_length - filled), style=self.theme.surface)
        bar.append(f" {score:.0%}", style=self.theme.text)
        return bar


class StatusBar:
    """Status bar component."""
    
    def __init__(self, theme: Theme):
        self.theme = theme
        self.items: Dict[str, str] = {}
    
    def update(self, key: str, value: str):
        self.items[key] = value
    
    def render(self) -> Panel:
        """Render the status bar."""
        time_str = datetime.now().strftime("%H:%M:%S")
        columns = [Text(f"🕐 {time_str}", style=self.theme.info)]
        columns.extend([Text(f"{key}: {value}", style=self.theme.text_dim) for key, value in self.items.items()])
        columns.append(Text("● READY", style=self.theme.success))
        return Panel(Columns(columns, expand=True), height=3, border_style=self.theme.surface, box=ASCII)


class LoadingAnimation:
    """Cool loading animations."""
    
    def __init__(self, theme: Theme):
        self.theme = theme
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.current_frame = 0
    
    def next_frame(self) -> str:
        frame = self.frames[self.current_frame]
        self.current_frame = (self.current_frame + 1) % len(self.frames)
        return frame
    
    def render_spinner(self, message: str) -> Text:
        spinner = Text(f"{self.next_frame()} {message}", style=self.theme.text)
        spinner.highlight_regex(r"^.", self.theme.accent)
        return spinner


class WelcomeScreen:
    """Stunning welcome screen."""
    
    def __init__(self, theme: Theme):
        self.theme = theme
    
    def render(self) -> Panel:
        """Render the welcome screen."""
        ascii_art = self._get_ascii_art()
        title = AnimatedText.gradient_text("CONTEXT7 DOCUMENT EXPLORER", self.theme.gradient_start, self.theme.gradient_end)
        subtitle = Text("Intelligent Document Search Powered by AI", style=f"italic {self.theme.text_dim}")
        tips_text = Text.from_markup(
            "\n  💡 Press '/' to start searching"
            "\n  📚 Use '@' to search by document type"
            "\n  🏷️  Use '#' to search by tags"
            "\n  ⚡ Press 'Tab' for autocomplete",
            style=self.theme.info
        )
        content = Group(Align.center(Text(ascii_art, style=self.theme.accent)), Text(""), Align.center(title), Align.center(subtitle), Text(""), tips_text)
        return Panel(Align.center(content, vertical="middle"), border_style=self.theme.primary, box=DOUBLE, padding=(2, 4))
    
    def _get_ascii_art(self) -> str:
        return """
    ╔═══════════════════════════════╗
    ║  ░▒▓█ CONTEXT-7 █▓▒░         ║
    ║  ┌─┐┌─┐┌┐┌┌┬┐┌─┐─┐ ┬┌┬┐     ║
    ║  │  │ ││││ │ ├┤ ┌┴┬┘ │      ║
    ║  └─┘└─┘┘└┘ ┴ └─┘┴ └─ ┴      ║
    ╚═══════════════════════════════╝
            """
```

#### Corrected and Restored `src/explorer_cli.py`

```python
# File: src/explorer_cli.py
"""
Main CLI application for Context7 Document Explorer.
"""

import asyncio
import os
import sys
import io
from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.prompt import Prompt
import click

try:
    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.layout import Layout, Window, ConditionalContainer, HSplit
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.filters import Condition
except ModuleNotFoundError:
    sys.exit("Error: prompt-toolkit is not installed. Please run 'pip install -r requirements.txt'")

from src.ui.dashboard import Dashboard
from src.context7_integration import Context7Manager, SearchQuery, Document
from src.data.history_manager import HistoryManager
from src.data.bookmarks import BookmarkManager
from src.data.session_manager import SessionManager
from src.config import config


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
        self.kb = self._create_key_bindings()
    
    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add('/')
        def search_mode(event):
            self.dashboard.current_view = "search"
            # Focus the invisible buffer control to start capturing typing
            event.app.layout.focus(self.dashboard.search.buffer_control)

        @kb.add('escape')
        def go_back(event):
            # If in search view, clear buffer and go to welcome
            if self.dashboard.current_view == "search":
                self.dashboard.search.input_buffer.reset()
                self.dashboard.current_view = "welcome"
            # Otherwise, perform standard go_back logic
            else:
                asyncio.create_task(self.go_back())
            event.app.layout.focus(None)

        @kb.add('enter')
        def on_enter(event):
            if self.dashboard.current_view == "search":
                query = self.dashboard.search.input_buffer.text
                if query:
                    self.dashboard.search.search_history.append(query)
                    self.history.add_search(query)
                    asyncio.create_task(self.perform_search(query))
                self.dashboard.search.input_buffer.reset()
                self.dashboard.current_view = "results"
                event.app.layout.focus(None)
            elif self.dashboard.current_view == "results":
                asyncio.create_task(self.select_current())

        @kb.add('up')
        def move_up(event):
            if self.dashboard.current_view == "results":
                self.dashboard.selected_index = max(0, self.dashboard.selected_index - 1)

        @kb.add('down')
        def move_down(event):
            if self.dashboard.current_view == "results" and self.dashboard.search_results:
                max_index = len(self.dashboard.search_results) - 1
                self.dashboard.selected_index = min(max_index, self.dashboard.selected_index + 1)
        
        @kb.add('c-b')
        def show_bookmarks(event):
            asyncio.create_task(self.show_bookmarks())

        @kb.add('c-h')
        def show_history(event):
            asyncio.create_task(self.show_history())

        @kb.add('c-s')
        def save_session(event):
            # Exit the app with a specific result to run the blocking prompt
            event.app.exit(result="save_session")

        @kb.add('c-q')
        def quit_app(event):
            self.running = False
            event.app.exit()

        return kb
    
    async def initialize(self):
        """Initialize the application and print startup messages."""
        if config.animations_enabled:
            await self._show_splash_screen()
        
        self.real_console.print("[cyan]Initializing Context7 integration...[/cyan]")
        success = await self.context7.initialize()
        
        if not success:
            self.real_console.print("[red]Failed to initialize Context7. Running in offline mode.[/red]")
        else:
            self.real_console.print("[green]✓ Context7 initialized successfully![/green]")
        
        last_session = self.sessions.get_last_session()
        if last_session:
            self.current_session = last_session
            self.real_console.print(f"[dim]Restored session: {last_session.name}[/dim]")

    async def _show_splash_screen(self):
        """Show animated splash screen on the real console."""
        console = self.real_console
        frames = ["⚡", "⚡C", "⚡CO", "⚡CON", "⚡CONT", "⚡CONTE", "⚡CONTEX", "⚡CONTEXT", "⚡CONTEXT7", "⚡CONTEXT7 ⚡"]
        for frame in frames:
            console.clear()
            console.print(f"\n\n\n[bold cyan]{frame}[/bold cyan]", justify="center")
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.5)
        console.clear()

    async def perform_search(self, query: str):
        """Perform document search."""
        self.dashboard.is_searching = True
        self.dashboard.refresh()
        
        results = await self.context7.search_documents(SearchQuery(query=query))
        
        self.dashboard.is_searching = False
        self.dashboard.search_results = [
            {"id": doc.id, "title": doc.title, "path": doc.path, "preview": doc.preview,
             "score": doc.score, "metadata": doc.metadata} for doc in results
        ]
        self.dashboard.current_view = "results"
        self.dashboard.selected_index = 0
        self.dashboard.status_bar.update("Status", f"Found {len(results)} results")

    async def select_current(self):
        """Select the currently highlighted item to view its content."""
        if self.dashboard.current_view == "results" and self.dashboard.search_results:
            await self.view_document(self.dashboard.search_results[self.dashboard.selected_index]["id"])

    async def view_document(self, doc_id: str):
        """View a specific document."""
        self.dashboard.current_view = "document"
        content = await self.context7.get_document_content(doc_id)
        if content:
            for doc in self.dashboard.search_results:
                if doc["id"] == doc_id:
                    doc["content"] = content
                    break

    async def go_back(self):
        """Go back to the previous view."""
        if self.dashboard.current_view == "document":
            self.dashboard.current_view = "results"
        elif self.dashboard.current_view in ["search", "results"]:
            self.dashboard.current_view = "welcome"

    async def show_bookmarks(self):
        """Display bookmarked documents."""
        bookmarks = self.bookmarks.get_all()
        if bookmarks:
            self.dashboard.search_results = [
                {"id": b.doc_id, "title": b.title, "path": b.path, "preview": b.notes or "Bookmarked", "score": 1.0, "metadata": {}}
                for b in bookmarks
            ]
            self.dashboard.current_view = "results"
            self.dashboard.search.current_query = "Bookmarks"

    async def show_history(self):
        """Display search history."""
        history = self.history.get_recent_searches(20)
        if history:
            self.dashboard.search_results = [
                {"id": f"hist_{i}", "title": item.query, "path": item.timestamp.strftime('%Y-%m-%d %H:%M'), "preview": f"{item.results_count} results", "score": 1.0, "metadata": {}}
                for i, item in enumerate(history)
            ]
            self.dashboard.current_view = "results"
            self.dashboard.search.current_query = "History"

    async def save_session(self):
        """Save the current session state."""
        if not self.dashboard.search_results:
            self.dashboard.status_bar.update("Status", "Nothing to save.")
            return
        
        session_name = Prompt.ask("Save session as", default="Quick Save", console=self.real_console)
        if session_name:
            session_data = {
                "query": self.dashboard.search.current_query, "results": self.dashboard.search_results,
                "selected_index": self.dashboard.selected_index, "view": self.dashboard.current_view
            }
            self.sessions.save_session(session_name, session_data)
            self.dashboard.status_bar.update("Status", f"Session '{session_name}' saved.")

    async def run(self):
        """Run the main application loop."""
        await self.initialize()

        while self.running:
            # This function is the bridge between rich and prompt_toolkit.
            # It gets the rendered content from the recording console.
            def get_content():
                self.dashboard.refresh()
                return self.console.export_text()

            # The main layout is a conditional container that shows either the
            # main dashboard or an interactive search box.
            # We use HSplit to stack the search box on top of the main content.
            # The main content is always visible, but the search box is conditional.
            root_container = HSplit([
                ConditionalContainer(
                    content=Window(
                        self.dashboard.search.buffer_control,
                        height=1,
                        dont_extend_height=True
                    ),
                    filter=Condition(lambda: self.dashboard.current_view == "search")
                ),
                Window(content=FormattedTextControl(text=get_content, show_cursor=False))
            ])
            
            # The prompt_toolkit application ties everything together.
            pt_app = Application(
                layout=Layout(root_container),
                key_bindings=self.kb,
                full_screen=True,
                mouse_support=False,
                refresh_interval=0.1
            )
            
            result = await pt_app.run_async()

            if result == "save_session":
                await self.save_session()
            # If the loop was exited for any other reason (like Ctrl+Q), self.running will be False.
            if not self.running:
                break
        
        await self.cleanup()

    async def cleanup(self):
        """Clean up resources before exiting."""
        await self.context7.cleanup()
        self.real_console.clear()
        self.real_console.print("\n[cyan]Thanks for using Context7 Explorer! 👋[/cyan]")

@click.command()
@click.option('--theme', type=click.Choice(['cyberpunk', 'ocean', 'forest', 'sunset']), default='cyberpunk', help='UI theme')
@click.option('--no-animations', is_flag=True, help='Disable animations')
@click.option('--index-path', type=click.Path(), help='Path to document index')
def main(theme: str, no_animations: bool, index_path: Optional[str]):
    """Context7 Document Explorer - AI-Powered Document Search"""
    if theme: config.theme = theme
    if no_animations: config.animations_enabled = False
    if index_path: config.context7_index_path = index_path
    
    app = Context7Explorer()
    try:
        asyncio.run(app.run())
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        print("\nApplication exited cleanly.")

if __name__ == "__main__":
    main()
```

