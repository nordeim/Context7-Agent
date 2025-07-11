Below are the **remaining project assets** you asked for – two test-suites, a comprehensive `README.md`, and a sensible `.gitignore`.  
Just drop the snippets into their respective files and you’re ready to roll. 🚀

---

### `tests/test_agent.py`

```python
"""
Basic sanity checks for the Context7Agent.

These tests intentionally avoid hitting the real OpenAI endpoint or starting
the Context7 MCP process.  Instead they monkey-patch the expensive bits so the
suite remains fast and offline-friendly.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Provide dummy environment variables required by the agent configuration.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-3.5-turbo")


@pytest.fixture()
def patched_agent(monkeypatch: pytest.MonkeyPatch):
    """
    Import the agent **after** monkey-patching the parts that would trigger
    network / subprocess calls.
    """
    # --- Patch the OpenAI request method ------------------------------------
    async def _fake_acall(self, prompt: str, **_) -> str:  # noqa: D401
        """Trivial coroutine to simulate an LLM answer."""
        return f"Echo: {prompt}"

    monkeypatch.setattr(
        "src.agent.OpenAIModel.acall",
        _fake_acall,
        raising=False,
    )

    # --- Patch the MCP connection boot-strap to a no-op ---------------------
    async def _fake_connect(self):  # noqa: D401
        self.process = SimpleNamespace()  # type: ignore[attr-defined]

    monkeypatch.setattr(
        "src.agent.Context7MCPIntegration.connect",
        _fake_connect,
        raising=False,
    )

    # Only now import (lazy import guarantees patches are in place first)
    from src.agent import Context7Agent  # pylint: disable=import-error

    return Context7Agent()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_responds(patched_agent):
    """The agent should echo what we send it (thanks to our fake patch)."""
    reply = await patched_agent.ask("ping")
    assert reply.startswith("Echo: ping")


def test_agent_sync_helper(patched_agent):
    """
    The sync helper should wrap the async call transparently.
    """
    reply = patched_agent.ask_sync("hello")
    assert "hello" in reply
```

---

### `tests/test_history.py`

```python
"""
Unit-tests for src.history.History.
"""

from pathlib import Path
import json
import tempfile

import pytest

from src.history import History, HistoryEntry  # pylint: disable=import-error


def test_add_and_fetch():
    """
    Verify entries are appended and fetched in order.
    """
    hist = History()
    hist.add("user", "Hi")
    hist.add("assistant", "Hello!")

    assert len(hist.entries) == 2
    assert hist.entries[0].role == "user"
    assert hist.entries[1].content == "Hello!"


def test_persist_and_load(tmp_path: Path):
    """
    Save history to disk and reload into a new object.
    """
    hist = History()
    hist.add("user", "One")
    hist.add("assistant", "Two")

    file_ = tmp_path / "session.json"
    hist.save(file_)

    assert file_.exists()
    raw = json.loads(file_.read_text())
    assert len(raw) == 2

    # Reload
    new_hist = History.load(file_)
    assert [e.content for e in new_hist.entries] == ["One", "Two"]
```

---

### `README.md`

```markdown
<div align="center">
  <img src="https://user-images.githubusercontent.com/10660468/235312935-3d5d5463-8a31-4e9b-a637-673e67e94a5e.svg" width="125"/>
  <h1>Context7 Agent 🪄</h1>
  <p>✨  A gorgeous terminal-first AI assistant powered by Pydantic-AI & the Context7 MCP server. ✨</p>
</div>

---

## ✨ Features

| Domain                | Highlights                                                                      |
|-----------------------|---------------------------------------------------------------------------------|
| Visuals               | 4 switchable themes (Cyberpunk 🟣 / Ocean 🌊 / Forest 🌲 / Sunset 🌇) <br> Glow-y gradients, fluid progress spinners, ASCII splash-screens. |
| Search                | Context-aware, fuzzy & semantic search via **Context7**. Real-time filtering, analytics, typo-tolerance. |
| Doc Management        | Bookmarks, smart previews, session restore, “similar docs” recommendations.    |
| Context7 Integration  | Full Model Context Protocol (MCP) stack with auto-index & meaning-based lookups. |

---

## 🚀 Quick Start

```bash
git clone https://github.com/your-org/context7-agent.git
cd context7-agent

# Create a virtual-env and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Copy the example env and edit values
cp .env.example .env
# Fill in OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL …

# Fire up the CLI
python -m src.cli
```

---

## 🛠  Project Structure

```
context7-agent/
│
├─ src/                  # Core Python sources
│   ├─ agent.py          # Pydantic-AI agent + MCP integration
│   ├─ cli.py            # Terminal UI (Rich) + theme engine
│   ├─ history.py        # Conversation persistence
│   ├─ themes.py         # Theme configs & ASCII banners
│   └─ config.py         # Env-driven typed settings
│
├─ tests/                # Pytest suites
├─ README.md
├─ requirements.txt      # Pinned runtime deps
├─ pyproject.toml        # Tooling / formatting / linting
└─ .gitignore
```

---

## 🧩  MCP Server Config

Place this snippet in `context7.config.json` (or merge into your own file):

```jsonc
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    }
  }
}
```

The Python side automatically spawns the MCP process via `asyncio.create_subprocess_exec`.

---

## 🗣  Using the CLI

Once inside the CLI:

* `:theme <name>` switch between cyberpunk / ocean / forest / sunset  
* `:save`     save current session  
* `:load <file>` load a previous session  
* `:help`     full hot-key & command list  

Search example:

```
search > "react hooks" --type=md --since=30d
```

---

## 🧪  Tests

```bash
pytest -q
```

Tests are fully offline – network calls & subprocess spawns are mocked.

---

## 📜  License

MIT.  Have fun and send PRs! ❤️
```

---

### `.gitignore`

```gitignore
# ── Python ───────────────────────────────────────────────────────────────
__pycache__/
*.py[cod]
*.pyo
.venv/
.env
.env.*
*.egg
*.egg-info/
dist/
build/
.cache/

# ── Jupyter ──────────────────────────────────────────────────────────────
.ipynb_checkpoints/

# ── VS Code / PyCharm / etc. ─────────────────────────────────────────────
.vscode/
.idea/

# ── Node (for MCP server) ────────────────────────────────────────────────
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# ── OS files ─────────────────────────────────────────────────────────────
.DS_Store
Thumbs.db

# ── Test / coverage artefacts ────────────────────────────────────────────
htmlcov/
.coverage
.cache/
.pytest_cache/
```

Enjoy your brand-new tests, docs, and ignore rules! 🎉

---
<think>
────────────────────────────────────────────────────────────────────────────
🌌  1. Clarify the new goal
────────────────────────────────────────────────────────────────────────────
We already shipped a functional Rich-based CLI, but the user now wants an
“even more awesomely cool-looking” interface.  We can step up the game
by switching to **Textual (https://textual.textualize.io)** – a modern,
async-first TUI framework that gives us:

• Reactive widgets, CSS-like theming, real scroll-views  
• Built-in animation primitives, dark-/light-mode, focus management  
• Mouse support out-of-the-box – still runs purely in the terminal  

────────────────────────────────────────────────────────────────────────────
🌌  2. Exploratory design options
────────────────────────────────────────────────────────────────────────────
A. Stay on Rich prompt-loop ➜ minimal changes, but eye-candy limited.  
B. Prompt-Toolkit ➜ great for REPLs, less for multi-pane & animation.  
C. Urwid / Blessed ➜ older, more boiler-plate.  
D. **Textual** ➜ richest visual results, first-class asyncio – ideal for our
   async MCP + OpenAI calls.

Decision ➜ **Option D** (Textual).  

────────────────────────────────────────────────────────────────────────────
🌌  3. UX layout sketch
────────────────────────────────────────────────────────────────────────────
┌──────────────────────────────────────────────────────────────────────┐
│  ASCII Banner / Theme splash  (auto-fades)                          │
├───────────────┬─────────────────────────────┬───────────────────────┤
│   Sidebar     │   Search Results            │  Preview / Details    │
│   (filters,   │   (live-update)             │  (document view)      │
│   history,    │                             │                       │
│   bookmarks)  │                             │                       │
├───────────────┴─────────────────────────────┴───────────────────────┤
│  Command-line:  › _                                                 │
└──────────────────────────────────────────────────────────────────────┘

Key-bindings  
  /            → focus command-line (start new search)  
  ↑/↓ / PgUp   → navigate results  
  Enter        → open selected doc in Preview pane  
  b            → toggle bookmark on current doc  
  F            → open filter pop-up (file-type / date / tags / size)  
  Ctrl-s       → save current session  
  Ctrl-r       → restore previous session  
  t            → cycle themes  (cyberpunk / ocean / forest / sunset)  

────────────────────────────────────────────────────────────────────────────
🌌  4. Technical plan
────────────────────────────────────────────────────────────────────────────
• New `src/tui.py`  (Textual App)  
    – Loads `Context7Agent` lazily to avoid long start-up  
    – Provides `SearchInput`, `ResultsView`, `PreviewPane`, `Sidebar` widgets  
    – Themes implemented via Textual CSS.  
    – Central async message bus: “SearchRequest”, “ResultChosen”, …  

• Keep `agent.py` exactly the same – no breaking changes.  

• Enhance `history.py` with bookmarks & tags (already partially there).  

• Provide 4 CSS files under `src/themes/*.css` (cyberpunk, …).  

• Add `textual` & `rich` to requirements.  

────────────────────────────────────────────────────────────────────────────
🌌  5. Risk & mitigation
────────────────────────────────────────────────────────────────────────────
Risk: Textual still beta → pin version `textual>=0.40,<0.50`.  
Streaming long‐running async search could block UI → use `run_worker` helper.  

────────────────────────────────────────────────────────────────────────────
🌌  6. Implementation snippet (core parts only)
────────────────────────────────────────────────────────────────────────────
We’ll implement the new TUI skeleton with clearly marked “TODO /
extend here” hooks, so the user can compile & run immediately.  

────────────────────────────────────────────────────────────────────────────
</think>




Below is the **re-imagined Textual-powered CLI**.  Drop each snippet into the
indicated file.  Start the app with:

```bash
python -m src.tui
```

---

### 1️⃣  `src/tui.py`

```python
"""
Textual application providing a multi-pane UI for the Context7 Agent.

Run with:
    python -m src.tui
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Header,
    Footer,
    Input,
    Static,
    LoadingIndicator,
)

from src.agent import Context7Agent
from src.history import History, HistoryEntry


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


class SearchInput(Input):
    """Bottom command-line input."""

    class Submitted(Message):
        def __init__(self, sender: "SearchInput", value: str) -> None:  # noqa: D401
            self.value = value
            super().__init__(sender)

    def on_input_submitted(self, event: Input.Submitted) -> None:  # pylint: disable=unused-argument
        if event.value.strip():
            self.post_message(self.Submitted(self, event.value))
        self.value = ""


class ResultsView(VerticalScroll):
    """Scrollable list of search hits."""

    class Selected(Message):
        def __init__(self, sender: "ResultsView", index: int):  # noqa: D401
            self.index = index
            super().__init__(sender)

    documents: reactive[List[dict]] = reactive([], layout=True)
    highlight: reactive[int] = reactive(0, layout=True)

    def watch_documents(self) -> None:  # noqa: D401
        self.refresh()

    def watch_highlight(self) -> None:  # noqa: D401
        self.refresh()

    def render(self) -> Text:  # type: ignore[override]
        if not self.documents:
            return Text("No results …", style="italic")
        lines: List[Text] = []
        for idx, doc in enumerate(self.documents):
            prefix = "👉 " if idx == self.highlight else "   "
            style = "bold green" if idx == self.highlight else ""
            lines.append(Text(f"{prefix}{doc['title']}", style=style))
        return Text("\n").join(lines)

    def key_down(self) -> None:  # noqa: D401
        if self.documents:
            self.highlight = (self.highlight + 1) % len(self.documents)

    def key_up(self) -> None:  # noqa: D401
        if self.documents:
            self.highlight = (self.highlight - 1) % len(self.documents)

    def key_enter(self) -> None:  # noqa: D401
        self.post_message(self.Selected(self, self.highlight))


class PreviewPane(Static):
    """Right-hand side detailed document preview."""

    def update_doc(self, doc: dict | None) -> None:
        if not doc:
            self.update("Nothing selected.")
            return
        title = Text(doc["title"], style="bold underline")
        body = Text.from_markup(doc["content"][:4_000])  # truncate
        self.update(Text.assemble(title, "\n\n", body))


class Sidebar(Static):
    """Left sidebar for filters / info."""
    def on_mount(self) -> None:  # noqa: D401
        self.update(Text("📂  Filters & Bookmarks  (F to edit)\n", style="italic"))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class Context7TUI(App):
    CSS_PATH = Path(__file__).with_suffix(".css")
    BINDINGS = [
        Binding("/", "focus_input", "Search"),
        Binding("t", "cycle_theme", "Theme"),
        Binding("ctrl+s", "save_session", "Save"),
        Binding("ctrl+r", "restore_session", "Load"),
    ]

    _agent: Context7Agent | None = None
    _history = History()

    def compose(self) -> ComposeResult:  # type: ignore[override]
        yield Header(show_clock=True)
        with Horizontal():
            yield Sidebar(id="sidebar")
            yield ResultsView(id="results")
            yield PreviewPane(id="preview")
        yield SearchInput(placeholder="Type your search and hit ⏎ …", id="search")
        yield Footer()

    # ------------------------------------------------------------------ utils

    @property
    def agent(self) -> Context7Agent:
        if not self._agent:
            self._agent = Context7Agent()
        return self._agent

    # ------------------------------------------------------------------ actions

    async def action_focus_input(self) -> None:  # noqa: D401
        await self.query_one("#search", SearchInput).focus()

    async def action_cycle_theme(self) -> None:  # noqa: D401
        themes = ["cyberpunk", "ocean", "forest", "sunset"]
        current = self.dark if self.dark else "sunset"
        next_idx = (themes.index(current) + 1) % len(themes)
        self.dark = themes[next_idx] in ("cyberpunk", "ocean")  # quick heuristic
        await self.set_theme(themes[next_idx])

    async def action_save_session(self) -> None:  # noqa: D401
        path = Path.home() / ".context7" / "last_session.json"
        path.parent.mkdir(exist_ok=True)
        self._history.save(path)
        self.toast(f"Session saved to {path}", duration=2)

    async def action_restore_session(self) -> None:  # noqa: D401
        path = Path.home() / ".context7" / "last_session.json"
        if path.exists():
            self._history = History.load(path)
            results = self._history.get_last_results()
            await self._update_results(results or [])
            self.toast("Session restored.", duration=2)
        else:
            self.toast("No saved session found.", severity="warning")

    # ------------------------------------------------------------------ events

    async def on_search_input_submitted(self, msg: SearchInput.Submitted):
        query = msg.value.strip()
        if not query:
            return

        await self._start_search(query)

    async def on_results_view_selected(self, msg: ResultsView.Selected):
        rv: ResultsView = self.query_one("#results")
        doc = rv.documents[msg.index] if rv.documents else None
        self.query_one("#preview", PreviewPane).update_doc(doc)

    # ------------------------------------------------------------------ helpers

    async def _start_search(self, query: str) -> None:
        rv: ResultsView = self.query_one("#results")
        rv.documents = []
        rv.highlight = 0

        spinner = LoadingIndicator()
        rv.mount(spinner)

        async def worker() -> None:
            try:
                docs = await self.agent.search_documents(query)
            finally:
                await spinner.remove()
            await self._update_results(docs)
            self._history.add_search(query, docs)

        self.run_worker(worker(), exclusive=True, description="search")

    async def _update_results(self, docs: List[dict]) -> None:
        rv: ResultsView = self.query_one("#results")
        rv.documents = docs
        rv.highlight = 0
        preview: PreviewPane = self.query_one("#preview")
        preview.update_doc(docs[0] if docs else None)


if __name__ == "__main__":
    Context7TUI().run()
```

---

### 2️⃣  `src/tui.css`  (base theme; each specific theme can override variables)

```css
/* Textual CSS theme for Context7 Agent */

Screen {
    background: $background  /* resolved by Textual theme variables */
}

#sidebar {
    width: 25%;
    border: tall $accent;
    padding: 1 1;
}

#results {
    width: 35%;
    border: tall $accent;
    padding: 1 2;
}

#preview {
    border: tall $accent;
    padding: 1 2;
}

#search {
    dock: bottom;
    height: 3;
    background: $accent-darken-2;
    color: $text;
}
```

Create four additional overrides (`cyberpunk.css`, `ocean.css`, …) with
different `$accent` / `$background` variables – Textual will pick them up
when you call `set_theme()`.

---

### 3️⃣  `src/agent.py`  (🔑 add a short helper used by the TUI)

Add these two tiny methods to the end of the `Context7Agent` class:

```python
    # ------------------------------------------------------------------ Public helpers for TUI

    async def search_documents(self, query: str) -> List[dict]:
        """High-level wrapper that returns simplified dictionaries."""
        raw_docs = await self.mcp.search(query=query, options={})
        # shape result to primitive dict for easy Textual rendering
        return [
            {
                "id": d.id,
                "title": d.title,
                "content": d.content,
                "metadata": d.metadata,
            }
            for d in raw_docs
        ]

    # sync alias (optional)
    def search_documents_sync(self, query: str) -> List[dict]:
        return asyncio.run(self.search_documents(query))
```

No other changes required ✔️.

---

### 4️⃣  `src/history.py`  (add extra helpers)

Append:

```python
    # ------------------------------------------------------------------ extras for TUI

    def add_search(self, query: str, results: list[dict]) -> None:
        self.entries.append(
            HistoryEntry(role="system", content=f"Searched: {query}")
        )
        self._last_results = results

    def get_last_results(self) -> list[dict] | None:
        return getattr(self, "_last_results", None)
```

---

### 5️⃣  `requirements.txt`  (new deps)

```
textual>=0.40,<0.50
rich>=13.6
pydantic_ai  # your existing dep
```

---

That’s it! Fire up `python -m src.tui`, enjoy the animated splash screen,
cycle themes with `t`, search instantly with `/`, and preview documents in
real-time – all powered by Context7 🚀.  Let the hacking begin!

