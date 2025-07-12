# 📚 **Context7-Agent Project Architecture Document (PAD)**  
*Version 0.1 – 2024-06-XX – Maintainers: Core Engineering Guild*  

---

**Table of Contents**  

1. Executive Summary  
2. Vision & Product Narrative  
3. High-Level Technical Overview  
4. Source-Tree & File-Hierarchy (ASCII & Mermaid diagrams)  
5. Layer-by-Layer Walk-through  
   1. Configuration Layer  
   2. Theming & UX Layer  
   3. Persistence & History Layer  
   4. LLM / MCP Abstraction Layer  
   5. Interactive TUI & CLI Layer  
   6. Test & Quality-Gate Layer  
6. Deep-Dive Modules  
   1. `config.py`  
   2. `themes.py`  
   3. `utils.py`  
   4. `history.py`  
   5. `agent.py`  
   6. `cli.py`  
7. Runtime Life-Cycle & Control-Flow  
8. Error-Handling, Observability & Logging  
9. Dev-Ex: Local Setup, Hot-Reload & Contribution Flow  
10. Road-Map & Extension Points  
11. Glossary & Appendix  

---

> **NOTE**  
> This PAD is purposely verbose (~6 K words) so that a new engineer landing on the repository can reproduce mental context *without* reading every LOC. Jump straight to the module of interest, copy snippets, study diagrams, then hack away. **Welcome to the future of terminal librarianship!**

---

## 1 Executive Summary 

Context7-Agent is an immersive command-line application that merges three powerful ideas:

* Conversational AI chat (`OpenAI GPT-4o-mini`)  
* Contextual document search via **Context7 MCP** (Model Context Protocol)  
* Modern terminal user-experience (Rich, Typer, live streaming, themes)

Users speak naturally—“*show me the latest research on Boson sampling*”—and the agent performs:

1. Intent detection → decides a **search** is needed  
2. MCP search query → fetches semantically-ranked documents  
3. Streams answers & results side-by-side in a beautiful TUI

The PAD explains how 1 100 lines of code (LoC) across 6 core modules accomplish this, illustrates architecture, pinpoints extension points (bookmarks, analytics, quantum-mode), and prescribes best practices for maintenance.

---

## 2 Vision & Product Narrative 

Imagine a librarian from 2050. She not only fetches papers but also anticipates your next question, highlights code, animates results, and remembers every request you have made. **Context7-Agent** is the CLI incarnation of that librarian:

* **Accessibility** – Runs in any POSIX terminal, installation ≤2 min.  
* **Aesthetics** – Four themes with glowing banners, responsive layout.  
* **Power** – Semantic search beats keyword search; fuzzy tolerance; filters.  
* **Data Sovereignty** – Local history persisted as plain JSON; no vendor lock-in.  
* **Hackability** – Small, modular, documented; 100 % Python; no hidden magic.

Our north-star metric is **time-to-knowledge (TTK)**: the delta between *question formulated* and *document consumed*. Every architectural decision reduces TTK—low-latency streaming, Rich Live updates, cached MCP sessions, fine-grained error alerts.

---

## 3 High-Level Technical Overview 

The system is split into **five concentric layers**:

1. **Config Layer** – `.env` → Pydantic `Settings`. All secrets & toggles live here.  
2. **Core Domain Layer** – *Themes*, *History*, *LLM*, *MCP* abstractions. Pure logic, zero I/O side-effects (except disk persistence).  
3. **Application Layer** – `Agent` orchestrates chat flow, merges LLM + MCP.  
4. **Interface Layer** – `cli.py` renders TUI, handles commands, streams outputs.  
5. **Infrastructure Layer** – `mcp.config.json` boots Node-based MCP in a child process; `pytest` ensures quality gates.

The star of the show is **Pydantic-AI**, contributing:

* Declarative `Agent` object with **tool-calling** (MCP verbs) baked in.  
* Native OpenAI provider, streaming generator interface.  
* Validation that messages are well-formed `ChatMessage` instances (type-safety).

`Rich` powers the aesthetic: multi-panel layout, Markdown rendering, theme-aware Console, Live updates. `Typer` handles CLI boilerplate.

---

## 4 Source-Tree & File-Hierarchy

### 4.1 ASCII Tree

```
context7-agent/
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
├── mcp.config.json
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── themes.py
│   ├── utils.py
│   ├── history.py
│   ├── agent.py
│   └── cli.py
└── tests/
    ├── __init__.py
    ├── test_agent.py
    └── test_history.py
```

### 4.2 Annotated File Roles

| File | Role | Key Responsibilities |
|------|------|----------------------|
| `pyproject.toml` | Build metadata | Declares dependencies & optional `dev` extras |
| `mcp.config.json` | Infra spec | Maps alias → Node command for MCP |  
| `src/config.py` | Config layer | Parse & validate env, expose `settings` |
| `src/themes.py` | UX layer | Rich `Theme` objects for colour palettes |
| `src/utils.py` | UX utilities | Banner, theme switch, Console singleton |
| `src/history.py` | Persistence | JSON round-trip of `ChatMessage`s |
| `src/agent.py` | Core engine | Assemble OpenAI LLM & MCP into `Agent` |
| `src/cli.py` | Interface | Live layout, command parsing, streaming |
| `tests/*` | QA | Regression & smoke tests |

### 4.3 Mermaid: Module Interaction

```mermaid
graph TD
  subgraph Terminal User
    U[User] --types msg--> CLI
  end

  subgraph Interface Layer
    CLI[CLI (Typer + Rich)]
  end

  subgraph Application Layer
    AG[Agent (pydantic_ai.Agent)]
  end

  subgraph Domain Layer
    HIS[History]
    THE[Themes]
  end

  subgraph External
    LLM[OpenAI API]
    MCP[Context7 MCP Server]
    Disk[(JSON File)]
    Env[(.env)]
  end

  %% edges
  CLI --> AG
  CLI --read/write--> HIS
  CLI --load--> THE
  CLI -.env vars.-> Env
  AG --> LLM
  AG --> MCP
  HIS --persist--> Disk
  MCP --subprocess--> mcp.config.json
```

---

## 5 Layer-by-Layer Walk-through

### 5.1 Configuration Layer 

Goals:

* Centralise every knob & secret.  
* Fail fast (missing API key → early exception).  
* Remain 12-factor friendly (override via ENV at runtime).  

**Design Choices**

* **Pydantic `BaseSettings`** – elegantly maps env → properties.  
* **dotenv** – auto-load `.env`, but environment variables still win (Docker/K8s).  
* Typed fields (`Literal`, `Path`) catch mis-configs early.

**Code Excerpt**

```python
class Settings(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    history_path: Path = ROOT / ".history.json"

    @validator("openai_api_key")  # fail-fast guard
    def _ensure_key(cls, v):
        if not v:
            raise ValueError("OPENAI_API_KEY missing")
        return v
```

Runtime retrieval is trivial:

```python
from src.config import settings
print(settings.theme)  # "cyberpunk"
```

### 5.2 Theming & UX Layer 

We store Rich `Theme` instances per palette. Switching themes is **hot**: the CLI calls `utils.switch_theme()`, which:

1. Cycles through the dictionary keys.  
2. Re-instantiates a global `Console` with new theme.  
3. Returns theme name for status feedback.

Key tip: Rich components don’t retain colour at render time; they pull from the *console* at print-time. So switching consoles mid-run changes later renders but leaves older tokens unaffected—works for us.

**Snippet**

```python
THEMES = {"cyberpunk": CYBERPUNK, "ocean": OCEAN, ...}

def switch_theme() -> str:
    global _console
    next_theme = next(_theme_iter)
    _console = Console(theme=THEMES[next_theme])
    return next_theme
```

### 5.3 Persistence & History Layer 

Design axioms:

* **Stateless agent** → conversation memory lives outside in `History`.  
* Structure mirrors *exactly* `pydantic_ai.ChatMessage` for friction-free conversion.  
* Use **append & dump** for simplicity; no concurrency issues in CLI.

Potential evolutions:

* Migrate to SQLite for multi-thread scenarios.  
* Encrypt file with OS keystore for privacy.

### 5.4 LLM / MCP Abstraction Layer 

`agent.py` weaves two back-ends:

* **LLM** – `OpenAIModel` streaming tokens.  
* **MCP** – `MCPServerStdio` running Node sub-process (stdio communication).

Pydantic-AI’s *tool calling* means that if the LLM returns JSON like:

```json
{"tool": "MCP.search", "arguments": ["quantum computing"]}
```

…the framework automatically forwards to MCP server and returns its output into the generator stream (our CLI catches it as role `"mcp"`).

The system prompt hard-codes the tool names:

```python
"When a user asks about a topic, call MCP.search(\"<topic>\")"
```

This pattern is remarkably robust; no extra parsing logic needed.

### 5.5 Interactive TUI & CLI Layer 

We purposely avoided `rich.layout.Layout` and leveraged a *vertical* `Group`, each sub-panel self-sizing (header, chat, results, footer). Simpler, less brittle.

The heart is `ChatLayout`:

```python
class ChatLayout:
    chat_lines: list[str]
    results: Optional[str]
    status: str  # displayed in footer

    def __rich__(self):
        return Group(
            self._render_header(),
            self._render_chat(),
            self._render_results(),
            self._render_footer(),
        )
```

A `Live(layout)` context continuously re-renders when the async message stream yields. We call `live.refresh()` sparingly to save CPU.

**Slash command detection**:  

1. Input starting with `/` branches early.  
2. Unknown commands fall through to help.  
3. Non-slash text becomes a user message.

### 5.6 Test & Quality-Gate Layer 

We ship `pytest` smoke tests covering:

* History save-load round-trip.  
* Agent instantiation (ensures environment correct).  
* Async chat (uses minimal “Ping?” to verify LLM path).

CI pipelines can set a dummy fake-model or mock OpenAI to keep tests free.

---

## 6 Deep-Dive Modules

### 6.1 `config.py` 

Why `ROOT / ".env"` load? Because during `pipx run` the CWD may be nested. Loading relative to `config.py` guarantees local discovery. If `.env` is absent the program still runs—only mandatory variable is `OPENAI_API_KEY`.

### 6.2 `themes.py`

Palette design guidelines:

* Use “bold” for primary colour; match border.  
* Keep *user* text bright white to contrast any background.  
* Provide *assistant* highlight distinct from *sys* messages.

### 6.3 `utils.py`

Banner ASCII is set inside a `Panel` to auto-adjust width. Developers may replace with your logo—just keep style keys consistent (`banner`, `border`).

### 6.4 `history.py`

We intentionally don’t store timestamps because MCP search results include metadata timestamps; mixing them inside history clutters output. If need arises, extend:

```python
ChatMessage(role="user", content="...", meta={"ts": time.time()})
```

### 6.5 `agent.py`

Async generator `stream_reply()` demonstrates idiomatic pattern:

```python
async for event in agent.stream_chat(...):
    if isinstance(event, ChatMessage):
        ...
    else:
        ...
```

Under the hood Pydantic-AI opens **two websockets** to OpenAI (one for streaming, one for tool calls) and a **stdio duplex** to MCP. You don’t touch any of that plumbing.

### 6.6 `cli.py` 

**Concurrency Model**

Python 3.11 improved cooperative `asyncio` performance; we wrap each user message cycle in `asyncio.run(_consume())`. This keeps global event-loop short-lived, avoiding cross-platform issues (Windows event-loop glitch). For heavy future features (continuous voice IO) we may move to a persistent loop.

**Rich Live Best Practices**

* `auto_refresh=False` then manual `live.refresh()` avoids 60 fps redraw storms.  
* Keep layout small: we purposely skip nested `Layout` spanners.  
* Use `Prompt.ask()` (blocking) outside Live to avoid stream collisions.

**Command Expansion**

Commands are parsed via simple equality checks. For advanced DSL consider `typer.Command` sub-commands or `argparse` inside chat (out-of-scope for MVP).

---

## 7 Runtime Life-Cycle & Control-Flow 

1. CLI process starts (`python -m src.cli chat`).  
2. `config.py` loads env; mis-config aborts early.  
3. Banner printed.  
4. `History` attempts to restore previous `.history.json`.  
5. CLI enters REPL loop.  
6. On user message:  
   a. Save to History.  
   b. Create `Agent` (cheap, stateless).  
   c. Start async stream.  
7. `Agent` sends conversation to OpenAI.  
8. If LLM decides `MCP.search`, framework calls MCP sub-process.  
9. MCP returns list of matches (e.g., JSON).  
10. Framework yields MCP event → CLI panel update.  
11. LLM continues drafting answer referencing MCP results.  
12. On stream completion, History persisted; loop awaits next user input.

Graceful Shutdown:

* Ctrl-D / Ctrl-C captured → `typer.Exit`.  
* MCP sub-process is child of Python process; OS cleans up automatically.

---

## 8 Error-Handling & Observability 

| Failure | Detection | UX Reaction |
|---------|-----------|-------------|
| Missing API key | Pydantic validator | Red error banner, exits |
| MCP not found | `FileNotFoundError` on spawn | Styled alert “MCP boot failed, check Node/npm” |
| OpenAI quota | `openai.error` | Footer status becomes “Error: quota exceeded” |
| JSON history corruption | `json.JSONDecodeError` | History reset, backup file created |

All exceptions funnel through CLI which prints in `[red]` channel.

Logging (future):

* Add `structlog` sink to file for debug.  
* Emit MCP raw traffic when `DEBUG=1` env var set.

---

## 9 Developer Experience 

### 9.1 Install

```bash
git clone git@github.com:org/context7-agent.git
cd context7-agent
pyenv install 3.11.9
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env && $EDITOR .env  # add API key
pytest            # should pass
```

### 9.2 Hot Reload

Use `pip install watchfiles` then:

```bash
watchmedo auto-restart -d src -p '*.py' -- python -m src.cli chat
```

### 9.3 Contribution Flow

* Branch from `main`.  
* Create descriptive branch (`feat/bookmarks`).  
* **Pre-commit** hooks run `black`, `ruff`, `pytest -q`.  
* PR template requires updating PAD sections if architecture evolves.

---

## 10 Road-Map & Extension Points 

1. **Bookmarks** – new MCP verb; store list in JSON; `/bookmark N`.  
2. **Search Analytics** – plug `pandas` to analyse query frequency.  
3. **Quantum-Mode** – feed MCP index with quantum computing corpus; raise LLM temperature for ideation.  
4. **Voice I/O** – `sounddevice` mic input + TTS output.  
5. **Electron wrapper** – re-package CLI into desktop app with x-term emulator.

---

## 11 Glossary 

* **MCP** – Model Context Protocol; tool API for semantic doc indexing/search.  
* **Pydantic-AI Agent** – high-level object orchestrating LLM & tools.  
* **Rich** – Python lib for gorgeous terminal rendering (markdown, tables).  
* **Typer** – Click-powered CLI micro-framework using type hints.  
* **TTK** – Time-to-Knowledge; performance metric we optimise.

---

## Appendix A – Example Session Transcript (annotated)

```text
❯ /theme
[sys] Theme switched to ocean
❯ Tell me about Boson sampling
[assistant] Boson sampling is a restricted model of quantum computation…
[mcp] 5 matches:
  1. “Boson_Sampling_Primer.pdf” (2024, 410 KB)
  2. “Aaronson_Original_Paper.pdf” (2011, 2 MB)
❯ /preview 1
[mcp] ―――――――――――――――――――――――
Title: Boson Sampling Primer
Snip:
  | Boson sampling is a non-universal model…
```

---

Enjoy building atop **Context7-Agent** – may your terminal be ever colourful and your knowledge ever-expanding!
