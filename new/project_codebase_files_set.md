# .gitignore
```txt
# virtual env / poetry / pipenv
.env
.venv/
env/
# caches, pyc
__pycache__/
*.py[cod]
# pytest
.pytest_cache/
# user data
.history.json

```

# .env.example
```example
# === OpenAI ==================================================================
OPENAI_API_KEY=sk-YOUR_KEY_HERE
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1

# === UI ======================================================================
THEME=cyberpunk             # cyberpunk | ocean | forest | sunset

```

# requirements.txt
```txt
pydantic-ai>=0.3.1
openai>=1.23.0
rich>=13.7.1
typer[all]>=0.9.0
python-dotenv>=1.0.1
pytest>=8.2.0
pytest-asyncio>=0.23.6

```

# pyproject.toml
```toml
[project]
name = "context7-agent"
version = "0.1.0"
description = "Immersive CLI agent powered by Pydantic-AI + Context7 MCP"
authors = [{ name = "You" }]
dependencies = [
    "pydantic-ai>=0.3.1",     # Adjust to actual version
    "openai>=1.23.0",
    "rich>=13.7.1",
    "typer[all]>=0.9.0",
    "python-dotenv>=1.0.1",
]

[project.optional-dependencies]
dev = ["pytest>=8.2.0", "pytest-asyncio>=0.23.6"]

[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

```

# mcp.config.json
```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    }
  }
}

```

# src/__init__.py
```py
# File: src/__init__.py
"""Context7 Agent – package root."""

```

# src/history.py
```py
# File: src/history.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List

from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str
    content: str

class History:
    """
    Persistent JSON history store compatible with pydantic-ai ChatMessage.
    """

    def __init__(self, file: Path):
        self.file = file
        self.messages: List[ChatMessage] = []
        self.load()

    # ---------------------------------------------------------------- load/save
    def load(self):
        if self.file.exists():
            try:
                raw = json.loads(self.file.read_text(encoding="utf-8"))
                self.messages = [ChatMessage(**m) for m in raw]
            except Exception:  # pragma: no cover
                self.messages = []

    def save(self):
        self.file.write_text(
            json.dumps([m.dict() for m in self.messages], indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------ helpers
    def add(self, role: str, content: str):
        self.messages.append(ChatMessage(role=role, content=content))
        self.save()

    def to_model_messages(self) -> List[ChatMessage]:
        return self.messages


```

# src/agent.py
```py
from __future__ import annotations
import asyncio
from typing import AsyncIterator

from pydantic_ai import Agent
from pydantic import BaseModel
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import settings
from .history import History

# --------------------------------------------------------------------- SYSTEM
SYSTEM_PROMPT = """
You are Context7, a futuristic librarian.
When a user asks about a *topic*, issue a MCP.search call:  MCP.search("<topic>")
When a user types /preview N or similar, call MCP.preview.
Always format factual answers in concise markdown.

If unsure, politely ask for clarification.
"""

class ChatMessage(BaseModel):
    role: str
    content: str

def _build_llm() -> OpenAIModel:
    provider = OpenAIProvider(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    return OpenAIModel(
        provider=provider,
        model=settings.openai_model,  # now only passed to OpenAIModel
        temperature=0.3,
        max_tokens=2048,
    )

def _build_mcp() -> MCPServerStdio:
    return MCPServerStdio(server=settings.mcp_alias)  # reads mcp.config.json

def create_agent() -> Agent:
    return Agent(model=_build_llm(), mcp_server=_build_mcp(), system_prompt=SYSTEM_PROMPT)

# ------------------------------------------------------------------ high-level
async def stream_reply(history: History) -> AsyncIterator[tuple[str, str]]:
    """
    Merge assistant tokens and MCP tool events into a single async stream.
    Yields tuples (role, payload)
      role == "assistant" -> markdown str
      role == "mcp"       -> already formatted message (string)
    """
    agent = create_agent()

    async for event in agent.stream_chat(messages=history.to_model_messages()):
        if isinstance(event, ChatMessage):
            yield event.role, event.content
        else:
            yield "mcp", str(event)

```

# src/config.py
```py
# File: src/config.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings  # <-- Correct import for v2+
from pydantic import Field, validator

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")      # cascade env vars


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini", env="OPENAI_MODEL")
    openai_base_url: str | None = Field(None, env="OPENAI_BASE_URL")

    # MCP
    mcp_alias: str = "context7"                         # key in mcp.config.json
    mcp_config_path: Path = ROOT / "mcp.config.json"

    # UX
    theme: Literal["cyberpunk", "ocean", "forest", "sunset"] = Field(
        default="cyberpunk", env="THEME"
    )
    history_path: Path = ROOT / ".history.json"

    class Config:
        env_file = ".env"

    # Extra validation
    @validator("openai_api_key", allow_reuse=True)
    def _check_key(cls, v: str):
        if not v or v.startswith("sk-YOUR"):
            raise ValueError("OPENAI_API_KEY missing or placeholder")
        return v


settings = Settings()
config = settings

```

# src/utils.py
```py
# File: src/utils.py
from __future__ import annotations
from itertools import cycle

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .themes import THEMES
from .config import settings

_console = Console(theme=THEMES[settings.theme])
_theme_iter = cycle(THEMES.keys())


def get_console() -> Console:
    """Return singleton console, re-instantiated if theme changed."""
    return _console


def banner() -> None:
    c = get_console()
    art = Text(
        """
 ██████╗ ██████╗ ██████╗ ███████╗████████╗ ██████╗██╗  ██╗
██╔════╝██╔═══██╗██╔══██╗██╔════╝╚══██╔══╝██╔════╝██║ ██╔╝
██║     ██║   ██║██████╔╝█████╗     ██║   ██║     █████╔╝ 
██║     ██║   ██║██╔══██╗██╔══╝     ██║   ██║     ██╔═██╗ 
╚██████╗╚██████╔╝██║  ██║███████╗   ██║   ╚██████╗██║  ██╗
 ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
""",
        style="banner",
    )
    c.print(Panel(art, border_style="border"))


def switch_theme() -> str:
    """
    Cycle through available themes and return the new theme name.
    """
    global _console
    next_theme = next(_theme_iter)
    _console = Console(theme=THEMES[next_theme])
    return next_theme

```

# src/cli.py
```py
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

```

# src/themes.py
```py
# File: src/themes.py
from __future__ import annotations
from rich.theme import Theme

CYBERPUNK = Theme(
    {
        "banner": "bold magenta",
        "border": "magenta",
        "user": "bold bright_white",
        "assistant": "bright_magenta",
        "sys": "bright_cyan",
    }
)

OCEAN = Theme(
    {
        "banner": "bold blue",
        "border": "blue",
        "user": "bright_white",
        "assistant": "bright_cyan",
        "sys": "cyan",
    }
)

FOREST = Theme(
    {
        "banner": "bold green",
        "border": "green",
        "user": "bright_white",
        "assistant": "green",
        "sys": "yellow",
    }
)

SUNSET = Theme(
    {
        "banner": "bold red",
        "border": "red",
        "user": "bright_white",
        "assistant": "bright_red",
        "sys": "orange1",
    }
)

THEMES = {
    "cyberpunk": CYBERPUNK,
    "ocean": OCEAN,
    "forest": FOREST,
    "sunset": SUNSET,
}

```

# src/config.py.bak2
```bak2
# File: src/config.py
"""
Enhanced configuration module for Context7 Document Explorer.
"""

import os
from dataclasses import dataclass, field
from typing import List
from pathlib import Path


@dataclass
class Config:
    """Configuration for the Context7 Document Explorer."""
    
    # OpenAI configuration
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    
    # Context7 configuration
    context7_workspace: str = "default"
    context7_index_path: str = "./documents"
    
    # UI configuration
    theme: str = "cyberpunk"  # cyberpunk, ocean, forest, sunset
    animations_enabled: bool = True
    sound_effects: bool = True
    
    # Search configuration
    max_results: int = 20
    preview_lines: int = 5
    highlight_matches: bool = True
    fuzzy_search: bool = True
    
    # Application settings
    data_dir: Path = field(default_factory=lambda: Path.home() / ".context7_explorer")
    history_file: str = "search_history.json"
    bookmarks_file: str = "bookmarks.json"
    sessions_dir: str = "sessions"
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config = cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4"),
            context7_workspace=os.getenv("CONTEXT7_WORKSPACE", "default"),
            context7_index_path=os.getenv("CONTEXT7_INDEX_PATH", "./documents"),
            theme=os.getenv("THEME", "cyberpunk"),
            animations_enabled=os.getenv("ANIMATIONS_ENABLED", "true").lower() == "true",
            sound_effects=os.getenv("SOUND_EFFECTS", "false").lower() == "true",
        )
        
        # Ensure directories exist
        config.data_dir.mkdir(parents=True, exist_ok=True)
        (config.data_dir / config.sessions_dir).mkdir(exist_ok=True)
        
        return config
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY environment variable is required")
        
        if not Path(self.context7_index_path).exists():
            Path(self.context7_index_path).mkdir(parents=True, exist_ok=True)
        
        return errors


# Global config instance
config = Config.from_env()

```

# tests/__init__.py
```py
# File: tests/__init__.py
"""py.test package marker."""

```

# tests/test_history.py
```py
# File: tests/test_history.py
from pathlib import Path
from src.history import History

def test_roundtrip(tmp_path: Path):
    f = tmp_path / "hist.json"
    h = History(f)
    h.add("user", "hello")
    h2 = History(f)
    assert len(h2.messages) == 1

```

# tests/test_agent.py
```py
# File: tests/test_agent.py
import pytest
import asyncio
from src.agent import create_agent

@pytest.mark.asyncio
async def test_agent_basic():
    agent = create_agent()
    reply = await agent.chat("Hello!")
    assert isinstance(reply.content, str)

```

# README.md
```md
# ⚡ Context7 Agent

A dazzling terminal librarian that lets you **chat** while it hunts documents
via the Context7 **MCP** protocol.  
Rich animations, four flashy themes, live-streamed answers, bookmarks (soon) –
all in one convenient Python package.

## 📸 Screenshot

```
╭────────────────────────────────────────────────────────╮
│ ⚡  Context7 Agent  ⚡                                 │
├────────────────────────────────────────────────────────┤
│ **You:** Tell me something about quantum computing     │
│ **AI:** Sure! First, here’s a bite-size intro …        │
├──────────────── Results ────────────────┬─────────────┤
│ search#42  Quantum Computing Basics.pdf │  1 MB  2023 │
│ search#43  Shor’s Algorithm.md          │  5 KB  2022 │
╰────────────────────────────────────────────────────────╯
```

## 🚀 Quick start

```bash
git clone …
cd context7-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env       # add your OpenAI key

python -m src.cli chat
```

First run will auto-download and boot the MCP server using  
`npx -y @upstash/context7-mcp@latest`.

### Slash commands
| Command | Description            |
|---------|------------------------|
| `/theme`| Cycle colour themes    |
| `/help` | Show quick cheatsheet  |
| `/exit` | Quit & save history    |

---

## 🧪 Tests
```bash
pytest
```
---

## 📂 Project layout
```
src/         main code (agent, cli, themes…)
tests/       pytest suite
mcp.config.json  Context7 server definition
```

---

Enjoy the future of terminal knowledge discovery! ✨

```

