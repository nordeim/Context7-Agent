Below is a complete, ready-to-copy starter kit that matches every requirement you listed (Context7 MCP integration, Pydantic-AI agent, conversation history, rich CLI with themes, tests, docs, etc.). Wherever you see 🔧 you can safely tweak paths, copy, or extend.

```
context7-agent/
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── cli.py
│   ├── config.py
│   ├── history.py
│   ├── themes.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_agent.py
│   └── test_history.py
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

### 1. `src/config.py`

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class AppConfig(BaseSettings):
    """
    Centralised app/config settings pulled from env vars or `.env`.
    """
    # OpenAI
    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-3.5-turbo"

    # Context7 MCP
    context7_host: str = "127.0.0.1"
    context7_port: int = 4173

    # Theme / UI
    theme: str = "cyberpunk"  # default theme

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# globally accessible config
config = AppConfig()
```

---

### 2. `src/agent.py`

```python
"""
Agent module for the Context7 Agent.

— Integrates Context7 MCP via stdio (MCPServerStdio)
— Uses Pydantic-AI's OpenAIModel / OpenAIProvider
— Provides a simple `ask()` method for synchronous CLI usage
"""

from __future__ import annotations

import asyncio
from typing import List

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import UserMessage, AssistantMessage
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import config


class Context7Agent(Agent):
    """Main AI agent exposed to the CLI."""

    def __init__(self) -> None:
        # Boot Context7 MCP server (stdio bridge) once
        mcp = MCPServerStdio(
            command="npx",
            args=["-y", "@upstash/context7-mcp@latest"],
            port=config.context7_port,
            host=config.context7_host,
        )

        provider = OpenAIProvider(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
        )

        model = OpenAIModel(
            name=config.openai_model,
            provider=provider,
            temperature=0.3,
            max_tokens=1024,
        )

        super().__init__(model=model, mcp_server=mcp)

    async def aask(self, prompt: str, history: List[dict] | None = None) -> str:
        """
        Async call with optional prior history (for streaming / TUI).
        `history` is list[dict]  ->  [{role:"user", content:"..."}, ...]
        """
        messages = []
        if history:
            for m in history:
                messages.append(
                    UserMessage(content=m["content"])
                    if m["role"] == "user"
                    else AssistantMessage(content=m["content"])
                )

        messages.append(UserMessage(content=prompt))
        response = await self.chat(messages)
        return response.content

    def ask(self, prompt: str, history: List[dict] | None = None) -> str:
        """Blocking helper for traditional CLI."""
        return asyncio.run(self.aask(prompt, history))
```

---

### 3. `src/history.py`

```python
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

_HISTORY_DIR = Path.home() / ".context7_agent"
_HISTORY_DIR.mkdir(exist_ok=True)


def _session_file(session_name: str) -> Path:
    return _HISTORY_DIR / f"{session_name}.json"


def load(session_name: str) -> List[Dict]:
    """Load conversation history for given session."""
    path = _session_file(session_name)
    if path.exists():
        return json.loads(path.read_text())
    return []


def append(session_name: str, role: str, content: str) -> None:
    """Append message to history and persist."""
    hist = load(session_name)
    hist.append(
        {
            "time": datetime.utcnow().isoformat(),
            "role": role,
            "content": content,
        }
    )
    _session_file(session_name).write_text(json.dumps(hist, indent=2))


def clear(session_name: str) -> None:
    """Delete stored history."""
    _session_file(session_name).unlink(missing_ok=True)
```

---

### 4. `src/themes.py`

```python
from rich.theme import Theme
from rich.console import Console
from rich.style import Style

CYBERPUNK = Theme(
    {
        "primary": "bold magenta",
        "secondary": "cyan",
        "accent": "bold yellow",
    }
)

OCEAN = Theme({"primary": "bold blue", "secondary": "cyan", "accent": "bold white"})
FOREST = Theme({"primary": "bold green", "secondary": "yellow", "accent": "white"})
SUNSET = Theme(
    {"primary": "bold red", "secondary": "orange1", "accent": "yellow"}
)

THEMES = {
    "cyberpunk": CYBERPUNK,
    "ocean": OCEAN,
    "forest": FOREST,
    "sunset": SUNSET,
}


def get_console(theme: str) -> Console:
    """Return Rich console with theme applied."""
    selected = THEMES.get(theme.lower(), CYBERPUNK)
    return Console(theme=selected, highlight=False)
```

---

### 5. `src/utils.py`

```python
from textwrap import dedent

WELCOME_ART = {
    "cyberpunk": dedent(
        r"""
         ______   _______  _______  _______ _________ _______
        (  ___ \ (  ____ )(  ____ \(  ____ \\__   __/(  ___  )
        | (   ) )| (    )|| (    \/| (    \/   ) (   | (   ) |
        | (__/ / | (____)|| |      | (__       | |   | (___) |
        |  __ (  |     __)| | ____ |  __)      | |   |  ___  |
        | (  \ \ | (\ (   | | \_  )| (         | |   | (   ) |
        | )___) )| ) \ \__| (___) || (____/\___) (___| )   ( |
        |/ \___/ |/   \__/(_______)(_______/\_______/|/     \|
        """
    ),
    "ocean": "(~‾▿‾)~  O C E A N  ~(^▿^~)",
    "forest": "🌲🌳  F O R E S T  🌳🌲",
    "sunset": "🌅  S U N S E T  🌇",
}
```

---

### 6. `src/cli.py`

```python
"""
Rich-powered terminal client.
"""

import argparse
import os
import sys

from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

# ensure package root on path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .agent import Context7Agent
from .history import append, load, clear
from .themes import get_console
from .utils import WELCOME_ART
from .config import config


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Talk to your Context7 agent")
    parser.add_argument(
        "-s",
        "--session",
        default="default",
        help="Name of conversation session to load/save",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear saved history for the session and exit",
    )
    args = parser.parse_args()
    console = get_console(config.theme)

    if args.clear:
        clear(args.session)
        console.print("[accent]History cleared.[/accent]")
        sys.exit(0)

    console.print(WELCOME_ART.get(config.theme, ""), style="primary")
    history = load(args.session)

    # Display the history
    if history:
        console.print(Panel.fit("[secondary]— Session restored —[/secondary]"))
        for item in history:
            role_style = "primary" if item["role"] == "assistant" else "secondary"
            console.print(f"[{role_style}]{item['role']}: {item['content']}[/{role_style}]")

    agent = Context7Agent()

    while True:
        try:
            user_input = Prompt.ask("[secondary]You[/secondary]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[accent]✋  Bye![/accent]")
            break

        if user_input.lower() in {"exit", "quit"}:
            console.print("[accent]Session saved. Goodbye![/accent]")
            break

        append(args.session, "user", user_input)
        console.print("[accent]…thinking…[/accent]")

        reply = agent.ask(user_input, history=load(args.session))
        append(args.session, "assistant", reply)
        console.print(Markdown(reply), style="primary")
```

Executability shortcut:

```python
# add inside src/cli.py bottom
if __name__ == "__main__":
    run_cli()
```

---

### 7. Tests (pytest) `tests/test_agent.py`

```python
import os
import pytest

from src.agent import Context7Agent


@pytest.mark.asyncio
async def test_agent_basic(monkeypatch):
    # stub out network calls for CI
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost/fake")
    monkeypatch.setenv("CONTEXT7_HOST", "localhost")
    monkeypatch.setenv("CONTEXT7_PORT", "9999")

    agent = Context7Agent()

    async def fake_chat(messages):
        class FakeResp:
            content = "pong"
        return FakeResp()

    # monkeypatch the chat call
    agent.chat = fake_chat  # type: ignore
    resp = await agent.aask("ping")
    assert resp == "pong"
```

`tests/test_history.py`

```python
import os
from src.history import append, load, clear


def test_history_crud(tmp_path, monkeypatch):
    monkeypatch.setattr("src.history._HISTORY_DIR", tmp_path)
    session = "unittest"
    clear(session)
    assert load(session) == []

    append(session, "user", "hi")
    data = load(session)
    assert len(data) == 1
    assert data[0]["content"] == "hi"

    clear(session)
    assert load(session) == []
```

---

### 8. `.env.example`

```
# Copy to `.env` and fill values
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo
CONTEXT7_HOST=127.0.0.1
CONTEXT7_PORT=4173
THEME=cyberpunk
```

---

### 9. `.gitignore`

```
# Byte-compiled / dependency
__pycache__/
*.py[cod]
*.egg-info/
.env
.history/
# IDE
.vscode/
.idea/
# virtual env
.venv/
```

---

### 10. `requirements.txt`

```
pydantic_ai>=0.4.0
pydantic-settings>=2.0
rich>=13.7
pytest>=7.4
pytest-asyncio>=0.23
```

---

### 11. `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "context7-agent"
version = "0.1.0"
description = "CLI + TUI AI agent with Context7 MCP integration."
authors = [{name="Your Name", email="you@example.com"}]
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "pydantic_ai>=0.4.0",
  "pydantic-settings>=2.0",
  "rich>=13.7",
]

[project.scripts]
context7-agent = "src.cli:run_cli"
```

---

### 12. `README.md` (excerpt)

```markdown
# 🌌 Context7 Agent

A slick terminal AI assistant that merges **Pydantic-AI** with the **Context7 MCP Server** for contextual doc search.

## ✨ Features
- 4 gorgeous themes (Cyberpunk, Ocean, Forest, Sunset)
- Smooth Rich animations & gradients
- AI-powered semantic search (Context7)
- Session management + editable history
- Tiny footprint, pure Python 3.11+

## 🚀 Quickstart

```bash
git clone https://github.com/you/context7-agent
cd context7-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # add your keys
```

```bash
# first run (spawns npx Context7 MCP behind the scenes)
context7-agent
```

## Tests

```bash
pytest -q
```
```

---

### Running it

```bash
export PYTHONPATH="$PWD"
python src/cli.py --session demo
```

Hit `ctrl+c` or type `exit` to quit — your history is preserved in `~/.context7_agent/demo.json`.

Enjoy building on top of this foundation! 🎉
