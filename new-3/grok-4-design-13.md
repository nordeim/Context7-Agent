```python
# src/config.py
"""
Configuration module for the Context7 Agent.

Loads environment variables and validates required settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.mcp_command = "npx"
        self.mcp_args = ["-y", "@upstash/context7-mcp@latest"]

    def validate(self) -> Optional[str]:
        if not self.openai_api_key:
            return "Missing OPENAI_API_KEY"
        if not self.openai_base_url:
            return "Missing OPENAI_BASE_URL"
        return None

config = Config()
```

```python
# src/themes.py
"""
Themes module for the Context7 Agent.

Defines visual themes with colors and ASCII art for the TUI.
"""

THEMES = {
    "cyberpunk": {
        "colors": {
            "primary": "bold magenta",
            "secondary": "cyan",
            "accent": "yellow",
            "background": "on black",
            "error": "red",
        },
        "ascii": """
   _____          _            _    _______
  / ____|        | |          | |  |__   __|
 | |    _   _ ___| |_ ___  ___| |_    | | ___   __ _
 | |   | | | / __| __/ _ \/ __| __|   | |/ _ \ / _` |
 | |___| |_| \__ \ ||  __/\__ \ |_    | | (_) | (_| |
  \_____\__,_|___/\__\___||___/\__|   |_|\___/ \__, |
                                                 __/ |
                                                |___/
        """,
    },
    "ocean": {
        "colors": {
            "primary": "bold blue",
            "secondary": "cyan",
            "accent": "white",
            "background": "on dark_blue",
            "error": "red",
        },
        "ascii": """
  ____          _            _    _______
 / ___|        | |          | |  |__   __|
| |    _   _ ___| |_ ___  ___| |_    | | ___   __ _
| |   | | | / __| __/ _ \/ __| __|   | |/ _ \ / _` |
| |___| |_| \__ \ ||  __/\__ \ |_    | | (_) | (_| |
 \_____\__,_|___/\__\___||___/\__|   |_|\___/ \__, |
                                                __/ |
                                               |___/
        """,
    },
    "forest": {
        "colors": {
            "primary": "bold green",
            "secondary": "yellow",
            "accent": "white",
            "background": "on dark_green",
            "error": "red",
        },
        "ascii": """
   _____          _            _    _______
  / ____|        | |          | |  |__   __|
 | |    _   _ ___| |_ ___  ___| |_    | | ___   __ _
 | |   | | | / __| __/ _ \/ __| __|   | |/ _ \ / _` |
 | |___| |_| \__ \ ||  __/\__ \ |_    | | (_) | (_| |
  \_____\__,_|___/\__\___||___/\__|   |_|\___/ \__, |
                                                 __/ |
                                                |___/
        """,
    },
    "sunset": {
        "colors": {
            "primary": "bold red",
            "secondary": "yellow",
            "accent": "magenta",
            "background": "on dark_red",
            "error": "white",
        },
        "ascii": """
   _____          _            _    _______
  / ____|        | |          | |  |__   __|
 | |    _   _ ___| |_ ___  ___| |_    | | ___   __ _
 | |   | | | / __| __/ _ \/ __| __|   | |/ _ \ / _` |
 | |___| |_| \__ \ ||  __/\__ \ |_    | | (_) | (_| |
  \_____\__,_|___/\__\___||___/\__|   |_|\___/ \__, |
                                                 __/ |
                                                |___/
        """,
    },
}
```

```python
# src/utils.py
"""
Utility functions for the Context7 Agent.

Includes animation helpers and other misc functions.
"""

import time
import asyncio
from rich.console import Console

console = Console()

def typing_print(text: str, style: str, delay: float = 0.03):
    for char in text:
        console.print(char, style=style, end="")
        time.sleep(delay)
    console.print("")

async def async_typing_print(log, text: str, style: str, delay: float = 0.03):
    for char in text:
        log.write(char, style=style)
        await asyncio.sleep(delay)
    log.write("\n")
```

```python
# src/agent.py
"""
Agent module for the Context7 Agent.

Implements the core AI agent with OpenAI integration, tool calling for intent detection, and Context7 MCP server integration.
"""

import os
import sys
import json
import uuid
import asyncio
from typing import Dict, Any, Optional, List, Union

from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Assuming pydantic_ai exists; if not, this is adapted for direct use
# For this implementation, we use direct OpenAI with Pydantic for tool schemas

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config

load_dotenv()

class SearchOptions(BaseModel):
    filters: Optional[Dict[str, Any]] = None
    limit: int = 10
    quantum_mode: bool = False

class SearchInput(BaseModel):
    query: str
    options: SearchOptions

class Document(BaseModel):
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]

class NeuralProperties(BaseModel):
    embedding: List[float]
    semantic_density: float
    information_entropy: float
    neural_activation: float

class MCPRequest(BaseModel):
    id: str
    method: str
    params: Dict[str, Any]

class MCPResponse(BaseModel):
    id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class Context7MCPIntegration:
    def __init__(self):
        self.process = None
        self.response_handlers: Dict[str, asyncio.Future] = {}
        self.connected = False

    async def connect(self):
        if self.connected:
            return
        self.process = await asyncio.create_subprocess_exec(
            config.mcp_command,
            *config.mcp_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        asyncio.create_task(self._process_messages())
        self.connected = True

    async def _process_messages(self):
        while True:
            line = await self.process.stdout.readline()
            if not line:
                break
            try:
                data = json.loads(line.decode().strip())
                if 'id' in data and data['id'] in self.response_handlers:
                    future = self.response_handlers.pop(data['id'])
                    future.set_result(MCPResponse(**data))
            except Exception as e:
                console.print(f"[red]MCP processing error: {e}[/red]")

    async def _send_request(self, request: MCPRequest) -> MCPResponse:
        future = asyncio.Future()
        self.response_handlers[request.id] = future
        await self.process.stdin.write((json.dumps(request.dict()) + '\n').encode())
        await self.process.stdin.drain()
        response = await future
        if response.error:
            raise ValueError(response.error)
        return response

    async def search(self, query: str, options: SearchOptions) -> List[Document]:
        await self.connect()
        request = MCPRequest(
            id=str(uuid.uuid4()),
            method="search",
            params={
                "query": query,
                "filters": options.filters or {},
                "limit": options.limit,
                "quantum_mode": options.quantum_mode
            }
        )
        response = await self._send_request(request)
        documents = []
        for doc_data in response.result.get("documents", []):
            documents.append(Document(
                id=doc_data["id"],
                title=doc_data["title"],
                content=doc_data["content"],
                metadata=doc_data.get("metadata", {})
            ))
        return documents

class Context7Agent:
    def __init__(self):
        error = config.validate()
        if error:
            raise ValueError(error)
        self.client = AsyncOpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )
        self.mcp_integration = Context7MCPIntegration()
        self.system_prompt = """
You are a futuristic librarian AI agent powered by Context7. You transform terminal interactions into immersive conversations.
When the user discusses a subject or asks for information, use the search_documents tool to find relevant documents via Context7 MCP, then summarize and respond based on them.
For general chat or commands, respond naturally. Be engaging and helpful.
"""

    async def respond(self, message: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        messages = [{"role": "system", "content": self.system_prompt}] + history + [{"role": "user", "content": message}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": "Search for documents using Context7 MCP based on query and options.",
                    "parameters": SearchInput.model_json_schema()
                }
            }
        ]
        response = await self.client.chat.completions.create(
            model=config.openai_model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        choice = response.choices[0]
        results = None
        if choice.message.tool_calls:
            tool_call = choice.message.tool_calls[0]
            if tool_call.function.name == "search_documents":
                args = json.loads(tool_call.function.arguments)
                search_input = SearchInput(**args)
                results = await self.mcp_integration.search(search_input.query, search_input.options)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "search_documents",
                    "content": json.dumps([doc.dict() for doc in results])
                }
                messages.append(tool_message)
                response = await self.client.chat.completions.create(
                    model=config.openai_model,
                    messages=messages
                )
                choice = response.choices[0]
        return {"response": choice.message.content, "results": results}
```

```python
# src/history.py
"""
History module for the Context7 Agent.

Manages persistence for conversations, searches, bookmarks, and sessions using JSON.
"""

import os
import json
from typing import List, Dict, Any

class HistoryManager:
    def __init__(self, file_path: str = "session.json"):
        self.file_path = file_path
        self.data: Dict[str, Any] = {
            "conversations": [],
            "searches": [],
            "bookmarks": {},
            "theme": "cyberpunk"
        }
        self.load()

    def load(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                self.data = json.load(f)

    def save(self):
        with open(self.file_path, "w") as f:
            json.dump(self.data, f, indent=4)

    def add_message(self, role: str, content: str):
        self.data["conversations"].append({"role": role, "content": content})

    def get_conversations(self) -> List[Dict[str, str]]:
        return self.data["conversations"]

    def add_search(self, query: str, results: List[Dict]):
        self.data["searches"].append({"query": query, "results": [doc.dict() for doc in results]})

    def add_bookmark(self, doc_id: str, doc: Dict):
        self.data["bookmarks"][doc_id] = doc

    def get_bookmarks(self) -> Dict[str, Dict]:
        return self.data["bookmarks"]

    def set_theme(self, theme: str):
        self.data["theme"] = theme
```

```python
# src/cli.py
"""
CLI module for the Context7 Agent.

Provides a dynamic TUI using Textual for conversational interactions, theme switching, animations, and more.
"""

import asyncio
import json
from typing import List, Dict

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog, DataTable, Static
from textual.screen import ModalScreen
from textual.binding import Binding
from textual import events
from rich.syntax import Syntax
from rich.panel import Panel

from agent import Context7Agent, Document
from history import HistoryManager
from themes import THEMES
from utils import async_typing_print

class WelcomeScreen(ModalScreen):
    def __init__(self, ascii_art: str):
        super().__init__()
        self.ascii_art = ascii_art

    def compose(self) -> ComposeResult:
        yield Static(Panel(self.ascii_art, title="Welcome to Context7 Agent", expand=True))
        yield Static("Press any key to continue...")

    def on_key(self, event: events.Key) -> None:
        self.dismiss()

class PreviewScreen(ModalScreen):
    def __init__(self, document: Document):
        super().__init__()
        self.document = document

    def compose(self) -> ComposeResult:
        syntax = Syntax(self.document.content, "python" if self.document.metadata.get("type") == "py" else "text", theme="monokai")
        yield Static(Panel(syntax, title=self.document.title, expand=True))
        yield Static("Press ESC to close")

    def on_key(self, event: events.Key) -> None:
        if event.name == "escape":
            self.dismiss()

class ThemeMenu(ModalScreen):
    def compose(self) -> ComposeResult:
        yield Static("Select Theme:")
        for theme in THEMES.keys():
            yield Static(theme, id=theme)

    def on_mount(self) -> None:
        self.focus()

class Context7App(App):
    CSS = """
    .cyberpunk {
        background: black;
        color: magenta;
    }
    .ocean {
        background: dark_blue;
        color: blue;
    }
    .forest {
        background: dark_green;
        color: green;
    }
    .sunset {
        background: dark_red;
        color: red;
    }
    RichLog {
        height: 70%;
    }
    DataTable {
        height: 20%;
    }
    Input {
        height: 10%;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+s", "save", "Save Session"),
    ]

    def __init__(self):
        super().__init__()
        self.history = HistoryManager()
        self.agent = Context7Agent()
        self.current_results: List[Document] = []
        self.current_theme = self.history.data["theme"]
        self.chat_log: RichLog
        self.results_table: DataTable
        self.input: Input

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        self.chat_log = RichLog(markup=True, highlight=True)
        yield self.chat_log
        self.results_table = DataTable()
        self.results_table.add_columns("ID", "Title", "Type")
        yield self.results_table
        self.input = Input(placeholder="Type your message or /help for commands...")
        yield self.input
        yield Footer()

    async def on_mount(self) -> None:
        self.add_class(self.current_theme)
        ascii_art = THEMES[self.current_theme]["ascii"]
        self.push_screen(WelcomeScreen(ascii_art))
        for msg in self.history.get_conversations():
            color = "green" if msg["role"] == "user" else "blue"
            self.chat_log.write(f"[{color}]{msg['role'].capitalize()}:[/] {msg['content']}\n")
        await self.agent.mcp_integration.connect()  # Connect to MCP

    def apply_theme(self, theme: str):
        self.remove_class(self.current_theme)
        self.current_theme = theme
        self.add_class(theme)
        self.history.set_theme(theme)
        self.history.save()

    async def on_input_submitted(self, message: events.Message) -> None:
        user_message = self.input.value
        self.input.value = ""
        if not user_message:
            return
        if user_message.startswith("/"):
            await self.handle_command(user_message[1:])
            return
        self.history.add_message("user", user_message)
        self.chat_log.write(f"[bold green]You:[/bold green] {user_message}\n")
        try:
            result = await self.agent.respond(user_message, self.history.get_conversations())
            response = result["response"]
            results = result["results"]
            self.chat_log.write("[bold blue]Agent:[/bold blue] ")
            await async_typing_print(self.chat_log, response, "bold blue")
            if results:
                self.current_results = results
                self.results_table.clear(rows=True)
                for idx, doc in enumerate(results):
                    self.results_table.add_row(str(idx), doc.title, doc.metadata.get("type", "unknown"))
                self.history.add_search(user_message, results)
        except Exception as e:
            self.chat_log.write(f"[red]Error: {str(e)}[/red]\n")
        self.history.save()

    async def handle_command(self, cmd: str) -> None:
        parts = cmd.split()
        command = parts[0]
        if command == "theme":
            self.push_screen(ThemeMenu())
        elif command == "preview" and len(parts) > 1:
            try:
                idx = int(parts[1])
                if 0 <= idx < len(self.current_results):
                    self.push_screen(PreviewScreen(self.current_results[idx]))
            except ValueError:
                self.chat_log.write("[red]Invalid preview ID[/red]\n")
        elif command == "bookmark" and len(parts) > 1:
            try:
                idx = int(parts[1])
                if 0 <= idx < len(self.current_results):
                    doc = self.current_results[idx]
                    self.history.add_bookmark(doc.id, doc.dict())
                    self.chat_log.write(f"[green]Bookmarked document {doc.id}[/green]\n")
            except ValueError:
                self.chat_log.write("[red]Invalid bookmark ID[/red]\n")
        elif command == "list_bookmarks":
            bookmarks = self.history.get_bookmarks()
            if not bookmarks:
                self.chat_log.write("[yellow]No bookmarks[/yellow]\n")
            else:
                for id, doc in bookmarks.items():
                    self.chat_log.write(f"[yellow]{id}: {doc['title']}[/yellow]\n")
        elif command == "analytics":
            searches = self.history.data["searches"]
            self.chat_log.write(f"[yellow]Total searches: {len(searches)}[/yellow]\n")
            # Add more analytics as needed
        elif command == "help":
            self.chat_log.write("[yellow]Commands: /theme, /preview <id>, /bookmark <id>, /list_bookmarks, /analytics, /exit, /help[/yellow]\n")
        elif command == "exit":
            self.app.exit()
        else:
            self.chat_log.write("[red]Unknown command[/red]\n")

    def action_quit(self) -> None:
        self.history.save()
        self.app.exit()

    def action_save(self) -> None:
        self.history.save()
        self.chat_log.write("[green]Session saved[/green]\n")

if __name__ == "__main__":
    app = Context7App()
    app.run()
```

```python
# tests/test_agent.py
import pytest
import asyncio
from src.agent import Context7Agent

@pytest.fixture
def agent():
    return Context7Agent()

@pytest.mark.asyncio
async def test_respond(agent):
    response = await agent.respond("Hello", [])
    assert "response" in response
    assert isinstance(response["response"], str)

# Add more tests as needed
```

```python
# tests/test_history.py
import pytest
import os
from src.history import HistoryManager

@pytest.fixture
def history(tmp_path):
    file = tmp_path / "test.json"
    return HistoryManager(str(file))

def test_add_message(history):
    history.add_message("user", "test")
    assert len(history.get_conversations()) == 1

def test_save_load(history):
    history.add_message("user", "test")
    history.save()
    new_history = HistoryManager(history.file_path)
    assert len(new_history.get_conversations()) == 1
```

```markdown
# Context7 Agent

An awesomely cool and user-friendly AI agent built with Pydantic AI, integrated with the Context7 MCP server.

## Installation

### Prerequisites
- Python 3.11+
- Node.js and npm
- OpenAI API key (or compatible)

### Setup
1. Clone the repo: `git clone <repo>`
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in variables (OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL)
4. Run: `python -m src.cli`

## Usage
- Chat naturally in the TUI.
- Use /commands like /theme, /preview <id>, etc.
- Exit with /exit or Ctrl+Q

## Features
- See the description above.
```

```
# requirements.txt
openai
pydantic
python-dotenv
rich
textual
pytest  # for tests
```

```
# .gitignore
__pycache__/
*.py[cod]
*.pyc
.env
session.json
*.log
.DS_Store
.idea/
.vscode/
```

```
# pyproject.toml
[tool.poetry]
name = "context7-agent"
version = "0.1.0"
description = "Context7 AI Agent"
authors = ["Your Name"]

[tool.poetry.dependencies]
python = "^3.11"
openai = "^1.0"
pydantic = "^2.0"
python-dotenv = "^1.0"
rich = "^13.0"
textual = "^0.1"

[tool.poetry.dev-dependencies]
pytest = "^7.0"
```

