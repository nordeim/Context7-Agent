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

# pyproject.toml
```toml
[tool.poetry]
name = "context7-agent"
version = "0.1.0"
description = "AI agent with Pydantic AI and Context7 MCP"
authors = ["Your Name"]

[tool.poetry.dependencies]
python = "^3.11"
pydantic-ai = "*"
rich = "*"
openai = "*"
python-dotenv = "*"
pytest = "*"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

```

# requirements.txt
```txt
pydantic_ai 
rich
openai      
python-dotenv
pytest
anyio

```

# src/agent.py
```py
# File: src/agent.py
"""
Agent module for the Context7 Agent.

This module implements a Pydantic AI agent with Context7 MCP server integration.
The agent uses an OpenAI model with configuration from environment variables.
"""

import os
import sys
from typing import Dict, Any, Optional, List, Union

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel as OpenAI_LLM
from pydantic_ai.providers.openai import OpenAIProvider
import openai

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from history import History


class Context7Agent:
    """
    Context7 Agent implementation using Pydantic AI.

    This agent integrates with the Context7 MCP server for enhanced context management
    and uses an OpenAI model with OpenAIProvider as the underlying LLM provider.
    Supports intent detection, MCP searches, and conversational responses.
    """

    def __init__(self):
        """
        Initialize the Context7 Agent with configuration from environment variables.

        Sets up the OpenAI model with OpenAIProvider and Context7 MCP server integration.
        """
        error = config.validate()
        if error:
            raise ValueError(error)

        # The provider is needed for its synchronous client and for the Agent.
        self.provider = OpenAIProvider(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
        )
        
        # We need a separate, dedicated async client for our async methods.
        self.async_client = openai.AsyncOpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
        )

        self.llm = OpenAI_LLM(
            model_name=config.openai_model,
            provider=self.provider
        )

        self.mcp_server = MCPServerStdio(**config.mcp_config["mcpServers"]["context7"])
        self.agent = Agent(model=self.llm, mcp_servers=[self.mcp_server])
        self.history = History()

    def detect_intent(self, message: str) -> str:
        """Detect if the message intends a search or command."""
        if "/search" in message or any(
            keyword in message.lower()
            for keyword in ["tell me about", "find docs on", "search for"]
        ):
            return "search"
        elif message.startswith("/"):
            return "command"
        return "chat"

    def query_mcp(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Query the Context7 MCP server for documents. (Mocked for demo; integrate real MCP calls.)"""
        mock_results = [
            {
                "id": 1,
                "title": f"Doc on {query}",
                "content": "Sample content...",
                "tags": ["ai"],
                "date": "2025-07-13",
            },
            {
                "id": 2,
                "title": f"Related to {query}",
                "content": "More info...",
                "tags": ["ethics"],
                "date": "2025-07-12",
            },
        ]
        self.history.add_search(query, mock_results)
        return mock_results

    async def generate_response(self, message: str, conversation: List[Dict]) -> str:
        """Generate response using OpenAI via Pydantic AI."""
        intent = self.detect_intent(message)
        if intent == "search":
            search_query = (
                message.split("about")[-1].strip() if "about" in message else message
            )
            results = self.query_mcp(search_query)
            summary = f"Found {len(results)} docs: " + ", ".join(
                r["title"] for r in results
            )
            prompt = f"Summarize these search results for the user: {summary}"
            # Access the synchronous client via the stored provider instance.
            response = self.provider.client.chat.completions.create(
                model=config.openai_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content + "\nUse /preview <id> to view."
        elif intent == "command":
            return self.handle_command(message)
        else:
            # Use our dedicated async client for the chat functionality.
            raw_msgs = conversation + [{"role": "user", "content": message}]
            response = await self.async_client.chat.completions.create(
                model=config.openai_model,
                messages=raw_msgs,
            )
            return response.choices[0].message.content

    def handle_command(self, command: str) -> str:
        """Handle hotkey commands."""
        if command == "/help":
            return "Commands: /search <query>, /preview <id>, /bookmark <id>, /theme <name>, /analytics, /exit"
        # Add more handlers...
        return "Unknown command."

    def preview_document(self, doc_id: int) -> str:
        """Syntax-highlighted preview (simple text for now)."""
        docs = (
            self.history.get_searches()[-1]["results"]
            if self.history.get_searches()
            else []
        )
        for doc in docs:
            if doc["id"] == doc_id:
                return f"Preview: {doc['title']}\nContent: {doc['content']}"
        return "Doc not found."

    # Note: MCPServerStdio lifecycle is managed by agent.run_mcp_servers(); no manual cleanup needed.

```

# src/cli.py
```py
# File: src/cli.py
"""
CLI module for the Context7 Agent.

Provides a user-friendly terminal interface for chatting with the agent,
with dynamic TUI, themes, animations, and hotkeys.
"""

import os
import sys
import time
import anyio
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
            # Use a lambda to pass keyword arguments to the function running in the thread.
            user_input = await anyio.to_thread.run_sync(lambda: Prompt.ask("[bold]You[/bold]", console=console))
            if user_input == "/exit":
                await anyio.to_thread.run_sync(self.agent.history.save)
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
                preview = await anyio.to_thread.run_sync(self.agent.preview_document, doc_id)
                rprint(Panel(preview, title="Document Preview", style=self.styles["preview"]))
                continue
            # Add more hotkeys...

            self.agent.history.add_message("user", user_input)
            await anyio.to_thread.run_sync(self.particle_loader)

            response = await self.agent.generate_response(user_input, self.conversation)
            self.agent.history.add_message("assistant", response)
            self.conversation = self.agent.history.get_conversation()

            rprint(Panel(response, title="Agent", style=self.styles["response"]))
            await anyio.to_thread.run_sync(self.typing_animation, "")

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
    anyio.run(CLI().run)

```

# src/config.py
```py
# File: src/config.py
"""
Configuration module for the Context7 Agent.

Handles environment variables and validation.
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.mcp_config = {
            "mcpServers": {
                "context7": {
                    "command": "npx",
                    "args": ["-y", "@upstash/context7-mcp@latest"]
                }
            }
        }

    def validate(self) -> str:
        if not self.openai_api_key:
            return "OPENAI_API_KEY is required."
        return None

config = Config()

```

# src/history.py
```py
# File: src/history.py
"""
History module for the Context7 Agent.

Handles JSON-based persistence for conversations, searches, bookmarks, and sessions.
"""

import json
import os
from typing import List, Dict

HISTORY_FILE = "history.json"

class History:
    def __init__(self):
        self.data = {"conversations": [], "searches": [], "bookmarks": [], "sessions": {}}
        self.load()

    def load(self):
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r") as f:
                    self.data = json.load(f)
            except json.JSONDecodeError:
                # Corrupted or empty JSON: reset history
                self.data = {
                    "conversations": [],
                    "searches": [],
                    "bookmarks": [],
                    "sessions": {}
                }
                self.save()

    def save(self):
        with open(HISTORY_FILE, "w") as f:
            json.dump(self.data, f, indent=4)

    def add_message(self, role: str, content: str):
        self.data["conversations"].append({"role": role, "content": content})
        self.save()

    def get_conversation(self) -> List[Dict]:
        return self.data["conversations"]

    def add_search(self, query: str, results: List[Dict]):
        self.data["searches"].append({"query": query, "results": results})
        self.save()

    def get_searches(self) -> List[Dict]:
        return self.data["searches"]

    def add_bookmark(self, doc: Dict):
        self.data["bookmarks"].append(doc)
        self.save()

    # Add more methods for sessions, analytics...

```

# src/utils.py
```py
# File: src/utils.py
"""
Utility functions for the Context7 Agent.
"""

def fuzzy_match(query: str, text: str) -> bool:
    """Simple fuzzy matching."""
    return query.lower() in text.lower()

# Add more utils...

```

# src/themes.py
```py
# File: src/themes.py
"""
Themes module for the Context7 Agent.

Defines themes, styles, and ASCII art.
"""

THEMES = ["cyberpunk", "ocean", "forest", "sunset"]

ASCII_ART = {
    "cyberpunk": """
   _____           _            _   ___      
  / ____|         | |          | | |__ \\     
 | |    |__   ___ | |_ ___  ___| |_   ) |    
 | |    / _ \\ / _ \\| __/ _ \\/ __| __| / /     
 | |___| (_) | (_) | ||  __/\\__ \\ |_ / /_     
  \\_____\\___/ \\___/ \\__\\___||___/\\__|____|    
    """,
    # Add more ASCII art...
}

def get_theme_styles(theme: str) -> dict:
    if theme == "cyberpunk":
        return {
            "panel": "bold magenta on black",
            "response": "cyan on black",
            "preview": "green on black",
            "header": "bold yellow on black",
            "footer": "dim white on black",
            "loader": "blink magenta"
        }
    # Add styles for other themes...
    return {}

```

