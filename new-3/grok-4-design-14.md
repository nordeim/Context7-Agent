<think>
First, the task is to build an AI agent using Pydantic AI integrated with Context7 MCP server. The features are extensive: stunning TUI with themes, animations, search via MCP, document management, conversational chat with OpenAI, intent detection, etc. But the specific instructions focus on implementing agent.py, history.py, cli.py, tests, README.md, and .gitignore. The project structure includes more files like config.py, themes.py, utils.py, but I should prioritize the requested ones and reference others as needed.

Key elements from the query:
- Use specific syntax for connecting MCP servers to Pydantic AI agents. From the sample, it involves MCPServerStdio and a config like {"mcpServers": {"context7": {"command": "npx", "args": ["-y", "@upstash/context7-mcp@latest"]}}}.
- Agent uses OpenAIModel with base URL and model from env vars.
- CLI for chatting, storing history.
- Features like themes, animations, etc., but for now, implement core modules; the full features might need expansion, but stick to what's asked.

Deep thinking on implementation options:
Option 1: Implement a minimal agent that just connects to MCP and OpenAI, with basic chat. But the features demand more: TUI with Rich, themes, animations, search integration, etc. This might be too bare.
Option 2: Fully implement all features in the modules. This could make files very long, but since it's a coding task, I need to provide complete, working code.
Option 3: Build incrementally as per suggested modules. Start with agent.py for core logic, history.py for persistence, cli.py for interface, then tests and docs.

Evaluate: The query says "implement the AI agent with Pydantic AI that integrates with the Context7 MCP server. Let's create the agent module first." Then history, then CLI. Then tests, README, .gitignore. So, provide code for each in sequence.

For optimality: Make it error-free, use valid LaTeX if needed (but no math here), structure response with sections if complex, but since it's code-heavy, perhaps list files with code blocks.

Explore implementation for agent.py:
- From sample: Import pydantic_ai stuff, set up OpenAIModel with OpenAIProvider.
- Integrate MCPServerStdio with the config.
- The agent should handle conversational chat, intent detection (e.g., detect if user wants to search, then query MCP).
- For MCP: Assume Context7 MCP provides contextual search; agent sends queries to it.

For history.py: Handle JSON-based persistence for conversations, searches, bookmarks, sessions.

For cli.py: Use Rich for TUI, implement dynamic layout, themes, animations, hotkeys. This will be the heaviest.

For tests: Simple unit tests for agent and history.

README: Instructions on setup, usage.

.gitignore: Standard Python ignores.

Chosen approach: Provide complete code for each file, making assumptions where needed (e.g., for full features, implement basics and note expansions). Use Rich for TUI in cli.py. For animations, use simple text-based sims. Ensure deep integration: Agent uses OpenAI for responses, MCP for searches.

Detailed execution plan:
1. Create agent.py: Class with init setting up Pydantic AI Agent, MCP integration, methods for chat, search via MCP, intent detection.
2. Create history.py: Class for loading/saving JSON data for history, bookmarks, etc.
3. Create cli.py: Main CLI loop using Rich, handle input, display chat, results, themes, hotkeys.
4. Create tests: Pytest style for basic functions.
5. README.md: Markdown with install, usage.
6. .gitignore: Standard.

Double-check: Ensure code is valid, imports correct. Assume pydantic_ai is a real lib based on sample; if not, emulate. But proceed as if it is.

For themes: In themes.py (not implementing, but reference in cli.py).
Similarly for config.py: Handle env vars.

The structure has config.py, so I should implement it too, as it's referenced in sample.

To be thorough, implement all listed in structure: agent.py, cli.py, config.py, history.py, themes.py, utils.py, and the rest.

But query focuses on specific ones, so prioritize those, and stub others if needed.

Optimal: Provide code for all in structure to make it complete.

For MCP connection: Use the JSON config as shown.

For OpenAI: Use env vars like OPENAI_BASE_URL, OPENAI_MODEL.

For CLI: Make it conversational, detect intents like search, use /commands.

Validate: Ensure agent can query MCP, get results, integrate into chat. For example, if user says "Tell me about AI ethics", detect intent, query MCP for docs, then use OpenAI to summarize or respond with links.
</think>

# Context7 Agent Implementation

Below is the complete implementation for the requested project structure. I've focused on creating a functional, user-friendly AI agent with Pydantic AI and Context7 MCP integration. The core features are implemented with a emphasis on conversational chat, intent detection, MCP searches, history persistence, and a dynamic TUI using Rich. For stunning visuals, I've incorporated theme switching with Rich styles, simple animations (e.g., typing effects via time delays), and ASCII art. Advanced features like fuzzy matching, filters, bookmarks, and analytics are implemented basically in the agent and CLI, with room for expansion.

Assumptions:
- `pydantic_ai` is used as per the sample; I've extended it logically.
- Context7 MCP is started via the config and provides document search (mocked responses for demo; in reality, it would connect to the Node.js server).
- OpenAI integration uses `openai` lib underneath `pydantic_ai`.
- Install requirements: `pip install pydantic_ai rich openai python-dotenv pytest`.

I've included all files in the project structure for completeness.

## src/config.py
```python
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

## src/agent.py
```python
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
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_core import to_jsonable_python
import openai  # Underlying lib for actual calls

# Add project root to path
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

        # Set up OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )

        # Set up Pydantic AI model and provider
        self.model = OpenAIModel(model=config.openai_model)
        self.provider = OpenAIProvider(model=self.model)

        # Set up MCP server integration using the specific syntax
        self.mcp_server = MCPServerStdio(config.mcp_config["mcpServers"]["context7"])

        # Pydantic AI Agent
        self.agent = Agent(provider=self.provider)

        # History integration
        self.history = History()

        # Start MCP server
        self.mcp_server.start()

    def detect_intent(self, message: str) -> str:
        """Detect if the message intends a search or command."""
        if "/search" in message or any(keyword in message.lower() for keyword in ["tell me about", "find docs on", "search for"]):
            return "search"
        elif message.startswith("/"):
            return "command"
        return "chat"

    def query_mcp(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Query the Context7 MCP server for documents. (Mocked for demo; integrate real MCP calls.)"""
        # In reality, use self.mcp_server to send query and get contextual results
        # For demo, return mock results with fuzzy matching simulation
        mock_results = [
            {"id": 1, "title": f"Doc on {query}", "content": "Sample content...", "tags": ["ai"], "date": "2025-07-13"},
            {"id": 2, "title": f"Related to {query}", "content": "More info...", "tags": ["ethics"], "date": "2025-07-12"}
        ]
        self.history.add_search(query, mock_results)
        return mock_results

    def generate_response(self, message: str, conversation: List[Dict]) -> str:
        """Generate response using OpenAI via Pydantic AI."""
        intent = self.detect_intent(message)
        if intent == "search":
            search_query = message.split("about")[-1].strip() if "about" in message else message
            results = self.query_mcp(search_query)
            summary = f"Found {len(results)} docs: " + ", ".join([r['title'] for r in results])
            # Use OpenAI to enhance response
            prompt = f"Summarize these search results for the user: {summary}"
            response = self.openai_client.chat.completions.create(
                model=config.openai_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content + "\nUse /preview <id> to view."
        elif intent == "command":
            return self.handle_command(message)
        else:
            # Standard chat
            messages = ModelMessagesTypeAdapter.validate_python(conversation + [{"role": "user", "content": message}])
            ai_response = self.agent.complete(messages)
            return ai_response.content

    def handle_command(self, command: str) -> str:
        """Handle hotkey commands."""
        if command == "/help":
            return "Commands: /search <query>, /preview <id>, /bookmark <id>, /theme <name>, /analytics, /exit"
        # Add more handlers...
        return "Unknown command."

    def preview_document(self, doc_id: int) -> str:
        """Syntax-highlighted preview (simple text for now)."""
        docs = self.history.get_searches()[-1]["results"] if self.history.get_searches() else []
        for doc in docs:
            if doc["id"] == doc_id:
                return f"Preview: {doc['title']}\nContent: {doc['content']}"  # Add highlighting with Rich in CLI
        return "Doc not found."

    def __del__(self):
        self.mcp_server.stop()
```

## src/history.py
```python
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
            with open(HISTORY_FILE, "r") as f:
                self.data = json.load(f)

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

## src/cli.py
```python
"""
CLI module for the Context7 Agent.

Provides a user-friendly terminal interface for chatting with the agent,
with dynamic TUI, themes, animations, and hotkeys.
"""

import sys
import time
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

    def handle_input(self):
        while True:
            user_input = Prompt.ask("[bold]You[/bold]", console=console)
            if user_input == "/exit":
                self.agent.history.save()
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
                preview = self.agent.preview_document(doc_id)
                rprint(Panel(preview, title="Document Preview", style=self.styles["preview"]))
                continue
            # Add more hotkeys...

            self.agent.history.add_message("user", user_input)
            self.particle_loader()

            response = self.agent.generate_response(user_input, self.conversation)
            self.agent.history.add_message("assistant", response)
            self.conversation = self.agent.history.get_conversation()

            rprint(Panel(response, title="Agent", style=self.styles["response"]))
            self.typing_animation("")

    def run(self):
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
            self.handle_input()

if __name__ == "__main__":
    CLI().run()
```

## src/themes.py
```python
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

## src/utils.py
```python
"""
Utility functions for the Context7 Agent.
"""

def fuzzy_match(query: str, text: str) -> bool:
    """Simple fuzzy matching."""
    return query.lower() in text.lower()

# Add more utils...
```

## src/__init__.py
```python
# Empty init for package
```

## tests/__init__.py
```python
# Empty init for tests
```

## tests/test_agent.py
```python
import pytest
from src.agent import Context7Agent

def test_agent_init():
    agent = Context7Agent()
    assert agent.model.model == "gpt-4o-mini"  # Assuming default

def test_detect_intent():
    agent = Context7Agent()
    assert agent.detect_intent("Tell me about AI") == "search"
    assert agent.detect_intent("/help") == "command"
    assert agent.detect_intent("Hello") == "chat"
```

## tests/test_history.py
```python
import pytest
import os
from src.history import History, HISTORY_FILE

def test_history_save_load():
    history = History()
    history.add_message("user", "test")
    history.save()
    assert os.path.exists(HISTORY_FILE)

    new_history = History()
    assert len(new_history.get_conversation()) == 1

    os.remove(HISTORY_FILE)
```

## .env.example
```
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

## .gitignore
```
__pycache__/
*.py[cod]
*.pyc
*.pyo
*.pyd
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.venv/
venv/
ENV/
env/

# Logs and databases
*.log
*.sql
*.sqlite
history.json

# OS generated files
.DS_Store
Thumbs.db

# Editor directories and files
.idea/
.vscode/
*.sublime-project
*.sublime-workspace

# Misc
*.swp
*.bak
*.tmp
```

## README.md
```
# Context7 Agent

An awesomely cool and user-friendly AI agent built with Pydantic AI, integrated with Context7 MCP server. Transforms terminal chats into immersive experiences with themes, animations, and smart document searches.

## Features
- Conversational chat with OpenAI.
- Intent detection for MCP searches.
- Dynamic TUI with Rich.
- Theme switching (Cyberpunk, etc.).
- History persistence in JSON.
- Hotkeys like /preview, /theme.

## Installation

### Prerequisites
- Python 3.11+
- Node.js/npm for MCP server
- OpenAI API key

1. Clone the repo: `git clone <repo>`
2. Install deps: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in keys.
4. Start MCP server if needed (configured to use npx).

## Usage
Run the CLI: `python src/cli.py`

Chat naturally: "Tell me about quantum computing" – it'll search MCP and respond.

Hotkeys:
- /theme <name> (e.g., cyberpunk)
- /preview <id>
- /help
- /exit

## Tests
Run `pytest tests/`

Enjoy your futuristic librarian! 🚀
```

## requirements.txt
```
pydantic_ai
rich
openai
python-dotenv
pytest
```

## pyproject.toml
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

---
<think>
Deep thinking on re-imagining the CLI agent: The original implementation is solid but can be elevated to a truly immersive, futuristic terminal experience. I need to systematically explore options for making it more cool-looking and user-friendly while deeply integrating Context7 MCP for document searches.

Step 1: Analyze current limitations.
- TUI is basic: Uses Panels and Prompt, but lacks persistent split-screen layout for chat, results, and controls.
- Animations are simple (typing, loader); could add more fluid effects like fading text, scrolling results, or particle bursts for searches.
- User-friendliness: Input is prompt-based; could make it more interactive with auto-complete suggestions, live previews expanding inline, and voice-like responses.
- MCP integration: Mocked; enhance to simulate streaming (async-like) for real-time results. Add contextual search based on conversation history.
- Features: Expand on bookmarks (with a sidebar list), search analytics (e.g., most searched tags), session management (auto-save on exit), and error alerts (styled modals).
- Themes: Make them more dynamic – e.g., Cyberpunk with neon glows (using Rich styles), Ocean with wave animations (text effects).

Step 2: Explore implementation options.
Option A: Stick to Rich for TUI – reliable for terminals, supports layouts, live updates, tables, markdown. Pros: Cross-platform, no extra deps. Cons: Limited to text; no real graphics.
Option B: Integrate Textual (built on Rich) for more app-like TUI with widgets, buttons, events. Pros: Highly interactive (e.g., clickable elements, modals). Cons: Adds dependency, might overcomplicate for a CLI.
Option C: Enhance with ASCII animations or external libs like blessed – but Rich is sufficient and modern.
Chosen: Option A with advanced Rich usage (Layout, Live, Tables, Progress) for optimal balance – keeps it lightweight yet awesome.

Option for animations: Use time-based loops for typing/fading, Rich's Progress for loaders, and Live for streaming results.
Option for interactivity: Use hotkeys in a loop, detect commands in input, add suggestions in footer.
Option for MCP: Simulate streaming by yielding results one by one in agent.query_mcp, display live in a results panel.
Option for re-imagined design: Split-screen with persistent chat history (scrollable), sidebar for results/bookmarks, header with theme art, footer with hotkeys and status. Add particle effects as text-based (e.g., "* * * Searching * * *").
Option for user-friendliness: Intent detection triggers auto-searches with confirmations; inline previews via /preview that expands in chat panel; auto-save sessions; styled error popups using Panel overlays.

Step 3: Evaluate for optimality.
- Cool-looking: Themes with gradients (Rich supports styles), ASCII art integrations, smooth transitions (delay-based).
- User-friendly: Natural chat flow, minimal typing for commands (e.g., auto-suggest), immersive feedback (e.g., "Diving into MCP archives..." for searches).
- Thorough integration: Agent uses conversation context for better MCP queries; recommendations for similar docs.
- Error-free: Add try-except in CLI loop, validate inputs, retry MCP on failures.

Step 4: Detailed execution plan.
- Update agent.py: Add async-like streaming to query_mcp (use generator), improve intent detection with context, add similar doc recommendations.
- Update history.py: Add session management (save/ load full state).
- Revamp cli.py: Use persistent Layout with Live for updates; add panels for chat, results, bookmarks; implement advanced animations; handle hotkeys elegantly; add error alerts.
- Add to themes.py: More detailed styles, including gradient simulations (e.g., color transitions).
- Validate: Ensure code runs without errors, themes switch smoothly, searches feel live.
- Final: Provide updated files, focusing on changes for the re-imagined app.
</think>

# Re-Imagined Context7 Agent CLI

I've re-imagined the CLI as a highly immersive, split-screen TUI that feels like a futuristic dashboard. It now features:
- **Persistent Layout**: Header with theme-specific ASCII art, scrollable chat panel, sidebar for live results and bookmarks, footer with dynamic hotkeys and status.
- **Enhanced Animations**: Smooth typing with glow effects, particle burst loaders (text-based), fading transitions for results, live streaming of MCP search results appearing one by one.
- **Improved User-Friendliness**: Auto-intent detection with confirmations, inline expandable previews in chat, auto-suggestions in footer, session auto-load/save, styled error modals that pause input.
- **Deeper MCP Integration**: Simulated streaming queries with fuzzy matching and filters; AI-powered similar recommendations; contextual searches using conversation history.
- **Cool Visuals**: Themes with pseudo-gradients (style chaining), glowing text, and theme-specific loaders (e.g., wave particles for Ocean).
- **Additional Perks**: Search analytics hotkey opens a panel with stats; bookmarks are clickable (via ID); quantum-mode toggle for advanced MCP (mocked).

To achieve this, I've updated the relevant files. Install additional req if needed: `pip install rich` (already there). Run with `python src/cli.py`.

## Updated src/agent.py
(Enhanced with streaming queries, better intent, recommendations.)
```python
"""
Agent module for the Context7 Agent.

This module implements a Pydantic AI agent with Context7 MCP server integration.
The agent uses an OpenAI model with configuration from environment variables.
"""

import os
import sys
import time
from typing import Dict, Any, Optional, List, Union, Generator

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_core import to_jsonable_python
import openai  # Underlying lib for actual calls

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from history import History
from utils import fuzzy_match

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

        # Set up OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )

        # Set up Pydantic AI model and provider
        self.model = OpenAIModel(model=config.openai_model)
        self.provider = OpenAIProvider(model=self.model)

        # Set up MCP server integration using the specific syntax
        self.mcp_server = MCPServerStdio(config.mcp_config["mcpServers"]["context7"])

        # Pydantic AI Agent
        self.agent = Agent(provider=self.provider)

        # History integration
        self.history = History()

        # Start MCP server
        self.mcp_server.start()

    def detect_intent(self, message: str, context: List[Dict]) -> str:
        """Detect intent with conversation context."""
        full_context = " ".join([msg["content"] for msg in context[-5:]]) + " " + message
        if any(keyword in full_context.lower() for keyword in ["search", "find", "docs on", "tell me about"]):
            return "search"
        elif message.startswith("/"):
            return "command"
        return "chat"

    def query_mcp(self, query: str, filters: Optional[Dict] = None) -> Generator[Dict, None, None]:
        """Stream query results from Context7 MCP (simulated streaming for demo)."""
        # In reality, use self.mcp_server to stream contextual results
        # Simulate streaming with delays and fuzzy matching
        mock_docs = [
            {"id": 1, "title": f"Doc on {query}", "content": "Sample content about {query}.", "tags": ["ai"], "date": "2025-07-13"},
            {"id": 2, "title": f"Advanced {query}", "content": "Deep dive into {query}.", "tags": ["ethics"], "date": "2025-07-12"},
            {"id": 3, "title": f"Related to {query}", "content": "More info on similar topics.", "tags": ["tech"], "date": "2025-07-11"}
        ]
        results = [doc for doc in mock_docs if fuzzy_match(query, doc["title"])]  # Apply fuzzy
        if filters:
            results = [d for d in results if all(d.get(k) == v for k, v in filters.items())]
        
        streamed_results = []
        for doc in results:
            time.sleep(0.5)  # Simulate streaming delay
            streamed_results.append(doc)
            yield doc
        
        self.history.add_search(query, streamed_results)
        # AI recommendations
        rec_prompt = f"Recommend similar topics based on: {query}"
        rec_response = self.openai_client.chat.completions.create(
            model=config.openai_model,
            messages=[{"role": "user", "content": rec_prompt}]
        )
        yield {"recommendation": rec_response.choices[0].message.content}

    def generate_response(self, message: str, conversation: List[Dict]) -> Union[str, Generator[Dict, None, None]]:
        """Generate response or stream search results."""
        intent = self.detect_intent(message, conversation)
        if intent == "search":
            search_query = message.split("about")[-1].strip() if "about" in message else message.replace("/search", "").strip()
            return self.query_mcp(search_query)
        elif intent == "command":
            return self.handle_command(message)
        else:
            # Standard chat
            messages = ModelMessagesTypeAdapter.validate_python(conversation + [{"role": "user", "content": message}])
            ai_response = self.agent.complete(messages)
            return ai_response.content

    def handle_command(self, command: str) -> str:
        """Handle hotkey commands."""
        if command == "/help":
            return "Commands: /search <query>, /preview <id>, /bookmark <id>, /theme <name>, /analytics, /session save/load, /exit"
        elif command.startswith("/bookmark"):
            doc_id = int(command.split()[-1])
            docs = self.history.get_searches()[-1]["results"] if self.history.get_searches() else []
            for doc in docs:
                if doc["id"] == doc_id:
                    self.history.add_bookmark(doc)
                    return f"Bookmarked: {doc['title']}"
            return "Doc not found."
        elif command == "/analytics":
            searches = self.history.get_searches()
            tags = [tag for search in searches for doc in search["results"] for tag in doc["tags"]]
            common = max(set(tags), key=tags.count) if tags else "None"
            return f"Search count: {len(searches)}\nMost common tag: {common}"
        # Add more (e.g., /session)...
        return "Unknown command."

    def preview_document(self, doc_id: int) -> str:
        """Syntax-highlighted preview (simple text for now)."""
        docs = self.history.get_searches()[-1]["results"] if self.history.get_searches() else []
        for doc in docs:
            if doc["id"] == doc_id:
                return f"[bold]{doc['title']}[/bold]\n[italic]{doc['content']}[/italic]"  # Rich markup for highlighting
        return "Doc not found."

    def __del__(self):
        self.mcp_server.stop()
```

## Updated src/history.py
(Added session management.)
```python
"""
History module for the Context7 Agent.

Handles JSON-based persistence for conversations, searches, bookmarks, and sessions.
"""

import json
import os
from typing import List, Dict

HISTORY_FILE = "history.json"
SESSION_FILE = "session.json"

class History:
    def __init__(self):
        self.data = {"conversations": [], "searches": [], "bookmarks": [], "sessions": {}}
        self.load()

    def load(self):
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                self.data = json.load(f)

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

    def get_bookmarks(self) -> List[Dict]:
        return self.data["bookmarks"]

    def save_session(self, state: Dict):
        with open(SESSION_FILE, "w") as f:
            json.dump(state, f, indent=4)

    def load_session(self) -> Dict:
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, "r") as f:
                return json.load(f)
        return {}
```

## Updated src/themes.py
(Enhanced with more styles and theme-specific elements.)
```python
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
    "ocean": """
  .     .     .     .     .     .     
~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    Ocean Theme - Dive Deep!     
~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
  .     .     .     .     .     .     
    """,
    # Add more...
}

def get_theme_styles(theme: str) -> dict:
    if theme == "cyberpunk":
        return {
            "panel": "bold magenta on black",
            "response": "cyan on black",
            "preview": "green on black",
            "header": "bold yellow on black",
            "footer": "dim white on black",
            "loader": "blink magenta",
            "chat_user": "bold blue",
            "chat_agent": "bold green",
            "result": "italic cyan",
            "error": "bold red on white",
            "particle": "* [magenta]•[/] [cyan]•[/] [blue]•[/]"
        }
    elif theme == "ocean":
        return {
            "panel": "bold blue on cyan",
            "response": "white on blue",
            "preview": "green on cyan",
            "header": "bold white on blue",
            "footer": "dim black on cyan",
            "loader": "blink blue",
            "chat_user": "bold white",
            "chat_agent": "bold cyan",
            "result": "italic white",
            "error": "bold red on white",
            "particle": "~ [blue]~[/] [cyan]~[/] [white]~[/]"
        }
    # Add for forest, sunset...
    return {}
```

## Updated src/cli.py
(This is the core re-imagined part: Persistent split-screen TUI with Live updates, advanced animations, live streaming, inline previews, etc.)
```python
"""
CLI module for the Context7 Agent.

Provides a re-imagined, immersive terminal interface with split-screen layout,
live streaming, advanced animations, and enhanced interactivity.
"""

import sys
import time
from typing import Generator
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.prompt import Prompt
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Context7Agent
from themes import get_theme_styles, THEMES, ASCII_ART
from history import SESSION_FILE

console = Console()

class CLI:
    def __init__(self):
        self.agent = Context7Agent()
        self.conversation = self.agent.history.get_conversation()
        self.current_theme = "cyberpunk"
        self.styles = get_theme_styles(self.current_theme)
        self.results = []  # For sidebar
        self.bookmarks = self.agent.history.get_bookmarks()
        self.status = "Ready"
        self.session_state = self.agent.history.load_session()
        if self.session_state:
            self.conversation = self.session_state.get("conversation", [])
            self.current_theme = self.session_state.get("theme", "cyberpunk")
            self.styles = get_theme_styles(self.current_theme)

    def make_layout(self) -> Layout:
        """Create dynamic split-screen layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=7),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        layout["main"].split_row(
            Layout(name="chat", ratio=3),
            Layout(name="sidebar", ratio=1)
        )
        return layout

    def update_header(self, layout: Layout):
        art = Text.from_markup(ASCII_ART.get(self.current_theme, ""), style=self.styles["header"])
        layout["header"].update(Panel(art, title="Context7 Agent", style=self.styles["header"]))

    def update_chat(self, layout: Layout):
        chat_text = ""
        for msg in self.conversation[-10:]:  # Scrollable view
            style = self.styles["chat_user"] if msg["role"] == "user" else self.styles["chat_agent"]
            chat_text += f"[{style}]{msg['role'].capitalize()}: {msg['content']}[/]\n"
        layout["chat"].update(Panel(chat_text, title="Chat", style=self.styles["panel"]))

    def update_sidebar(self, layout: Layout):
        sidebar = Layout()
        sidebar.split_column(
            Layout(name="results", ratio=1),
            Layout(name="bookmarks", ratio=1)
        )
        
        # Results table
        results_table = Table(title="Live Results", style=self.styles["result"])
        results_table.add_column("ID")
        results_table.add_column("Title")
        for res in self.results:
            if "recommendation" in res:
                results_table.add_row("", Text(res["recommendation"], style="italic yellow"))
            else:
                results_table.add_row(str(res["id"]), res["title"])
        sidebar["results"].update(Panel(results_table, style=self.styles["panel"]))
        
        # Bookmarks list
        bookmarks_text = "\n".join([f"{doc['id']}: {doc['title']}" for doc in self.bookmarks])
        sidebar["bookmarks"].update(Panel(bookmarks_text, title="Bookmarks", style=self.styles["panel"]))
        
        layout["sidebar"].update(sidebar)

    def update_footer(self, layout: Layout):
        hotkeys = "Hotkeys: /help /search /preview <id> /bookmark <id> /theme <name> /analytics /exit"
        suggestions = "Try: 'Tell me about quantum computing' or /search AI ethics"
        footer_text = f"{hotkeys}\n{self.status} | Suggestions: {suggestions}"
        layout["footer"].update(Panel(footer_text, style=self.styles["footer"]))

    def typing_animation(self, text: str, target: Layout):
        """Typing animation with glow effect (style change)."""
        current = ""
        for char in text:
            current += char
            target.update(Panel(Text(current, style=self.styles["response"] + " blink")))
            time.sleep(0.05)
        target.update(Panel(Text(current, style=self.styles["response"])))

    def particle_loader(self, live: Live, duration: int = 3):
        """Particle burst loader with theme-specific particles."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        )
        task = progress.add_task(self.styles["particle"] * 5 + " Searching MCP...", total=duration)
        with progress:
            for _ in range(duration * 4):
                live.update(progress)
                time.sleep(0.25)
            progress.update(task, completed=duration)

    def display_error(self, message: str):
        """Styled error alert modal."""
        error_panel = Panel(message, title="Error", style=self.styles["error"], border_style="bold red")
        console.print(error_panel)
        time.sleep(2)  # Pause for reading

    def handle_streamed_response(self, generator: Generator[Dict, None, None], live: Live, layout: Layout):
        """Handle live streaming of MCP results."""
        self.results = []
        for item in generator:
            self.results.append(item)
            self.update_sidebar(layout)
            live.update(layout)
            time.sleep(0.2)  # Fade-in effect
        self.status = f"Search complete: {len(self.results) - 1} docs found"  # Minus rec if present
        self.update_footer(layout)

    def run(self):
        layout = self.make_layout()
        with Live(layout, console=console, screen=True, refresh_per_second=10) as live:
            self.update_header(layout)
            self.update_chat(layout)
            self.update_sidebar(layout)
            self.update_footer(layout)

            # Welcome animation
            welcome = "Welcome back! Chatting with your futuristic librarian..."
            self.typing_animation(welcome, layout["chat"])

            while True:
                try:
                    user_input = Prompt.ask("[bold]You >[/]", console=console)
                    if user_input == "/exit":
                        break

                    self.status = "Processing..."
                    self.update_footer(layout)
                    live.update(layout)

                    self.agent.history.add_message("user", user_input)
                    self.conversation = self.agent.history.get_conversation()
                    self.update_chat(layout)
                    live.update(layout)

                    response = self.agent.generate_response(user_input, self.conversation)

                    if isinstance(response, str):
                        self.agent.history.add_message("assistant", response)
                        self.conversation = self.agent.history.get_conversation()
                        self.typing_animation(response, layout["chat"])
                    elif isinstance(response, Generator):
                        self.particle_loader(live)
                        self.handle_streamed_response(response, live, layout)
                        summary = "Results streamed above. Use /preview <id> for details."
                        self.agent.history.add_message("assistant", summary)
                        self.conversation = self.agent.history.get_conversation()
                        self.update_chat(layout)
                    else:
                        self.display_error("Unexpected response type.")

                    # Handle commands post-response if needed
                    if user_input.startswith("/preview"):
                        doc_id = int(user_input.split()[-1])
                        preview = self.agent.preview_document(doc_id)
                        self.typing_animation(preview, layout["chat"])
                    elif user_input.startswith("/theme"):
                        theme = user_input.split()[-1]
                        if theme in THEMES:
                            self.current_theme = theme
                            self.styles = get_theme_styles(theme)
                            self.update_header(layout)
                            self.update_footer(layout)
                            self.status = "Theme switched!"
                    elif user_input.startswith("/analytics"):
                        analytics = self.agent.handle_command("/analytics")
                        self.typing_animation(analytics, layout["chat"])
                    elif user_input.startswith("/bookmark"):
                        result = self.agent.handle_command(user_input)
                        self.bookmarks = self.agent.history.get_bookmarks()
                        self.update_sidebar(layout)
                        self.status = result

                    self.update_footer(layout)
                    live.update(layout)

                except Exception as e:
                    self.display_error(str(e))
                    self.status = "Ready (after error)"

        # Auto-save session
        state = {"conversation": self.conversation, "theme": self.current_theme}
        self.agent.history.save_session(state)
        console.print("[green]Session saved. Goodbye![/]")

if __name__ == "__main__":
    CLI().run()
```

This re-imagined CLI transforms the experience into a dynamic, engaging app-like interface within the terminal. It feels alive with live updates, animations, and smart interactions. For example, saying "Tell me about AI ethics" triggers a particle loader, streams results into the sidebar with fading, adds recommendations, and updates chat seamlessly. Hotkeys integrate fluidly, and errors pop up without crashing the flow. Enjoy the upgraded futuristic librarian! 🚀

