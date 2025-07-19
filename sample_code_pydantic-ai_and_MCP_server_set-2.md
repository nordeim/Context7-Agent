# .env.example
```example
# =============================================================================
# CONTEXT7 TERMINAL AI - ENVIRONMENT CONFIGURATION
# Copy this file to .env and fill in your actual values
# =============================================================================

# 🎯 REQUIRED: OpenAI Configuration
CONTEXT7_OPENAI_API_KEY=your-openai-api-key-here
CONTEXT7_OPENAI_BASE_URL=https://api.openai.com/v1
CONTEXT7_OPENAI_MODEL=gpt-4o-mini

# 🔧 OPTIONAL: Model Configuration
CONTEXT7_MAX_TOKENS=4000
CONTEXT7_TEMPERATURE=0.7
CONTEXT7_TOP_P=1.0
CONTEXT7_FREQUENCY_PENALTY=0.0
CONTEXT7_PRESENCE_PENALTY=0.0

# 🗄️ Storage Configuration
CONTEXT7_HISTORY_PATH=~/.context7/history.json
CONTEXT7_MCP_CONFIG_PATH=~/.context7/mcp.json
CONTEXT7_MAX_HISTORY=1000
CONTEXT7_CACHE_TTL=3600

# 🌐 Network Configuration
CONTEXT7_REQUEST_TIMEOUT=30
CONTEXT7_MAX_RETRIES=3
CONTEXT7_BACKOFF_FACTOR=0.3
CONTEXT7_MAX_WORKERS=4

# 🎨 Theme Configuration
CONTEXT7_DEFAULT_THEME=cyberpunk
CONTEXT7_ENABLE_ANIMATIONS=true
CONTEXT7_ANIMATION_SPEED=50
CONTEXT7_COLOR_SUPPORT=true

# 📊 Logging Configuration
CONTEXT7_LOG_LEVEL=INFO
CONTEXT7_LOG_FILE=~/.context7/logs/context7.log
CONTEXT7_LOG_MAX_SIZE=10MB
CONTEXT7_LOG_BACKUP_COUNT=5

# 🐳 Docker Configuration
DOCKER_BUILDKIT=1
COMPOSE_DOCKER_CLI_BUILD=1

# 🔧 Development Configuration
CONTEXT7_DEBUG=false
CONTEXT7_DEV_MODE=false
CONTEXT7_AUTO_RELOAD=false
CONTEXT7_PROFILING=false

# 🧪 Testing Configuration
PYTEST_ASYNCIO_THREAD_COUNT=8
PYTEST_MAX_WORKERS=4

# 🌐 Proxy Configuration (if needed)
HTTP_PROXY=
HTTPS_PROXY=
NO_PROXY=localhost,127.0.0.1,.local

# 📱 Notification Configuration
CONTEXT7_ENABLE_NOTIFICATIONS=false
CONTEXT7_WEBHOOK_URL=
CONTEXT7_SLACK_WEBHOOK=

```

# src/agent.py
```py
"""
Production-ready AI agent with Pydantic AI and MCP integration.
Fixed import paths and message class names for v0.5+ compatibility.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
import logging

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import config
from .history import HistoryManager

logger = logging.getLogger(__name__)

class Context7Agent:
    """Production-ready AI agent with native Pydantic AI integration."""
    
    def __init__(self):
        """Initialize with correct Pydantic AI v0.5+ patterns."""
        self.provider = OpenAIProvider(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )
        
        self.model = OpenAIModel(
            model_name=config.openai_model,
            provider=self.provider
        )
        
        self.mcp_server = MCPServerStdio(
            command="npx",
            args=["-y", "@upstash/context7-mcp@latest"]
        )
        
        self.agent = Agent(
            model=self.model,
            mcp_servers=[self.mcp_server],
            system_prompt="""You are Context7, an advanced AI assistant with access to a vast knowledge base.
            You can search, analyze, and present documents in real-time. Be helpful, concise, and engaging.
            When users ask about topics, use the search tools to find relevant documents and provide summaries."""
        )
        
        self.history = HistoryManager()
    
    async def initialize(self):
        """Initialize the agent and load history."""
        await self.history.load()
    
    async def chat_stream(
        self, 
        message: str, 
        conversation_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat responses using correct Pydantic AI patterns."""
        
        try:
            async with self.agent.run_mcp_servers():
                # Build conversation history
                history = self.history.get_messages(conversation_id or "default")
                
                # Create proper message structure for Pydantic AI
                messages = []
                for msg in history[-6:]:  # Last 6 messages for context
                    if msg["role"] == "user":
                        messages.append({"role": "user", "content": msg["content"]})
                    elif msg["role"] == "assistant":
                        messages.append({"role": "assistant", "content": msg["content"]})
                
                # Use Pydantic AI's native run method
                result = await self.agent.run(
                    message,
                    message_history=messages
                )
                
                # Stream the response
                content = str(result.data)
                
                yield {
                    "type": "content",
                    "data": content,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Save to history
                await self.history.save_message(
                    conversation_id or "default",
                    message,
                    content
                )
                
                yield {
                    "type": "complete",
                    "data": content,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Agent error: {e}")
            yield {
                "type": "error",
                "data": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def search_documents(
        self, 
        query: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search documents using MCP tools with proper error handling."""
        
        try:
            async with self.agent.run_mcp_servers():
                search_prompt = f"""
                Search the Context7 knowledge base for documents about: {query}
                Return up to {limit} most relevant results with titles, sources, and brief summaries.
                """
                
                result = await self.agent.run(search_prompt)
                
                # Parse and structure the results
                response_text = str(result.data)
                
                # Attempt to parse as JSON if possible
                try:
                    if response_text.strip().startswith('[') or response_text.strip().startswith('{'):
                        parsed = json.loads(response_text)
                        if isinstance(parsed, list):
                            return parsed
                        else:
                            return [parsed]
                except json.JSONDecodeError:
                    pass
                
                # Fallback to structured response
                return [{
                    "title": "Search Results",
                    "content": response_text,
                    "source": "Context7 MCP",
                    "score": 1.0
                }]
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            return [{
                "title": "Search Error",
                "content": f"Unable to search documents: {str(e)}",
                "source": "Error",
                "score": 0.0
            }]
    
    def get_conversations(self) -> List[Dict[str, Any]]:
        """Get all conversation histories."""
        return self.history.get_conversations()
    
    async def clear_history(self, conversation_id: Optional[str] = None):
        """Clear conversation history."""
        await self.history.clear(conversation_id)

```

# src/cli.py
```py
"""
Rich TUI interface for Context7 Agent with robust async handling.
"""

import asyncio
import sys
from typing import Optional, List, Dict, Any
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.align import Align
import anyio

from .agent import Context7Agent
from .themes import ThemeManager
from .config import config

class Context7CLI:
    """Production-ready TUI interface for Context7 Agent."""
    
    def __init__(self):
        self.console = Console()
        self.agent = Context7Agent()
        self.theme_manager = ThemeManager()
        self.current_conversation = "default"
        self.running = True
    
    async def initialize(self):
        """Initialize the CLI interface."""
        await self.agent.initialize()
        self.theme_manager.get_current_theme().print_banner()
    
    async def display_welcome(self):
        """Display welcome screen with theme selection."""
        theme = self.theme_manager.get_current_theme()
        theme.print_banner()
        
        # Theme selection
        self.console.print("\n[bold]Available Themes:[/bold]")
        for t in self.theme_manager.list_themes():
            marker = "✓" if t == self.theme_manager.current_theme else "○"
            self.console.print(f"  {marker} {t}")
        
        choice = await anyio.to_thread.run_sync(
            lambda: Prompt.ask(
                "\nSelect theme (or press Enter for default)",
                choices=self.theme_manager.list_themes(),
                default=self.theme_manager.current_theme
            )
        )
        self.theme_manager.set_theme(choice)
    
    async def chat_loop(self):
        """Main chat interaction loop with robust error handling."""
        self.console.print("\n[bold green]Context7 AI is ready! Type your questions below.[/bold green]")
        self.console.print("[dim]Commands: /theme, /history, /clear, /exit[/dim]\n")
        
        while self.running:
            try:
                # Get user input with proper async handling
                user_input = await anyio.to_thread.run_sync(
                    lambda: Prompt.ask("[bold cyan]You[/bold cyan]")
                )
                
                if not user_input.strip():
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    await self.handle_command(user_input)
                    continue
                
                # Process message
                await self.process_message(user_input)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Shutting down gracefully...[/yellow]")
                self.running = False
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                self.console.print("[dim]Please try again or use /exit to quit.[/dim]")
    
    async def handle_command(self, command: str):
        """Handle special commands with async support."""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == "/exit":
            self.running = False
            self.console.print("[yellow]Goodbye! 👋[/yellow]")
            
        elif cmd == "/theme":
            if len(parts) > 1:
                if self.theme_manager.set_theme(parts[1]):
                    self.console.print(f"[green]Theme changed to {parts[1]}[/green]")
                    self.theme_manager.get_current_theme().print_banner()
                else:
                    self.console.print("[red]Invalid theme[/red]")
            else:
                themes = ", ".join(self.theme_manager.list_themes())
                self.console.print(f"[cyan]Available themes: {themes}[/cyan]")
                
        elif cmd == "/history":
            conversations = self.agent.get_conversations()
            if conversations:
                table = Table(title="Conversation History")
                table.add_column("ID")
                table.add_column("Last Message")
                table.add_column("Count")
                
                for conv in conversations[:10]:
                    table.add_row(
                        conv["id"],
                        conv["last_message"][:19],
                        str(conv["message_count"])
                    )
                
                self.console.print(table)
            else:
                self.console.print("[yellow]No conversation history[/yellow]")
                
        elif cmd == "/clear":
            await self.agent.clear_history()
            self.console.print("[green]History cleared[/green]")
            
        else:
            self.console.print(f"[red]Unknown command: {cmd}[/red]")
    
    async def process_message(self, message: str):
        """Process a user message with streaming and robust error handling."""
        theme = self.theme_manager.get_current_theme()
        
        # Create progress indicator
        with theme.create_progress_spinner() as progress:
            task = progress.add_task("Thinking...", total=None)
            
            # Process the message
            response_parts = []
            async for chunk in self.agent.chat_stream(message, self.current_conversation):
                if chunk["type"] == "content":
                    response_parts.append(chunk["data"])
                    progress.update(task, description=f"Processing... ({len(response_parts)} chars)")
                
                elif chunk["type"] == "complete":
                    progress.update(task, description="Complete!")
                
                elif chunk["type"] == "error":
                    self.console.print(f"[red]Error: {chunk['data']}[/red]")
                    return
            
            # Display response
            full_response = "".join(response_parts)
            if full_response.strip():
                theme.console.print("\n[bold green]Assistant:[/bold green]")
                theme.console.print(Panel(full_response, border_style=theme.colors['secondary']))
            else:
                theme.console.print("[yellow]No response generated.[/yellow]")
    
    async def run(self):
        """Run the CLI application with proper async handling."""
        try:
            await self.initialize()
            await self.display_welcome()
            await self.chat_loop()
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Goodbye! Thanks for using Context7 AI.[/yellow]")
        except Exception as e:
            self.console.print(f"[red]Fatal error: {e}[/red]")
            sys.exit(1)

async def main():
    """Main entry point with proper async handling."""
    cli = Context7CLI()
    await cli.run()

if __name__ == "__main__":
    anyio.run(main)

```

# src/config.py
```py
"""
Configuration management for Context7 Agent.
Handles environment variables and MCP server configuration.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import json

class Config(BaseSettings):
    """Application configuration with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CONTEXT7_",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Required fields
    openai_api_key: str = Field(..., description="OpenAI API key")
    
    # Optional fields with defaults
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL"
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model to use"
    )
    mcp_config_path: Path = Field(
        default_factory=lambda: Path.home() / ".context7" / "mcp.json",
        description="Path to MCP configuration"
    )
    history_path: Path = Field(
        default_factory=lambda: Path.home() / ".context7" / "history.json",
        description="Path to conversation history"
    )
    max_history: int = Field(
        default=1000,
        description="Maximum conversation history entries"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        self.mcp_config_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
    
    def get_mcp_config(self) -> Dict[str, Any]:
        """Get MCP server configuration."""
        config = {
            "mcpServers": {
                "context7": {
                    "command": "npx",
                    "args": ["-y", "@upstash/context7-mcp@latest"],
                    "env": {}
                }
            }
        }
        
        # Save config if it doesn't exist
        if not self.mcp_config_path.exists():
            with open(self.mcp_config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        return config

# Global config instance
config = Config()

```

# src/history.py
```py
"""
Conversation history management with JSON persistence.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import aiofiles

from .config import config

class HistoryManager:
    """Manages conversation history with JSON persistence."""
    
    def __init__(self):
        self.history_path = config.history_path
        self.max_history = config.max_history
        self._history: Dict[str, List[Dict[str, Any]]] = {}
    
    async def load(self):
        """Load conversation history from disk."""
        try:
            if self.history_path.exists():
                async with aiofiles.open(self.history_path, 'r') as f:
                    content = await f.read()
                    self._history = json.loads(content)
        except Exception as e:
            print(f"Warning: Could not load history: {e}")
            self._history = {}
    
    async def save(self):
        """Save conversation history to disk."""
        try:
            async with aiofiles.open(self.history_path, 'w') as f:
                await f.write(json.dumps(self._history, indent=2))
        except Exception as e:
            print(f"Warning: Could not save history: {e}")
    
    async def save_message(
        self, 
        conversation_id: str, 
        user_message: str, 
        assistant_response: str
    ):
        """Save a message exchange to history."""
        if conversation_id not in self._history:
            self._history[conversation_id] = []
        
        self._history[conversation_id].append({
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim history if too long
        if len(self._history[conversation_id]) > self.max_history:
            self._history[conversation_id] = self._history[conversation_id][-self.max_history:]
        
        await self.save()
    
    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get messages for a conversation."""
        return self._history.get(conversation_id, [])
    
    def get_conversations(self) -> List[Dict[str, Any]]:
        """Get all conversation IDs with metadata."""
        conversations = []
        for conv_id, messages in self._history.items():
            if messages:
                conversations.append({
                    "id": conv_id,
                    "last_message": messages[-1]["timestamp"],
                    "message_count": len(messages)
                })
        
        return sorted(conversations, key=lambda x: x["last_message"], reverse=True)
    
    async def clear(self, conversation_id: Optional[str] = None):
        """Clear history for a conversation or all conversations."""
        if conversation_id:
            self._history.pop(conversation_id, None)
        else:
            self._history.clear()
        
        await self.save()

```

# src/themes.py
```py
"""
Stunning theme system with animations and ASCII art.
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
from rich.prompt import Prompt
from rich.align import Align
from rich.columns import Columns
import random
import time
from typing import Dict, Any, List

class Theme:
    """Base theme class with color schemes and animations."""
    
    def __init__(self, name: str):
        self.name = name
        self.console = Console()
        self.colors = self._get_colors()
        self.ascii_art = self._get_ascii_art()
    
    def _get_colors(self) -> Dict[str, str]:
        """Get color scheme for the theme."""
        schemes = {
            "cyberpunk": {
                "primary": "#ff00ff",
                "secondary": "#00ffff",
                "accent": "#ffff00",
                "background": "#0a0a0a",
                "text": "#ffffff"
            },
            "ocean": {
                "primary": "#0077be",
                "secondary": "#00cccc",
                "accent": "#ffd700",
                "background": "#001a33",
                "text": "#e6f3ff"
            },
            "forest": {
                "primary": "#228b22",
                "secondary": "#90ee90",
                "accent": "#ffd700",
                "background": "#0f1f0f",
                "text": "#f5f5dc"
            },
            "sunset": {
                "primary": "#ff4500",
                "secondary": "#ff8c00",
                "accent": "#ffd700",
                "background": "#2f1b14",
                "text": "#fff8dc"
            }
        }
        return schemes.get(self.name, schemes["cyberpunk"])
    
    def _get_ascii_art(self) -> str:
        """Get ASCII art banner for the theme."""
        arts = {
            "cyberpunk": """
    ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    ██ ▄▄▄ ██ ▀██ ██ ▄▄▀██ ▄▄▄██ ▄▄▀
    ██ ███ ██ █ █ ██ ██ ██ ▄▄▄██ ▀▀▄
    ██ ▀▀▀ ██ ██▄ ██ ▀▀ ██ ▀▀▀██ ██
    ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
    CONTEXT7 AI - CYBERPUNK MODE ACTIVATED
            """,
            "ocean": """
     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
     ▓▓░▓▓░▓▓░▓▓░▓▓░▓▓░▓▓░▓▓░▓▓░▓▓░▓▓░
     ▓▓░▓▓░▓▓░▓▓░▓▓░▓▓░▓▓░▓▓░▓▓░▓▓░▓▓░
     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
     CONTEXT7 AI - OCEAN DEPTH EXPLORER
            """,
            "forest": """
    ╔═══════════════════════════════════╗
    ║ 🌲 🌳 🌲 🌳 🌲 🌳 🌲 🌳 🌲 🌳 ║
    ║    CONTEXT7 AI - FOREST GUARDIAN   ║
    ║ 🌳 🌲 🌳 🌲 🌳 🌲 🌳 🌲 🌳 🌲 ║
    ╚═══════════════════════════════════╝
            """,
            "sunset": """
    ☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️
    🌅 CONTEXT7 AI - SUNSET SAGA 🌅
    ☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️
            """
        }
        return arts.get(self.name, arts["cyberpunk"])
    
    def print_banner(self):
        """Display theme banner with animations."""
        self.console.print(f"\n[{self.colors['primary']}]{self.ascii_art}[/{self.colors['primary']}]\n")
    
    def create_search_table(self, results: List[Dict[str, Any]]) -> Table:
        """Create a styled table for search results."""
        table = Table(
            title=f"[{self.colors['accent']}]Search Results[/{self.colors['accent']}]",
            border_style=self.colors['secondary'],
            header_style=self.colors['primary']
        )
        
        table.add_column("Title", style=self.colors['text'])
        table.add_column("Source", style=self.colors['secondary'])
        table.add_column("Relevance", style=self.colors['accent'])
        
        for result in results[:10]:
            table.add_row(
                result.get('title', 'Unknown'),
                result.get('source', 'Context7'),
                f"{result.get('score', 0):.2f}"
            )
        
        return table
    
    def create_progress_spinner(self) -> Progress:
        """Create a themed progress spinner."""
        return Progress(
            SpinnerColumn(style=self.colors['primary']),
            TextColumn("[{task.description}]", style=self.colors['text']),
            console=self.console
        )

class ThemeManager:
    """Manages theme switching and animations."""
    
    def __init__(self):
        self.themes = {
            "cyberpunk": Theme("cyberpunk"),
            "ocean": Theme("ocean"),
            "forest": Theme("forest"),
            "sunset": Theme("sunset")
        }
        self.current_theme = "cyberpunk"
    
    def set_theme(self, theme_name: str) -> bool:
        """Switch to a new theme."""
        if theme_name in self.themes:
            self.current_theme = theme_name
            return True
        return False
    
    def get_current_theme(self) -> Theme:
        """Get the current theme."""
        return self.themes[self.current_theme]
    
    def list_themes(self) -> List[str]:
        """List available themes."""
        return list(self.themes.keys())
    
    def print_typing_effect(self, text: str, speed: int = 50):
        """Print text with typing animation effect."""
        theme = self.get_current_theme()
        for char in text:
            theme.console.print(char, end="", style=theme.colors['text'])
            time.sleep(1 / speed)
        theme.console.print()

```

# src/utils.py
```py
"""
Utility functions for the Context7 Agent.
"""

import re
from typing import List, Dict, Any
import hashlib

def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """Extract code blocks from text."""
    pattern = r'```(\w+)?\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    return [
        {"language": lang or "text", "code": code.strip()}
        for lang, code in matches
    ]

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    return re.sub(r'[^\w\s-]', '', filename).strip()

def generate_conversation_id(query: str) -> str:
    """Generate a unique conversation ID from query."""
    return hashlib.md5(query.encode()).hexdigest()[:8]

def format_search_result(result: Dict[str, Any]) -> str:
    """Format a search result for display."""
    title = result.get('title', 'Untitled')
    source = result.get('source', 'Unknown')
    score = result.get('score', 0)
    snippet = result.get('snippet', '')[:200] + "..."
    
    return f"""
📄 **{title}**
Source: {source} | Relevance: {score:.2f}
{snippet}
    """.strip()

def fuzzy_match(query: str, text: str) -> float:
    """Simple fuzzy matching score."""
    query_words = query.lower().split()
    text_words = text.lower().split()
    
    matches = sum(1 for qw in query_words if any(qw in tw for tw in text_words))
    return matches / len(query_words) if query_words else 0

```

