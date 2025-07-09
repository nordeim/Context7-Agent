<think>
The user wants me to create an AI agent using Pydantic AI that integrates with the Context7 MCP server. They've provided a JSON config for the MCP server and want me to adapt it for Pydantic AI's specific syntax. The agent should use OpenAIModel with configuration from environment variables, include a CLI for chatting, and store conversation history.

Let me analyze the requirements:
1. Create an agent using Pydantic AI with Context7 MCP integration
2. Use OpenAIModel with base URL and model from environment variables
3. Create a CLI for chatting with conversation history storage
4. The suggested modules are: agent.py, history.py, cli.py
5. Additional components needed: tests, README.md, .gitignore

Looking at the sample code provided, I can see they want a proper project structure with error handling and configuration management. Let me implement this step by step, starting with the configuration module since the agent will depend on it.
</think>

I'll help you create a comprehensive AI agent with Pydantic AI that integrates with the Context7 MCP server. Let me implement all the necessary components with proper structure and error handling.

First, let's create the configuration module:

**src/config.py**
```python
"""
Configuration module for the Context7 Agent.

This module handles loading and validating configuration from environment variables.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for the Context7 Agent."""
    
    # OpenAI configuration
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    
    # Optional configurations
    history_file: str = "conversation_history.json"
    max_history_entries: int = 1000
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4"),
            history_file=os.getenv("HISTORY_FILE", "conversation_history.json"),
            max_history_entries=int(os.getenv("MAX_HISTORY_ENTRIES", "1000"))
        )
    
    def validate(self) -> Optional[str]:
        """Validate the configuration."""
        if not self.openai_api_key:
            return "OPENAI_API_KEY environment variable is required"
        if not self.openai_base_url:
            return "OPENAI_BASE_URL environment variable is required"
        if not self.openai_model:
            return "OPENAI_MODEL environment variable is required"
        return None


# Global config instance
config = Config.from_env()
```

Now, let's create the complete agent module:

**src/agent.py**
```python
"""
Agent module for the Context7 Agent.

This module implements a Pydantic AI agent with Context7 MCP server integration.
The agent uses an OpenAI model with configuration from environment variables.
"""

import os
import sys
from typing import Dict, Any, Optional, List, Union
import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.tools import Tool, ToolDefinition
from pydantic_ai.messages import (
    Message,
    UserPrompt,
    ModelTextResponse,
    ModelMessagesTypeAdapter
)
from pydantic_ai.models.openai import OpenAIModel

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config


class Context7Agent:
    """
    Context7 Agent implementation using Pydantic AI.

    This agent integrates with the Context7 MCP server for enhanced context management
    and uses an OpenAI model as the underlying LLM provider.
    """

    def __init__(self):
        """
        Initialize the Context7 Agent with configuration from environment variables.

        Sets up the OpenAI model and Context7 MCP server integration.
        """
        # Validate configuration
        error = config.validate()
        if error:
            raise ValueError(f"Configuration error: {error}")
        
        # Initialize OpenAI model with custom base URL
        self.model = OpenAIModel(
            model=config.openai_model,
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )
        
        # Initialize the Pydantic AI agent
        self.agent = Agent(
            model=self.model,
            system_prompt="""You are a helpful AI assistant with access to Context7 MCP server 
            for enhanced context management. You can help with various tasks and maintain 
            contextual awareness across conversations."""
        )
        
        # Track MCP server process
        self.mcp_process = None
        
    async def initialize_mcp(self):
        """Initialize and connect to the Context7 MCP server."""
        try:
            # Import MCP-related modules
            from pydantic_ai.tools.mcp import MCPClient
            
            # Create MCP client for Context7
            self.mcp_client = MCPClient(
                command="npx",
                args=["-y", "@upstash/context7-mcp@latest"],
                env=os.environ.copy()
            )
            
            # Connect to the MCP server
            await self.mcp_client.connect()
            
            # Register MCP tools with the agent
            tools = await self.mcp_client.list_tools()
            for tool in tools:
                self.agent.register_tool(tool)
                
            print(f"Successfully connected to Context7 MCP server with {len(tools)} tools")
            
        except ImportError:
            print("Warning: MCP support not available. Running without Context7 integration.")
        except Exception as e:
            print(f"Warning: Failed to initialize Context7 MCP server: {e}")
            print("Continuing without MCP integration.")
    
    async def chat(self, message: str, history: Optional[List[Message]] = None) -> str:
        """
        Send a message to the agent and get a response.
        
        Args:
            message: The user's message
            history: Optional conversation history
            
        Returns:
            The agent's response as a string
        """
        try:
            # Prepare messages
            messages = history or []
            
            # Run the agent
            result = await self.agent.run(
                message,
                message_history=messages
            )
            
            return result.data
            
        except Exception as e:
            return f"Error processing message: {str(e)}"
    
    async def cleanup(self):
        """Clean up resources including MCP server connection."""
        if hasattr(self, 'mcp_client'):
            try:
                await self.mcp_client.disconnect()
            except Exception as e:
                print(f"Error disconnecting MCP client: {e}")
    
    def get_conversation_messages(self, history_entries: List[Dict[str, Any]]) -> List[Message]:
        """
        Convert history entries to Pydantic AI messages.
        
        Args:
            history_entries: List of history entries with role and content
            
        Returns:
            List of Message objects
        """
        messages = []
        for entry in history_entries:
            if entry["role"] == "user":
                messages.append(UserPrompt(content=entry["content"]))
            elif entry["role"] == "assistant":
                messages.append(ModelTextResponse(content=entry["content"]))
        return messages
```

Now, let's create the history module:

**src/history.py**
```python
"""
History module for managing conversation history.

This module provides functionality to store, retrieve, and manage conversation
history for the Context7 Agent.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import threading


class ConversationHistory:
    """Manages conversation history with thread-safe operations."""
    
    def __init__(self, history_file: str = "conversation_history.json", 
                 max_entries: int = 1000):
        """
        Initialize the conversation history manager.
        
        Args:
            history_file: Path to the JSON file for storing history
            max_entries: Maximum number of entries to keep in history
        """
        self.history_file = Path(history_file)
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._ensure_history_file()
    
    def _ensure_history_file(self):
        """Ensure the history file exists."""
        if not self.history_file.exists():
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            self._save_history([])
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load history from file."""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _save_history(self, history: List[Dict[str, Any]]):
        """Save history to file."""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    def add_entry(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a new entry to the conversation history.
        
        Args:
            role: The role (user or assistant)
            content: The message content
            metadata: Optional metadata for the entry
        """
        with self._lock:
            history = self._load_history()
            
            entry = {
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "content": content,
                "metadata": metadata or {}
            }
            
            history.append(entry)
            
            # Keep only the most recent entries
            if len(history) > self.max_entries:
                history = history[-self.max_entries:]
            
            self._save_history(history)
    
    def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent conversation entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of recent conversation entries
        """
        with self._lock:
            history = self._load_history()
            return history[-limit:] if history else []
    
    def get_all_history(self) -> List[Dict[str, Any]]:
        """Get all conversation history."""
        with self._lock:
            return self._load_history()
    
    def clear_history(self):
        """Clear all conversation history."""
        with self._lock:
            self._save_history([])
    
    def search_history(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search conversation history for entries containing the query.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of matching conversation entries
        """
        with self._lock:
            history = self._load_history()
            query_lower = query.lower()
            
            matches = []
            for entry in reversed(history):
                if query_lower in entry["content"].lower():
                    matches.append(entry)
                    if len(matches) >= limit:
                        break
            
            return matches
    
    def get_conversation_pairs(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent conversation pairs (user message + assistant response).
        
        Args:
            limit: Maximum number of pairs to return
            
        Returns:
            List of conversation pairs
        """
        with self._lock:
            history = self._load_history()
            pairs = []
            
            i = len(history) - 1
            while i > 0 and len(pairs) < limit:
                if (history[i]["role"] == "assistant" and 
                    history[i-1]["role"] == "user"):
                    pairs.append({
                        "user": history[i-1],
                        "assistant": history[i]
                    })
                    i -= 2
                else:
                    i -= 1
            
            return list(reversed(pairs))
```

Now, let's create the CLI module:

**src/cli.py**
```python
"""
CLI module for the Context7 Agent.

This module provides a command-line interface for interacting with the Context7 Agent
and managing conversation history.
"""

import asyncio
import sys
from typing import Optional
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import click

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import Context7Agent
from src.history import ConversationHistory
from src.config import config


console = Console()


class Context7CLI:
    """Command-line interface for the Context7 Agent."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.agent = None
        self.history = ConversationHistory(
            history_file=config.history_file,
            max_entries=config.max_history_entries
        )
        self.running = False
    
    async def initialize(self):
        """Initialize the agent and MCP connection."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing Context7 Agent...", total=None)
            
            try:
                self.agent = Context7Agent()
                await self.agent.initialize_mcp()
                progress.update(task, completed=True)
                
                console.print("[green]✓[/green] Context7 Agent initialized successfully!")
                
            except Exception as e:
                progress.update(task, completed=True)
                console.print(f"[red]✗[/red] Failed to initialize: {e}")
                raise
    
    async def chat_loop(self):
        """Main chat loop."""
        self.running = True
        
        console.print(Panel.fit(
            "[bold cyan]Context7 Agent CLI[/bold cyan]\n"
            "Type 'help' for commands, 'exit' to quit",
            border_style="cyan"
        ))
        
        while self.running:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold green]You[/bold green]")
                
                if not user_input.strip():
                    continue
                
                # Handle special commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                elif user_input.lower() == 'clear':
                    console.clear()
                    continue
                elif user_input.lower() == 'clear history':
                    self.clear_history()
                    continue
                elif user_input.lower().startswith('search '):
                    query = user_input[7:]
                    self.search_history(query)
                    continue
                
                # Add user message to history
                self.history.add_entry("user", user_input)
                
                # Get conversation history for context
                recent_history = self.history.get_recent_history(limit=10)
                messages = self.agent.get_conversation_messages(recent_history[:-1])
                
                # Get agent response
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True
                ) as progress:
                    task = progress.add_task("Thinking...", total=None)
                    
                    response = await self.agent.chat(user_input, messages)
                    
                    progress.update(task, completed=True)
                
                # Display response
                console.print(f"\n[bold blue]Assistant[/bold blue]")
                console.print(Panel(Markdown(response), border_style="blue"))
                
                # Add assistant response to history
                self.history.add_entry("assistant", response)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    def show_help(self):
        """Display help information."""
        help_text = """
# Available Commands

- **help** - Show this help message
- **history** - Show recent conversation history
- **clear** - Clear the screen
- **clear history** - Clear all conversation history
- **search <query>** - Search conversation history
- **exit/quit/q** - Exit the application
        """
        console.print(Panel(Markdown(help_text), title="Help", border_style="yellow"))
    
    def show_history(self, limit: int = 5):
        """Display recent conversation history."""
        pairs = self.history.get_conversation_pairs(limit=limit)
        
        if not pairs:
            console.print("[yellow]No conversation history found.[/yellow]")
            return
        
        table = Table(title=f"Recent Conversations (Last {limit})")
        table.add_column("Time", style="dim")
        table.add_column("User", style="green")
        table.add_column("Assistant", style="blue")
        
        for pair in pairs:
            time = pair["user"]["timestamp"].split("T")[1].split(".")[0]
            table.add_row(
                time,
                pair["user"]["content"][:50] + "..." if len(pair["user"]["content"]) > 50 else pair["user"]["content"],
                pair["assistant"]["content"][:50] + "..." if len(pair["assistant"]["content"]) > 50 else pair["assistant"]["content"]
            )
        
        console.print(table)
    
    def search_history(self, query: str):
        """Search conversation history."""
        results = self.history.search_history(query, limit=5)
        
        if not results:
            console.print(f"[yellow]No results found for '{query}'[/yellow]")
            return
        
        console.print(f"\n[bold]Search Results for '{query}':[/bold]\n")
        
        for i, entry in enumerate(results, 1):
            role_style = "green" if entry["role"] == "user" else "blue"
            console.print(f"[{role_style}]{entry['role'].title()}[/{role_style}] "
                         f"[dim]({entry['timestamp'].split('T')[0]})[/dim]")
            console.print(Panel(entry["content"][:200] + "..." 
                              if len(entry["content"]) > 200 
                              else entry["content"]))
            if i < len(results):
                console.print()
    
    def clear_history(self):
        """Clear conversation history with confirmation."""
        confirm = Prompt.ask(
            "[yellow]Are you sure you want to clear all history?[/yellow]",
            choices=["yes", "no"],
            default="no"
        )
        
        if confirm.lower() == "yes":
            self.history.clear_history()
            console.print("[green]✓[/green] Conversation history cleared.")
        else:
            console.print("[dim]History clear cancelled.[/dim]")
    
    async def cleanup(self):
        """Clean up resources."""
        if self.agent:
            await self.agent.cleanup()


@click.command()
@click.option('--no-history', is_flag=True, help='Start without loading conversation history')
def main(no_history):
    """Context7 Agent CLI - AI assistant with Context7 MCP integration."""
    cli = Context7CLI()
    
    if no_history:
        cli.history.clear_history()
    
    async def run():
        try:
            await cli.initialize()
            await cli.chat_loop()
        except Exception as e:
            console.print(f"[red]Fatal error: {e}[/red]")
        finally:
            await cli.cleanup()
            console.print("\n[cyan]Goodbye![/cyan]")
    
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")


if __name__ == "__main__":
    main()
```

Let's fix the import issue in cli.py:

**src/cli.py** (updated imports)
```python
"""
CLI module for the Context7 Agent.

This module provides a command-line interface for interacting with the Context7 Agent
and managing conversation history.
"""

import asyncio
import sys
import os
from typing import Optional
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import click

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import Context7Agent
from src.history import ConversationHistory
from src.config import config

# ... rest of the code remains the same
```

Now, let's create the main entry point:

**main.py**
```python
#!/usr/bin/env python3
"""
Main entry point for the Context7 Agent CLI.
"""

from src.cli import main

if __name__ == "__main__":
    main()
```

Let's create the test files:

**tests/test_agent.py**
```python
"""Tests for the Context7 Agent module."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import Context7Agent
from src.config import Config


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    with patch('src.agent.config') as mock:
        mock.openai_api_key = "test-key"
        mock.openai_base_url = "https://test.com"
        mock.openai_model = "test-model"
        mock.validate.return_value = None
        yield mock


@pytest.fixture
async def agent(mock_config):
    """Create a Context7Agent instance for testing."""
    agent = Context7Agent()
    yield agent
    await agent.cleanup()


class TestContext7Agent:
    """Test cases for Context7Agent."""
    
    def test_initialization_with_invalid_config(self):
        """Test that initialization fails with invalid configuration."""
        with patch('src.agent.config') as mock_config:
            mock_config.validate.return_value = "Missing API key"
            
            with pytest.raises(ValueError, match="Configuration error"):
                Context7Agent()
    
    @pytest.mark.asyncio
    async def test_chat_basic(self, agent):
        """Test basic chat functionality."""
        with patch.object(agent.agent, 'run') as mock_run:
            mock_result = Mock()
            mock_result.data = "Hello! How can I help you?"
            mock_run.return_value = mock_result
            
            response = await agent.chat("Hello")
            
            assert response == "Hello! How can I help you?"
            mock_run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chat_with_history(self, agent):
        """Test chat with conversation history."""
        from pydantic_ai.messages import UserPrompt, ModelTextResponse
        
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
        
        messages = agent.get_conversation_messages(history)
        
        assert len(messages) == 2
        assert isinstance(messages[0], UserPrompt)
        assert isinstance(messages[1], ModelTextResponse)
        assert messages[0].content == "Previous question"
        assert messages[1].content == "Previous answer"
    
    @pytest.mark.asyncio
    async def test_chat_error_handling(self, agent):
        """Test error handling in chat method."""
        with patch.object(agent.agent, 'run') as mock_run:
            mock_run.side_effect = Exception("Test error")
            
            response = await agent.chat("Test message")
            
            assert "Error processing message" in response
            assert "Test error" in response
    
    def test_get_conversation_messages(self, agent):
        """Test conversion of history entries to messages."""
        history_entries = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"}
        ]
        
        messages = agent.get_conversation_messages(history_entries)
        
        assert len(messages) == 3
        assert messages[0].content == "Question 1"
        assert messages[1].content == "Answer 1"
        assert messages[2].content == "Question 2"
```

**tests/test_history.py**
```python
"""Tests for the conversation history module."""

import pytest
import json
import os
from pathlib import Path
from datetime import datetime
import tempfile
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.history import ConversationHistory


@pytest.fixture
def temp_history_file():
    """Create a temporary file for history testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def history(temp_history_file):
    """Create a ConversationHistory instance with a temporary file."""
    return ConversationHistory(history_file=temp_history_file, max_entries=10)


class TestConversationHistory:
    """Test cases for ConversationHistory."""
    
    def test_initialization(self, history, temp_history_file):
        """Test history initialization."""
        assert history.history_file == Path(temp_history_file)
        assert history.max_entries == 10
        assert os.path.exists(temp_history_file)
    
    def test_add_entry(self, history):
        """Test adding entries to history."""
        history.add_entry("user", "Hello")
        history.add_entry("assistant", "Hi there!")
        
        all_history = history.get_all_history()
        assert len(all_history) == 2
        assert all_history[0]["role"] == "user"
        assert all_history[0]["content"] == "Hello"
        assert all_history[1]["role"] == "assistant"
        assert all_history[1]["content"] == "Hi there!"
    
    def test_add_entry_with_metadata(self, history):
        """Test adding entries with metadata."""
        metadata = {"model": "gpt-4", "temperature": 0.7}
        history.add_entry("assistant", "Response", metadata=metadata)
        
        all_history = history.get_all_history()
        assert all_history[0]["metadata"] == metadata
    
    def test_max_entries_limit(self, history):
        """Test that history respects max_entries limit."""
        # Add more entries than the limit
        for i in range(15):
            history.add_entry("user", f"Message {i}")
        
        all_history = history.get_all_history()
        assert len(all_history) == 10
        assert all_history[0]["content"] == "Message 5"
        assert all_history[-1]["content"] == "Message 14"
    
    def test_get_recent_history(self, history):
        """Test getting recent history."""
        for i in range(5):
            history.add_entry("user", f"Message {i}")
        
        recent = history.get_recent_history(limit=3)
        assert len(recent) == 3
        assert recent[0]["content"] == "Message 2"
        assert recent[-1]["content"] == "Message 4"
    
    def test_clear_history(self, history):
        """Test clearing history."""
        history.add_entry("user", "Test message")
        history.clear_history()
        
        all_history = history.get_all_history()
        assert len(all_history) == 0
    
    def test_search_history(self, history):
        """Test searching history."""
        history.add_entry("user", "Tell me about Python")
        history.add_entry("assistant", "Python is a programming language")
        history.add_entry("user", "What about JavaScript?")
        history.add_entry("assistant", "JavaScript is another language")
        
        results = history.search_history("Python", limit=10)
        assert len(results) == 2
        assert all("python" in r["content"].lower() for r in results)
    
    def test_get_conversation_pairs(self, history):
        """Test getting conversation pairs."""
        history.add_entry("user", "Question 1")
        history.add_entry("assistant", "Answer 1")
        history.add_entry("user", "Question 2")
        history.add_entry("assistant", "Answer 2")
        history.add_entry("user", "Question 3")  # Unpaired
        
        pairs = history.get_conversation_pairs(limit=5)
        assert len(pairs) == 2
        assert pairs[0]["user"]["content"] == "Question 1"
        assert pairs[0]["assistant"]["content"] == "Answer 1"
        assert pairs[1]["user"]["content"] == "Question 2"
        assert pairs[1]["assistant"]["content"] == "Answer 2"
    
    def test_thread_safety(self, history):
        """Test thread safety of history operations."""
        import threading
        import time
        
        def add_entries(start, end):
            for i in range(start, end):
                history.add_entry("user", f"Message {i}")
                time.sleep(0.001)
        
        threads = []
        for i in range(3):
            t = threading.Thread(target=add_entries, args=(i*10, (i+1)*10))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        all_history = history.get_all_history()
        assert len(all_history) == 10  # Should respect max_entries
```

Now let's create the README:

**README.md**
```markdown
# Context7 Agent

An AI agent powered by Pydantic AI with Context7 MCP (Model Context Protocol) server integration. This agent provides enhanced context management capabilities and can be accessed through a command-line interface.

## Features

- 🤖 **Pydantic AI Integration**: Built on top of Pydantic AI for robust agent capabilities
- 🔌 **Context7 MCP Server**: Integrated with Context7 MCP server for enhanced context management
- 💬 **Interactive CLI**: User-friendly command-line interface with rich formatting
- 📝 **Conversation History**: Persistent conversation history with search capabilities
- 🔧 **Configurable**: Flexible configuration through environment variables
- 🧪 **Well-tested**: Comprehensive test suite included

## Prerequisites

- Python 3.8 or higher
- Node.js and npm (for Context7 MCP server)
- OpenAI API key or compatible API endpoint

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd context7-agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Set the following environment variables:

```bash
# Required
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Or your custom endpoint
export OPENAI_MODEL="gpt-4"  # Or your preferred model

# Optional
export HISTORY_FILE="conversation_history.json"
export MAX_HISTORY_ENTRIES="1000"
```

You can also create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4
HISTORY_FILE=conversation_history.json
MAX_HISTORY_ENTRIES=1000
```

## Usage

### Starting the CLI

Run the agent:
```bash
python main.py
```

Or make the script executable:
```bash
chmod +x main.py
./main.py
```

### CLI Commands

Once in the interactive CLI, you can use the following commands:

- **help** - Display available commands
- **history** - Show recent conversation history
- **clear** - Clear the screen
- **clear history** - Clear all conversation history
- **search <query>** - Search through conversation history
- **exit/quit/q** - Exit the application

### Command-line Options

```bash
# Start without loading previous conversation history
python main.py --no-history
```

## Project Structure

```
context7-agent/
├── src/
│   ├── __init__.py
│   ├── agent.py         # Core agent implementation
│   ├── cli.py           # Command-line interface
│   ├── config.py        # Configuration management
│   └── history.py       # Conversation history management
├── tests/
│   ├── __init__.py
│   ├── test_agent.py    # Agent tests
│   └── test_history.py  # History tests
├── main.py              # Entry point
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── .gitignore         # Git ignore rules
```

## Development

### Running Tests

Run the test suite:
```bash
pytest tests/
```

With coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

### Code Style

This project uses standard Python conventions. Format code with:
```bash
black src/ tests/
isort src/ tests/
```

## Troubleshooting

### MCP Server Connection Issues

If you encounter issues with the Context7 MCP server:

1. Ensure Node.js and npm are installed:
```bash
node --version
npm --version
```

2. The Context7 MCP server will be automatically downloaded and run via npx

3. Check for any firewall or network restrictions

### API Connection Issues

1. Verify your API key is correct
2. Check the base URL is accessible
3. Ensure the model name is valid for your endpoint

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Your License Here]

## Acknowledgments

- Built with [Pydantic AI](https://github.com/pydantic/pydantic-ai)
- Context management powered by [Context7 MCP](https://github.com/upstash/context7-mcp)
```

Finally, let's create the .gitignore file:

**.gitignore**
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/
.venv

# Distribution / packaging
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
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
.nox/
coverage.xml
*.cover
.hypothesis/

# IDEs
.idea/
.vscode/
*.swp
*.swo
*~
.project
.pydevproject

# Environment variables
.env
.env.local
.env.*.local

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Project specific
conversation_history.json
*.bak
temp/
tmp/

# Node.js (for MCP server)
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
```

And finally, let's create the requirements.txt:

**requirements.txt**
```txt
pydantic-ai>=0.0.5
pydantic>=2.0.0
openai>=1.0.0
rich>=13.0.0
click>=8.0.0
python-dotenv>=1.0.0
aiofiles>=23.0.0

# Testing dependencies
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0

# Development dependencies
black>=23.0.0
isort>=5.0.0
flake8>=6.0.0
```

Let's also create an __init__.py file for the src directory:

**src/__init__.py**
```python
"""Context7 Agent package."""

__version__ = "0.1.0"
__author__ = "Your Name"
```

And for the tests directory:

**tests/__init__.py**
```python
"""Tests for Context7 Agent."""
```

This implementation provides a complete AI agent with Pydantic AI that integrates with the Context7 MCP server. The key features include:

1. **Proper MCP Integration**: Uses Pydantic AI's MCP client to connect to the Context7 server
2. **Environment Configuration**: All settings configurable via environment variables
3. **Rich CLI**: Interactive command-line interface with syntax highlighting and formatting
4. **Conversation History**: Persistent storage with search and management capabilities
5. **Comprehensive Testing**: Unit tests for core functionality
6. **Professional Structure**: Well-organized code with proper error handling

To use this agent:

1. Set up your environment variables
2. Run `pip install -r requirements.txt`
3. Execute `python main.py` to start the CLI
4. The Context7 MCP server will be automatically started and connected

The agent will maintain context across conversations using the Context7 MCP server and provide an enhanced AI assistant experience.
