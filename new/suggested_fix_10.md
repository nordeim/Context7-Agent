The error indicates that the MCP server isn't running when the agent tries to use it. You need to start the MCP server before using it with the agent. Here's how to fix your code:

```python
# File: src/agent.py
from __future__ import annotations
import asyncio
import json
from typing import AsyncIterator

from pydantic_ai import Agent, TextOutput
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

# Global MCP server instance
_mcp_server = None

def _build_llm() -> OpenAIModel:
    """
    Constructs the LLM with static configuration (provider, model name).
    Generation parameters are passed per-request in `stream_reply`.
    """
    provider = OpenAIProvider(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    return OpenAIModel(
        settings.openai_model,  # The model name is the first positional argument
        provider=provider,
    )

def _build_mcp() -> MCPServerStdio:
    """
    Loads the MCP server configuration from mcp.config.json,
    finds the server by its alias, and constructs the MCPServerStdio
    instance with the correct command and args.
    """
    with open(settings.mcp_config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    server_config = config_data.get("mcpServers", {}).get(settings.mcp_alias)

    if not server_config:
        raise ValueError(
            f"MCP server alias '{settings.mcp_alias}' not found in {settings.mcp_config_path}"
        )

    command = server_config.get("command")
    args = server_config.get("args", [])

    if not command:
        raise ValueError(
            f"'command' not specified for MCP server '{settings.mcp_alias}' in {settings.mcp_config_path}"
        )

    return MCPServerStdio(command, args)

async def get_or_create_mcp_server() -> MCPServerStdio:
    """Get the global MCP server instance, starting it if necessary."""
    global _mcp_server
    
    if _mcp_server is None:
        _mcp_server = _build_mcp()
    
    # Start the server if it's not running
    if not _mcp_server.is_running:
        await _mcp_server.__aenter__()
    
    return _mcp_server

def create_agent(mcp_server: MCPServerStdio) -> Agent:
    """Create agent with the provided MCP server."""
    return Agent(
        model=_build_llm(), 
        mcp_servers=(mcp_server,), 
        system_prompt=SYSTEM_PROMPT
    )

# ------------------------------------------------------------------ high-level
async def stream_reply(history: History) -> AsyncIterator[tuple[str, str]]:
    """
    Merge assistant tokens and MCP tool events into a single async stream.
    Yields tuples (role, payload)
      role == "assistant" -> markdown str
      role == "mcp"       -> already formatted message (string)
    """
    # Ensure MCP server is running
    mcp_server = await get_or_create_mcp_server()
    
    # Create agent with running MCP server
    agent = create_agent(mcp_server)
    
    # Get the last user message as the prompt
    messages = history.to_model_messages()
    user_prompt = None
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            user_prompt = msg.get('content')
            break
    
    if not user_prompt:
        return
    
    # Use message_history for previous messages (excluding the last user message)
    message_history = messages[:-1] if len(messages) > 1 else None
    
    # Model settings for temperature and max_tokens
    model_settings = {
        'temperature': 0.3,
        'max_tokens': 4096
    }
    
    # Call agent.run_stream with the correct parameters
    async with agent.run_stream(
        user_prompt,
        message_history=message_history,
        model_settings=model_settings,
    ) as stream:
        async for event in stream:
            if isinstance(event, TextOutput):
                yield "assistant", event.content
            else:
                yield "mcp", str(event)

# Optional: Clean up function to stop the MCP server when done
async def cleanup():
    """Stop the MCP server if it's running."""
    global _mcp_server
    if _mcp_server and _mcp_server.is_running:
        await _mcp_server.__aexit__(None, None, None)
        _mcp_server = None
```

The key changes are:

1. **Global MCP server instance**: Instead of creating a new MCP server for each request, we maintain a global instance that stays running.

2. **`get_or_create_mcp_server()` function**: This ensures the MCP server is started before use. It uses the async context manager protocol (`__aenter__`) to start the server.

3. **Pass the running server to the agent**: The agent now receives an already-running MCP server instance.

4. **Optional cleanup function**: You can call this when your application shuts down to properly close the MCP server.

You might also want to add cleanup logic to your CLI when the user exits:

```python
# In src/cli.py, add this at the end of the chat() function:
@app.command()
def chat():
    """
    Launch the interactive CLI with Live layout and streaming responses.
    """
    banner()
    c = get_console()
    layout = ChatLayout()

    try:
        with Live(layout, console=c, auto_refresh=False, screen=False) as live:
            while True:
                # ... existing code ...
    finally:
        # Clean up the MCP server when exiting
        import asyncio
        from .agent import cleanup
        asyncio.run(cleanup())
```

