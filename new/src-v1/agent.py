# File: src/agent.py
from __future__ import annotations
import asyncio
from typing import List

from pydantic_ai import Agent, ChatMessage
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel

from .config import settings
from .history import History


SYSTEM_PROMPT = """
You are Context7 — an expert, friendly document-librarian.
When the user’s intent is **information retrieval**, 
call MCP.search with a natural-language query.
For management actions, call the proper MCP verbs (preview, bookmark …).
Respond in a concise, helpful style.
"""

def build_llm() -> OpenAIModel:
    provider = OpenAIProvider(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    return OpenAIModel(
        provider=provider,
        model=settings.openai_model,
        temperature=0.2,
        max_tokens=2048
    )

def build_mcp() -> MCPServerStdio:
    """
    Boot the Context7 Node.js server in a background stdio subprocess.

    It picks the command from the JSON config located at settings.mcp_config_file
    (see README).  You can add many servers and reference them by alias.
    """
    return MCPServerStdio(server=settings.mcp_server_alias)

def create_agent() -> Agent:
    """
    Factory that assembles the LLM + MCP toolchain into a single Agent instance.
    Conversation memory is *externalised* to History; the agent is transient.
    """
    llm = build_llm()
    mcp = build_mcp()

    agent = Agent(
        model=llm,
        mcp_server=mcp,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


# Async helper (used by the CLI) ---------------------------------------------
async def chat_stream(history: History):
    agent = create_agent()

    async for event in agent.stream_chat(messages=history.to_model_messages()):
        # Each event is a ChatMessage or a MCP tool result.
        if isinstance(event, ChatMessage):
            yield event.role, event.content
        else:
            # Handle MCP-side events (search results, preview, …)
            yield "mcp", event  # event already pretty-formatted by pydantic-ai
