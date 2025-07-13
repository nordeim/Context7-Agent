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
        model=settings.openai_model,
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

