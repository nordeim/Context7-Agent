"""
Pydantic-AI agent with native MCP integration.
python agent.py
"""
import asyncio, json, os, sys
from typing import List, Dict, Any
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel, Field
import httpx
import websockets

OPENAI_KEY = os.getenv("OPENAI_API_KEY") or sys.exit("Need OPENAI_API_KEY")

# --- MCP client helpers -------------------------------------------------------
class MCPClient:
    def __init__(self, uri="ws://localhost:8766"):
        self.uri = uri
        self.ws = None

    async def connect(self):
        self.ws = await websockets.connect(self.uri)

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        payload = {"name": name, "arguments": arguments}
        await self.ws.send(json.dumps(payload))
        resp = json.loads(await self.ws.recv())
        return resp

    async def close(self):
        if self.ws:
            await self.ws.close()

# --- Pydantic-AI agent --------------------------------------------------------
class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str

class AgentDeps(BaseModel):
    mcp: MCPClient

agent = Agent(
    model=OpenAIModel("gpt-4o-mini", api_key=OPENAI_KEY),
    deps_type=AgentDeps,
    result_type=List[SearchResult],
    system_prompt=(
        "You are a futuristic librarian. Use the MCP server tools "
        "(search_docs, bookmark_doc, list_bookmarks) to help the user. "
        "Always return a concise list of relevant docs."
    ),
)

@agent.tool
async def search_docs(ctx: RunContext[AgentDeps], query: str, limit: int = 5) -> List[SearchResult]:
    """Search Context7 docs."""
    raw = await ctx.deps.mcp.call_tool("search_docs", {"query": query, "limit": limit})
    return [SearchResult(**d) for d in raw]

@agent.tool
async def bookmark_doc(ctx: RunContext[AgentDeps], url: str, note: str = "") -> Dict[str, Any]:
    """Bookmark a document."""
    return await ctx.deps.mcp.call_tool("bookmark_doc", {"url": url, "note": note})

@agent.tool
async def list_bookmarks(ctx: RunContext[AgentDeps]) -> List[Dict[str, Any]]:
    """List all bookmarks."""
    return await ctx.deps.mcp.call_tool("list_bookmarks", {})

# --- Entry point (for standalone tests) ---------------------------------------
if __name__ == "__main__":
    async def _main():
        async with MCPClient() as mcp:
            await mcp.connect()
            result = await agent.run("quantum computing", deps=AgentDeps(mcp=mcp))
            print(result.data)

    asyncio.run(_main())
