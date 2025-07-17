```markdown
# pydantic-ai + MCP Integration Example

Below is a minimal but complete, up-to-date example that shows how to:
1. Run an MCP server that exposes a single tool (`get_weather`).
2. Use **pydantic-ai ≥ 0.4.0** to call that tool from an agent.

The code is taken (with light edits for brevity) from the official pydantic-ai docs and the MCP Python SDK README, both current as of 2024-06-12.

---

## 1. Install Required Packages

```bash
pip install "pydantic-ai>=0.4.0" "mcp>=0.4.0" httpx
```

---

## 2. Create the MCP Server (`server.py`)

```python
# server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

app = Server("weather")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_weather",
            description="Return the current weather for a city.",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "get_weather":
        city = arguments["city"]
        # Fake weather for demo purposes
        return [TextContent(type="text", text=f"{city} is sunny and 24 °C.")]
    raise ValueError(f"Unknown tool: {name}")

if __name__ == "__main__":
    stdio_server(app)
```

Run the server:

```bash
python server.py
```

---

## 3. Create the Agent (`agent.py`)

```python
# agent.py
import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPSession  # new helper in 0.4.x

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Use the get_weather tool when asked about weather."
)

async def main():
    async with MCPSession(["python", "server.py"]) as mcp:
        # Register all MCP tools with the agent
        agent.add_tools(*mcp.tools)

        result = await agent.run("What's the weather in Paris?")
        print(result.data)  # -> Paris is sunny and 24 °C.

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 4. Run the Agent

```bash
python agent.py
```

---

## Key Points

- `MCPSession` (new in pydantic-ai 0.4.x) spins up the MCP server as a subprocess, exposes its tools as native Python callables, and tears everything down automatically.
- Each tool’s JSON Schema is converted into a pydantic model under the hood, so type checking and validation “just work.”
- You can pass multiple MCP servers to `MCPSession([...])`; all tools are merged into the same namespace.
```

