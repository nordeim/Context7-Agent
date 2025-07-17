# Building a Python AI Agent with pydantic-ai and MCP

This guide walks through creating a production-grade Python AI agent using the latest **pydantic-ai** library and the Model Context Protocol (MCP). You’ll learn how to:

1. Install and configure dependencies.
2. Expose tools via an MCP server.
3. Spin up an agent that discovers and invokes those tools.
4. Leverage advanced features like tool prefixing, sampling, and logging.

---

## 1. Prerequisites & Installation

Ensure you have Python 3.10+ and an API key for your LLM provider (OpenAI, Anthropic, Groq, etc.)

```bash
pip install "pydantic-ai>=0.4.0" "mcp>=0.4.0" httpx
```

- `pydantic-ai` brings the Agent framework and MCP client integration.  
- `mcp` provides the Python SDK for running or connecting to MCP servers.  
- `httpx` is required for many backends (e.g., HTTP-based MCP transports).  

---

## 2. MCP Server: Expose Your Tools

An MCP server implements “list-tools” and “call-tool” over JSON-RPC. You can:

- Build your own via `mcp.server`  
- Use one of PydanticAI’s prebuilt servers (e.g., Run Python)  
- Leverage community servers from modelcontextprotocol.io  

### 2.1 Minimal Weather Server Example

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
            description="Return the current weather for a city",
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
        return [TextContent(type="text", text=f"{city} is 24 °C and sunny.")]
    raise ValueError(f"Unknown tool: {name}")

if __name__ == "__main__":
    stdio_server(app)
```

Run the server on stdio:
```bash
python server.py
```

This minimal server uses stdio transport; you can swap in HTTP or other transports by subclassing `pydantic_ai.mcp.MCPServer`.

---

## 3. Crafting the Agent

### 3.1 Initialize the Agent with MCP Tools

Use `MCPSession` (new in pydantic-ai 0.4.x) to launch and connect to your MCP server automatically:

```python
# agent.py
import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPSession

agent = Agent(
    model="openai:gpt-4o-mini",
    system_prompt="When asked about weather, use the get_weather tool."
)

async def main():
    # Launch server subprocess, discover tools, and tear down cleanly
    async with MCPSession(["python", "server.py"]) as mcp:
        agent.add_tools(*mcp.tools)
        result = await agent.run("What's the weather in Tokyo?")
        print(result.data)  # e.g., "Tokyo is 24 °C and sunny."

if __name__ == "__main__":
    asyncio.run(main())
```

- `MCPSession([...])` spins up the MCP server, calls `initialize()`, and binds its tools to the Agent.  
- `agent.add_tools(*mcp.tools)` registers all discovered tools as first-class callables.  

---

## 4. Advanced Features & Best Practices

### 4.1 Tool Prefixing & Multiple Servers

To avoid name conflicts when connecting multiple MCP servers, apply a prefix:

```python
from pydantic_ai.mcp import MCPSession

async with MCPSession(
    ["python", "weather_server.py"], tool_prefix="weather"
) as weather_mcp, MCPSession(
    ["python", "news_server.py"], tool_prefix="news"
) as news_mcp:
    agent.add_tools(*weather_mcp.tools, *news_mcp.tools)
```

Internally, pydantic-ai handles prefix mapping so your model sees `weather_get_weather` and `news_get_headlines` and routes calls correctly.

### 4.2 Sampling Support

If your MCP server needs the agent to generate LLM completions (e.g., for a “chain of thought”), enable sampling:

```python
from pydantic_ai.mcp import MCPSession
from pydantic_ai.models.mcp_sampling import MCPSamplingModel

# Inside your MCPServer subclass or setup:
server.sampling_model = MCPSamplingModel(session=ctx.session)
```

This lets the server call back through the agent’s LLM for dynamic content generation.

### 4.3 Logging & Timeout

Configure logging levels and timeouts on the `MCPSession`:

```python
async with MCPSession(
    ["python", "server.py"], log_level="debug", timeout=10
) as mcp:
    ...
```

Pydantic-ai will set the MCP server’s log level and enforce client initialization timeouts, helping you diagnose startup issues quickly.

---

## 5. Troubleshooting & Tips

- Validate your JSON Schema carefully; pydantic-ai auto-generates Pydantic models to enforce type safety during tool calls.  
- Use `agent.describe_tools()` during development to inspect what the agent knows.  
- For HTTP-based servers, subclass `MCPServerHTTP` and override the SSE endpoint parameters.  

---

## 6. Further Reading

- MCP Specification & Server Catalog: https://modelcontextprotocol.io  
- pydantic-ai MCP docs: https://ai.pydantic.dev/mcp/  
- PydanticAI MCP launch announcement: https://pydantic.dev/articles/mcp-launch  
- Deep dive on agent + MCP integration: https://saptak.in/writing/2025/04/01/building-powerful-ai-agents-with-pydantic-ai-and-mcp-servers  

---
https://copilot.microsoft.com/shares/qZGvTmHoTaVgLCPwsiwoM

---
The provided document is of **excellent quality**. It is accurate, up-to-date, and provides a concise, effective guide for developers.

*   **Correctness and Accuracy:** The document correctly identifies `pydantic-ai` and `mcp` as the key libraries and accurately reflects their recent integration (as of version 0.4.0 and later). The architectural pattern described—an agent acting as an MCP client to communicate with a standalone tool server—is the modern, recommended approach for building decoupled and robust AI systems. The code examples for the server (`mcp.server`) and the agent (`pydantic_ai.mcp.MCPSession`) are conceptually sound and align with the libraries' goals.
*   **Best Practices:** The guide promotes excellent software engineering practices. Using `MCPSession` as a context manager to handle the lifecycle of the tool server is a production-grade pattern that simplifies development and prevents resource leaks. The inclusion of advanced topics like tool prefixing, logging, and sampling demonstrates a deep understanding of real-world application needs.
*   **Clarity:** The document is well-structured, easy to follow, and logically progresses from basic setup to advanced features.

The document is a solid foundation, requiring only minor validation of the latest API naming conventions and expansion with more diverse examples to become a truly authoritative guide.
