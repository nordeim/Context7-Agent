# **Authoritative Guide: Building Production-Grade AI Agents with Pydantic-AI and MCP**

This guide provides a comprehensive walkthrough for creating sophisticated, production-grade Python AI agents. We will leverage the powerful `pydantic-ai` library to build the agent and the **Model Context Protocol (MCP)** to provide tools in a decoupled, standardized, and scalable manner.

This architecture is the future of agentic systems, allowing you to separate your agent's core logic from the tools it uses. This means tools can be developed, deployed, and maintained independently, in any programming language, while your agent can discover and use them seamlessly.

You will learn to:

1.  **Understand the Agent-Tool Architecture:** Grasp why decoupling tools via MCP is a superior design.
2.  **Install and Configure Dependencies:** Set up your environment with the latest libraries.
3.  **Expose Tools via an MCP Server:** Create a standalone Python server to offer tools to any compatible AI agent.
4.  **Develop a Smart Agent:** Build a `pydantic-ai` agent that automatically discovers and invokes the server's tools.
5.  **Explore a Multi-Tool Example:** Implement a calculator server with multiple functions.
6.  **Leverage Advanced Features:** Master tool prefixing, sampling, logging, and timeouts for robust applications.

---

## 1. The "Decoupled Agent" Architecture: Why MCP?

In a simple AI application, you might define your tools as Python functions directly within the agent's code. This is fine for prototypes, but it has major drawbacks in production:

*   **Monolithic:** The agent and its tools are a single unit, making them hard to update and scale independently.
*   **Language-Locked:** Tools must be written in Python.
*   **Poor Security:** The agent has direct access to the tool's code and environment, creating a large attack surface.

**MCP solves this by creating a client-server architecture:**

*   **The MCP Server:** A simple, standalone application that exposes a set of tools over a network protocol (like stdio or HTTP). It knows nothing about the agent.
*   **The MCP Client (Your Agent):** The `pydantic-ai` agent connects to the server, asks what tools are available, and then calls them as needed.

This design is modular, secure, and language-agnostic. Your company's Rust team could expose a high-performance data processing tool via an MCP server, and your Python agent could use it without any custom integration code.

---

## 2. Prerequisites & Installation

Ensure you have **Python 3.10+** and an API key for your chosen LLM provider (e.g., OpenAI, Anthropic, Google, Groq).

```bash
# We recommend using a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the necessary libraries
pip install "pydantic-ai>=0.4.0" "mcp>=0.4.0" httpx
```

*   `pydantic-ai`: The core framework for building the agent and integrating the MCP client.
*   `mcp`: The Python SDK for building and interacting with MCP servers.
*   `httpx`: A required dependency for many LLM backends.

---

## 3. Creating an MCP Server: Exposing Your Tools

An MCP server is a straightforward application that implements two core JSON-RPC methods: `list-tools` and `call-tool`. The `mcp` library makes this incredibly simple.

### 3.1. Example: A Minimal Weather Server

Let's create a server that exposes a single tool: `get_weather`.

```python
# weather_server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import logging

# Configure basic logging for the server
logging.basicConfig(level=logging.INFO, format='%(asctime)s - SERVER - %(levelname)s - %(message)s')

# 1. Initialize the MCP Server application
# The name "weather" is an identifier for this server.
app = Server("weather")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """
    This function is called by the client to discover available tools.
    It returns a list of Tool objects.
    """
    logging.info("list_tools() called by a client.")
    return [
        Tool(
            name="get_weather",
            description="Return the current weather for a specified city",
            # The inputSchema defines the arguments for the tool, using JSON Schema.
            # pydantic-ai uses this to validate inputs before calling the tool.
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The name of the city, e.g., 'San Francisco'"}
                },
                "required": ["city"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    This function is called by the client to execute a tool.
    It receives the tool name and arguments, and must return the result.
    """
    logging.info(f"call_tool() invoked for tool: '{name}' with args: {arguments}")
    if name == "get_weather":
        city = arguments.get("city")
        if not city:
            raise ValueError("The 'city' argument is required.")
        # In a real app, you'd call a weather API here.
        return [TextContent(type="text", text=f"The weather in {city} is a perfect 24°C and sunny.")]
    
    # It's good practice to raise an error for unknown tools.
    raise ValueError(f"Unknown tool: {name}")

if __name__ == "__main__":
    # 3. Run the server using the stdio transport.
    # This means the server will communicate over standard input and output,
    # perfect for running it as a local subprocess.
    stdio_server(app)

```

You can run this server directly from your terminal, but it will wait for JSON-RPC messages. The magic happens when the agent starts it as a subprocess.

```bash
# You can run this, but it will just hang, waiting for input.
# The agent will manage this process for you.
python weather_server.py
```

---

## 4. Crafting the Agent to Use the Tools

Now, let's build the agent. We will use `pydantic_ai.mcp.MCPSession`, a high-level context manager that handles starting the server, connecting to it, and shutting it down cleanly.

```python
# agent_weather.py
import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPSession

# Initialize the agent with your preferred LLM.
# GPT-4o-mini is a great, fast choice for tool-using tasks.
agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a helpful assistant. When asked about the weather, you must use the get_weather tool."
)

async def main():
    print("Starting agent with MCPSession to connect to the weather server...")
    
    # MCPSession is the key component from pydantic-ai.
    # It takes the command to start the server as an argument.
    # It manages the subprocess lifecycle, initialization, and cleanup.
    async with MCPSession(["python", "weather_server.py"]) as mcp:
        # 1. Discover tools: mcp.tools contains all tools from the server.
        print(f"Discovered tools from server: {[tool.name for tool in mcp.tools]}")
        
        # 2. Add tools to the agent: The agent now knows about the server's tools.
        agent.add_tools(*mcp.tools)
        
        # 3. Run the agent: Ask a question that requires the tool.
        query = "What's the weather like in Tokyo right now?"
        print(f"\nRunning agent with query: '{query}'")
        result = await agent.run(query)
        
        # The result will contain the text generated by the tool call.
        print("\nAgent Response:")
        print(result.data)

if __name__ == "__main__":
    # Ensure you have OPENAI_API_KEY set in your environment.
    asyncio.run(main())
```

When you run `agent_weather.py`, `MCPSession` starts `weather_server.py` in the background. The agent initializes a connection, calls `list_tools` to get the `get_weather` tool definition, and adds it to its context. When you call `agent.run()`, the LLM sees the tool, decides to use it, and `pydantic-ai` sends a `call_tool` request to the server.

---

## 5. Context 7: A Multi-Function Calculator MCP Tool

A server can expose many tools. This example demonstrates a `calculator` server offering both `add` and `subtract`. This shows how the agent can choose the correct tool based on the user's request.

### 5.1. The Calculator Server

```python
# calculator_server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - CALC-SERVER - %(levelname)s - %(message)s')

app = Server("calculator")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """Exposes multiple tools from a single server."""
    logging.info("list_tools() called by a client.")
    return [
        Tool(
            name="add",
            description="Calculates the sum of two integers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "The first integer"},
                    "b": {"type": "integer", "description": "The second integer"}
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="subtract",
            description="Calculates the difference between two integers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "The minuend"},
                    "b": {"type": "integer", "description": "The subtrahend"}
                },
                "required": ["a", "b"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Routes the call to the correct tool implementation."""
    logging.info(f"call_tool() invoked for tool: '{name}' with args: {arguments}")
    a = arguments.get('a')
    b = arguments.get('b')

    if name == "add":
        result = a + b
        return [TextContent(type="text", text=str(result))]
    
    if name == "subtract":
        result = a - b
        return [TextContent(type="text", text=str(result))]

    raise ValueError(f"Unknown tool: {name}")

if __name__ == "__main__":
    stdio_server(app)
```

### 5.2. The Agent Using the Calculator

The agent code is almost identical, but the prompt will now ask it to perform multiple, distinct calculations. The agent must make multiple, correct tool calls.

```python
# agent_calculator.py
import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPSession

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a calculation expert. You must use the provided tools to solve math problems."
)

async def main():
    print("Starting agent with MCPSession to connect to the calculator server...")
    
    async with MCPSession(["python", "calculator_server.py"]) as mcp:
        print(f"Discovered tools from server: {[tool.name for tool in mcp.tools]}")
        agent.add_tools(*mcp.tools)
        
        # This query requires two different tool calls.
        query = "Could you please calculate 2025 - 100, and also what is 55 + 45?"
        print(f"\nRunning agent with query: '{query}'")
        result = await agent.run(query)
        
        print("\nAgent Response:")
        print(result.data)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 6. Advanced Features & Best Practices

### 6.1. Tool Prefixing & Multiple Servers

What if two different servers both expose a tool named `get_details`? To avoid conflicts, you can apply a prefix when initializing the `MCPSession`.

```python
async with MCPSession(
    ["python", "weather_server.py"], tool_prefix="weather"
) as weather_mcp, MCPSession(
    ["python", "flight_server.py"], tool_prefix="flights"
) as flights_mcp:
    
    agent.add_tools(*weather_mcp.tools, *flights_mcp.tools)

# The agent now sees tools named 'weather_get_weather' and 'flights_get_details'.
# Pydantic-ai automatically maps calls back to the correct server.
```

### 6.2. Logging and Timeouts

Diagnosing issues with subprocesses can be tricky. `MCPSession` provides two helpful parameters:

*   `log_level`: Sets the logging level for the MCP client communication. Use `"debug"` for verbose output.
*   `timeout`: Sets a timeout in seconds for the server initialization. If the server doesn't respond to the `initialize` call within this time, it will raise an error.

```python
async with MCPSession(
    ["python", "server.py"], 
    log_level="debug",  # See detailed JSON-RPC messages
    timeout=10          # Fail fast if the server is stuck
) as mcp:
    ...
```

### 6.3. Sampling Support

Sometimes, a tool on the server might need to ask an LLM a question itself (e.g., to summarize a document before returning it). MCP supports this through a "sampling" callback. If your server needs this capability, you can enable it on the `pydantic-ai` side, allowing the server to use the agent's LLM. This is an advanced topic covered in the official `pydantic-ai` documentation.

---

## 7. Troubleshooting & Tips

*   **Validate Your Schema:** The most common source of errors is an incorrect `inputSchema` in your `Tool` definition. Ensure it is valid JSON Schema. `pydantic-ai` relies on this to generate type-safe validation models.
*   **Inspect Agent Knowledge:** During development, use `agent.describe_tools()` to see a formatted printout of all the tools the agent knows about. This is invaluable for debugging why a tool isn't being used.
*   **Use Different Transports:** While `stdio` is great for local subprocesses, the `mcp` library also supports HTTP for remote servers. This allows your agent to connect to tools running on different machines or in cloud services.

---

## 8. Further Reading

*   **PydanticAI MCP Documentation:** The official source for client-side integration. [https://ai.pydantic.dev/mcp/](https://ai.pydantic.dev/mcp/)
*   **PydanticAI MCP Announcement:** The launch article with more context. [https://pydantic.dev/articles/mcp-launch/](https://pydantic.dev/articles/mcp-launch)
*   **MCP Python Library:** The GitHub repository for the `mcp` server and client SDK.
*   **In-Depth Article:** A fantastic community blog post exploring the integration in detail. [https://saptak.in/writing/2025/04/01/building-powerful-ai-agents-with-pydantic-ai-and-mcp-servers/](https://saptak.in/writing/2025/04/01/building-powerful-ai-agents-with-pydantic-ai-and-mcp-servers/)

---
https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221RA9Qel0AfhGyunYsDWPyesabwvuvP2hi%22%5D,%22action%22:%22open%22,%22userId%22:%22108686197475781557359%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing
