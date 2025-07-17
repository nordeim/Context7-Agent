# **The Authoritative Guide to Production-Grade AI Agents with Pydantic-AI and MCP**

This guide provides a comprehensive, start-to-finish walkthrough for creating sophisticated, resilient, and production-ready Python AI agents. You will learn the modern architecture for building decoupled agentic systems using `pydantic-ai` and the **Model Context Protocol (MCP)**.

This is not just a "hello world" tutorial. This guide is forged from real-world debugging experience and is designed to teach you the patterns that lead to stable, maintainable applications.

**What You Will Master:**

1.  **The Decoupled Architecture:** Understand *why* separating your agent's logic from its tools is the superior approach for scalability and security.
2.  **Building an MCP Tool Server:** Create a standalone Python server to offer tools—like a multi-function calculator—to any compatible AI agent.
3.  **The Hybrid Agent Pattern:** Implement the definitive, battle-tested pattern for agent development. Your agent will robustly handle both complex tool-calling and simple conversational chat by intelligently using the right client for the right job.
4.  **Asynchronous Best Practices:** Learn to correctly manage the application's lifecycle using `anyio`, the concurrency backend used by `pydantic-ai`, to prevent subtle and hard-to-debug errors.
5.  **Developer Survival Guide:** Equip yourself with essential principles and commands to navigate the rapidly evolving landscape of AI libraries.

---

## 1. The Core Concept: The Decoupled Agent Architecture

In simple prototypes, developers often define tools as functions directly inside the agent's code. This monolithic approach is brittle and does not scale.

*   **The Old Way (Monolithic):** The agent and its tools are a single, tightly-coupled application. A change to a tool requires redeploying the entire agent. Tools must be written in the same language (Python).
*   **The Modern Way (Decoupled with MCP):** The system is split into two independent parts that communicate over a standardized protocol.
    *   **The MCP Server:** A simple, standalone application whose only job is to expose tools. It can be written in any language (Rust, Go, Node.js) and scaled independently.
    *   **The Agent (as MCP Client):** Your `pydantic-ai` agent connects to the server, asks what tools are available, and executes them by sending a request.

This architecture is modular, secure, and language-agnostic—the blueprint for professional-grade agentic systems.

---

## 2. Foundational Setup

Ensure you have **Python 3.10+** and an API key for your chosen LLM provider (e.g., OpenAI).

### 2.1. Installation

We strongly recommend using a virtual environment. The dependencies below include the agent framework, the MCP server library, and the required concurrency backend.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install all necessary libraries from requirements.txt
pip install "pydantic-ai>=0.4.0" "mcp>=0.4.0" "openai>=1.0.0" "anyio>=4.0.0"
```

*   `pydantic-ai`: The core agent framework.
*   `mcp`: The Python SDK for building MCP servers.
*   `openai`: The official client library for your LLM.
*   `anyio`: **Crucially**, this is the asynchronous framework used by `pydantic-ai`. Your application *must* use it to avoid concurrency conflicts.

---

## 3. Building the Tool Server: A Multi-Function Calculator

Our first step is to create the tool provider. This server will offer two tools, `add` and `subtract`, and will know nothing about the agent that will eventually use it.

```python
# calculator_server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import logging

# It's good practice to have logging in your server
logging.basicConfig(level=logging.INFO, format='%(asctime)s - CALC-SERVER - %(message)s')

# 1. Initialize the MCP Server with a unique name
app = Server("calculator")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """
    This function is called by clients to discover the tools this server provides.
    It returns a list of Tool objects, each with a detailed schema.
    """
    logging.info("A client is requesting the list of tools.")
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
    """
    This function executes a tool when a client requests it.
    It routes the call to the correct logic based on the tool 'name'.
    """
    logging.info(f"Client called tool '{name}' with arguments: {arguments}")
    a = arguments.get('a')
    b = arguments.get('b')

    if name == "add":
        result = a + b
        # The result must be wrapped in a Content object (e.g., TextContent)
        return [TextContent(type="text", text=str(result))]
    
    if name == "subtract":
        result = a - b
        return [TextContent(type="text", text=str(result))]

    raise ValueError(f"Unknown tool requested: {name}")

if __name__ == "__main__":
    # 3. Run the server using the stdio transport. This allows our agent
    # to run it as a clean subprocess.
    stdio_server(app)

```

This server is now ready to be run and used by our agent.

---

## 4. The Hybrid Agent: The Definitive Pattern for Robustness

Now we build the agent. We will implement the **Hybrid Pattern**, which is the key to a stable application.

**The Philosophy:** High-level abstractions like `agent.run()` are powerful for their intended purpose (like tool-calling) but can sometimes be fragile for other tasks (like simple chat). A robust application uses the abstraction for its strengths and bypasses it for everything else, using a more direct, stable API.

Our agent will therefore have **two clients**:
1.  **The `pydantic-ai` `Agent`:** We will use this to manage the MCP server's lifecycle and handle all tool-related queries.
2.  **A standard `openai.AsyncOpenAI` client:** We will use this *directly* for all non-tool-calling, conversational chat.

This separation isolates the two functionalities, making our application resilient and easier to debug.

```python
# hybrid_agent.py
import os
from typing import List, Dict

# Core Pydantic-AI imports for the agent and its components
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# The direct OpenAI client for our bypass strategy
import openai

# --- Configuration (replace with your own config loader) ---
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = "gpt-4o-mini"
    MCP_SERVER_CONFIG = {"command": "python", "args": ["calculator_server.py"]}
config = Config()
# ---

class HybridAgent:
    def __init__(self):
        """
        Initializes the agent using the Hybrid Pattern.
        """
        # --- Client 1: For Pydantic-AI Tool Calling ---
        # The provider handles synchronous communication details.
        provider = OpenAIProvider(api_key=config.OPENAI_API_KEY)
        
        # The LLM model wrapper, used by the Pydantic-AI Agent.
        llm = OpenAIModel(model_name=config.OPENAI_MODEL, provider=provider)
        
        # Define the MCP server the agent will connect to.
        mcp_server = MCPServerStdio(**config.MCP_SERVER_CONFIG)

        # The Pydantic-AI Agent is configured for tools.
        self.tool_agent = Agent(model=llm, mcp_servers=[mcp_server])

        # --- Client 2: For Direct, Stable Conversational Chat ---
        # A dedicated, standard async client. This is our bypass.
        self.chat_client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    def detect_intent(self, message: str) -> str:
        """A simple function to decide whether to use tools or just chat."""
        if "calculate" in message.lower() or "+" in message or "-" in message:
            return "tool_use"
        return "chat"

    async def generate_response(self, message: str, conversation_history: List[Dict]) -> str:
        """
        Generates a response by routing to the correct client based on intent.
        """
        intent = self.detect_intent(message)
        
        if intent == "tool_use":
            print("[Agent is using a tool...]")
            # Use the Pydantic-AI agent to handle the tool-related query.
            # The .run() method orchestrates the LLM call and the MCP communication.
            result = await self.tool_agent.run(message)
            return result.data

        else: # intent == "chat"
            print("[Agent is performing a direct chat...]")
            # Use our direct, stable chat client (the bypass).
            messages = conversation_history + [{"role": "user", "content": message}]
            response = await self.chat_client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
            )
            return response.choices[0].message.content

```

---

## 5. Bringing it All Together: The Application CLI

Finally, we create the command-line interface that runs our `HybridAgent`. This entry point correctly uses `anyio` for its event loop and the `pydantic-ai` agent's context manager to handle the tool server's lifecycle.

```python
# cli.py
import anyio
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from hybrid_agent import HybridAgent

console = Console()

class CLI:
    def __init__(self):
        self.agent = HybridAgent()
        self.conversation = []

    async def main_loop(self):
        console.print("[bold green]Hybrid Agent Initialized. Ask a math question or just chat![/]")
        console.print("Try: 'What is 55 + 45?' or 'Hello, who are you?'")
        
        while True:
            # Use a lambda to correctly pass keyword args in a thread
            user_input = await anyio.to_thread.run_sync(
                lambda: Prompt.ask("[bold cyan]You[/bold cyan]")
            )
            
            if user_input.lower() == "/exit":
                break

            # Generate the response using our hybrid agent
            response = await self.agent.generate_response(user_input, self.conversation)
            
            # Update history and display the response
            self.conversation.append({"role": "user", "content": user_input})
            self.conversation.append({"role": "assistant", "content": response})
            console.print(Panel(response, title="Agent", border_style="magenta"))

    async def run(self):
        """
        The main entry point, managing the MCP server's lifecycle.
        """
        # This context manager starts the MCP server when entered
        # and cleanly terminates it on exit.
        async with self.agent.tool_agent.run_mcp_servers():
            await self.main_loop()

if __name__ == "__main__":
    # The application MUST be run with anyio.
    anyio.run(CLI().run)

```

To run your agent, save the three files (`calculator_server.py`, `hybrid_agent.py`, `cli.py`) and execute:

```bash
# Make sure your OPENAI_API_KEY is set in your environment
export OPENAI_API_KEY="sk-..."

python cli.py
```

---

## 6. Developer's Survival Guide: Avoiding Pitfalls

The world of AI libraries is fast-moving. These principles will help you write stable code and debug effectively.

*   **Principle 1: The Source is the Truth.** Documentation and blog posts can become outdated. If you get a `TypeError` on a class constructor, your first step should be to inspect the installed library's source code. The `__init__` signature is the ultimate source of truth. Pin your dependencies in production (`pip freeze > requirements.txt`).

*   **Principle 2: Unify Your Async Backend.** This is not optional. `pydantic-ai` uses `anyio`. Your application must also use `anyio`. Mixing `anyio` and `asyncio` will lead to `TimeoutError` and other state corruption issues that are nearly impossible to debug.

*   **Principle 3: Interpret Tracebacks Intelligently.**
    *   **`TypeError`:** You are calling a function incorrectly. Fix the call.
    *   **`AttributeError`:** An object you expected to exist is `None` or of a different type. The error likely happened *before* the line that failed, during object creation.
    *   **`AssertionError` (from a library):** This is a critical warning. You are likely using the library in a way the authors did not intend. Do not try to satisfy the assertion; rethink your entire approach. This is what led to the discovery of the Hybrid Pattern.

*   **Principle 4: Embrace the Hybrid Pattern.** When a high-level abstraction is leaky or unstable for a specific use case, don't be afraid to bypass it. Using a direct, underlying client for simple tasks while leveraging the abstraction for its core strengths is a sign of a mature, pragmatic developer.

---
https://drive.google.com/file/d/1FLYoDd_1c5fNum_xTFxUZbOtTPifHbv8/view?usp=sharing, https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221RA9Qel0AfhGyunYsDWPyesabwvuvP2hi%22%5D,%22action%22:%22open%22,%22userId%22:%22108686197475781557359%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing, https://drive.google.com/file/d/1Rge-RBbQKH91OnCVcYGggTzm1v47Q3p1/view?usp=sharing, https://drive.google.com/file/d/1Ueg9XjXTmIYayAnH8hFvcRbpgQtxmX8F/view?usp=sharing

