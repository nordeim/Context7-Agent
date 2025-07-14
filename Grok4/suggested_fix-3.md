<think>
1. The immediate error is `AttributeError: 'MCPServerStdio' object has no attribute 'start'` (and similarly for `.terminate()`). Those methods were removed in the latest PydanticAI release.
2. MCPServerStdio is now managed via an async context manager (`agent.run_mcp_servers()`), so we must remove manual calls to `.start()` and `.terminate()`.
3. I will:
   a. Remove the `self.mcp_server.start()` call in `__init__`.
   b. Drop the `__del__` method entirely (no more manual cleanup).
   c. Leave the rest of the agent logic untouched.
4. I will produce a line‐by‐line diff to verify no other code was changed besides these two spots.
5. Then I will give you the complete updated `src/agent.py` file.
6. After that, I’ll recap the change and how to correctly start/stop your MCP server in your CLI (using `async with agent.run_mcp_servers():`).

Proceeding carefully now.
</think>

Here’s the minimal patch:

```diff
diff --git a/src/agent.py b/src/agent.py
index abcdef1..1234567 100644
--- a/src/agent.py
+++ b/src/agent.py
@@ class Context7Agent:
-        # Start MCP server
-        self.mcp_server.start()
+        # MCPServerStdio is managed by the agent.run_mcp_servers context manager

@@
-    def __del__(self):
-        # Gracefully stop the MCP server if it was successfully started
-        if hasattr(self, 'mcp_server'):
-            self.mcp_server.terminate()
+    # Note: MCPServerStdio lifecycle is handled by agent.run_mcp_servers(), 
+    # so no manual __del__ cleanup is needed.
```

And here is the complete, updated `src/agent.py`:

```python
import os
import sys
from typing import Dict, Any, Optional, List, Union

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel as OpenAI_LLM
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_core import to_jsonable_python
import openai  # Underlying lib for actual calls

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from history import History


class Context7Agent:
    """
    Context7 Agent implementation using Pydantic AI.

    This module implements a Pydantic AI agent with Context7 MCP server integration.
    The agent uses an OpenAI model with configuration from environment variables.
    """

    def __init__(self):
        """
        Initialize the Context7 Agent with configuration from environment variables.

        Sets up the OpenAI model with OpenAIProvider and Context7 MCP server integration.
        """
        error = config.validate()
        if error:
            raise ValueError(error)

        # Set up a unified LLM object that encapsulates the provider and model
        self.llm = OpenAI_LLM(
            model_name=config.openai_model,
            provider=OpenAIProvider(
                api_key=config.openai_api_key,
                base_url=config.openai_base_url
            )
        )

        # Set up MCP server integration using the specific syntax
        self.mcp_server = MCPServerStdio(**config.mcp_config["mcpServers"]["context7"])

        # Pydantic AI Agent is configured with the Model instance
        # MCPServerStdio lifecycle will be handled by agent.run_mcp_servers()
        self.agent = Agent(model=self.llm, mcp_servers=[self.mcp_server])

        # History integration
        self.history = History()

        # MCPServerStdio is managed by agent.run_mcp_servers(); no manual start() here.

    def detect_intent(self, message: str) -> str:
        """Detect if the message intends a search or command."""
        if "/search" in message or any(
            keyword in message.lower() for keyword in ["tell me about", "find docs on", "search for"]
        ):
            return "search"
        elif message.startswith("/"):
            return "command"
        return "chat"

    def query_mcp(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Query the Context7 MCP server for documents. (Mocked for demo; integrate real MCP calls.)"""
        # In reality, use self.mcp_server to send query and get contextual results
        mock_results = [
            {"id": 1, "title": f"Doc on {query}", "content": "Sample content...", "tags": ["ai"], "date": "2025-07-13"},
            {"id": 2, "title": f"Related to {query}", "content": "More info...", "tags": ["ethics"], "date": "2025-07-12"}
        ]
        self.history.add_search(query, mock_results)
        return mock_results

    def generate_response(self, message: str, conversation: List[Dict]) -> str:
        """Generate response using OpenAI via Pydantic AI."""
        intent = self.detect_intent(message)
        if intent == "search":
            search_query = message.split("about")[-1].strip() if "about" in message else message
            results = self.query_mcp(search_query)
            summary = f"Found {len(results)} docs: " + ", ".join([r['title'] for r in results])
            # Use OpenAI to enhance response via the provider client
            prompt = f"Summarize these search results for the user: {summary}"
            response = self.llm.provider.client.chat.completions.create(
                model=config.openai_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content + "\nUse /preview <id> to view."
        elif intent == "command":
            return self.handle_command(message)
        else:
            # Standard chat
            messages = ModelMessagesTypeAdapter.validate_python(
                conversation + [{"role": "user", "content": message}]
            )
            ai_response = self.agent.run(messages)
            return ai_response

    def handle_command(self, command: str) -> str:
        """Handle hotkey commands."""
        if command == "/help":
            return "Commands: /search <query>, /preview <id>, /bookmark <id>, /theme <name>, /analytics, /exit"
        return "Unknown command."

    def preview_document(self, doc_id: int) -> str:
        """Syntax-highlighted preview (simple text for now)."""
        docs = self.history.get_searches()[-1]["results"] if self.history.get_searches() else []
        for doc in docs:
            if doc["id"] == doc_id:
                return f"Preview: {doc['title']}\nContent: {doc['content']}"
        return "Doc not found."

    # MCPServerStdio lifecycle is managed by agent.run_mcp_servers(); no manual cleanup needed.
```

**Validation Steps:**
1. Run `python3 src/cli.py` again—no `start`/`terminate` errors.
2. When you want to actually spin up the MCP server, wrap your agent calls in:

   ```python
   import asyncio

   async def main():
       cli = Context7Agent()
       async with cli.agent.run_mcp_servers():
           # Now the stdio subprocess is running.
           # Call cli.generate_response(...) here as needed.
           pass

   asyncio.run(main())
   ```

With these changes, you’ll avoid the `AttributeError` and align with the latest Pydantic AI MCP API. Let me know if you need any further tweaks!

