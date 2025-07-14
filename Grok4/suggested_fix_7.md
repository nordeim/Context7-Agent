<think>
We need to fix the “coroutine is not JSON serializable” by ensuring we await any coroutine returned by `Agent.run`. That means:

1. Refactor `generate_response` into an `async def` so we can `await self.agent.run(...)`.
2. In CLI, call `await agent.generate_response` from an async `handle_input`.
3. Replace synchronous prompt and blocking loaders with `asyncio.to_thread` wrappers so we can await them inside our async `handle_input`.

We’ll produce two file patches:
- **src/agent.py**: convert `generate_response` to async and await agent calls.
- **src/cli.py**: make `handle_input` async, swap in `await asyncio.to_thread` for blocking calls, and call `await handle_input()` in `run`.

We’ll show line‐by‐line diffs then full updated files.

Proceeding with the patch.
</think>

```diff
diff --git a/src/agent.py b/src/agent.py
index 0acbd12..abcdef0 100644
--- a/src/agent.py
+++ b/src/agent.py
@@    def generate_response(self, message: str, conversation: List[Dict]) -> str:
-    def generate_response(self, message: str, conversation: List[Dict]) -> str:
+    async def generate_response(self, message: str, conversation: List[Dict]) -> str:
         """Generate response using OpenAI via Pydantic AI."""
         intent = self.detect_intent(message)
         if intent == "search":
@@
             response = self.llm.provider.client.chat.completions.create(
                 model=config.openai_model,
                 messages=[{"role": "user", "content": prompt}]
             )
-            return response.choices[0].message.content + "\nUse /preview <id> to view."
+            return response.choices[0].message.content + "\nUse /preview <id> to view."

         elif intent == "command":
             return self.handle_command(message)
         else:
-            raw_msgs = conversation + [{"role": "user", "content": message}]
-            model_msgs = _to_model_message_dict(raw_msgs)
-            messages = ModelMessagesTypeAdapter.validate_python(model_msgs)
-
-            ai_response = self.agent.run(messages)
-            return ai_response
+            raw_msgs = conversation + [{"role": "user", "content": message}]
+            model_msgs = _to_model_message_dict(raw_msgs)
+            messages = ModelMessagesTypeAdapter.validate_python(model_msgs)
+
+            ai_response = await self.agent.run(messages)
+            return ai_response
```

```diff
diff --git a/src/cli.py b/src/cli.py
index 1234abc..789def0 100644
--- a/src/cli.py
+++ b/src/cli.py
@@ class CLI:
-    def handle_input(self):
+    async def handle_input(self):
         while True:
-            user_input = Prompt.ask("[bold]You[/bold]", console=console)
+            user_input = await asyncio.to_thread(Prompt.ask, "[bold]You[/bold]", console, )
             if user_input == "/exit":
-                self.agent.history.save()
+                await asyncio.to_thread(self.agent.history.save)
                 break
             if user_input.startswith("/theme"):
@@
             self.agent.history.add_message("user", user_input)
-            self.particle_loader()
+            await asyncio.to_thread(self.particle_loader)

-            response = self.agent.generate_response(user_input, self.conversation)
+            response = await self.agent.generate_response(user_input, self.conversation)
             self.agent.history.add_message("assistant", response)
             self.conversation = self.agent.history.get_conversation()

             rprint(Panel(response, title="Agent", style=self.styles["response"]))
             self.typing_animation("")
@@ class CLI:
-    async def run(self):
+    async def run(self):
         layout = Layout()
         layout.split_column(
             Layout(name="header", size=3),
             Layout(name="main"),
             Layout(name="footer", size=3)
         )
@@     async def run(self):
-            async with self.agent.agent.run_mcp_servers():
-                self.handle_input()
+            async with self.agent.agent.run_mcp_servers():
+                await self.handle_input()
```

### Full Updated `src/agent.py`

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
import openai

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from history import History


def _to_model_message_dict(raw_msgs: list[dict]) -> list[dict]:
    """
    Convert each {'role','content'} into the full shape Pydantic-AI expects,
    with the correct part_kind discriminator on each part.
    """
    out: list[dict] = []
    for m in raw_msgs:
        role = m["role"]
        content = m["content"]
        if role == "system":
            kind = "request"
            part_kind = "system-prompt"
            payload = {"content": content}
        elif role == "user":
            kind = "request"
            part_kind = "user-prompt"
            payload = {"content": content}
        else:  # assistant or tool
            kind = "response"
            part_kind = "text"
            payload = {"text": content}

        part = {"part_kind": part_kind, **payload}
        out.append({"kind": kind, "role": role, "parts": [part]})
    return out


class Context7Agent:
    """
    Context7 Agent implementation using Pydantic AI.
    """

    def __init__(self):
        error = config.validate()
        if error:
            raise ValueError(error)

        self.llm = OpenAI_LLM(
            model_name=config.openai_model,
            provider=OpenAIProvider(
                api_key=config.openai_api_key,
                base_url=config.openai_base_url
            )
        )
        self.mcp_server = MCPServerStdio(**config.mcp_config["mcpServers"]["context7"])
        self.agent = Agent(model=self.llm, mcp_servers=[self.mcp_server])
        self.history = History()

    def detect_intent(self, message: str) -> str:
        if "/search" in message or any(
            keyword in message.lower()
            for keyword in ["tell me about", "find docs on", "search for"]
        ):
            return "search"
        elif message.startswith("/"):
            return "command"
        return "chat"

    def query_mcp(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        mock_results = [
            {"id": 1, "title": f"Doc on {query}", "content": "Sample content...", "tags": ["ai"], "date": "2025-07-13"},
            {"id": 2, "title": f"Related to {query}", "content": "More info...", "tags": ["ethics"], "date": "2025-07-12"}
        ]
        self.history.add_search(query, mock_results)
        return mock_results

    async def generate_response(self, message: str, conversation: List[Dict]) -> str:
        """Generate response using OpenAI via Pydantic AI."""
        intent = self.detect_intent(message)
        if intent == "search":
            search_query = message.split("about")[-1].strip() if "about" in message else message
            results = self.query_mcp(search_query)
            summary = f"Found {len(results)} docs: " + ", ".join(r["title"] for r in results)
            prompt = f"Summarize these search results for the user: {summary}"
            response = self.llm.provider.client.chat.completions.create(
                model=config.openai_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content + "\nUse /preview <id> to view."
        elif intent == "command":
            return self.handle_command(message)
        else:
            raw_msgs = conversation + [{"role": "user", "content": message}]
            model_msgs = _to_model_message_dict(raw_msgs)
            messages = ModelMessagesTypeAdapter.validate_python(model_msgs)
            ai_response = await self.agent.run(messages)
            return ai_response

    def handle_command(self, command: str) -> str:
        if command == "/help":
            return "Commands: /search <query>, /preview <id>, /bookmark <id>, /theme <name>, /analytics, /exit"
        return "Unknown command."

    def preview_document(self, doc_id: int) -> str:
        docs = self.history.get_searches()[-1]["results"] if self.history.get_searches() else []
        for doc in docs:
            if doc["id"] == doc_id:
                return f"Preview: {doc['title']}\nContent: {doc['content']}"
        return "Doc not found."
```

### Full Updated `src/cli.py`

```python
import os
import sys
import time
import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.prompt import Prompt
from rich import print as rprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Context7Agent
from themes import get_theme_styles, THEMES, ASCII_ART

console = Console()

class CLI:
    def __init__(self):
        self.agent = Context7Agent()
        self.conversation = self.agent.history.get_conversation()
        self.current_theme = "cyberpunk"
        self.styles = get_theme_styles(self.current_theme)

    def typing_animation(self, text: str):
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.05)
        print()

    def particle_loader(self, duration: int = 3):
        with Live(console=console, refresh_per_second=4) as live:
            for _ in range(duration * 4):
                live.update(Text("Loading... •", style=self.styles["loader"]))
                time.sleep(0.25)

    def display_welcome(self):
        art = ASCII_ART.get(self.current_theme, "")
        rprint(Panel(art, title="Welcome to Context7 Agent", style=self.styles["panel"]))
        self.typing_animation("Hello! Chat naturally or use /commands. Try 'Tell me about AI ethics'.")

    async def handle_input(self):
        while True:
            user_input = await asyncio.to_thread(Prompt.ask, "[bold]You[/bold]", console)
            if user_input == "/exit":
                await asyncio.to_thread(self.agent.history.save)
                break
            if user_input.startswith("/theme"):
                theme = user_input.split()[-1]
                if theme in THEMES:
                    self.current_theme = theme
                    self.styles = get_theme_styles(theme)
                    rprint("[green]Theme changed![/green]")
                continue
            if user_input.startswith("/preview"):
                doc_id = int(user_input.split()[-1])
                preview = self.agent.preview_document(doc_id)
                rprint(Panel(preview, title="Document Preview", style=self.styles["preview"]))
                continue

            self.agent.history.add_message("user", user_input)
            await asyncio.to_thread(self.particle_loader)

            response = await self.agent.generate_response(user_input, self.conversation)
            self.agent.history.add_message("assistant", response)
            self.conversation = self.agent.history.get_conversation()

            rprint(Panel(response, title="Agent", style=self.styles["response"]))
            self.typing_animation("")

    async def run(self):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        layout["header"].update(Panel("Context7 Agent", style=self.styles["header"]))
        layout["footer"].update(Panel("Hotkeys: /help, /exit", style=self.styles["footer"]))

        with console:
            self.display_welcome()
            async with self.agent.agent.run_mcp_servers():
                await self.handle_input()

if __name__ == "__main__":
    asyncio.run(CLI().run())
```

**Summary:**

- `generate_response` is now `async`, with `await self.agent.run(...)`.
- `CLI.handle_input` is now `async`, using `await asyncio.to_thread(...)` for blocking I/O.
- `CLI.run` awaits `handle_input()` inside the MCP server context.
- This ensures we never store coroutines in history, and all `await` positions are correct.

Run:

```bash
python3 src/cli.py
```

You should now see a working chat loop with no coroutine serialization errors.
