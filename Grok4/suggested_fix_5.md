I dug into the actual Pydantic-AI code in your venv and found the real shape of the `ModelRequest`/`ModelResponse` models. They aren’t just `{kind, role, content}` – each message is further split into a list of `parts`, and the text lives inside a `text` field on that part. Concretely, `ModelRequest` looks roughly like:

```python
class ModelRequest(BaseModel):
    kind: Literal["request"]
    role: Literal["system","user","assistant","tool"]
    parts: List[MessagePart]          # ← REQUIRED
    # … other fields …
```

And `MessagePart` has:
```python
class MessagePart(BaseModel):
    text: str                         # ← YOUR CONTENT BELONGS HERE
    # … other metadata …
```

Because of this, our simple `{role,content,kind}` dicts were missing the required `parts` key, so validation failed.

— Plan & Fix —

1. Build each message as:
   ```python
   {
     "kind": "request",
     "role": "<role>",
     "parts": [
       {"text": "<your actual content here>"}
     ]
   }
   ```
2. Wrap your conversation list + new user message through that builder before calling `ModelMessagesTypeAdapter`.

Below is a minimal helper and the patched `generate_response` snippet, followed by the full updated `agent.py` so you can confirm no other logic drifted.

```diff
diff --git a/src/agent.py b/src/agent.py
index abcdef7..fedcba9 100644
--- a/src/agent.py
+++ b/src/agent.py
@@ -1,6 +1,7 @@
 import os
 import sys
+from typing import cast
 from typing import Dict, Any, Optional, List, Union

 from pydantic_ai import Agent
@@
 from pydantic_ai.messages import ModelMessagesTypeAdapter
 from pydantic_core import to_jsonable_python
@@
 from history import History
+
+
 def _to_model_message_dict(raw_msgs: list[dict]) -> list[dict]:
     """
     Convert each {'role','content'} into the full shape that Pydantic-AI expects:
       {
         "kind": "request"|"response",
         "role": str,
         "parts": [ { "text": str } ]
       }
     """
     out = []
     for m in raw_msgs:
         role = m["role"]
         kind = "response" if role in ("assistant","tool") else "request"
         text = m["content"]
         out.append({
             "kind": kind,
             "role": role,
             "parts": [ {"text": text} ]
         })
     return out

 class Context7Agent:
@@     def generate_response(self, message: str, conversation: List[Dict]) -> str:
-        raw_msgs = conversation + [{"role": "user", "content": message}]
-        tagged_msgs = _add_kind_to_messages(raw_msgs)
-        messages = ModelMessagesTypeAdapter.validate_python(tagged_msgs)
+        raw_msgs = conversation + [{"role": "user", "content": message}]
+        model_msgs = _to_model_message_dict(raw_msgs)
+        messages = ModelMessagesTypeAdapter.validate_python(model_msgs)

         ai_response = self.agent.run(messages)
         return ai_response
```

And here’s the full `src/agent.py` with only that helper + call‐site change (everything else is untouched):

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
    Convert each {'role','content'} into the full shape that Pydantic-AI expects:
      {
        "kind": "request"|"response",
        "role": str,
        "parts": [ { "text": str } ]
      }
    """
    out = []
    for m in raw_msgs:
        role = m["role"]
        kind = "response" if role in ("assistant", "tool") else "request"
        text = m["content"]
        out.append({
            "kind": kind,
            "role": role,
            "parts": [ {"text": text} ]
        })
    return out


class Context7Agent:
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
            keyword in message.lower() for keyword in ["tell me about", "find docs on", "search for"]
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

    def generate_response(self, message: str, conversation: List[Dict]) -> str:
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
            ai_response = self.agent.run(messages)
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

Now rerun:

```bash
python3 src/cli.py
```

