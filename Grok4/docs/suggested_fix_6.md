Here’s a precise fix for the “part_kind” discriminator error. We need to build each message part with the correct `part_kind` and field name:

- System messages → part_kind “system-prompt” + `content`
- User messages → part_kind “user-prompt” + `content`
- Assistant/tool responses → part_kind “text” + `text`

Below is a line‐by‐line patch for just the helper, followed by the full updated `src/agent.py` so you can confirm no other logic drifted.

```diff
diff --git a/src/agent.py b/src/agent.py
index fedcba9..0acbd12 100644
--- a/src/agent.py
+++ b/src/agent.py
@@
-def _to_model_message_dict(raw_msgs: list[dict]) -> list[dict]:
-    """
-    Convert each {'role','content'} into the full shape that Pydantic-AI expects:
-      {
-        "kind": "request"|"response",
-        "role": str,
-        "parts": [ { "text": str } ]
-      }
-    """
-    out = []
-    for m in raw_msgs:
-        role = m["role"]
-        kind = "response" if role in ("assistant", "tool") else "request"
-        text = m["content"]
-        out.append({
-            "kind": kind,
-            "role": role,
-            "parts": [ {"text": text} ]
-        })
-    return out
+def _to_model_message_dict(raw_msgs: list[dict]) -> list[dict]:
+    """
+    Convert each {'role','content'} into the full shape Pydantic-AI expects,
+    with the correct part_kind discriminator on each part.
+    """
+    out: list[dict] = []
+    for m in raw_msgs:
+        role = m["role"]
+        content = m["content"]
+        if role == "system":
+            kind = "request"
+            part_kind = "system-prompt"
+            payload = {"content": content}
+        elif role == "user":
+            kind = "request"
+            part_kind = "user-prompt"
+            payload = {"content": content}
+        else:  # assistant or tool
+            kind = "response"
+            part_kind = "text"
+            payload = {"text": content}
+
+        part = {"part_kind": part_kind, **payload}
+        out.append({"kind": kind, "role": role, "parts": [part]})
+    return out
```

And here’s the full, updated `src/agent.py` (unchanged sections elided with `…` comments for brevity):

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

# Add project root to path
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
        # … (unchanged) …
        ...

    def query_mcp(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        # … (unchanged) …
        ...

    def generate_response(self, message: str, conversation: List[Dict]) -> str:
        intent = self.detect_intent(message)
        if intent == "search":
            # … (unchanged search code) …
            ...
        elif intent == "command":
            return self.handle_command(message)
        else:
            raw_msgs = conversation + [{"role": "user", "content": message}]
            model_msgs = _to_model_message_dict(raw_msgs)
            messages = ModelMessagesTypeAdapter.validate_python(model_msgs)
            ai_response = self.agent.run(messages)
            return ai_response

    def handle_command(self, command: str) -> str:
        # … (unchanged) …
        ...

    def preview_document(self, doc_id: int) -> str:
        # … (unchanged) …
        ...
```

With this change, each message part carries the required `"part_kind"` discriminator and the correct field name (`"content"` or `"text"`), satisfying the Pydantic schema. Rerun your CLI:

```bash
python3 src/cli.py
```

You should now clear the `part_kind` validation errors and enter normal conversation flow.
