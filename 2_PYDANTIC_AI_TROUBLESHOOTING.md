# Pydantic-AI & MCP Server – The Definitive Cheat-Sheet
> **Battle-tested recipes for a production-grade terminal AI agent**  
> *Context7 case study distilled into one page*

---

## 0. Golden Rules (memorise these)

1. **Read the source** – `__init__` signatures > docs.
2. **Pin versions** – `pip install "pydantic-ai>=0.5,<0.6"`.
3. **Use `anyio`** everywhere – never mix with `asyncio`.
4. **Let the library speak** – stop hand-crafting message dicts.
5. **Graceful exits** – wrap your CLI loop in `try/except KeyboardInterrupt`.

---

## 1. Quick-Fix Matrix

| Symptom | Root Cause | One-Line Fix |
|---------|------------|--------------|
| `cannot import name 'UserPrompt'` | class renamed | **Delete the import** – use plain dicts |
| `AssertionError: Expected code to be unreachable` | manual dict schema | **Stop DIY message objects** |
| `'async for' requires __aiter__` | wrong streaming API | **Use `agent.run(...)` not `run_stream()`** |
| `TimeoutError + GeneratorExit` | mixed async backends | **Replace `asyncio.run` with `anyio.run`** |
| `KeyboardInterrupt` traceback | no graceful shutdown | **Wrap CLI loop in `try/except`** |

---

## 2. Production-Ready Skeleton

```python
# src/agent.py
import anyio, openai
from pydantic_ai import Agent
from pydantic_ai.models.openai  import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStdio

class ProdAgent:
    def __init__(self, api_key: str, model: str="gpt-4o-mini"):
        # 1. Sync client via provider (for MCP tools)
        self.provider = OpenAIProvider(api_key=api_key)

        # 2. Async client for fast chat
        self.openai = openai.AsyncOpenAI(api_key=api_key)

        # 3. Pydantic-AI model
        self.llm = OpenAIModel(model_name=model, provider=self.provider)

        # 4. Agent with MCP
        self.agent = Agent(
            model=self.llm,
            mcp_servers=[MCPServerStdio(command="npx", args=["-y", "@upstash/context7-mcp"])]
        )

    async def chat(self, user_text: str, history: list[dict]=None):
        history = history or []
        async with self.agent.run_mcp_servers():
            result = await self.agent.run(user_text, message_history=history)
            return str(result.data)
```

---

## 3. CLI Entry Pattern

```python
# src/cli.py
import anyio
from rich.prompt import Prompt

async def main():
    agent = ProdAgent(api_key="sk-...")
    while True:
        user = await anyio.to_thread.run_sync(
            lambda: Prompt.ask("[bold cyan]You[/bold cyan]")
        )
        if user.lower() == "/exit":
            break
        reply = await agent.chat(user)
        print(reply)

if __name__ == "__main__":
    anyio.run(main)
```

---

## 4. Pre-flight Checklist

```bash
pip install "pydantic-ai>=0.5,<0.6" anyio openai rich
```

| Step | Command |
|------|---------|
| 1. Check versions | `pip show pydantic-ai` |
| 2. Verify MCP | `python -c "from pydantic_ai.mcp import MCPServerStdio; print('OK')"` |
| 3. Test async | `python -c "import anyio; print('anyio version:', anyio.__version__)"` |
| 4. Pin deps | `pip freeze > requirements.txt` |

---

## 5. Debug Flow Chart

```
Error appears
├─ TypeError on constructor → fix signature per matrix
├─ AssertionError → stop manual dict construction
├─ Async issues → switch to `anyio.run`
└─ Other → check installed source code
```

---

## 6. Common Pitfalls & One-Liner Fixes

| Pitfall | One-liner |
|---------|-----------|
| Passing `model=` to `OpenAIProvider` | **Remove it** – model goes to `Agent` |
| Using `asyncio.run` with `anyio.to_thread` | **Replace with `anyio.run`** |
| Hand-building message dicts | **Just use dicts with `role`/`content`** |
| Not handling Ctrl-C | **Wrap CLI loop in `try/except KeyboardInterrupt`** |

---

## 7. Emergency Bypass (when abstractions leak)

```python
# Fallback direct OpenAI call
response = await openai.AsyncOpenAI().chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
return response.choices[0].message.content
```

---

## 8. Reference Links

- [Pydantic AI Docs](https://ai.pydantic.dev)  
- [AnyIO Guide](https://anyio.readthedocs.io)  
- [OpenAI Python SDK](https://github.com/openai/openai-python)  
- [MCP Spec](https://github.com/modelcontextprotocol/spec)

---

## 9. TL;DR Cheat-Sheet

| Task | Code snippet |
|------|--------------|
| **Create Agent** | `Agent(model=OpenAIModel(...), mcp_servers=[...])` |
| **Run with MCP** | `async with agent.run_mcp_servers(): result = await agent.run(...)` |
| **Chat loop** | `anyio.run(cli.run)` |
| **Graceful exit** | `try/except KeyboardInterrupt` |

> Keep this sheet pinned; it is the distilled wisdom of the trenches.
