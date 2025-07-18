# Context7 Terminal AI – Complete Troubleshooting Guide
> **From ImportError to Production-Ready in 5 Phases**  
> *A field-tested survival manual for Pydantic AI + MCP integration*

---

## 1. Executive Summary

| Phase | Error Encountered | Root Cause | Final Fix |
|-------|-------------------|------------|-----------|
| 1 | `ImportError: cannot import name 'UserPrompt'` | **Class renamed** in Pydantic AI v0.5+ | Use `UserPromptPart` (or simply remove unused import) |
| 2 | `AssertionError: Expected code to be unreachable` | **Manual dict construction** violates internal schema | Let `Agent.run()` handle messages natively |
| 3 | `TypeError: async for requires __aiter__` | **Mis-used streaming API** | Use `Agent.run()` with `message_history` |
| 4 | `TimeoutError` + `GeneratorExit` | **Mixed async backends** (`asyncio` vs `anyio`) | Unify on `anyio` |
| 5 | `KeyboardInterrupt` chaos | **No graceful shutdown** | Add `try/except` around CLI loop |

---

## 2. Phase-by-Phase Journey

### **Phase 1 – Import Hell**  
**Error:**  
```
ImportError: cannot import name 'UserPrompt' from 'pydantic_ai.messages'
```

**Diagnosis**  
- Pydantic AI v0.5+ **renamed** all message classes.  
- `UserPrompt` → `UserPromptPart`, `SystemPrompt` → `SystemPromptPart`, etc. 

**Solution**  
```python
# ❌ OLD (v0.4.x)
from pydantic_ai.messages import UserPrompt, SystemPrompt

# ✅ NEW (v0.5+)
# Simply remove the import; Agent.run() accepts dicts
messages = [{"role": "user", "content": "hello"}]
```

---

### **Phase 2 – AssertionError “Unreachable Code”**  
**Error:**  
```
AssertionError: Expected code to be unreachable, but got: {'role': 'user', ...}
```

**Diagnosis**  
- We were manually crafting message dicts with nested `parts` and `kind` fields.  
- Pydantic AI’s internal validator rejects these hand-rolled dicts.

**Solution**  
Stop building messages manually. Let `Agent.run()` accept plain dicts.

---

### **Phase 3 – Async Generator Confusion**  
**Error:**  
```
TypeError: 'async for' requires an object with __aiter__ method, got _AsyncGeneratorContextManager
```

**Diagnosis**  
- `agent.run_stream()` **is not** an async generator in v0.5+.  
- It returns a context manager that yields a single `RunResult`.

**Solution**  
Use `agent.run()` with `message_history` for stable streaming.

```python
result = await agent.run(
    message,
    message_history=[{"role": "user", "content": "prev"}]
)
yield result.data  # already the full string
```

---

### **Phase 4 – Event-Loop Wars**  
**Error:**  
```
TimeoutError + GeneratorExit + RuntimeError
```

**Diagnosis**  
- `pydantic-ai` uses **anyio** internally.  
- Mixing `asyncio.run()` with `anyio.to_thread.run_sync()` causes deadlocks.

**Solution**  
Unify the entire app under `anyio`.

```python
import anyio
anyio.run(main)
```

---

### **Phase 5 – Graceful Shutdown**  
**Symptom:** Ctrl-C raises ugly tracebacks.

**Solution**  
Wrap CLI loop in `try/except KeyboardInterrupt`.

---

## 3. TL;DR Checklist

| Check | Command |
|-------|---------|
| **Pin versions** | `pip install "pydantic-ai>=0.5,<0.6"` |
| **Import correctly** | *No manual `UserPrompt`, use dicts* |
| **Async runtime** | `anyio.run(cli.run)` |
| **Message format** | `messages = [{"role": "user", "content": "..."}]` |
| **Graceful exit** | `try/except KeyboardInterrupt` |

---

## 4. Quick Debug Commands

```bash
# 1. Verify installed versions
pip show pydantic-ai anyio openai

# 2. Check class signatures
python -c "from pydantic_ai.models.openai import OpenAIModel; help(OpenAIModel.__init__)"

# 3. Test MCP import
python -c "from pydantic_ai.mcp import MCPServerStdio; print('OK')"

# 4. Validate async backend
python -c "import anyio; print('anyio version:', anyio.__version__)"
```

---

## 5. Key Takeaways

1. **Read the source** – the `__init__` signature is the ultimate API doc .  
2. **Never mix async backends** – pick `anyio` or `asyncio`, not both .  
3. **Let the library do the work** – avoid DIY message schemas.  
4. **Pin dependencies** – prevents surprise API changes .  
5. **Handle Ctrl-C gracefully** – keeps UX polished.

---

## 6. References

- [Pydantic AI Troubleshooting](https://ai.pydantic.dev/troubleshooting/)   
- [Pydantic V2 Migration Guide](https://docs.pydantic.dev/latest/migration/)   
- [AnyIO Docs](https://anyio.readthedocs.io/en/stable/)  
- [OpenAI Python SDK](https://github.com/openai/openai-python)  

> *This guide was forged in the trenches of real-world debugging. Bookmark it, share it, and may your builds stay green.* 🚀
