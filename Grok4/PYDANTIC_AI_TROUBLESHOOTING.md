# Pydantic-AI & MCP Server Troubleshooting Guide

Quick reference for resolving common Pydantic-AI integration issues based on real-world debugging experience.

## 🔍 Error Pattern Quick Reference

### Initialization Errors

| Error Message | Cause | Solution |
|---------------|--------|----------|
| `TypeError: OpenAIModel.__init__() got unexpected keyword argument 'model'` | API signature changed - model parameter moved | Use `model_name` parameter in `OpenAIModel` constructor |
| `TypeError: OpenAIProvider.__init__() got unexpected keyword argument 'client'` | Provider doesn't accept pre-initialized client | Pass `api_key` and `base_url` directly |
| `TypeError: OpenAIProvider.__init__() got unexpected keyword argument 'model'` | Model parameter belongs to Agent, not Provider | Remove model from Provider, use in Agent |
| `TypeError: MCPServerStdio.__init__() missing 1 required positional argument: 'args'` | Missing argument unpacking | Use `**config_dict` to unpack configuration |
| `UserError: Unknown keyword arguments: 'provider', 'llm_model'` | Agent constructor signature changed | Use `model` parameter instead of `llm_model` |

### Import/Module Errors

| Error Message | Cause | Solution |
|---------------|--------|----------|
| `ModuleNotFoundError: No module named 'pydantic_ai.llm'` | Package reorganization | Import from `pydantic_ai.models.openai` |
| `ImportError: cannot import name 'OpenAI'` | Class renamed | Use `OpenAIModel` instead of `OpenAI` |

### MCP Server Lifecycle Errors

| Error Message | Cause | Solution |
|---------------|--------|----------|
| `AttributeError: 'MCPServerStdio' object has no attribute 'start'` | Manual lifecycle deprecated | Use async context manager: `async with agent.run_mcp_servers()` |
| `AttributeError: 'MCPServerStdio' object has no attribute 'terminate'` | Manual cleanup deprecated | Let Agent manage lifecycle via context manager |

### Message Formatting Errors

| Error Message | Cause | Solution |
|---------------|--------|----------|
| `ValidationError: Missing 'kind' discriminator` | Incorrect message structure | Use Pydantic model objects, not dicts |
| `ValidationError: Missing 'parts' field` | Message format too simplistic | Use proper `UserPromptPart`, `SystemPromptPart` objects |
| `ValidationError: Missing 'part_kind' discriminator` | Nested structure missing | Use library's internal message classes |
| `AssertionError: Expected code to be unreachable` | Manual dict creation incorrect | Use official Pydantic model objects |

### Async Runtime Errors

| Error Message | Cause | Solution |
|---------------|--------|----------|
| `TimeoutError` + `GeneratorExit` + `RuntimeError` | Async backend conflict | Unify on `anyio` (not `asyncio`) |
| `TypeError: anyio.to_thread.run_sync() got unexpected keyword argument 'console'` | Wrong async API usage | Use lambda wrapper: `lambda: func(arg, kwarg=value)` |
| `TypeError: PromptBase.ask() takes from 1 to 2 positional arguments but 3 were given` | Incorrect threading call | Use keyword arguments properly in thread wrapper |

## 🛠️ Defensive Initialization Pattern

```python
# Safe initialization pattern based on debugging experience
import openai
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

class SafeAgent:
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", base_url: str = None):
        # 1. Provider for sync operations
        self.provider = OpenAIProvider(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1"
        )
        
        # 2. Dedicated async client for async operations
        self.async_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1"
        )
        
        # 3. Model for pydantic-ai integration
        self.llm = OpenAIModel(
            model_name=model_name,
            provider=self.provider
        )
        
        # 4. Agent with MCP servers
        self.agent = Agent(
            model=self.llm,
            mcp_servers=[MCPServerStdio(**mcp_config)]
        )
```

## 📊 Version Compatibility Matrix

| Pydantic-AI Version | OpenAI Model Class | Provider Pattern | Recommended Approach |
|---------------------|--------------------|------------------|---------------------|
| 0.4.x and earlier | `OpenAI` | Direct client | Legacy - avoid |
| 0.5.x+ | `OpenAIModel` | With `OpenAIProvider` | Use current pattern |
| 0.6.x+ (future) | Check source code | Check source code | Always verify |

## 🔄 Async Runtime Best Practices

### 1. Unify Async Backend
```python
# ❌ Don't mix async backends
import asyncio
# ... later use anyio.to_thread.run_sync()

# ✅ Use anyio consistently
import anyio
# ... use anyio.to_thread.run_sync()
# ... use anyio.run() as entry point
```

### 2. Context Manager Pattern
```python
# ✅ Proper MCP server lifecycle
async def run(self):
    async with self.agent.run_mcp_servers():
        await self.main_loop()

# Entry point
if __name__ == "__main__":
    anyio.run(MyApp().run)
```

### 3. Threading Safe Calls
```python
# ✅ Safe blocking calls in async context
user_input = await anyio.to_thread.run_sync(
    lambda: Prompt.ask("[bold]You[/bold]", console=console)
)
```

## 🎯 Message Formatting Checklist

When sending messages to Pydantic-AI:

- [ ] **Don't use raw dictionaries** - Use Pydantic model objects
- [ ] **Use proper message types**:
  - `UserPromptPart` for user messages
  - `SystemPromptPart` for system messages
  - `TextPart` for text content
- [ ] **Avoid manual schema construction** - Let the library handle validation
- [ ] **Test with simple messages first** - Verify basic functionality

## 🔧 Quick Debug Commands

```bash
# Check installed versions
pip show pydantic-ai openai anyio

# Verify class signatures
python -c "from pydantic_ai.models.openai import OpenAIModel; help(OpenAIModel.__init__)"

# Test MCP server connection
python -c "from pydantic_ai.mcp import MCPServerStdio; print('MCP server import OK')"

# Check async backend compatibility
python -c "import anyio; print('anyio version:', anyio.__version__)"
```

## 🚨 Emergency Bypass Strategy

When Pydantic-AI abstractions fail:

1. **Identify the stable layer** - Usually the underlying OpenAI client
2. **Bypass problematic methods** - Use direct client calls
3. **Maintain MCP integration** - Keep using Agent for server lifecycle
4. **Document the workaround** - Note why bypass was necessary

```python
# Emergency bypass example
async def generate_response_bypass(self, messages):
    # When agent.run() fails, use direct OpenAI client
    response = await self.async_client.chat.completions.create(
        model=self.model_name,
        messages=messages
    )
    return response.choices[0].message.content
```

## 📋 Pre-flight Checklist

Before running Pydantic-AI applications:

- [ ] Verify all imports work without errors
- [ ] Check constructor signatures match current API
- [ ] Ensure async backend consistency (anyio)
- [ ] Test MCP server lifecycle with context manager
- [ ] Validate message formatting with simple test
- [ ] Confirm environment variables are set
- [ ] Test file I/O error handling (history.json)

## 🔍 Common Log Patterns

Watch for these patterns in logs:

- `"TypeError: ... got unexpected keyword argument"` → API drift
- `"ValidationError: Missing 'kind' discriminator"` → Message format issue
- `"AttributeError: ... has no attribute 'start'"` → MCP lifecycle change
- `"TimeoutError"` + `"GeneratorExit"` → Async backend conflict
- `"AssertionError: Expected code to be unreachable"` → Abstraction misuse

## 📚 Reference Commands

```bash
# Get current API signatures
python -c "
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStdio
print('Agent:', Agent.__init__.__annotations__)
print('OpenAIModel:', OpenAIModel.__init__.__annotations__)
print('OpenAIProvider:', OpenAIProvider.__init__.__annotations__)
"
```

Remember: When in doubt, check the source code. The installed library is the ultimate source of truth.