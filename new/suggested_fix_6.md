The error is `AttributeError: 'Agent' object has no attribute 'astream_chat'`, meaning the previous guess of `astream_chat` was incorrect—the method doesn't exist on the `Agent` class.

Based on common conventions in AI agent libraries (e.g., LangChain's `astream` for async streaming with tool/event handling), the method is likely `astream`. It probably takes the messages as the first argument, with generation params as kwargs.

### Fix
Update `stream_reply` in `src/agent.py` (starting at line 78) to use `astream`:

```python
async def stream_reply(history: History) -> AsyncIterator[tuple[str, str]]:
    """
    Merge assistant tokens and MCP tool events into a single async stream.
    Yields tuples (role, payload)
      role == "assistant" -> markdown str
      role == "mcp"       -> already formatted message (string)
    """
    agent = create_agent()

    async for event in agent.astream(
        history.to_model_messages(),
        temperature=0.3,
        max_tokens=4096,
    ):
        if isinstance(event, TextOutput):
            yield "assistant", event.content
        else:
            yield "mcp", str(event)
```

### Why This Works
- Removes the `_chat` suffix to match likely library naming (e.g., `astream` for async streaming of chat-like responses with tools/events).
- Passes messages positionally (common in such APIs), with params as kwargs.
- Retains the event handling logic, assuming `astream` yields `TextOutput` for text chunks and other events for MCP/tool outputs.

Rerun `$ python3 -m src.cli`. If it fails (e.g., no `astream` or wrong args), try wrapping messages in a dict: `agent.astream({"messages": history.to_model_messages()}, temperature=0.3, max_tokens=4096)`. Check `pydantic_ai` docs/source for the exact method (possibly `arun`, `ainvoke`, or `generate` with `stream=True`). If non-streaming is needed as a fallback, replace with `event = await agent.achat(messages=..., temperature=..., max_tokens=...)` and yield once.

