The error is an `AttributeError: 'Agent' object has no attribute 'stream_chat'`, indicating that the `Agent` class from `pydantic_ai` lacks a `stream_chat` method, despite the code calling it in `stream_reply`.

This is likely a naming convention issue in the library (common in async AI frameworks like LangChain, where methods are prefixed with 'a' for async versions). The method probably exists as `astream_chat` to return an async iterator of events (e.g., `TextOutput` for text chunks and others for tool/MCP events).

### Fix
Update `stream_reply` in `src/agent.py` (starting at line 78) to use `astream_chat`:

```python
async def stream_reply(history: History) -> AsyncIterator[tuple[str, str]]:
    """
    Merge assistant tokens and MCP tool events into a single async stream.
    Yields tuples (role, payload)
      role == "assistant" -> markdown str
      role == "mcp"       -> already formatted message (string)
    """
    agent = create_agent()

    async for event in agent.astream_chat(
        messages=history.to_model_messages(),
        temperature=0.3,
        max_tokens=4096,
    ):
        if isinstance(event, TextOutput):
            yield "assistant", event.content
        else:
            yield "mcp", str(event)
```

### Why This Works
- The loop is already `async for`, so it needs an async iterable method.
- If `astream_chat` exists, it should yield the expected events (`TextOutput` for streamed text, others for MCP/tool outputs).
- This aligns with patterns in similar libraries (e.g., `agent.astream` or `chain.astream`).

Rerun `$ python3 -m src.cli` after the change. If the error persists (e.g., no `astream_chat` either), inspect the `pydantic_ai` library source/docs for the correct streaming method (possibly `stream`, `achat` with `stream=True`, or `generate_stream`). Update `pydantic_ai` if it's outdated, or consider a non-streaming fallback using `agent.chat` and manual chunking if needed.

