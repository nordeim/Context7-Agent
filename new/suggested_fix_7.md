The error is `AttributeError: 'Agent' object has no attribute 'astream'`, indicating the method doesn't exist. Based on patterns in AI libraries (e.g., OpenAI's async client where `create(stream=True)` returns an async iterator of chunks/events), the `Agent` likely supports streaming via an `achat` method with a `stream=True` parameter.

### Fix
Update `stream_reply` in `src/agent.py` (starting at line 78) to use `achat` with `stream=True`:

```python
async def stream_reply(history: History) -> AsyncIterator[tuple[str, str]]:
    """
    Merge assistant tokens and MCP tool events into a single async stream.
    Yields tuples (role, payload)
      role == "assistant" -> markdown str
      role == "mcp"       -> already formatted message (string)
    """
    agent = create_agent()

    async for event in agent.achat(
        messages=history.to_model_messages(),
        temperature=0.3,
        max_tokens=4096,
        stream=True,
    ):
        if isinstance(event, TextOutput):
            yield "assistant", event.content
        else:
            yield "mcp", str(event)
```

### Why This Works
- `achat` is the async chat method (common naming for async operations).
- Adding `stream=True` turns it into a streaming response, yielding events like `TextOutput` for text chunks and others for MCP/tool outputs.
- Matches the existing `async for` loop and event handling.

Rerun `$ python3 -m src.cli`. If it fails (e.g., no `achat` or `stream` not supported), try `agent.chat` (sync version) wrapped in `asyncio.to_thread` for async compatibility, or check `pydantic_ai` source/docs for the exact streaming API (possibly `ainvoke` with `stream=True` or `generate_stream`).
