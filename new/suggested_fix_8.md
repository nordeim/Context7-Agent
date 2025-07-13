Looking at your error, the issue is that `agent.run()` returns a coroutine that needs to be awaited to get the async iterator. You're trying to iterate directly over the coroutine instead of the result.

Here's the fix for your `stream_reply` function:

```python
async def stream_reply(history: History) -> AsyncIterator[tuple[str, str]]:
    """
    Merge assistant tokens and MCP tool events into a single async stream.
    Yields tuples (role, payload)
      role == "assistant" -> markdown str
      role == "mcp"       -> already formatted message (string)
    """
    agent = create_agent()

    # Await the coroutine to get the async iterator
    stream = await agent.run(
        messages=history.to_model_messages(),
        stream=True,
        temperature=0.3,
        max_tokens=4096,
    )
    
    # Now iterate over the stream
    async for event in stream:
        if isinstance(event, TextOutput):
            yield "assistant", event.content
        else:
            yield "mcp", str(event)
```

The key change is splitting the operation into two steps:
1. First await `agent.run()` to get the stream object
2. Then iterate over that stream with `async for`

Alternatively, if `pydantic_ai` expects you to use `agent.run()` differently for streaming, you might need to check their documentation for the correct streaming API. Some libraries use a pattern like `agent.run_stream()` or `agent.stream()` for streaming responses.
