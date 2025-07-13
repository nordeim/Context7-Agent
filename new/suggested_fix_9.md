Looking at the error, it seems `pydantic_ai`'s `Agent.run()` method has a different API than what you're using. The error indicates it doesn't accept `messages`, `stream`, `temperature`, or `max_tokens` as direct arguments.

Based on the error traceback, here's how to fix your `stream_reply` function:

```python
async def stream_reply(history: History) -> AsyncIterator[tuple[str, str]]:
    """
    Merge assistant tokens and MCP tool events into a single async stream.
    Yields tuples (role, payload)
      role == "assistant" -> markdown str
      role == "mcp"       -> already formatted message (string)
    """
    agent = create_agent()
    
    # Get the messages from history
    messages = history.to_model_messages()
    
    # Get the last user message as the prompt
    user_prompt = None
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            user_prompt = msg.get('content')
            break
    
    if not user_prompt:
        return
    
    # Use message_history for previous messages (excluding the last user message)
    message_history = messages[:-1] if len(messages) > 1 else None
    
    # Model settings for temperature and max_tokens
    model_settings = {
        'temperature': 0.3,
        'max_tokens': 4096
    }
    
    # Call agent.run with the correct parameters
    async with agent.run_stream(
        user_prompt,
        message_history=message_history,
        model_settings=model_settings
    ) as stream:
        async for event in stream:
            if isinstance(event, TextOutput):
                yield "assistant", event.content
            else:
                yield "mcp", str(event)
```

However, I notice that `pydantic_ai` might use a different streaming API. Based on typical patterns, you might need to use something like:

```python
async def stream_reply(history: History) -> AsyncIterator[tuple[str, str]]:
    """
    Merge assistant tokens and MCP tool events into a single async stream.
    Yields tuples (role, payload)
      role == "assistant" -> markdown str
      role == "mcp"       -> already formatted message (string)
    """
    agent = create_agent()
    
    # Get the messages from history
    messages = history.to_model_messages()
    
    # Get the last user message as the prompt
    user_prompt = None
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            user_prompt = msg.get('content')
            break
    
    if not user_prompt:
        return
    
    # Try using the agent's streaming interface
    result = await agent.run(user_prompt)
    
    # If result is streamable, iterate over it
    if hasattr(result, '__aiter__'):
        async for event in result:
            if isinstance(event, TextOutput):
                yield "assistant", event.content
            else:
                yield "mcp", str(event)
    else:
        # Non-streaming response
        yield "assistant", str(result)
```

Since I don't have the exact `pydantic_ai` documentation, you'll need to check their docs for the correct API. Look for:
1. How to pass conversation history (might be `message_history` parameter)
2. How to enable streaming (might be a separate method like `run_stream()` or `stream()`)
3. How to pass model settings like temperature and max_tokens (likely through `model_settings` dict)

The key issue is that `pydantic_ai` has its own API conventions that differ from typical LLM libraries.

