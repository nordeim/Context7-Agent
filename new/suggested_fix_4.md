The error stems from a keyword argument mismatch in the `Agent` constructor. The `pydantic_ai` library expects `mcp_servers` (plural, likely a tuple or iterable for multiple servers), but the code passes `mcp_server` (singular). This triggers `UserError: Unknown keyword arguments: `mcp_server`` during validation.

### Fix
Update `create_agent` in `src/agent.py` (line 66) to use the plural form and wrap the single server in a tuple:

```python
def create_agent() -> Agent:
    return Agent(model=_build_llm(), mcp_servers=(_build_mcp(),), system_prompt=SYSTEM_PROMPT)
```

### Why This Works
- The traceback shows `mcp_servers=()` in the `Agent.__init__` locals, confirming it's a supported parameter.
- Passing a tuple with one `MCPServerStdio` instance adapts to the plural expectation without changing the single-server logic.
- If the library requires a list instead, use `mcp_servers=[_build_mcp()]`—test both if needed.

Rerun `$ python3 -m src.cli` after the change. If issues persist (e.g., version mismatch), ensure `pydantic_ai` is up-to-date or check its docs for `Agent` params.
