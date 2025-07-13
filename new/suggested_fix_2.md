### Error Analysis
The `TypeError` shows `MCPServerStdio.__init__()` requires two positional arguments: 'command' (likely the executable/path to the MCP server) and 'args' (a list of command-line arguments). Calling it without arguments (`MCPServerStdio()`) fails. The comment `# reads mcp.config.json` suggests it interacts with a config file, but the class still needs these args to spawn a process via stdio.

### Fix
Update `_build_mcp` in `src/agent.py` to provide the required arguments. Use `settings.mcp_alias` as the 'command' (assuming it's the server executable/alias), and an empty list for 'args' (or add config-related args if needed, e.g., `['--config', 'mcp.config.json']`):

```python
def _build_mcp() -> MCPServerStdio:
    return MCPServerStdio(command=settings.mcp_alias, args=[])  # Adjust args if config loading requires flags
```

If the signature expects positional args without keywords, use:
```python
def _build_mcp() -> MCPServerStdio:
    return MCPServerStdio(settings.mcp_alias, [])  # reads mcp.config.json via args if needed
```

### Why This Works
- Supplies the missing 'command' and 'args', matching the init signature.
- Leverages `settings.mcp_alias` as the command, per prior code intent.
- If config loading fails, add args like `['--config', 'mcp.config.json']` and check `pydantic_ai.mcp` docs for details.
- Restart the CLI; it should now create the agent, invoke `MCP.search("n8n json")` or similar, and stream results. Verify `mcp.config.json` exists and `settings.mcp_alias` is correctly set (e.g., a valid executable).

---
### Error Analysis
The `TypeError` indicates `MCPServerStdio.__init__()` does not accept a 'server' keyword argument. Based on the code comment (`# reads mcp.config.json`), this class likely initializes without arguments and loads config from a file (e.g., `mcp.config.json`). The 'server' param is extraneous and causing the failure during agent creation.

### Fix
Update `src/agent.py` by removing the 'server' argument from `MCPServerStdio` init:

```python
def _build_mcp() -> MCPServerStdio:
    return MCPServerStdio()  # reads mcp.config.json
```

### Why This Works
- `MCPServerStdio` auto-loads from config, per the comment.
- This avoids the invalid keyword error.
- Ensure `mcp.config.json` exists and is correctly set up (e.g., with any alias or server details if needed internally).
- Restart the CLI; the agent should now initialize, process "n8n json format" (likely via `MCP.search("n8n json format")`), and stream a response in markdown. If config issues arise, verify the JSON file's contents against `pydantic_ai.mcp` docs.

