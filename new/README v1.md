# ⚡ Context7 Agent

An immersive conversational librarian bridging OpenAI and the Context7 **MCP** server.

## ✨ Features
– Stunning Rich-TUI (4 themes, animations, ASCII banners)  
– AI-powered contextual search (meaning-based, fuzzy) through Context7 MCP  
– Bookmarks, history, hotkeys, live streaming responses  

## 🚀 Quick Start

```bash
git clone https://github.com/you/context7-agent
cd context7-agent

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # add your OpenAI key

# Boot the chat!
python -m src.cli chat
```

The first run spawns the Context7 MCP server via `npx @upstash/context7-mcp@latest` (configured in `mcp.config.json`).  
All chat history is saved to `.history.json`.

## 🛠️ Configuration

Env | Default | Description
----|---------|------------
`OPENAI_API_KEY` | — | Your OpenAI token
`OPENAI_MODEL` | gpt-4o-mini | Any supported model
`OPENAI_BASE_URL` | — | Optional custom endpoint
`THEME` | cyberpunk | cyberpunk/ocean/forest/sunset

## 🧪 Tests
```
pytest
```

---

### ⓫ `mcp.config.json` (root)

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    }
  }
}
```

---

💡 Where to extend from here  
• Add `/theme` command to live-switch Rich themes.  
• Implement bookmarks & analytics as MCP verbs (`MCPServerStdio.register_tool`).  
• Enhance animations using `rich.live.Live` & custom spinners.  
• Deploy as a standalone binary with `shiv` / `pex`.

Enjoy your **awesomely cool** Context7 agent!

