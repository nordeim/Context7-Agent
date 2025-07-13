# ⚡ Context7 Agent

A dazzling terminal librarian that lets you **chat** while it hunts documents
via the Context7 **MCP** protocol.  
Rich animations, four flashy themes, live-streamed answers, bookmarks (soon) –
all in one convenient Python package.

## 📸 Screenshot

```
╭────────────────────────────────────────────────────────╮
│ ⚡  Context7 Agent  ⚡                                 │
├────────────────────────────────────────────────────────┤
│ **You:** Tell me something about quantum computing     │
│ **AI:** Sure! First, here’s a bite-size intro …        │
├──────────────── Results ────────────────┬─────────────┤
│ search#42  Quantum Computing Basics.pdf │  1 MB  2023 │
│ search#43  Shor’s Algorithm.md          │  5 KB  2022 │
╰────────────────────────────────────────────────────────╯
```

## 🚀 Quick start

```bash
git clone …
cd context7-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env       # add your OpenAI key

python -m src.cli chat
```

First run will auto-download and boot the MCP server using  
`npx -y @upstash/context7-mcp@latest`.

### Slash commands
| Command | Description            |
|---------|------------------------|
| `/theme`| Cycle colour themes    |
| `/help` | Show quick cheatsheet  |
| `/exit` | Quit & save history    |

---

## 🧪 Tests
```bash
pytest
```
---

## 📂 Project layout
```
src/         main code (agent, cli, themes…)
tests/       pytest suite
mcp.config.json  Context7 server definition
```

---

Enjoy the future of terminal knowledge discovery! ✨
