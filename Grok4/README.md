# Context7 Agent

An awesomely cool and user-friendly AI agent built with Pydantic AI, integrated with Context7 MCP server. Transforms terminal chats into immersive experiences with themes, animations, and smart document searches.

## Features
- Conversational chat with OpenAI.
- Intent detection for MCP searches.
- Dynamic TUI with Rich.
- Theme switching (Cyberpunk, etc.).
- History persistence in JSON.
- Hotkeys like /preview, /theme.

## Installation

### Prerequisites
- Python 3.11+
- Node.js/npm for MCP server
- OpenAI API key

1. Clone the repo: `git clone <repo>`
2. Install deps: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in keys.
4. Start MCP server if needed (configured to use npx).

## Usage
Run the CLI: `python src/cli.py`

Chat naturally: "Tell me about quantum computing" – it'll search MCP and respond.

Hotkeys:
- /theme <name> (e.g., cyberpunk)
- /preview <id>
- /help
- /exit

## Tests
Run `pytest tests/`

Enjoy your futuristic librarian! 🚀
