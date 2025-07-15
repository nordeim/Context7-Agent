# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Context7 Agent is an AI-powered terminal chat application that integrates with Context7 MCP server for intelligent document retrieval. Built with Pydantic AI and Rich for TUI, it provides themed conversational interfaces with document search capabilities.

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env  # Fill in OPENAI_API_KEY

# Run the application
python src/cli.py
```

## Core Architecture

### Main Components
- **CLI Layer** (`src/cli.py`): Rich-based terminal interface with themes and animations
- **Agent Core** (`src/agent.py`): Pydantic AI integration with OpenAI + Context7 MCP
- **Configuration** (`src/config.py`): Environment-based configuration management
- **History** (`src/history.py`): JSON persistence for conversations, searches, bookmarks
- **Themes** (`src/themes.py`): Visual theming system with ASCII art
- **Utils** (`src/utils.py`): Shared utility functions

### Key Design Patterns
- **Agent Pattern**: `Context7Agent` class wraps Pydantic AI with MCP integration
- **Repository Pattern**: `History` class handles all persistence operations
- **Strategy Pattern**: Theme system allows runtime style switching
- **Async/Await**: Full async support for MCP operations and API calls

## Development Commands

### Setup
```bash
# Install Poetry (if using)
pip install poetry
poetry install

# Or use pip directly
pip install -r requirements.txt
```

### Running
```bash
# Main application
python src/cli.py

# With specific Python path
PYTHONPATH=src python src/cli.py
```

### Environment Variables
Required in `.env`:
- `OPENAI_API_KEY` - OpenAI API key
- `OPENAI_BASE_URL` - Optional custom endpoint (default: https://api.openai.com/v1)
- `OPENAI_MODEL` - Model name (default: gpt-4o-mini)

## Key Files & Responsibilities

| File | Purpose | Key Classes |
|------|---------|-------------|
| `src/cli.py` | Terminal interface | `CLI` class with Rich TUI |
| `src/agent.py` | AI agent logic | `Context7Agent` with Pydantic AI |
| `src/config.py` | Configuration | `Config` singleton |
| `src/history.py` | Data persistence | `History` with JSON storage |
| `src/themes.py` | Visual theming | Theme definitions and styles |
| `mcp.config.json` | MCP server config | Context7 MCP integration |

## MCP Integration
- Uses `@upstash/context7-mcp@latest` via npx
- Configured in `mcp.config.json`
- Managed through Pydantic AI's `MCPServerStdio`
- Provides document search and retrieval capabilities

## Testing
```bash
# Run tests (when tests/ directory exists)
pytest tests/

# Or with Poetry
poetry run pytest
```

## Hotkeys
- `/theme <name>` - Switch theme (cyberpunk, ocean, forest, sunset)
- `/preview <id>` - Preview search result
- `/help` - Show available commands
- `/exit` - Exit application

## Data Storage
- `history.json` - Stores conversations, searches, bookmarks
- Auto-saved on each interaction
- JSON format for easy inspection/debugging