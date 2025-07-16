# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Context7 AI Agent** - A sophisticated conversational AI agent with Pydantic AI integration and Context7 MCP server for intelligent document discovery. Features stunning TUI with 4 themes, real-time chat, document search, and session management.

## Architecture

### Core Components

- **Agent Layer** (`src/agent.py:33`) - Main AI agent using Pydantic AI with OpenAI integration and MCP server lifecycle management
- **CLI Interface** (`src/cli.py:43`) - Rich-based terminal UI with 4 themes, animations, and interactive commands
- **History Management** (`src/history.py:93`) - Persistent conversation storage with JSON serialization and analytics
- **Theme System** (`src/themes.py:19`) - 4 visual themes (cyberpunk, ocean, forest, sunset) with gradient text and animations
- **Configuration** (`src/config.py:13`) - Environment-based config with validation and MCP server setup

### Key Patterns

- **Async/Await** throughout - Uses `anyio` for async operations compatible with Pydantic AI
- **MCP Integration** - Context7 MCP server via `MCPServerStdio` for document search
- **Session Management** - Persistent sessions with JSON storage in `~/.context7-agent/`
- **Intent Detection** - Automatic document search based on user messages
- **Real-time Streaming** - Live chat interface with status animations

## Development Commands

### Setup & Installation
```bash
# Install dependencies
poetry install
# or
pip install -r requirements.txt

# Set required environment variables
export OPENAI_API_KEY="your-key-here"
```

### Running the Application
```bash
# Run the CLI application
python src/cli.py

# Or using poetry
poetry run python src/cli.py
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_agent.py

# Run with coverage
pytest --cov=src tests/
```

### Development Workflow
```bash
# Check code style
python -m flake8 src/

# Type checking
python -m mypy src/

# Format code
python -m black src/

# Interactive development
python -c "from src.agent import create_agent; agent = create_agent(); print('Agent ready')"
```

## Key Classes & Methods

### Context7Agent (`src/agent.py:33`)
- `chat(user_message: str) -> str` - Main chat interface
- `search_documents(query: str) -> List[Dict]` - Document search via MCP
- `detect_intent(message: str) -> Dict[str, Any]` - Intent analysis
- `save_session(filepath: Path)` - Session persistence

### Context7CLI (`src/cli.py:43`)
- `run()` - Main application entry point
- `process_chat_message(user_message: str)` - Chat processing with search
- `handle_command(command_input: str)` - Command processing
- Available commands: /help, /theme, /search, /bookmark, /history, /analytics

### ThemeManager (`src/themes.py:19`)
- `set_theme(theme_name: str)` - Switch between themes
- `create_gradient_text(text: str)` - Animated gradient text
- `particle_animation(duration: float)` - Visual effects

## Environment Configuration

Required environment variables:
- `OPENAI_API_KEY` - OpenAI API key
- `OPENAI_BASE_URL` (optional) - Custom API endpoint
- `OPENAI_MODEL` (optional) - Model name (default: gpt-4o-mini)
- `DEFAULT_THEME` (optional) - Visual theme (default: cyberpunk)

## File Structure

```
src/
├── agent.py         # Core AI agent with Pydantic AI
├── cli.py           # Rich terminal interface
├── config.py        # Configuration management
├── history.py       # Conversation persistence
├── themes.py        # Visual themes and animations
└── utils.py         # Utility functions

~/.context7-agent/   # Data directory for sessions and history
├── conversation_history.json
├── search_history.json
├── bookmarks.json
└── session_*.json   # Individual session files
```

## Common Development Tasks

### Adding New Themes
1. Add theme configuration to `ThemeManager.THEMES` in `themes.py:22`
2. Include color palette, ASCII art, and gradient colors
3. Test with `python -c "from src.themes import ThemeManager; ..."`

### Extending Commands
1. Add command method to `Context7CLI.commands` dict in `cli.py:71`
2. Implement async method with signature `async def command_name(self, args: List[str])`
3. Update help text in `show_help()` method

### MCP Server Changes
- Modify `Config.get_mcp_config()` in `config.py:59` for server configuration
- Update MCP server command/args in environment variables
- Test MCP integration with `agent.search_documents("test query")`

## Testing Patterns

- Use `pytest` with async support via `pytest-asyncio`
- Mock OpenAI responses using `unittest.mock`
- Test MCP integration with mock server responses
- Validate theme switching and TUI interactions

## Performance Notes

- Conversation history auto-saves asynchronously
- Sessions auto-save on exit and switch
- Large search results are truncated to top 10 items
- History files are JSON for easy debugging and inspection