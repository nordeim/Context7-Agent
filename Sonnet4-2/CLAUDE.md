# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**QuantumSearch Pro** (also known as Context7 Agent) is a revolutionary AI-powered document discovery system that combines Pydantic AI with Context7 MCP server integration. It features stunning terminal UI themes, conversational AI, and advanced document search capabilities.

## Quick Commands

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt
npm install  # For Context7 MCP server

# Run the application
python -m src.cli

# Run with custom theme
python -m src.cli --theme ocean
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_agent.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Environment Setup
```bash
# Create .env file
cp .env.example .env

# Required environment variables
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
DEFAULT_THEME=cyberpunk
```

## Architecture Overview

### Core Components

**🧠 Agent Layer** (`src/agent.py`)
- Pydantic AI agent with OpenAI integration
- Context7 MCP server integration for document context
- Intent detection and conversation management
- Search, analysis, and recommendation engines

**🎨 UI Layer** (`src/cli.py`)
- Rich-based terminal interface
- Multi-panel layout (chat, results, header, footer)
- Real-time updates and animations
- Theme switching and visual effects

**🎯 Theme System** (`src/themes.py`)
- 4 built-in themes: cyberpunk, ocean, forest, sunset
- Customizable colors, ASCII art, and animations
- Gradient text and particle effects

**💾 Data Management** (`src/history.py`)
- Conversation history with persistence
- Search history and analytics
- Bookmark management
- Session management with auto-save

### MCP Integration

The system uses Context7 MCP server for document context:
- **Server**: `@upstash/context7-mcp@latest`
- **Integration**: Pydantic AI's MCP server support
- **Configuration**: via `mcp.config.json` and environment variables

### Key Features

1. **Conversational AI**: Natural language queries with context awareness
2. **Document Search**: Semantic search with MCP-powered context
3. **Theme System**: 4 beautiful themes with animations
4. **History & Bookmarks**: Persistent conversation and search history
5. **Session Management**: Multiple sessions with auto-switching
6. **Analytics**: Usage statistics and search insights

## File Structure

```
src/
├── cli.py          # Main TUI interface
├── agent.py        # Core AI agent with MCP integration
├── themes.py       # Visual themes and styling
├── config.py       # Configuration management
├── history.py      # Data persistence (history, bookmarks)
├── utils.py        # Utility functions
└── init.py         # Package initialization
```

## Development Workflow

### Code Style
- **Black**: Code formatting (88 char line length)
- **MyPy**: Type checking (Python 3.11+)
- **Tests**: pytest with asyncio support

### Key Patterns
- **Async/Await**: All I/O operations are async
- **Data Classes**: Used for configuration and data models
- **Error Handling**: Graceful error handling with user-friendly messages
- **MCP Integration**: Async context management for server connections

### Adding New Features

1. **New Themes**: Add to `THEMES` dict in `themes.py`
2. **Commands**: Extend `_handle_command` in `agent.py` and `cli.py`
3. **Intents**: Update `detect_intent` method in `agent.py`
4. **Tests**: Add to corresponding test file in `tests/`

## Common Tasks

### Changing Themes
```bash
/theme cyberpunk  # Built-in themes: cyberpunk, ocean, forest, sunset
```

### Search Operations
```bash
"Find Python tutorials"  # Natural language search
"Tell me about quantum computing"  # Contextual queries
```

### Session Management
```bash
/sessions          # List available sessions
/history          # View search history
/bookmark title   # Bookmark current results
```

### Debugging
```bash
# Check MCP server connection
python -c "from src.agent import Context7Agent; print('Agent initialized')"

# Test theme loading
python -c "from src.themes import get_theme; print(get_theme('ocean').name)"
```

## Environment Configuration

Required `.env` variables:
```bash
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini
MCP_SERVER_COMMAND=npx
MCP_SERVER_ARGS=-y,@upstash/context7-mcp@latest
DEFAULT_THEME=cyberpunk
```

## Performance Notes

- **Memory**: Moderate usage due to conversation history caching
- **Network**: Requires internet for OpenAI API and Context7 MCP
- **Terminal**: Best with 256-color support and minimum 80x24 terminal
- **Storage**: JSON files for persistence in `data/` directory